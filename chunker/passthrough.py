"""SQL pass-through — the database tables a ``PROC SQL`` step reaches directly.

SAS reaches a database's own tables with *explicit SQL pass-through*::

    proc sql;
    connect to oracle (user=&ora_user password=&ora_pass path=&ora_path);
    create table nonip as select * from connection to oracle
    (select cov_month, count as count from edw_export.current_nonip where table_cd='MED');
    disconnect from oracle;
    quit;

Everything inside ``CONNECTION TO oracle ( ... )`` and ``EXECUTE ( ... ) BY
oracle`` is the database's own SQL, sent as written. While the chunker had no
grammar for it, its SAS SQL patterns read that text as SAS: ``from connection``
became the dataset ``work.connection``, ``disconnect from oracle`` the dataset
``work.oracle``, and the Oracle owner ``edw_export`` a SAS libref that a batch
then reported as a LIBNAME it needed. This module recognises the statements,
records each table in the database's own terms as a
:class:`~chunker.models.SasDbTableRef`, and returns the spans every SAS-side
dataset scan must have **masked** — native SQL is never scanned as SAS.

Two forms of the same text
--------------------------
Structure — statement ends, the parentheses around native SQL, the keywords —
is found on the fully sanitised text (``mt``), where string interiors are blank,
so a ``;`` or ``)`` inside ``path="(DESCRIPTION=(HOST=x))"`` cannot end
anything and no quote tracking is needed. Text — option values, quoted
identifiers like ``"EDW"."T"`` — is read from the strings-intact form (``cf``)
at the same offsets. The two are character-aligned by construction: the
sanitiser replaces what it blanks with the same number of characters.

The native-SQL scan
-------------------
A small token walk, not a regex, because a FROM clause is a comma-separated list
whose items may be subqueries, table functions or aliased, and because ``FROM``
also appears inside ``EXTRACT(YEAR FROM d)``. Oracle ``'...'`` literals and
``--`` comments are blanked first — neither is SAS syntax, so the SAS sanitiser
left both in place. Names are lowercased in the parsed fields, including quoted
identifiers (Oracle treats those as case-sensitive; ``raw`` keeps the spelling
for a consumer that has to care).

Logger name: ``chunker.passthrough``.
"""

from __future__ import annotations

import logging
import re
from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass

from .macro_vars import DS_REF_TOKEN
from .models import DbTableAccess, DbTableVia, SasDbTableRef, _db_table_sort_key
from .paths import _ENGINE_OPTION_RE, ENGINE_LIBNAMES, _option_value
from .scanner import _blank_span

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAS-side statements — matched on the sanitised text
# ---------------------------------------------------------------------------

# CONNECT TO <engine> [AS <alias>]; the options parenthesis, when there is one,
# is found from the end of the match.
# A connection name — engine, alias or libref — in any of these statements. A
# macro reference is allowed (``connection to &db``): parameterised connection
# names are ordinary in utility macros, and while this required a bare
# identifier such a statement matched nothing, so its native SQL went unmasked
# and was read as SAS again. Resolved later, by the macro-variable pass.
_CONN_NAME = r"[A-Za-z_&][\w&]*\.?"

_CONNECT_TO_RE = re.compile(
    rf"\bconnect\s+to\s+(?P<engine>{_CONN_NAME})"
    rf"(?:\s+as\s+(?P<alias>{_CONN_NAME}))?",
    re.IGNORECASE,
)
# CONNECT USING <libref> [AS <alias>] — reuses a LIBNAME's connection, so the
# engine is the LIBNAME's: chunker.metadata.resolve_db_librefs fills it in.
_CONNECT_USING_RE = re.compile(
    rf"\bconnect\s+using\s+(?P<libref>{_CONN_NAME})"
    rf"(?:\s+as\s+(?P<alias>{_CONN_NAME}))?",
    re.IGNORECASE,
)
_DISCONNECT_RE = re.compile(
    rf"\bdisconnect\s+from\s+(?P<alias>{_CONN_NAME})", re.IGNORECASE
)
# FROM|JOIN CONNECTION TO <alias> ( — the match ends on the query's open paren.
_CONNECTION_TO_RE = re.compile(
    rf"\b(?:from|join)\s+connection\s+to\s+(?P<alias>{_CONN_NAME})\s*\(",
    re.IGNORECASE,
)
# EXECUTE ( <stmt> ) BY <alias>, or the newer EXECUTE BY <alias> ( <stmt> ). A
# plain ``execute(`` with no BY after its parenthesis is not pass-through —
# that is a DATA step's CALL EXECUTE — and is skipped.
_EXECUTE_RE = re.compile(
    rf"\bexecute\s*(?:by\s+(?P<alias>{_CONN_NAME})\s*)?\(", re.IGNORECASE
)
_BY_ALIAS_RE = re.compile(rf"\s*by\s+(?P<alias>{_CONN_NAME})", re.IGNORECASE)

# The SAS statement a CONNECTION TO feeds: the dataset it creates or inserts
# into. Searched for, not anchored, so ``proc sql`` written without its own
# semicolon still leaves ``create table nonip`` findable.
_SAS_TARGET_RE = re.compile(
    rf"\b(?:create\s+(?:table|view)|insert\s+into)\s+({DS_REF_TOKEN})",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Native SQL
# ---------------------------------------------------------------------------

# Oracle string literals ('' escaped) and -- line comments, blanked before the
# table scan. "QUOTED" identifiers are names and survive.
_NATIVE_NOISE_RE = re.compile(r"'(?:[^']|'')*'|--[^\n]*")

_IDENT = r'(?:"[^"\n]*"|[A-Za-z_&][\w$#&]*)'
# A table reference: an identifier chain with an optional @dblink. Dots come in
# runs because macro references consume them: ``&schema..tbl`` is a delimiter
# then the separator, and an indirect ``&&sch_&env...tbl`` needs one more per
# rescan. _name_parts decides which dot does what.
_NATIVE_REF = rf"{_IDENT}(?:\.+{_IDENT})*(?:@[\w$#.&]+)?"
_NATIVE_TOKEN_RE = re.compile(rf"{_NATIVE_REF}|[(),;]")
_WORD_RE = re.compile(r"[A-Za-z_]\w*")

_PUNCT = frozenset({"(", ")", ",", ";"})

# The only characters _Structure indexes.
_STRUCTURE_RE = re.compile(r"[();]")

# The statement verbs that decide what a native statement writes.
_VERBS = frozenset(
    {"select", "insert", "update", "delete", "merge", "create", "truncate", "drop", "alter"}
)

# Functions whose argument syntax contains FROM without naming a table.
_FUNCTION_FROM = frozenset({"extract", "trim", "substring", "overlay", "position"})

# Words that end a FROM item — never a table, never an alias.
_CLAUSE_WORDS = frozenset(
    {
        "where", "group", "order", "having", "connect", "start", "union",
        "intersect", "minus", "except", "fetch", "offset", "for", "model",
        "pivot", "unpivot", "join", "inner", "left", "right", "full", "cross",
        "natural", "outer", "on", "using", "window", "qualify", "limit", "when",
        "then", "else", "end", "set", "values", "select", "returning", "with",
        "into", "by", "partition", "subpartition", "sample", "as", "lateral",
        "apply", "versions", "of", "log", "errors",
    }
)

# Words allowed between CREATE and TABLE/VIEW (Oracle and Teradata forms).
_CREATE_MODIFIERS = frozenset(
    {
        "or", "replace", "global", "private", "temporary", "materialized",
        "force", "noforce", "editionable", "noneditionable", "sharded",
        "duplicated", "immutable", "blockchain", "volatile", "multiset", "set",
    }
)


# One piece of a table reference, for splitting it into parts: a macro
# reference *with* its delimiter dot, a "quoted" identifier, a run of other
# name characters, or a separator dot.
_NAME_PIECE_RE = re.compile(r'&+[A-Za-z_]\w*\.?|"[^"\n]*"|[^."&]+|\.|&')
# A part ending in a macro reference with no delimiter after it: "&sch" once
# its quotes are stripped, or a LIBNAME's schema=&sch.
_ENDS_IN_REF_RE = re.compile(r"&+[A-Za-z_]\w*\Z")


def _name_parts(body: str) -> list[str]:
    """*body*'s dot-separated parts, reading dots the way SAS's macro processor does.

    The dot right after a macro reference is its *delimiter* — SAS consumes it
    — so it separates nothing: ``t_&sfx._v`` is one table, ``t_<sfx>_v``; in
    ``&sch..t`` the first dot ends ``&sch`` and only the second separates.
    Splitting on every dot made an unresolved ``edw_export.t_&sfx._v`` read as
    schema ``edw_export.t_&sfx``, table ``_v``. A delimiter stays with its part,
    and so does each further dot after it — an indirect ``&&sch_&env...t`` needs
    one per rescan — so the parts join back with single dots into the name as
    written. A dot run anywhere else (SQL Server's ``db..t``) is one separator.
    """
    parts = [""]
    for m in _NAME_PIECE_RE.finditer(body):
        piece = m.group(0)
        if piece != ".":
            parts[-1] += piece
        elif not parts[-1] and len(parts) > 1 and parts[-2].endswith("."):
            parts[-2] += "."
        else:
            parts.append("")
    return [p for p in parts if p]


def _delimited(part: str) -> str:
    """*part*, ready for a separator dot to follow it.

    One ending in a bare macro reference gets the delimiter SAS consumes first,
    or the joined name would say something else: ``&sch.t`` is *one* name, the
    value of ``sch`` followed by ``t``; ``&sch..t`` is schema and table.
    """
    return f"{part}." if _ENDS_IN_REF_RE.search(part) else part


def split_table_name(name: str) -> tuple[str | None, str, str | None]:
    """``(db_schema, table, dblink)`` of a native table reference.

    Lowercased, quotes stripped. Everything before the last part is the schema,
    so SQL Server's ``db.dbo.t`` keeps its database as ``db.dbo`` rather than
    losing it. Macro references split as SAS would read them — see
    :func:`_name_parts`. A schema that ends in an unresolved reference keeps its
    delimiter (``&sch..t`` → ``&sch.``), so schema and table join back into what
    SAS would read, ``&sch..t``; the table's own trailing delimiter is dropped,
    since ``&tbl.`` and ``&tbl`` name one variable.
    """
    body, _, link = name.partition("@")
    parts = [p.strip('"').lower() for p in _name_parts(body)]
    if not parts:
        return None, body.strip().lower(), (link.lower() or None)
    *schema, table = parts
    db_schema = ".".join(_delimited(p) for p in schema) or None
    return db_schema, table.rstrip(".") or table, (link.lower() or None)


def db_table_ref(
    raw: str,
    *,
    access: DbTableAccess,
    via: DbTableVia,
    connection: str,
    engine: str | None = None,
    sas_targets: tuple[str, ...] = (),
    options: tuple[tuple[str, str], ...] = (),
    name: str | None = None,
    default_schema: str | None = None,
    macro: str | None = None,
    parameterised: bool = False,
) -> SasDbTableRef:
    """The single builder of a :class:`~chunker.models.SasDbTableRef`.

    *name* is the reference to parse when it differs from *raw* — a resolved
    macro reference, or a LIBNAME member whose SAS spelling (*raw*,
    ``edw.accounts``) names a libref rather than a schema. *default_schema*
    fills an unqualified name: the LIBNAME's ``schema=`` option. Keeping every
    construction here is what keeps :attr:`~SasDbTableRef.has_macro_ref` honest.
    """
    db_schema, table, dblink = split_table_name(raw if name is None else name)
    if db_schema is None and default_schema:
        db_schema = _delimited(default_schema)
    return SasDbTableRef(
        engine=engine,
        connection=connection,
        db_schema=db_schema,
        table=table,
        access=access,
        via=via,
        sas_targets=sas_targets,
        dblink=dblink,
        options=options,
        has_macro_ref="&" in f"{db_schema or ''}.{table}@{dblink or ''}",
        raw=raw,
        macro=macro,
        parameterised=parameterised,
    )


class _NativeTables:
    """Table references in one native SQL statement, as written.

    Built once per statement: tokenises, matches every parenthesis in one pass
    (so skipping a subquery is a lookup, never a rescan), finds CTE names and
    the statement verb, then reads the FROM/JOIN/USING sources and the write
    targets the verb implies.
    """

    def __init__(self, sql: str) -> None:
        text = _NATIVE_NOISE_RE.sub(lambda m: _blank_span(m.group(0)), sql)
        self.toks = [m.group(0) for m in _NATIVE_TOKEN_RE.finditer(text)]
        self.words = [
            t.lower() if _WORD_RE.fullmatch(t) else None for t in self.toks
        ]
        self.n = len(self.toks)
        self.match: dict[int, int] = {}
        self.depth: list[int] = []
        stack: list[int] = []
        for i, tok in enumerate(self.toks):
            if tok == ")" and stack:
                self.match[stack.pop()] = i
            self.depth.append(len(stack))
            if tok == "(":
                stack.append(i)
        self.ctes = self._cte_names()

    # -- helpers -----------------------------------------------------------

    def _after(self, j: int) -> int:
        """The index past the parenthesised group opening at *j*."""
        return self.match.get(j, self.n - 1) + 1

    def _is_name(self, j: int) -> bool:
        return (
            j < self.n
            and self.toks[j] not in _PUNCT
            and self.words[j] not in _CLAUSE_WORDS
        )

    def _cte_names(self) -> set[str]:
        """Names a ``WITH`` clause defines — queries, not tables."""
        names: set[str] = set()
        for i, word in enumerate(self.words):
            if word != "with":
                continue
            j = i + 1
            while j < self.n and (name := self.words[j]) is not None:
                j += 1
                if j < self.n and self.toks[j] == "(":  # column list
                    j = self._after(j)
                if j >= self.n or self.words[j] != "as":
                    break
                j += 1
                while j < self.n and self.words[j] in {"materialized", "not"}:
                    j += 1
                if j >= self.n or self.toks[j] != "(":
                    break
                names.add(name)
                j = self._after(j)
                if j < self.n and self.toks[j] == ",":
                    j += 1
                    continue
                break
        return names

    def _items(self, j: int, *, many: bool) -> list[str]:
        """The table items of a FROM list (or one JOIN/USING item) at *j*."""
        found: list[str] = []
        while j < self.n:
            tok = self.toks[j]
            if tok == "(":  # inline view — its own FROMs are walked separately
                j = self._after(j)
            elif tok not in _PUNCT and j + 1 < self.n and self.toks[j + 1] == "(":
                # TABLE(...), XMLTABLE(...), LATERAL(...) — checked before the
                # clause words, since LATERAL is one and must not end the list
                j = self._after(j + 1)
            elif tok in _PUNCT or self.words[j] in _CLAUSE_WORDS:
                break
            else:
                found.append(tok)
                j += 1
            # partition / sample clauses, then an optional alias
            while j < self.n and self.words[j] in {"partition", "subpartition", "sample"}:
                k = j + 1
                if k < self.n and self.words[k] == "block":
                    k += 1
                if k < self.n and self.toks[k] == "(":
                    j = self._after(k)
                else:
                    break
            if j < self.n and self.words[j] == "as":
                j += 2
            elif self._is_name(j):
                j += 1
            if many and j < self.n and self.toks[j] == ",":
                j += 1
                continue
            break
        return found

    def _verb(self) -> tuple[int, str | None]:
        for i, word in enumerate(self.words):
            if word in _VERBS and self.depth[i] == 0:
                return i, word
        return -1, None

    # -- the scan ----------------------------------------------------------

    def reads(self) -> list[str]:
        """Every table the statement reads, first-seen order."""
        found: list[str] = []
        openers: list[str | None] = []
        _, verb = self._verb()
        for i, tok in enumerate(self.toks):
            word = self.words[i]
            if tok == "(":
                openers.append(self.words[i - 1] if i else None)
            elif tok == ")":
                if openers:
                    openers.pop()
            elif word == "from":
                if openers and openers[-1] in _FUNCTION_FROM:
                    continue  # EXTRACT(YEAR FROM d) — an argument, not a table
                if i and self.words[i - 1] == "delete":
                    continue  # DELETE FROM t — a write
                found += self._items(i + 1, many=True)
            elif word == "join":
                found += self._items(i + 1, many=False)
            elif word == "using" and verb == "merge":
                found += self._items(i + 1, many=False)
        return [r for r in found if self._is_table(r)]

    def writes(self) -> list[str]:
        """Every table the statement creates, changes or removes."""
        at, verb = self._verb()
        if verb is None:
            return []
        found: list[str] = []
        if verb in {"insert", "merge"}:
            # INSERT ALL INTO a ... INTO b ... names several targets.
            found = [
                self.toks[j + 1]
                for j in range(at, self.n - 1)
                if self.words[j] == "into" and self.depth[j] == 0 and self._is_name(j + 1)
            ]
        elif verb == "update" and self._is_name(at + 1):
            found = [self.toks[at + 1]]
        elif verb == "delete":
            j = at + 1
            if j < self.n and self.words[j] == "from":
                j += 1
            if self._is_name(j):
                found = [self.toks[j]]
        elif verb == "create":
            j = at + 1
            while j < self.n and self.words[j] in _CREATE_MODIFIERS:
                j += 1
            if j < self.n and self.words[j] in {"table", "view"} and self._is_name(j + 1):
                found = [self.toks[j + 1]]
        elif verb in {"truncate", "drop", "alter"}:
            if (
                at + 1 < self.n
                and self.words[at + 1] in {"table", "view"}
                and self._is_name(at + 2)
            ):
                found = [self.toks[at + 2]]
        return [r for r in found if self._is_table(r)]

    def _is_table(self, ref: str) -> bool:
        db_schema, table, _ = split_table_name(ref)
        if db_schema is None and table in self.ctes:
            return False
        return not (table == "dual" and db_schema in {None, "sys"})


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PassThroughScan:
    """What :func:`scan_pass_through` found in one chunk.

    ``spans`` are half-open ``[start, end)`` offsets of every pass-through
    statement or ``CONNECTION TO (...)`` clause — the text :func:`mask` blanks
    before any SAS-side dataset scan runs. ``tables`` are sorted by
    ``_db_table_sort_key``, duplicates removed.
    """

    spans: tuple[tuple[int, int], ...] = ()
    tables: tuple[SasDbTableRef, ...] = ()


@dataclass(frozen=True)
class _Connection:
    engine: str | None
    name: str  # what SasDbTableRef.connection records
    options: tuple[tuple[str, str], ...] = ()


class _Structure:
    """Where every statement ends and every parenthesis closes, indexed once.

    Built from one pass over the sanitised text visiting only ``(``, ``)`` and
    ``;``, so finding a body's end is a lookup rather than a rescan — which is
    what keeps a chunk of thousands of unclosed ``execute(`` calls (each one a
    CALL EXECUTE, so never claimed as a span) linear instead of quadratic.
    """

    def __init__(self, mt: str) -> None:
        self.size = len(mt)
        self.semis: list[int] = []
        self.parens: dict[int, int] = {}
        stack: list[int] = []
        for m in _STRUCTURE_RE.finditer(mt):
            ch, at = m.group(0), m.start()
            if ch == "(":
                stack.append(at)
            elif ch == ")":
                if stack:
                    self.parens[stack.pop()] = at
            else:
                # SAS ends the statement here whatever the parentheses say, so
                # an unbalanced body never swallows the rest of the step.
                self.semis.append(at)
                stack.clear()

    def next_semi(self, pos: int) -> int:
        """The offset of the first ``;`` at or after *pos*, or the text's end."""
        i = bisect_left(self.semis, pos)
        return self.semis[i] if i < len(self.semis) else self.size

    def statement_start(self, pos: int) -> int:
        """The offset just past the ``;`` before *pos*, or ``0``."""
        i = bisect_left(self.semis, pos)
        return self.semis[i - 1] + 1 if i else 0

    def statement_end(self, pos: int) -> int:
        """The offset just past the ``;`` ending the statement at *pos*."""
        return min(self.next_semi(pos) + 1, self.size)

    def body(self, open_idx: int) -> tuple[int, int]:
        """``(body_end, span_end)`` for the parenthesised body opening at *open_idx*.

        The body runs to the matching ``)``; unbalanced, it runs to the
        statement's end and is still scanned for what it names.
        """
        close = self.parens.get(open_idx)
        if close is not None:
            return close, close + 1
        stop = self.next_semi(open_idx)
        return stop, stop


def _conn_name(token: str) -> str:
    """A connection name as recorded: lowercased, a macro delimiter dot dropped."""
    return token.lower().rstrip(".")


def _options(text: str) -> tuple[tuple[str, str], ...]:
    """``key=value`` options, parsed by the engine-LIBNAME grammar in :mod:`chunker.paths`."""
    return tuple(
        (m.group("key").lower(), _option_value(m.group("value")))
        for m in _ENGINE_OPTION_RE.finditer(text)
    )


def _sas_targets(mt: str, structure: _Structure, at: int) -> tuple[str, ...]:
    """The SAS dataset the statement containing offset *at* creates or fills."""
    m = _SAS_TARGET_RE.search(mt, structure.statement_start(at), at)
    return (m.group(1).lower(),) if m else ()


def scan_pass_through(cf: str, mt: str) -> PassThroughScan:
    """Every pass-through statement in one chunk, and the tables they name.

    *cf* is the comments-blanked, strings-intact text and *mt* the fully
    sanitised one (``cf`` / ``mt`` in :func:`chunker.metadata._metadata_for`).
    Statements are processed in source order, so an alias means whatever the
    latest ``CONNECT`` before it said — and a ``CONNECTION TO`` whose ``CONNECT``
    was made elsewhere (a macro call) still records its tables, with the engine
    taken from the alias when the alias *is* an engine name, else ``None``.
    """
    events = sorted(
        [(m.start(), "connect", m) for m in _CONNECT_TO_RE.finditer(mt)]
        + [(m.start(), "using", m) for m in _CONNECT_USING_RE.finditer(mt)]
        + [(m.start(), "disconnect", m) for m in _DISCONNECT_RE.finditer(mt)]
        + [(m.start(), "query", m) for m in _CONNECTION_TO_RE.finditer(mt)]
        + [(m.start(), "execute", m) for m in _EXECUTE_RE.finditer(mt)],
        key=lambda event: event[0],
    )
    structure = _Structure(mt)
    aliases: dict[str, _Connection] = {}
    spans: list[tuple[int, int]] = []
    tables: list[SasDbTableRef] = []

    def connection_for(alias: str) -> _Connection:
        return aliases.get(alias) or _Connection(
            alias if alias in ENGINE_LIBNAMES else None, alias
        )

    def record(
        refs: Iterable[str],
        access: DbTableAccess,
        via: DbTableVia,
        conn: _Connection,
        sas_targets: tuple[str, ...] = (),
    ) -> None:
        tables.extend(
            db_table_ref(
                ref,
                access=access,
                via=via,
                connection=conn.name,
                engine=conn.engine,
                sas_targets=sas_targets,
                options=conn.options,
            )
            for ref in refs
        )

    claimed = 0  # end of the last span — anything starting before it is inside native SQL
    for start, kind, m in events:
        if start < claimed:
            continue
        if kind == "connect":
            engine = _conn_name(m.group("engine"))
            alias = _conn_name(m.group("alias") or engine)
            j = m.end()
            while j < len(mt) and mt[j].isspace():
                j += 1
            options: tuple[tuple[str, str], ...] = ()
            if j < len(mt) and mt[j] == "(":
                body_end, _ = structure.body(j)
                options = _options(cf[j + 1 : body_end])
            aliases[alias] = _Connection(engine, alias, options)
            end = structure.statement_end(j)
        elif kind == "using":
            libref = _conn_name(m.group("libref"))
            alias = _conn_name(m.group("alias") or libref)
            aliases[alias] = _Connection(None, libref)
            end = structure.statement_end(m.end())
        elif kind == "disconnect":
            aliases.pop(_conn_name(m.group("alias")), None)
            end = structure.statement_end(m.end())
        elif kind == "query":
            open_idx = m.end() - 1
            body_end, end = structure.body(open_idx)
            native = _NativeTables(cf[open_idx + 1 : body_end])
            record(
                native.reads(),
                DbTableAccess.READ,
                DbTableVia.CONNECTION_TO,
                connection_for(_conn_name(m.group("alias"))),
                _sas_targets(mt, structure, start),
            )
        else:  # execute
            open_idx = m.end() - 1
            body_end, after = structure.body(open_idx)
            alias = m.group("alias")
            if alias is None:
                by = _BY_ALIAS_RE.match(mt, after)
                if by is None:
                    continue  # CALL EXECUTE(...) — not pass-through
                alias, after = by.group("alias"), by.end()
            conn = connection_for(_conn_name(alias))
            native = _NativeTables(cf[open_idx + 1 : body_end])
            record(native.reads(), DbTableAccess.READ, DbTableVia.EXECUTE, conn)
            record(native.writes(), DbTableAccess.WRITE, DbTableVia.EXECUTE, conn)
            end = structure.statement_end(after)
        spans.append((start, end))
        claimed = end

    found = sorted(dict.fromkeys(tables), key=_db_table_sort_key)
    if found and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"scan_pass_through: {[str(t) for t in found]}")
    return PassThroughScan(tuple(spans), tuple(found))


def mask(text: str, spans: Iterable[tuple[int, int]]) -> str:
    """*text* with every span blanked, length and newlines preserved."""
    out: list[str] = []
    prev = 0
    for start, end in sorted(spans):
        start = max(start, prev)
        if end <= start:
            continue
        out += [text[prev:start], _blank_span(text[start:end])]
        prev = end
    out.append(text[prev:])
    return "".join(out)
