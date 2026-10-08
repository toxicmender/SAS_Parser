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
:class:`chunker.sql.SqlStatement` in its NATIVE dialect: the one SQL grammar,
shared with PROC SQL. Names are lowercased in the parsed fields, including
quoted identifiers (Oracle treats those as case-sensitive; ``raw`` keeps the
spelling for a consumer that has to care).

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
from .sql import SqlStatement, _delimited, split_table_name

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


# The only characters _Structure indexes.
_STRUCTURE_RE = re.compile(r"[();]")


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
            native = SqlStatement(cf[open_idx + 1 : body_end])
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
            native = SqlStatement(cf[open_idx + 1 : body_end])
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
