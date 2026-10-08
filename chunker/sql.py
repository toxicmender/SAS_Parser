"""One SQL grammar: the tables a SQL statement reads and writes. See chunker/README.md.

A small token walk, not a regex, because a FROM clause is a comma-separated list
whose items may be subqueries, table functions or aliased, and because ``FROM``
also appears inside ``EXTRACT(YEAR FROM d)``. Built once per statement, it
tokenises, matches every parenthesis in one pass (so skipping a subquery is a
lookup, never a rescan), finds CTE names and the statement's verb, then reads
the FROM/JOIN/USING sources and the targets the verb implies.

Two dialects share the walk:

- :attr:`Dialect.NATIVE` — a database's own SQL, sent by pass-through
  (:mod:`chunker.passthrough`). ``'...'`` literals and ``--`` comments are
  blanked first: neither is SAS syntax, so the SAS sanitiser left both in
  place. ``"quoted"`` identifiers are names, a name may carry an ``@dblink``, a
  CTE and DUAL are no tables, and ``name(`` in a FROM list is a table function.
- :attr:`Dialect.SAS` — PROC SQL and PROC FEDSQL. A quoted string is a literal
  except where a table stands: there it is a physical path or a name literal
  (``'my data'n``). ``name(…)`` there is the table with its dataset options, a
  macro call names nothing that can be seen, and DICTIONARY tables are SAS's
  own metadata, no datasets. :meth:`SqlStatement.refs` gives each table what
  the statement does to it.

Names come back as written; the callers parse them —
:func:`split_table_name` a database's, :mod:`chunker.statements` SAS's. Pure; no
logging.
"""

from __future__ import annotations

import re
from enum import StrEnum

from .macro_vars import DS_REF_TOKEN
from .models import DatasetRole
from .scanner import _blank_span

# ---------------------------------------------------------------------------
# Table names
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


class Dialect(StrEnum):
    """Whose SQL a statement is: a database's (pass-through) or SAS's."""

    NATIVE = "native"
    SAS = "sas"


# Oracle string literals ('' escaped) and -- line comments, blanked before the
# native scan. "QUOTED" identifiers are names and survive.
_NATIVE_NOISE_RE = re.compile(r"'(?:[^']|'')*'|--[^\n]*")

_IDENT = r'(?:"[^"\n]*"|[A-Za-z_&][\w$#&]*)'
# A table reference: an identifier chain with an optional @dblink. Dots come in
# runs because macro references consume them: ``&schema..tbl`` is a delimiter
# then the separator, and an indirect ``&&sch_&env...tbl`` needs one more per
# rescan. _name_parts decides which dot does what.
_NATIVE_REF = rf"{_IDENT}(?:\.+{_IDENT})*(?:@[\w$#.&]+)?"
_NATIVE_TOKEN_RE = re.compile(rf"{_NATIVE_REF}|[(),;]")
# SAS: a string (a path or, with its n, a name literal where a table stands), a
# macro call's name, a dataset name that may hold &refs, punctuation.
_SAS_TOKEN_RE = re.compile(
    rf"""'(?:[^']|'')*'n?|"(?:[^"]|"")*"n?|%\s*[A-Za-z_]\w*|{DS_REF_TOKEN}|[(),;]""",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z_]\w*")

_PUNCT = frozenset({"(", ")", ",", ";"})

# The statement verbs that decide what a statement writes.
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

READ, WRITE, UPDATE, DROP = (
    DatasetRole.READ,
    DatasetRole.WRITE,
    DatasetRole.UPDATE,
    DatasetRole.DROP,
)


# ---------------------------------------------------------------------------
# The statement
# ---------------------------------------------------------------------------


class SqlStatement:
    """The tables one SQL statement names, as written (see the module docstring)."""

    __slots__ = ("dialect", "toks", "words", "n", "match", "depth", "ctes")

    def __init__(self, text: str, dialect: Dialect = Dialect.NATIVE) -> None:
        self.dialect = dialect
        if dialect is Dialect.NATIVE:
            text = _NATIVE_NOISE_RE.sub(lambda m: _blank_span(m.group(0)), text)
            tokens = _NATIVE_TOKEN_RE
        else:
            tokens = _SAS_TOKEN_RE
        self.toks = [m.group(0) for m in tokens.finditer(text)]
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
            and self.toks[j][0] != "%"
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

    def _items(self, j: int, *, many: bool) -> list[int]:
        """Where the table items of a FROM list (or one JOIN/USING item) at *j* are."""
        sas = self.dialect is Dialect.SAS
        found: list[int] = []
        while j < self.n:
            tok = self.toks[j]
            if tok == "(":  # inline view — its own FROMs are walked separately
                j = self._after(j)
            elif sas and tok[0] == "%":  # a macro call: nothing that can be seen
                j += 1
                if j < self.n and self.toks[j] == "(":
                    j = self._after(j)
            elif (
                sas
                and tok not in _PUNCT
                and self.words[j] not in _CLAUSE_WORDS
                and j + 1 < self.n
                and self.toks[j + 1] == "("
            ):
                found.append(j)  # a dataset with its options
                j = self._after(j + 1)
            elif not sas and tok not in _PUNCT and j + 1 < self.n and self.toks[j + 1] == "(":
                # TABLE(...), XMLTABLE(...), LATERAL(...) — checked before the
                # clause words, since LATERAL is one and must not end the list
                j = self._after(j + 1)
            elif tok in _PUNCT or self.words[j] in _CLAUSE_WORDS:
                break
            else:
                found.append(j)
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

    def _sources(self) -> list[tuple[int, str]]:
        """``(where, clause)`` of every table the statement reads from."""
        found: list[tuple[int, str]] = []
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
                found += [(j, "from") for j in self._items(i + 1, many=True)]
            elif word == "join":
                found += [(j, "join") for j in self._items(i + 1, many=False)]
            elif word == "using" and verb == "merge":
                found += [(j, "using") for j in self._items(i + 1, many=False)]
        return found

    def _targets(self) -> list[tuple[int, str]]:
        """``(where, verb)`` of every table the statement creates, changes or removes."""
        at, verb = self._verb()
        if verb is None:
            return []
        found: list[int] = []
        if verb in {"insert", "merge"}:
            # INSERT ALL INTO a ... INTO b ... names several targets.
            found = [
                j + 1
                for j in range(at, self.n - 1)
                if self.words[j] == "into" and self.depth[j] == 0 and self._is_name(j + 1)
            ]
        elif verb == "update" and self._is_name(at + 1):
            found = [at + 1]
        elif verb == "delete":
            j = at + 1
            if j < self.n and self.words[j] == "from":
                j += 1
            if self._is_name(j):
                found = [j]
        elif verb == "create":
            j = at + 1
            while j < self.n and self.words[j] in _CREATE_MODIFIERS:
                j += 1
            if j < self.n and self.words[j] in {"table", "view"} and self._is_name(j + 1):
                found = [j + 1]
        elif verb in {"truncate", "drop", "alter"}:
            if (
                at + 1 < self.n
                and self.words[at + 1] in {"table", "view"}
                and self._is_name(at + 2)
            ):
                found = [at + 2]
                # SAS drops a list: DROP TABLE a, b;
                j = at + 2
                while (
                    verb == "drop"
                    and self.dialect is Dialect.SAS
                    and j + 2 < self.n
                    and self.toks[j + 1] == ","
                    and self._is_name(j + 2)
                ):
                    j += 2
                    found.append(j)
        return [(j, verb) for j in found]

    def _is_table(self, ref: str) -> bool:
        if self.dialect is Dialect.SAS:
            low = ref.lower()
            return low not in self.ctes and not low.startswith("dictionary.")
        db_schema, table, _ = split_table_name(ref)
        if db_schema is None and table in self.ctes:
            return False
        return not (table == "dual" and db_schema in {None, "sys"})

    # -- the scan ----------------------------------------------------------

    def reads(self) -> list[str]:
        """Every table the statement reads, first-seen order."""
        return [r for j, _ in self._sources() if self._is_table(r := self.toks[j])]

    def writes(self) -> list[str]:
        """Every table the statement creates, changes or removes."""
        return [r for j, _ in self._targets() if self._is_table(r := self.toks[j])]

    def refs(self) -> list[tuple[str, DatasetRole, str]]:
        """``(name, role, clause)`` of every table the statement names, in
        source order: FROM and JOIN read; CREATE TABLE|VIEW writes, and reads
        its LIKE table; INSERT, UPDATE, DELETE and ALTER rewrite one in place
        (UPDATE); DROP deletes. ``CREATE INDEX`` and ``DESCRIBE`` name none."""
        found: list[tuple[int, DatasetRole, str]] = [
            (j, READ, clause) for j, clause in self._sources()
        ]
        for j, verb in self._targets():
            if verb == "create":
                found.append((j, WRITE, verb))
                k = j + 1
                if k < self.n and self.toks[k] == "(":  # the new table's options
                    k = self._after(k)
                if k < self.n and self.words[k] == "like" and self._is_name(k + 1):
                    found.append((k + 1, READ, "like"))
            else:
                found.append((j, DROP if verb == "drop" else UPDATE, verb))
        found.sort(key=lambda item: item[0])
        return [
            (self.toks[j], role, clause)
            for j, role, clause in found
            if self._is_table(self.toks[j])
        ]
