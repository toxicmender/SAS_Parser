"""Rewriting table names and physical paths in *generated* code, after conversion.

Two languages, two parsers, one rule.

Spark SQL
    ``sqlglot.parse`` under the configured dialect, walk every ``exp.Table``,
    and replace each name whose ``db.name`` matches a mapping key where it
    stands in the source — sqlglot records every identifier's offsets, so
    nothing is regenerated.

PySpark
    The ``ast`` module over the string literals that name tables:
    ``spark.table("x")``, ``spark.read.table("x")``, ``.saveAsTable("x")``, and
    ``spark.sql("...")`` — the last recursing into the sqlglot path, since the
    SQL inside it is still SQL. Rewriting is done by **source-span
    substitution** rather than ``ast.unparse``, so comments, formatting and
    string quoting survive untouched; only the matched literals change.

Paths, not just tables
----------------------
``xref.pre`` remaps a ``LIBNAME`` / ``INFILE`` / ``%INCLUDE`` path in the SAS
source before conversion. A path can still reach the generated code — no mapping
row covered it when ``pre`` ran, or the model wrote one of its own — so the post
pass rewrites those too: :func:`rewrite_python_paths` over the reader/writer and
``dbutils.fs`` calls, :func:`rewrite_sql_paths` over ``LOCATION`` and friends.
Both resolve through :mod:`xref.mapping`, the same longest-prefix resolver
``pre`` uses, so one path cannot come out of the two halves as two targets.

The hard rule
-------------
**Unparseable input is left exactly as the model wrote it.** A rewriter that
corrupts generated code is worse than one that no-ops: the no-op leaves a
reviewer with working code and a wrong table name, which is visible and
fixable, while a corrupted file is neither. ``xref.on_parse_failure`` chooses
between a WARNING (default) and raising, and nothing in between.

``sqlglot`` is imported lazily — it is the optional ``sql`` extra — so this
module costs nothing to import and only the ``"post"`` mode needs it
installed.

Logger name: ``xref.rewrite``.
"""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import app_config
from target_language import (
    SPARKSQL,
    mask_identifier_clauses,
    restore_identifier_clauses,
    session_variable,
    unqualify_declared_names,
)

from .mapping import map_path, ordered_keys

if TYPE_CHECKING:  # sqlglot is imported where a parse needs it
    from sqlglot import exp
    from sqlglot.tokenizer_core import Token

logger = logging.getLogger(__name__)

ON_PARSE_FAILURE_WARN = "warn"
ON_PARSE_FAILURE_ERROR = "error"
_ON_PARSE_FAILURE = (ON_PARSE_FAILURE_WARN, ON_PARSE_FAILURE_ERROR)

# Attribute calls whose first string argument names a table.
_TABLE_CALLS = frozenset({"table", "saveAsTable", "insertInto", "createOrReplaceTempView"})
# Attribute calls whose first string argument is SQL to recurse into.
_SQL_CALLS = frozenset({"sql"})

# Attribute calls whose first string argument is a *path* rather than a table:
# the DataFrame reader/writer formats, and the dbutils filesystem utilities.
# ``head`` and ``text`` are also DataFrame methods that take something else —
# the first-argument-is-a-string test and the mapping lookup both have to pass
# before anything is rewritten, so a `df.head(5)` is never touched.
_PATH_CALLS = frozenset(
    {
        # DataFrameReader / DataFrameWriter
        "csv", "parquet", "json", "orc", "text", "load", "save",
        # dbutils.fs
        "ls", "cp", "mv", "rm", "mkdirs", "mount", "head", "put",
    }
)
# Builtins taking a path as their first argument.
_PATH_BUILTINS = frozenset({"open"})
# ``.option("path", "<path>")`` — the one form whose path is the *second*
# argument, so it needs its own branch rather than joining _PATH_CALLS.
_OPTION_CALL = "option"
_OPTION_PATH_KEYS = frozenset({"path"})


class XrefRewriteError(RuntimeError):
    """Generated code could not be parsed and ``xref.on_parse_failure`` is
    ``"error"``. The output is unchanged either way — this only decides
    whether the run stops."""


def default_dialect() -> str:
    """The run's target language, as a sqlglot dialect.

    Resolved from ``pipeline.output_language`` rather than held as a constant
    here: the syntax checker parses generated SQL under the *same* target's
    :attr:`~target_language.TargetLanguage.sqlglot_dialect`, and a rewriter
    reading it differently would silently return code un-rewritten (the hard
    rule above) on exactly the syntax the checker had just called valid.

    A non-SQL target (PySpark, Spark Scala) has no dialect of its own, but this
    module still reaches sqlglot for the SQL inside ``spark.sql("...")`` — so
    those fall back to the SQL target's dialect rather than to nothing.
    """
    return SPARKSQL.sqlglot_dialect or "databricks"

def on_parse_failure() -> str:
    """What to do with unparseable generated code (``xref.on_parse_failure``).

    An unrecognised value degrades to ``"warn"`` with a WARNING, the same
    degrade-don't-crash rule the rest of the config follows — and the safer
    direction, since ``"warn"`` never fails a run.
    """
    configured = app_config.get_typed_value(
        "xref", "on_parse_failure", str, ON_PARSE_FAILURE_WARN
    )
    if configured not in _ON_PARSE_FAILURE:
        logger.warning(
            f"on_parse_failure: config.json xref.on_parse_failure "
            f"{configured!r} is not one of {'/'.join(_ON_PARSE_FAILURE)}; "
            f"ignoring it ({ON_PARSE_FAILURE_WARN!r} applies)"
        )
        return ON_PARSE_FAILURE_WARN
    return configured


def _unparseable(what: str, exc: Exception, mode: str | None) -> None:
    """Handle a parse failure per *mode*, having changed nothing."""
    resolved = mode if mode in _ON_PARSE_FAILURE else on_parse_failure()
    message = (
        f"could not parse the generated {what} ({exc}); leaving it exactly as "
        f"the model wrote it — no XREF substitution was applied"
    )
    if resolved == ON_PARSE_FAILURE_ERROR:
        raise XrefRewriteError(message) from exc
    logger.warning(f"rewrite: {message}")


def _match(name: str, mapping: dict[str, str]) -> tuple[str, str] | None:
    """The mapping key a table name matches, and its target, case-insensitively.

    Both the qualified name (``schema.table``) and the bare table name are
    tried, in that order: a mapping is written against the SAS-side
    ``schema.table``, but generated code may name the table alone.
    """
    lowered = name.strip().lower()
    hit = mapping.get(lowered)
    if hit is not None:
        return lowered, hit
    _, dot, bare = lowered.rpartition(".")
    if dot and bare and (hit := mapping.get(bare)) is not None:
        return bare, hit
    return None


def _lookup(name: str, mapping: dict[str, str]) -> str | None:
    """The target for a table name (see :func:`_match`)."""
    hit = _match(name, mapping)
    return hit[1] if hit else None


def _name_span(table: exp.Table) -> tuple[int, int] | None:
    """Where *table*'s name stands in the text it was parsed from, from its
    first name part to its last (end exclusive), or ``None`` when a part has
    no recorded position. sqlglot keeps each identifier's offsets, quotes
    included, in its ``meta``."""
    from sqlglot import exp

    parts = [
        part
        for part in (table.args.get("catalog"), table.args.get("db"), table.this)
        if part is not None
    ]
    if not parts or not all(
        isinstance(part, exp.Identifier) and "start" in part.meta for part in parts
    ):
        return None
    return parts[0].meta["start"], parts[-1].meta["end"] + 1


@dataclass
class _Cell:
    """One SQL cell as the rewrite reads it: masked, parsed, and the edits
    gathered for it."""

    masked: str
    identifiers: dict[str, str]
    statements: list[exp.Expr]
    edits: list[tuple[int, int, str]] = field(default_factory=list)
    tokens: list[Token] | None = None


@dataclass(eq=False)
class _Dependent:
    """A table name built from a prefix, ``CONCAT(session.P, '.member')``.

    ``column`` and ``suffix`` are the ``session.P`` reference and the
    ``'.member'`` literal; both are ``None`` for a name built inside an
    IDENTIFIER clause, which the rewrite cannot reach. ``active`` is each
    variable's prefix where the name is read, so a group is re-pointed only
    to a prefix that still holds what it held when it was chosen.
    """

    cell: int
    column: exp.Column | None
    suffix: exp.Literal | None
    member: str
    key: str | None
    target: tuple[str, str, str] | None
    active: dict[str, _Prefix] = field(default_factory=dict)


@dataclass(eq=False)
class _Prefix:
    """A session variable given ``'catalog.schema'``: the prefix the names of
    a library's tables are built from."""

    cell: int
    name: str
    literal: exp.Literal
    catalog: str
    schema: str
    dependents: list[_Dependent] = field(default_factory=list)
    # Something other than CONCAT(session.P, '.member') reads the variable,
    # so its value must stay where it is.
    pinned: bool = False
    # Where every table built from it moves, when its own literal moves there.
    moves_to: tuple[str, str] | None = None

    @property
    def home(self) -> tuple[str, str]:
        """The ``(catalog, schema)`` the prefix holds once the rewrite is done."""
        return self.moves_to or (self.catalog, self.schema)


def _string_value(node: object) -> str | None:
    """The text a string literal holds, or ``None``."""
    from sqlglot import exp

    if isinstance(node, exp.Literal) and node.is_string:
        return node.this
    return None


def _evaluate(node: object, values: dict[str, str | None]) -> str | None:
    """The string *node* builds from literals, session variables, ``CONCAT``
    and ``||``, or ``None`` when any part of it is unknown."""
    from sqlglot import exp

    if (text := _string_value(node)) is not None:
        return text
    if isinstance(node, exp.Paren):
        return _evaluate(node.this, values)
    if isinstance(node, exp.Column):
        return values.get(session_variable(node) or "")
    if isinstance(node, exp.DPipe):
        parts = [node.this, node.expression]
    elif isinstance(node, exp.Concat):
        parts = node.expressions
    else:
        return None
    texts = [_evaluate(part, values) for part in parts]
    return None if any(text is None for text in texts) else "".join(texts)  # type: ignore[arg-type]


def _assignments(statement: exp.Expr) -> list[tuple[str, exp.Expr | None, bool]]:
    """``(variable, value, shared)`` for each value *statement* gives a
    session variable: a ``DECLARE``'s default, or a ``SET VAR``. ``shared``
    marks a ``DECLARE a, b ... DEFAULT x``, whose one literal is two values."""
    from sqlglot import exp

    found: list[tuple[str, exp.Expr | None, bool]] = []
    if isinstance(statement, exp.Declare):
        for item in statement.expressions:
            names = item.this if isinstance(item.this, list) else [item.this]
            default = item.args.get("default")
            value = default if isinstance(default, exp.Expr) else None
            for name in names:
                found.append((str(getattr(name, "name", name)).upper(), value, len(names) > 1))
    elif (
        isinstance(statement, exp.Set)
        and statement.expressions
        and statement.expressions[0].args.get("kind") == "VARIABLE"
    ):
        for item in statement.expressions:
            eq = item.this
            if isinstance(eq, exp.EQ) and (name := session_variable(eq.this)):
                found.append((name, eq.expression, False))
    return found


def _built_from_prefix(
    value: object, prefix_of: dict[str, _Prefix]
) -> tuple[_Prefix, exp.Column, exp.Literal, str] | None:
    """``(prefix, column, suffix, member)`` when *value* is
    ``CONCAT(session.P, '.member')`` or ``session.P || '.member'`` and ``P``
    holds a prefix."""
    from sqlglot import exp

    if isinstance(value, exp.Concat) and len(value.expressions) == 2:
        column, suffix = value.expressions
    elif isinstance(value, exp.DPipe):
        column, suffix = value.this, value.expression
    else:
        return None
    member = _string_value(suffix)
    if not isinstance(column, exp.Column) or not isinstance(suffix, exp.Literal) or member is None:
        return None
    prefix = prefix_of.get(session_variable(column) or "")
    if prefix is None or not member.startswith(".") or "." in member[1:] or len(member) < 2:
        return None
    return prefix, column, suffix, member[1:]


def _literal(text: str, read: str) -> str:
    from sqlglot import exp

    return exp.Literal.string(text).sql(dialect=read)


def _span(node: exp.Expr) -> tuple[int, int] | None:
    """Where a parsed identifier or literal stands in the source, end
    exclusive, or ``None`` when sqlglot recorded no position."""
    meta = node.meta
    return (meta["start"], meta["end"] + 1) if "start" in meta else None


class _VariableRewrite:
    """The edits that move table names held in session variables to their
    mapped targets, across one item's SQL cells in the order they run.

    Each variable's value is evaluated as the item runs. A table variable
    built as ``CONCAT(session.P, '.member')`` from a prefix ``P`` holding
    ``'catalog.schema'`` is looked up as ``schema.member``. Then, per prefix:

    * every table built from it mapped to one ``catalog.schema``, and nothing
      else reading it: the prefix's own literal changes;
    * otherwise each mapped group gets a prefix of its own — an existing one
      that holds that ``catalog.schema`` wherever the group reads it, once the
      rewrite is done, or a new ``<SCHEMA>_PREFIX`` declared right after the
      original — and the group's tables are re-pointed to it;
    * a mapping that renames the table rewrites its ``'.member'`` literal.

    Anything else that reads a prefix — a query, a command's text, an
    IDENTIFIER clause that builds some other name or names the prefix itself —
    keeps the prefix where it is, so moving its tables never moves what that
    reads.

    A table variable holding a literal ``'catalog.schema.table'`` gets the
    target literal. A mapped name built any other way — inside an IDENTIFIER
    clause, or from more than a prefix and a member — is reported, not
    rewritten.
    """

    def __init__(self, cells: list[_Cell | None], mapping: dict[str, str], read: str) -> None:
        self.cells = cells
        self.mapping = mapping
        self.read = read
        self.prefixes: list[_Prefix] = []
        self.declared: set[str] = set()
        self.remapped: set[str] = set()
        # New prefixes declared by this rewrite: (catalog, schema, cell, offset, name).
        self.added: list[tuple[str, str, int, int, str]] = []

    def run(self) -> list[str]:
        from sqlglot import exp

        values: dict[str, str | None] = {}
        prefix_of: dict[str, _Prefix] = {}
        shared_read: set[int] = set()
        for index, cell in enumerate(self.cells):
            if cell is None:
                continue
            for statement in cell.statements:
                assignments = _assignments(statement)
                for name, value, shared in assignments:
                    self.declared.add(name)
                    prefix_of.pop(name, None)
                    text = _evaluate(value, values)
                    # `DECLARE a, b ... DEFAULT x`: one value, read (and edited) once.
                    if not shared or id(value) not in shared_read:
                        self._assign(index, name, value, text, shared, prefix_of)
                    if shared:
                        shared_read.add(id(value))
                    values[name] = text
                if isinstance(statement, exp.Command):
                    rest = statement.expression
                    command = str(rest.this) if isinstance(rest, exp.Literal) else str(rest or "")
                    self._pin_text(command, cell, prefix_of)
                elif not assignments:
                    self._pin(statement, prefix_of)
                self._inline(index, cell, statement, values, prefix_of)
        # Every prefix's final value is settled before any group is re-pointed,
        # so a group never follows a prefix that is about to move away.
        for prefix in self.prefixes:
            self._plan(prefix)
        for prefix in self.prefixes:
            self._settle(prefix)
        return sorted(self.remapped)

    # ── reading ─────────────────────────────────────────────────────────────

    def _assign(
        self,
        index: int,
        name: str,
        value: exp.Expr | None,
        text: str | None,
        shared: bool,
        prefix_of: dict[str, _Prefix],
    ) -> None:
        built = _built_from_prefix(value, prefix_of)
        if built is not None:
            prefix, column, suffix, member = built
            prefix.dependents.append(self._dependent(index, column, suffix, prefix, member, prefix_of))
            return
        self._pin(value, prefix_of)
        literal = value if _string_value(value) is not None else None
        catalog, _, schema = (text or "").partition(".")
        if literal is not None and catalog and schema and "." not in schema and not shared:
            prefix = _Prefix(index, name, literal, catalog, schema)  # type: ignore[arg-type]
            self.prefixes.append(prefix)
            prefix_of[name] = prefix
        elif text is not None and text.count(".") == 2:
            self._whole_name(index, literal if not shared else None, text, name)

    def _inline(
        self,
        index: int,
        cell: _Cell,
        statement: exp.Expr,
        values: dict[str, str | None],
        prefix_of: dict[str, _Prefix],
    ) -> None:
        """The statement's IDENTIFIER clauses, wherever they stand: a table
        name built in one from a prefix is a table variable that has no
        variable; any other read of a prefix keeps the prefix in place."""
        import sqlglot
        from sqlglot import exp

        seen: set[str] = set()
        for node in statement.walk():
            if not isinstance(node, (exp.Table, exp.Column, exp.Anonymous)):
                continue
            clause = cell.identifiers.get(node.name)
            if clause is None or node.name in seen:
                continue
            seen.add(node.name)
            try:
                inner = sqlglot.parse_one(clause[clause.index("(") + 1 : clause.rindex(")")], read=self.read)
            except Exception:  # sqlglot raises several unrelated types
                self._pin_text(clause, cell, prefix_of)
                continue
            if session_variable(inner) is not None:
                # IDENTIFIER(session.T): T's own value is rewritten, and a
                # prefix named whole stays where it is.
                self._pin(inner, prefix_of)
                continue
            names_table = isinstance(node, exp.Table)
            built = _built_from_prefix(inner, prefix_of) if names_table else None
            if built is not None:
                prefix, _, _, member = built
                prefix.dependents.append(self._dependent(index, None, None, prefix, member, prefix_of))
                continue
            self._pin(inner, prefix_of)
            if names_table and (text := _evaluate(inner, values)) is not None and text.count(".") == 2:
                self._whole_name(index, None, text, clause)

    @staticmethod
    def _pin(expression: object, prefix_of: dict[str, _Prefix]) -> None:
        """Keep in place every prefix *expression* reads."""
        from sqlglot import exp

        if not isinstance(expression, exp.Expr):
            return
        for column in expression.find_all(exp.Column):
            if (prefix := prefix_of.get(session_variable(column) or "")) is not None:
                prefix.pinned = True

    def _pin_text(self, text: str, cell: _Cell, prefix_of: dict[str, _Prefix]) -> None:
        """Keep in place every prefix raw SQL *text* names: what follows a
        command's keyword, which sqlglot keeps as one string (``EXECUTE
        IMMEDIATE``, ``SHOW``), or a clause it cannot read."""
        from sqlglot.dialects.dialect import Dialect

        try:
            tokens = Dialect.get_or_raise(self.read).tokenize(text)
            words = [t.text for t in tokens if "STRING" not in t.token_type.name]
        except Exception:  # an unclosed literal or comment: read it as words
            words = re.findall(r"\w+", text)
        for word in words:
            if (prefix := prefix_of.get(word.upper())) is not None:
                prefix.pinned = True
            elif (clause := cell.identifiers.get(word.lower())) is not None:
                self._pin_text(clause, cell, prefix_of)

    def _dependent(
        self,
        index: int,
        column: exp.Column | None,
        suffix: exp.Literal | None,
        prefix: _Prefix,
        member: str,
        prefix_of: dict[str, _Prefix],
    ) -> _Dependent:
        hit = _match(f"{prefix.schema}.{member}", self.mapping)
        if hit is None:
            return _Dependent(index, column, suffix, member, None, None)
        key, target = hit
        parts = target.split(".")
        if len(parts) != 3:
            logger.warning(
                f"rewrite_sql: {key!r} maps to {target!r}, which is not "
                f"catalog.schema.table; the session variable that builds it "
                f"is left as written"
            )
            return _Dependent(index, column, suffix, member, None, None)
        return _Dependent(
            index, column, suffix, member, key, (parts[0], parts[1], parts[2]), dict(prefix_of)
        )

    def _whole_name(self, index: int, literal: exp.Expr | None, text: str, holder: str) -> None:
        """A table name held whole: rewrite its literal, or report it."""
        hit = _match(text.split(".", 1)[1], self.mapping)
        if hit is None:
            return
        key, target = hit
        if target.lower() == text.lower():
            return
        cell = self.cells[index]
        span = _span(literal) if literal is not None else None
        if cell is None or span is None:
            logger.warning(
                f"rewrite_sql: {text!r} maps to {target!r}, but {holder} builds "
                f"it in a way the post rewrite does not change; left as written"
            )
            return
        cell.edits.append((*span, _literal(target, self.read)))
        self.remapped.add(key)

    # ── rewriting ───────────────────────────────────────────────────────────

    def _plan(self, prefix: _Prefix) -> None:
        """Whether every table built from *prefix* moves to one
        ``catalog.schema`` and nothing else reads it, so that its own literal
        moves there (``moves_to``)."""
        mapped = [d for d in prefix.dependents if d.target is not None]
        homes = {(d.target[0], d.target[1]) for d in mapped if d.target}
        if (
            mapped
            and not prefix.pinned
            and len(homes) == 1
            and all(d.target is not None for d in prefix.dependents)
            and all(d.column is not None or (d.target and d.target[2] == d.member) for d in mapped)
        ):
            prefix.moves_to = homes.pop()

    def _settle(self, prefix: _Prefix) -> None:
        mapped = [d for d in prefix.dependents if d.target is not None]
        if not mapped:
            return
        if prefix.moves_to is not None:
            moved = prefix.moves_to != (prefix.catalog, prefix.schema)
            if moved:
                self._edit_literal(prefix.cell, prefix.literal, ".".join(prefix.moves_to))
            for dependent in mapped:
                self._rename_member(dependent, moved=moved)
            return
        homes = sorted({(d.target[0], d.target[1]) for d in mapped if d.target})
        for home in homes:
            group = [d for d in mapped if d.target and (d.target[0], d.target[1]) == home]
            editable = [d for d in group if d.column is not None]
            for dependent in group:
                if dependent.column is None:
                    logger.warning(
                        f"rewrite_sql: {dependent.key!r} maps to "
                        f"{'.'.join(dependent.target or ())!r}, but its name is built "
                        f"inside an IDENTIFIER clause, which the post rewrite does not "
                        f"change; left as written"
                    )
            if not editable:
                continue
            moved = home != (prefix.catalog, prefix.schema)
            if moved:
                name = self._prefix_for(home, prefix, editable)
                for dependent in editable:
                    column = dependent.column
                    span = _span(column.this) if column is not None else None
                    if span is not None:
                        self._edit(dependent.cell, span, name)
            for dependent in editable:
                self._rename_member(dependent, moved=moved)

    def _rename_member(self, dependent: _Dependent, *, moved: bool) -> None:
        """Rewrite the ``'.member'`` a mapping renames, and count the name as
        remapped when it moved or was renamed."""
        assert dependent.target is not None and dependent.key is not None  # a mapped one
        renamed = dependent.target[2] != dependent.member and dependent.suffix is not None
        if renamed and dependent.suffix is not None:
            self._edit_literal(dependent.cell, dependent.suffix, f".{dependent.target[2]}")
        if moved or renamed:
            self.remapped.add(dependent.key)

    def _prefix_for(self, home: tuple[str, str], prefix: _Prefix, group: list[_Dependent]) -> str:
        """The prefix variable *group* is re-pointed to: one that holds *home*
        once the rewrite is done and is the variable's prefix wherever the
        group reads it, or a new one."""
        first = min(
            (d.cell, d.column.this.meta.get("start", 0) if d.column else 0) for d in group
        )
        for other in self.prefixes:
            if other.home == home and all(d.active.get(other.name) is other for d in group):
                return other.name
        for catalog, schema, cell, offset, name in self.added:
            if (catalog, schema) == home and (cell, offset) < first:
                return name
        name = self._new_name(home)
        cell = self.cells[prefix.cell]
        assert cell is not None
        offset, ended = self._statement_end(cell, prefix.literal)
        declare = (
            f"DECLARE OR REPLACE VARIABLE {name} STRING DEFAULT "
            f"{_literal('.'.join(home), self.read)};"
        )
        text = f"\n{declare}" if ended else f";\n{declare}"
        cell.edits.append((offset, offset, text))
        self.added.append((*home, prefix.cell, offset, name))
        self.declared.add(name)
        return name

    def _new_name(self, home: tuple[str, str]) -> str:
        catalog, schema = (re.sub(r"\W+", "_", part).upper() for part in home)
        for name in (f"{schema}_PREFIX", f"{catalog}_{schema}_PREFIX"):
            if name not in self.declared:
                return name
        n = 2
        while f"{catalog}_{schema}_PREFIX_{n}" in self.declared:
            n += 1
        return f"{catalog}_{schema}_PREFIX_{n}"

    def _statement_end(self, cell: _Cell, literal: exp.Expr) -> tuple[int, bool]:
        """Where a statement can follow the one that holds *literal*: just
        past its semicolon (``True``), or, when the cell ends without one,
        just past its last token (``False``), ahead of a comment trailing it."""
        from sqlglot.dialects.dialect import Dialect

        if cell.tokens is None:
            cell.tokens = Dialect.get_or_raise(self.read).tokenize(cell.masked)
        after = literal.meta.get("end", 0)
        for token in cell.tokens:
            if token.token_type.name == "SEMICOLON" and token.start > after:
                return token.end + 1, True
        return cell.tokens[-1].end + 1, False

    def _edit_literal(self, index: int, literal: exp.Expr, text: str) -> None:
        span = _span(literal)
        if span is not None:
            self._edit(index, span, _literal(text, self.read))

    def _edit(self, index: int, span: tuple[int, int], text: str) -> None:
        cell = self.cells[index]
        if cell is not None:
            cell.edits.append((*span, text))


def _parse_cell(sql: str, read: str, on_failure: str | None) -> _Cell | None:
    """*sql* masked and parsed, or ``None`` (reported) when it does not parse."""
    import sqlglot

    masked, identifiers = mask_identifier_clauses(sql, read)
    try:
        statements = sqlglot.parse(unqualify_declared_names(masked, read), read=read)
    except Exception as exc:  # sqlglot raises several unrelated types
        _unparseable("Spark SQL", exc, on_failure)
        return None
    return _Cell(masked, identifiers, [s for s in statements if s is not None])


def _name_edits(cell: _Cell, mapping: dict[str, str], read: str, on_failure: str | None) -> bool:
    """Gather the edits for every mapped table name *cell* writes out. False
    (reported) when one has no recorded position, so the cell stays as is."""
    from sqlglot import exp

    for statement in cell.statements:
        for table in statement.find_all(exp.Table):
            qualified = ".".join(
                part.name for part in (table.args.get("db"), table.this) if part
            )
            target = _lookup(qualified, mapping)
            if target is None:
                continue
            span = _name_span(table)
            if span is None:
                _unparseable(
                    "Spark SQL",
                    ValueError(f"no source position for table {qualified!r}"),
                    on_failure,
                )
                return False
            cell.edits.append((*span, exp.to_table(target, dialect=read).sql(dialect=read)))
    return True


def rewrite_sql_cells(
    cells: Sequence[str], mapping: dict[str, str], *, on_failure: str | None = None
) -> tuple[list[str], list[str]]:
    """
    One item's SQL *cells*, in the order they run, with every mapped table
    name rewritten to its target — and the mapping keys of the names that
    session variables held.

    Two kinds of name are rewritten:

    * **Names written out** (``FROM sales.orders``): each is replaced where
      it stands in the source, so nothing is regenerated and every other
      byte comes through as written — the spelling of a ``DECLARE``,
      comments, a placeholder in function position, and the SQL scripting
      blocks sqlglot reads but cannot write back (``WHILE ... DO``).
    * **Names held in session variables**, the way the Spark SQL
      instructions name permanent tables
      (``DECLARE ... ORDERS STRING DEFAULT CONCAT(session.MART_PREFIX,
      '.orders')`` then ``IDENTIFIER(session.ORDERS)``): see
      :class:`_VariableRewrite`. The cells are read together because a
      prefix declared in one can build the names of the next.

    A cell that does not parse, or that has a mapped name without a recorded
    position, is left exactly as written (``xref.on_parse_failure`` decides
    whether that is a warning or an error); so are all cells when ``sqlglot``
    is not installed.
    """
    out = list(cells)
    if not mapping or not any(cell.strip() for cell in cells):
        return out, []
    try:
        import sqlglot  # noqa: F401  (the import is the availability check)
    except ImportError:
        logger.warning(
            "rewrite_sql: sqlglot is not installed, so the post-conversion "
            "XREF rewrite cannot run; install the 'sql' extra "
            "('pip install \"sas-parser[sql]\"'). The code is unchanged"
        )
        return out, []

    read = default_dialect()
    parsed = [_parse_cell(cell, read, on_failure) if cell.strip() else None for cell in cells]
    for index, cell in enumerate(parsed):
        if cell is not None and not _name_edits(cell, mapping, read, on_failure):
            parsed[index] = None
    remapped = _VariableRewrite(parsed, mapping, read).run()
    for index, cell in enumerate(parsed):
        if cell is None or not cell.edits:
            continue
        edits = _without_overlaps(cell.edits)
        if edits is None:
            _unparseable("Spark SQL", ValueError("two rewrites of the same text overlap"), on_failure)
            continue
        out[index] = restore_identifier_clauses(_apply_edits(cell.masked, edits), cell.identifiers)
    if out != list(cells):
        logger.info(f"rewrite_sql: rewrote table references under dialect {read!r}")
    return out, remapped


def rewrite_sql(
    sql: str, mapping: dict[str, str], *, on_failure: str | None = None
) -> str:
    """
    *sql* with every mapped table reference rewritten to its target: one cell
    of :func:`rewrite_sql_cells`, which says what is rewritten and how.

    Returns *sql* unchanged when it does not parse, when ``sqlglot`` is not
    installed, or when nothing matched.
    """
    return rewrite_sql_cells([sql], mapping, on_failure=on_failure)[0][0]


def _string_literal(node: ast.AST) -> tuple[ast.Constant, str] | None:
    """*node* and its text when it is a plain string constant, else ``None``.

    The text comes back alongside the node because ``ast.Constant.value`` is
    typed as the union of every literal type; returning it here is where the
    narrowing happens, once.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node, node.value
    return None


def _call_name(node: ast.Call) -> str | None:
    """The attribute a call was made through (``spark.table`` -> ``table``)."""
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else None


def rewrite_python(
    source: str, mapping: dict[str, str], *, on_failure: str | None = None
) -> str:
    """
    *source* with the table names in its Spark calls rewritten.

    Only string literals in the recognised call positions are touched
    (:data:`_TABLE_CALLS`, :data:`_SQL_CALLS`), and only by replacing their
    source span — everything else in the file, comments included, comes
    through byte-identical. Returns *source* unchanged when it does not parse
    or nothing matched.
    """
    if not mapping or not source.strip():
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        _unparseable("PySpark", exc, on_failure)
        return source

    offset = _offsetter(source)
    # (start, end, replacement text), collected then applied back-to-front so
    # earlier spans keep their offsets.
    edits: list[tuple[int, int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = _call_name(node)
        if name is None:
            continue
        found = _string_literal(node.args[0])
        if found is None:
            continue
        literal, text = found
        if literal.end_lineno is None or literal.end_col_offset is None:
            continue
        if name in _TABLE_CALLS:
            target = _lookup(text, mapping)
            if target is None:
                continue
            # The literal's own quoting style, like the path half and the SQL
            # branch below. ``repr`` is the fallback for a value whose style
            # cannot be reproduced (it embeds the quote, or a backslash), not
            # the default: normalising every rewritten "x" to 'x' put quoting
            # churn in the diff of a file whose table name was the only change.
            new_text = _requote(source, literal, target) or repr(target)
        elif name in _SQL_CALLS:
            rewritten = rewrite_sql(text, mapping, on_failure=on_failure)
            if rewritten == text:
                continue
            # Keep the literal's own quoting style where it is single-line;
            # repr() would mangle a triple-quoted block.
            new_text = _requote(source, literal, rewritten)
            if new_text is None:
                continue
        else:
            continue
        edits.append(
            (
                offset(literal.lineno, literal.col_offset),
                offset(literal.end_lineno, literal.end_col_offset),
                new_text,
            )
        )

    if not edits:
        return source
    logger.info(f"rewrite_python: rewrote {len(edits)} table reference(s)")
    return _apply_edits(source, edits)


def _offsetter(source: str):
    """A ``(lineno, col) -> character offset`` function for *source*.

    ``ast`` reports columns as utf-8 byte offsets into the line, so the prefix
    is encoded and decoded rather than assumed to agree with the character
    index. Built once per rewrite and shared by every span in it.
    """
    lines = source.splitlines(keepends=True)
    starts: list[int] = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def offset(lineno: int, col: int) -> int:
        line = lines[lineno - 1]
        return starts[lineno - 1] + len(line.encode("utf-8")[:col].decode("utf-8"))

    return offset


def _without_overlaps(edits: list[tuple[int, int, str]]) -> list[tuple[int, int, str]] | None:
    """*edits* with each repeated one kept once, or ``None`` when two of them
    change overlapping text, which applying both would corrupt. Insertions at
    one offset do not overlap; they are applied in a fixed order."""
    unique = sorted(set(edits))
    for (_, end, _), (start, _, _) in zip(unique, unique[1:]):
        if start < end:
            return None
    return unique


def _apply_edits(source: str, edits: list[tuple[int, int, str]]) -> str:
    """*source* with each ``(start, end, replacement)`` span substituted.

    Applied back-to-front so an earlier span's offsets are still valid when it
    is reached. Everything outside the listed spans — comments, formatting,
    quoting — comes through byte-identical, which is what makes this module's
    hard rule (leave unparseable input alone) worth anything: a rewrite that
    reformatted would be indistinguishable from a corruption.
    """
    out = source
    for start, end, replacement in sorted(edits, reverse=True):
        out = out[:start] + replacement + out[end:]
    return out


def _path_literal(node: ast.Call) -> ast.Constant | None:
    """The string literal holding a path in *node*, or ``None``.

    Three call shapes, in the order they are cheapest to reject:
    ``dbutils.fs.ls("<path>")`` and the reader/writer formats
    (:data:`_PATH_CALLS`), ``open("<path>")`` (:data:`_PATH_BUILTINS`, a bare
    name rather than an attribute), and ``.option("path", "<path>")`` — the
    only one whose path is not the first argument.
    """
    if not node.args:
        return None
    func = node.func
    if isinstance(func, ast.Name):
        if func.id not in _PATH_BUILTINS:
            return None
        found = _string_literal(node.args[0])
        return found[0] if found else None
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr == _OPTION_CALL:
        key = _string_literal(node.args[0])
        if not key or key[1].strip().lower() not in _OPTION_PATH_KEYS:
            return None
        if len(node.args) < 2:
            return None
        found = _string_literal(node.args[1])
        return found[0] if found else None
    if func.attr not in _PATH_CALLS:
        return None
    found = _string_literal(node.args[0])
    return found[0] if found else None


def rewrite_python_paths(
    source: str, by_path: dict[str, str], *, on_failure: str | None = None
) -> str:
    """
    *source* with the physical paths in its Spark and filesystem calls remapped.

    The path counterpart of :func:`rewrite_python`: same ``ast`` walk, same
    source-span substitution, same hard rule — unparseable input comes back
    exactly as the model wrote it. Only the literals in the recognised path
    positions (:func:`_path_literal`) are considered, and only those a
    ``by_path`` mapping actually addresses are changed, so a string that merely
    looks like a path is left alone.

    Resolution is :func:`xref.mapping.map_path`, shared with the pre pass, so
    one path cannot be rewritten to two different targets by the two halves.
    """
    if not by_path or not source.strip():
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        _unparseable("PySpark", exc, on_failure)
        return source

    keys = ordered_keys(by_path)
    offset = _offsetter(source)
    edits: list[tuple[int, int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        literal = _path_literal(node)
        if literal is None or literal.end_lineno is None:
            continue
        if literal.end_col_offset is None or not isinstance(literal.value, str):
            continue
        target = map_path(literal.value, by_path, keys)
        if target is None:
            continue
        new_text = _requote(source, literal, target)
        if new_text is None:
            continue
        edits.append(
            (
                offset(literal.lineno, literal.col_offset),
                offset(literal.end_lineno, literal.end_col_offset),
                new_text,
            )
        )

    if not edits:
        return source
    logger.info(f"rewrite_python_paths: rewrote {len(edits)} path(s)")
    return _apply_edits(source, edits)


# Where a physical path appears in generated Spark SQL. Named groups match
# chunker.paths' convention — ``head`` is everything up to the opening quote,
# ``q`` the quote, ``path`` the value — so the substitution is one helper for
# every entry.
#
# Matched by *position* rather than by parsing, unlike the table rewriter above:
# sqlglot regenerates a whole statement to change a node, which reformats
# everything it touches. That is worth paying to re-emit a table as structured
# identifiers; it is not worth paying to change a string literal.
_SQL_PATH_POSITIONS: tuple[re.Pattern[str], ...] = (
    # CREATE TABLE ... LOCATION '<path>'
    re.compile(r"(?P<head>\blocation\s+)(?P<q>['\"])(?P<path>[^'\"\n]*)(?P=q)", re.I),
    # OPTIONS ('path' = '<path>') — the quoted-key form, matched before the bare
    # one so the key is consumed as part of `head` rather than being mistaken
    # for the start of a bare `path` clause.
    re.compile(
        r"(?P<head>['\"]path['\"]\s*=\s*)(?P<q>['\"])(?P<path>[^'\"\n]*)(?P=q)", re.I
    ),
    # OPTIONS (path '<path>') and PATH = '<path>'. A separator is required, so
    # this cannot latch onto the quoted key above and rewrite the ` = ` between
    # them as though it were the value.
    re.compile(
        r"(?P<head>\bpath\s*(?:=\s*|\s+))(?P<q>['\"])(?P<path>[^'\"\n]*)(?P=q)", re.I
    ),
    # COPY INTO ... FROM '<path>' — a quoted literal in FROM position is a
    # location, not a table name (a table there is an identifier).
    re.compile(r"(?P<head>\bfrom\s+)(?P<q>['\"])(?P<path>[^'\"\n]*)(?P=q)", re.I),
    # read_files('<path>')
    re.compile(
        r"(?P<head>\bread_files\s*\(\s*)(?P<q>['\"])(?P<path>[^'\"\n]*)(?P=q)", re.I
    ),
)


def rewrite_sql_paths(sql: str, by_path: dict[str, str]) -> str:
    """
    *sql* with the paths in its location clauses remapped.

    Needs no parser — see :data:`_SQL_PATH_POSITIONS` — so unlike
    :func:`rewrite_sql` this does not degrade when ``sqlglot`` is absent, and
    SQL it cannot make sense of is simply left untouched rather than
    round-tripped.
    """
    if not by_path or not sql.strip():
        return sql

    keys = ordered_keys(by_path)
    changed = 0

    def _substitute(match: re.Match[str]) -> str:
        nonlocal changed
        target = map_path(match.group("path"), by_path, keys)
        if target is None:
            return match.group(0)
        changed += 1
        quote = match.group("q")
        return f"{match.group('head')}{quote}{target}{quote}"

    out = sql
    for pattern in _SQL_PATH_POSITIONS:
        out = pattern.sub(_substitute, out)
    if changed:
        logger.info(f"rewrite_sql_paths: rewrote {changed} path(s)")
    return out


def _requote(source: str, literal: ast.Constant, value: str) -> str | None:
    """*value* re-emitted in the same quoting style the original literal used.

    ``None`` when the original's style cannot be reproduced safely, in which
    case the literal is left alone — the module's rule again: no change beats
    a wrong one.
    """
    if literal.end_lineno is None or literal.end_col_offset is None:
        return None
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    start = starts[literal.lineno - 1] + literal.col_offset
    end = starts[literal.end_lineno - 1] + literal.end_col_offset
    original = source[start:end]
    for quote in ('"""', "'''", '"', "'"):
        if original.startswith(quote) and original.endswith(quote):
            if quote in value or "\\" in value:
                return None  # would need escaping; leave it alone
            return f"{quote}{value}{quote}"
    return None
