"""Whether generated Spark SQL names its permanent tables through session variables.

The Spark SQL instructions (``prompt_builder/instructions/sparksql/``) ask the
model to name every permanent table — one in a SAS library other than WORK,
or a database table — through Databricks SQL session variables, so that the
catalog and schema a table lands in are set in one place:

    DECLARE OR REPLACE VARIABLE MART_PREFIX STRING DEFAULT 'catalog_name.mart';
    DECLARE OR REPLACE VARIABLE DEMO_TABLE STRING
        DEFAULT CONCAT(session.MART_PREFIX, '.demo_table');
    CREATE OR REPLACE TABLE IDENTIFIER(session.DEMO_TABLE) AS SELECT ... ;

A session variable needs Databricks Runtime 14.1 or above ("DECLARE
VARIABLE"); the IDENTIFIER clause takes it wherever a name goes, the target
of CREATE, INSERT, MERGE, UPDATE, DELETE, COPY INTO and DROP included
("IDENTIFIER clause"). :func:`check_table_names` reads one answer's SQL, in
order, and judges each table reference the convention covers:

- a reference through ``IDENTIFIER(...)`` passes when every variable it reads
  is declared earlier in the answer, and when it is one variable, that
  variable's value is built from another (its library's prefix) rather than
  being a literal name;
- a literal qualified name (``main.sales.orders``, ``mart.orders``) fails;
- an unqualified literal written as a permanent table fails — the target of
  CREATE TABLE (or a view that is not temporary), INSERT, MERGE, UPDATE,
  DELETE, COPY INTO, DROP TABLE, TRUNCATE or ALTER.

What is not a permanent table is left alone: temporary views (a WORK dataset)
and the CTEs and views a query reads by a bare name, table functions
(``read_files``), Databricks' own ``system`` catalog and
``information_schema``, and a file read in place (``delta.`/Volumes/...```).

Logger name: ``validation.table_names``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from target_language import SPARKSQL, mask_identifier_clauses

if TYPE_CHECKING:  # sqlglot is imported where a parse needs it
    from sqlglot import exp

logger = logging.getLogger(__name__)

_DIALECT = SPARKSQL.sqlglot_dialect or "databricks"

# The file formats Spark reads a path in place with: `delta.`/Volumes/...``.
_PATH_FORMATS = frozenset(
    {"delta", "parquet", "csv", "json", "orc", "avro", "text", "binaryfile"}
)

# A DECLARE sqlglot reads only as a command — a qualified name, `DECLARE OR
# REPLACE session.x ...`: its names, and its default after DEFAULT or `=`.
_DECLARE_NAMES_RE = re.compile(
    r"\s*(?:OR\s+REPLACE\s+)?(?:VARIABLE\s+)?(?P<names>[\w.]+(?:\s*,\s*[\w.]+)*)",
    re.IGNORECASE,
)
_DECLARE_DEFAULT_RE = re.compile(r"(?:\bDEFAULT\b|=)\s*(?P<default>.+?)\s*;?\s*\Z", re.I | re.S)

# What a statement creates, alters or drops when it is no table: a schema or a
# catalog names no table, and neither does a function.
_NOT_TABLES = frozenset({"SCHEMA", "DATABASE", "CATALOG", "FUNCTION"})


@dataclass
class TableNameCheck:
    """What :func:`check_table_names` found in one answer.

    Attributes
    ----------
    checked
        The table references the convention covers.
    problems
        One line per reference that breaks it, in the answer's order, saying
        what to change.
    unparsed
        Blocks sqlglot could not read; ``target_syntax`` reports why.
    """

    checked: int = 0
    problems: list[str] = field(default_factory=list)
    unparsed: int = 0


def _variable(column: object) -> str | None:
    """The session variable *column* reads — ``session.x``,
    ``system.session.x`` or a bare ``x`` — upper-cased, or ``None``."""
    from sqlglot import exp

    if not isinstance(column, exp.Column):
        return None
    qualifier = (column.text("db").lower(), column.table.lower())
    if qualifier in (("", ""), ("", "session"), ("system", "session")):
        return column.name.upper()
    return None


def _variables(expression: object) -> list[str]:
    """Every session variable *expression* reads, in order."""
    from sqlglot import exp

    if not isinstance(expression, exp.Expr):
        return []
    found = [_variable(c) for c in expression.find_all(exp.Column)]
    return [name for name in found if name]


def _declared_name(node: object) -> str:
    """The variable name a DECLARE item names, its ``session.`` dropped."""
    return str(getattr(node, "name", node)).rsplit(".", 1)[-1].upper()


def _kind(statement: exp.Expr) -> str:
    """What *statement* creates, alters or drops (``TABLE``, ``VIEW``, ...)."""
    return str(statement.args.get("kind") or "").upper()


def _target_table(statement: exp.Expr) -> exp.Table | None:
    """The table *statement* writes, when it writes one."""
    from sqlglot import exp

    if isinstance(statement, exp.TruncateTable):
        node = statement.expressions[0] if statement.expressions else None
    elif isinstance(statement, (exp.Insert, exp.Merge, exp.Update, exp.Delete, exp.Copy)):
        node = statement.this
    elif isinstance(statement, (exp.Create, exp.Alter)) and _kind(statement) in ("TABLE", "VIEW", ""):
        node = statement.this
    elif isinstance(statement, exp.Drop) and _kind(statement) == "TABLE":
        node = statement.this
    else:
        return None
    if isinstance(node, exp.Schema):
        node = node.this
    return node if isinstance(node, exp.Table) else None


def _temporary_view(statement: exp.Expr) -> bool:
    """Whether *statement* creates a temporary view: a WORK dataset."""
    from sqlglot import exp

    if not isinstance(statement, exp.Create) or _kind(statement) != "VIEW":
        return False
    properties = statement.args.get("properties")
    return bool(
        properties and any(isinstance(p, exp.TemporaryProperty) for p in properties.expressions)
    )


class _Answer:
    """One answer's session variables as its statements declare and set them."""

    def __init__(self) -> None:
        # Declared variable -> the expression it holds (None until it has one).
        self.values: dict[str, object | None] = {}
        self.temp_views: set[str] = set()
        # The IDENTIFIER clauses of the block being read, by placeholder.
        self.identifiers: dict[str, str] = {}
        self.result = TableNameCheck()

    def declare(self, names: Iterable[str], default: object | None) -> None:
        for name in names:
            self.values[name] = default

    def set(self, name: str, value: object) -> None:
        if name in self.values:  # `SET x = ...` on no variable is Spark config
            self.values[name] = value

    def statement(self, statement: exp.Expr) -> None:
        from sqlglot import exp

        if isinstance(statement, exp.Declare):
            for item in statement.expressions:
                names = item.this if isinstance(item.this, list) else [item.this]
                # sqlglot keeps `default=False` for a DECLARE with none.
                default = item.args.get("default")
                self.declare(
                    map(_declared_name, names),
                    default if isinstance(default, exp.Expr) else None,
                )
            return
        if isinstance(statement, exp.Command) and statement.name.upper() == "DECLARE":
            self._declare_command(str(statement.expression or ""))
            return
        if isinstance(statement, exp.Set):
            for item in statement.expressions:
                eq = item.this
                if isinstance(eq, exp.EQ) and (name := _variable(eq.this)):
                    self.set(name, eq.expression)
            return
        if isinstance(statement, exp.Use) or _kind(statement) in _NOT_TABLES:
            return  # USE CATALOG / SCHEMA, CREATE SCHEMA, ...: no table
        target = _target_table(statement)
        if _temporary_view(statement) and isinstance(target, exp.Table):
            # A WORK dataset: no permanent write, though its query still reads.
            self.temp_views.add(target.name.lower())
            target = None
        for table in statement.find_all(exp.Table):
            self._table(table, writes=table is target)

    def _declare_command(self, text: str) -> None:
        import sqlglot

        names = _DECLARE_NAMES_RE.match(text)
        if names is None:
            return
        default = None
        if found := _DECLARE_DEFAULT_RE.search(text, names.end()):
            try:
                default = sqlglot.parse_one(found["default"], read=_DIALECT)
            except Exception:  # sqlglot raises several unrelated types
                default = None
        self.declare(map(_declared_name, re.split(r"\s*,\s*", names["names"])), default)

    def _table(self, table: exp.Table, *, writes: bool) -> None:
        from sqlglot import exp

        if not isinstance(table.this, exp.Identifier):
            return  # a table function: read_files(...)
        name, db, catalog = table.name, table.db, table.catalog
        if name in self.identifiers:
            self.result.checked += 1
            self._identifier(self.identifiers[name])
            return
        if db or catalog:
            if catalog.lower() == "system" or db.lower() == "information_schema":
                return
            if db.lower() in _PATH_FORMATS and ("/" in name or ":" in name):
                return  # a file read in place: a path, not a table
            self.result.checked += 1
            spelled = ".".join(p for p in (catalog, db, name) if p)
            self._problem(
                f"hard-coded table name `{spelled}`: declare it as a session "
                f"variable and write IDENTIFIER(session.<VAR>)"
            )
            return
        if writes and name.lower() not in self.temp_views:
            self.result.checked += 1
            self._problem(
                f"`{name}` is written as a permanent table by a literal name: "
                f"write IDENTIFIER(session.<VAR>), or CREATE OR REPLACE TEMP "
                f"VIEW for a WORK dataset"
            )

    def _identifier(self, clause: str) -> None:
        import sqlglot

        inner = clause[clause.index("(") + 1 : clause.rindex(")")]
        try:
            expression = sqlglot.parse_one(inner, read=_DIALECT)
        except Exception:  # sqlglot raises several unrelated types
            self._problem(f"{clause} could not be read")
            return
        variables = _variables(expression)
        if not variables:
            self._problem(
                f"{clause} hard-codes the name: put it in a session variable"
            )
            return
        undeclared = [v for v in variables if v not in self.values]
        if undeclared:
            self._problem(
                f"`session.{undeclared[0]}` is not declared before it is used in "
                f"this answer: DECLARE OR REPLACE VARIABLE it first"
            )
            return
        if len(variables) == 1 and _variable(expression) == variables[0]:
            value = self.values[variables[0]]
            if value is None:
                self._problem(f"`{variables[0]}` is never given a value")
            elif not [v for v in _variables(value) if v != variables[0]]:
                self._problem(
                    f"`{variables[0]}` holds a literal name: build it from its "
                    f"library's prefix, CONCAT(session.<LIBREF>_PREFIX, "
                    f"'.<table>')"
                )

    def _problem(self, line: str) -> None:
        self.result.problems.append(line)


def check_table_names(blocks: Iterable[str]) -> TableNameCheck:
    """How one answer's SQL *blocks* name their permanent tables, read in
    order: a variable is usable once a block before it, or a statement
    earlier in its own, has declared it (see the module docstring)."""
    import sqlglot

    answer = _Answer()
    for block in blocks:
        masked, identifiers = mask_identifier_clauses(block)
        try:
            statements = sqlglot.parse(masked, read=_DIALECT)
        except Exception:  # sqlglot raises several unrelated types
            answer.result.unparsed += 1
            continue
        answer.identifiers = identifiers
        for statement in statements:
            if statement is not None:
                answer.statement(statement)
    if answer.result.problems and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"check_table_names: {answer.result.problems}")
    return answer.result
