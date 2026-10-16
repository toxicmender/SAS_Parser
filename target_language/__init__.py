"""The run's supported target output languages.

Target resolution happens once at each public boundary. Every downstream
consumer receives a TargetLanguage, never a free-form string, so prompting,
notebook rendering, validation, complexity, and XREF agree.
"""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_OUTPUT_LANGUAGE",
    "KNOWN_TARGETS",
    "PROSE_FENCE_INFOS",
    "PYSPARK",
    "SPARKSQL",
    "TargetLanguage",
    "UnknownTargetLanguage",
    "mask_identifier_clauses",
    "normalize_language",
    "resolve_target_language",
    "restore_identifier_clauses",
]

DEFAULT_OUTPUT_LANGUAGE = "Spark SQL"
PROSE_FENCE_INFOS = frozenset(
    {"sas", "text", "txt", "log", "output", "console", "markdown", "md"}
)


class UnknownTargetLanguage(ValueError):
    """Raised when an output language is not a supported target."""


def normalize_language(name: str) -> str:
    """Return the case/space/hyphen/underscore-insensitive comparison key."""
    return re.sub(r"[\s_-]+", "", name.lower())


def _check_python(source: str) -> str | None:
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return f"{exc.msg} (line {exc.lineno})"
    return None


# The IDENTIFIER clause: an object name computed from a constant string — a
# session variable, a parameter marker or a literal — anywhere Databricks SQL
# takes a name ("IDENTIFIER clause", Databricks SQL / Runtime 13.3 and above).
_IDENTIFIER_RE = re.compile(r"IDENTIFIER\s*\(", re.IGNORECASE)
_PLACEHOLDER = "__sasp_identifier_{}__"


def _skip_quoted(sql: str, start: int) -> int:
    """The index after the string literal or backquoted name opening at
    *start*: a backslash escapes the next character in a literal, and a
    doubled backquote stands for one inside a name."""
    quote = sql[start]
    i = start + 1
    while i < len(sql):
        char = sql[i]
        if char == "\\" and quote != "`":
            i += 2
        elif char == quote and quote == "`" and sql.startswith("``", i):
            i += 2
        elif char == quote:
            return i + 1
        else:
            i += 1
    return len(sql)


def _skip_comment(sql: str, start: int) -> int:
    """The index after the comment opening at *start*: ``--`` to the end of
    its line, or a bracketed comment, which nests."""
    if sql.startswith("--", start):
        end = sql.find("\n", start)
        return len(sql) if end < 0 else end + 1
    depth = 0
    i = start
    while i < len(sql):
        if sql.startswith("/*", i):
            depth += 1
            i += 2
        elif sql.startswith("*/", i):
            depth -= 1
            i += 2
            if depth == 0:
                return i
        else:
            i += 1
    return len(sql)


def _closing_paren(sql: str, start: int) -> int:
    """The index of the parenthesis closing the one at *start*, strings and
    comments aside, or ``-1`` when it is never closed."""
    depth = 0
    i = start
    while i < len(sql):
        char = sql[i]
        if char in "'\"`":
            i = _skip_quoted(sql, i)
            continue
        if sql.startswith(("--", "/*"), i):
            i = _skip_comment(sql, i)
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def mask_identifier_clauses(sql: str) -> tuple[str, dict[str, str]]:
    """*sql* with each IDENTIFIER clause replaced by a plain placeholder name,
    and the clause each placeholder stands for.

    sqlglot reads ``SELECT * FROM IDENTIFIER(session.t)`` but not
    ``CREATE TABLE IDENTIFIER(session.t) AS ...``, ``INSERT INTO``, ``COPY
    INTO`` or ``DROP TABLE`` with one, which Databricks accepts — so a parse
    of generated Spark SQL masks them first. A placeholder is an ordinary
    identifier, so the statement around it parses as it would with a literal
    name, and :func:`restore_identifier_clauses` puts the clauses back. A
    clause inside a string or a comment is left alone, as is one never
    closed, which the parse then reports.
    """
    out: list[str] = []
    originals: dict[str, str] = {}
    i = 0
    while i < len(sql):
        char = sql[i]
        if char in "'\"`":
            end = _skip_quoted(sql, i)
        elif sql.startswith(("--", "/*"), i):
            end = _skip_comment(sql, i)
        elif (
            char in "iI"
            and (i == 0 or not (sql[i - 1].isalnum() or sql[i - 1] in "_$"))
            and (match := _IDENTIFIER_RE.match(sql, i)) is not None
            and (close := _closing_paren(sql, match.end() - 1)) >= 0
        ):
            name = _PLACEHOLDER.format(len(originals))
            originals[name] = sql[i : close + 1]
            out.append(name)
            i = close + 1
            continue
        else:
            end = i + 1
        out.append(sql[i:end])
        i = end
    return "".join(out), originals


def restore_identifier_clauses(sql: str, originals: dict[str, str]) -> str:
    """*sql*, masked by :func:`mask_identifier_clauses`, with its clauses back."""
    for name, clause in originals.items():
        sql = sql.replace(name, clause)
    return sql


def _check_sql(source: str, dialect: str) -> str | None:
    """Parse a statement list with the required sqlglot dependency, its
    IDENTIFIER clauses masked (:func:`mask_identifier_clauses`)."""
    import sqlglot

    masked, _ = mask_identifier_clauses(source)
    try:
        sqlglot.parse(masked, dialect=dialect)
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    return None


def _check_none(source: str) -> str | None:
    return None


@dataclass(frozen=True)
class TargetLanguage:
    """One fully-supported SAS translation target."""

    key: str
    display_name: str
    aliases: frozenset[str]
    fence_infos: frozenset[str]
    default_fence: str
    cell_language: str
    kernelspec: dict[str, str]
    language_info: dict[str, str]
    complexity_profile: str
    comment_prefix: str
    sqlglot_dialect: str | None = None
    syntax_checker: Callable[[str], str | None] = field(
        repr=False, default=_check_none
    )

    def owns_fence(self, info: str) -> bool:
        tag = normalize_language(info)
        return tag == "" or tag in self.fence_infos

    def check_syntax(self, source: str) -> str | None:
        """Return a syntax error string, or None when source parses."""
        if self.sqlglot_dialect is not None:
            return _check_sql(source, self.sqlglot_dialect)
        return self.syntax_checker(source)

    @property
    def checker_name(self) -> str:
        if self.sqlglot_dialect is not None:
            return "sqlglot"
        if self.syntax_checker is _check_python:
            return "ast"
        return "none"

    @property
    def checks_syntax(self) -> bool:
        return (
            self.sqlglot_dialect is not None
            or self.syntax_checker is not _check_none
        )


_PYTHON_KERNELSPEC = {
    "name": "python3",
    "display_name": "Python 3",
    "language": "python",
}
_PYTHON_LANGUAGE_INFO = {
    "name": "python",
    "file_extension": ".py",
    "mimetype": "text/x-python",
}

PYSPARK = TargetLanguage(
    key="pyspark",
    display_name="PySpark",
    aliases=frozenset({"python", "python3", "py", "sparkpython"}),
    fence_infos=frozenset({"python", "python3", "py", "pyspark"}),
    default_fence="python",
    cell_language="python",
    kernelspec=_PYTHON_KERNELSPEC,
    language_info=_PYTHON_LANGUAGE_INFO,
    complexity_profile="pyspark",
    comment_prefix="#",
    syntax_checker=_check_python,
)

SPARKSQL = TargetLanguage(
    key="sparksql",
    display_name="Spark SQL",
    aliases=frozenset({"sql", "databrickssql", "ansisql"}),
    fence_infos=frozenset({"sql", "sparksql", "databrickssql"}),
    default_fence="sql",
    cell_language="sql",
    kernelspec={"name": "sql", "display_name": "SQL", "language": "sql"},
    language_info={
        "name": "sql",
        "file_extension": ".sql",
        "mimetype": "application/sql",
    },
    complexity_profile="sparksql",
    comment_prefix="--",
    sqlglot_dialect="databricks",
)

KNOWN_TARGETS: tuple[TargetLanguage, ...] = (PYSPARK, SPARKSQL)


def _index(targets: tuple[TargetLanguage, ...]) -> dict[str, TargetLanguage]:
    index: dict[str, TargetLanguage] = {}
    for target in targets:
        for alias in (target.key, *target.aliases):
            if alias in index:
                raise RuntimeError(
                    f"target-language alias {alias!r} is claimed by both "
                    f"{index[alias].display_name} and {target.display_name}"
                )
            index[alias] = target
    return index


_BY_KEY = _index(KNOWN_TARGETS)


def resolve_target_language(name: str | None) -> TargetLanguage:
    """Resolve a configured target or raise for every unsupported value."""
    if name is None:
        name = _configured_default()
    key = normalize_language(name)
    target = _BY_KEY.get(key)
    if target is not None:
        if key != normalize_language(target.display_name):
            logger.debug(
                "resolve_target_language: output language %r resolved to %r",
                name,
                target.display_name,
            )
        return target
    known = ", ".join(target.display_name for target in KNOWN_TARGETS)
    raise UnknownTargetLanguage(
        f"unknown output language {name!r}; known targets are {known}"
    )


def _configured_default() -> str:
    import app_config

    value = app_config.get_typed_value(
        "pipeline", "output_language", str, DEFAULT_OUTPUT_LANGUAGE
    )
    return value or DEFAULT_OUTPUT_LANGUAGE
