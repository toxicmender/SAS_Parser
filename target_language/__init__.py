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
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # sqlglot is imported where a parse needs it
    from sqlglot.tokens import Token

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
_PLACEHOLDER = "__sasp_identifier_{}__"


def _clause_end(tokens: list[Token], i: int) -> int | None:
    """The index of the token closing the IDENTIFIER clause that opens at
    token *i*, or ``None`` when none opens there or it never closes.

    A clause opens at an unquoted ``IDENTIFIER`` followed by ``(``, and not
    after a dot: a backquoted name, a string or a comment spelling it is a
    different token, or none at all.
    """
    token = tokens[i]
    if (
        token.token_type.name != "VAR"
        or token.text.upper() != "IDENTIFIER"
        or i + 1 >= len(tokens)
        or tokens[i + 1].token_type.name != "L_PAREN"
        or (i > 0 and tokens[i - 1].token_type.name == "DOT")
    ):
        return None
    depth = 0
    for j in range(i + 1, len(tokens)):
        kind = tokens[j].token_type.name
        if kind == "L_PAREN":
            depth += 1
        elif kind == "R_PAREN":
            depth -= 1
            if depth == 0:
                return j
    return None


def mask_identifier_clauses(
    sql: str, dialect: str = "databricks"
) -> tuple[str, dict[str, str]]:
    """*sql* with each IDENTIFIER clause replaced by a plain placeholder name,
    and the clause each placeholder stands for.

    sqlglot reads ``SELECT * FROM IDENTIFIER(session.t)`` but not
    ``CREATE TABLE IDENTIFIER(session.t) AS ...``, ``INSERT INTO``, ``COPY
    INTO`` or ``DROP TABLE`` with one, which Databricks accepts — so a parse
    of generated Spark SQL masks them first. A placeholder is an ordinary
    identifier, so the statement around it parses as it would with a literal
    name, and :func:`restore_identifier_clauses` puts the clauses back.

    Clauses are found on *dialect*'s own tokens, so a string, a raw string, a
    comment or a backquoted name that only spells ``IDENTIFIER(`` is never
    one. A clause never closed masks nothing, and neither does a literal or a
    comment never closed, which the tokenizer refuses: the parse reports them.
    Nor does a clause in a statement sqlglot keeps as raw text after its
    keyword (``OPTIMIZE``, ``EXECUTE``, ``SHOW``): it parses as a command
    either way, and whoever reads that text masks it there.
    """
    try:
        from sqlglot.dialects.dialect import Dialect

        tokens = Dialect.get_or_raise(dialect).tokenize(sql)
    except Exception:  # sqlglot's TokenError: an unclosed literal or comment
        return sql, {}
    out: list[str] = []
    originals: dict[str, str] = {}
    copied = i = 0
    while i < len(tokens):
        close = _clause_end(tokens, i)
        if close is None:
            i += 1
            continue
        name = _PLACEHOLDER.format(len(originals))
        start, end = tokens[i].start, tokens[close].end + 1
        originals[name] = sql[start:end]
        out += (sql[copied:start], name)
        copied, i = end, close + 1
    out.append(sql[copied:])
    return "".join(out), originals


def restore_identifier_clauses(sql: str, originals: dict[str, str]) -> str:
    """*sql*, masked by :func:`mask_identifier_clauses`, with its clauses back.

    Matched case-insensitively, so a generator that normalises names — sqlglot
    upper-cases a function name — still leaves no placeholder behind.
    """
    if not originals:
        return sql
    by_name = {name.lower(): clause for name, clause in originals.items()}
    pattern = re.compile("|".join(map(re.escape, originals)), re.IGNORECASE)
    return pattern.sub(lambda match: by_name[match.group(0).lower()], sql)


def _syntax_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0]}"


def _check_sql(source: str, dialect: str) -> str | None:
    """Parse a statement list with the required sqlglot dependency, its
    IDENTIFIER clauses masked (:func:`mask_identifier_clauses`) and each
    clause's own expression parsed by itself, so masking hides no error."""
    import sqlglot

    masked, clauses = mask_identifier_clauses(source, dialect)
    try:
        sqlglot.parse(masked, dialect=dialect)
    except Exception as exc:
        return _syntax_error(exc)
    for clause in clauses.values():
        inner = clause[clause.index("(") + 1 : clause.rindex(")")]
        try:
            sqlglot.parse_one(inner, dialect=dialect)
        except Exception as exc:
            return f"{_syntax_error(exc)} (in {clause})"
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
