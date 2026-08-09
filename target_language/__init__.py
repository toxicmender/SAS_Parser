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
    "normalize_language",
    "resolve_target_language",
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


def _check_sql(source: str, dialect: str) -> str | None:
    """Parse a statement list with the required sqlglot dependency."""
    import sqlglot

    try:
        sqlglot.parse(source, dialect=dialect)
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
