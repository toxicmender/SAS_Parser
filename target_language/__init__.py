"""The run's target output language, as one resolved object.

``output_language`` used to be a free-form string that four layers each
interpreted on their own — the system prompt formatted it, ``prompt_builder``
folded it to match ``[lang: ...]`` tags, ``pipeline.notebook`` mapped it to a
kernelspec, and ``validation`` ignored it entirely and checked Python. This
module is the single place that says what a target *is*: how it is spelled,
which fence tags belong to it, what a notebook cell carries, and how to tell
whether a block of code is syntactically that language.

Resolution happens once, at pipeline construction
(:func:`resolve_target_language`); everything downstream is handed the resulting
:class:`TargetLanguage` rather than a string, so a typo can no longer degrade
into a silently-wrong default halfway down the stack.

Dependency-free by design (stdlib only, no ``chunker`` / ``pipeline`` /
``validation`` import), because every one of those packages imports *it*.
``sqlglot`` is used for SQL syntax checking when installed and degraded to a
conservative structural check when not — same pattern as
``llm_client.tokens`` degrading to ``chars // 4`` offline.

Logger name: ``target_language``.
"""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_OUTPUT_LANGUAGE",
    "KNOWN_TARGETS",
    "PROSE_FENCE_INFOS",
    "PYSPARK",
    "SPARKSCALA",
    "SPARKSQL",
    "TargetLanguage",
    "UnknownTargetLanguage",
    "normalize_language",
    "resolve_target_language",
]

# The one code default. Both SasLLMPipeline and main.py read it, so the two
# entry points can no longer disagree (they used to: "PySpark" vs "SparkSQL").
# config.json's pipeline.output_language overrides it; an explicit argument
# overrides that.
DEFAULT_OUTPUT_LANGUAGE = "SparkSQL"

# Fence info strings that never denote target-language code, whichever target
# is in force: the echoed SAS source and plain prose/log blocks.
PROSE_FENCE_INFOS = frozenset(
    {"sas", "text", "txt", "log", "output", "console", "markdown", "md"}
)


class UnknownTargetLanguage(ValueError):
    """Raised for an ``output_language`` no target in the registry claims."""


def normalize_language(name: str) -> str:
    """Fold an output-language name to its comparison key.

    Case-, space-, hyphen-, and underscore-insensitive, so ``"SparkSQL"``,
    ``"Spark SQL"``, and ``"spark_sql"`` all fold together — this is also the
    rule ``prompt_builder`` matches ``[lang: ...]`` directive tokens with, and
    it lives here so the two can never drift apart.
    """
    return re.sub(r"[\s_-]+", "", name.lower())


# ---------------------------------------------------------------------------
# Syntax checking
# ---------------------------------------------------------------------------


def _check_python(source: str) -> str | None:
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return f"{exc.msg} (line {exc.lineno})"
    return None


def _check_sql(source: str) -> str | None:
    """Parse *source* as Databricks SQL with sqlglot, or check it structurally.

    sqlglot is optional (the ``sql`` extra). Without it the fallback flags
    only what is unambiguous — unbalanced brackets or quotes — because this
    result can fail an item and drive a retry, and a heuristic that guesses
    would spend LLM calls re-answering correct translations.

    The ``databricks`` dialect, not ``spark``: this target's guidance emits
    ``QUALIFY``, SQL scripting, and ``EXECUTE IMMEDIATE``, which open-source
    Spark does not have. sqlglot's ``spark`` dialect happens to accept all of
    them today, so the two agree — but it would be within its rights to tighten,
    and that would start failing correct translations.
    """
    try:
        import sqlglot
    except ImportError:
        return _check_sql_structure(source)
    try:
        # A statement list: a translation cell is routinely several statements.
        sqlglot.parse(source, dialect="databricks")
    except Exception as exc:  # sqlglot raises ParseError/TokenError/...
        return f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    return None


# Single-quoted string, double-quoted identifier, backtick identifier, line
# comment, block comment — masked out before brackets and quotes are counted.
_SQL_MASKABLE_RE = re.compile(
    r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|`[^`]*`|--[^\n]*|/\*.*?\*/",
    re.DOTALL,
)


def _check_sql_structure(source: str) -> str | None:
    """The sqlglot-less fallback: brackets and quotes must balance."""
    masked = _SQL_MASKABLE_RE.sub(" ", source)
    for opener, closer, label in (("(", ")", "parenthes"), ("[", "]", "bracket")):
        if masked.count(opener) != masked.count(closer):
            return f"unbalanced {label}es ({opener}{closer})"
    for quote, label in (("'", "single"), ('"', "double"), ("`", "backtick")):
        if masked.count(quote):
            return f"unterminated {label} quote"
    return None


def _check_none(source: str) -> str | None:
    """No checker for this target — never reports an error."""
    return None


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetLanguage:
    """One target the pipeline can translate SAS into.

    Attributes
    ----------
    key
        The normalized comparison key (:func:`normalize_language` of the
        display name), e.g. ``"sparksql"``.
    display_name
        The canonical spelling, and the only one that reaches a prompt, a
        report, or a notebook's metadata — whatever the caller typed.
    aliases
        Extra normalized keys that resolve here.
    fence_infos
        Markdown fence info strings that mark a block as this language.
    default_fence
        The tag applied when the model emits code with no info string.
    cell_language
        What a notebook code cell records as its language, and the value the
        structured schema's ``TranslationCell.language`` is asked for.
    comment_prefix
        The line-comment token (``--``, ``#``, ``//``). The system prompt
        interpolates it so the "not convertible" marker is spelled in the
        target's own comment syntax — one place resolves it, rather than each
        caller guessing.
    kernelspec, language_info
        The notebook-level metadata blocks (nbformat v4.5).
    complexity_profile
        The ``complexity/profiles/*.json`` rule set that rates SAS constructs
        against this target.
    syntax_checker
        ``source -> error message | None``. ``None`` means "parses".
    """

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
    syntax_checker: Callable[[str], str | None] = field(
        repr=False, default=_check_none
    )

    def owns_fence(self, info: str) -> bool:
        """True when a fence tagged *info* is code in this language.

        An empty info string counts: an untagged fence inside a translation is
        the translation.
        """
        tag = normalize_language(info)
        return tag == "" or tag in self.fence_infos

    def check_syntax(self, source: str) -> str | None:
        """An error message for *source*, or ``None`` when it is well-formed."""
        return self.syntax_checker(source)

    @property
    def checker_name(self) -> str:
        """Which checker :meth:`check_syntax` actually runs, for reporting.

        Resolved live rather than at construction because it depends on
        whether an optional dependency is importable *now*.
        """
        if self.syntax_checker is _check_python:
            return "ast"
        if self.syntax_checker is _check_sql:
            try:
                import sqlglot  # noqa: F401
            except ImportError:
                return "structural"
            return "sqlglot"
        return "none"

    @property
    def checks_syntax(self) -> bool:
        """False when this target has no syntax checker at all (the metric
        skips rather than scoring everything as valid)."""
        return self.syntax_checker is not _check_none


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
    syntax_checker=_check_sql,
)

SPARKSCALA = TargetLanguage(
    key="sparkscala",
    display_name="Spark Scala",
    aliases=frozenset({"scala"}),
    fence_infos=frozenset({"scala", "sparkscala"}),
    default_fence="scala",
    cell_language="scala",
    kernelspec={"name": "scala", "display_name": "Scala", "language": "scala"},
    language_info={
        "name": "scala",
        "file_extension": ".scala",
        "mimetype": "text/x-scala",
    },
    # No Scala rule set of its own; the DataFrame-API profile is the closer
    # of the two (same constructs, different surface syntax).
    complexity_profile="pyspark",
    comment_prefix="//",
)

KNOWN_TARGETS: tuple[TargetLanguage, ...] = (PYSPARK, SPARKSQL, SPARKSCALA)

def _index(targets: tuple[TargetLanguage, ...]) -> dict[str, TargetLanguage]:
    """``{normalized name or alias: target}``, rejecting a shared alias.

    Two targets claiming one alias would make resolution order-dependent, so
    it fails at import — it can only ever be an editing mistake in the
    registry above, never a runtime condition.
    """
    index: dict[str, TargetLanguage] = {}
    for target in targets:
        for alias in (target.key, *target.aliases):
            if alias in index:  # pragma: no cover - registry is a constant
                raise RuntimeError(
                    f"target-language alias {alias!r} is claimed by both "
                    f"{index[alias].display_name} and {target.display_name}"
                )
            index[alias] = target
    return index


_BY_KEY = _index(KNOWN_TARGETS)


def resolve_target_language(
    name: str | None, *, allow_unknown: bool = False
) -> TargetLanguage:
    """The :class:`TargetLanguage` *name* denotes.

    ``None`` resolves the configured default (config.json
    ``pipeline.output_language``, else :data:`DEFAULT_OUTPUT_LANGUAGE`).

    An unrecognised name raises :class:`UnknownTargetLanguage` rather than
    quietly behaving like Python, which is what every layer used to do — the
    kernelspec fell back to python3, the syntax metric checked Python, and the
    prompt asked for a language nothing downstream understood. Pass
    *allow_unknown* to keep that old lenient behaviour: the name is preserved
    for the prompt and the ``[lang: ...]`` axis, and everything that needs
    real knowledge of the target borrows PySpark's, with a warning.
    """
    if name is None:
        name = _configured_default()
    key = normalize_language(name)
    target = _BY_KEY.get(key)
    if target is not None:
        if key != normalize_language(target.display_name):
            logger.debug(
                f"resolve_target_language: output language {name!r} resolved to "
                f"{target.display_name!r}"
            )
        return target
    known = ", ".join(t.display_name for t in KNOWN_TARGETS)
    if not allow_unknown:
        raise UnknownTargetLanguage(
            f"unknown output language {name!r}; known targets are {known} "
            f"(pass allow_unknown=True to translate into it anyway, with "
            f"PySpark's notebook and syntax handling)"
        )
    logger.warning(
        f"resolve_target_language: unknown output language {name!r} (known: {known}); "
        f"prompting for it anyway with PySpark's notebook kernel, fence tags, "
        f"and syntax checking — validation of the emitted language will be "
        f"unreliable"
    )
    return replace(PYSPARK, key=key, display_name=name)


def _configured_default() -> str:
    """``pipeline.output_language`` from config.json, else the code default.

    Imported lazily: ``app_config`` is dependency-free but this module is
    imported by it in no direction, and keeping the import local means a
    caller who passes an explicit name never touches the config file.
    """
    import app_config

    value = app_config.get_typed_value(
        "pipeline", "output_language", str, DEFAULT_OUTPUT_LANGUAGE
    )
    return value or DEFAULT_OUTPUT_LANGUAGE
