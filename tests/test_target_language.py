"""The target-language registry: resolution, fence ownership, syntax checks.

These are the guarantees every other layer leans on — the pipeline resolves
once and hands the object down, so a bug here is a bug in the prompt, the
notebook, and the validation verdict at the same time.
"""

from __future__ import annotations

import sys
import types

import pytest

from target_language import (
    DEFAULT_OUTPUT_LANGUAGE,
    KNOWN_TARGETS,
    PYSPARK,
    SPARKSCALA,
    SPARKSQL,
    TargetLanguage,
    UnknownTargetLanguage,
    normalize_language,
    resolve_target_language,
)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["SparkSQL", "Spark SQL", "spark sql", "spark-sql", "spark_sql", "SPARKSQL", "sql"],
)
def test_spellings_of_one_target_all_resolve_together(name):
    assert resolve_target_language(name) is SPARKSQL


@pytest.mark.parametrize(
    "name", ["PySpark", "pyspark", "python", "Python3", "py"]
)
def test_python_spellings_resolve_to_pyspark(name):
    assert resolve_target_language(name) is PYSPARK


def test_resolution_canonicalises_the_display_name():
    # Whatever the caller typed, one spelling reaches prompts and reports.
    assert resolve_target_language("sparksql").display_name == "Spark SQL"
    assert resolve_target_language("scala").display_name == "Spark Scala"


def test_none_resolves_the_configured_default():
    assert resolve_target_language(None) is resolve_target_language(
        DEFAULT_OUTPUT_LANGUAGE
    )


def test_unknown_target_raises_and_names_the_known_ones():
    with pytest.raises(UnknownTargetLanguage) as excinfo:
        resolve_target_language("Cobol")
    message = str(excinfo.value)
    assert "Cobol" in message
    for target in KNOWN_TARGETS:
        assert target.display_name in message


def test_unknown_target_can_be_forced_through_and_keeps_its_name():
    forced = resolve_target_language("Cobol", allow_unknown=True)
    assert forced.display_name == "Cobol"  # the prompt still asks for it
    assert forced.cell_language == PYSPARK.cell_language  # borrowed handling


def test_every_alias_is_claimed_by_exactly_one_target():
    seen: dict[str, TargetLanguage] = {}
    for target in KNOWN_TARGETS:
        for alias in (target.key, *target.aliases):
            assert alias not in seen, f"{alias!r} claimed twice"
            seen[alias] = target


def test_display_names_normalise_back_to_their_key():
    for target in KNOWN_TARGETS:
        assert normalize_language(target.display_name) == target.key


# ---------------------------------------------------------------------------
# Fence ownership
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target,info,owned",
    [
        (SPARKSQL, "sql", True),
        (SPARKSQL, "sparksql", True),
        (SPARKSQL, "", True),  # untagged inside a translation is the translation
        (SPARKSQL, "python", False),
        (SPARKSQL, "sas", False),
        (PYSPARK, "python", True),
        (PYSPARK, "py", True),
        (PYSPARK, "PYTHON", True),
        (PYSPARK, "sql", False),
        (SPARKSCALA, "scala", True),
        (SPARKSCALA, "python", False),
    ],
)
def test_owns_fence(target, info, owned):
    assert target.owns_fence(info) is owned


@pytest.mark.parametrize(
    "target,prefix",
    [(SPARKSQL, "--"), (PYSPARK, "#"), (SPARKSCALA, "//")],
)
def test_comment_prefix(target, prefix):
    """The system prompt interpolates this to spell the NOT CONVERTIBLE
    marker in the target's own comment syntax."""
    assert target.comment_prefix == prefix


def test_every_target_defines_a_comment_prefix():
    from target_language import KNOWN_TARGETS

    assert all(t.comment_prefix for t in KNOWN_TARGETS)


# ---------------------------------------------------------------------------
# Syntax checking
# ---------------------------------------------------------------------------


def test_python_syntax_check_accepts_and_rejects():
    assert PYSPARK.check_syntax('df = spark.table("x")') is None
    assert PYSPARK.check_syntax("def broken(:") is not None
    assert PYSPARK.checker_name == "ast"


def test_scala_has_no_checker_so_nothing_is_flagged():
    assert not SPARKSCALA.checks_syntax
    assert SPARKSCALA.check_syntax("this is not scala at all {{{") is None
    assert SPARKSCALA.checker_name == "none"


def test_sql_structural_fallback_flags_only_the_unambiguous(monkeypatch):
    """Without sqlglot the check must not guess: it can fail a real item."""
    monkeypatch.setitem(sys.modules, "sqlglot", None)  # forces ImportError

    assert SPARKSQL.checker_name == "structural"
    # Well-formed, including quotes and comments that contain brackets.
    assert SPARKSQL.check_syntax("SELECT a FROM t WHERE b = 1") is None
    assert SPARKSQL.check_syntax("SELECT '((' AS a -- ) note\nFROM t") is None
    # Unambiguously broken.
    assert "parenthes" in (SPARKSQL.check_syntax("SELECT a FROM (t") or "")
    assert "quote" in (SPARKSQL.check_syntax("SELECT 'unclosed FROM t") or "")
    # Not SQL at all, but structurally balanced — deliberately NOT flagged,
    # because a false failure here spends a retry re-answering a good item.
    assert SPARKSQL.check_syntax("df = 1") is None


def test_sql_uses_sqlglot_when_it_is_importable(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    class ParseError(Exception):
        pass

    def parse(sql: str, dialect: str | None = None) -> list[object]:
        calls.append((sql, dialect))
        if "BROKEN" in sql:
            raise ParseError("no viable alternative\nat line 1")
        return []

    fake = types.ModuleType("sqlglot")
    fake.parse = parse  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sqlglot", fake)

    assert SPARKSQL.checker_name == "sqlglot"
    assert SPARKSQL.check_syntax("SELECT 1") is None
    error = SPARKSQL.check_syntax("BROKEN")
    assert error is not None and "ParseError" in error
    # Databricks, not the generic dialect and not `spark`: this target emits
    # QUALIFY, SQL scripting and EXECUTE IMMEDIATE, which open-source Spark
    # does not have.
    assert {dialect for _, dialect in calls} == {"databricks"}


def test_sparksql_declares_the_databricks_dialect():
    """The bundled guidance emits QUALIFY, SQL scripting and EXECUTE
    IMMEDIATE, which open-source Spark does not have."""
    assert SPARKSQL.sqlglot_dialect == "databricks"
    assert PYSPARK.sqlglot_dialect is None
    assert SPARKSCALA.sqlglot_dialect is None


def test_xref_rewriter_takes_its_dialect_from_the_target(monkeypatch):
    """The two sqlglot consumers read the same SQL the same way.

    `xref.rewrite` parses the *generated* SQL to rewrite table references and
    returns it unchanged when it does not parse. If the checker accepted a
    statement the rewriter could not read, the rewrite would silently no-op on
    exactly the Databricks-only syntax this target emits — a split-brain that
    shows up as missing table rewrites, not as an error. Resolving both from
    one field is what makes that unrepresentable.
    """
    import app_config
    import xref.rewrite as rewrite

    monkeypatch.setattr(
        app_config, "get_value", lambda s, k, default=None: "SparkSQL"
    )
    assert rewrite.default_dialect() == SPARKSQL.sqlglot_dialect

    # A non-SQL target still needs a dialect: the rewriter reaches sqlglot for
    # the SQL inside `spark.sql("...")`.
    monkeypatch.setattr(
        app_config, "get_value", lambda s, k, default=None: "PySpark"
    )
    assert rewrite.default_dialect() == SPARKSQL.sqlglot_dialect

    # A config typo is the pipeline's error to raise, not the rewriter's.
    monkeypatch.setattr(
        app_config, "get_value", lambda s, k, default=None: "Klingon"
    )
    assert rewrite.default_dialect() == SPARKSQL.sqlglot_dialect
