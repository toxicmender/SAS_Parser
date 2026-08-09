"""Contract tests for the supported target-language registry."""

from __future__ import annotations

import types

import pytest

from target_language import (
    DEFAULT_OUTPUT_LANGUAGE,
    KNOWN_TARGETS,
    PYSPARK,
    SPARKSQL,
    TargetLanguage,
    UnknownTargetLanguage,
    normalize_language,
    resolve_target_language,
)


@pytest.mark.parametrize(
    "name",
    ["SparkSQL", "Spark SQL", "spark sql", "spark-sql", "spark_sql", "sql"],
)
def test_sql_aliases_resolve_to_spark_sql(name):
    assert resolve_target_language(name) is SPARKSQL


@pytest.mark.parametrize("name", ["PySpark", "pyspark", "python", "Python3", "py"])
def test_pyspark_aliases_resolve_to_pyspark(name):
    assert resolve_target_language(name) is PYSPARK


def test_only_two_supported_targets_are_registered():
    assert KNOWN_TARGETS == (PYSPARK, SPARKSQL)
    assert resolve_target_language(None) is resolve_target_language(
        DEFAULT_OUTPUT_LANGUAGE
    )


@pytest.mark.parametrize("name", ["Cobol", "scala", "Spark Scala"])
def test_unsupported_targets_fail_at_resolution(name):
    with pytest.raises(UnknownTargetLanguage) as excinfo:
        resolve_target_language(name)
    assert name in str(excinfo.value)


def test_every_alias_is_claimed_once_and_display_names_are_canonical():
    seen: dict[str, TargetLanguage] = {}
    for target in KNOWN_TARGETS:
        assert normalize_language(target.display_name) == target.key
        for alias in (target.key, *target.aliases):
            assert alias not in seen
            seen[alias] = target


@pytest.mark.parametrize(
    "target,info,owned",
    [
        (SPARKSQL, "sql", True),
        (SPARKSQL, "", True),
        (SPARKSQL, "python", False),
        (PYSPARK, "python", True),
        (PYSPARK, "py", True),
        (PYSPARK, "sql", False),
    ],
)
def test_fence_ownership(target, info, owned):
    assert target.owns_fence(info) is owned


def test_targets_define_target_specific_comment_prefixes():
    assert SPARKSQL.comment_prefix == "--"
    assert PYSPARK.comment_prefix == "#"


def test_pyspark_uses_python_ast_syntax_checking():
    assert PYSPARK.check_syntax('df = spark.table("x")') is None
    assert PYSPARK.check_syntax("def broken(:") is not None
    assert PYSPARK.checker_name == "ast"


def test_sql_uses_required_sqlglot_with_databricks_dialect(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    class ParseError(Exception):
        pass

    def parse(sql: str, dialect: str | None = None) -> list[object]:
        calls.append((sql, dialect))
        if "BROKEN" in sql:
            raise ParseError("no viable alternative")
        return []

    fake = types.ModuleType("sqlglot")
    fake.parse = parse  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "sqlglot", fake)

    assert SPARKSQL.check_syntax("SELECT * FROM main.sales.orders QUALIFY 1 = 1") is None
    assert "ParseError" in (SPARKSQL.check_syntax("BROKEN") or "")
    assert SPARKSQL.checker_name == "sqlglot"
    assert {dialect for _, dialect in calls} == {"databricks"}
