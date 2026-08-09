"""Core deterministic validation tests."""

from __future__ import annotations

from validation import (
    EvaluationRun,
    LanguageComplianceMetric,
    TargetSyntaxMetric,
    default_metrics,
)


def _response(language: str, code: str) -> str:
    fence = chr(96) * 3
    return f"## Translation\n\n{fence}{language}\n{code}\n{fence}"


def _run(response: str) -> EvaluationRun:
    return EvaluationRun(
        run_id="case",
        prompts=["data work.x; run;"],
        outputs=[{"response": response}],
    )


def test_target_syntax_uses_python_ast_for_pyspark():
    result = TargetSyntaxMetric(output_language="PySpark").evaluate(
        _run(_response("python", "df = spark.table('x')"))
    )
    assert result.metric == "target_syntax"
    assert result.score == 1.0
    assert "ast" in result.details


def test_target_syntax_uses_sqlglot_for_databricks_sql():
    sql = (
        "SELECT * FROM main.sales.orders QUALIFY row_number() "
        "OVER (PARTITION BY customer_id ORDER BY updated_at DESC) = 1"
    )
    result = TargetSyntaxMetric(output_language="Spark SQL").evaluate(
        _run(_response("sql", sql))
    )
    assert result.metric == "target_syntax"
    assert result.score == 1.0
    assert "sqlglot" in result.details


def test_target_syntax_rejects_invalid_target_code():
    result = TargetSyntaxMetric(output_language="PySpark").evaluate(
        _run(_response("python", "def broken(:"))
    )
    assert result.score == 0.0


def test_language_compliance_rejects_an_off_target_fence():
    result = LanguageComplianceMetric(output_language="Spark SQL").evaluate(
        _run(_response("python", "df = 1"))
    )
    assert result.score == 0.0


def test_default_metrics_publish_the_target_syntax_contract():
    assert "target_syntax" in {
        metric.name for metric in default_metrics(output_language="Spark SQL")
    }
