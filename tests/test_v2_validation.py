"""Phase 8 validation, target-result, token-report, and adapter gates."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pymupdf
import pytest

from sas_migrate.adapters.validation import JsonlValidationReportRepository, render_pdf
from sas_migrate.application.validation import (
    DatasetFidelityMetric,
    EvaluationRun,
    Evaluator,
    LanguageComplianceMetric,
    ReferenceSimilarityMetric,
    RequiredTermsMetric,
    ResponseCoverageMetric,
    TargetSyntaxMetric,
    TokenBudgetPolicy,
    ValidationService,
    ValidationUnit,
    build_token_budget_report,
    default_metrics,
    metric_names,
    render_json,
    render_markdown,
)
from sas_migrate.core.targets import (
    ResponseValidationResult,
    TargetId,
    TargetIssueCode,
    TargetValidationIssue,
)
from sas_migrate.core.tokens import CallTokenRecord, TokenCallLedger, TokenCategory


def _run(
    response: str = "```python\ndf = spark.table('work.source')\n```",
    **changes: object,
) -> EvaluationRun:
    values: dict[str, object] = {
        "run_id": "case-1",
        "target": TargetId.PYSPARK,
        "units": (
            ValidationUnit(
                unit_id="unit-1",
                source="data work.target; set work.source; run;",
                response=response,
                input_datasets=("work.source",),
                output_datasets=("work.target",),
                target_validation=ResponseValidationResult.accepted(TargetId.PYSPARK),
            ),
        ),
    }
    values.update(changes)
    return EvaluationRun.model_validate(values)


def _record(
    *,
    item: str = "item-1",
    attempt: int = 1,
    accepted: bool = True,
    recovered: bool = False,
    sas: int = 20,
    instructions: int = 10,
    output: int = 5,
) -> CallTokenRecord:
    input_by_category = {
        TokenCategory.SAS_SOURCE: sas,
        TokenCategory.PROJECT_INSTRUCTIONS: instructions,
    }
    return CallTokenRecord(
        run_id="run-1",
        thread_id="thread-1",
        item_id=item,
        attempt=attempt,
        target=TargetId.PYSPARK,
        estimator="test",
        encoding="cl100k_base",
        estimated_input_by_category=input_by_category,
        estimated_input_total=sum(input_by_category.values()),
        estimated_output_by_category={TokenCategory.CODE_OUTPUT: output},
        accepted_attempt=accepted,
        recovered=recovered,
    )


def test_default_metric_names_and_thresholds_match_legacy_contract() -> None:
    metrics = default_metrics()
    assert metric_names(metrics) == (
        "response_coverage",
        "dataset_fidelity",
        "language_compliance",
        "target_syntax",
        "required_terms",
        "reference_similarity",
    )
    assert [metric.threshold for metric in metrics] == [1.0, 0.75, 1.0, 1.0, 1.0, 0.5]


@pytest.mark.parametrize(
    ("metric", "run"),
    [
        (DatasetFidelityMetric(), _run(units=(ValidationUnit(unit_id="u"),))),
        (RequiredTermsMetric(), _run(required_terms=())),
        (ReferenceSimilarityMetric(), _run(reference_translation=None)),
        (TargetSyntaxMetric(), _run("plain response")),
    ],
)
def test_metrics_preserve_no_signal_skip_semantics(metric: object, run: EvaluationRun) -> None:
    result = metric.evaluate(run)  # type: ignore[attr-defined]
    assert result.skipped and result.passed and result.score == 1.0


def test_deterministic_metrics_score_response_content() -> None:
    run = _run(
        required_terms=("spark.table",),
        reference_translation="df = spark.table('work.source')",
    )
    results = {result.metric: result for result in Evaluator().evaluate(run).metrics}
    assert results["response_coverage"].score == 1.0
    assert results["language_compliance"].score == 1.0
    assert results["target_syntax"].score == 1.0
    assert results["required_terms"].score == 1.0
    assert results["reference_similarity"].score > 0.5
    assert results["dataset_fidelity"].score == 0.5


def test_response_coverage_uses_expected_units() -> None:
    result = ResponseCoverageMetric().evaluate(_run(expected_units=2))
    assert result.score == 0.5 and not result.passed


def test_language_compliance_rejects_foreign_fences() -> None:
    result = LanguageComplianceMetric().evaluate(_run("```sql\nselect 1\n```"))
    assert result.score == 0.0 and not result.passed


def test_sql_syntax_uses_databricks_dialect(monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlglot

    calls: list[str | None] = []
    real_parse = sqlglot.parse

    def capture(value: str, *, read: str | None = None, **kwargs: object) -> object:
        calls.append(read)
        return real_parse(value, read=read, **kwargs)

    monkeypatch.setattr(sqlglot, "parse", capture)
    run = _run(
        "```sql\nSELECT * FROM work.source\n```",
        target=TargetId.SPARK_SQL,
        units=(ValidationUnit(unit_id="sql", response="```sql\nSELECT * FROM work.source\n```"),),
    )
    assert TargetSyntaxMetric().evaluate(run).passed
    assert calls == ["databricks"]


def test_token_budget_is_broken_down_and_counts_retry_and_recovery() -> None:
    ledger = TokenCallLedger(
        records=(
            _record(accepted=False),
            _record(item="item-2", recovered=True, sas=8, instructions=2, output=3),
        )
    )
    report = build_token_budget_report(ledger, TokenBudgetPolicy(max_run_tokens=40))
    assert report.input_by_category == {"sas_source": 28, "project_instructions": 12}
    assert report.output_by_category == {"code_output": 8}
    assert report.current_run_tokens == 35
    assert report.recovered_tokens == 13
    assert report.retry_overhead_tokens == 35
    assert report.compliant


def test_token_budget_reports_per_call_and_run_violations() -> None:
    report = build_token_budget_report(
        TokenCallLedger(records=(_record(),)),
        TokenBudgetPolicy(
            max_input_tokens_per_call=29,
            max_output_tokens_per_call=4,
            max_run_tokens=34,
        ),
    )
    assert not report.compliant
    assert len(report.violations) == 3


def test_service_keeps_translation_and_judge_budgets_separate() -> None:
    report = ValidationService().validate(
        _run(),
        model="gateway/model",
        translation_ledger=TokenCallLedger(records=(_record(sas=20),)),
        judge_ledger=TokenCallLedger(records=(_record(sas=2, instructions=1),)),
        translation_policy=TokenBudgetPolicy(max_run_tokens=100),
        judge_policy=TokenBudgetPolicy(max_run_tokens=100),
    )
    assert report.translation_tokens is not None
    assert report.judge_tokens is not None
    assert report.translation_tokens.current_run_tokens == 35
    assert report.judge_tokens.current_run_tokens == 8
    budget_results = [m for m in report.results[0].metrics if m.metric == "token_budget_compliance"]
    assert len(budget_results) == 2


def test_invalid_target_and_budget_fail_the_aggregate() -> None:
    invalid = ResponseValidationResult(
        valid=False,
        resolved_target=TargetId.PYSPARK,
        issues=(TargetValidationIssue(code=TargetIssueCode.SYNTAX_ERROR, message="broken"),),
    )
    run = _run(units=(ValidationUnit(unit_id="u", response="```python\nx = 1\n```", target_validation=invalid),))
    report = ValidationService().validate(
        run,
        model="model",
        translation_ledger=TokenCallLedger(records=(_record(),)),
        translation_policy=TokenBudgetPolicy(max_run_tokens=1),
    )
    assert not report.passed


def test_markdown_and_json_render_target_and_component_budget_tables() -> None:
    report = ValidationService().validate(
        _run(),
        model="model",
        translation_ledger=TokenCallLedger(records=(_record(),)),
        translation_policy=TokenBudgetPolicy(max_run_tokens=100),
    )
    markdown = render_markdown(report)
    assert "Target resolution validation" in markdown
    assert "token_budget_compliance" in markdown
    assert "| input | sas_source | 20 |" in markdown
    payload = json.loads(render_json(report))
    assert payload["translation_tokens"]["input_by_category"]["project_instructions"] == 10
    assert payload["judge_tokens"] is None


def test_pdf_contains_target_and_token_sections() -> None:
    report = ValidationService().validate(
        _run(),
        model="model",
        translation_ledger=TokenCallLedger(records=(_record(),)),
        translation_policy=TokenBudgetPolicy(max_run_tokens=100),
    )
    payload = render_pdf(report)
    with pymupdf.open(stream=payload, filetype="pdf") as document:
        text = "\n".join(page.get_text() for page in document)
    assert "Target resolution validation" in text
    assert "Translation token budget" in text
    assert "sas_source" in text


def test_jsonl_tracking_round_trips_full_report(tmp_path: Path) -> None:
    path = tmp_path / "validation.jsonl"
    repository = JsonlValidationReportRepository(path)
    report = ValidationService().validate(_run(), model="model")

    async def exercise() -> None:
        assert await repository.load() == ()
        assert await repository.append(report) == str(path)
        assert await repository.load() == (report,)

    asyncio.run(exercise())
