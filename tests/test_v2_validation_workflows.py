"""Phase 8 judged, memory, transcript, inline, and offline validation gates."""

from __future__ import annotations

import asyncio

import pytest

from sas_migrate.application.ports.validation import JudgeRequest, JudgeVerdict
from sas_migrate.application.validation import (
    JUDGED_METRIC_NAMES,
    ConversationTurn,
    EvaluationRun,
    MemoryExtractionMetric,
    MemoryLeakageMetric,
    TokenBudgetPolicy,
    ValidationCase,
    ValidationRunner,
    ValidationUnit,
    judged_metrics,
    memory_metrics,
    run_from_transcript,
    validate_response,
)
from sas_migrate.core.targets import ResponseValidationResult, TargetId


class Judge:
    def __init__(self, score: float = 0.75) -> None:
        self.score = score
        self.requests: list[JudgeRequest] = []

    def evaluate(self, request: JudgeRequest) -> JudgeVerdict:
        self.requests.append(request)
        return JudgeVerdict(score=self.score, details=f"scored {request.metric}")


def _run(**changes: object) -> EvaluationRun:
    values: dict[str, object] = {
        "run_id": "thread-1",
        "target": TargetId.PYSPARK,
        "units": (
            ValidationUnit(
                unit_id="turn-1",
                prompt="Translate the source",
                source="data work.a; set work.b; run;",
                response="```python\ndf = spark.table('work.b')\n```",
                retrieval_context=("Use spark.table for SAS inputs.",),
            ),
        ),
        "prompt_instructions": ("Use PySpark",),
        "summary": "work.a comes from work.b",
        "summary_source": "The source reads work.b and writes work.a.",
        "task_policy": ("Preserve source datasets",),
        "thread_notes": ("Prefer DataFrame operations",),
    }
    values.update(changes)
    return EvaluationRun.model_validate(values)


def test_judged_metric_catalogue_preserves_names_and_thresholds() -> None:
    metrics = judged_metrics(Judge())
    assert tuple(metric.name for metric in metrics) == JUDGED_METRIC_NAMES
    assert [metric.threshold for metric in metrics] == [
        0.7,
        0.7,
        0.8,
        0.6,
        0.5,
        0.8,
        0.7,
        0.7,
        0.6,
        0.6,
        0.8,
        0.9,
    ]


def test_judged_metrics_keep_canonical_order_when_filtered() -> None:
    metrics = judged_metrics(Judge(), include=("task_completion", "faithfulness"))
    assert tuple(metric.name for metric in metrics) == ("faithfulness", "task_completion")
    with pytest.raises(ValueError, match="unknown judged metric"):
        judged_metrics(Judge(), include=("faithfullness",))


def test_judged_metrics_use_only_the_judge_port() -> None:
    judge = Judge(0.83)
    results = [metric.evaluate(_run()) for metric in judged_metrics(judge)]
    assert len(judge.requests) == len(JUDGED_METRIC_NAMES)
    assert all(result.score == 0.83 for result in results)
    assert judge.requests[0].contexts == ("data work.a; set work.b; run;",)
    assert judge.requests[3].contexts == ("Use spark.table for SAS inputs.",)


def test_judged_metrics_skip_when_their_specific_signal_is_absent() -> None:
    run = _run(
        units=(ValidationUnit(unit_id="u", response="answer"),),
        prompt_instructions=(),
        summary=None,
        summary_source=None,
        task_policy=(),
        thread_notes=(),
    )
    results = {metric.name: metric.evaluate(run) for metric in judged_metrics(Judge())}
    for name in (
        "contextual_precision",
        "contextual_relevancy",
        "prompt_alignment",
        "summarization",
        "policy_adherence",
        "override_compliance",
    ):
        assert results[name].skipped and results[name].passed


def test_memory_extraction_scores_precision_and_recall() -> None:
    run = _run(
        expected_memories=("Prefer email", "Escalate large refunds"),
        extracted_memories=("Prefer email communication", "Fabricated discount"),
    )
    result = MemoryExtractionMetric().evaluate(run)
    assert result.score == pytest.approx(0.5)
    assert "precision 0.50" in result.details and "recall 0.50" in result.details


def test_memory_extraction_does_not_reuse_one_extraction() -> None:
    run = _run(
        expected_memories=("Prefer email", "Prefer email communication"),
        extracted_memories=("Prefer email",),
    )
    result = MemoryExtractionMetric().evaluate(run)
    assert result.score < 1.0


def test_memory_leakage_normalizes_case_and_whitespace() -> None:
    run = _run(
        foreign_notes=("Do  NOT offer the discount", "Unrelated note"),
        units=(ValidationUnit(unit_id="u", response="do not offer the discount"),),
    )
    result = MemoryLeakageMetric().evaluate(run)
    assert result.score == 0.5 and not result.passed
    assert tuple(metric.name for metric in memory_metrics()) == (
        "memory_extraction",
        "memory_leakage",
    )


def test_transcript_reconstruction_preserves_turn_alignment_and_memory() -> None:
    run = run_from_transcript(
        "thread-7",
        TargetId.SPARK_SQL,
        (
            ConversationTurn(turn_id="1", prompt="first", response="one"),
            {"turn_id": "2", "prompt": "second", "response": "two", "source": "data x; run;"},
        ),
        task_policy=("Be exact",),
        foreign_notes=("other thread",),
    )
    assert [unit.prompt for unit in run.units] == ["first", "second"]
    assert run.joined_responses == "one\n\ntwo"
    assert run.task_policy == ("Be exact",)


def test_inline_validator_carries_target_result_into_report() -> None:
    result = ResponseValidationResult.accepted(TargetId.PYSPARK)
    report = validate_response(
        run_id="run",
        unit_id="unit",
        target=TargetId.PYSPARK,
        model="model",
        response="```python\nx = 1\n```",
        target_validation=result,
    )
    assert report.target_results == (result,)


def test_offline_runner_uses_producer_port_and_aggregates_cases() -> None:
    class Producer:
        async def produce(self, case: ValidationCase) -> EvaluationRun:
            return EvaluationRun(
                run_id=case.case_id,
                target=case.target,
                units=(
                    ValidationUnit(
                        unit_id=case.case_id,
                        source=case.sas_source,
                        response="```python\nx = 1\n```",
                    ),
                ),
                required_terms=case.required_terms,
                reference_translation=case.reference_translation,
                prompt_instructions=case.prompt_instructions,
            )

    cases = (
        ValidationCase(case_id="a", target=TargetId.PYSPARK, sas_source="data a; run;"),
        ValidationCase(case_id="b", target=TargetId.PYSPARK, sas_source="data b; run;"),
    )
    report = asyncio.run(
        ValidationRunner(Producer()).run(
            cases,
            model="model",
            translation_policy=TokenBudgetPolicy(max_run_tokens=100),
        )
    )
    assert [result.case_id for result in report.results] == ["a", "b"]


def test_offline_runner_rejects_empty_or_misidentified_cases() -> None:
    class Producer:
        async def produce(self, case: ValidationCase) -> EvaluationRun:
            return _run(run_id="wrong")

    runner = ValidationRunner(Producer())
    with pytest.raises(ValueError, match="at least one"):
        asyncio.run(runner.run((), model="model"))
    case = ValidationCase(case_id="right", target=TargetId.PYSPARK, sas_source="data x; run;")
    with pytest.raises(ValueError, match="identity"):
        asyncio.run(runner.run((case,), model="model"))
