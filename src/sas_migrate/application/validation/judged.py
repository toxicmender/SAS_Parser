"""Port-driven judged metrics with stable names, thresholds, and skip signals."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sas_migrate.application.ports.validation import JudgeRequest, ValidationJudge

from .metrics import ValidationMetric
from .models import EvaluationRun, MetricResult


@dataclass(frozen=True)
class _MetricSpec:
    name: str
    threshold: float
    signal: Callable[[EvaluationRun], tuple[str, tuple[str, ...], tuple[str, ...]] | None]


def _units(run: EvaluationRun) -> tuple[str, tuple[str, ...], tuple[str, ...]] | None:
    if not any(unit.response.strip() for unit in run.units):
        return None
    prompt = "\n\n".join(unit.prompt for unit in run.units)
    contexts = tuple(unit.source for unit in run.units if unit.source)
    return prompt, contexts, ()


def _retrieval(run: EvaluationRun) -> tuple[str, tuple[str, ...], tuple[str, ...]] | None:
    base = _units(run)
    contexts = tuple(value for unit in run.units for value in unit.retrieval_context)
    return None if base is None or not contexts else (base[0], contexts, ())


def _instructions(run: EvaluationRun) -> tuple[str, tuple[str, ...], tuple[str, ...]] | None:
    base = _units(run)
    return None if base is None or not run.prompt_instructions else (base[0], base[1], run.prompt_instructions)


def _summary(run: EvaluationRun) -> tuple[str, tuple[str, ...], tuple[str, ...]] | None:
    if not run.summary or not run.summary_source:
        return None
    return run.summary_source, (run.summary_source,), ()


def _policy(run: EvaluationRun) -> tuple[str, tuple[str, ...], tuple[str, ...]] | None:
    base = _units(run)
    return None if base is None or not run.task_policy else (base[0], base[1], run.task_policy)


def _override(run: EvaluationRun) -> tuple[str, tuple[str, ...], tuple[str, ...]] | None:
    base = _units(run)
    return None if base is None or not run.thread_notes else (base[0], base[1], run.thread_notes)


_SPECS = (
    _MetricSpec("faithfulness", 0.7, _units),
    _MetricSpec("answer_relevancy", 0.7, _units),
    _MetricSpec("hallucination", 0.8, _units),
    _MetricSpec("contextual_precision", 0.6, _retrieval),
    _MetricSpec("contextual_relevancy", 0.5, _retrieval),
    _MetricSpec("prompt_alignment", 0.8, _instructions),
    _MetricSpec("plan_adherence", 0.7, _units),
    _MetricSpec("task_completion", 0.7, _units),
    _MetricSpec("summarization", 0.6, _summary),
    _MetricSpec("analysis_summarization", 0.6, _units),
    _MetricSpec("policy_adherence", 0.8, _policy),
    _MetricSpec("override_compliance", 0.9, _override),
)
JUDGED_METRIC_NAMES = tuple(spec.name for spec in _SPECS)


class JudgedMetric(ValidationMetric):
    def __init__(self, judge: ValidationJudge, spec: _MetricSpec) -> None:
        self.name = spec.name
        self.default_threshold = spec.threshold
        self._judge = judge
        self._signal = spec.signal
        super().__init__()

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        signal = self._signal(run)
        if signal is None:
            return self.result(1.0, f"no signal for {self.name}", skipped=True)
        prompt, contexts, instructions = signal
        verdict = self._judge.evaluate(
            JudgeRequest(
                metric=self.name,
                prompt=prompt,
                response=run.joined_responses if self.name != "summarization" else run.summary or "",
                contexts=contexts,
                instructions=instructions,
            )
        )
        return self.result(verdict.score, verdict.details)


def judged_metrics(
    judge: ValidationJudge,
    include: Sequence[str] | None = None,
) -> tuple[JudgedMetric, ...]:
    requested = set(JUDGED_METRIC_NAMES if include is None else include)
    unknown = requested - set(JUDGED_METRIC_NAMES)
    if unknown:
        raise ValueError(f"unknown judged metric(s): {', '.join(sorted(unknown))}")
    return tuple(JudgedMetric(judge, spec) for spec in _SPECS if spec.name in requested)


__all__ = ["JUDGED_METRIC_NAMES", "JudgedMetric", "judged_metrics"]
