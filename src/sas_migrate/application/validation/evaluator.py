"""Pure validation metric orchestration."""

from __future__ import annotations

from collections.abc import Sequence

from .metrics import ValidationMetric, default_metrics
from .models import CaseResult, EvaluationRun


class Evaluator:
    def __init__(self, metrics: Sequence[ValidationMetric] | None = None) -> None:
        self.metrics = tuple(default_metrics() if metrics is None else metrics)

    def evaluate(self, run: EvaluationRun) -> CaseResult:
        return CaseResult(
            case_id=run.run_id,
            item_count=run.unit_count,
            metrics=tuple(metric.evaluate(run) for metric in self.metrics),
        )


__all__ = ["Evaluator"]
