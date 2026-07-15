"""Evaluator: the scoring core, decoupled from pipeline execution.

One :class:`Evaluator` holds a metric suite and turns one
:class:`~validation.models.EvaluationRun` into one
:class:`~validation.models.CaseResult` — no pipeline, no network, no store.
Everything that *produces* an ``EvaluationRun`` (the offline
:class:`~validation.runner.ValidationRunner`, the live entry points in
``validation.conversation``) funnels through here, so the metric loop exists
exactly once.

Logger name: ``validation.evaluator``.
"""

from __future__ import annotations

import logging
from typing import Sequence

from .metrics import ValidationMetric, default_metrics
from .models import CaseResult, EvaluationRun

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Score :class:`EvaluationRun`s with a fixed metric suite.

    Parameters
    ----------
    metrics : Sequence[ValidationMetric] | None
        Metric suite; ``None`` (default) uses :func:`default_metrics` — the
        deterministic, offline suite. Append an
        :class:`~validation.judge.LLMJudgeMetric` for judged runs.
    """

    def __init__(
        self, *, metrics: Sequence[ValidationMetric] | None = None
    ) -> None:
        self.metrics = list(metrics) if metrics is not None else default_metrics()

    def evaluate(self, run: EvaluationRun) -> CaseResult:
        """Run every metric over *run* and assemble the result."""
        results = [metric.evaluate(run) for metric in self.metrics]
        case_result = CaseResult(
            case_id=run.run_id,
            item_count=run.expected_units,
            metrics=results,
        )
        logger.info(
            f"evaluate: '{run.run_id}'  units={case_result.item_count}  "
            f"score={case_result.score:.3f}  passed={case_result.passed}"
        )
        return case_result
