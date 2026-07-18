"""ValidationRunner: cases -> pipeline -> metrics -> report.

The runner drives an existing :class:`chunker.SasLLMPipeline` and never
constructs one itself, so the caller decides everything about the model
(real vs ``FakeListChatModel``), guidance, memory backend, and history
policy — validation adds scoring, not configuration.

Item alignment: metrics need the batcher's items (for metadata such as
dataset names), but ``run_text`` returns only output dicts. Chunking and
batching are deterministic for identical input, so the runner re-derives the
items with the pipeline's own chunker/batcher and aligns them positionally
with the outputs, warning if the ids ever disagree.

Logger name: ``validation.runner``.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence
from uuid import uuid4

from chunker.models import SasBatch
from chunker.pipeline import SasLLMPipeline

from .evaluator import Evaluator
from .metrics import ValidationMetric
from .models import CaseResult, CaseRun, ValidationCase, ValidationReport

logger = logging.getLogger(__name__)


class ValidationRunner:
    """
    Run :class:`ValidationCase`s through a pipeline and score the outputs.

    Parameters
    ----------
    pipeline : SasLLMPipeline
        The pipeline under test, fully configured by the caller.
    metrics : Sequence[ValidationMetric] | None
        Metric suite; ``None`` (default) uses :func:`default_metrics` — the
        deterministic, offline suite. Append an
        :class:`~validation.judge.LLMJudgeMetric` for judged runs.
    """

    def __init__(
        self,
        pipeline: SasLLMPipeline,
        *,
        metrics: Sequence[ValidationMetric] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self._evaluator = Evaluator(metrics=metrics)
        logger.info(
            f"ValidationRunner: model={pipeline.model}  "
            f"metrics=[{', '.join(m.name for m in self.metrics)}]"
        )

    @property
    def metrics(self) -> list[ValidationMetric]:
        return self._evaluator.metrics

    def run_case(self, case: ValidationCase) -> CaseResult:
        """Run one case through the pipeline and every metric."""
        source_id = f"{case.case_id}.sas"
        # Fresh thread per invocation: reruns of the same case must not see
        # (or be scored against) an earlier run's accumulated history.
        thread_id = f"validation::{case.case_id}::{uuid4().hex[:8]}"
        logger.info(f"run_case: '{case.case_id}'  thread='{thread_id}'")

        chunk_result = self.pipeline.chunker.chunk_text(
            case.sas_source, source_id=source_id
        )
        batch_result = self.pipeline.batcher.batch(chunk_result)
        items = batch_result.all_ordered_items

        t0 = time.perf_counter()
        outputs = self.pipeline.run_text(
            case.sas_source, source_id=source_id, thread_id=thread_id
        )
        elapsed = time.perf_counter() - t0

        derived_ids = [
            i.batch_id if isinstance(i, SasBatch) else i.chunk_id for i in items
        ]
        output_ids = [o["item_id"] for o in outputs]
        if derived_ids != output_ids:
            logger.warning(
                f"run_case: '{case.case_id}' item ids diverge between the "
                f"re-derived batching and the pipeline outputs "
                f"({derived_ids} vs {output_ids}); scoring positionally"
            )

        # run_id and the expectation fields are filled in from `case` by
        # CaseRun._derive_from_case, a before-validator pyright can't see.
        run = CaseRun(case=case, items=items, outputs=outputs)  # pyright: ignore[reportCallIssue]
        case_result = self._evaluator.evaluate(run)
        logger.info(
            f"run_case: '{case.case_id}' done  items={len(items)}  "
            f"score={case_result.score:.3f}  passed={case_result.passed}  "
            f"llm_elapsed={elapsed:.3f}s"
        )
        return case_result

    def run(self, cases: Sequence[ValidationCase]) -> ValidationReport:
        """Run every case and aggregate into a :class:`ValidationReport`."""
        logger.info(f"run: {len(cases)} case(s)  model={self.pipeline.model}")
        report = ValidationReport(
            model=self.pipeline.model,
            instructions_fingerprint=self.pipeline.instructions_fingerprint,
            results=[self.run_case(case) for case in cases],
        )
        logger.info(
            f"run: aggregate score={report.score:.3f}  passed={report.passed}"
        )
        return report
