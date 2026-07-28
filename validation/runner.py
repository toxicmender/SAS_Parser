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
from llm_client import TokenUsage

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

    @property
    def judge_token_usage(self) -> TokenUsage:
        """
        Tokens billed by LLM-backed *metrics* so far, summed across the suite.

        Any metric exposing a ``token_usage`` is counted, so a second judged
        metric added later is picked up without touching this. Zero when no
        metric reports usage — the deterministic suite, or a judge wired to a
        raw chat model rather than an :class:`llm_client.LLMClient`.
        """
        total = TokenUsage()
        for metric in self.metrics:
            usage = getattr(metric, "token_usage", None)
            if isinstance(usage, TokenUsage):
                total = total + usage
        return total

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
        """Run every case and aggregate into a :class:`ValidationReport`.

        Token totals are attributed to *this* run by snapshotting before and
        subtracting after: the pipeline and judge are the caller's, and may
        well have been used before this call or be reused after it.
        """
        logger.info(f"run: {len(cases)} case(s)  model={self.pipeline.model}")
        pipeline_before = self.pipeline.token_usage
        judge_before = self.judge_token_usage

        results = [self.run_case(case) for case in cases]

        pipeline_usage = self.pipeline.token_usage - pipeline_before
        judge_usage = self.judge_token_usage - judge_before
        report = ValidationReport(
            model=self.pipeline.model,
            instructions_fingerprint=self.pipeline.instructions_fingerprint,
            # getattr: the runner duck-types its pipeline (tests inject
            # stand-ins), and a pipeline predating long-term memory has no
            # policy to fingerprint.
            policy_fingerprint=getattr(self.pipeline, "policy_fingerprint", None),
            results=results,
            # A run that billed nothing reported nothing — carry None rather
            # than a row of zeroes claiming the run was free.
            token_usage=pipeline_usage if pipeline_usage.calls else None,
            judge_token_usage=judge_usage if judge_usage.calls else None,
        )
        logger.info(
            f"run: aggregate score={report.score:.3f}  passed={report.passed}  "
            f"pipeline tokens[{pipeline_usage.summary()}]  "
            f"judge tokens[{judge_usage.summary()}]"
        )
        return report
