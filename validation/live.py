"""Inline (per-batch) validation during a live pipeline run.

Where the other entry points score *after the fact* — ``validation.runner``
drives cases offline, ``validation.conversation`` scores a finished thread —
:class:`LiveValidator` scores each item the moment its response returns and
writes the verdict into the **same conversation memory the run itself uses**,
so a run's validation lives beside the run and is queryable per thread.

The pipeline calls :meth:`LiveValidator.validate_item` after every item it
sends to the LLM (see ``SasLLMPipeline._process``); nothing here re-runs the
model. Scoring funnels through the shared :class:`~validation.evaluator.Evaluator`
core, so an inline verdict on one item is identical to what a post-hoc pass
over that single item would produce — inline validation adds *when*, not a
different *how*.

Storage
-------
One record per item in the conversation KV store, keyed::

    validation::{thread_id}::item::{item_id}

deliberately mirroring the run facts the pipeline writes at
``run::{thread_id}::item::*`` (see ``SasLLMPipeline._record_run_fact``): the
same thread, the same per-item granularity, the small-facts-only rule (the
full response already lives in the ``msg::`` history and is never duplicated
here). The stored value is the item's :class:`~validation.models.CaseResult`
(``score`` / ``passed`` / per-metric results) plus ``index`` / ``total`` /
``ts``.

Observe-only
------------
A verdict is stored and returned; a failing item is never retried and never
aborts the run (the pipeline additionally swallows any validator error).
This matches the standing failure-handling policy in ``validation/README.md``.

Logger name: ``validation.live``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

from chunker.models import SasBatch, SasChunk

from .evaluator import Evaluator
from .metrics import ValidationMetric
from .models import CaseResult, EvaluationRun

logger = logging.getLogger(__name__)


def _item_id(item: SasBatch | SasChunk) -> str:
    return item.batch_id if isinstance(item, SasBatch) else item.chunk_id


def _validation_key(thread_id: str, item_id: str) -> str:
    return f"validation::{thread_id}::item::{item_id}"


_VALIDATION_PREFIX_TMPL = "validation::{thread_id}::item::"


class LiveValidator:
    """
    Score pipeline items inline and persist each verdict to conversation memory.

    Parameters
    ----------
    metrics : Sequence[ValidationMetric] | None
        Metric suite; ``None`` (default) uses the deterministic, offline
        :func:`~validation.metrics.default_metrics` — no network, so inline
        validation never slows a run with an extra model call. Append an
        :class:`~validation.judge.LLMJudgeMetric` to grade each item with a
        judge model (that call *is* per-item, and blocking).
    """

    def __init__(
        self, *, metrics: Sequence[ValidationMetric] | None = None
    ) -> None:
        self._evaluator = Evaluator(metrics=metrics)
        logger.info(
            "LiveValidator: metrics="
            f"[{', '.join(m.name for m in self._evaluator.metrics)}]"
        )

    @property
    def metrics(self) -> list[ValidationMetric]:
        return self._evaluator.metrics

    def validate_item(
        self,
        item: SasBatch | SasChunk,
        response: str,
        *,
        thread_id: str,
        kv: Any,
        index: int | None = None,
        total: int | None = None,
    ) -> CaseResult:
        """Score one item's *response* and store the verdict in *kv*.

        Builds a single-item :class:`EvaluationRun` (the item carries its own
        dataset/metadata, so ``dataset_fidelity`` scores it precisely rather
        than skipping the way a metadata-less thread would), evaluates it, and
        upserts the result under
        ``validation::{thread_id}::item::{item_id}`` on the conversation KV.

        Parameters
        ----------
        item, response
            The batch/chunk just sent to the LLM and the text it produced.
        thread_id
            The conversation thread the item belongs to — the KV namespace
            the verdict is filed under, beside that thread's run facts.
        kv
            The conversation's KV store (``memory.store.KVMemoryStore`` —
            duck-typed, ``.set(key, value, tags=..., source=...)``).
        index, total
            1-based position of the item in the run and the run size, stored
            for ordering/reporting (as on the run facts). Optional.
        """
        item_id = _item_id(item)
        run = EvaluationRun(
            run_id=item_id,
            items=[item],
            outputs=[{"item_id": item_id, "response": response}],
        )
        result = self._evaluator.evaluate(run)

        fact = {
            **result.model_dump(),
            "index": index,
            "total": total,
            "ts": time.time(),
        }
        kv.set(
            _validation_key(thread_id, item_id),
            fact,
            tags=["validation", thread_id],
            source="validation",
        )
        logger.log(
            logging.INFO if result.passed else logging.WARNING,
            f"validate_item: thread='{thread_id}' item={item_id} "
            f"score={result.score:.3f} passed={result.passed}",
        )
        return result


def validations_for_thread(kv: Any, thread_id: str) -> list[dict[str, Any]]:
    """Per-item validation verdicts stored for *thread_id*, in item order.

    Reads the records :meth:`LiveValidator.validate_item` wrote under
    ``validation::{thread_id}::item::*`` back out of the conversation KV
    (``kv.all_items()`` — the same shape ``SasLLMPipeline.get_run_facts``
    reads), each augmented with its ``item_id``. Ordered by the stored
    ``index`` (unindexed records sort first).
    """
    prefix = _VALIDATION_PREFIX_TMPL.format(thread_id=thread_id)
    facts = [
        {"item_id": item["key"][len(prefix) :], **item["value"]}
        for item in kv.all_items()
        if item["key"].startswith(prefix)
    ]
    facts.sort(key=lambda f: f.get("index") or 0)
    return facts
