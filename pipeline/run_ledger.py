"""Run control for the SAS LLM pipeline: per-item facts, resume, fork.

:class:`RunLedger` owns the KV-side bookkeeping of a run — the
``run::{thread}::item::{item_id}`` outcome facts, the
``validation::{thread}::item::{item_id}`` verdicts written beside them, and
the machinery built on those rows: what a resume may skip and must redo,
the rewind that keeps a redo history consistent, and the fact-copying half
of a fork. The engine (`pipeline.engine`) delegates here; nothing in this
module calls an LLM.

Logger name: ``pipeline.run_ledger``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

from chunker.models import SasBatch
from langchain_core.messages import AIMessage, BaseMessage
from memory.store import MemoryHub

logger = logging.getLogger(__name__)


# Key under which a turn's TranslationDocument travels on the AI message's
# additional_kwargs (and so into the KV store, and back out on resume).
DOCUMENT_KEY = "translation_document"

# Bookkeeping keys the pipeline adds around a stored inline-validation verdict
# (index/total/ts, plus item_id added by the reader). Stripped when a verdict
# is recovered on resume so the recovered `validation` value matches the shape
# a freshly-scored item carries (the bare CaseResult dump).
_RECOVERED_VALIDATION_DROP = frozenset({"item_id", "index", "total", "ts"})


def document_of(message: BaseMessage | None) -> dict[str, Any] | None:
    """The TranslationDocument dict carried by *message*, or ``None``."""
    if message is None:
        return None
    document = getattr(message, "additional_kwargs", {}).get(DOCUMENT_KEY)
    return document if isinstance(document, dict) else None


class RunLedger:
    """KV-backed run bookkeeping for one pipeline's threads.

    Parameters
    ----------
    memory : MemoryHub
        The store the facts (and the message history they describe) live in.
    validation_retries : int
        The pipeline's validation-retry budget. ``0`` keeps the original
        resume policy (every ``ok`` item is skipped); a positive value makes
        resume validation-aware — an ``ok`` item whose stored verdict failed
        is redone (see :meth:`resume_state`).
    """

    def __init__(self, memory: MemoryHub, *, validation_retries: int = 0) -> None:
        self._memory = memory
        self._validation_retries = validation_retries

    # ------------------------------------------------------------------
    # Fact writes and reads
    # ------------------------------------------------------------------

    def record_run_fact(
        self, thread_id: str, item_id: str, fact: dict[str, Any]
    ) -> None:
        """Write one per-item outcome record to the KV layer (the "write
        context" channel): small facts only — the full response already
        lives in the msg:: history and is never duplicated here."""
        self._memory.kv.set(
            f"run::{thread_id}::item::{item_id}",
            fact,
            tags=["run-item", thread_id],
            source="pipeline",
        )

    def record_validation_fact(
        self, thread_id: str, item_id: str, fact: dict[str, Any]
    ) -> None:
        """Upsert one inline-validation verdict under the same key schema
        ``validation.live.LiveValidator`` uses (``validation::{thread_id}::``
        ``item::{item_id}``), so a fork's copied verdicts read back through
        :meth:`validation_facts` exactly like inline-written ones."""
        self._memory.kv.set(
            f"validation::{thread_id}::item::{item_id}",
            fact,
            tags=["validation", thread_id],
            source="pipeline",
        )

    def run_facts(self, thread_id: str) -> list[dict[str, Any]]:
        """Per-item outcome records written during a run, in item order.

        One record per processed item, stored in the KV layer under
        ``run::{thread_id}::item::{item_id}`` as the run progresses —
        durable evidence of *which items completed* (status, index,
        timing) that later batches, later runs, and resume query without
        replaying the message history.
        """
        prefix = f"run::{thread_id}::item::"
        # Prefix-filtered read: Delta mode fetches only this thread's rows
        # instead of collecting the whole kv namespace.
        facts = [
            {"item_id": item["key"][len(prefix) :], **item["value"]}
            for item in self._memory.kv.items_with_prefix(prefix)
        ]
        facts.sort(key=lambda f: f.get("index", 0))
        return facts

    def validation_facts(self, thread_id: str) -> list[dict[str, Any]]:
        """Per-item inline-validation verdicts recorded for *thread_id*.

        Present only when the pipeline was built with a ``validator``: each
        item scored during the run leaves one record under
        ``validation::{thread_id}::item::{item_id}`` (score, passed,
        per-metric results, index/total), stored beside the run facts by
        ``validation.live.LiveValidator``. Ordered by item index; empty when
        no validator ran on the thread.
        """
        prefix = f"validation::{thread_id}::item::"
        facts = [
            {"item_id": item["key"][len(prefix) :], **item["value"]}
            for item in self._memory.kv.items_with_prefix(prefix)
        ]
        facts.sort(key=lambda f: f.get("index") or 0)
        return facts

    # ------------------------------------------------------------------
    # Recovery of stored outputs (resume)
    # ------------------------------------------------------------------

    @staticmethod
    def recovered_response(
        messages: list[BaseMessage], fact: dict[str, Any]
    ) -> str | None:
        """The stored AI response for a completed item, or ``None``.

        A run halts on its first failure and each item persists exactly one
        (human, AI) pair, so completed item *i* (1-based fact index) maps to
        message ``2 * i - 1``. Anything inconsistent — a hand-edited thread,
        retention pruning — degrades to ``None`` rather than guessing.
        """
        position = 2 * fact.get("index", 0) - 1
        if 0 < position < len(messages) and isinstance(messages[position], AIMessage):
            return messages[position].content
        return None

    @staticmethod
    def recovered_document(
        messages: list[BaseMessage], fact: dict[str, Any]
    ) -> dict[str, Any] | None:
        """The stored document for a completed item — see
        :meth:`recovered_response` for how the position is derived."""
        position = 2 * fact.get("index", 0) - 1
        if 0 < position < len(messages) and isinstance(messages[position], AIMessage):
            return document_of(messages[position])
        return None

    @staticmethod
    def recovered_validation(
        fact: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """The stored inline-validation verdict for a recovered item, or ``None``.

        Normalises the KV-stored fact back to the bare ``CaseResult`` dump a
        freshly-scored item carries — dropping the pipeline's bookkeeping keys
        (see :data:`_RECOVERED_VALIDATION_DROP`) — so a resumed run's outputs
        are shaped identically whether an item was replayed or recovered.
        ``None`` when the original attempt had no validator (or scoring failed
        then and left no verdict).
        """
        if not fact:
            return None
        return {
            k: v for k, v in fact.items() if k not in _RECOVERED_VALIDATION_DROP
        }

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def resume_state(
        self, items: Sequence[SasBatch], thread_id: str
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[BaseMessage]]:
        """Resolve what a resume can skip and what it must redo.

        Returns ``(completed, completed_validations, recovered)`` where
        *completed* maps item_id -> run fact for items that may be skipped and
        their responses recovered, *completed_validations* maps item_id ->
        stored verdict, and *recovered* is the pre-rewind message snapshot the
        skipped items' responses are read from.

        With validation-driven retry active, an ``ok`` item whose stored
        verdict *failed* is not "done": the thread is rewound to the earliest
        such (or otherwise-missing) item and that item's — and every later
        item's — run/validation facts and turns are dropped, so the main loop
        regenerates from there on a clean, consistent history. Without it,
        this reproduces the original policy: every ``ok`` item is skipped.
        """
        ok_facts = {
            f["item_id"]: f
            for f in self.run_facts(thread_id)
            if f.get("status") == "ok"
        }
        completed_validations = {
            f["item_id"]: f for f in self.validation_facts(thread_id)
        }
        recovered: list[BaseMessage] = []
        if ok_facts:
            recovered = self._memory.get_thread(thread_id).messages

        if not self._validation_retries:
            if ok_facts:
                logger.info(
                    f"resume_state: thread='{thread_id}'  {len(ok_facts)} "
                    f"item(s) already complete  "
                    f"{len(completed_validations)} stored verdict(s)"
                )
            return ok_facts, completed_validations, recovered

        # Validation-aware resume: find the first item that is not "done and
        # good" (missing, errored, or an ok item whose stored verdict failed).
        def _satisfied(item_id: str) -> bool:
            if item_id not in ok_facts:
                return False
            verdict = completed_validations.get(item_id)
            # No verdict → cannot call it a failure; treat as satisfied.
            return verdict is None or verdict.get("passed", True)

        redo_start: int | None = None
        for pos, item in enumerate(items, start=1):
            if not _satisfied(item.batch_id):
                redo_start = pos
                break

        if redo_start is None:
            logger.info(
                f"resume_state: thread='{thread_id}'  all {len(items)} "
                f"item(s) complete and passing; nothing to redo"
            )
            return ok_facts, completed_validations, recovered

        self._rewind_for_resume(items, thread_id, redo_start)
        completed = {
            item.batch_id: ok_facts[item.batch_id]
            for pos, item in enumerate(items, start=1)
            if pos < redo_start
        }
        logger.info(
            f"resume_state: thread='{thread_id}'  keeping {len(completed)} "
            f"passing item(s), regenerating from item {redo_start}/{len(items)}"
        )
        return completed, completed_validations, recovered

    def _rewind_for_resume(
        self, items: Sequence[SasBatch], thread_id: str, redo_start: int
    ) -> None:
        """Rewind *thread_id* to just before item *redo_start* (1-based).

        Truncates the thread to the ``redo_start - 1`` completed (human, AI)
        pairs that precede it and drops the run/validation facts of item
        *redo_start* and every later item, so the main loop regenerates them
        onto a clean, append-only history instead of leaving stale turns and
        facts behind.
        """
        keep_pairs = redo_start - 1
        removed = self._memory.get_thread(thread_id).truncate_to(keep_pairs * 2)
        for pos, item in enumerate(items, start=1):
            if pos < redo_start:
                continue
            self._memory.kv.delete(f"run::{thread_id}::item::{item.batch_id}")
            self._memory.kv.delete(f"validation::{thread_id}::item::{item.batch_id}")
        logger.info(
            f"_rewind_for_resume: thread='{thread_id}'  rewound to item "
            f"{redo_start} (kept {keep_pairs} pair(s), removed {removed} message(s))"
        )

    # ------------------------------------------------------------------
    # Fork
    # ------------------------------------------------------------------

    def fork(
        self,
        src_thread_id: str,
        dst_thread_id: str,
        *,
        upto_items: int | None = None,
    ) -> int:
        """Copy a run's turns and facts onto an empty thread; return the
        number of messages copied.

        Copies the first ``upto_items`` (human, AI) turn pairs of
        *src_thread_id* — every pair when ``None`` — plus their ``ok`` run
        facts and stored inline verdicts onto *dst_thread_id*, so a
        forked-then-resumed run recovers them the same way a plain resume
        does. Short-term thread notes are the engine's concern (they live in
        ``memory.thread_mem``, not in this ledger's rows).
        """
        copied = self._memory.fork_thread(
            src_thread_id,
            dst_thread_id,
            upto_messages=None if upto_items is None else upto_items * 2,
        )
        for fact in self.run_facts(src_thread_id):
            index = fact.get("index", 0)
            if upto_items is not None and index > upto_items:
                continue
            if fact.get("status") != "ok":
                continue
            value = {k: v for k, v in fact.items() if k != "item_id"}
            self.record_run_fact(dst_thread_id, fact["item_id"], value)
        # Carry the copied items' inline verdicts onto the fork too (no-op
        # when the source run had no validator).
        for fact in self.validation_facts(src_thread_id):
            if upto_items is not None and (fact.get("index") or 0) > upto_items:
                continue
            value = {k: v for k, v in fact.items() if k != "item_id"}
            self.record_validation_fact(dst_thread_id, fact["item_id"], value)
        return copied

    def error_fact(
        self, index: int, total: int, exc: Exception
    ) -> dict[str, Any]:
        """The outcome fact recorded for an item whose LLM call raised."""
        return {
            "status": "error",
            "index": index,
            "total": total,
            "is_batch": True,
            "error": repr(exc),
            "ts": time.time(),
        }

    def ok_fact(
        self, index: int, total: int, *, elapsed_s: float, response_chars: int,
        attempts: int,
    ) -> dict[str, Any]:
        """The outcome fact recorded for a successfully answered item."""
        return {
            "status": "ok",
            "index": index,
            "total": total,
            "is_batch": True,
            "elapsed_s": round(elapsed_s, 3),
            "response_chars": response_chars,
            "attempts": attempts,
            "ts": time.time(),
        }
