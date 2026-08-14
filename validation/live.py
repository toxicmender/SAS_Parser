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
``run::{thread_id}::item::*`` (see ``pipeline.run_ledger.RunLedger.record_run_fact``): the
same thread, the same per-item granularity, the small-facts-only rule (the
full response already lives in the ``msg::`` history and is never duplicated
here). The stored value is the item's :class:`~validation.models.CaseResult`
(``score`` / ``passed`` / per-metric results) plus ``index`` / ``total`` /
``ts``.

Observe-only by default
-----------------------
The validator itself only scores and stores — it never re-runs the model or
aborts the run, and the pipeline additionally swallows any validator error.
Whether a *failing* verdict is then acted upon is the pipeline's decision:
with ``SasLLMPipeline(validation=ValidationSetup(validator=...))`` (retries=0,
the default) it is not, matching
the standing policy in ``validation/README.md``; with a positive budget the
pipeline re-generates the item with corrective feedback (and treats a stored
failing verdict as not-done on resume). See ``SasLLMPipeline._answer_item``.

Logger name: ``validation.live``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Sequence

from chunker.models import SasBatch, SasChunk
from llm_client import TokenUsage
from target_language import TargetLanguage

from .evaluator import Evaluator
from .metrics import ValidationMetric, _as_target, default_metrics
from .models import CaseResult, EvaluationRun, MetricResult, ValidationReport

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
    output_language : str | TargetLanguage | None
        The target the default suite scores the emitted code against —
        normally the pipeline's own
        :attr:`~pipeline.engine.SasLLMPipeline.target_language`. ``None``
        resolves the configured default. A validator scoring a different
        target than the pipeline prompts for fails every item on
        ``language_compliance``, so pass the pipeline's when you have it.
        Ignored when *metrics* is given: an explicit suite carries its own
        targets.
    """

    def __init__(
        self,
        *,
        metrics: Sequence[ValidationMetric] | None = None,
        output_language: "str | TargetLanguage | None" = None,
    ) -> None:
        self._target: TargetLanguage | None = None
        if metrics is None:
            self._target = _as_target(output_language)
            metrics = default_metrics(self._target)
        elif output_language is not None:
            logger.warning(
                "LiveValidator: output_language is ignored when an explicit "
                "metric suite is passed; build the suite with "
                "default_metrics(output_language=...) instead"
            )
        self._evaluator = Evaluator(metrics=metrics)
        # Evaluators for the targets individual items fell back to, built on
        # first use and kept: default_metrics() is not free, and a run that
        # falls back once usually falls back several times.
        self._evaluators_by_target: dict[str, Evaluator] = {}
        logger.info(
            "LiveValidator: metrics="
            f"[{', '.join(m.name for m in self._evaluator.metrics)}]"
        )

    def _evaluator_for(
        self, target: "TargetLanguage | None"
    ) -> tuple[Evaluator, bool]:
        """The evaluator that should score an item translated into *target*.

        Returns ``(evaluator, is_override)``. The run's own evaluator whenever
        *target* is absent or is already the run's — the common path, and the
        one that must stay byte-for-byte what it was.

        A caller who supplied explicit metrics gets its own suite back
        regardless: this class cannot rebuild a suite it did not compose, and
        silently swapping in the default one would score the run against
        metrics its owner never chose.
        """
        if target is None or self._target is None or target.key == self._target.key:
            return self._evaluator, False
        cached = self._evaluators_by_target.get(target.key)
        if cached is None:
            cached = Evaluator(metrics=default_metrics(target))
            self._evaluators_by_target[target.key] = cached
            logger.info(
                f"LiveValidator: scoring fallback item(s) against "
                f"{target.display_name} rather than "
                f"{self._target.display_name}"
            )
        return cached, True

    @property
    def target_language(self) -> TargetLanguage | None:
        """The target the default suite was built for, or ``None``.

        ``None`` means the caller supplied its own metrics, so this validator
        has no single target to speak for. :class:`SasLLMPipeline` reads this
        to warn when it is about to score a run against a language other than
        the one it prompts for.
        """
        return self._target

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
        prompt: str | None = None,
        retrieval_context: Sequence[str] = (),
        target_language: "TargetLanguage | None" = None,
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
        prompt, retrieval_context
            The message the item was answered from and the reference chunks
            retrieved for it, in rank order — what the judged metrics
            (``validation.rag_metrics`` and friends) score as ``input`` and
            retrieved context. Both optional: the deterministic default suite
            ignores them, and a judged metric with neither skips.
        target_language
            The target *this item* was actually translated into, when it is not
            the run's — see :mod:`complexity.fallback`. Scoring a PySpark
            fallback against the Spark SQL suite would find no SQL blocks and
            return 0.0 on a correct translation, and fail ``language_compliance``
            for code the run itself asked for. ``None`` (the default) scores
            against the run's target, so every existing caller is unaffected.
        """
        item_id = _item_id(item)
        run = EvaluationRun(
            run_id=item_id,
            items=[item],
            prompts=[prompt] if prompt is not None else [],
            outputs=[
                {
                    "item_id": item_id,
                    "response": response,
                    "retrieval_context": list(retrieval_context),
                }
            ],
        )
        evaluator, is_override = self._evaluator_for(target_language)
        result = evaluator.evaluate(run)

        fact = {
            **result.model_dump(),
            "index": index,
            "total": total,
            "ts": time.time(),
        }
        if is_override and target_language is not None:
            # Stored on the verdict so a report reading it back can say which
            # language the score was against — a 1.0 for PySpark and a 1.0 for
            # Spark SQL are not the same claim.
            fact["target_language"] = target_language.display_name
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
    (``kv.items_with_prefix()`` — a prefix-filtered read, so Delta mode
    fetches only this thread's rows), each augmented with its ``item_id``.
    Ordered by the stored ``index`` (unindexed records sort first).
    """
    prefix = _VALIDATION_PREFIX_TMPL.format(thread_id=thread_id)
    facts = [
        {"item_id": item["key"][len(prefix) :], **item["value"]}
        for item in kv.items_with_prefix(prefix)
    ]
    facts.sort(key=lambda f: f.get("index") or 0)
    return facts


def report_from_verdicts(
    verdicts: Iterable[dict[str, Any]],
    *,
    model: str = "live-validation",
    instructions_fingerprint: str | None = None,
    policy_fingerprint: str | None = None,
    token_usage: TokenUsage | None = None,
    judge_token_usage: TokenUsage | None = None,
) -> ValidationReport:
    """Aggregate stored inline verdicts into a :class:`ValidationReport`.

    Each verdict is one item's :class:`~validation.models.CaseResult` as a dict
    — an ``out["validation"]`` value from a pipeline run, or a record from
    :func:`validations_for_thread` /
    :meth:`~pipeline.engine.SasLLMPipeline.get_validation_facts`. The extra
    ``item_id`` / ``index`` / ``total`` / ``ts`` keys those readers add are
    ignored (``CaseResult`` drops unknown fields), and the per-item order is
    taken as given — pass already-ordered verdicts.

    The point is reuse: an inline run gets the same report surface the offline
    runner produces, so its verdicts flow straight into ``to_markdown()``,
    :mod:`validation.pdf`, and :func:`validation.tracking.log_report`.

    Parameters
    ----------
    model : str
        Label for the run under test — pass the pipeline's real model string;
        the default marks a report whose model was not supplied.
    instructions_fingerprint : str | None
        The active user-instruction fingerprint
        (:attr:`SasLLMPipeline.instructions_fingerprint`), so an inline report
        is never compared as equal to one scored under different instructions.
    policy_fingerprint : str | None
        The same, for the long-term task policy the run was prompted under
        (:attr:`SasLLMPipeline.policy_fingerprint`). Both reach the run
        history as their own columns.
    token_usage : TokenUsage | None
        What the run cost — pass :attr:`SasLLMPipeline.token_usage`. Verdicts
        do not carry token counts (they are per item, usage is per run), so
        this cannot be recovered from *verdicts* alone; ``None`` leaves the
        report's cost section out entirely.
    judge_token_usage : TokenUsage | None
        What *grading* cost, kept separate so a thoroughly-checked run does
        not read as an expensive one. ``None`` for an inline run, whose
        metrics are the deterministic suite.
    """
    results = [_case_result(v) for v in verdicts]
    return ValidationReport(
        model=model,
        instructions_fingerprint=instructions_fingerprint,
        policy_fingerprint=policy_fingerprint,
        results=results,
        token_usage=token_usage,
        judge_token_usage=judge_token_usage,
    )


def _case_result(verdict: dict[str, Any]) -> CaseResult:
    """One inline verdict dict rebuilt as a :class:`CaseResult`.

    Reads the ``CaseResult`` fields (``case_id`` / ``item_count`` / ``metrics``)
    and recomputes ``score`` / ``passed``, tolerating both a full
    :meth:`CaseResult.model_dump` and a leaner record: the item id falls back to
    the reader-supplied ``item_id`` then a placeholder, and ``item_count`` to 1.
    """
    return CaseResult(
        case_id=verdict.get("case_id") or verdict.get("item_id") or "item",
        item_count=verdict.get("item_count", 1),
        metrics=[MetricResult.model_validate(m) for m in verdict.get("metrics", [])],
    )


def report_from_thread(
    kv: Any,
    thread_id: str,
    *,
    model: str = "live-validation",
    instructions_fingerprint: str | None = None,
    policy_fingerprint: str | None = None,
) -> ValidationReport:
    """A :class:`ValidationReport` over *thread_id*'s stored inline verdicts.

    Convenience over :func:`report_from_verdicts` that first reads the verdicts
    back from the conversation KV with :func:`validations_for_thread` (their
    stored ``index`` already orders them).
    """
    return report_from_verdicts(
        validations_for_thread(kv, thread_id),
        model=model,
        instructions_fingerprint=instructions_fingerprint,
        policy_fingerprint=policy_fingerprint,
    )
