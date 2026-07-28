"""Live-conversation validation: score existing threads and transcripts.

Two post-hoc entry points, both observe-only (they return results; nothing
gates or retries a run):

- :func:`validate_thread` — reconstruct the (prompt, response) turns of an
  existing pipeline thread from the memory store and score them **without
  re-running the pipeline**. The chunker/batcher items are not persisted, so
  metrics that need item metadata (``dataset_fidelity``) skip; the LLM judge
  falls back to grading against the turn's prompt, which for pipeline
  threads embeds the SAS chunk text. The reference guidance is ephemeral too,
  so the retrieval-context metrics skip. What thread mode uniquely *can*
  score is everything the memory layers persisted: the rolling summary
  (handed to :class:`~validation.summarization.SummarizationMetric`), the
  task policy the thread ran under, its own short-term notes, and every
  *other* thread's notes — which is what lets ``validation.memory_metrics``
  audit instruction-following and note leakage after the fact.
- :func:`validate_transcript` — score any (prompt, response) transcript,
  even one not produced by :class:`~chunker.pipeline.SasLLMPipeline`.

Both build a case-free :class:`~validation.models.EvaluationRun` and score
it with :class:`~validation.evaluator.Evaluator`; results are ordinary
:class:`~validation.models.CaseResult`s (``case_id`` carries the thread /
transcript id), so ``ValidationReport`` and ``tracking.log_report`` work on
them unchanged.

Logger name: ``validation.conversation``.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence, cast

from langchain_core.messages import BaseMessage
from memory.turns import group_turns, turn_text

from .evaluator import Evaluator
from .metrics import ValidationMetric
from .models import CaseResult, EvaluationRun

logger = logging.getLogger(__name__)

# list[BaseMessage] (grouped into turns) or explicit (prompt, response) pairs.
Transcript = Sequence[BaseMessage] | Sequence[tuple[str, str]]


# ---------------------------------------------------------------------------
# Turn reconstruction
# ---------------------------------------------------------------------------


def _pairs_from_messages(
    messages: Sequence[BaseMessage],
) -> tuple[list[str], list[str]]:
    """(prompts, responses) from a chronological message list.

    ``memory.turns.group_turns`` opens a turn per HumanMessage; everything
    that follows (AI, tool, ...) is that turn's response, joined. A turn
    with no reply yields an empty response — response_coverage then counts
    it unanswered, which is the honest verdict on a half-finished turn.
    """
    prompts: list[str] = []
    responses: list[str] = []
    for turn in group_turns(list(messages)):
        prompts.append(str(turn[0].content))
        responses.append("\n".join(str(m.content) for m in turn[1:]))
    return prompts, responses


def _thread_messages(source: Any, thread_id: str) -> list[BaseMessage]:
    """History for *thread_id* from a ``SasLLMPipeline`` or a ``MemoryHub``
    (duck-typed so this module imports neither)."""
    getter = getattr(source, "get_thread_messages", None)
    if getter is not None:  # SasLLMPipeline
        return getter(thread_id)
    return source.get_thread(thread_id).messages  # MemoryHub


def _thread_item_ids(source: Any, thread_id: str) -> dict[int, str]:
    """0-based turn index -> real pipeline item id, from the run facts the
    pipeline records under ``run::{thread_id}::item::*`` (empty when the
    source carries none). One completed item == one persisted turn pair, so
    fact index *i* (1-based) labels turn ``i - 1``."""
    if hasattr(source, "get_run_facts"):  # SasLLMPipeline
        facts = source.get_run_facts(thread_id)
    elif hasattr(source, "kv"):  # MemoryHub
        prefix = f"run::{thread_id}::item::"
        facts = [
            {"item_id": item["key"][len(prefix) :], **item["value"]}
            for item in source.kv.items_with_prefix(prefix)
        ]
    else:
        facts = []
    return {
        fact["index"] - 1: fact["item_id"]
        for fact in facts
        if isinstance(fact.get("index"), int) and fact["index"] > 0
    }


def _thread_kv(source: Any) -> Any | None:
    """The conversation KV store behind *source*, or ``None``.

    ``MemoryHub`` exposes it as ``kv``; a ``SasLLMPipeline`` holds a hub it
    does not re-export, so its ``_memory`` is reached the same way
    validation/README.md documents for reading verdicts back. Duck-typed, like
    :func:`_thread_messages`, so this module imports neither class.
    """
    kv = getattr(source, "kv", None)
    if kv is not None:
        return kv
    return getattr(getattr(source, "_memory", None), "kv", None)


def _rolling_summary(
    source: Any, thread_id: str, messages: Sequence[BaseMessage]
) -> tuple[str | None, str | None]:
    """``(summary, summary_source)`` for *thread_id*'s rolling summary.

    Reads the record :class:`memory.summarize.RollingSummarizer` stores under
    ``summary::{thread_id}`` and rebuilds the text it covers — the first
    ``covered_turns`` turns of the thread, which is exactly the prefix the
    summarizer folded. ``(None, None)`` when no summary has been folded yet
    (the common case: folding waits for ``trigger_tokens``), which skips
    :class:`~validation.summarization.SummarizationMetric`.
    """
    kv = _thread_kv(source)
    if kv is None:
        return None, None
    try:
        stored = kv.get(f"summary::{thread_id}", None)
    except Exception:
        logger.debug(f"_rolling_summary: no summary readable for '{thread_id}'")
        return None, None
    if not isinstance(stored, dict):
        return None, None
    summary = str(stored.get("summary") or "").strip()
    covered = int(stored.get("covered_turns", 0) or 0)
    if not summary or covered < 1:
        return None, None
    turns = group_turns(list(messages))[:covered]
    if not turns:
        return None, None
    logger.info(
        f"_rolling_summary: '{thread_id}' summary covers {len(turns)} turn(s)"
    )
    return summary, "\n\n".join(turn_text(turn) for turn in turns)


def _memory_state(source: Any, thread_id: str) -> dict[str, list[Any]]:
    """The memory a thread ran under: policy, its own notes, others' notes.

    Read straight off the KV rows (``policy::…``, ``tmem::…``) the way
    :func:`_rolling_summary` reads the summary, so this module imports
    nothing from ``memory``. The long-term policy comes from the pipeline
    that ran the thread (it knows its task id); a bare ``MemoryHub`` has no
    way to know which task a thread belonged to, so it contributes notes
    only.

    ``foreign`` is every *other* thread's notes — the corpus
    ``memory_leakage`` checks these answers against. Empty (and the metric
    skips) whenever the store holds only this thread's notes.
    """
    state: dict[str, list[Any]] = {"policy": [], "notes": [], "foreign": []}
    policy = getattr(source, "task_policy", None)
    as_dicts = getattr(policy, "as_dicts", None)
    if callable(as_dicts):
        entries = as_dicts()
        if isinstance(entries, list):
            state["policy"] = [e for e in entries if isinstance(e, dict)]
    kv = _thread_kv(source)
    if kv is None:
        return state
    try:
        own = kv.get(f"tmem::{thread_id}", None)
        items = getattr(kv, "items_with_prefix", None)
        entries = items("tmem::") if items is not None else []
    except Exception:
        logger.debug(f"_memory_state: no notes readable for '{thread_id}'")
        return state
    state["notes"] = _note_texts(own)
    for entry in entries:
        if entry.get("key") == f"tmem::{thread_id}":
            continue
        state["foreign"].extend(_note_texts(entry.get("value")))
    logger.info(
        f"_memory_state: '{thread_id}'  policy={len(state['policy'])}  "
        f"notes={len(state['notes'])}  foreign_notes={len(state['foreign'])}"
    )
    return state


def _note_texts(stored: Any) -> list[str]:
    """Note texts out of one stored ``tmem::`` record (see
    ``memory.thread_mem``); ``[]`` for anything unreadable."""
    if not isinstance(stored, list):
        return []
    return [
        str(raw["text"]).strip()
        for raw in stored
        if isinstance(raw, dict) and str(raw.get("text", "")).strip()
    ]


def _outputs(
    responses: list[str], item_ids: dict[int, str]
) -> list[dict[str, Any]]:
    return [
        {"item_id": item_ids.get(i, f"turn-{i + 1}"), "response": response}
        for i, response in enumerate(responses)
    ]


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_from_thread(
    source: Any,
    thread_id: str,
    *,
    required_terms: Sequence[str] = (),
    reference_translation: str | None = None,
) -> EvaluationRun:
    """
    Reconstruct an :class:`EvaluationRun` from an existing thread.

    Parameters
    ----------
    source : SasLLMPipeline | MemoryHub
        Where the thread lives. A pipeline also supplies its run facts, so
        outputs carry real item ids; otherwise turns are labelled
        ``turn-<n>``.
    thread_id : str
        The thread to reconstruct. An empty/unknown thread raises
        ``ValueError`` — silently passing an empty conversation would be a
        false green.
    required_terms, reference_translation
        Optional expectations, as on :class:`ValidationCase`.
    """
    messages = _thread_messages(source, thread_id)
    if not messages:
        raise ValueError(f"thread '{thread_id}' has no messages to validate")
    prompts, responses = _pairs_from_messages(messages)
    item_ids = _thread_item_ids(source, thread_id)
    summary, summary_source = _rolling_summary(source, thread_id, messages)
    memory = _memory_state(source, thread_id)
    logger.info(
        f"run_from_thread: '{thread_id}'  turns={len(prompts)}  "
        f"labelled_items={len(item_ids)}  summary={'yes' if summary else 'no'}  "
        f"policy={len(memory['policy'])}  notes={len(memory['notes'])}"
    )
    return EvaluationRun(
        run_id=thread_id,
        prompts=prompts,
        outputs=_outputs(responses, item_ids),
        required_terms=list(required_terms),
        reference_translation=reference_translation,
        summary=summary,
        summary_source=summary_source,
        task_policy=memory["policy"],
        thread_notes=memory["notes"],
        foreign_notes=memory["foreign"],
    )


def validate_thread(
    source: Any,
    thread_id: str,
    *,
    metrics: Sequence[ValidationMetric] | None = None,
    required_terms: Sequence[str] = (),
    reference_translation: str | None = None,
) -> CaseResult:
    """Score an existing thread: :func:`run_from_thread` + the metric suite
    (``None`` = the deterministic :func:`~validation.metrics.default_metrics`)."""
    run = run_from_thread(
        source,
        thread_id,
        required_terms=required_terms,
        reference_translation=reference_translation,
    )
    return Evaluator(metrics=metrics).evaluate(run)


def run_from_transcript(
    transcript: Transcript,
    *,
    run_id: str = "transcript",
    required_terms: Sequence[str] = (),
    reference_translation: str | None = None,
) -> EvaluationRun:
    """
    Build an :class:`EvaluationRun` from an arbitrary transcript.

    *transcript* is either a chronological ``list[BaseMessage]`` (grouped
    into turns like a thread) or explicit ``(prompt, response)`` string
    pairs. Empty transcripts raise ``ValueError``.
    """
    items = list(transcript)
    if not items:
        raise ValueError("empty transcript: nothing to validate")
    # A Transcript is one shape or the other throughout, so the first item
    # decides for the whole list.
    if isinstance(items[0], BaseMessage):
        prompts, responses = _pairs_from_messages(cast("list[BaseMessage]", items))
    else:
        pairs = cast("list[tuple[str, str]]", items)
        prompts = [str(p) for p, _ in pairs]
        responses = [str(r) for _, r in pairs]
    logger.info(f"run_from_transcript: '{run_id}'  turns={len(prompts)}")
    return EvaluationRun(
        run_id=run_id,
        prompts=prompts,
        outputs=_outputs(responses, {}),
        required_terms=list(required_terms),
        reference_translation=reference_translation,
    )


def validate_transcript(
    transcript: Transcript,
    *,
    run_id: str = "transcript",
    metrics: Sequence[ValidationMetric] | None = None,
    required_terms: Sequence[str] = (),
    reference_translation: str | None = None,
) -> CaseResult:
    """Score an arbitrary transcript: :func:`run_from_transcript` + the
    metric suite (``None`` = the deterministic default suite)."""
    run = run_from_transcript(
        transcript,
        run_id=run_id,
        required_terms=required_terms,
        reference_translation=reference_translation,
    )
    return Evaluator(metrics=metrics).evaluate(run)
