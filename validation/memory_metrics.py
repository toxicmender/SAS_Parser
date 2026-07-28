"""Metrics over the memory layers. See validation/README.md.

The rest of this package scores what the model *said*. These four score what
it was *told to remember* — the long-term task policy (``memory.policy``),
the short-term thread notes (``memory.thread_mem``), and the extractor that
writes them (``memory.extractor``):

    policy_adherence    every standing task instruction was followed  (judged)
    override_compliance thread-scoped overrides honoured, fixed rules
                        still respected                               (judged)
    memory_extraction   what the extractor wrote vs what it should
                        have written                            (deterministic)
    memory_leakage      one thread's notes did not surface in
                        another's answers                       (deterministic)

The split matters. ``policy_adherence`` is ``prompt_alignment`` pointed at a
different instruction source — the persisted, task-scoped policy rather than
the system prompt's standing contract — and both are worth having: a run can
follow the pipeline's own contract perfectly while ignoring the operator's
policy. ``override_compliance`` is the metric with no analogue elsewhere: it
scores *both* directions of the precedence rule, because a model that ignores
an approved exception and one that lets an exception talk it out of a fixed
guardrail are both failures, and only one of them looks like disobedience.

Every metric reads its expectations off the :class:`~validation.models.EvaluationRun`
(``task_policy``, ``thread_notes``, ``foreign_notes``, ``expected_memories``,
``extracted_memories``), so a run that carries none simply skips — the same
degradation the rest of the suite applies to a case with no golden reference.

Logger name: ``validation.memory_metrics``.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Iterable, Sequence

from .judged import JudgedMetric, numbered

# _token_f1 is the same order-insensitive overlap `reference_similarity`
# scores with; matching two phrasings of one memory is the same question, so
# it is reused rather than reimplemented with a second definition of "close".
from .metrics import ValidationMetric, _token_f1
from .models import EvaluationRun, MetricResult

logger = logging.getLogger(__name__)


_POLICY_PROMPT = """\
You are checking whether an answer followed the standing instructions its
operator set for this task.

The request it answered:
{input}

The answer:
{answer}

The standing instructions:
{instructions}

For each instruction, in order, say whether the answer followed it:
- "yes" — followed
- "no"  — not followed
- "idk" — the instruction does not apply to this request

Reply with JSON: {{"verdicts": [{{"verdict": "yes|no|idk", "reason": "..."}}, ...]}}
with exactly {count} verdict(s), in the same order as the instructions.
"""

_OVERRIDE_PROMPT = """\
You are checking whether an answer respected the instructions in force for
one specific conversation.

Some instructions are specific to this conversation and override the standing
task instructions. Others are fixed guardrails that a conversation-specific
instruction may NOT override, no matter what was asked.

The request it answered:
{input}

The answer:
{answer}

The checks, in order:
{checks}

For each check, say whether the answer satisfied it:
- "yes" — satisfied
- "no"  — violated
- "idk" — nothing in the answer bears on this check

Reply with JSON: {{"verdicts": [{{"verdict": "yes|no|idk", "reason": "..."}}, ...]}}
with exactly {count} verdict(s), in the same order as the checks.
"""

# Fraction of a note's words that must land inside one note-sized window of
# the answer for it to count as leaked. Below this, overlap on short texts is
# not distinguishable from two sentences happening to share common words.
_LEAK_SIMILARITY = 0.8

_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    """Lowercased word tokens, punctuation dropped.

    Deliberately *not* ``metrics._tokens`` (which keeps ``.`` inside tokens so
    ``lib.member`` survives): a leaked note is prose, and "supervisor." must
    match "supervisor".
    """
    return _WORD_RE.findall(text.lower())


def _policy_texts(
    policy: Any | None, run: EvaluationRun, *, overridable: bool | None = None
) -> list[str]:
    """Instruction texts from an explicit policy, else the run's own.

    *policy* may be a :class:`~memory.policy.TaskPolicy` (duck-typed on
    ``texts``), a plain sequence of strings, or ``None``. Filtering by
    *overridable* is only possible where the source carries the flag — a bare
    list of strings has no notion of a fixed rule and yields nothing when a
    filter is asked for.
    """
    if policy is not None:
        texts = getattr(policy, "texts", None)
        if callable(texts):
            resolved = texts(overridable=overridable)
            return (
                [str(entry) for entry in resolved]
                if isinstance(resolved, (list, tuple))
                else []
            )
        if isinstance(policy, (list, tuple)):
            return [] if overridable is not None else [str(entry) for entry in policy]
    entries = []
    for entry in run.task_policy:
        if not isinstance(entry, dict):
            continue
        if overridable is not None and bool(entry.get("overridable", True)) != overridable:
            continue
        text = str(entry.get("text", "")).strip()
        if text:
            entries.append(text)
    return entries


class PolicyAdherenceMetric(JudgedMetric):
    """
    ``policy instructions followed / total`` — the long-term memory channel's
    counterpart to :class:`~validation.agentic_metrics.PromptAlignmentMetric`.

    Instructions resolve as: explicit constructor *policy* > the run's
    ``task_policy`` (which ``validation.conversation`` fills from the
    ``TaskPolicy`` a thread ran under). A run with neither skips — unlike
    ``prompt_alignment``, there is no sensible default policy to fall back
    on: a task nobody wrote instructions for has none.

    One judge call per unit.
    """

    name = "policy_adherence"
    default_threshold = 0.8

    def __init__(
        self,
        llm: Any,
        *,
        threshold: float | None = None,
        policy: Any | None = None,
    ) -> None:
        super().__init__(llm, threshold=threshold)
        self._policy = policy

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        instructions = _policy_texts(self._policy, run)
        if not instructions:
            return self._result(1.0, "no task policy on this run", skipped=True)
        inputs = run.inputs
        scores: list[float] = []
        judged = 0
        first_reason = ""
        for index, response in enumerate(run.responses):
            if not response.strip():
                continue
            judged += 1
            verdicts = self._classify(
                _POLICY_PROMPT.format(
                    input=inputs[index] if index < len(inputs) else "",
                    answer=response,
                    instructions=numbered(instructions),
                    count=len(instructions),
                ),
                len(instructions),
            )
            if verdicts is None:
                scores.append(0.0)
                continue
            scores.append(self._ratio(verdicts))
            first_reason = first_reason or self._first_reason(verdicts)
        if not judged:
            return self._result(1.0, "no responses to check", skipped=True)
        details = f"{len(instructions)} policy instruction(s) over {judged} unit(s)"
        fingerprint = getattr(self._policy, "fingerprint", "")
        if fingerprint:
            details += f"; policy {fingerprint}"
        if first_reason:
            details += f"; e.g. {first_reason}"
        return self._result(sum(scores) / len(scores), details)


class OverrideComplianceMetric(JudgedMetric):
    """
    Both halves of the short-term/long-term precedence rule, in one score.

    The checklist per unit is:

    * every live thread note — *was this honoured?*
    * every **fixed** policy instruction — *was this still respected, despite
      the notes?*

    So a model that quietly ignores "exception approved by supervisor" and a
    model that lets that exception talk it past "escalate refunds above $500"
    both lose points, which is the property that makes short-term memory safe
    to enable at all. Skips when the run carries no thread notes.

    One judge call per unit.
    """

    name = "override_compliance"
    default_threshold = 0.9

    def __init__(
        self,
        llm: Any,
        *,
        threshold: float | None = None,
        policy: Any | None = None,
    ) -> None:
        super().__init__(llm, threshold=threshold)
        self._policy = policy

    def _checks(self, run: EvaluationRun) -> list[str]:
        checks = [
            f"The answer honours this conversation-specific instruction: {note}"
            for note in run.thread_notes
            if str(note).strip()
        ]
        checks += [
            "The answer still respects this fixed standing instruction, even "
            f"if a conversation-specific instruction pushed against it: {text}"
            for text in _policy_texts(self._policy, run, overridable=False)
        ]
        return checks

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        if not run.thread_notes:
            return self._result(
                1.0, "no conversation-specific instructions in force", skipped=True
            )
        checks = self._checks(run)
        inputs = run.inputs
        scores: list[float] = []
        judged = 0
        first_reason = ""
        for index, response in enumerate(run.responses):
            if not response.strip():
                continue
            judged += 1
            verdicts = self._classify(
                _OVERRIDE_PROMPT.format(
                    input=inputs[index] if index < len(inputs) else "",
                    answer=response,
                    checks=numbered(checks),
                    count=len(checks),
                ),
                len(checks),
            )
            if verdicts is None:
                scores.append(0.0)
                continue
            scores.append(self._ratio(verdicts))
            first_reason = first_reason or self._first_reason(verdicts)
        if not judged:
            return self._result(1.0, "no responses to check", skipped=True)
        fixed = len(checks) - len(run.thread_notes)
        details = (
            f"{len(run.thread_notes)} override(s) + {fixed} fixed rule(s) "
            f"over {judged} unit(s)"
        )
        if first_reason:
            details += f"; e.g. {first_reason}"
        return self._result(sum(scores) / len(scores), details)


class MemoryExtractionMetric(ValidationMetric):
    """
    F1 of what the extractor wrote against what it should have written.

    Deterministic, no judge: the run declares ``expected_memories`` (a
    labelled turn set) and ``extracted_memories`` (what
    ``memory.extractor`` actually produced), and they are matched greedily by
    token overlap — a memory rewritten as "Escalate refunds over $500" and one
    labelled "Escalate refund requests above $500" are the same memory, and
    demanding string equality would measure phrasing rather than behaviour.

    Precision matters more than recall here and the F1 hides that, so the
    details string reports both: a missed memory is a lost instruction, but a
    *fabricated* one is a durable instruction nobody gave.

    Skips when the run declares no expectations.
    """

    name = "memory_extraction"
    default_threshold = 0.7
    # Token-F1 above which two phrasings are treated as the same memory.
    match_similarity = 0.6

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        expected = [text for text in run.expected_memories if text.strip()]
        extracted = [text for text in run.extracted_memories if text.strip()]
        if not expected and not extracted:
            return self._result(1.0, "no memory expectations declared", skipped=True)
        if not expected:
            return self._result(
                0.0, f"{len(extracted)} memory/ies extracted, none expected"
            )
        if not extracted:
            return self._result(0.0, f"0/{len(expected)} expected memory/ies extracted")
        matched = _greedy_matches(extracted, expected, self.match_similarity)
        precision = matched / len(extracted)
        recall = matched / len(expected)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return self._result(
            f1,
            f"precision {precision:.2f} ({matched}/{len(extracted)}), "
            f"recall {recall:.2f} ({matched}/{len(expected)})",
        )


class MemoryLeakageMetric(ValidationMetric):
    """
    None of *another* thread's notes may surface in this run's answers.

    The failure this guards against is the reason short-term memory is
    thread-scoped in the first place: "exception approved by supervisor" is
    true of one conversation and false of every other, and a store-wide note
    lookup — or a thread id collision — would spread it silently.

    Deterministic and deliberately narrow: it catches a foreign note quoted
    verbatim or near-verbatim (normalised substring, or token overlap at
    :data:`_LEAK_SIMILARITY`). Leakage that shows up only as changed
    *behaviour* is not detectable from text — that is what
    ``override_compliance`` judges. Skips when the run declares no
    ``foreign_notes``.
    """

    name = "memory_leakage"
    default_threshold = 1.0

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        foreign = [note for note in run.foreign_notes if note.strip()]
        if not foreign:
            return self._result(
                1.0, "no foreign thread notes to check against", skipped=True
            )
        haystack = _words(run.joined_responses)
        if not haystack:
            return self._result(1.0, "no responses to check", skipped=True)
        leaked = [note for note in foreign if _leaked(note, haystack)]
        if leaked:
            logger.warning(
                f"{self.name}: {len(leaked)} foreign note(s) surfaced in run "
                f"'{run.run_id}'; first: {leaked[0][:80]!r}"
            )
        return self._result(
            1.0 - len(leaked) / len(foreign),
            f"{len(leaked)}/{len(foreign)} foreign note(s) surfaced"
            + (f"; e.g. {leaked[0][:60]}" if leaked else ""),
        )


def _leaked(note: str, haystack: Sequence[str]) -> bool:
    """Whether *note* surfaces in the answer's word stream *haystack*.

    Scanned as a sliding window the size of the note, not against the whole
    answer: a note quoted inside a long response is still a leak, and
    comparing it to the entire text would drown that in length.
    """
    needle = _words(note)
    if not needle or not haystack:
        return False
    needed = Counter(needle)
    span = len(needle)
    best = 0.0
    for start in range(max(len(haystack) - span, 0) + 1):
        window = Counter(haystack[start : start + span])
        best = max(best, sum((needed & window).values()) / span)
        if best >= 1.0:
            break
    return best >= _LEAK_SIMILARITY


def _greedy_matches(
    candidates: Sequence[str], references: Sequence[str], floor: float
) -> int:
    """How many *candidates* pair off with a distinct reference above *floor*.

    Greedy best-first over all pairs: one extracted memory can satisfy at most
    one expectation, so a duplicate extraction cannot inflate recall.
    """
    scored = sorted(
        (
            (_token_f1(candidate, reference), c_index, r_index)
            for c_index, candidate in enumerate(candidates)
            for r_index, reference in enumerate(references)
        ),
        reverse=True,
    )
    used_candidates: set[int] = set()
    used_references: set[int] = set()
    matched = 0
    for score, c_index, r_index in scored:
        if score < floor:
            break
        if c_index in used_candidates or r_index in used_references:
            continue
        used_candidates.add(c_index)
        used_references.add(r_index)
        matched += 1
    return matched


def memory_metrics(
    llm: Any | None = None,
    *,
    policy: Any | None = None,
    include: Iterable[str] | None = None,
) -> list[ValidationMetric]:
    """The memory suite: the deterministic pair, plus the judged pair when a
    judge is given.

    Kept out of :func:`~validation.metrics.default_metrics` on purpose — the
    deterministic two are cheap but score nothing unless a run declares memory
    expectations, and every extra always-skipped row makes a report harder to
    read. The judged two are additionally in
    :data:`~validation.metrics.JUDGED_METRIC_NAMES`, so the full judged suite
    already includes them.

    Parameters
    ----------
    llm : Any | None
        Judge model for ``policy_adherence`` / ``override_compliance``.
        ``None`` builds the deterministic pair only.
    policy : Any | None
        A :class:`~memory.policy.TaskPolicy` (or list of instruction texts)
        to score against, when the runs themselves do not carry one.
    include : Iterable[str] | None
        Metric names to build; ``None`` builds everything available.
    """
    builders: dict[str, Any] = {
        "memory_extraction": MemoryExtractionMetric,
        "memory_leakage": MemoryLeakageMetric,
    }
    if llm is not None:
        builders["policy_adherence"] = lambda: PolicyAdherenceMetric(
            llm, policy=policy
        )
        builders["override_compliance"] = lambda: OverrideComplianceMetric(
            llm, policy=policy
        )
    wanted = list(builders) if include is None else list(include)
    unknown = [name for name in wanted if name not in builders]
    if unknown:
        raise ValueError(
            f"unknown (or judge-less) memory metric(s): {', '.join(sorted(unknown))}; "
            f"available: {', '.join(builders)}"
        )
    selected = [name for name in builders if name in set(wanted)]
    logger.info(f"memory_metrics: [{', '.join(selected)}]")
    return [builders[name]() for name in selected]
