"""Shared machinery for the LLM-judged metrics. See validation/README.md.

Everything in :mod:`validation.rag_metrics`, :mod:`validation.agentic_metrics`,
and :mod:`validation.summarization` is built on :class:`JudgedMetric`: a
:class:`~validation.metrics.ValidationMetric` that reaches a judge model,
reports what judging cost, and — crucially — never lets a misbehaving judge
break a run. That last rule is inherited from
:class:`~validation.judge.LLMJudgeMetric`: an unusable reply degrades to a
warning and a zero (or a skip), never an exception.

Two judge shapes are supported through one call path (:meth:`JudgedMetric._ask`):

- an :class:`llm_client.LLMClient`, which is asked for a **schema** via
  ``invoke_structured`` (gateway auth, retries, input budget, token accounting
  all come along);
- any LangChain-style ``invoke(input) -> message`` object — a raw chat model,
  or a ``FakeListChatModel`` scripted with JSON in the tests — whose reply text
  is parsed as JSON instead.

The verdict schemas below are deliberately the small, flat shapes deepeval's
own prompts ask for (a list of statements; a list of yes/no/idk verdicts with
reasons), so the same prompt works in either mode.

Logger name: ``validation.judged``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, Sequence, TypeVar

from pydantic import BaseModel, Field

from chunker.models import SasBatch, SasChunk

from .metrics import ValidationMetric

logger = logging.getLogger(__name__)

_SchemaT = TypeVar("_SchemaT", bound=BaseModel)


# ---------------------------------------------------------------------------
# Verdict schemas
# ---------------------------------------------------------------------------


class Statements(BaseModel):
    """An extraction step's output: the atomic units to be judged next."""

    statements: list[str] = Field(default_factory=list)


class Verdict(BaseModel):
    """One yes/no/idk judgement with the reason behind it.

    ``idk`` is deepeval's third option and is *not* a failure by default: an
    ambiguous claim is one the source neither supports nor contradicts, and
    counting it against the model would punish honest hedging. Metrics that
    want the strict reading pass ``penalize_ambiguous=True``.
    """

    verdict: Literal["yes", "no", "idk"] = "idk"
    reason: str = ""


class Verdicts(BaseModel):
    """A classification step's output, positionally aligned with its input."""

    verdicts: list[Verdict] = Field(default_factory=list)


class ScoreVerdict(BaseModel):
    """A direct 0–1 alignment score with its justification — the shape the
    single-call metrics (task completion, plan adherence) ask for."""

    score: float = 0.0
    reason: str = ""


# ---------------------------------------------------------------------------
# Text helpers, shared across the judged metrics
# ---------------------------------------------------------------------------


def item_source(item: SasBatch | SasChunk) -> str:
    """The SAS text of one work item (a batch's members joined in order)."""
    if isinstance(item, SasBatch):
        return "\n".join(chunk.text for chunk in item.chunks)
    return item.text


def markdown_section(text: str, heading: str) -> str:
    """The body of the ``## {heading}`` section of *text*, or ``""``.

    Used to recover the four sections
    (:mod:`chunker.pipeline_constants`'s contract) from a response that was
    *not* produced through structured output, so the judged metrics score an
    unstructured run the same way they score a structured one. Matches only
    level-2 headings, so the ``### {comment}`` headings
    :meth:`~chunker.response_models.TranslationDocument.to_markdown` writes
    inside a section do not end it.
    """
    match = re.search(
        rf"^##[ \t]+{re.escape(heading)}[ \t]*$\n(.*?)(?=^##[ \t]|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def numbered(entries: Sequence[str]) -> str:
    """*entries* as a 1-based numbered list — the format every classification
    prompt uses, so a verdict list can be aligned back by position."""
    return "\n".join(f"{i}. {entry}" for i, entry in enumerate(entries, start=1))


# ---------------------------------------------------------------------------
# JSON recovery from a prose reply
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _json_candidates(text: str) -> list[str]:
    """Substrings of *text* that might be the JSON object, best guess first:
    the text itself, then any fenced block, then the first brace-balanced
    span (so a judge that wraps its JSON in prose is still understood)."""
    candidates = [text.strip()]
    candidates.extend(body.strip() for body in _FENCE_RE.findall(text))
    start = text.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            char = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
    return candidates


def parse_json_reply(schema: type[_SchemaT], text: str) -> _SchemaT | None:
    """*text* validated against *schema*, or ``None`` when nothing parses."""
    for candidate in _json_candidates(text):
        if not candidate:
            continue
        try:
            return schema.model_validate(json.loads(candidate))
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class JudgedMetric(ValidationMetric):
    """
    Base for metrics that need a judge model.

    Parameters
    ----------
    llm : Any
        The judge. An :class:`llm_client.LLMClient` gets structured output and
        token accounting; anything with a LangChain-style ``invoke`` works too
        and is parsed from prose.
    threshold : float | None
        As :class:`~validation.metrics.ValidationMetric` — explicit argument >
        config.json ``validation.<name>_threshold`` > the class default.

    Subclasses implement ``evaluate`` and reach the judge through
    :meth:`_ask`. They must not raise on a bad reply: score the unit 0, or
    skip, and log.
    """

    default_threshold = 0.7

    def __init__(self, llm: Any, *, threshold: float | None = None) -> None:
        super().__init__(threshold)
        self._llm = llm
        # supports_structured_output() binds the schema to decide, which is
        # worth doing once per schema rather than once per unit.
        self._structured: dict[str, bool] = {}

    @property
    def token_usage(self) -> Any | None:
        """
        Tokens this judge has billed so far (an :class:`llm_client.TokenUsage`),
        or ``None`` when the judge reports none — a raw chat model or a fake.

        :attr:`~validation.runner.ValidationRunner.judge_token_usage` sums this
        across every metric that exposes it, so a judged suite's cost lands in
        ``ValidationReport.judge_token_usage`` with no further wiring.
        """
        return getattr(self._llm, "usage", None)

    # ------------------------------------------------------------------
    # Judge call
    # ------------------------------------------------------------------

    def _supports_structured(self, schema: type[BaseModel]) -> bool:
        supports = getattr(self._llm, "supports_structured_output", None)
        if supports is None:
            return False
        key = schema.__name__
        cached = self._structured.get(key)
        if cached is None:
            cached = bool(supports(schema))
            self._structured[key] = cached
        return cached

    def _ask(self, schema: type[_SchemaT], prompt: str) -> _SchemaT | None:
        """Ask the judge for *schema*, or ``None`` when the reply is unusable.

        Exactly one model call either way: in structured mode a schema the
        gateway would not honour arrives as ``parsed=None`` with the model's
        prose in ``raw`` (see :meth:`llm_client.LLMClient.invoke_structured`),
        and that prose is parsed as JSON rather than paying for a retry.
        """
        text = ""
        if self._supports_structured(schema):
            envelope = self._llm.invoke_structured(schema, prompt)
            parsed = envelope.get("parsed")
            if isinstance(parsed, schema):
                return parsed
            raw = envelope.get("raw")
            text = str(getattr(raw, "content", "") or "")
        else:
            reply = self._llm.invoke(prompt)
            text = str(getattr(reply, "content", reply))

        result = parse_json_reply(schema, text)
        if result is None:
            logger.warning(
                f"{self.name}: unusable judge reply for {schema.__name__}: "
                f"{text[:120]!r}"
            )
        return result

    # ------------------------------------------------------------------
    # Shared scoring shapes
    # ------------------------------------------------------------------

    def _extract(self, prompt: str) -> list[str] | None:
        """One extraction call — the statements/claims/questions a later
        classification call judges. ``None`` on an unusable reply."""
        result = self._ask(Statements, prompt)
        return None if result is None else [s for s in result.statements if s.strip()]

    def _classify(self, prompt: str, expected: int) -> list[Verdict] | None:
        """One classification call, returning exactly *expected* verdicts.

        A judge that returns too few is padded with ``idk`` (unknown, not
        guilty) and one that returns too many is truncated, both with a
        warning: a length mismatch is the judge's error, and silently
        re-scaling the denominator would hide it.
        """
        result = self._ask(Verdicts, prompt)
        if result is None:
            return None
        verdicts = list(result.verdicts)
        if len(verdicts) != expected:
            logger.warning(
                f"{self.name}: judge returned {len(verdicts)} verdict(s) for "
                f"{expected} item(s); padding/truncating"
            )
            verdicts = verdicts[:expected]
            verdicts += [Verdict() for _ in range(expected - len(verdicts))]
        return verdicts

    @staticmethod
    def _ratio(verdicts: Sequence[Verdict], *, penalize_ambiguous: bool = False) -> float:
        """Fraction of *verdicts* that count as a pass — deepeval's numerator
        for every ``relevant / total``-shaped metric."""
        if not verdicts:
            return 0.0
        ok = {"yes"} if penalize_ambiguous else {"yes", "idk"}
        return sum(1 for v in verdicts if v.verdict in ok) / len(verdicts)

    @staticmethod
    def _first_reason(verdicts: Sequence[Verdict]) -> str:
        """The first failing verdict's reason, for the result's ``details``."""
        for verdict in verdicts:
            if verdict.verdict == "no" and verdict.reason.strip():
                return verdict.reason.strip()
        return ""
