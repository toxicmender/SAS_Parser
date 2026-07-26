"""Summarization metrics, judged by a model. See validation/README.md.

deepeval's *Summarization* is ``min(alignment, coverage)``:

* **alignment** — nothing in the summary contradicts the source (no
  hallucinated detail);
* **coverage** — the summary still answers the yes/no questions the source
  answers (nothing important dropped).

The minimum, not the mean, because the two failures are not interchangeable: a
summary that invents facts is not redeemed by being thorough.

Two things in this repo are summaries, and each gets a metric:

``summarization``
    :class:`memory.summarize.RollingSummarizer`'s per-thread summary against
    the conversation turns it covers. That summarizer's prompt promises to
    "preserve exact identifiers (dataset, table, macro, file, function, and
    variable names)" because later turns depend on recalling them — coverage
    over auto-generated questions is precisely the check on that promise.

``analysis_summarization``
    Each item's ``## Analysis`` against the SAS source it describes. Always
    present (the system prompt requires it), so it scores every run.

Logger name: ``validation.summarization``.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from .judged import JudgedMetric, markdown_section, numbered
from .models import EvaluationRun, MetricResult

logger = logging.getLogger(__name__)


_SUMMARY_CLAIMS_PROMPT = """\
Extract every factual claim the following summary makes. Split compound
sentences; keep each claim self-contained and verifiable.

Summary:
{summary}

Reply with JSON: {{"statements": ["<claim>", ...]}}
"""

_ALIGNMENT_PROMPT = """\
You are checking a summary against the text it summarises.

Original text:
{source}

Claims made by the summary:
{claims}

For each claim, in order, say whether the original text supports it:
- "yes" — the original text supports the claim
- "no"  — the original text contradicts it, or the claim is invented
- "idk" — genuinely undecidable from the original text

Reply with JSON: {{"verdicts": [{{"verdict": "yes|no|idk", "reason": "..."}}, ...]}}
with exactly {count} verdict(s), in the same order as the claims.
"""

_QUESTIONS_PROMPT = """\
Write {n} closed yes/no questions about the following text. Each must be
answerable "yes" from the text itself, and each must target a different key
point — the facts, identifiers and decisions a reader must not lose.

Text:
{source}

Reply with JSON: {{"statements": ["<question>", ...]}}
"""

_COVERAGE_PROMPT = """\
Answer each question using ONLY the text below. Answer "yes" when the text
answers the question affirmatively, "no" when it does not, and "idk" when the
text does not address the question at all.

Text:
{text}

Questions:
{questions}

Reply with JSON: {{"verdicts": [{{"verdict": "yes|no|idk", "reason": "..."}}, ...]}}
with exactly {count} verdict(s), in the same order as the questions.
"""


class _SummarizationScorer(JudgedMetric):
    """Shared ``min(alignment, coverage)`` scoring for the two metrics below.

    Four judge calls per (source, summary) pair — claims from the summary,
    verdicts against the source, coverage questions from the source, those
    questions answered from the summary — or three when the caller supplies
    ``assessment_questions``. deepeval asks its questions of both texts; the
    questions are generated from the source and required to be answerable
    "yes" there, so only the summary needs asking, which is one call less for
    the same meaning.
    """

    def __init__(
        self,
        llm: Any,
        *,
        threshold: float | None = None,
        n: int = 5,
        assessment_questions: Sequence[str] | None = None,
    ) -> None:
        super().__init__(llm, threshold=threshold)
        self._n = n
        self._questions = list(assessment_questions) if assessment_questions else []

    def _alignment(self, source: str, summary: str) -> tuple[float, str]:
        claims = self._extract(_SUMMARY_CLAIMS_PROMPT.format(summary=summary))
        if claims is None:
            return 0.0, "unusable judge reply extracting claims"
        if not claims:
            # A summary asserting nothing cannot contradict the source.
            return 1.0, ""
        verdicts = self._classify(
            _ALIGNMENT_PROMPT.format(
                source=source, claims=numbered(claims), count=len(claims)
            ),
            len(claims),
        )
        if verdicts is None:
            return 0.0, "unusable judge reply classifying claims"
        return self._ratio(verdicts), self._first_reason(verdicts)

    def _coverage(self, source: str, summary: str) -> tuple[float, str]:
        questions = self._questions
        if not questions:
            generated = self._extract(
                _QUESTIONS_PROMPT.format(source=source, n=self._n)
            )
            if not generated:
                return 0.0, "unusable judge reply generating questions"
            questions = generated
        verdicts = self._classify(
            _COVERAGE_PROMPT.format(
                text=summary, questions=numbered(questions), count=len(questions)
            ),
            len(questions),
        )
        if verdicts is None:
            return 0.0, "unusable judge reply answering questions"
        # Strict: an unanswerable question is a dropped key point, so "idk"
        # counts against coverage where it would not against alignment.
        answered = sum(1 for v in verdicts if v.verdict == "yes")
        reason = ""
        for question, verdict in zip(questions, verdicts):
            if verdict.verdict != "yes":
                reason = f"unanswered: {question}"
                break
        return answered / len(questions), reason

    def _score_pair(self, source: str, summary: str) -> tuple[float, str]:
        alignment, alignment_reason = self._alignment(source, summary)
        coverage, coverage_reason = self._coverage(source, summary)
        reason = alignment_reason if alignment <= coverage else coverage_reason
        detail = f"alignment {alignment:.3f}, coverage {coverage:.3f}"
        if reason:
            detail += f"; {reason}"
        return min(alignment, coverage), detail


class SummarizationMetric(_SummarizationScorer):
    """
    ``min(alignment, coverage)`` for the rolling conversation summary — the
    summary :class:`memory.summarize.RollingSummarizer` folds older turns into
    and prompts back on every subsequent turn.

    Needs both halves on the run (``summary`` and ``summary_source``);
    :func:`validation.conversation.run_from_thread` fills them in from the
    thread's stored summary, and other modes leave them ``None``. Skips when
    either is missing — which is the common case, since a summary only exists
    once the folded turns pass ``trigger_tokens``.

    Four judge calls per run.
    """

    name = "summarization"
    default_threshold = 0.6

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        summary = (run.summary or "").strip()
        source = (run.summary_source or "").strip()
        if not summary or not source:
            return self._result(
                1.0, "no rolling summary folded for this run", skipped=True
            )
        score, detail = self._score_pair(source, summary)
        return self._result(score, detail)


class AnalysisSummarizationMetric(_SummarizationScorer):
    """
    ``min(alignment, coverage)`` for each item's ``## Analysis`` against the SAS
    source it describes, averaged over the run.

    The Analysis section is the model's account of what the SAS does before it
    translates anything, so scoring it as a summary catches the failure that
    matters upstream of the code: an Analysis that invents behaviour the source
    does not have, or omits a step the translation then also omits.

    Needs chunker metadata for the source text, so it skips in thread and
    transcript modes. Four judge calls per item.
    """

    name = "analysis_summarization"
    default_threshold = 0.6

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        documents = run.documents
        sources = run.item_sources
        scores: list[float] = []
        first_detail = ""
        for index, response in enumerate(run.responses):
            if index >= len(sources) or not sources[index].strip():
                continue
            document = documents[index] if index < len(documents) else None
            analysis = ""
            if document:
                analysis = str(document.get("analysis") or "").strip()
            analysis = analysis or markdown_section(response, "Analysis")
            if not analysis:
                continue
            score, detail = self._score_pair(sources[index], analysis)
            scores.append(score)
            first_detail = first_detail or detail
        if not scores:
            return self._result(
                1.0, "no Analysis section over a known SAS source", skipped=True
            )
        details = f"mean over {len(scores)} item(s)"
        if first_detail:
            details += f"; first: {first_detail}"
        return self._result(sum(scores) / len(scores), details)
