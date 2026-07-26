"""RAG-quality metrics, judged by a model. See validation/README.md.

Five metrics over the retrieval half of the pipeline — the reference-PDF
guidance :mod:`prompt_builder` selects per item, which nothing measured before:

    faithfulness          the translation does not contradict the guidance
    answer_relevancy      the response actually answers the item it was given
    hallucination         the translation does not contradict the SAS source
    contextual_precision  the useful guidance chunks ranked above the useless
    contextual_relevancy  the retrieved guidance was on-topic at all

They implement deepeval's published definitions (deepeval.com/docs/metrics-*)
natively rather than importing the package — see validation/README.md for why —
with three deliberate departures, each called out on the class that makes it:

* ``hallucination`` reports **groundedness** (``1 - contradicted/total``), so
  that higher-is-better holds across the whole report;
* a unit with no signal is ``skipped``, not scored 1.0;
* ``faithfulness`` passes the retrieval context verbatim as the fact set
  instead of paying for deepeval's separate truths-extraction call.

Each metric scores one unit (item/turn) at a time and averages, as
:class:`~validation.judge.LLMJudgeMetric` does: one ``EvaluationRun`` here is a
whole run of N items, not a single deepeval test case.

Logger name: ``validation.rag_metrics``.
"""

from __future__ import annotations

import logging

from .judged import JudgedMetric, numbered
from .models import EvaluationRun, MetricResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CLAIMS_PROMPT = """\
Extract every factual claim the following answer makes about the code it
translates or the platform it targets. Split compound sentences; keep each
claim self-contained and verifiable. Ignore hedging and pure formatting.

Answer:
{answer}

Reply with JSON: {{"statements": ["<claim>", ...]}}
"""

_FAITHFULNESS_PROMPT = """\
You are checking whether an answer stays faithful to the reference material it
was given.

Reference material:
{context}

Claims made by the answer:
{claims}

For each claim, in order, say whether the reference material contradicts it:
- "yes" — the reference material does not contradict the claim (faithful)
- "no"  — the reference material contradicts the claim
- "idk" — the reference material neither supports nor contradicts it

A claim the reference material simply does not mention is "idk", not "no".

Reply with JSON: {{"verdicts": [{{"verdict": "yes|no|idk", "reason": "..."}}, ...]}}
with exactly {count} verdict(s), in the same order as the claims.
"""

_STATEMENTS_PROMPT = """\
Break the following answer into its individual statements — each a single
self-contained assertion, instruction, or block of code with its purpose.

Answer:
{answer}

Reply with JSON: {{"statements": ["<statement>", ...]}}
"""

_RELEVANCY_PROMPT = """\
You are checking whether an answer stays on the question it was asked.

The request:
{input}

Statements made by the answer:
{statements}

For each statement, in order, say whether it addresses the request:
- "yes" — it contributes to answering the request
- "no"  — it is off-topic or irrelevant filler
- "idk" — it is background that neither helps nor hurts

Reply with JSON: {{"verdicts": [{{"verdict": "yes|no|idk", "reason": "..."}}, ...]}}
with exactly {count} verdict(s), in the same order as the statements.
"""

_HALLUCINATION_PROMPT = """\
You are checking an answer against ground-truth reference material.

Answer:
{answer}

Reference contexts:
{contexts}

For each context, in order, say whether the answer contradicts it:
- "yes" — the answer agrees with this context (or says nothing about it)
- "no"  — the answer contradicts this context
- "idk" — genuinely undecidable

Reply with JSON: {{"verdicts": [{{"verdict": "yes|no|idk", "reason": "..."}}, ...]}}
with exactly {count} verdict(s), in the same order as the contexts.
"""

_PRECISION_PROMPT = """\
You are auditing a retriever's ranking.

The request:
{input}

The ideal answer:
{expected}

Retrieved contexts, in the order the retriever ranked them:
{contexts}

For each context, in order, say whether it is relevant to producing the ideal
answer for this request:
- "yes" — relevant
- "no"  — not relevant

Reply with JSON: {{"verdicts": [{{"verdict": "yes|no", "reason": "..."}}, ...]}}
with exactly {count} verdict(s), in the same order as the contexts.
"""

_CONTEXT_STATEMENTS_PROMPT = """\
Break the following retrieved reference material into its individual
statements — each a single self-contained fact, rule, or example.

Reference material:
{context}

Reply with JSON: {{"statements": ["<statement>", ...]}}
"""

_CONTEXT_RELEVANCY_PROMPT = """\
You are checking whether retrieved reference material is on-topic.

The request:
{input}

Statements found in the retrieved material:
{statements}

For each statement, in order, say whether it is relevant to the request:
- "yes" — relevant to answering this request
- "no"  — irrelevant to this request
- "idk" — tangentially related

Reply with JSON: {{"verdicts": [{{"verdict": "yes|no|idk", "reason": "..."}}, ...]}}
with exactly {count} verdict(s), in the same order as the statements.
"""


def _joined(contexts: list[str]) -> str:
    return "\n\n---\n\n".join(contexts)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class FaithfulnessMetric(JudgedMetric):
    """
    ``truthful claims / total claims`` — deepeval's *Faithfulness*, scored
    against the reference guidance retrieved for each item.

    Two judge calls per unit: extract the claims the translation makes, then
    classify each against the retrieval context. deepeval extracts a separate
    "truths" list from the context first; the context here is already a short,
    curated set of manual sections (``prompt_builder`` word-budgets it), so it
    is handed to the classifier verbatim and that third call is skipped.

    A unit with no retrieval context contributes nothing; a run where no unit
    had any (guidance off, or a thread whose ephemeral guidance was never
    persisted) is skipped.
    """

    name = "faithfulness"
    default_threshold = 0.7

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        scores: list[float] = []
        judged = 0
        first_reason = ""
        for index, response in enumerate(run.responses):
            context = run.retrieval_context_at(index)
            if not context or not response.strip():
                continue
            judged += 1
            claims = self._extract(_CLAIMS_PROMPT.format(answer=response))
            if claims is None:
                scores.append(0.0)
                continue
            if not claims:
                # An answer that asserts nothing cannot be unfaithful.
                scores.append(1.0)
                continue
            verdicts = self._classify(
                _FAITHFULNESS_PROMPT.format(
                    context=_joined(context),
                    claims=numbered(claims),
                    count=len(claims),
                ),
                len(claims),
            )
            if verdicts is None:
                scores.append(0.0)
                continue
            scores.append(self._ratio(verdicts))
            first_reason = first_reason or self._first_reason(verdicts)
        if not judged:
            return self._result(
                1.0, "no retrieval context to check against", skipped=True
            )
        score = sum(scores) / len(scores)
        details = f"mean over {judged} unit(s) with retrieval context"
        if first_reason:
            details += f"; e.g. {first_reason}"
        return self._result(score, details)


class AnswerRelevancyMetric(JudgedMetric):
    """
    ``relevant statements / total statements`` — deepeval's *Answer Relevancy*.

    Referenceless: it needs only the item's prompt and the response, so it
    scores every mode (offline case, inline item, reconstructed thread, bare
    transcript). Two judge calls per unit: split the response into statements,
    then classify each against the request.
    """

    name = "answer_relevancy"
    default_threshold = 0.7

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        inputs = run.inputs
        scores: list[float] = []
        judged = 0
        first_reason = ""
        for index, response in enumerate(run.responses):
            request = inputs[index] if index < len(inputs) else ""
            if not request.strip() or not response.strip():
                continue
            judged += 1
            statements = self._extract(_STATEMENTS_PROMPT.format(answer=response))
            if statements is None:
                scores.append(0.0)
                continue
            if not statements:
                scores.append(0.0)
                continue
            verdicts = self._classify(
                _RELEVANCY_PROMPT.format(
                    input=request,
                    statements=numbered(statements),
                    count=len(statements),
                ),
                len(statements),
            )
            if verdicts is None:
                scores.append(0.0)
                continue
            scores.append(self._ratio(verdicts))
            first_reason = first_reason or self._first_reason(verdicts)
        if not judged:
            return self._result(1.0, "no request/response pair to judge", skipped=True)
        details = f"mean over {judged} unit(s)"
        if first_reason:
            details += f"; e.g. {first_reason}"
        return self._result(sum(scores) / len(scores), details)


class HallucinationMetric(JudgedMetric):
    """
    Groundedness against the item's own SAS source — deepeval's *Hallucination*,
    **inverted**.

    deepeval scores ``contradicted contexts / total contexts``, where lower is
    better and the threshold is a maximum. Every other verdict in this package
    is higher-is-better (``MetricResult.passed`` is ``score >= threshold``,
    ``CaseResult.score`` is a mean, and a failing metric is fed back to the
    model as "score X < threshold Y" by
    ``SasLLMPipeline._validation_feedback_message``), so this reports
    ``1 - contradicted/total`` and puts deepeval's raw rate in ``details``.

    The context is deepeval's ``context``, not ``retrieval_context``: the SAS
    source being translated, plus the golden translation when the run has one.
    Contradicting the source is the failure that matters here — a translation
    that silently changes what the program does.
    """

    name = "hallucination"
    default_threshold = 0.8

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        contradicted = 0
        total = 0
        first_reason = ""
        for index, response in enumerate(run.responses):
            contexts = run.contexts_at(index)
            if not contexts or not response.strip():
                continue
            verdicts = self._classify(
                _HALLUCINATION_PROMPT.format(
                    answer=response,
                    contexts=numbered(contexts),
                    count=len(contexts),
                ),
                len(contexts),
            )
            if verdicts is None:
                # An unusable reply is scored as a contradiction rather than a
                # free pass: this metric guards against silent wrongness.
                contradicted += len(contexts)
                total += len(contexts)
                continue
            contradicted += sum(1 for v in verdicts if v.verdict == "no")
            total += len(verdicts)
            first_reason = first_reason or self._first_reason(verdicts)
        if not total:
            return self._result(
                1.0, "no ground-truth context (no item metadata)", skipped=True
            )
        rate = contradicted / total
        details = (
            f"{contradicted}/{total} context(s) contradicted "
            f"(deepeval hallucination rate {rate:.3f})"
        )
        if first_reason:
            details += f"; e.g. {first_reason}"
        return self._result(1.0 - rate, details)


class ContextualPrecisionMetric(JudgedMetric):
    """
    Ranking-aware precision over the retrieved guidance — deepeval's
    *Contextual Precision*, formula unchanged::

        (1 / |relevant|) * sum_k (relevant_up_to_k / k) * r_k

    so an irrelevant chunk hurts more the higher the retriever ranked it. The
    picks arrive in ``InstructionSelector``'s own priority order (user rules,
    pinned, hazard, construct, topical), which *is* the ranking under test.

    Needs an expected output to judge relevance against, exactly as deepeval
    does, so it skips a run with no ``reference_translation`` — the same
    condition ``reference_similarity`` skips on. One judge call per unit.
    """

    name = "contextual_precision"
    default_threshold = 0.6

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        expected = run.reference_translation
        if not expected:
            return self._result(
                1.0,
                "no reference translation to rank relevance against",
                skipped=True,
            )
        inputs = run.inputs
        scores: list[float] = []
        judged = 0
        for index in range(len(run.responses)):
            context = run.retrieval_context_at(index)
            if not context:
                continue
            judged += 1
            verdicts = self._classify(
                _PRECISION_PROMPT.format(
                    input=inputs[index] if index < len(inputs) else "",
                    expected=expected,
                    contexts=numbered(context),
                    count=len(context),
                ),
                len(context),
            )
            if verdicts is None:
                scores.append(0.0)
                continue
            relevant = [v.verdict == "yes" for v in verdicts]
            scores.append(_weighted_precision(relevant))
        if not judged:
            return self._result(
                1.0, "no retrieval context to rank", skipped=True
            )
        return self._result(
            sum(scores) / len(scores),
            f"weighted precision over {judged} ranked context list(s)",
        )


def _weighted_precision(relevant: list[bool]) -> float:
    """deepeval's weighted cumulative precision over a ranked relevance list.

    0.0 when nothing is relevant (the ``1/|relevant|`` factor is undefined);
    1.0 when every relevant node precedes every irrelevant one.
    """
    total_relevant = sum(relevant)
    if not total_relevant:
        return 0.0
    running = 0
    accumulated = 0.0
    for position, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            running += 1
            accumulated += running / position
    return accumulated / total_relevant


class ContextualRelevancyMetric(JudgedMetric):
    """
    ``relevant statements / total statements`` over the retrieved guidance —
    deepeval's *Contextual Relevancy*.

    Where :class:`ContextualPrecisionMetric` asks whether the *ranking* was
    right, this asks whether the retrieved text was on-topic at all — the
    signal that catches a query or construct mapping sending
    ``InstructionSelector`` after the wrong manual sections. Two judge calls
    per unit: split the retrieved material into statements, then classify each
    against the request.
    """

    name = "contextual_relevancy"
    default_threshold = 0.5

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        inputs = run.inputs
        scores: list[float] = []
        judged = 0
        first_reason = ""
        for index in range(len(run.responses)):
            context = run.retrieval_context_at(index)
            request = inputs[index] if index < len(inputs) else ""
            if not context or not request.strip():
                continue
            judged += 1
            statements = self._extract(
                _CONTEXT_STATEMENTS_PROMPT.format(context=_joined(context))
            )
            if not statements:
                scores.append(0.0)
                continue
            verdicts = self._classify(
                _CONTEXT_RELEVANCY_PROMPT.format(
                    input=request,
                    statements=numbered(statements),
                    count=len(statements),
                ),
                len(statements),
            )
            if verdicts is None:
                scores.append(0.0)
                continue
            scores.append(self._ratio(verdicts))
            first_reason = first_reason or self._first_reason(verdicts)
        if not judged:
            return self._result(
                1.0, "no retrieval context to assess", skipped=True
            )
        details = f"mean over {judged} unit(s) with retrieval context"
        if first_reason:
            details += f"; e.g. {first_reason}"
        return self._result(sum(scores) / len(scores), details)
