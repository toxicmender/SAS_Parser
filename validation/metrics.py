"""Deterministic validation metrics. See validation/README.md.

Every metric scores one :class:`~validation.models.CaseRun` to a value in
[0, 1] and compares it against a threshold resolved with the repo-wide
precedence rule (see the ``app_config`` package):

    explicit constructor argument > config.json ``validation.<name>_threshold``
    > the metric's class default

None of the metrics here needs a network, an API key, or MLflow — they are
plain functions of the pipeline's inputs and outputs, so the same suite runs
identically in CI (with a fake chat model) and against a live model.

Logger name: ``validation.metrics``.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Callable, Iterable

import app_config

from chunker.models import SasBatch, SasChunk
from target_language import (
    PROSE_FENCE_INFOS,
    TargetLanguage,
    normalize_language,
    resolve_target_language,
)

from .models import EvaluationRun, MetricResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared text helpers
# ---------------------------------------------------------------------------

# Fenced code block: info string (may be empty) + body. Non-greedy body so
# multiple blocks in one response are matched individually.
_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)

# A "## " section heading at the start of a line, matching pipeline.notebook's
# — the translation region is scored, not the whole response.
_SECTION_RE = re.compile(r"^## +(.+?) *$", re.MULTILINE)

_TOKEN_RE = re.compile(r"[a-z0-9_.]+")


def _fenced_blocks(text: str) -> list[tuple[str, str]]:
    """``(info, body)`` for every fenced block in *text*, info lower-cased."""
    return [(info.strip().lower(), body) for info, body in _FENCE_RE.findall(text)]


def _translation_region(response: str) -> str:
    """The ``## Translation`` section of *response*, or all of it.

    Language and syntax are judged on the translation only: a ```sas``` echo
    under ``## Analysis`` is illustration, and a response that never emitted
    the heading (the model deviated) is scored whole — the same fallback
    ``pipeline.notebook.markdown_to_cells`` makes, so the metrics see the
    blocks the notebook would actually run.
    """
    matches = list(_SECTION_RE.finditer(response))
    for i, match in enumerate(matches):
        if not match.group(1).strip().lower().startswith("translation"):
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(response)
        return response[match.end() : end]
    return response


def _target_blocks(text: str, target: TargetLanguage) -> list[str]:
    """Bodies of the translation-region blocks that are *target* code."""
    return [
        body
        for info, body in _fenced_blocks(_translation_region(text))
        if target.owns_fence(info)
    ]


def _as_target(value: str | TargetLanguage | None) -> TargetLanguage:
    """Accept a target, a language name, or ``None`` (the configured default).

    Metrics are constructed by callers that hold either — the pipeline has a
    resolved :class:`TargetLanguage`, a CLI or config-driven runner has a
    string — and neither should have to convert.
    """
    if isinstance(value, TargetLanguage):
        return value
    return resolve_target_language(value)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _token_f1(candidate: str, reference: str) -> float:
    """Multiset token F1 — order-insensitive overlap between two texts."""
    cand, ref = Counter(_tokens(candidate)), Counter(_tokens(reference))
    overlap = sum((cand & ref).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(cand.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def _item_datasets(item: SasBatch | SasChunk) -> list[str]:
    """The item's canonical input+output dataset names, physical paths
    excluded (quoted paths keep a leading ``'`` at extraction and cannot be
    expected verbatim in a translation)."""
    if isinstance(item, SasBatch):
        names = set(item.input_datasets) | set(item.output_datasets)
    else:
        names = set(item.metadata.input_datasets) | set(
            item.metadata.output_datasets
        )
    return sorted(n for n in names if not n.startswith("'"))


def _mentions(name: str, response: str) -> bool:
    """True when *name* (``lib.member``) or its bare member name appears in
    *response* as a standalone identifier."""
    low = response.lower()
    if name.lower() in low:
        return True
    bare = name.split(".")[-1].lower()
    return re.search(rf"(?<![\w.]){re.escape(bare)}(?![\w.])", low) is not None


# ---------------------------------------------------------------------------
# Metric base
# ---------------------------------------------------------------------------


class ValidationMetric(ABC):
    """
    Base class: one :class:`EvaluationRun` in, one :class:`MetricResult` out.

    Subclasses set ``name`` (also the config.json key prefix,
    ``validation.<name>_threshold``) and ``default_threshold``, and implement
    :meth:`evaluate`. Runs without chunker metadata (``items`` empty — live
    threads, arbitrary transcripts) must degrade cleanly: score from what is
    present, or skip.
    """

    name: str = "metric"
    default_threshold: float = 0.7

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = app_config.resolve(
            threshold,
            "validation",
            f"{self.name}_threshold",
            type(self).default_threshold,
        )

    @abstractmethod
    def evaluate(self, run: EvaluationRun) -> MetricResult: ...

    def _result(
        self, score: float, details: str = "", *, skipped: bool = False
    ) -> MetricResult:
        score = max(0.0, min(1.0, score))
        result = MetricResult(
            metric=self.name,
            score=score,
            threshold=self.threshold,
            passed=skipped or score >= self.threshold,
            skipped=skipped,
            details=details,
        )
        logger.debug(
            f"{self.name}: score={result.score:.3f}  threshold={self.threshold}  "
            f"passed={result.passed}  skipped={skipped}"
        )
        return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class ResponseCoverageMetric(ValidationMetric):
    """Every expected unit must have produced a non-empty response.

    The denominator is :attr:`EvaluationRun.expected_units` — items when
    chunker metadata exists, turns otherwise — so both an empty response and
    a missing output (unit/output count mismatch) lower the score.
    """

    name = "response_coverage"
    default_threshold = 1.0

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        total = run.expected_units
        if not total:
            return self._result(0.0, "no items or turns to cover")
        answered = sum(
            1 for response in run.responses[:total] if response.strip()
        )
        return self._result(
            answered / total,
            f"{answered}/{total} unit(s) answered",
        )


class DatasetFidelityMetric(ValidationMetric):
    """The translation of each item should account for that item's datasets.

    For every item, each canonical input/output dataset name must appear in
    that item's response (full ``lib.member`` name, or the bare member name
    as a standalone identifier). Scores ``mentioned / expected`` over the
    whole case; skipped when no item carries any dataset.
    """

    name = "dataset_fidelity"
    default_threshold = 0.75

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        expected = 0
        mentioned = 0
        missing: list[str] = []
        for item, response in zip(run.items, run.responses):
            for name in _item_datasets(item):
                expected += 1
                if _mentions(name, response):
                    mentioned += 1
                else:
                    missing.append(name)
        if expected == 0:
            detail = (
                "no item metadata (datasets unknown)"
                if not run.items
                else "no datasets to check"
            )
            return self._result(1.0, detail, skipped=True)
        details = f"{mentioned}/{expected} dataset name(s) covered"
        if missing:
            details += f"; missing: {', '.join(sorted(set(missing)))}"
        return self._result(mentioned / expected, details)


class PythonSyntaxMetric(ValidationMetric):
    """Translated code in the responses must actually parse as the target.

    Despite the name — kept because it is the stable key in config.json
    (``validation.python_syntax_threshold``), in stored verdicts, and in the
    report tables — this checks the run's **target** language, not Python. It
    used to check Python unconditionally, which scored a correct Spark SQL
    run 0.0 ("no fenced Python code blocks") and, with
    ``validation_retries`` on, drove retries that could never pass. The
    ``details`` string names the target and the checker that ran, so a report
    is unambiguous about what was verified.

    Extracts the target's fenced blocks from each response's ``## Translation``
    section and runs the target's checker over each. Scores
    ``valid blocks / blocks``; a case whose responses contain no target code
    at all scores 0 — a translation run that emits only prose is a failure,
    not a skip. Targets with no checker (Spark Scala) skip.
    """

    name = "python_syntax"
    default_threshold = 1.0

    def __init__(
        self,
        threshold: float | None = None,
        *,
        output_language: str | TargetLanguage | None = None,
    ) -> None:
        super().__init__(threshold)
        self.target = _as_target(output_language)

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        blocks: list[str] = []
        for response in run.responses:
            blocks.extend(_target_blocks(response, self.target))
        lang = self.target.display_name
        if not blocks:
            return self._result(0.0, f"no fenced {lang} code blocks in responses")
        if not self.target.checks_syntax:
            return self._result(
                1.0,
                f"{len(blocks)} {lang} block(s) found; no syntax checker for "
                f"{lang}",
                skipped=True,
            )
        ok = 0
        first_error = ""
        for body in blocks:
            error = self.target.check_syntax(body)
            if error is None:
                ok += 1
            elif not first_error:
                first_error = f"; first error: {error}"
        return self._result(
            ok / len(blocks),
            f"{ok}/{len(blocks)} {lang} block(s) parse "
            f"({self.target.checker_name}){first_error}",
        )


class LanguageComplianceMetric(ValidationMetric):
    """The translated code must be in the target language and no other.

    Scores ``target blocks / code blocks`` over each response's
    ``## Translation`` section: a block tagged for a *different* target — a
    ```python``` fence in a Spark SQL run — is the failure this catches, and
    ``details`` names the offending tags so the corrective note the pipeline
    injects before a retry (``SasLLMPipeline._validation_feedback_message``)
    tells the model exactly what to change.

    Prose and source-echo fences (```sas```, ```text```, ...) are not code and
    are ignored rather than counted against the score. A translation section
    with no code block at all scores 0, matching ``python_syntax``: prose is
    not a translation.
    """

    name = "language_compliance"
    default_threshold = 1.0

    def __init__(
        self,
        threshold: float | None = None,
        *,
        output_language: str | TargetLanguage | None = None,
    ) -> None:
        super().__init__(threshold)
        self.target = _as_target(output_language)

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        on_target = 0
        off_target: Counter[str] = Counter()
        for response in run.responses:
            for info, _ in _fenced_blocks(_translation_region(response)):
                if normalize_language(info) in PROSE_FENCE_INFOS:
                    continue
                if self.target.owns_fence(info):
                    on_target += 1
                else:
                    off_target[info] += 1
        total = on_target + sum(off_target.values())
        lang = self.target.display_name
        if not total:
            return self._result(0.0, f"no code blocks to check against {lang}")
        details = f"{on_target}/{total} code block(s) are {lang}"
        if off_target:
            # An untagged fence is never off-target (TargetLanguage.owns_fence
            # claims it), so every tag named here is a real one.
            named = ", ".join(
                f"```{info} x{count}" for info, count in sorted(off_target.items())
            )
            details += f"; off-target: {named}"
        return self._result(on_target / total, details)


class RequiredTermsMetric(ValidationMetric):
    """Caller-declared substrings that must appear somewhere in the output.

    Case-insensitive containment over the concatenated responses; skipped
    when the run declares no ``required_terms``.
    """

    name = "required_terms"
    default_threshold = 1.0

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        terms = run.required_terms
        if not terms:
            return self._result(1.0, "no required terms declared", skipped=True)
        haystack = run.joined_responses.lower()
        found = [t for t in terms if t.lower() in haystack]
        missing = [t for t in terms if t.lower() not in haystack]
        details = f"{len(found)}/{len(terms)} term(s) present"
        if missing:
            details += f"; missing: {', '.join(missing)}"
        return self._result(len(found) / len(terms), details)


class ReferenceSimilarityMetric(ValidationMetric):
    """Token-F1 similarity against the run's golden translation.

    Order-insensitive multiset F1 over identifier-ish tokens — deliberately
    lexical (two correct translations can differ structurally), so treat it
    as a drift alarm against a known-good baseline, not a correctness proof.
    Skipped when the run has no ``reference_translation``.
    """

    name = "reference_similarity"
    default_threshold = 0.5

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        reference = run.reference_translation
        if not reference:
            return self._result(1.0, "no reference translation", skipped=True)
        score = _token_f1(run.joined_responses, reference)
        return self._result(score, f"token F1 vs reference = {score:.3f}")


def default_metrics(
    output_language: str | TargetLanguage | None = None,
) -> list[ValidationMetric]:
    """The deterministic suite the runner uses when none is passed.

    *output_language* is the target the two language-aware metrics score
    against; ``None`` resolves the configured default. Pass the pipeline's
    own :attr:`~pipeline.engine.SasLLMPipeline.target_language` so scoring and
    prompting cannot disagree about what the run was asked to emit.

    The LLM-judged metrics are *not* included — they need a judge model and
    are opted into explicitly (see :func:`judged_metrics`).
    """
    target = _as_target(output_language)
    return [
        ResponseCoverageMetric(),
        DatasetFidelityMetric(),
        LanguageComplianceMetric(output_language=target),
        PythonSyntaxMetric(output_language=target),
        RequiredTermsMetric(),
        ReferenceSimilarityMetric(),
    ]


# Every judged metric, in the order a report reads best: instruction
# following, then answer quality, then the retrieval layer, then the
# whole-run verdicts. `llm_judge` leads as the coarsest, cheapest grade.
JUDGED_METRIC_NAMES = (
    "llm_judge",
    "prompt_alignment",
    "policy_adherence",
    "override_compliance",
    "answer_relevancy",
    "faithfulness",
    "hallucination",
    "contextual_precision",
    "contextual_relevancy",
    "plan_adherence",
    "analysis_summarization",
    "summarization",
    "task_completion",
)


def judged_metrics(
    llm: Any,
    *,
    output_language: str = "PySpark",
    include: Iterable[str] | None = None,
) -> list[ValidationMetric]:
    """The LLM-judged suite, built against one judge model.

    Opt-in and never part of :func:`default_metrics`: these metrics cost
    roughly **17 judge calls per item plus 5 per run** with everything
    enabled, against ``llm_judge``'s one. Pass *include* to enable a subset
    (any of :data:`JUDGED_METRIC_NAMES`); ``None`` builds them all.

    Parameters
    ----------
    llm : Any
        The judge — an :class:`llm_client.LLMClient` (structured output plus
        token accounting) or any LangChain-style ``invoke`` object. See
        :class:`~validation.judged.JudgedMetric`.
    output_language : str
        Target language named in the judge prompts, matching the pipeline's
        own ``output_language``.
    include : Iterable[str] | None
        Metric names to build. Unknown names raise ``ValueError`` rather than
        silently scoring less than the caller asked for.

    Imports are local: the judged modules import this one for
    :class:`ValidationMetric`, so a module-level import would be circular.
    """
    from .agentic_metrics import (
        PlanAdherenceMetric,
        PromptAlignmentMetric,
        TaskCompletionMetric,
    )
    from .judge import LLMJudgeMetric
    from .memory_metrics import OverrideComplianceMetric, PolicyAdherenceMetric
    from .rag_metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        HallucinationMetric,
    )
    from .summarization import AnalysisSummarizationMetric, SummarizationMetric

    builders: dict[str, Callable[[], ValidationMetric]] = {
        "llm_judge": lambda: LLMJudgeMetric(
            llm=llm, output_language=output_language
        ),
        "prompt_alignment": lambda: PromptAlignmentMetric(llm),
        "policy_adherence": lambda: PolicyAdherenceMetric(llm),
        "override_compliance": lambda: OverrideComplianceMetric(llm),
        "answer_relevancy": lambda: AnswerRelevancyMetric(llm),
        "faithfulness": lambda: FaithfulnessMetric(llm),
        "hallucination": lambda: HallucinationMetric(llm),
        "contextual_precision": lambda: ContextualPrecisionMetric(llm),
        "contextual_relevancy": lambda: ContextualRelevancyMetric(llm),
        "plan_adherence": lambda: PlanAdherenceMetric(llm),
        "analysis_summarization": lambda: AnalysisSummarizationMetric(llm),
        "summarization": lambda: SummarizationMetric(llm),
        "task_completion": lambda: TaskCompletionMetric(
            llm, output_language=output_language
        ),
    }
    wanted = list(JUDGED_METRIC_NAMES) if include is None else list(include)
    unknown = [name for name in wanted if name not in builders]
    if unknown:
        raise ValueError(
            f"unknown judged metric(s): {', '.join(sorted(unknown))}; "
            f"known: {', '.join(JUDGED_METRIC_NAMES)}"
        )
    # Built in canonical order regardless of how the caller listed them, so a
    # report's metric rows are comparable between runs.
    selected = [name for name in JUDGED_METRIC_NAMES if name in set(wanted)]
    logger.info(f"judged_metrics: [{', '.join(selected)}]")
    return [builders[name]() for name in selected]
