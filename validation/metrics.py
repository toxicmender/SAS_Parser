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

import ast
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter

import app_config

from chunker.models import SasBatch, SasChunk

from .models import CaseRun, MetricResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared text helpers
# ---------------------------------------------------------------------------

# Fenced code block: info string (may be empty) + body. Non-greedy body so
# multiple blocks in one response are matched individually.
_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)

# Info strings treated as Python for syntax checking. An untagged fence in a
# PySpark translation is overwhelmingly Python, so "" is included.
_PYTHON_INFO_STRINGS = {"", "python", "python3", "py", "pyspark"}

_TOKEN_RE = re.compile(r"[a-z0-9_.]+")


def _python_blocks(text: str) -> list[str]:
    """Bodies of fenced blocks whose info string marks them as Python."""
    return [
        body
        for info, body in _FENCE_RE.findall(text)
        if info.strip().lower() in _PYTHON_INFO_STRINGS
    ]


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
    Base class: one case in, one :class:`MetricResult` out.

    Subclasses set ``name`` (also the config.json key prefix,
    ``validation.<name>_threshold``) and ``default_threshold``, and implement
    :meth:`evaluate`.
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
    def evaluate(self, run: CaseRun) -> MetricResult: ...

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
    """Every batch/singleton item must have produced a non-empty response.

    Scores ``non-empty responses / items``, so both an empty response and a
    missing output (item/output count mismatch) lower the score.
    """

    name = "response_coverage"
    default_threshold = 1.0

    def evaluate(self, run: CaseRun) -> MetricResult:
        if not run.items:
            return self._result(0.0, "chunker/batcher produced no items")
        answered = sum(
            1
            for response in run.responses[: len(run.items)]
            if response.strip()
        )
        return self._result(
            answered / len(run.items),
            f"{answered}/{len(run.items)} item(s) answered",
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

    def evaluate(self, run: CaseRun) -> MetricResult:
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
            return self._result(1.0, "no datasets to check", skipped=True)
        details = f"{mentioned}/{expected} dataset name(s) covered"
        if missing:
            details += f"; missing: {', '.join(sorted(set(missing)))}"
        return self._result(mentioned / expected, details)


class PythonSyntaxMetric(ValidationMetric):
    """Fenced Python/PySpark code in the responses must actually parse.

    Extracts fenced code blocks whose info string is empty or Python-ish and
    runs ``ast.parse`` on each. Scores ``valid blocks / blocks``; a case
    whose responses contain no code blocks at all scores 0 — a translation
    run that emits only prose is a failure, not a skip.
    """

    name = "python_syntax"
    default_threshold = 1.0

    def evaluate(self, run: CaseRun) -> MetricResult:
        blocks: list[str] = []
        for response in run.responses:
            blocks.extend(_python_blocks(response))
        if not blocks:
            return self._result(0.0, "no fenced Python code blocks in responses")
        ok = 0
        first_error = ""
        for body in blocks:
            try:
                ast.parse(body)
                ok += 1
            except SyntaxError as exc:
                if not first_error:
                    first_error = f"; first error: {exc.msg} (line {exc.lineno})"
        return self._result(
            ok / len(blocks),
            f"{ok}/{len(blocks)} code block(s) parse{first_error}",
        )


class RequiredTermsMetric(ValidationMetric):
    """Case-authored substrings that must appear somewhere in the output.

    Case-insensitive containment over the concatenated responses; skipped
    when the case declares no ``required_terms``.
    """

    name = "required_terms"
    default_threshold = 1.0

    def evaluate(self, run: CaseRun) -> MetricResult:
        terms = run.case.required_terms
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
    """Token-F1 similarity against the case's golden translation.

    Order-insensitive multiset F1 over identifier-ish tokens — deliberately
    lexical (two correct translations can differ structurally), so treat it
    as a drift alarm against a known-good baseline, not a correctness proof.
    Skipped when the case has no ``reference_translation``.
    """

    name = "reference_similarity"
    default_threshold = 0.5

    def evaluate(self, run: CaseRun) -> MetricResult:
        reference = run.case.reference_translation
        if not reference:
            return self._result(1.0, "no reference translation", skipped=True)
        score = _token_f1(run.joined_responses, reference)
        return self._result(score, f"token F1 vs reference = {score:.3f}")


def default_metrics() -> list[ValidationMetric]:
    """The deterministic suite the runner uses when none is passed.

    The LLM-judge metric is *not* included — it needs a judge model and is
    opted into explicitly (see ``validation.judge``).
    """
    return [
        ResponseCoverageMetric(),
        DatasetFidelityMetric(),
        PythonSyntaxMetric(),
        RequiredTermsMetric(),
        ReferenceSimilarityMetric(),
    ]
