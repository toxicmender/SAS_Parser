"""Deterministic validation metrics with legacy-compatible names and skips."""

from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable

import sqlglot

from sas_migrate.core.targets import TargetId

from .models import EvaluationRun, MetricResult

_TOKEN_RE = re.compile(r"[a-z0-9_.]+")
_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


def _token_f1(candidate: str, reference: str) -> float:
    left, right = Counter(_tokens(candidate)), Counter(_tokens(reference))
    overlap = sum((left & right).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(left.values())
    recall = overlap / sum(right.values())
    return 2 * precision * recall / (precision + recall)


def _mentions(value: str, text: str) -> bool:
    lowered = text.casefold()
    name = value.casefold()
    if name in lowered:
        return True
    bare = name.rsplit(".", 1)[-1]
    return re.search(rf"(?<![\w.]){re.escape(bare)}(?![\w.])", lowered) is not None


def _code_blocks(response: str) -> list[tuple[str, str]]:
    return [(info.strip().casefold(), body) for info, body in _FENCE_RE.findall(response)]


class ValidationMetric(ABC):
    name = "metric"
    default_threshold = 0.7

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = self.default_threshold if threshold is None else threshold
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("metric threshold must be between 0 and 1")

    @abstractmethod
    def evaluate(self, run: EvaluationRun) -> MetricResult: ...

    def result(
        self,
        score: float,
        details: str = "",
        *,
        skipped: bool = False,
    ) -> MetricResult:
        bounded = min(1.0, max(0.0, score))
        return MetricResult(
            metric=self.name,
            score=bounded,
            threshold=self.threshold,
            passed=skipped or bounded >= self.threshold,
            skipped=skipped,
            details=details,
        )


class ResponseCoverageMetric(ValidationMetric):
    name = "response_coverage"
    default_threshold = 1.0

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        expected = run.unit_count
        if expected == 0:
            return self.result(1.0, "no expected response units", skipped=True)
        produced = sum(bool(unit.response.strip()) for unit in run.units)
        return self.result(produced / expected, f"{produced}/{expected} responses present")


class DatasetFidelityMetric(ValidationMetric):
    name = "dataset_fidelity"
    default_threshold = 0.75

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        expected = sorted(
            {
                dataset
                for unit in run.units
                for dataset in (*unit.input_datasets, *unit.output_datasets)
            }
        )
        if not expected:
            return self.result(1.0, "no dataset metadata", skipped=True)
        found = sum(_mentions(dataset, run.joined_responses) for dataset in expected)
        return self.result(found / len(expected), f"{found}/{len(expected)} datasets mentioned")


class LanguageComplianceMetric(ValidationMetric):
    name = "language_compliance"
    default_threshold = 1.0

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        aliases = (
            {"python", "py", "pyspark"}
            if run.target is TargetId.PYSPARK
            else {"sql", "sparksql", "spark_sql", "databricks"}
        )
        blocks = [block for unit in run.units for block in _code_blocks(unit.response)]
        if not blocks:
            return self.result(0.0, "no fenced target code")
        owned = sum(info in aliases for info, _ in blocks)
        return self.result(owned / len(blocks), f"{owned}/{len(blocks)} code fences match target")


class TargetSyntaxMetric(ValidationMetric):
    name = "target_syntax"
    default_threshold = 1.0

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        aliases = (
            {"python", "py", "pyspark"}
            if run.target is TargetId.PYSPARK
            else {"sql", "sparksql", "spark_sql", "databricks"}
        )
        bodies = [
            body
            for unit in run.units
            for info, body in _code_blocks(unit.response)
            if info in aliases
        ]
        if not bodies:
            return self.result(1.0, "no target code to parse", skipped=True)
        valid = 0
        for body in bodies:
            try:
                if run.target is TargetId.PYSPARK:
                    ast.parse(body)
                else:
                    sqlglot.parse(body, read="databricks")
            except (SyntaxError, sqlglot.errors.ParseError):
                continue
            valid += 1
        return self.result(valid / len(bodies), f"{valid}/{len(bodies)} target blocks parse")


class RequiredTermsMetric(ValidationMetric):
    name = "required_terms"
    default_threshold = 1.0

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        if not run.required_terms:
            return self.result(1.0, "no required terms", skipped=True)
        text = run.joined_responses.casefold()
        found = sum(term.casefold() in text for term in run.required_terms)
        return self.result(found / len(run.required_terms), f"{found}/{len(run.required_terms)} required terms present")


class ReferenceSimilarityMetric(ValidationMetric):
    name = "reference_similarity"
    default_threshold = 0.5

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        if not run.reference_translation:
            return self.result(1.0, "no reference translation", skipped=True)
        return self.result(
            _token_f1(run.joined_responses, run.reference_translation),
            "multiset token F1 against reference",
        )


def default_metrics() -> tuple[ValidationMetric, ...]:
    return (
        ResponseCoverageMetric(),
        DatasetFidelityMetric(),
        LanguageComplianceMetric(),
        TargetSyntaxMetric(),
        RequiredTermsMetric(),
        ReferenceSimilarityMetric(),
    )


def metric_names(metrics: Iterable[ValidationMetric]) -> tuple[str, ...]:
    return tuple(metric.name for metric in metrics)


__all__ = [
    "DatasetFidelityMetric",
    "LanguageComplianceMetric",
    "ReferenceSimilarityMetric",
    "RequiredTermsMetric",
    "ResponseCoverageMetric",
    "TargetSyntaxMetric",
    "ValidationMetric",
    "default_metrics",
    "metric_names",
]
