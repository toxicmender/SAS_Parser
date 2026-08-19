"""Deterministic memory extraction and cross-thread leakage metrics."""

from __future__ import annotations

import re

from .metrics import ValidationMetric, _token_f1
from .models import EvaluationRun, MetricResult


class MemoryExtractionMetric(ValidationMetric):
    name = "memory_extraction"
    default_threshold = 0.7

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        expected = run.expected_memories
        extracted = run.extracted_memories
        if not expected and not extracted:
            return self.result(1.0, "no declared memory extraction", skipped=True)
        matches: set[tuple[int, int]] = set()
        used_extractions: set[int] = set()
        for left, expectation in enumerate(expected):
            candidates = [
                (right, _token_f1(expectation, value))
                for right, value in enumerate(extracted)
                if right not in used_extractions
            ]
            if candidates:
                right, score = max(candidates, key=lambda item: item[1])
                if score >= 0.5:
                    matches.add((left, right))
                    used_extractions.add(right)
        count = len(matches)
        precision = count / len(extracted) if extracted else 0.0
        recall = count / len(expected) if expected else 0.0
        score = 0.0 if not precision or not recall else 2 * precision * recall / (precision + recall)
        return self.result(score, f"precision {precision:.2f}; recall {recall:.2f}")


class MemoryLeakageMetric(ValidationMetric):
    name = "memory_leakage"
    default_threshold = 1.0

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        if not run.foreign_notes:
            return self.result(1.0, "no foreign notes", skipped=True)
        normalized = re.sub(r"\s+", " ", run.joined_responses.casefold())
        leaked = sum(
            re.sub(r"\s+", " ", note.casefold()) in normalized
            for note in run.foreign_notes
        )
        return self.result(
            1.0 - leaked / len(run.foreign_notes),
            f"{leaked}/{len(run.foreign_notes)} foreign note(s) surfaced",
        )


def memory_metrics() -> tuple[ValidationMetric, ...]:
    return MemoryExtractionMetric(), MemoryLeakageMetric()


__all__ = ["MemoryExtractionMetric", "MemoryLeakageMetric", "memory_metrics"]
