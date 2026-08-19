"""Pure cross-file assessment, sizing, and optional review orchestration."""

from __future__ import annotations

import math
from collections.abc import Sequence

from sas_migrate.application.ports.assessment import (
    AssessmentProfileRepository,
    AssessmentReviewer,
)
from sas_migrate.core.targets import TargetId

from .models import (
    AssessmentProfile,
    AssessmentReport,
    AssessmentUnit,
    ComplexitySignal,
    ComplexityTier,
    DependencyEdge,
    FileAssessment,
    TranslationParity,
    TShirtSize,
)
from .profiles import load_profile

_TIER_ORDER = {
    ComplexityTier.LOW: 0,
    ComplexityTier.MEDIUM: 1,
    ComplexityTier.HIGH: 2,
}
_PARITY_ORDER = {
    TranslationParity.DIRECT: 0,
    TranslationParity.SUPPORTED: 1,
    TranslationParity.PARTIAL: 2,
    TranslationParity.HARD: 3,
    TranslationParity.MANUAL: 4,
}


def profile_name(target: TargetId) -> str:
    return "pyspark" if target is TargetId.PYSPARK else "sparksql"


def dependency_edges(units: Sequence[AssessmentUnit]) -> tuple[DependencyEdge, ...]:
    producers: dict[str, list[str]] = {}
    for unit in units:
        for dataset in unit.output_datasets:
            producers.setdefault(dataset.casefold(), []).append(unit.source_id)
    edges = {
        (producer, unit.source_id, dataset.casefold())
        for unit in units
        for dataset in unit.input_datasets
        for producer in producers.get(dataset.casefold(), ())
        if producer != unit.source_id
    }
    return tuple(
        DependencyEdge(producer=producer, consumer=consumer, dataset=dataset)
        for producer, consumer, dataset in sorted(edges)
    )


def _number(mapping: object, key: str, default: float) -> float:
    return float(mapping.get(key, default)) if isinstance(mapping, dict) else default


def _size(raw: float, sizes: dict[str, object]) -> tuple[float, TShirtSize]:
    anchor_block = sizes.get("anchor", {})
    anchor = _number(anchor_block, "raw", 18.0)
    scale = sizes.get("scale", {})
    rungs = {
        TShirtSize.SMALL: _number(scale, "SMALL", 2.0),
        TShirtSize.MEDIUM: _number(scale, "MEDIUM", 3.0),
        TShirtSize.LARGE: _number(scale, "LARGE", 5.0),
        TShirtSize.EXTRA_LARGE: _number(scale, "EXTRA_LARGE", 8.0),
    }
    ratio = max(raw, 0.01) / max(anchor, 0.01)
    continuous = rungs[TShirtSize.MEDIUM] * math.sqrt(ratio)
    selected = min(rungs, key=lambda item: abs(math.log(rungs[item] / continuous)))
    return rungs[selected], selected


def _assess_file(
    unit: AssessmentUnit,
    profile: AssessmentProfile,
    edges: tuple[DependencyEdge, ...],
) -> FileAssessment:
    volume = profile.sizes.get("volume", {})
    raw = (
        unit.chunk_count * _number(volume, "chunk", 0.5)
        + unit.line_count * _number(volume, "line", 0.01)
        + unit.step_count * _number(volume, "step", 1.0)
        + (len(unit.input_datasets) + len(unit.output_datasets)) * _number(volume, "io", 1.0)
        + unit.parameter_count * _number(volume, "param", 0.5)
    )
    signals: list[ComplexitySignal] = []
    for occurrence in unit.constructs:
        rule = profile.constructs.get(occurrence.kind, {}).get(occurrence.name.casefold())
        if rule is None:
            continue
        score = profile.weights[rule.tier] * occurrence.count
        raw += score
        signals.append(
            ComplexitySignal(
                kind=occurrence.kind,
                name=occurrence.name,
                count=occurrence.count,
                category=rule.category,
                tier=rule.tier,
                parity=rule.parity,
                weighted_score=score,
                note=rule.note,
            )
        )
    uncertainty = profile.sizes.get("uncertainty", {})
    raw += len(unit.unresolved_references) * _number(uncertainty, "unresolved_ref", 3.0)
    raw += len(unit.diagnostics) * _number(uncertainty, "diagnostic", 1.5)
    story_points, size = _size(raw, profile.sizes)
    tier = max(
        (signal.tier for signal in signals),
        key=lambda value: _TIER_ORDER[value],
        default=ComplexityTier.LOW,
    )
    parity = max(
        (signal.parity for signal in signals),
        key=lambda value: _PARITY_ORDER[value],
        default=TranslationParity.DIRECT,
    )
    return FileAssessment(
        source_id=unit.source_id,
        raw_score=round(raw, 2),
        story_points=story_points,
        size=size,
        tier=tier,
        parity=parity,
        signals=tuple(signals),
        dependencies=tuple(edge for edge in edges if edge.consumer == unit.source_id),
        unresolved_references=unit.unresolved_references,
        diagnostics=unit.diagnostics,
    )


class AssessmentService:
    def __init__(self, profiles: AssessmentProfileRepository) -> None:
        self._profiles = profiles

    def assess(self, units: Sequence[AssessmentUnit], target: TargetId) -> AssessmentReport:
        name = profile_name(target)
        profile = load_profile(name, self._profiles)
        edges = dependency_edges(units)
        return AssessmentReport(
            target=target,
            profile=name,
            files=tuple(_assess_file(unit, profile, edges) for unit in units),
            dependencies=edges,
        )

    async def review(
        self,
        report: AssessmentReport,
        reviewer: AssessmentReviewer,
    ) -> AssessmentReport:
        lines = [
            f"Review SAS migration assessment for target {report.target.value}.",
            "Return concise risks and sequencing advice grounded only in these facts:",
        ]
        lines.extend(
            f"- {file.source_id}: {file.size.value}, {file.tier.value}, "
            f"{file.parity.value}, {file.story_points:g} points"
            for file in report.files
        )
        response = await reviewer.review("\n".join(lines))
        return report.model_copy(update={"review": response})


__all__ = ["AssessmentService", "dependency_edges", "profile_name"]
