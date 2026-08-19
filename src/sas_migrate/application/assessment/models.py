"""Stable output contracts for migration assessment."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, computed_field

from sas_migrate.core.models import ContractModel, VersionedContract
from sas_migrate.core.targets import TargetId


class ComplexityTier(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TranslationParity(StrEnum):
    DIRECT = "DIRECT"
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    HARD = "HARD"
    MANUAL = "MANUAL"


class TShirtSize(StrEnum):
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"
    EXTRA_LARGE = "EXTRA_LARGE"


class ConstructRule(ContractModel):
    category: str
    tier: ComplexityTier
    parity: TranslationParity
    note: str = ""


class AssessmentProfile(VersionedContract):
    name: str
    target: str
    display_name: str
    description: str = ""
    extends: str | None = None
    weights: dict[ComplexityTier, float]
    sizes: dict[str, object]
    constructs: dict[str, dict[str, ConstructRule]]
    flags: tuple[dict[str, object], ...] = ()


class ConstructOccurrence(ContractModel):
    kind: str
    name: str
    count: int = Field(default=1, ge=1)


class AssessmentUnit(ContractModel):
    source_id: str
    line_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    step_count: int = Field(default=0, ge=0)
    parameter_count: int = Field(default=0, ge=0)
    input_datasets: tuple[str, ...] = ()
    output_datasets: tuple[str, ...] = ()
    constructs: tuple[ConstructOccurrence, ...] = ()
    unresolved_references: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()


class DependencyEdge(ContractModel):
    producer: str
    consumer: str
    dataset: str


class ComplexitySignal(ContractModel):
    kind: str
    name: str
    count: int
    category: str
    tier: ComplexityTier
    parity: TranslationParity
    weighted_score: float = Field(ge=0)
    note: str = ""


class FileAssessment(ContractModel):
    source_id: str
    raw_score: float = Field(ge=0)
    story_points: float = Field(gt=0)
    size: TShirtSize
    tier: ComplexityTier
    parity: TranslationParity
    signals: tuple[ComplexitySignal, ...]
    dependencies: tuple[DependencyEdge, ...] = ()
    unresolved_references: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()


class AssessmentReport(VersionedContract):
    target: TargetId
    profile: str
    files: tuple[FileAssessment, ...]
    dependencies: tuple[DependencyEdge, ...] = ()
    review: str | None = None

    @computed_field
    @property
    def total_story_points(self) -> float:
        return sum(file.story_points for file in self.files)

    def to_json(self) -> str:
        return self.model_dump_json(exclude_computed_fields=True)


__all__ = [
    "AssessmentProfile",
    "AssessmentReport",
    "AssessmentUnit",
    "ComplexitySignal",
    "ComplexityTier",
    "ConstructOccurrence",
    "ConstructRule",
    "DependencyEdge",
    "FileAssessment",
    "TShirtSize",
    "TranslationParity",
]
