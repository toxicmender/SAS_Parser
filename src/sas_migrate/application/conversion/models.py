"""Versioned contracts for request-driven conversion workflows."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from sas_migrate.application.ports.source_repository import SourceObject
from sas_migrate.application.translation.artifacts import ArtifactLocator
from sas_migrate.core.models import VersionedContract
from sas_migrate.core.targets import ResolvedTarget


class ConversionStatus(StrEnum):
    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    FAILED = "Failed"


class ConversionRequest(VersionedContract):
    request_id: str = Field(min_length=1)
    application_name: str = Field(min_length=1)
    input_language: str = "SAS"
    output_language: str | None = None
    macro_file_name: str | None = None
    validation_required: bool = False
    status: str = ""

    @property
    def pending(self) -> bool:
        return self.status.strip().casefold() != ConversionStatus.COMPLETED.casefold()


class ConversionModelPreference(VersionedContract):
    request_id: str | None = None
    script_name: str = ""
    model: str | None = None
    status: str = ""


class ConversionTranslationCommand(VersionedContract):
    request: ConversionRequest
    target: ResolvedTarget
    model: str = Field(min_length=1)
    sources: tuple[SourceObject, ...] = Field(min_length=1)
    dry_run: bool = False


class ConversionTranslationResult(VersionedContract):
    ok: bool
    artifacts: tuple[ArtifactLocator, ...] = Field(default_factory=tuple)
    error: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> ConversionTranslationResult:
        if self.ok and self.error is not None:
            raise ValueError("successful conversion cannot contain an error")
        if not self.ok and not self.error:
            raise ValueError("failed conversion requires an error")
        return self


class ConversionOutcome(VersionedContract):
    request_id: str
    application_name: str
    status: ConversionStatus
    model: str
    target: ResolvedTarget | None = None
    artifacts: tuple[ArtifactLocator, ...] = Field(default_factory=tuple)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is ConversionStatus.COMPLETED


class ConversionBatchOutcome(VersionedContract):
    outcomes: tuple[ConversionOutcome, ...] = Field(default_factory=tuple)

    @property
    def failed_count(self) -> int:
        return sum(not outcome.ok for outcome in self.outcomes)

    @property
    def exit_code(self) -> int:
        return int(self.failed_count > 0)


__all__ = [
    "ConversionBatchOutcome",
    "ConversionModelPreference",
    "ConversionOutcome",
    "ConversionRequest",
    "ConversionStatus",
    "ConversionTranslationCommand",
    "ConversionTranslationResult",
]
