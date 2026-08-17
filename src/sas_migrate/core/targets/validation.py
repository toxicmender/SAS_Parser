"""Target-validation result contracts shared by response and reporting layers."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from ..models import ContractModel, VersionedContract
from .models import TargetId


class TargetIssueCode(StrEnum):
    TARGET_MISMATCH = "target_mismatch"
    FOREIGN_LANGUAGE = "foreign_language"
    SYNTAX_ERROR = "syntax_error"
    EMPTY_CODE = "empty_code"
    UNKNOWN_CHUNK = "unknown_chunk"
    MIXED_TARGETS = "mixed_targets"
    ROUND_TRIP_MISMATCH = "round_trip_mismatch"


class TargetValidationIssue(ContractModel):
    code: TargetIssueCode
    message: str
    cell_index: int | None = Field(default=None, ge=0)
    chunk_id: str | None = None


class ResponseValidationResult(VersionedContract):
    valid: bool
    resolved_target: TargetId
    reported_target: TargetId | None = None
    issues: tuple[TargetValidationIssue, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_outcome(self) -> ResponseValidationResult:
        if self.valid:
            if self.issues:
                raise ValueError("a valid target result cannot contain issues")
            if self.reported_target != self.resolved_target:
                raise ValueError("a valid response must report the resolved target")
        elif not self.issues:
            raise ValueError("an invalid target result must explain at least one issue")
        return self

    @classmethod
    def accepted(cls, target: TargetId) -> ResponseValidationResult:
        return cls(valid=True, resolved_target=target, reported_target=target)
