"""Resolved translation-target values passed through the v2 pipeline."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from ..models import ContractModel, VersionedContract


class TargetId(StrEnum):
    """The complete v2 translation target set."""

    PYSPARK = "pyspark"
    SPARK_SQL = "spark_sql"


class TargetSource(StrEnum):
    """Boundary value that determined a resolved target."""

    REQUEST = "request"
    EXPLICIT = "explicit"
    CONFIG = "config"
    DEFAULT = "default"
    COMPATIBILITY_FALLBACK = "compatibility_fallback"


class TargetDefinition(ContractModel):
    """Static registry metadata for a supported target."""

    target: TargetId
    display_name: str
    aliases: frozenset[str]
    canonical_language: str
    fence: str


class ResolvedTarget(VersionedContract):
    """Canonical target identity plus auditable resolution provenance."""

    target: TargetId
    display_name: str
    canonical_language: str
    fence: str
    source: TargetSource
    requested_value: str | None = None
    fallback_from: TargetId | None = None
    reasons: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_fallback(self) -> ResolvedTarget:
        if self.source is TargetSource.COMPATIBILITY_FALLBACK:
            if self.fallback_from is not TargetId.SPARK_SQL:
                raise ValueError("compatibility fallback must originate from Spark SQL")
            if self.target is not TargetId.PYSPARK:
                raise ValueError("compatibility fallback may only select PySpark")
        elif self.fallback_from is not None:
            raise ValueError("fallback_from requires compatibility_fallback provenance")
        return self
