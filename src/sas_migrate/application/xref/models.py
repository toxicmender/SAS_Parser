"""Typed XREF mappings and rewrite reports."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, field_validator

from sas_migrate.core.models import ContractModel
from sas_migrate.core.sas import SasBatchResult


class XrefApplyMode(StrEnum):
    PRE = "pre"
    POST = "post"
    BOTH = "both"


class ParseFailureMode(StrEnum):
    WARN = "warn"
    ERROR = "error"


class XrefRow(ContractModel):
    """Source-neutral projection of one XREF row."""

    application_name: str = ""
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    marker: str = ""

    @field_validator("application_name", "source", "target", "marker")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class XrefMappings(ContractModel):
    """Mappings split by exact dataset, library and physical path."""

    exact: dict[str, str] = Field(default_factory=dict)
    by_libref: dict[str, str] = Field(default_factory=dict)
    by_path: dict[str, str] = Field(default_factory=dict)

    @field_validator("exact", "by_libref", "by_path")
    @classmethod
    def normalize_keys(cls, values: dict[str, str]) -> dict[str, str]:
        return {
            source.strip().casefold(): target.strip()
            for source, target in values.items()
            if source.strip() and target.strip()
        }

    @property
    def dataset_mapping(self) -> dict[str, str]:
        return {**self.by_libref, **self.exact}

    def __bool__(self) -> bool:
        return bool(self.exact or self.by_libref or self.by_path)

    def __len__(self) -> int:
        return len(self.exact) + len(self.by_libref) + len(self.by_path)


class PreRewriteReport(ContractModel):
    rewritten: dict[str, str] = Field(default_factory=dict)
    unresolved: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.rewritten)

    def __bool__(self) -> bool:
        return bool(self.rewritten or self.unresolved)


class BothRewriteResult(ContractModel):
    code: str
    result: SasBatchResult | None = None
    pre_applied: bool = False
    post_changed: bool = False
    only_post: tuple[str, ...] = ()


__all__ = [
    "BothRewriteResult",
    "ParseFailureMode",
    "PreRewriteReport",
    "XrefApplyMode",
    "XrefMappings",
    "XrefRow",
]
