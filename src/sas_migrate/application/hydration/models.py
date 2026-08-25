"""Versioned, secret-free contracts for hydration planning and execution."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import Field, model_validator

from sas_migrate.core.models import VersionedContract


class SourceKind(StrEnum):
    ORACLE = "oracle"
    SFTP = "sftp"
    ADLS = "adls"
    BLOB = "blob"
    FILE = "file"
    SAS_DATASET = "sas7bdat"
    SPDE = "spde"
    SAS_SESSION = "sas_session"


class PartitionStrategy(StrEnum):
    NONE = "none"
    NATIVE = "native"
    ROW_RANGE = "row_range"
    COLUMN_RANGE = "column_range"


class WriteMode(StrEnum):
    OVERWRITE = "overwrite"
    APPEND = "append"


class ItemStatus(StrEnum):
    WRITTEN = "written"
    SKIPPED = "skipped"
    FAILED = "failed"


class HydrationSettings(VersionedContract):
    """Non-secret inputs fixed once for a planning/execution run."""

    catalog: str | None = None
    schema_name: str | None = None
    table_template: str = "<catalog_name>.<schema_name>.<table_name>"
    stage: str | None = None
    date_format: str = "%Y%m%d"
    planned_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    num_partitions: int = Field(default=8, ge=1)
    batch_rows: int = Field(default=10_000, ge=1)
    sas_session_configured: bool = False
    apply_index_clustering: bool = False
    on_error: str = Field(default="continue", pattern="^(continue|stop)$")

    @property
    def run_date(self) -> str:
        try:
            return self.planned_at.strftime(self.date_format)
        except ValueError:
            return self.planned_at.strftime("%Y%m%d")


class HydrationSource(VersionedContract):
    kind: SourceKind
    locator: str = ""
    object_name: str = ""
    source_name: str = ""
    libref: str | None = None
    engine: str | None = None
    options: tuple[tuple[str, str], ...] = ()
    has_macro_ref: bool = False
    source_id: str | None = None

    @property
    def option_map(self) -> dict[str, str]:
        return dict(self.options)


class HydrationPartition(VersionedContract):
    name: str = Field(min_length=1)
    predicate: str | None = None
    row_offset: int | None = Field(default=None, ge=0)
    row_limit: int | None = Field(default=None, ge=1)
    component: str | None = None


class HydrationItem(VersionedContract):
    source: HydrationSource
    target_table: str = Field(min_length=1)
    write_mode: WriteMode = WriteMode.OVERWRITE
    strategy: PartitionStrategy = PartitionStrategy.NONE
    strategy_reason: str = ""
    partition: HydrationPartition | None = None
    notes: tuple[str, ...] = ()
    cluster_by: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    @property
    def needs_operator_input(self) -> bool:
        return bool(self.blockers)


class HydrationPlan(VersionedContract):
    run_date: str = ""
    items: tuple[HydrationItem, ...] = ()

    @property
    def target_tables(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.target_table for item in self.items))

    @property
    def blocked_count(self) -> int:
        return sum(item.needs_operator_input for item in self.items)

    def by_source_id(self, source_id: str) -> tuple[HydrationItem, ...]:
        return tuple(item for item in self.items if item.source.source_id == source_id)

    def counts_by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            kind = item.source.kind.value
            counts[kind] = counts.get(kind, 0) + 1
        return dict(sorted(counts.items()))


class HydrationItemOutcome(VersionedContract):
    item: HydrationItem
    status: ItemStatus
    rows: int | None = Field(default=None, ge=0)
    error: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> HydrationItemOutcome:
        if self.status is ItemStatus.WRITTEN and self.error:
            raise ValueError("written hydration item cannot contain an error")
        if self.status is ItemStatus.FAILED and not self.error:
            raise ValueError("failed hydration item requires an error")
        return self


class HydrationReport(VersionedContract):
    plan: HydrationPlan
    outcomes: tuple[HydrationItemOutcome, ...] = ()
    dry_run: bool = False

    @property
    def written(self) -> int:
        return sum(item.status is ItemStatus.WRITTEN for item in self.outcomes)

    @property
    def skipped(self) -> int:
        return sum(item.status is ItemStatus.SKIPPED for item in self.outcomes)

    @property
    def failed(self) -> int:
        return sum(item.status is ItemStatus.FAILED for item in self.outcomes)

    @property
    def ok(self) -> bool:
        return self.failed == 0


__all__ = [
    "HydrationItem",
    "HydrationItemOutcome",
    "HydrationPartition",
    "HydrationPlan",
    "HydrationReport",
    "HydrationSettings",
    "HydrationSource",
    "ItemStatus",
    "PartitionStrategy",
    "SourceKind",
    "WriteMode",
]
