"""Guarded Delta maintenance commands for the memory store.

Maintenance is intentionally separate from the hot-path backend.  A pipeline
must never compact or vacuum its own state during a request; an operator or
scheduled job owns these explicit, observable actions instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .store import _quoted_table_name

MIN_VACUUM_HOURS = 7 * 24
"""Delta's safe default: one week."""

MAX_VACUUM_HOURS = 4 * 30 * 24
"""Application policy ceiling: four 30-day months."""


@dataclass(frozen=True)
class VacuumPolicy:
    """Retention guardrail for a memory Delta table and its CDF consumers."""

    retention_hours: int
    max_cdf_outage_hours: int = 0

    def __post_init__(self) -> None:
        if not MIN_VACUUM_HOURS <= self.retention_hours <= MAX_VACUUM_HOURS:
            raise ValueError(
                "retention_hours must be between "
                f"{MIN_VACUUM_HOURS} and {MAX_VACUUM_HOURS}"
            )
        if self.max_cdf_outage_hours < 0:
            raise ValueError("max_cdf_outage_hours must be >= 0")
        if self.retention_hours <= self.max_cdf_outage_hours:
            raise ValueError(
                "retention_hours must exceed max_cdf_outage_hours so every "
                "CDF consumer can resume before VACUUM removes its changes"
            )


class DeltaMemoryMaintenance:
    """Explicitly inspect, compact, or vacuum one validated Delta table."""

    def __init__(self, spark: Any, table: str, *, policy: VacuumPolicy) -> None:
        self._spark = spark
        self.table_name = table
        self._table = _quoted_table_name(table)
        self.policy = policy

    def status(self) -> dict[str, Any]:
        """Return the current Delta history/details needed by an operator."""
        latest = self._spark.sql(f"DESCRIBE HISTORY {self._table} LIMIT 1").first()
        detail = self._spark.sql(f"DESCRIBE DETAIL {self._table}").first()
        return {
            "table": self.table_name,
            "latest_version": int(latest.version) if latest is not None else None,
            "last_operation": getattr(latest, "operation", None),
            "size_in_bytes": getattr(detail, "sizeInBytes", None),
            "num_files": getattr(detail, "numFiles", None),
            "vacuum_retention_hours": self.policy.retention_hours,
        }

    def optimize(self) -> None:
        """Run Delta compaction as a scheduled maintenance action."""
        self._spark.sql(f"OPTIMIZE {self._table}")

    def vacuum(self, *, dry_run: bool = True) -> list[Any]:
        """Vacuum within policy; dry-run by default for safe operations."""
        command = (
            f"VACUUM {self._table} RETAIN {self.policy.retention_hours} HOURS"
            + (" DRY RUN" if dry_run else "")
        )
        return self._spark.sql(command).collect()
