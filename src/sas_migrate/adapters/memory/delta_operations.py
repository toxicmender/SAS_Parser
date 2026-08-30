"""Guarded, operator-owned maintenance for the v2 Delta memory adapter."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

MIN_VACUUM_HOURS = 7 * 24
MAX_VACUUM_HOURS = 4 * 30 * 24

_IDENTIFIER_PART = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def quoted_table_name(table: str) -> str:
    """Validate and quote a one- to three-part Unity Catalog identifier."""

    if not table:
        raise ValueError("Delta table must be a non-empty identifier")
    parts = table.split(".")
    if not 1 <= len(parts) <= 3 or any(
        _IDENTIFIER_PART.fullmatch(part) is None for part in parts
    ):
        raise ValueError(
            "Delta table must be a one- to three-part identifier containing "
            "only letters, numbers, and underscores"
        )
    return ".".join(f"`{part}`" for part in parts)


@dataclass(frozen=True)
class VacuumPolicy:
    """Retention guardrails that keep CDF consumers able to resume."""

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
    """Inspect, compact, or vacuum memory outside the request hot path."""

    def __init__(self, spark: Any, table: str, *, policy: VacuumPolicy) -> None:
        self._spark = spark
        self.table_name = table
        self._table = quoted_table_name(table)
        self.policy = policy

    def status(self) -> dict[str, Any]:
        latest = self._spark.sql(
            f"DESCRIBE HISTORY {self._table} LIMIT 1"
        ).first()
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
        self._spark.sql(f"OPTIMIZE {self._table}")

    def vacuum(self, *, dry_run: bool = True) -> list[Any]:
        command = (
            f"VACUUM {self._table} RETAIN {self.policy.retention_hours} HOURS"
            + (" DRY RUN" if dry_run else "")
        )
        return self._spark.sql(command).collect()


__all__ = [
    "MAX_VACUUM_HOURS",
    "MIN_VACUUM_HOURS",
    "DeltaMemoryMaintenance",
    "VacuumPolicy",
    "quoted_table_name",
]
