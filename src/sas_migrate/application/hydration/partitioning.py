"""Pure partition selection; live discovery is available only through a probe port."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sas_migrate.application.ports.hydration import HydrationSourceProbe

from .models import HydrationPartition, HydrationSource, PartitionStrategy, SourceKind


@dataclass(frozen=True)
class PartitionPlan:
    strategy: PartitionStrategy
    partitions: tuple[HydrationPartition, ...]
    reason: str


def _whole(reason: str) -> PartitionPlan:
    return PartitionPlan(PartitionStrategy.NONE, (), reason)


def _safe(call: Callable[[], Any]) -> Any:
    try:
        return call()
    except Exception:  # noqa: BLE001 - a failed probe only removes an optimisation
        return None


def _row_ranges(rows: int, slices: int) -> tuple[HydrationPartition, ...]:
    if rows <= 0 or slices <= 1:
        return ()
    size = max(1, -(-rows // slices))
    return tuple(
        HydrationPartition(
            name=f"rows_{offset}_{offset + min(size, rows - offset)}",
            row_offset=offset,
            row_limit=min(size, rows - offset),
        )
        for offset in range(0, rows, size)
    )


def _column_ranges(
    column: str, low: float, high: float, slices: int
) -> tuple[HydrationPartition, ...]:
    if slices <= 1 or high <= low:
        return ()
    width = (high - low) / slices
    parts = []
    for index in range(slices):
        lo = low + width * index
        last = index == slices - 1
        bound = high if last else low + width * (index + 1)
        operator = "<=" if last else "<"
        parts.append(
            HydrationPartition(
                name=f"{column}_{index}",
                predicate=f"{column} >= {lo} AND {column} {operator} {bound}",
            )
        )
    return tuple(parts)


def plan_partitions(
    source: HydrationSource,
    *,
    num_partitions: int = 8,
    probe: HydrationSourceProbe | None = None,
) -> PartitionPlan:
    if probe is None:
        return _whole("planned without a live source probe; reading the source whole")

    native = _safe(lambda: probe.native_partitions(source))
    if source.kind is SourceKind.SPDE:
        if native:
            return _whole(
                f"SPD Engine stores this dataset across {len(native)} component(s), "
                "which must be read together through SAS"
            )
        return _whole("SPD Engine component discovery was unavailable; reading the dataset whole")

    if source.kind is SourceKind.ORACLE:
        if native:
            return PartitionPlan(
                PartitionStrategy.NATIVE,
                tuple(
                    HydrationPartition(name=name, predicate=f"PARTITION ({name})")
                    for name in native
                ),
                f"table declares {len(native)} native partition(s)",
            )
        ranged = _safe(lambda: probe.range_column(source))
        if ranged:
            column, low, high = ranged
            partitions = _column_ranges(column, low, high, num_partitions)
            if partitions:
                return PartitionPlan(
                    PartitionStrategy.COLUMN_RANGE,
                    partitions,
                    f"split indexed column '{column}' across {len(partitions)} range(s)",
                )
        return _whole("no native partitions or indexed range column were available")

    if source.kind is SourceKind.SAS_DATASET:
        rows = _safe(lambda: probe.row_count(source))
        if rows:
            partitions = _row_ranges(rows, num_partitions)
            if partitions:
                return PartitionPlan(
                    PartitionStrategy.ROW_RANGE,
                    partitions,
                    f"{rows} rows split into {len(partitions)} row range(s)",
                )
        return _whole("row count unavailable; reading the dataset whole")

    return _whole(f"{source.kind.value} sources are read whole")


__all__ = ["PartitionPlan", "plan_partitions"]
