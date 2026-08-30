"""Composition helpers for the operational v2 hydration command."""

from __future__ import annotations

from sas_migrate.adapters.hydration import (
    DeltaHydrationSink,
    LazyHydrationDriverRegistry,
    LocalFileDriver,
    SasDatasetDriver,
)
from sas_migrate.application.hydration import SourceKind


def hydration_driver_registry(
    *,
    batch_rows: int,
) -> LazyHydrationDriverRegistry:
    """Compose only the concrete source drivers currently owned by v2."""

    return LazyHydrationDriverRegistry(
        {
            SourceKind.FILE: lambda: LocalFileDriver(batch_rows=batch_rows),
            SourceKind.SAS_DATASET: SasDatasetDriver,
        }
    )


def hydration_delta_sink(*, apply_index_clustering: bool) -> DeltaHydrationSink:
    """Compose the managed Delta sink without resolving Spark eagerly."""

    return DeltaHydrationSink(
        apply_index_clustering=apply_index_clustering,
    )


__all__ = ["hydration_delta_sink", "hydration_driver_registry"]
