"""Ports isolating hydration planning and execution from optional runtimes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from sas_migrate.application.hydration.models import (
        HydrationItem,
        HydrationSource,
        SourceKind,
    )


class HydrationSourceProbe(Protocol):
    def native_partitions(self, source: HydrationSource) -> tuple[str, ...] | None: ...

    def row_count(self, source: HydrationSource) -> int | None: ...

    def range_column(self, source: HydrationSource) -> tuple[str, float, float] | None: ...


class HydrationSourceDriver(Protocol):
    def batches(self, item: HydrationItem) -> Iterable[Any]: ...

    def close(self) -> None: ...


class HydrationDriverRegistry(Protocol):
    def driver_for(self, kind: SourceKind) -> HydrationSourceDriver: ...


class HydrationSink(Protocol):
    def write(self, item: HydrationItem, batches: Iterable[Any]) -> int: ...


__all__ = ["HydrationDriverRegistry", "HydrationSink", "HydrationSourceDriver", "HydrationSourceProbe"]
