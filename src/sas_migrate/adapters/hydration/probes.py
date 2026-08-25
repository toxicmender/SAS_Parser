"""Filesystem-backed source discovery adapters."""

from __future__ import annotations

from pathlib import Path

from sas_migrate.application.hydration import HydrationSource, SourceKind


class FilesystemSpdeProbe:
    def native_partitions(self, source: HydrationSource) -> tuple[str, ...] | None:
        if source.kind is not SourceKind.SPDE:
            return None
        stem = source.object_name.casefold()
        try:
            return tuple(
                str(path)
                for path in sorted(Path(source.locator).iterdir())
                if path.suffix.casefold() == ".dpf" and path.stem.casefold().startswith(stem)
            ) or None
        except OSError:
            return None

    def row_count(self, source: HydrationSource) -> int | None:
        del source
        return None

    def range_column(self, source: HydrationSource) -> tuple[str, float, float] | None:
        del source
        return None


__all__ = ["FilesystemSpdeProbe"]
