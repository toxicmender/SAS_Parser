"""Lazy hydration drivers and optional-dependency boundaries."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from sas_migrate.application.hydration import HydrationItem, SourceKind
from sas_migrate.application.ports.hydration import HydrationSourceDriver

OPTIONAL_DEPENDENCIES: dict[SourceKind, tuple[str, str]] = {
    SourceKind.ORACLE: ("oracledb", "hydration"),
    SourceKind.SFTP: ("paramiko", "hydration"),
    SourceKind.ADLS: ("azure.storage.filedatalake", "hydration"),
    SourceKind.BLOB: ("azure.storage.blob", "hydration"),
    SourceKind.FILE: ("pandas", "hydration"),
    SourceKind.SAS_DATASET: ("pyreadstat", "hydration"),
    SourceKind.SPDE: ("saspy", "hydration"),
    SourceKind.SAS_SESSION: ("saspy", "hydration"),
}

# Keep this explicit until every source kind has a concrete v2 driver. The
# optional-dependency matrix alone is not evidence that a driver exists.
CONCRETE_DRIVER_KINDS = frozenset({SourceKind.FILE, SourceKind.SAS_DATASET})


class HydrationDriverUnavailable(RuntimeError):
    pass


def require_optional_dependency(kind: SourceKind) -> Any:
    module_name, extra = OPTIONAL_DEPENDENCIES[kind]
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise HydrationDriverUnavailable(
            f"{module_name} is required for {kind.value}; install 'sas-parser[{extra}]'"
        ) from exc


DriverFactory = Callable[[], HydrationSourceDriver]


class LazyHydrationDriverRegistry:
    """Construct only the driver requested by the current plan item."""

    def __init__(self, factories: Mapping[SourceKind, DriverFactory]) -> None:
        self._factories = dict(factories)

    def driver_for(self, kind: SourceKind) -> HydrationSourceDriver:
        try:
            factory = self._factories[kind]
        except KeyError as exc:
            module_name, extra = OPTIONAL_DEPENDENCIES[kind]
            raise HydrationDriverUnavailable(
                f"no {kind.value} hydration driver is configured "
                f"({module_name}; install 'sas-parser[{extra}]')"
            ) from exc
        return factory()


class SasDatasetDriver:
    def __init__(self) -> None:
        self._pyreadstat = require_optional_dependency(SourceKind.SAS_DATASET)

    def batches(self, item: HydrationItem) -> Iterable[Any]:
        name = item.source.source_name or f"{item.source.object_name}.sas7bdat"
        path = Path(item.source.locator) / name
        kwargs: dict[str, int] = {}
        if item.partition and item.partition.row_offset is not None:
            kwargs["row_offset"] = item.partition.row_offset
        if item.partition and item.partition.row_limit is not None:
            kwargs["row_limit"] = item.partition.row_limit
        frame, _metadata = self._pyreadstat.read_sas7bdat(str(path), **kwargs)
        return (frame,)

    def close(self) -> None:
        return None


class LocalFileDriver:
    def __init__(self, *, batch_rows: int = 10_000) -> None:
        self._pandas = require_optional_dependency(SourceKind.FILE)
        self._batch_rows = batch_rows

    def batches(self, item: HydrationItem) -> Iterable[Any]:
        path = Path(item.source.locator) / (
            item.source.source_name or item.source.object_name
        )
        suffix = path.suffix.casefold()
        if suffix == ".parquet":
            return (self._pandas.read_parquet(path),)
        if suffix in {".xlsx", ".xls"}:
            return (self._pandas.read_excel(path),)
        separator = "\t" if suffix in {".tsv", ".tab"} else ","
        return self._pandas.read_csv(path, sep=separator, chunksize=self._batch_rows)

    def close(self) -> None:
        return None


__all__ = [
    "CONCRETE_DRIVER_KINDS",
    "OPTIONAL_DEPENDENCIES",
    "HydrationDriverUnavailable",
    "LazyHydrationDriverRegistry",
    "LocalFileDriver",
    "SasDatasetDriver",
    "require_optional_dependency",
]
