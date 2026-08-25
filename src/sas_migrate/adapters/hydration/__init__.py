"""Optional-runtime adapters for v2 hydration."""

from .delta import DeltaHydrationSink
from .drivers import (
    OPTIONAL_DEPENDENCIES,
    HydrationDriverUnavailable,
    LazyHydrationDriverRegistry,
    LocalFileDriver,
    SasDatasetDriver,
    require_optional_dependency,
)
from .probes import FilesystemSpdeProbe
from .ranged_io import DEFAULT_BLOCK_SIZE, RangedRawIO, open_buffered

__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "OPTIONAL_DEPENDENCIES",
    "DeltaHydrationSink",
    "FilesystemSpdeProbe",
    "HydrationDriverUnavailable",
    "LazyHydrationDriverRegistry",
    "LocalFileDriver",
    "RangedRawIO",
    "SasDatasetDriver",
    "open_buffered",
    "require_optional_dependency",
]
