"""Reading a file from Azure Data Lake Storage Gen2.

The sibling of :mod:`data_hydration.sources.blob`, and the difference is only the
endpoint: ADLS Gen2 speaks ``dfs.core.windows.net``, which understands the
hierarchical namespace that makes ``/raw/2026/08/sales.csv`` a real directory
tree rather than a blob whose name happens to contain slashes. A migration
reading a lake path wants the Gen2 client for that reason.

Everything else is shared — the ranged-read adapter
(:func:`data_hydration.rawio.AdlsRawIO`), the Entra ID credential, and the
DataFrame conversion.

Logger name: ``data_hydration.sources.adls``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from ..rawio import AdlsRawIO, open_buffered
from .base import SourceInfo, frames_from_file

if TYPE_CHECKING:
    from ..config import HydrationConfig
    from ..models import HydrationItem

logger = logging.getLogger(__name__)


class AdlsReader:
    """One ADLS Gen2 file, streamed as DataFrames."""

    def __init__(self, item: "HydrationItem", config: "HydrationConfig") -> None:
        self._item = item
        self._config = config
        self._handle: Any = None
        source = item.source
        options = source.option_map
        self._account = options.get("account") or config.adls_account or ""
        self._filesystem = (
            options.get("filesystem") or config.adls_filesystem or source.locator
        )
        self._path = source.object_name

    def _open(self) -> Any:
        if self._handle is None:
            if not self._account:
                raise ValueError(
                    "no ADLS account configured: set data_hydration.adls_account"
                )
            self._handle = open_buffered(
                AdlsRawIO(self._account, self._filesystem, self._path)
            )
        return self._handle

    def info(self) -> SourceInfo:
        handle = self._open()
        return SourceInfo(size_bytes=handle.raw.size)  # type: ignore[attr-defined]

    def batches(self) -> Iterator[Any]:
        logger.info(f"AdlsReader: {self._account}/{self._filesystem}/{self._path}")
        yield from frames_from_file(
            self._open(), self._path, self._config.fetch_size
        )

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
