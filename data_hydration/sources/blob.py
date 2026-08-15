"""Reading a file from Azure Blob storage.

The blob is presented as an ordinary buffered file object by
:func:`data_hydration.rawio.BlobRawIO`, so everything after ``open`` is the same
code every other file source runs — see
:func:`data_hydration.sources.base.frames_from_file`.

Authentication is an Entra ID token, not a storage key: the identity is the one
:mod:`app_config.azure` already logs in as, reached through
:class:`data_hydration.secrets.EntraCredential`. A deployment that has granted
its service principal ``Storage Blob Data Reader`` needs nothing else configured.

Logger name: ``data_hydration.sources.blob``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from ..rawio import BlobRawIO, open_buffered
from .base import SourceInfo, frames_from_file

if TYPE_CHECKING:
    from ..config import HydrationConfig
    from ..models import HydrationItem

logger = logging.getLogger(__name__)


class BlobReader:
    """One blob, streamed as DataFrames."""

    def __init__(self, item: "HydrationItem", config: "HydrationConfig") -> None:
        self._item = item
        self._config = config
        self._handle: Any = None
        source = item.source
        options = source.option_map
        self._account = options.get("account") or config.blob_account or ""
        self._container = (
            options.get("container") or config.blob_container or source.locator
        )
        self._name = source.object_name

    def _open(self) -> Any:
        if self._handle is None:
            if not self._account:
                raise ValueError(
                    "no Azure storage account configured: set "
                    "data_hydration.blob_account"
                )
            self._handle = open_buffered(
                BlobRawIO(self._account, self._container, self._name)
            )
        return self._handle

    def info(self) -> SourceInfo:
        handle = self._open()
        return SourceInfo(size_bytes=handle.raw.size)  # type: ignore[attr-defined]

    def batches(self) -> Iterator[Any]:
        logger.info(
            f"BlobReader: {self._account}/{self._container}/{self._name}"
        )
        yield from frames_from_file(
            self._open(), self._name, self._config.fetch_size
        )

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
