"""The surface every source presents, so the runner never branches on kind.

A reader answers two questions: what is there (:meth:`Reader.info`) and give me
the rows (:meth:`Reader.batches`). Batches are pandas DataFrames — the one
container ``oracledb``, ``pyreadstat`` and ``pyarrow`` all produce or accept
cheaply, and the one :meth:`pyspark.sql.SparkSession.createDataFrame` takes
directly. Streaming them rather than materialising one frame is what keeps a
40 GB table from being a 40 GB allocation.

:func:`reader_for` is the only entry point the sink uses. Drivers are imported
inside it, per source kind, so a run that touches Oracle never needs the sFTP
extra installed — and ``import data_hydration`` needs none of them.

Logger name: ``data_hydration.sources.base``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..config import HydrationConfig
    from ..models import HydrationItem

logger = logging.getLogger(__name__)


class UnsupportedSource(RuntimeError):
    """This source kind has no reader, or the one it needs is unavailable.

    Raised rather than returning ``None`` so the runner records it as a failed
    item with a message naming what is missing — usually an extra to install.
    """


@dataclass(frozen=True)
class SourceInfo:
    """What a probe could learn about a source without reading all of it.

    Every field is optional because every source answers a different subset:
    a ``sas7bdat`` header gives rows and columns for free, an sFTP file gives
    only a byte count, and an Oracle table gives rows only if somebody pays for
    a ``COUNT(*)``.

    Attributes
    ----------
    rows
        Row count, when cheaply knowable.
    columns
        Column names in order, when the source declares a schema.
    size_bytes
        Bytes on disk or in storage.
    index_columns
        Columns an index covers — the clustering hint. See
        :mod:`data_hydration.sources.sas_files` for why this is best-effort.
    """

    rows: int | None = None
    columns: tuple[str, ...] = ()
    size_bytes: int | None = None
    index_columns: tuple[str, ...] = field(default=())

    def __str__(self) -> str:
        parts = []
        if self.rows is not None:
            parts.append(f"{self.rows} rows")
        if self.columns:
            parts.append(f"{len(self.columns)} columns")
        if self.size_bytes is not None:
            parts.append(f"{self.size_bytes} bytes")
        return ", ".join(parts) or "unknown"


class Reader(Protocol):
    """What :mod:`data_hydration.sinks.delta` needs from any source."""

    def info(self) -> SourceInfo:
        """Schema and size, without reading the rows."""
        ...

    def batches(self) -> Iterator[Any]:
        """The rows, as a stream of pandas DataFrames."""
        ...

    def close(self) -> None:
        """Release the connection or file handle."""
        ...


def reader_for(item: "HydrationItem", config: "HydrationConfig") -> Reader:
    """The :class:`Reader` for one item, with its driver imported on demand.

    Raises
    ------
    UnsupportedSource
        No reader exists for this kind, or its driver is not installed. The
        message names the extra to install, because that is the fix.
    """
    from ..models import SourceKind

    kind = item.source.kind
    if kind is SourceKind.ORACLE:
        from .oracle import OracleReader

        return OracleReader(item, config)
    if kind is SourceKind.SAS_DATASET:
        from .sas_files import SasDatasetReader

        return SasDatasetReader(item, config)
    if kind is SourceKind.SPDE:
        from .sas_session import SpdeReader

        return SpdeReader(item, config)
    if kind is SourceKind.SFTP:
        from .sftp import SftpReader

        return SftpReader(item, config)
    if kind is SourceKind.BLOB:
        from .blob import BlobReader

        return BlobReader(item, config)
    if kind is SourceKind.ADLS:
        from .adls import AdlsReader

        return AdlsReader(item, config)
    if kind is SourceKind.FILE:
        from .sas_files import LocalFileReader

        return LocalFileReader(item, config)
    raise UnsupportedSource(f"no reader for source kind '{kind}'")


def frames_from_file(handle: Any, name: str, chunk_rows: int) -> Iterator[Any]:
    """A delimited or columnar file object as a stream of DataFrames.

    The shared tail of every *file* source: sFTP, Blob and ADLS all differ in
    how they open a handle and not at all in what to do with it. Format is taken
    from the name's suffix, which is the only signal a raw byte stream carries.

    ``pandas`` arrives with ``pyreadstat``; a deployment with neither installed
    is not reading files, so the import is not guarded separately.
    """
    import pandas as pd

    lowered = name.lower()
    if lowered.endswith(".parquet"):
        # Parquet is columnar: it is read whole rather than in row chunks,
        # because a row-group scan needs the footer either way.
        yield pd.read_parquet(handle)
        return
    if lowered.endswith((".xlsx", ".xls")):
        yield pd.read_excel(handle)
        return
    separator = "\t" if lowered.endswith((".tsv", ".tab")) else ","
    for chunk in pd.read_csv(handle, sep=separator, chunksize=chunk_rows):
        yield chunk
