"""SAS data on disk: ``sas7bdat``, ``sas7bndx``, and plain files.

Three formats and only one of them is data. Being precise about which is which
is the whole job of this module, because the failure mode of getting it wrong is
silent: an index file read as a table produces rows, and they are garbage.

``.sas7bdat`` — a dataset. Read in full, no SAS needed.
    ``pyreadstat`` rather than ``pandas.read_sas``: ``metadataonly=True`` answers
    :meth:`SasDatasetReader.info` from the header alone — row count, column
    names, labels and formats — without touching a row, and ``row_offset`` /
    ``row_limit`` implement the planner's row-range partitions directly. The
    pandas reader can do neither.

``.sas7bndx`` — an index. Detected, never read for rows.
    It contains B-tree pages pointing into the dataset, not data, and its layout
    is undocumented — what is known about it comes from reverse engineering.
    :func:`index_columns` therefore promises very little: the *presence* of the
    file is reliable, the column names inside it are not. What matters for a
    migration is the hint, since a SAS index and Delta clustering answer the same
    question — which columns are queried by — so an unparsed index still tells
    the operator something useful. It is applied only when
    ``data_hydration.apply_index_clustering`` is on.

``SPD Engine`` — see :mod:`data_hydration.sources.sas_session`. The component
    files have no open-source reader, so they go through SAS.

Anything else with a path — CSV, Parquet, Excel — is a
:class:`LocalFileReader`, which is the same code every remote file source ends
in.

Logger name: ``data_hydration.sources.sas_files``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import SourceInfo, frames_from_file

if TYPE_CHECKING:
    from ..config import HydrationConfig
    from ..models import HydrationItem

logger = logging.getLogger(__name__)

INDEX_SUFFIX = ".sas7bndx"
DATASET_SUFFIX = ".sas7bdat"


def _pyreadstat() -> Any:
    try:
        import pyreadstat
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(
            "pyreadstat is required to read .sas7bdat files; install it with "
            "'pip install \"sas-parser[sasdata]\"'"
        ) from exc
    return pyreadstat


def index_path_for(dataset: str | Path) -> Path | None:
    """The ``.sas7bndx`` sitting beside *dataset*, if there is one.

    Convention only — SAS names an index file after the dataset it indexes —
    but the convention is reliable in a way the file's *contents* are not.
    """
    path = Path(dataset).with_suffix(INDEX_SUFFIX)
    return path if path.exists() else None


def index_columns(dataset: str | Path) -> tuple[str, ...]:
    """Best-effort column names from the index beside *dataset*.

    Returns an empty tuple whenever the names cannot be read — which is most of
    the time, and is not an error. The caller should treat a non-empty result as
    a hint and the *existence* of :func:`index_path_for` as the reliable signal.

    The header is scanned for the ASCII name records SAS writes near the start
    of the file. This is reverse-engineered and deliberately conservative: it
    only accepts plausible SAS names, so a mis-parse yields nothing rather than
    nonsense a migration would then cluster a table on.
    """
    path = index_path_for(dataset)
    if path is None:
        return ()
    try:
        header = path.read_bytes()[:4096]
    except OSError as exc:
        logger.debug(f"index_columns: cannot read '{path}': {exc}")
        return ()

    import re

    # SAS names: letter or underscore, then alphanumerics/underscores, at most
    # 32 characters. Anything shorter than three characters is far too likely to
    # be a coincidence in binary data to be worth reporting.
    candidates = re.findall(rb"[A-Za-z_][A-Za-z0-9_]{2,31}", header)
    stem = Path(dataset).stem.lower().encode()
    names = []
    for candidate in candidates:
        lowered = candidate.lower()
        if lowered in {stem, b"sas", b"index"} or lowered.startswith(b"sas"):
            continue
        name = candidate.decode("ascii", "ignore").lower()
        if name not in names:
            names.append(name)
    if not names:
        logger.debug(f"index_columns: '{path}' present but no names recovered")
    return tuple(names[:8])


class SasDatasetReader:
    """One ``.sas7bdat``, whole or one row range of it."""

    def __init__(self, item: "HydrationItem", config: "HydrationConfig") -> None:
        self._item = item
        self._config = config
        source = item.source
        self._path = str(Path(source.locator) / f"{source.object_name}{DATASET_SUFFIX}")

    def info(self) -> SourceInfo:
        """Schema and row count from the header, with no rows read."""
        pyreadstat = _pyreadstat()
        _, meta = pyreadstat.read_sas7bdat(self._path, metadataonly=True)
        return SourceInfo(
            rows=meta.number_rows,
            columns=tuple(meta.column_names or ()),
            size_bytes=_size_of(self._path),
            index_columns=index_columns(self._path),
        )

    def batches(self) -> Iterator[Any]:
        """The item's rows as one DataFrame, or its row-range slice.

        ``pyreadstat`` returns a whole frame rather than a stream, so the
        planner's row ranges are what bound memory here — an unpartitioned read
        of a very large dataset is one allocation, which is why
        :class:`~data_hydration.models.PartitionStrategy` ``ROW_RANGE`` exists.
        """
        pyreadstat = _pyreadstat()
        partition = self._item.partition
        kwargs: dict[str, Any] = {}
        if partition is not None and partition.row_offset is not None:
            kwargs["row_offset"] = partition.row_offset
            kwargs["row_limit"] = partition.row_limit
        logger.info(f"SasDatasetReader: {self._path} {kwargs or '(whole)'}")
        frame, _ = pyreadstat.read_sas7bdat(self._path, **kwargs)
        yield frame

    def close(self) -> None:
        """Nothing to release — ``pyreadstat`` closes the file itself."""


class LocalFileReader:
    """A file on a mounted filesystem: CSV, TSV, Parquet, Excel.

    The terminal case for every *path* source that is not a SAS dataset, and the
    same body the remote readers reach after they have opened a handle.
    """

    def __init__(self, item: "HydrationItem", config: "HydrationConfig") -> None:
        self._item = item
        self._config = config
        source = item.source
        self._path = str(Path(source.locator) / source.object_name)
        self._handle: Any = None

    def info(self) -> SourceInfo:
        return SourceInfo(size_bytes=_size_of(self._path))

    def batches(self) -> Iterator[Any]:
        logger.info(f"LocalFileReader: {self._path}")
        self._handle = open(self._path, "rb")
        yield from frames_from_file(
            self._handle, self._path, self._config.fetch_size
        )

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _size_of(path: str) -> int | None:
    """File size, or ``None`` when it cannot be stat'ed."""
    try:
        return Path(path).stat().st_size
    except OSError:
        return None
