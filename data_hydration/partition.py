"""Deciding how a source is divided, and saying why.

"Import the table entirely, or a partition if one can be identified" is the
requirement, and the interesting half is *identified*: a source is only worth
splitting when the split is one the source itself can serve efficiently. Four
strategies, tried in this order, and the first that applies wins:

1. **Native** — partitions the source declares. Oracle table partitions, and the
   ``.dpf`` component files of an SPD Engine library. Best by a distance: the
   reader touches exactly one partition's storage.
2. **Row range** — an offset/limit pair, for a reader that can seek by row.
   ``sas7bdat`` through ``pyreadstat`` can; most cannot.
3. **Column range** — ``WHERE col >= lo AND col < hi`` over an indexed numeric
   or date column. Needs a probe for the bounds, and needs the column to be
   indexed or the database re-scans the table once per slice.
4. **None** — read it whole. A real answer, not a failure.

Every decision carries a sentence saying why, onto
:attr:`~data_hydration.models.HydrationItem.strategy_reason`, because "this
40 GB table was not partitioned" is the first thing an operator will question and
the plan should answer without being re-run.

**Probing is optional.** With no probe — which is how :mod:`complexity` builds a
plan to print it — only strategies knowable from static information apply: SPD
Engine components (a directory listing) and nothing else. That asymmetry is the
point: a report must not open a database connection.

Logger name: ``data_hydration.partition``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from .models import Partition, PartitionStrategy, SourceKind

logger = logging.getLogger(__name__)

#: SPD Engine data component suffix. The metadata (``.mdf``) and index
#: (``.idx``/``.hbx``) components are not data and are never partitions.
SPDE_DATA_SUFFIX = ".dpf"


class SourceProbe(Protocol):
    """What :func:`plan_partitions` needs to ask a live source.

    A protocol, not a base class, so a test can hand in an object with two
    methods and the real drivers stay out of the import graph. Every method may
    return ``None``, meaning "cannot answer" — never an exception, because a
    probe that fails should downgrade the strategy, not end the run.
    """

    def native_partitions(self, owner: str, table: str) -> list[str] | None:
        """Names of the table's own partitions, or ``None`` if it has none."""

    def row_count(self, owner: str, table: str) -> int | None:
        """Total rows, for splitting into ranges."""

    def range_column(self, owner: str, table: str) -> tuple[str, float, float] | None:
        """An indexed numeric column and its ``(min, max)``, if one exists."""


class PartitionPlan:
    """The chosen strategy, its partitions, and the reason for both.

    A small carrier rather than a pydantic model: it never leaves this module's
    call, and :class:`~data_hydration.models.HydrationItem` stores the three
    fields separately.
    """

    __slots__ = ("strategy", "partitions", "reason")

    def __init__(
        self,
        strategy: PartitionStrategy,
        partitions: list[Partition],
        reason: str,
    ) -> None:
        self.strategy = strategy
        self.partitions = partitions
        self.reason = reason

    def __str__(self) -> str:
        return f"{self.strategy} x{len(self.partitions)}: {self.reason}"


def _whole(reason: str) -> PartitionPlan:
    """The unpartitioned answer, with its justification."""
    return PartitionPlan(PartitionStrategy.NONE, [], reason)


def spde_partitions(directory: str, dataset: str) -> list[Partition]:
    """One :class:`~data_hydration.models.Partition` per ``.dpf`` component.

    An SPD Engine dataset stores its rows across numbered data-partition files
    in the library directory, which is why this is the cleanest partitionable
    source the module handles: the split already exists on disk, and finding it
    is a directory listing rather than a query.

    Components are matched on the dataset stem so one library holding several
    datasets does not hand every dataset every other dataset's partitions.
    Returned in sorted order — the filesystem's order is not stable, and item
    ordering is observable output.

    An unreadable directory yields no partitions rather than raising: the caller
    downgrades to a whole-object read, which still works.
    """
    stem = dataset.lower()
    try:
        entries = sorted(Path(directory).iterdir())
    except OSError as exc:
        logger.debug(f"spde_partitions: cannot list '{directory}': {exc}")
        return []
    return [
        Partition(name=path.name, component=str(path))
        for path in entries
        if path.suffix.lower() == SPDE_DATA_SUFFIX and path.stem.lower().startswith(stem)
    ]


def _row_ranges(rows: int, slices: int) -> list[Partition]:
    """*rows* split into at most *slices* contiguous offset/limit windows."""
    if rows <= 0 or slices <= 1:
        return []
    size = max(1, -(-rows // slices))  # ceiling division
    parts: list[Partition] = []
    for offset in range(0, rows, size):
        limit = min(size, rows - offset)
        parts.append(
            Partition(
                name=f"rows_{offset}_{offset + limit}",
                row_offset=offset,
                row_limit=limit,
            )
        )
    return parts


def _column_ranges(column: str, low: float, high: float, slices: int) -> list[Partition]:
    """``[low, high]`` split into *slices* half-open ranges over *column*.

    The last range closes with ``<=`` rather than ``<`` so the maximum value is
    included — an off-by-one here silently drops rows, which is the worst kind
    of bug this module could have.
    """
    if slices <= 1 or high <= low:
        return []
    width = (high - low) / slices
    parts: list[Partition] = []
    for index in range(slices):
        lo = low + width * index
        hi = low + width * (index + 1)
        last = index == slices - 1
        operator = "<=" if last else "<"
        bound = high if last else hi
        parts.append(
            Partition(
                name=f"{column}_{index}",
                predicate=f"{column} >= {lo} AND {column} {operator} {bound}",
            )
        )
    return parts


def plan_partitions(
    kind: SourceKind,
    *,
    locator: str = "",
    object_name: str = "",
    num_partitions: int = 8,
    probe: SourceProbe | None = None,
) -> PartitionPlan:
    """How to divide one source, and why.

    Parameters
    ----------
    kind
        What the source is — decides which strategies are even candidates.
    locator, object_name
        The source's coordinates: for SPD Engine, the library directory and the
        dataset name.
    num_partitions
        Ceiling on how many slices a non-native strategy produces.
    probe
        A live connection to ask, or ``None`` for static planning only. See the
        module docstring: ``None`` is what a report passes.
    """
    if kind is SourceKind.SPDE:
        # The components are found and reported, but NOT fanned out into an item
        # each. Reading one ``.dpf`` in isolation means bypassing the engine that
        # owns the layout, and the only supported reader — a SAS session — has no
        # way to select a single component. Emitting one item per component would
        # therefore read the whole dataset once per component and write it N
        # times. The count is real information, so it goes in the reason.
        parts = spde_partitions(locator, object_name)
        if parts:
            return _whole(
                f"SPD Engine library stores this dataset across {len(parts)} "
                f"{SPDE_DATA_SUFFIX} component(s), which are only readable "
                f"together through SAS — reading the dataset whole"
            )
        return _whole(
            f"no {SPDE_DATA_SUFFIX} components found under '{locator}' — "
            f"reading the dataset whole"
        )

    if probe is None:
        return _whole(
            "planned without a live connection, so only partitioning visible "
            "on disk was considered"
        )

    if kind is SourceKind.ORACLE:
        owner = _owner_of(object_name)
        table = _table_of(object_name)
        native = _safe(lambda: probe.native_partitions(owner, table))
        if native:
            return PartitionPlan(
                PartitionStrategy.NATIVE,
                [
                    Partition(name=name, predicate=f"PARTITION ({name})")
                    for name in native
                ],
                f"table declares {len(native)} native partition(s)",
            )
        ranged = _safe(lambda: probe.range_column(owner, table))
        if ranged:
            column, low, high = ranged
            parts = _column_ranges(column, low, high, num_partitions)
            if parts:
                return PartitionPlan(
                    PartitionStrategy.COLUMN_RANGE,
                    parts,
                    f"no native partitions; split on indexed column "
                    f"'{column}' across {len(parts)} range(s)",
                )
        return _whole(
            "no native partitions and no indexed numeric column to split on"
        )

    if kind is SourceKind.SAS_DATASET:
        rows = _safe(lambda: probe.row_count("", object_name))
        if rows:
            parts = _row_ranges(rows, num_partitions)
            if parts:
                return PartitionPlan(
                    PartitionStrategy.ROW_RANGE,
                    parts,
                    f"{rows} rows split into {len(parts)} row range(s)",
                )
        return _whole("row count unavailable — reading the dataset whole")

    return _whole(f"{kind} sources are read whole")


def _safe(call):  # type: ignore[no-untyped-def]
    """Run a probe call, turning any failure into ``None``.

    A probe exists to *improve* a plan. One that raises — a permission error on
    the data dictionary, a dropped connection — must downgrade the strategy to
    something that still works, not abort planning.
    """
    try:
        return call()
    except Exception as exc:
        logger.debug(f"plan_partitions: probe failed, downgrading strategy: {exc}")
        return None


def _owner_of(qualified: str) -> str:
    """The schema half of ``owner.table``, or empty when unqualified."""
    return qualified.split(".", 1)[0] if "." in qualified else ""


def _table_of(qualified: str) -> str:
    """The table half of ``owner.table``."""
    return qualified.split(".", 1)[1] if "." in qualified else qualified
