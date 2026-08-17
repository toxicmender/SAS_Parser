"""What a hydration run is made of, as data.

Every model here is inert: building one opens no connection, reads no file, and
needs no driver installed. That is the property :mod:`data_hydration.planner`
depends on and :mod:`complexity` relies on — a report renderer builds a plan to
print it, and must not be able to reach the network by doing so.

The shape is a two-level one. A :class:`HydrationSource` is *what the SAS named*
— one Oracle table, one file on an sFTP host, one SPD Engine library. A
:class:`HydrationItem` is *one unit of work against it*: a whole table, or one
partition of one. A source with four partitions produces four items sharing one
source, which is why the source is frozen and the item is not.

Logger name: ``data_hydration.models``.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from pydantic import BaseModel, Field, computed_field

logger = logging.getLogger(__name__)


class SourceKind(StrEnum):
    """What kind of system a :class:`HydrationSource` reads from.

    The distinction that matters is not the protocol but what the module has to
    *do*: ``ORACLE`` issues SQL, the file kinds move bytes, and ``SPDE`` needs a
    SAS session because no open-source reader for its component files exists.
    """

    ORACLE = "oracle"
    SFTP = "sftp"
    ADLS = "adls"
    BLOB = "blob"
    FILE = "file"
    SAS_DATASET = "sas7bdat"
    SPDE = "spde"
    SAS_SESSION = "sas_session"


class PartitionStrategy(StrEnum):
    """How one source was divided into items.

    ``NONE`` is a real answer, not a failure: most files are read whole, and
    saying so is what lets a report distinguish "not partitioned" from "we did
    not look".
    """

    NONE = "none"
    #: Partitions the source itself declares — Oracle table partitions, or the
    #: ``.dpf`` component files of an SPD Engine library.
    NATIVE = "native"
    #: Row offset/limit slices, for a reader that can seek by row.
    ROW_RANGE = "row_range"
    #: ``WHERE col >= lo AND col < hi`` over a numeric or date column.
    COLUMN_RANGE = "column_range"


class WriteMode(StrEnum):
    """How an item writes into its target table.

    Decided when the plan is built, never at run time. The first item for a
    table overwrites and the rest append, so a re-run is idempotent and the
    result does not depend on the order the runner happens to execute in.
    """

    OVERWRITE = "overwrite"
    APPEND = "append"


class HydrationSource(BaseModel, frozen=True):
    """One external object a SAS program reads, as coordinates.

    Frozen because the planner groups items by source to assign write modes, and
    a mapping key has to be hashable.

    Attributes
    ----------
    kind
        See :class:`SourceKind`.
    libref
        The SAS libref or fileref this was bound to, when it was bound to one.
        Carried so a report can say which ``LIBNAME`` a row came from.
    locator
        What addresses the *system*: an Oracle DSN or service name, an sFTP
        host, a storage account, or the directory holding a file.
    object_name
        What addresses the *object* inside it: a table name, a file name, an SPD
        Engine dataset name.
    options
        The connection options as declared, keys lowercased and values exactly
        as the SAS wrote them — including unresolved ``&macro`` references.
        Tuple-of-pairs rather than a dict so the model stays hashable, the same
        trade :class:`chunker.models.SasEngineRef` makes.
    has_macro_ref
        Some coordinate contains a macro reference and is therefore not the
        value SAS would resolve at run time.
    source_id
        The SAS file this was found in, for reporting. Not part of the identity
        of the source itself — two files naming one table name it once.
    """

    kind: SourceKind
    locator: str = ""
    object_name: str = ""
    libref: str | None = None
    options: tuple[tuple[str, str], ...] = ()
    has_macro_ref: bool = False
    source_id: str | None = None

    @property
    def option_map(self) -> dict[str, str]:
        """:attr:`options` as a mapping."""
        return dict(self.options)

    def __str__(self) -> str:
        bound = f" ({self.libref})" if self.libref else ""
        # Only the join is cleaned up; a leading slash is part of the path and
        # stripping it turns an absolute path into a misleading relative one.
        where = "/".join(p for p in (self.locator, self.object_name) if p)
        return f"{self.kind}:{where}{bound}"


class Partition(BaseModel, frozen=True):
    """One slice of a source, in whichever terms its reader understands.

    Exactly one addressing field is set, decided by the strategy that produced
    it: a ``NATIVE`` Oracle partition has :attr:`predicate`, a ``NATIVE`` SPD
    Engine one has :attr:`component`, a ``ROW_RANGE`` has the offset/limit pair.
    Keeping them on one model rather than three subclasses is deliberate — the
    report prints :attr:`name` and nothing else cares.
    """

    name: str
    #: A SQL fragment: ``PARTITION (p2024_01)`` or ``col >= 0 AND col < 1000``.
    predicate: str | None = None
    row_offset: int | None = None
    row_limit: int | None = None
    #: Path to a single component file of a multi-file source.
    component: str | None = None

    def __str__(self) -> str:
        return self.name


class HydrationItem(BaseModel):
    """One unit of work: read this much of this source, write it there.

    Attributes
    ----------
    source
        What is being read.
    target_table
        The fully-qualified managed table this lands in, already rendered
        through :mod:`data_hydration.naming`. A real name, not a template —
        by the time an item exists the template has been resolved and
        validated.
    write_mode
        See :class:`WriteMode`.
    strategy
        Which :class:`PartitionStrategy` produced this item.
    strategy_reason
        Why that strategy and not another, in a sentence a report can print.
        Recorded because "not partitioned" is a decision somebody will want to
        second-guess, and the plan should answer without being re-run.
    partition
        The slice, or ``None`` when :attr:`strategy` is ``NONE``.
    notes
        Things worth reporting that block nothing — "a SAS index sits beside
        this dataset", say. Distinct from :attr:`blockers`, which stop the item
        running; a note is for the reader, not the runner.
    cluster_by
        Columns to cluster the Delta table on. Empty at planning time: the
        column names live inside the ``.sas7bndx`` binary, and reading it is
        the reader's job, not the planner's — see
        :mod:`data_hydration.sources.sas_files`. Applied only when
        ``data_hydration.apply_index_clustering`` is set, because a SAS index
        and Delta clustering solve overlapping but different problems.
    blockers
        Reasons this item cannot run unattended — an unresolved macro in the
        connection, an SPD Engine library with no SAS session configured. A
        non-empty list means the plan is reportable but not fully executable,
        which is a state worth showing rather than failing on.
    """

    source: HydrationSource
    target_table: str
    write_mode: WriteMode = WriteMode.OVERWRITE
    strategy: PartitionStrategy = PartitionStrategy.NONE
    strategy_reason: str = ""
    partition: Partition | None = None
    notes: tuple[str, ...] = ()
    cluster_by: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def needs_operator_input(self) -> bool:
        """True when something must be decided by a human before this can run."""
        return bool(self.blockers)

    def __str__(self) -> str:
        part = f" [{self.partition}]" if self.partition else ""
        flag = " (needs operator input)" if self.blockers else ""
        return f"{self.source}{part} -> {self.target_table}{flag}"


class HydrationPlan(BaseModel):
    """Every item a run would execute, plus the instant it was planned at.

    Attributes
    ----------
    run_date
        The formatted date every item's target name was rendered with, captured
        **once** when the plan was built. A run that crosses midnight must not
        write half its partitions into yesterday's table and half into today's,
        so the value lives here rather than being re-derived per item.
    items
        In planning order. Items sharing a target table are adjacent, and the
        first of each group carries ``WriteMode.OVERWRITE``.
    """

    run_date: str = ""
    items: list[HydrationItem] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:  # noqa: ANN001
        logger.info(
            f"HydrationPlan: {len(self.items)} item(s), "
            f"{len(self.target_tables)} table(s), "
            f"{self.blocked_count} needing operator input"
        )

    @property
    def target_tables(self) -> list[str]:
        """Distinct target tables, in first-appearance order."""
        seen: dict[str, None] = {}
        for item in self.items:
            seen.setdefault(item.target_table, None)
        return list(seen)

    @property
    def blocked_count(self) -> int:
        """How many items need a human before they can run."""
        return sum(1 for item in self.items if item.blockers)

    def by_source_id(self, source_id: str) -> list[HydrationItem]:
        """The items that came from one SAS file — what a per-file report needs."""
        return [i for i in self.items if i.source.source_id == source_id]

    def counts_by_kind(self) -> dict[str, int]:
        """Item count per :class:`SourceKind`, for the corpus summary."""
        counts: dict[str, int] = {}
        for item in self.items:
            counts[str(item.source.kind)] = counts.get(str(item.source.kind), 0) + 1
        return dict(sorted(counts.items()))

    def __str__(self) -> str:
        return (
            f"HydrationPlan({len(self.items)} items, "
            f"{len(self.target_tables)} tables, date={self.run_date})"
        )


class ItemStatus(StrEnum):
    """What became of one item."""

    WRITTEN = "written"
    #: Planned but not executed — a dry run, or a blocked item.
    SKIPPED = "skipped"
    FAILED = "failed"


class ItemOutcome(BaseModel):
    """What one item did, successful or not.

    A failure is recorded rather than raised: one unreachable host must not cost
    the operator the other forty tables, so the runner catches per item and the
    report carries the error text.
    """

    item: HydrationItem
    status: ItemStatus
    rows: int | None = None
    error: str | None = None

    def __str__(self) -> str:
        detail = f" — {self.error}" if self.error else ""
        rows = f" ({self.rows} rows)" if self.rows is not None else ""
        return f"[{self.status}] {self.item}{rows}{detail}"


class HydrationReport(BaseModel):
    """The result of executing a plan.

    Shaped like :class:`conversion.run.RunOutcome`: counts a caller can act on,
    plus the per-item detail behind them.
    """

    plan: HydrationPlan
    outcomes: list[ItemOutcome] = Field(default_factory=list)
    dry_run: bool = False

    def model_post_init(self, __context: object) -> None:  # noqa: ANN001
        logger.info(
            f"HydrationReport: {self.written} written, {self.skipped} skipped, "
            f"{self.failed} failed{' (dry run)' if self.dry_run else ''}"
        )

    @property
    def written(self) -> int:
        return sum(1 for o in self.outcomes if o.status is ItemStatus.WRITTEN)

    @property
    def skipped(self) -> int:
        return sum(1 for o in self.outcomes if o.status is ItemStatus.SKIPPED)

    @property
    def failed(self) -> int:
        return sum(1 for o in self.outcomes if o.status is ItemStatus.FAILED)

    @property
    def ok(self) -> bool:
        """True when nothing failed. A dry run with no failures is ok."""
        return self.failed == 0

    def __str__(self) -> str:
        return (
            f"HydrationReport({self.written} written, {self.skipped} skipped, "
            f"{self.failed} failed)"
        )
