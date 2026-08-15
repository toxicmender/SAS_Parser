"""Turning what the chunker found into a plan somebody can read.

The input is two lists the chunker already produces —
:class:`~chunker.models.SasEngineRef` for database LIBNAMEs and
:class:`~chunker.models.SasPathRef` for everything with a path — and the output
is a :class:`~data_hydration.models.HydrationPlan`.

**Nothing here does I/O by default.** No driver is imported, no socket is opened,
no file is read; with ``probe=None`` even partitioning is decided from what the
statement itself said, plus a directory listing for SPD Engine. That is what
makes ``--dry-run`` a real check and what lets :mod:`complexity` build a plan
purely to print it.

The chunker types are annotations only. They are imported under
``TYPE_CHECKING``, so ``import data_hydration`` never pulls in :mod:`chunker` —
the decoupling this package is meant to keep.

What cannot be decided is *recorded*, not guessed. A connection whose password is
``&user_pass.`` is planned, marked with a blocker naming the unresolved macro,
and reported. Guessing would produce a plan that looks executable and is not.

Logger name: ``data_hydration.planner``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

from .config import HydrationConfig
from .models import (
    HydrationItem,
    HydrationPlan,
    HydrationSource,
    SourceKind,
    WriteMode,
)
from .naming import TableNameError, render, validate_template
from .partition import SourceProbe, plan_partitions

if TYPE_CHECKING:  # annotations only — never imported at run time
    from chunker.models import SasEngineRef, SasPathRef

logger = logging.getLogger(__name__)

#: FILENAME device keywords that name a hydratable remote source. The other
#: remote devices SAS supports (``email``, ``pipe``) move no table and are
#: deliberately absent — a mailbox is not a source.
_REMOTE_KINDS: dict[str, SourceKind] = {
    "ftp": SourceKind.SFTP,
    "sftp": SourceKind.SFTP,
    "azure": SourceKind.BLOB,
}

#: Path suffixes that identify a SAS data file.
_SAS_DATA_SUFFIX = ".sas7bdat"
_SAS_INDEX_SUFFIX = ".sas7bndx"


def _basename(path: str) -> str:
    """The final path component, suffix included."""
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _stem(path: str) -> str:
    """The final path component with its suffix removed."""
    tail = _basename(path)
    return tail.rsplit(".", 1)[0] if "." in tail else tail


def _directory(path: str) -> str:
    """Everything before the final path component."""
    cleaned = path.replace("\\", "/").rstrip("/")
    return cleaned.rsplit("/", 1)[0] if "/" in cleaned else ""


def _macro_blocker(source: HydrationSource) -> tuple[str, ...]:
    """A blocker naming *which* coordinates are unresolved macro references.

    Naming the options is the point: "there is a macro in here somewhere" sends
    the operator back to the source to find it, and the statement is right here.
    """
    if not source.has_macro_ref:
        return ()
    unresolved = sorted(key for key, value in source.options if "&" in value)
    where = (
        f"option(s) {', '.join(unresolved)}" if unresolved else "the connection"
    )
    return (
        f"unresolved macro reference in {where} — SAS resolves these at run "
        f"time, so the values recorded here are not the ones it would use",
    )


def _library_blocker(source: HydrationSource) -> tuple[str, ...]:
    """A blocker for a LIBNAME that names a directory rather than one dataset.

    ``libname flat '/sasdata3/dataetl';`` binds a whole library. Which datasets
    are in it is only knowable by listing the directory, which static planning
    does not do — so the reference is reported (it is a real dependency of the
    corpus) and marked as needing expansion, rather than being silently modelled
    as a single file that does not exist.
    """
    if source.kind is not SourceKind.FILE or source.object_name:
        return ()
    return (
        f"'{source.locator}' is a library directory, not a single dataset — "
        f"list it and hydrate each member, or point the LIBNAME at one file",
    )


def _oracle_source(ref: "SasEngineRef", source_id: str | None) -> HydrationSource:
    """A :class:`HydrationSource` for one database-engine LIBNAME."""
    options = ref.option_map
    # SAS spells the service name `path=` on the Oracle engine; the schema is
    # what a table is qualified by.
    return HydrationSource(
        kind=SourceKind(ref.engine) if ref.engine in _ENGINE_KINDS else SourceKind.ORACLE,
        locator=options.get("path", "") or options.get("server", ""),
        object_name=options.get("schema", "") or ref.binds,
        libref=ref.binds,
        options=ref.options,
        has_macro_ref=ref.has_macro_ref,
        source_id=source_id,
    )


#: Engines with a :class:`SourceKind` of their own. Everything else is planned
#: as ``ORACLE`` — the SQL path — because the shape of the work is the same and
#: the plan records the real engine in ``options``.
_ENGINE_KINDS = frozenset({"oracle"})


def _path_source(ref: "SasPathRef", source_id: str | None) -> HydrationSource | None:
    """A :class:`HydrationSource` for one path reference, or ``None``.

    ``None`` for a reference that names no data to move — a shell pipe, a
    mailbox, an ``%INCLUDE`` of more SAS source, an ODS report destination.
    Being selective here is what keeps the plan an inventory of *data* rather
    than of every string in the corpus.
    """
    if ref.statement in {"include", "ods", "sasautos", "file", "proc_export"}:
        return None
    # A .sas7bndx is an INDEX, not data. Left in, it would be planned as an
    # ordinary file and — because it shares its stem with the dataset it indexes
    # — render the same target table, appending index pages into it as rows.
    # Its only role here is the note ``_index_note`` attaches to the dataset.
    if ref.path.endswith(_SAS_INDEX_SUFFIX):
        return None

    kind: SourceKind
    if ref.engine == "spde":
        kind = SourceKind.SPDE
    elif str(ref.location) == "remote":
        mapped = _REMOTE_KINDS.get(ref.device or "")
        if mapped is None:
            return None
        kind = mapped
    elif str(ref.location) != "filesystem":
        return None
    elif ref.path.endswith(_SAS_DATA_SUFFIX):
        kind = SourceKind.SAS_DATASET
    else:
        kind = SourceKind.FILE

    # An SPD Engine LIBNAME names the library directory; the dataset name is not
    # in the statement, so the libref stands in until a listing resolves it.
    if kind is SourceKind.SPDE:
        return HydrationSource(
            kind=kind,
            locator=ref.raw,
            object_name=ref.binds or _stem(ref.path),
            libref=ref.binds,
            has_macro_ref=ref.has_macro_ref,
            source_id=source_id,
        )
    # A path with no file suffix is a directory — a LIBNAME binding a whole
    # library, not one dataset. It keeps the whole path as its locator and no
    # object name, which is what ``_library_blocker`` reads.
    if kind is SourceKind.FILE and "." not in _basename(ref.path):
        return HydrationSource(
            kind=kind,
            locator=ref.raw,
            libref=ref.binds,
            has_macro_ref=ref.has_macro_ref,
            source_id=source_id,
        )
    return HydrationSource(
        kind=kind,
        locator=_directory(ref.raw),
        object_name=_stem(ref.path),
        libref=ref.binds,
        has_macro_ref=ref.has_macro_ref,
        source_id=source_id,
    )


def _index_note(
    source: HydrationSource, path_refs: Iterable["SasPathRef"]
) -> tuple[str, ...]:
    """A note when a SAS index sits beside this dataset.

    A ``.sas7bndx`` file names an index on the dataset of the same stem. Only
    its *presence* is knowable here: the indexed column names live inside a
    binary whose layout is undocumented, and reading it is the reader's job at
    write time — see :func:`data_hydration.sources.sas_files.index_columns`.

    So this returns prose, not column names. Putting a placeholder in
    :attr:`~data_hydration.models.HydrationItem.cluster_by` instead would reach
    the sink and be emitted as ``CLUSTER BY (`<indexed>`)``, which is not valid
    SQL and not a column.
    """
    if source.kind is not SourceKind.SAS_DATASET:
        return ()
    stem = source.object_name.lower()
    for ref in path_refs:
        if ref.path.endswith(_SAS_INDEX_SUFFIX) and _stem(ref.path).lower() == stem:
            return (
                "a SAS index sits beside this dataset; its columns are a "
                "candidate CLUSTER BY, read at load time and applied only "
                "when data_hydration.apply_index_clustering is set",
            )
    return ()


#: Stands in for a target name that could not be rendered. Never written to —
#: an item carrying it always carries a blocker too.
UNRESOLVED_TARGET = "<unresolved>"


def _target_for(
    source: HydrationSource, config: HydrationConfig, run_date: str
) -> tuple[str, tuple[str, ...]]:
    """The managed-table name *source* lands in, plus any blocker that stopped it.

    The schema defaults to the SAS libref when none is configured: a migration
    that keeps ``edwprod.accounts`` recognisable as
    ``<catalog>.edwprod.accounts`` is doing the least surprising thing. An
    ``INFILE`` naming a bare path has no libref, though, and then there is
    nothing to default to.

    A *value* the template needs and cannot get is recorded as a blocker on this
    one item rather than raised, because the plan is a report before it is a
    work queue: one path without a libref must not cost the operator the view of
    the other forty tables. A broken *template* still raises — that is
    :func:`~data_hydration.naming.validate_template`, checked once for the run.
    """
    try:
        return (
            render(
                config.table_template,
                catalog_name=config.catalog,
                schema_name=config.schema or source.libref,
                table_name=source.object_name,
                stage=config.stage,
                date=run_date,
                libref=source.libref,
                source=str(source.kind),
            ),
            (),
        )
    except TableNameError as exc:
        logger.debug(f"_target_for: no target name for {source}: {exc}")
        return (UNRESOLVED_TARGET, (f"no target table name: {exc}",))


def _sources_for(
    engine_refs: Sequence["SasEngineRef"],
    path_refs: Sequence["SasPathRef"],
    source_id: str | None,
) -> list[HydrationSource]:
    """Every hydratable source one file's refs name, in reading order."""
    sources = [_oracle_source(ref, source_id) for ref in engine_refs]
    for path_ref in path_refs:
        source = _path_source(path_ref, source_id)
        if source is not None:
            sources.append(source)
    return sources


def build_plan(
    engine_refs: Sequence["SasEngineRef"] = (),
    path_refs: Sequence["SasPathRef"] = (),
    *,
    config: HydrationConfig | None = None,
    probe: SourceProbe | None = None,
    source_id: str | None = None,
) -> HydrationPlan:
    """Everything a run would do against one file's references, without doing it.

    Parameters
    ----------
    engine_refs, path_refs
        What the chunker found — ``chunk.metadata.engine_refs`` and
        ``chunk.metadata.external_refs``, from as many chunks as the caller
        wants covered.
    config
        ``None`` builds one with :meth:`HydrationConfig.from_env`.
    probe
        A live connection for partition discovery, or ``None`` for static
        planning. :mod:`complexity` always passes ``None``.
    source_id
        The SAS file these refs came from, recorded on every source so
        :meth:`HydrationPlan.by_source_id` can group items per file. Use
        :func:`build_corpus_plan` for a whole corpus — it keeps write modes
        correct across files, which calling this once per file cannot.

    Raises
    ------
    ~data_hydration.naming.TableNameError
        The configured **template** is invalid — an unknown placeholder, or a
        shape that cannot produce a three-level name. Raised before any item is
        built, so a misconfiguration fails on the first line of a dry run
        rather than after a table has been written.

        A missing *value* is not this: a source that cannot fill a placeholder
        gets :data:`UNRESOLVED_TARGET` and a blocker, and the rest of the plan
        survives. See :func:`_target_for`.
    """
    return build_corpus_plan(
        {source_id or "": (engine_refs, path_refs)}, config=config, probe=probe
    )


def build_corpus_plan(
    by_source: Mapping[str, tuple[Sequence["SasEngineRef"], Sequence["SasPathRef"]]],
    *,
    config: HydrationConfig | None = None,
    probe: SourceProbe | None = None,
) -> HydrationPlan:
    """One plan across a whole corpus, keyed by SAS file.

    Building the corpus in one pass — rather than merging per-file plans — is
    what keeps :class:`~data_hydration.models.WriteMode` honest. Two files
    declaring the same LIBNAME produce items for one target table, and exactly
    one of them may overwrite it; per-file plans merged afterwards would each
    think they were first and the second would wipe the first's rows.

    Files are visited in the mapping's order, which the caller controls.
    """
    config = config or HydrationConfig.from_env()
    validate_template(config.table_template)
    run_date = config.run_date

    sources: list[HydrationSource] = []
    path_ref_list: list["SasPathRef"] = []
    for source_id, (engine_refs, path_refs) in by_source.items():
        sources += _sources_for(engine_refs, path_refs, source_id or None)
        path_ref_list += list(path_refs)

    items: list[HydrationItem] = []
    seen_tables: set[str] = set()
    for source in sources:
        target, name_blockers = _target_for(source, config, run_date)
        blockers = list(_macro_blocker(source))
        library = _library_blocker(source)
        blockers += library
        # A library directory has no table name *because* it is a directory, so
        # its rendering failure is already explained; reporting both would say
        # the same thing twice in less useful words.
        if not library:
            blockers += name_blockers
        if source.kind is SourceKind.SPDE and not config.has_sas_session:
            blockers.append(
                "SPD Engine components have no open-source reader; configure "
                "data_hydration.sas_host so the library can be read through SAS"
            )
        partitioned = plan_partitions(
            source.kind,
            locator=source.locator,
            object_name=source.object_name,
            num_partitions=config.num_partitions,
            probe=probe,
        )
        notes = _index_note(source, path_ref_list)

        # The first item for a table replaces it and the rest add to it, so a
        # re-run is idempotent whatever order the runner executes in.
        first = target not in seen_tables
        seen_tables.add(target)

        if not partitioned.partitions:
            items.append(
                HydrationItem(
                    source=source,
                    target_table=target,
                    write_mode=WriteMode.OVERWRITE if first else WriteMode.APPEND,
                    strategy=partitioned.strategy,
                    strategy_reason=partitioned.reason,
                    notes=notes,
                    blockers=tuple(blockers),
                )
            )
            continue
        for index, partition in enumerate(partitioned.partitions):
            items.append(
                HydrationItem(
                    source=source,
                    target_table=target,
                    write_mode=(
                        WriteMode.OVERWRITE if first and index == 0 else WriteMode.APPEND
                    ),
                    strategy=partitioned.strategy,
                    strategy_reason=partitioned.reason,
                    partition=partition,
                    notes=notes,
                    blockers=tuple(blockers),
                )
            )

    logger.info(
        f"build_corpus_plan: {len(sources)} source(s) across {len(by_source)} file(s) "
        f"-> {len(items)} item(s) (probe={'yes' if probe else 'no'})"
    )
    return HydrationPlan(run_date=run_date, items=items)
