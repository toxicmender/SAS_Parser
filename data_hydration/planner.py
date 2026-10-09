"""Turning what the chunker found into a plan somebody can read.

The input is what the chunker already produces —
:class:`~chunker.models.SasEngineRef` for database LIBNAMEs,
:class:`~chunker.models.SasPathRef` for everything with a path, and
:class:`~chunker.models.SasDbTableRef` for the individual database tables the
corpus reads (SQL pass-through, or a member of a database LIBNAME) — and the
output is a :class:`~data_hydration.models.HydrationPlan`.

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
    from chunker.models import SasDatasetRef, SasDbTableRef, SasEngineRef, SasPathRef

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
    parts = [f"option(s) {', '.join(unresolved)}"] if unresolved else []
    # A path's place; a database's is one of its options, named above.
    if "&" in source.locator and not source.options:
        parts.append(f"the location '{source.locator}'")
    if "&" in source.object_name:
        parts.append(f"the object name '{source.object_name}'")
    where = " and ".join(parts) or "the connection"
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
#: as ``ORACLE`` — the SQL path, the shape of the work being the same — and
#: blocked by :func:`_engine_blocker`, since only the Oracle reader exists.
_ENGINE_KINDS = frozenset({"oracle"})


def _engine_blocker(engine: str | None) -> tuple[str, ...]:
    """A blocker for a database no reader here connects to.

    Teradata, DB2, SQL Server and the rest are planned through the SQL path so
    the plan lists them, but the one SQL reader speaks Oracle: run as planned,
    it would query the configured Oracle connection for a table that lives
    somewhere else.
    """
    if engine is None or engine in _ENGINE_KINDS:
        return ()
    return (
        f"a {engine} database: the hydration readers connect to Oracle only — "
        f"load this table with a {engine} reader, or point the source at an "
        f"Oracle copy",
    )


def _list_blocker(ref: "SasDbTableRef") -> tuple[str, ...]:
    """A blocker for a read of a list of tables.

    ``set edw.acct_:;`` reads every table whose name starts ``acct_``, and
    ``proc copy in=edw`` every table there is (``edw.:``). Which tables those
    are only the database knows, so the list is one item, planned to no table.
    """
    if not ref.table.endswith(":"):
        return ()
    tables = f"every {ref.db_schema or 'default-schema'} table"
    if prefix := ref.table[:-1]:
        tables += f" whose name starts '{prefix}'"
    return (
        f"{ref.raw} reads a list of tables — {tables}: the plan cannot name them "
        f"without asking the database; list the tables the job needs",
    )


def _db_table_source(
    ref: "SasDbTableRef", source_id: str | None
) -> tuple[HydrationSource, tuple[str, ...]] | None:
    """A source for one database table the corpus *reads*, with the blockers
    only this kind of source can have — or ``None`` for a write, or for a
    macro-body template named by the macro's own parameters.

    A write is something the converted job produces, never a table to load.
    ``object_name`` is ``owner.table`` (``partition._owner_of`` and the Oracle
    reader split it back), and the connection options ride on the record, so
    no join by alias is needed. Compared by the string value of the chunker's
    enums, which keeps this module free of a run-time chunker import.
    """
    if str(ref.access) != "read" or ref.parameterised:
        # A write is the job's output; a parameterised name is a template in a
        # macro body, whose every call is recorded — resolved — on its own.
        return None
    options = ref.option_map
    blockers: list[str] = []
    if ref.engine is None:
        blockers.append(
            f"the database behind connection '{ref.connection}' is unknown — no "
            f"CONNECT TO for it in the same PROC SQL (a macro call probably made "
            f"it); confirm the engine and its options before loading"
        )
    if ref.dblink:
        blockers.append(
            f"read through database link '{ref.dblink}': the table lives in the "
            f"linked database, not the one this connection reaches — point the "
            f"source at that database"
        )
    if ref.engine is not None:
        blockers.extend(_engine_blocker(ref.engine))
    blockers.extend(_list_blocker(ref))
    source = HydrationSource(
        kind=SourceKind(ref.engine) if ref.engine in _ENGINE_KINDS else SourceKind.ORACLE,
        locator=options.get("path", "") or options.get("server", ""),
        object_name=ref.qualified,
        libref=ref.connection if str(ref.via) == "libname" else None,
        connection=ref.connection,
        options=ref.options,
        has_macro_ref=ref.has_macro_ref or any("&" in v for _, v in ref.options),
        source_id=source_id,
    )
    return source, tuple(blockers)


def _path_source(ref: "SasPathRef", source_id: str | None) -> HydrationSource | None:
    """A :class:`HydrationSource` for one path reference, or ``None``.

    ``None`` for a reference that names no data to move — a shell pipe, a
    mailbox, an ``%INCLUDE`` of more SAS source, an ODS report destination.
    Being selective here is what keeps the plan an inventory of *data* rather
    than of every string in the corpus.
    """
    # Where a job writes (FILE, PROC EXPORT, ODS, PROC PRINTTO's log and
    # listing) and the SAS it pulls in are no data to load.
    if ref.statement in {"include", "ods", "sasautos", "file", "proc_export", "printto"}:
        return None
    # ``infile in;`` reads through a fileref, at the path its FILENAME names
    # (chunker.metadata.resolve_filerefs): that FILENAME's own reference is
    # the source, planned once.
    if ref.statement == "infile" and ref.binds is not None:
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
            locator=ref.effective_path,
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
            locator=ref.effective_path,
            libref=ref.binds,
            has_macro_ref=ref.has_macro_ref,
            source_id=source_id,
        )
    return HydrationSource(
        kind=kind,
        locator=_directory(ref.effective_path),
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
    schema_name = config.schema or source.libref
    table_name = source.object_name
    if source.connection is not None and table_name.endswith(":"):
        # A list of tables has a target per table, so none of its own; its
        # blocker (_list_blocker) says why.
        return (UNRESOLVED_TARGET, ())
    if source.connection is not None and "." in table_name:
        # A database table is owner.table: the table part names the target, and
        # the owner stands in for the schema when there is no libref to keep —
        # edw_export.current_nonip lands as <catalog>.edw_export.current_nonip.
        owner, table_name = table_name.rsplit(".", 1)
        schema_name = schema_name or owner
    try:
        return (
            render(
                config.table_template,
                catalog_name=config.catalog,
                schema_name=schema_name,
                table_name=table_name,
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
    *,
    db_tables: Sequence["SasDbTableRef"] = (),
    named: frozenset[tuple[str, tuple[tuple[str, str], ...]]] = frozenset(),
) -> list[tuple[HydrationSource, tuple[str, ...]]]:
    """Every hydratable source one file's refs name, in reading order, each with
    the blockers only its kind can have.

    *named* holds the database LIBNAMEs — ``(libref, options)`` — whose tables
    the corpus names: those are planned table by table, so the LIBNAME's own
    schema-level item, which stands in when no table is known, is left out.
    """
    sources: list[tuple[HydrationSource, tuple[str, ...]]] = [
        (_oracle_source(ref, source_id), _engine_blocker(ref.engine))
        for ref in engine_refs
        if (ref.binds, ref.options) not in named
    ]
    for path_ref in path_refs:
        source = _path_source(path_ref, source_id)
        if source is not None:
            sources.append((source, ()))
    for table in db_tables:
        planned = _db_table_source(table, source_id)
        if planned is not None:
            sources.append(planned)
    return sources


def build_plan(
    engine_refs: Sequence["SasEngineRef"] = (),
    path_refs: Sequence["SasPathRef"] = (),
    *,
    db_tables: Sequence["SasDbTableRef"] = (),
    datasets: Sequence["SasDatasetRef"] = (),
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
    db_tables
        ``chunk.metadata.db_tables`` from the same chunks: every database table
        read is planned as its own item (writes are the job's output, not a
        source), and a database LIBNAME whose tables appear here is planned
        through them instead of as one schema-level item.
    datasets
        ``chunk.metadata.dataset_refs`` from the same chunks: a directory
        LIBNAME whose datasets are read here is planned per dataset (see
        :func:`build_corpus_plan`).
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
        {source_id or "": (engine_refs, path_refs)},
        db_tables={source_id or "": db_tables},
        datasets={source_id or "": datasets},
        config=config,
        probe=probe,
    )


#: LIBNAME engines whose directory holds each dataset as ``<member>.sas7bdat``:
#: the default, written or not.
_BASE_ENGINES = frozenset({"", "base", "v9", "v8", "v7"})


def _directory_library(ref: "SasPathRef") -> bool:
    """Whether *ref* is a LIBNAME binding a directory of SAS datasets."""
    return (
        ref.statement == "libname"
        and bool(ref.binds)
        and not (ref.binds or "").startswith("&")
        and str(ref.location) == "filesystem"
        and (ref.engine or "") in _BASE_ENGINES
        and "." not in _basename(ref.path)
    )


#: A directory library: its libref and directory, as a library item has them.
_Library = tuple[str | None, str]


def _library_members(
    by_source: Mapping[str, tuple[Sequence["SasEngineRef"], Sequence["SasPathRef"]]],
    datasets: Mapping[str, Sequence["SasDatasetRef"]],
) -> tuple[dict[_Library, list[tuple[str, str | None]]], set[_Library]]:
    """The datasets to load from each directory library, and the libraries a
    list is read from.

    A dataset the corpus reads, or updates, through a libref a directory
    LIBNAME binds, and that no step creates, is a member of that directory to
    load: ``set raw.customers;`` after ``libname raw '/data/raw';`` reads
    ``/data/raw/customers.sas7bdat``. The LIBNAME in force is the latest one
    before the read in corpus order, a file's LIBNAMEs taken before its reads,
    as SAS would have run them. A dataset some step creates is the job's own,
    not a source; a list (``raw.sales_:``) has members only a listing knows.

    Returns each library's members as ``(member, reading file)`` in first-read
    order, and the libraries a list is read from.
    """
    created = {
        ref.name for refs in datasets.values() for ref in refs if str(ref.role) == "write"
    }
    bound: dict[str, str] = {}
    members: dict[_Library, list[tuple[str, str | None]]] = {}
    listed: set[_Library] = set()
    seen: set[tuple[str, str]] = set()
    for source_id in dict.fromkeys([*by_source, *datasets]):
        for ref in by_source.get(source_id, ((), ()))[1]:
            if _directory_library(ref) and ref.binds:
                bound[ref.binds] = ref.effective_path
        for ref in datasets.get(source_id, ()):
            if str(ref.role) not in ("read", "update") or ref.param is not None:
                continue
            libref, _, member = ref.name.partition(".")
            if libref not in bound or not member:
                continue
            library = (libref, bound[libref])
            if ref.pattern:
                listed.add(library)
            elif ref.name not in created and (library[1], member) not in seen:
                seen.add((library[1], member))
                members.setdefault(library, []).append((member, source_id or None))
    return members, listed


def build_corpus_plan(
    by_source: Mapping[str, tuple[Sequence["SasEngineRef"], Sequence["SasPathRef"]]],
    *,
    db_tables: Mapping[str, Sequence["SasDbTableRef"]] | None = None,
    datasets: Mapping[str, Sequence["SasDatasetRef"]] | None = None,
    config: HydrationConfig | None = None,
    probe: SourceProbe | None = None,
) -> HydrationPlan:
    """One plan across a whole corpus, keyed by SAS file.

    Building the corpus in one pass — rather than merging per-file plans — is
    what keeps :class:`~data_hydration.models.WriteMode` honest. Two files
    declaring the same LIBNAME produce items for one target table, and exactly
    one of them may overwrite it; per-file plans merged afterwards would each
    think they were first and the second would wipe the first's rows.

    Files are visited in the mapping's order, which the caller controls; a file
    present only in *db_tables* or *datasets* follows them.

    *db_tables* maps the same file keys to ``chunk.metadata.db_tables``. Pass
    metadata resolved across the corpus (``chunker.resolve_corpus_references``)
    or a database LIBNAME in one file cannot reach the reads in another. A table
    read in several places is **one** item, owned by the first file that reads
    it: appending one copy per reader would load its rows several times over.

    *datasets* maps them to ``chunk.metadata.dataset_refs``, and is how a
    directory LIBNAME is planned per dataset rather than as one blocked
    library item (see :func:`_library_members`): each member read is a
    ``sas7bdat`` item, owned by the first file that reads it. The library item
    stays only when a list is read from it.
    """
    config = config or HydrationConfig.from_env()
    validate_template(config.table_template)
    run_date = config.run_date

    db_tables = db_tables or {}
    named = frozenset(
        (table.connection, table.options)
        for tables in db_tables.values()
        for table in tables
        if str(table.via) == "libname"
    )
    members, listed = _library_members(by_source, datasets or {})
    expanded: set[_Library] = set()

    sources: list[tuple[HydrationSource, tuple[str, ...]]] = []
    planned_tables: set[HydrationSource] = set()
    path_ref_list: list["SasPathRef"] = []
    for source_id in dict.fromkeys([*by_source, *db_tables]):
        engine_refs, path_refs = by_source.get(source_id, ((), ()))
        for source, extra in _sources_for(
            engine_refs,
            path_refs,
            source_id or None,
            db_tables=db_tables.get(source_id, ()),
            named=named,
        ):
            library = (source.libref, source.locator)
            if source.kind is SourceKind.FILE and not source.object_name and library in members:
                # A directory whose datasets are named: one item per dataset,
                # where the library is first declared.
                if library not in expanded:
                    expanded.add(library)
                    sources += [
                        (
                            HydrationSource(
                                kind=SourceKind.SAS_DATASET,
                                locator=source.locator,
                                object_name=member,
                                libref=source.libref,
                                has_macro_ref=source.has_macro_ref or "&" in member,
                                source_id=reader,
                            ),
                            (),
                        )
                        for member, reader in members[library]
                    ]
                if library not in listed:
                    continue
            if source.connection is not None:
                identity = source.model_copy(update={"source_id": None})
                if identity in planned_tables:
                    continue
                planned_tables.add(identity)
            sources.append((source, extra))
        path_ref_list += list(path_refs)

    items: list[HydrationItem] = []
    seen_tables: set[str] = set()
    for source, extra in sources:
        target, name_blockers = _target_for(source, config, run_date)
        blockers = [*_macro_blocker(source), *extra]
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
        f"build_corpus_plan: {len(sources)} source(s) across "
        f"{len(by_source.keys() | db_tables.keys())} file(s) "
        f"-> {len(items)} item(s) (probe={'yes' if probe else 'no'})"
    )
    return HydrationPlan(run_date=run_date, items=items)
