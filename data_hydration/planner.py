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
from typing import TYPE_CHECKING, NamedTuple

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

#: Sources whose :attr:`~data_hydration.models.HydrationSource.object_name` is
#: a file name, suffix included — what their readers open.
_FILE_KINDS = frozenset(
    {SourceKind.FILE, SourceKind.SFTP, SourceKind.BLOB, SourceKind.ADLS}
)

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
    # the source, planned once. So does PROC CIMPORT's ``infile=tf``.
    if ref.statement in ("infile", "proc_cimport") and ref.binds is not None:
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
    # The object is named as SAS reads it, case and suffix kept, from the
    # unnormalised path: on AIX, Linux and object storage `Sales.CSV` and
    # `sales` are two files, and a file reader takes the format from the suffix.
    # A SAS dataset keeps only its stem — its reader adds `.sas7bdat` — and
    # either way the target table is named from the stem (`_target_for`).
    name = _basename(ref.effective_path)
    return HydrationSource(
        kind=kind,
        locator=_directory(ref.effective_path),
        object_name=_stem(name) if kind is SourceKind.SAS_DATASET else name,
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
    if source.kind in _FILE_KINDS:
        # A file is named by its stem: `Sales.CSV` lands as `sales`, the name
        # its dataset would have had (render lowercases every part).
        table_name = _stem(table_name)
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
    code_filerefs: frozenset[str] = frozenset(),
    transport_filerefs: frozenset[str] = frozenset(),
    env_used: frozenset[tuple[str, str]] = frozenset(),
) -> list[tuple[HydrationSource, tuple[str, ...]]]:
    """Every hydratable source one file's refs name, in reading order, each with
    the blockers only its kind can have.

    *named* holds the database LIBNAMEs — ``(libref, options)`` — whose tables
    the corpus names: those are planned table by table, so the LIBNAME's own
    schema-level item, which stands in when no table is known, is left out.
    *code_filerefs* are the filerefs only ``%INCLUDE`` reads
    (:func:`_code_filerefs`): their FILENAMEs name SAS source, not data.
    *transport_filerefs* are the ones only PROC CIMPORT reads: their FILENAMEs
    name a transport file, planned blocked as one named by INFILE= is.
    *env_used* are the environment variables a read goes through,
    ``(name, place)`` (:func:`_env_in_use`): any other is just a variable.
    """
    sources: list[tuple[HydrationSource, tuple[str, ...]]] = [
        (_oracle_source(ref, source_id), _engine_blocker(ref.engine))
        for ref in engine_refs
        if (ref.binds, ref.options) not in named
    ]
    for path_ref in path_refs:
        if path_ref.statement == "filename" and path_ref.binds in code_filerefs:
            continue
        if (
            path_ref.statement == "setenv"
            and (path_ref.binds, path_ref.effective_path) not in env_used
        ):
            continue
        source = _path_source(path_ref, source_id)
        if source is not None:
            transport = path_ref.statement == "proc_cimport" or (
                path_ref.statement == "filename" and path_ref.binds in transport_filerefs
            )
            sources.append((source, _TRANSPORT_BLOCKER if transport else ()))
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
    datasets: Sequence[Sequence["SasDatasetRef"]] = (),
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
        Each of the same chunks' ``metadata.dataset_refs``, chunk by chunk: a
        directory LIBNAME whose datasets are read here is planned per dataset
        (see :func:`build_corpus_plan`).
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


def _code_filerefs(
    by_source: Mapping[str, tuple[Sequence["SasEngineRef"], Sequence["SasPathRef"]]],
) -> frozenset[str]:
    """The filerefs the corpus only ``%INCLUDE``s through.

    ``filename src '/code/macros'; %include src(util);`` names a directory of
    SAS source: more code to convert, not data to load, as an ``%INCLUDE`` of
    a quoted path is not. A fileref an INFILE or FILE statement also uses
    holds data, and stays.
    """
    return _filerefs_only(by_source, "include")


def _filerefs_only(
    by_source: Mapping[str, tuple[Sequence["SasEngineRef"], Sequence["SasPathRef"]]],
    statement: str,
) -> frozenset[str]:
    """The filerefs the corpus uses through *statement* and nothing else."""
    only: set[str] = set()
    other: set[str] = set()
    for _, path_refs in by_source.values():
        for ref in path_refs:
            if not ref.binds or ref.statement == "filename":
                continue
            (only if ref.statement == statement else other).add(ref.binds)
    return frozenset(only - other)


#: A transport file is CPORT's format: no reader here opens one.
_TRANSPORT_BLOCKER = (
    "a transport file PROC CIMPORT imports: no reader here opens one — import "
    "it with PROC CIMPORT in SAS, then load the data sets it creates",
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


def _env_library(ref: "SasPathRef") -> bool:
    """Whether *ref* is an environment variable naming a directory: on UNIX,
    a libref wherever no LIBNAME assigns the name (``x setenv EDUC
    /users/mydir/educ;`` lets ``set educ.class;`` read that library)."""
    return (
        ref.statement == "setenv"
        and bool(ref.binds)
        and str(ref.location) == "filesystem"
        and "." not in _basename(ref.path)
    )


def _concatenated(ref: "SasPathRef") -> bool:
    """Whether *ref* is one library of a LIBNAME concatenation (``libname x
    ('/a' v9);``): a directory, a libref, or a library of another engine,
    which takes its place in the search order all the same."""
    return (
        ref.statement == "libname"
        and getattr(ref, "concat_index", None) is not None
        and bool(ref.binds)
        and not (ref.binds or "").startswith("&")
    )


#: A directory library: its libref and directory, as a library item has them.
#: In a concatenation's search order, a directory of ``""`` stands for a
#: library named by a libref the corpus never assigns (or one spelled through
#: a macro variable nothing resolved): its place is somebody else's to say.
_Library = tuple[str | None, str]


#: The writes that take effect before their step ends: PROC DATASETS's CHANGE
#: and a library COPY make their tables for the statements after them.
_IMMEDIATE_WRITES = frozenset({"change", "copy"})


class _Members(NamedTuple):
    """What :func:`_library_members` finds."""

    #: Each directory library's datasets to load, as ``(member, reading
    #: file)``, in first-read order.
    named: dict[_Library, list[tuple[str, str | None]]]
    #: The libraries a list (``raw.sales_:``) is read from.
    listed: set[_Library]
    #: Some read names its dataset through a libref nothing resolved
    #: (``&lib..x``, ``&ds``): any library may hold what it reads.
    unknown: bool
    #: Each libref the corpus reads from but never assigns, with the datasets
    #: read, as the job names them, and the first file to read one, in
    #: first-read order. A concatenation that searches the library first
    #: reads it under its own libref.
    unassigned: dict[str, tuple[list[str], str | None]]
    #: Directories a concatenation searches after the first, whose datasets
    #: the job reads from the first: their own items are not needed.
    covered: set[_Library]
    #: For a dataset read through a concatenation, ``(directory, member)``,
    #: the note naming the libraries SAS would search after the one planned.
    notes: dict[tuple[str, str], str]


#: The librefs SAS assigns itself: in a SAS session they need no LIBNAME.
#: (Maps only with SAS/GRAPH, but a read from it is not data a job loads.)
_SAS_LIBRARIES = frozenset({"work", "sashelp", "sasuser", "maps", "mapsgfk", "mapssas"})


def _assigned_librefs(
    by_source: Mapping[str, tuple[Sequence["SasEngineRef"], Sequence["SasPathRef"]]],
) -> frozenset[str] | None:
    """Every libref a LIBNAME in the corpus assigns, by path or engine — or
    ``None`` when some LIBNAME's libref is spelled through a macro variable
    nothing resolved, which could be any of them."""
    librefs: set[str] = set()
    for engine_refs, path_refs in by_source.values():
        librefs.update(ref.binds for ref in engine_refs if ref.binds)
        for ref in path_refs:
            # An environment variable is a libref on UNIX too (_env_library).
            if ref.statement not in ("libname", "setenv") or not ref.binds:
                continue
            if ref.binds.startswith("&"):
                return None
            librefs.add(ref.binds)
    return frozenset(librefs)


def _unassigned_blocker(libref: str, read: Sequence[str]) -> str:
    """Why an item for a library nothing in the corpus assigns is blocked.
    *read* names the datasets as the job reads them: through the library's
    own libref, or a concatenation's that searches it first."""

    def shown_as(name: str) -> str:
        through = name.partition(".")[0]
        if through == libref:
            return name
        return f"{name} (the concatenation {through} searches {libref} first)"

    shown = ", ".join(map(shown_as, read[:5]))
    more = f" and {len(read) - 5} more" if len(read) > 5 else ""
    return (
        f"library '{libref}' is never assigned in the corpus, yet the job reads "
        f"{shown}{more} — its LIBNAME is elsewhere: an autoexec.sas (on UNIX, "
        f"in the current, home or !SASROOT directory), a configuration file, or "
        f"an environment variable {libref.upper()} holding the directory; add "
        f"it to the corpus or point the plan at that library"
    )


def _library_members(
    by_source: Mapping[str, tuple[Sequence["SasEngineRef"], Sequence["SasPathRef"]]],
    datasets: Mapping[str, Sequence[Sequence["SasDatasetRef"]]],
) -> _Members:
    """The datasets to load from each directory library.

    A dataset the corpus reads, or updates, through a libref a directory
    LIBNAME binds is a member of that directory to load — ``set raw.customers;``
    after ``libname raw '/data/raw';`` reads ``/data/raw/customers.sas7bdat`` —
    unless the job made it first. *datasets* holds each file's chunks' dataset
    references, in order, because the order decides that:

    - an earlier step of the same file, or a step of another file, that creates
      the dataset makes it the job's own (another file is taken to run first,
      as the batcher links it);
    - the step that reads it creating it too does not: ``data raw.a; set
      raw.a;`` and an in-place ``proc sort data=raw.a;`` need the table that
      was there before. A step writes when it ends — save for a rename or a
      copy, which takes effect for the statements after it: ``proc datasets;
      change old=new; modify new;`` modifies the ``new`` it just made.

    The LIBNAME in force is the latest one before the read in corpus order, as
    SAS would have run them: ``libname raw '/data/2025'; set raw.x; libname raw
    '/data/2026'; set raw.y;`` reads ``/data/2025/x`` and ``/data/2026/y``.
    Records that say where they stand in their file (an ``order``, as the
    inventory's do) are bound in place; without one, a file's LIBNAMEs are
    taken before its reads. A parameter's template in a ``%MACRO`` body reads nothing until a call names
    it, and each call's reading is its own reference
    (``chunker.resolve_corpus_datasets``).

    A concatenation (``libname x ('/a' '/b');``) is one library SAS searches
    in order, reading a member from the first library that holds it. Which
    one holds it only a listing can tell, so a member is planned from the
    first, with a note naming the rest, which need no items of their own
    (``covered``). A library a concatenation names by its libref is taken as
    that libref is assigned then: ``libname all (v9 '/x');`` reads
    ``all.m`` from v9's directory first.
    """
    writers: dict[str, set[str]] = {}
    for source_id, chunks in datasets.items():
        for refs in chunks:
            for ref in refs:
                if str(ref.role) == "write":
                    writers.setdefault(ref.name, set()).add(source_id)
    assigned = _assigned_librefs(by_source)
    libraries = _Libraries()
    found = _Members({}, set(), False, {}, set(), {})
    seen: set[tuple[str, str]] = set()
    for source_id in dict.fromkeys([*by_source, *datasets]):
        libnames = sorted(
            (
                ref
                for ref in by_source.get(source_id, ((), ()))[1]
                if (_directory_library(ref) or _env_library(ref) or _concatenated(ref))
                and ref.binds
            ),
            key=lambda ref: (_position(ref) is not None, _position(ref) or 0),
        )
        made_here: set[str] = set()  # by this file's earlier steps
        for refs in datasets.get(source_id, ()):
            # Every LIBNAME that stands before this chunk binds its libref now.
            start = min((p for r in refs if (p := _position(r)) is not None), default=None)
            while libnames and (
                start is None
                or (position := _position(libnames[0])) is None
                or position < start
            ):
                libraries.bind(libnames.pop(0))
            made_now: set[str] = set()  # renamed or copied by this step so far
            for ref in refs:
                if str(ref.role) == "write" and ref.via in _IMMEDIATE_WRITES:
                    made_now.add(ref.name)
                if str(ref.role) not in ("read", "update") or ref.param is not None:
                    continue
                if ref.name.startswith("&"):
                    found = found._replace(unknown=True)
                    continue
                libref, _, member = ref.name.partition(".")
                if not member:
                    continue
                made = ref.name in made_here | made_now or bool(
                    writers.get(ref.name, set()) - {source_id}
                )
                places = libraries.search(libref)
                outside = (places[0][0] or "") if places and not places[0][1] else None
                if not places or outside is not None:
                    # Searched first: a library no LIBNAME in the corpus
                    # assigns. SAS assigned it some other way, and its data
                    # is needed.
                    first = libref if outside is None else outside
                    if (
                        assigned is not None
                        and first not in assigned
                        and first not in _SAS_LIBRARIES
                        and not first.startswith("&")
                        and not made
                    ):
                        names, _ = found.unassigned.setdefault(first, ([], source_id or None))
                        if ref.name not in names:
                            names.append(ref.name)
                    elif outside is not None and not made:
                        # A concatenation that searches a library spelled
                        # through a macro variable first: any library may
                        # hold what it reads.
                        found = found._replace(unknown=True)
                    continue
                library = places[0]
                if ref.pattern:
                    found.listed.update(place for place in places if place[1])
                    continue
                if made:
                    continue
                if (library[1], member) not in seen:
                    seen.add((library[1], member))
                    found.named.setdefault(library, []).append((member, source_id or None))
                    if len(places) > 1:
                        found.notes[(library[1], member)] = _concatenation_note(
                            libref, places, member
                        )
                found.covered.update(place for place in places[1:] if place[1])
            made_here.update(ref.name for ref in refs if str(ref.role) == "write")
        for ref in libnames:  # after the file's last read: in force for the next
            libraries.bind(ref)
    return found


class _Libraries:
    """The libraries in force while the corpus is walked: each libref's
    directories in the order SAS searches them — one, or a concatenation's."""

    def __init__(self) -> None:
        self.bound: dict[str, tuple[_Library, ...]] = {}
        # Environment variables: librefs no LIBNAME assigns.
        self.env: dict[str, tuple[_Library, ...]] = {}

    def search(self, libref: str) -> tuple[_Library, ...]:
        """Where SAS looks for a member of *libref*, in order. A LIBNAME wins
        over an environment variable of the same name."""
        return self.bound.get(libref) or self.env.get(libref) or ()

    def bind(self, ref: "SasPathRef") -> None:
        """Put the LIBNAME, or environment variable, *ref* in force. A later
        library of a concatenation joins the ones before it."""
        if not ref.binds:
            return
        if ref.statement == "setenv":
            self.env[ref.binds] = ((ref.binds, ref.effective_path),)
            return
        places = self._places(ref)
        if getattr(ref, "concat_index", None):
            places = self.bound.get(ref.binds, ()) + places
        self.bound[ref.binds] = tuple(dict.fromkeys(places))

    def _places(self, ref: "SasPathRef") -> tuple[_Library, ...]:
        """The directories one library of a LIBNAME adds to the search."""
        if str(ref.location) == "libref":
            # As that libref is assigned now: assigning it again later does
            # not change the concatenation (SAS 9.4 Global Statements, LIBNAME
            # Statement: Rules for Library Concatenation). WORK and the other
            # libraries SAS assigns hold no data a job loads.
            name = ref.effective_path.lower()
            if name in _SAS_LIBRARIES:
                return ()
            return self.search(name) or ((name, ""),)
        if _directory_library(ref):
            return ((ref.binds, ref.effective_path),)
        return ()  # a library of another engine holds no member files


def _concatenation_note(libref: str, places: Sequence[_Library], member: str) -> str:
    """The note on a dataset read through the concatenation *libref*."""

    def where(place: _Library) -> str:
        name, directory = place
        if directory:
            return directory
        return f"{name} (unresolved)" if (name or "").startswith("&") else (
            f"library {name} (assigned outside the corpus)"
        )

    return (
        f"read as {libref}.{member} from a concatenation SAS searches in the order "
        f"{', '.join(map(where, places))}: it reads {member} from the first that "
        f"holds it — planned from {places[0][1]}; if {member} is not there, point "
        f"this item at the one that holds it"
    )


def _env_in_use(
    by_source: Mapping[str, tuple[Sequence["SasEngineRef"], Sequence["SasPathRef"]]],
    members: _Members,
) -> frozenset[tuple[str, str]]:
    """The environment variables a read goes through, as ``(name, place)``:
    as a libref (a library whose datasets are read) or as a fileref no
    FILENAME binds (``infile ined;``). Any other variable is just a variable —
    ``x setenv TMPDIR /tmp;`` names no data."""
    libraries = set(members.named) | members.listed
    filenames: set[str] = set()
    through: set[str] = set()
    for _, path_refs in by_source.values():
        for ref in path_refs:
            if not ref.binds:
                continue
            if ref.statement == "filename":
                filenames.add(ref.binds)
            elif ref.statement not in ("libname", "setenv"):
                through.add(ref.binds)
    used: set[tuple[str, str]] = set()
    for _, path_refs in by_source.values():
        for ref in path_refs:
            if ref.statement != "setenv" or not ref.binds:
                continue
            place = (ref.binds, ref.effective_path)
            if place in libraries or (ref.binds in through and ref.binds not in filenames):
                used.add(place)
    return frozenset(used)


def _position(ref: object) -> int | None:
    """Where *ref* stands in its file, when its record says (``order``)."""
    return getattr(ref, "order", None)


def build_corpus_plan(
    by_source: Mapping[str, tuple[Sequence["SasEngineRef"], Sequence["SasPathRef"]]],
    *,
    db_tables: Mapping[str, Sequence["SasDbTableRef"]] | None = None,
    datasets: Mapping[str, Sequence[Sequence["SasDatasetRef"]]] | None = None,
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

    *datasets* maps them to each chunk's ``metadata.dataset_refs``, chunk by
    chunk in order, from a corpus resolved with
    ``chunker.resolve_corpus_datasets`` (so each macro call carries what its
    macro reads). It is how a directory LIBNAME is planned per dataset rather
    than as one blocked library item (see :func:`_library_members`): each
    member read is a ``sas7bdat`` item, owned by the first file that reads it.
    The library item stays where no member is named, where a list is read
    from it, and while some read names its library through a macro variable
    nothing resolved — so no dataset the job needs drops out of the plan
    unannounced.
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
    members = _library_members(by_source, datasets or {})
    if members.unassigned and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"build_corpus_plan: unassigned libraries {sorted(members.unassigned)}")
    expanded: set[_Library] = set()
    code_filerefs = _code_filerefs(by_source)
    transport_filerefs = _filerefs_only(by_source, "proc_cimport")
    env_used = _env_in_use(by_source, members)

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
            code_filerefs=code_filerefs,
            transport_filerefs=transport_filerefs,
            env_used=env_used,
        ):
            library = (source.libref, source.locator)
            if (
                source.kind is SourceKind.FILE
                and not source.object_name
                and (library in members.named or library in members.covered)
            ):
                # A directory whose datasets are named: one item per dataset,
                # where the library is first declared. One a concatenation
                # searches after the directory its reads are planned from
                # needs none.
                if library in members.named and library not in expanded:
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
                        for member, reader in members.named[library]
                    ]
                if library not in members.listed and not members.unknown:
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
        if source.kind is SourceKind.SAS_DATASET and (
            concatenation := members.notes.get((source.locator, source.object_name))
        ):
            notes += (concatenation,)

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

    # A library the corpus reads from but never assigns: one item each, so
    # its data does not drop out of the plan unannounced. Only a SAS session
    # that assigns it can say where it is.
    for libref, (read, reader) in members.unassigned.items():
        items.append(
            HydrationItem(
                source=HydrationSource(
                    kind=SourceKind.SAS_SESSION, libref=libref, source_id=reader
                ),
                target_table=UNRESOLVED_TARGET,
                strategy_reason="not planned: the library is assigned outside the corpus",
                blockers=(_unassigned_blocker(libref, read),),
            )
        )

    logger.info(
        f"build_corpus_plan: {len(sources)} source(s) across "
        f"{len(by_source.keys() | db_tables.keys())} file(s) "
        f"-> {len(items)} item(s) (probe={'yes' if probe else 'no'})"
    )
    return HydrationPlan(run_date=run_date, items=items)
