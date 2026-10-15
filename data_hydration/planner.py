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

import dataclasses
import logging
import posixpath
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple

from pydantic import BaseModel

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
    # listing, PROC CPORT's transport file) and the code it runs (the SAS an
    # %INCLUDE pulls in, the Python PROC PYTHON's INFILE= names) are no data
    # to load.
    if ref.statement in {
        "include", "proc_python", "ods", "sasautos", "file", "proc_export", "proc_cport",
        "printto", "cd",
    }:
        return None
    # ``infile in;`` reads through a fileref, at the path its FILENAME names
    # (chunker.metadata.resolve_filerefs): that FILENAME's own reference is
    # the source, planned once. So do PROC CIMPORT's ``infile=tf`` and PROC
    # IMPORT's ``datafile=in``.
    if ref.statement in ("infile", "proc_cimport", "proc_import") and ref.binds is not None:
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
    # object name, which is what ``_library_blocker`` reads. A wildcard
    # (``'/data/*/*'``) names files, whatever its last part.
    if (
        kind is SourceKind.FILE
        and "." not in _basename(ref.path)
        and not _WILDCARD_RE.search(_basename(ref.path))
    ):
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


def _quoted_spelling(ref: "SasDatasetRef") -> str:
    """The path a data set named by its path (``set '/Data/Prod/Master';``)
    is read from, quotes and data set options dropped: as written, case kept,
    unless a double-quoted spelling holds a ``&``, which only the resolved
    name can expand."""
    raw = (ref.raw or ref.name).strip()
    quote = raw[:1]
    end = raw.find(quote, 1) if quote in ("'", '"') else -1
    if end > 0 and ("&" not in raw[:end] or quote == "'"):
        return raw[1:end].strip()
    return ref.name.strip().strip("'")


def _member_of(path: str) -> str:
    """The data set a path names: its last part, less the ``.sas7bdat`` SAS
    gives a data file."""
    name = _basename(path)
    return name[: -len(_SAS_DATA_SUFFIX)] if name.lower().endswith(_SAS_DATA_SUFFIX) else name


def _absolute(path: str) -> bool:
    """Whether *path* is a full path: from the root (``/data``), or, on a
    Windows server, a drive (``D:\\data``) or a share (``\\\\host\\share``)."""
    return path.startswith(("/", "\\\\")) or (
        len(path) > 2 and path[1] == ":" and path[2] in "/\\"
    )


def _full_path(source: HydrationSource) -> str:
    """*source*'s place in one path: its directory and object, or either."""
    if source.locator and source.object_name:
        return f"{source.locator}/{source.object_name}"
    return source.locator or source.object_name


def _relative(path: str) -> bool:
    """Whether *path* is relative: SAS reads it from its current directory.
    One the server completes, or spelled through a macro variable, is not
    taken for one."""
    return bool(path) and not _absolute(path) and not _server_parts(path) and "&" not in path


def _relative_blocker(source: HydrationSource) -> tuple[str, ...]:
    """A blocker for a filesystem source left relative: no cd before it in
    the corpus says which directory SAS reads it from."""
    if source.kind not in (SourceKind.FILE, SourceKind.SAS_DATASET, SourceKind.SPDE):
        return ()
    path = _full_path(source)
    if not _relative(path):
        return ()
    return (
        f"'{path}' is a relative path: SAS reads it from its current directory, "
        f"which no cd before it sets (x cd, CALL SYSTEM('cd ...'), %SYSEXEC cd) "
        f"— point this item at the directory it names",
    )


# An environment variable in a UNIX path: $HOME, $PROJ.
_ENV_VAR_RE = re.compile(r"\$([A-Za-z_]\w*)")


def _server_parts(path: str) -> list[str]:
    """What in *path* the SAS server fills in when it reads it, each said in
    words: on UNIX, ``~/`` is ``$HOME/``, ``~name/`` that user's home
    directory, ``!sasroot`` the directory SAS is installed in — each only at
    the start — and ``$VARIABLE`` an environment variable (SAS 9.4 Companion
    for UNIX Environments, Character Substitutions in Pathnames)."""
    parts: list[str] = []
    if path.startswith("~"):
        user = path[1:].replace("\\", "/").split("/", 1)[0]
        parts.append(
            f"~{user} is {user}'s home directory"
            if user
            else "~ is the home directory of the user SAS runs as"
        )
    elif path.lower().startswith("!sasroot"):
        parts.append("!SASROOT is the directory SAS is installed in")
    parts += [
        f"${name} is an environment variable of the SAS server"
        for name in _ENV_VAR_RE.findall(path)
    ]
    return parts


# A UNIX wildcard in an input path: *, ? or a [ ] set (SAS 9.4 Companion for
# UNIX Environments, Using Wildcards in Pathnames).
_WILDCARD_RE = re.compile(r"[*?]|\[[^\]]*\]")


def _wildcard_blocker(source: HydrationSource) -> tuple[str, ...]:
    """A blocker for a file read through a wildcard: ``filename wild
    '/data/in/wild*.dat';`` reads every file the pattern matches, one after
    another — ``*`` one or more characters, ``?`` one, ``[ ]`` one of a set
    — which only a listing of the server's directory can tell."""
    if source.kind is not SourceKind.FILE:
        return ()
    path = _full_path(source)
    if not _WILDCARD_RE.search(path):
        return ()
    return (
        f"'{path}' is a wildcard: SAS reads every file it matches, one after "
        f"another — list them on the SAS server and hydrate each, or point "
        f"this item at one",
    )


def _server_blocker(source: HydrationSource) -> tuple[str, ...]:
    """A blocker for a filesystem source whose path the SAS server completes
    (:func:`_server_parts`): the plan cannot know where it points."""
    if source.kind not in (SourceKind.FILE, SourceKind.SAS_DATASET, SourceKind.SPDE):
        return ()
    path = _full_path(source)
    parts = _server_parts(path)
    if not parts:
        return ()
    return (
        f"'{path}' is completed by the SAS server: {'; '.join(parts)} — point "
        f"this item at the place it names there",
    )


def _quoted_dataset(path: str, reader: str | None) -> tuple[HydrationSource, tuple[str, ...]]:
    """The source for a data set the job reads by its path, and why it is
    blocked, if it is.

    ``set '/u/2011/budgets/may';`` reads the SAS data file
    ``/u/2011/budgets/may.sas7bdat`` — on UNIX "if you do not specify an
    extension, SAS uses .sas7bdat" — so the item loads it as a SAS data set.
    One with another extension names no SAS data file. A relative path is
    the current directory's (:class:`_WorkingDirectories`).
    """
    member = _member_of(path)
    source = HydrationSource(
        kind=SourceKind.SAS_DATASET,
        locator=_directory(path),
        object_name=member,
        has_macro_ref="&" in path,
        source_id=reader,
    )
    blockers: list[str] = []
    if "." in member:
        blockers.append(
            f"'{path}' names no SAS data file (.sas7bdat) — check what the "
            f"job reads there"
        )
    return source, tuple(blockers)


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
    unloaded_filerefs: frozenset[str] = frozenset(),
    transport_filerefs: frozenset[str] = frozenset(),
    env_used: frozenset[tuple[str, str]] = frozenset(),
) -> list[tuple[HydrationSource, tuple[str, ...]]]:
    """Every hydratable source one file's refs name, in reading order, each with
    the blockers only its kind can have.

    *named* holds the database LIBNAMEs — ``(libref, options)`` — whose tables
    the corpus names: those are planned table by table, so the LIBNAME's own
    schema-level item, which stands in when no table is known, is left out.
    *unloaded_filerefs* are the filerefs the corpus only runs code or writes
    through (:func:`_unloaded_filerefs`): their FILENAMEs name no data to load.
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
        if path_ref.statement == "filename" and path_ref.binds in unloaded_filerefs:
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


def _unloaded_filerefs(
    by_source: Mapping[str, tuple[Sequence["SasEngineRef"], Sequence["SasPathRef"]]],
) -> frozenset[str]:
    """The filerefs the corpus only runs code or writes through: ``%INCLUDE``
    and PROC PYTHON's ``INFILE=``, and FILE, PROC EXPORT's ``OUTFILE=`` and
    PROC CPORT's ``FILE=``.

    ``filename src '/code/macros'; %include src(util);`` names a directory of
    SAS source, and ``filename py '/py/prep.py'; proc python infile=py;`` a
    Python script: more code to convert, not data to load, as either by a
    quoted path is not. ``filename rpt '/out/r.txt'; file rpt;`` names what
    the job writes, as ``file '/out/r.txt';`` does, and a file it writes and
    then includes is code it generates. A fileref an INFILE or PROC IMPORT
    also reads holds data, and stays.
    """
    return _filerefs_only(
        by_source, "include", "proc_python", "file", "proc_export", "proc_cport"
    )


def _filerefs_only(
    by_source: Mapping[str, tuple[Sequence["SasEngineRef"], Sequence["SasPathRef"]]],
    *statements: str,
) -> frozenset[str]:
    """The filerefs the corpus uses through *statements* and nothing else."""
    only: set[str] = set()
    other: set[str] = set()
    for _, path_refs in by_source.values():
        for ref in path_refs:
            if not ref.binds or ref.statement == "filename":
                continue
            (only if ref.statement in statements else other).add(ref.binds)
    return frozenset(only - other)


def _pipe_blocker(command: str) -> str:
    """Why an item for a read of a command's output is blocked. The paths
    the command names, if any, are said: they are often the data itself
    (``gunzip -c /data/x.gz``)."""
    named = [
        token.strip("'\"|;")
        for token in command.split()
        if token.strip("'\"|;").startswith(("/", "~", "$"))
    ]
    names = f" (it names {', '.join(named[:3])})" if named else ""
    return (
        f"read from a shell command: SAS runs '{command}' on its server and reads "
        f"what it prints{names} — load the data the command reads, or run it "
        f"there and load its output"
    )


#: A transport file is CPORT's format: no reader here opens one.
_TRANSPORT_BLOCKER = (
    "a transport file PROC CIMPORT imports: no reader here opens one — import "
    "it with PROC CIMPORT in SAS, then load the data sets it creates",
)


#: LIBNAME engines whose directory holds each dataset as ``<member>.sas7bdat``:
#: the default, written or not.
_BASE_ENGINES = frozenset({"", "base", "v9", "v8", "v7"})


def _directory_library(ref: "SasPathRef") -> bool:
    """Whether *ref* is a LIBNAME binding a directory of SAS datasets: in the
    session (``options user='/dir';`` binds the User libref so too), or, a
    clause of a CREATE VIEW's USING list, for the view."""
    return (
        ref.statement in ("libname", "using_libname", "user")
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
#: and AGE, which rename, and a library COPY make their tables for the
#: statements after them.
_IMMEDIATE_WRITES = frozenset({"change", "age", "copy"})


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
    #: Each data set read by its path (``set '/data/prod/master';``), as
    #: written, with the first file to read it, in first-read order.
    quoted: dict[str, str | None]


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
        # A view's own (statement using_libname) is assigned only in the view.
        librefs.update(
            ref.binds
            for ref in engine_refs
            if ref.binds and getattr(ref, "statement", "libname") == "libname"
        )
        for ref in path_refs:
            # An environment variable is a libref on UNIX too (_env_library),
            # and USER= assigns the User libref.
            if ref.statement not in ("libname", "setenv", "user") or not ref.binds:
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


def _unassigned_filerefs(
    by_source: Mapping[str, tuple[Sequence["SasEngineRef"], Sequence["SasPathRef"]]],
    directories: "_WorkingDirectories",
) -> dict[str, tuple[list[str], str | None, str | None]]:
    """Each fileref an INFILE reads through that no FILENAME or environment
    variable in the corpus assigns, in first-read order: the reads as the job
    spells them, the first file to read one, and for a bare word the file SAS
    on UNIX reads when no fileref of that name exists — the one
    :func:`chunker.metadata.resolve_filerefs` recorded, in the current
    directory a cd set where one did. A fileref a FILENAME assigns to a device
    and no place (``filename t temp;``) is assigned: its reads carry the
    device."""
    assigned = {
        ref.binds
        for _, path_refs in by_source.values()
        for ref in path_refs
        if ref.statement in ("filename", "setenv") and ref.binds
    }
    found: dict[str, tuple[list[str], str | None, str | None]] = {}
    for source_id, (_, path_refs) in by_source.items():
        for ref in path_refs:
            if (
                ref.statement != "infile"
                or str(ref.location) != "fileref"
                or ref.device
                or not ref.binds
                or ref.binds in assigned
            ):
                continue
            read, reader, place = found.setdefault(ref.binds, ([], source_id or None, None))
            if ref.raw not in read:
                read.append(ref.raw)
            file = ref.resolved_path
            if place is None and file and "(" not in file and "&" not in file:
                cwd = directories.at(source_id, _position(ref))
                found[ref.binds] = (read, reader, file if cwd is None else _under(cwd, file))
    return found


def _unassigned_fileref_blocker(fileref: str, read: Sequence[str], place: str | None) -> str:
    """Why an item for a fileref nothing in the corpus assigns is blocked.
    *read* names the reads as the job spells them; *place* is the file SAS
    on UNIX reads for a bare word when no fileref of that name exists."""
    shown = ", ".join(f"infile {r}" for r in read[:5])
    more = f" and {len(read) - 5} more" if len(read) > 5 else ""
    if place is None:
        otherwise = ""
    elif place.startswith("/"):
        otherwise = f"; with no fileref of that name, SAS on UNIX reads {place}"
    else:
        otherwise = (
            f"; with no fileref of that name, SAS on UNIX reads {place} from its "
            f"current directory, which no cd in the corpus sets"
        )
    return (
        f"fileref '{fileref}' is never assigned in the corpus, yet the job reads "
        f"through it ({shown}{more}) — its FILENAME is elsewhere: an autoexec.sas, "
        f"a configuration file, or an environment variable {fileref.upper()} holding "
        f"the file's path{otherwise}; add it to the corpus or point the plan at "
        f"that file"
    )


def _library_members(
    by_source: Mapping[str, tuple[Sequence["SasEngineRef"], Sequence["SasPathRef"]]],
    datasets: Mapping[str, Sequence[Sequence["SasDatasetRef"]]],
    *,
    directories: "_WorkingDirectories | None" = None,
    relative: dict[str, tuple[str, str]] | None = None,
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
    taken before its reads, and a view's USING LIBNAME is in force for the
    whole file, behind them. A parameter's template in a ``%MACRO`` body reads nothing until a call names
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
    found = _Members({}, set(), False, {}, set(), {}, {})
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
        # A view's own LIBNAMEs (a CREATE VIEW's USING list) are in force for
        # its chunk alone. With no position, one cannot be told to its chunk:
        # it is in force for the whole file, behind the file's other LIBNAMEs,
        # and for no file after.
        unplaced = [
            ref for ref in libnames if ref.statement == "using_libname" and _position(ref) is None
        ]
        file_views = libraries.scoped(unplaced)
        put = {libref: libraries.bound.get(libref) for libref in file_views}
        for refs in datasets.get(source_id, ()):
            # Every LIBNAME that stands before this chunk binds its libref now.
            start = min((p for r in refs if (p := _position(r)) is not None), default=None)
            views: list["SasPathRef"] = []  # a CREATE VIEW's own, for this chunk
            while libnames and (
                start is None
                or (position := _position(libnames[0])) is None
                or position < start
            ):
                ref = libnames.pop(0)
                if ref.statement != "using_libname":
                    libraries.bind(ref)
                elif _position(ref) is not None:  # unplaced, it is in file_views
                    views.append(ref)
            hidden = libraries.scoped(views)
            made_now: set[str] = set()  # renamed or copied by this step so far
            for ref in refs:
                # One inside the chunk binds the reads after it: a %MACRO body
                # or a PROC SQL holds several steps.
                here = _position(ref)
                while (
                    here is not None
                    and libnames
                    and (position := _position(libnames[0])) is not None
                    and position < here
                ):
                    later = libnames.pop(0)
                    if later.statement != "using_libname":
                        libraries.bind(later)
                    else:
                        for libref, was in libraries.scoped([later]).items():
                            hidden.setdefault(libref, was)
                if str(ref.role) == "write" and ref.via in _IMMEDIATE_WRITES:
                    made_now.add(ref.name)
                if str(ref.role) not in ("read", "update") or ref.param is not None:
                    continue
                if ref.name.startswith("&"):
                    found = found._replace(unknown=True)
                    continue
                made = ref.name in made_here | made_now or bool(
                    writers.get(ref.name, set()) - {source_id}
                )
                if ref.name.startswith("'"):
                    # Named by its path, through no libref: its own item, a
                    # relative path read from the current directory.
                    path = _quoted_spelling(ref)
                    cwd = directories.at(source_id, _position(ref)) if directories else None
                    if cwd is not None and _relative(path):
                        if relative is not None:
                            relative.setdefault(_under(cwd, path), (path, cwd))
                        path = _under(cwd, path)
                    key = (_directory(path), _member_of(path).lower())
                    if not made and key not in seen:
                        seen.add(key)
                        found.quoted.setdefault(path, source_id or None)
                    continue
                libref, _, member = ref.name.partition(".")
                if not member:
                    continue
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
                if member[:1] in ("'", '"') and member.endswith("n"):
                    # A name literal, 'my data'n: its file is my data.sas7bdat.
                    member = member[1:-2]
                if (library[1], member) not in seen:
                    seen.add((library[1], member))
                    found.named.setdefault(library, []).append((member, source_id or None))
                    if len(places) > 1:
                        found.notes[(library[1], member)] = _concatenation_note(
                            libref, places, member
                        )
                found.covered.update(place for place in places[1:] if place[1])
            libraries.restore(hidden)
            made_here.update(ref.name for ref in refs if str(ref.role) == "write")
        for ref in libnames:  # after the file's last read: in force for the next
            if ref.statement != "using_libname":
                libraries.bind(ref)
        # What the unplaced views hid comes back, save where a LIBNAME of the
        # file has replaced them since.
        libraries.restore(
            {lib: was for lib, was in file_views.items() if libraries.bound.get(lib) == put[lib]}
        )
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

    def scoped(self, refs: Sequence["SasPathRef"]) -> dict[str, tuple[_Library, ...] | None]:
        """Put *refs* — the LIBNAME clauses of a CREATE VIEW's USING list — in
        force, and return the bindings they hide, for :meth:`restore`."""
        hidden = {ref.binds: self.bound.get(ref.binds) for ref in refs if ref.binds}
        for ref in refs:
            self.bind(ref)
        return hidden

    def restore(self, hidden: Mapping[str, tuple[_Library, ...] | None]) -> None:
        """Put back the bindings :meth:`scoped` hid."""
        for libref, places in hidden.items():
            if places is None:
                self.bound.pop(libref, None)
            else:
                self.bound[libref] = places

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
            elif ref.statement not in ("libname", "setenv", "user"):
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


def _change_directory(current: str | None, target: str) -> str | None:
    """The current directory after ``cd target`` from *current*: *target*
    when it is a full path (or one the server completes), else *target*
    under *current*; ``None`` when that cannot be known."""
    if "&" in target:
        return None
    if not _relative(target):
        return posixpath.normpath(target) if target.startswith("/") else target
    return None if current is None else _under(current, target)


def _under(directory: str, relative: str) -> str:
    """*relative* read from *directory*: ``data/x`` under ``/users/smith`` is
    ``/users/smith/data/x``, ``..`` and ``.`` taken as the shell takes them."""
    return posixpath.normpath(f"{directory.rstrip('/')}/{relative}")


class _WorkingDirectories:
    """SAS's current directory as the corpus runs: the one the last single
    cd before a statement set — ``x cd /users/smith/report;``, ``call
    system('cd ...');`` or ``%sysexec cd ...;`` — kept from one file to the
    next, in corpus order. "The search for any relative (partial) filenames
    during the SAS session now begins in" it (SAS 9.4 Companion for UNIX
    Environments). Before the first, or after a cd it cannot follow, it is
    unknown (``None``): the directory SAS was started in."""

    def __init__(
        self, by_source: Mapping[str, tuple[Sequence["SasEngineRef"], Sequence["SasPathRef"]]]
    ) -> None:
        self._start: dict[str, str | None] = {}
        self._changes: dict[str, list[tuple[int | None, str | None]]] = {}
        current: str | None = None
        for source_id, (_, path_refs) in by_source.items():
            self._start[source_id] = current
            changes: list[tuple[int | None, str | None]] = []
            cds = [ref for ref in path_refs if ref.statement == "cd"]
            for ref in sorted(cds, key=lambda r: (_position(r) is not None, _position(r) or 0)):
                current = _change_directory(current, ref.effective_path)
                changes.append((_position(ref), current))
            self._changes[source_id] = changes

    def at(self, source_id: str, position: int | None) -> str | None:
        """The current directory where *position* stands in *source_id*. A
        record that does not say where it stands sees all of its file's."""
        current = self._start.get(source_id)
        for changed_at, directory in self._changes.get(source_id, ()):
            if position is not None and changed_at is not None and changed_at >= position:
                break
            current = directory
        return current


def _placed(ref: "SasPathRef", place: str) -> "SasPathRef":
    """*ref*, a chunk's record or an inventory row's, read at *place*."""
    if isinstance(ref, BaseModel):
        return ref.model_copy(update={"resolved_path": place})
    return dataclasses.replace(ref, resolved_path=place)  # type: ignore[type-var]


def _from_current_directory(
    by_source: Mapping[str, tuple[Sequence["SasEngineRef"], Sequence["SasPathRef"]]],
    directories: _WorkingDirectories,
    relative: dict[str, tuple[str, str]],
) -> dict[str, tuple[Sequence["SasEngineRef"], Sequence["SasPathRef"]]]:
    """*by_source* with each relative filesystem path read from the current
    directory where it stands, when a cd says which: ``x cd /users/smith;
    infile 'in/x.csv';`` reads ``/users/smith/in/x.csv``. *relative* gains
    each such place, with its spelling and the directory, for the item's
    note. One no cd before it explains stays relative, and its item is
    blocked (:func:`_relative_blocker`)."""
    placed: dict[str, tuple[Sequence["SasEngineRef"], Sequence["SasPathRef"]]] = {}
    for source_id, (engine_refs, path_refs) in by_source.items():
        refs: list["SasPathRef"] = []
        for ref in path_refs:
            spelled = ref.effective_path
            if (
                str(ref.location) == "filesystem"
                and ref.statement not in ("cd", "setenv")
                and _relative(spelled)
                and (cwd := directories.at(source_id, _position(ref))) is not None
            ):
                place = _under(cwd, spelled)
                relative.setdefault(place, (spelled, cwd))
                ref = _placed(ref, place)
            refs.append(ref)
        placed[source_id] = (engine_refs, refs)
    return placed


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
    # Relative paths read from the current directory a cd sets, where one does.
    relative: dict[str, tuple[str, str]] = {}
    directories = _WorkingDirectories(by_source)
    by_source = _from_current_directory(by_source, directories, relative)
    members = _library_members(
        by_source, datasets or {}, directories=directories, relative=relative
    )
    if members.unassigned and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"build_corpus_plan: unassigned libraries {sorted(members.unassigned)}")
    expanded: set[_Library] = set()
    unloaded_filerefs = _unloaded_filerefs(by_source)
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
            unloaded_filerefs=unloaded_filerefs,
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
    # A data set read by its path, through no libref.
    sources += [_quoted_dataset(path, reader) for path, reader in members.quoted.items()]

    items: list[HydrationItem] = []
    seen_tables: set[str] = set()
    for source, extra in sources:
        target, name_blockers = _target_for(source, config, run_date)
        wildcard = _wildcard_blocker(source)
        blockers = [
            *_macro_blocker(source),
            *_server_blocker(source),
            *_relative_blocker(source),
            *wildcard,
            *extra,
        ]
        library = _library_blocker(source)
        blockers += library
        # A library directory has no table name *because* it is a directory, so
        # its rendering failure is already explained; reporting both would say
        # the same thing twice in less useful words. So has a wildcard.
        if not library and not wildcard:
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
        # A relative place: the file's own, or its library directory's.
        spelled = relative.get(_full_path(source)) or relative.get(source.locator)
        if spelled is not None:
            notes += (
                f"'{spelled[0]}' is relative: planned in {spelled[1]}, the "
                f"current directory the last cd before it set",
            )
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

    # A read of a shell command's output — `filename f pipe 'gunzip -c
    # /data/x.gz'; infile f;` or `infile '...' pipe;` — one item per command:
    # only the SAS server can run it.
    pipes: dict[str, tuple[str | None, str | None, bool]] = {}
    for source_id, (_, path_refs) in by_source.items():
        for ref in path_refs:
            if ref.statement == "infile" and str(ref.location) == "pipe":
                pipes.setdefault(
                    ref.effective_path, (source_id or None, ref.binds, ref.has_macro_ref)
                )
    for command, (reader, fileref, has_macro_ref) in pipes.items():
        items.append(
            HydrationItem(
                source=HydrationSource(
                    kind=SourceKind.SAS_SESSION,
                    locator=command,
                    libref=fileref,
                    has_macro_ref=has_macro_ref,
                    source_id=reader,
                ),
                target_table=UNRESOLVED_TARGET,
                strategy_reason="not planned: the data is a shell command's output",
                blockers=(_pipe_blocker(command),),
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

    # A fileref the corpus reads through but never assigns: one item each, as
    # for a library, so the read does not drop out of the plan unannounced.
    for fileref, (read, reader, place) in _unassigned_filerefs(by_source, directories).items():
        items.append(
            HydrationItem(
                source=HydrationSource(
                    kind=SourceKind.SAS_SESSION, libref=fileref, source_id=reader
                ),
                target_table=UNRESOLVED_TARGET,
                strategy_reason="not planned: the fileref is assigned outside the corpus",
                blockers=(_unassigned_fileref_blocker(fileref, read, place),),
            )
        )

    logger.info(
        f"build_corpus_plan: {len(sources)} source(s) across "
        f"{len(by_source.keys() | db_tables.keys())} file(s) "
        f"-> {len(items)} item(s) (probe={'yes' if probe else 'no'})"
    )
    return HydrationPlan(run_date=run_date, items=items)
