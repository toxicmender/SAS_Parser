"""The reference inventory: every place and dataset a SAS corpus names, as rows.

The chunker records four kinds of reference on each chunk —
:class:`~chunker.models.SasPathRef` (a path a statement names),
:class:`~chunker.models.SasDatasetRef` (a SAS dataset a step reads or writes),
:class:`~chunker.models.SasDbTableRef` (a table inside a database) and
:class:`~chunker.models.SasEngineRef` (a database LIBNAME). The inventory flattens
them into one :class:`InventoryRow` each, **resolved or not**: a path still
spelled ``&root/in`` is a row with ``resolved = false``, which is exactly what a
migration needs told about.

The rows live in a Delta table (:func:`write_inventory`), one *run* per write,
so the history of a corpus's references is kept and the latest run is one
``max(run_id)`` away. The planner reads them back (:func:`read_inventory`,
:func:`plan_from_inventory`) without the SAS source or the chunker: the rows
carry everything :func:`~data_hydration.planner.build_corpus_plan` reads, and
``plan_from_inventory(inventory_rows(results))`` is the plan the chunks give.

Two rules the rows keep:

* **Only top-level chunks are read.** A chunk split for size has children whose
  text is part of the parent's; reading both would list each reference twice.
* **No secret is stored.** A connection option keeps its value only when its
  key says where the connection goes or as whom (:data:`_KEPT_OPTIONS`:
  ``path``, ``server``, ``schema``, ``user``, ...); every other value is stored
  as ``<redacted>``, and a connection string (``noprompt="DSN=dw;PWD=x"``) is
  redacted pair by pair the same way. SAS/ACCESS spells a password many ways —
  ``PASSWORD=``, ``PASS=``, ``PW=``, ``PWD=``, ``USING=`` — so a list of the
  safe keys holds where a list of the secret ones leaks. A macro reference
  (``&user_pass.``) is kept: it is no secret, and the planner's blocker names
  it. No reader takes a password from these options —
  :mod:`data_hydration.secrets` resolves it.

Like the planner, this module imports :mod:`chunker` for annotations only, and
pyspark only inside the two functions that touch Delta.

Logger name: ``data_hydration.inventory``.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, field_validator

if TYPE_CHECKING:  # annotations only — never imported at run time
    from .config import HydrationConfig
    from .models import HydrationPlan
    from .partition import SourceProbe

logger = logging.getLogger(__name__)


class RefKind(StrEnum):
    """Which chunker record an :class:`InventoryRow` came from."""

    PATH = "path"  # SasPathRef
    DATASET = "dataset"  # SasDatasetRef
    DB_TABLE = "db_table"  # SasDbTableRef
    ENGINE = "engine"  # SasEngineRef


#: What a statement does with the place it names: a LIBNAME or FILENAME binds a
#: name to it, and a cd binds the current directory; an INFILE or %INCLUDE
#: reads it, a FILE or ODS destination writes.
_PATH_ROLES = {
    "libname": "bind",
    "using_libname": "bind",
    "user": "bind",
    "filename": "bind",
    "infile": "read",
    "include": "read",
    "proc_import": "read",
    "sasautos": "read",
    "file": "write",
    "proc_export": "write",
    "ods": "write",
    "printto": "write",
    "cd": "bind",
}


class InventoryRow(BaseModel, frozen=True):
    """One reference a SAS corpus makes, as a row of the inventory table.

    Attributes
    ----------
    run_id
        The inventory run: ``<UTC stamp>-<8 hex>``, so the latest run of a
        corpus is its greatest ``run_id``.
    corpus
        Which corpus the run took — a name the writer gives it, the source
        directory by default — so one table can keep several corpora and
        :func:`read_inventory` reads the latest run of the one asked for.
    recorded_at
        When the run was taken.
    file_order, ref_order
        Where the reference stands: the file's place in the corpus, and the
        reference's place in the file. The planner gives a table read by
        several files to the first, so the order is part of the data.
    source_id, chunk_id, start_line, end_line
        Which file, chunk and lines named it.
    kind
        See :class:`RefKind`.
    statement
        What named it: a path's statement (``libname``, ``infile``, ...), a
        dataset's ``via`` (``set``, ``data=``, ``from``, ...), a database
        table's ``via`` (``connection_to``, ``execute``, ``libname``), or
        ``libname`` for a database LIBNAME. A LIBNAME clause of a CREATE
        VIEW's USING list, path or database, is ``using_libname``: its libref
        is the view's alone.
    role
        What the statement does with it: a dataset's role (``read``,
        ``write``, ``update``, ``drop``, ``mention``), a database table's access
        (``read``, ``write``), a path's ``bind`` / ``read`` / ``write``, or
        ``bind`` for a database LIBNAME.
    name
        The comparison key: a path normalised (lowercased, forward slashes), a
        canonical dataset name (``work.x``, ``lib.sales_:``), a database
        table's ``schema.table``, a database LIBNAME's libref.
    raw
        As the SAS wrote it.
    value
        The best-known value, case kept: for a path the place SAS reads
        (macro variables expanded, a fileref followed), for the rest
        :attr:`name`.
    resolved
        The reference names a definite place or object: no unresolved ``&``
        remains, and a path is not left at a fileref no FILENAME binds.
    has_macro_ref
        An unresolved ``&`` remains — in the name, or in a connection option.
    libref
        The libref or fileref: a path's ``binds``, a dataset's libref, a
        database table's connection, a database LIBNAME's libref.
    member
        A dataset's member name, or a database table's table name.
    db_schema, dblink, sas_targets, macro, parameterised
        A database table's — see :class:`~chunker.models.SasDbTableRef`.
    location, device
        A path's — see :class:`~chunker.models.SasPathRef`.
    engine
        A path's LIBNAME engine (``spde``), or a database's (``oracle``).
    options
        Connection options, every value but a connection coordinate's
        redacted (see the module docstring).
    pattern
        A list rather than one object: ``lib.sales_:``, or ``edw.acct_:``.
    in_macro_body, macro_param, macro_param_pos
        A dataset named inside a ``%MACRO`` body, and the parameter that
        spells it when a call supplies it.
    found_local, found_sharepoint
        For a ``%INCLUDE``, the scripts of its file name found in the local
        corpus and in the application's SharePoint scripts folder
        (:func:`data_hydration.includes.match_includes`); ``None`` where
        nobody looked.
    concat_index
        A path's place in a concatenation (``libname x ('/a' '/b');``), from
        0 — see :attr:`~chunker.models.SasPathRef.concat_index`. The rows of
        one concatenation follow each other in ``ref_order``.
    """

    run_id: str
    corpus: str
    recorded_at: datetime
    file_order: int
    ref_order: int
    source_id: str
    chunk_id: str
    start_line: int
    end_line: int
    kind: RefKind
    statement: str
    role: str
    name: str
    raw: str
    value: str
    resolved: bool
    has_macro_ref: bool
    libref: str | None = None
    member: str | None = None
    db_schema: str | None = None
    location: str | None = None
    engine: str | None = None
    device: str | None = None
    dblink: str | None = None
    options: tuple[tuple[str, str], ...] = ()
    sas_targets: tuple[str, ...] = ()
    pattern: bool = False
    in_macro_body: bool = False
    macro_param: str | None = None
    macro_param_pos: int | None = None
    macro: str | None = None
    parameterised: bool = False
    found_local: tuple[str, ...] | None = None
    found_sharepoint: tuple[str, ...] | None = None
    concat_index: int | None = None

    @field_validator("options", mode="before")
    @classmethod
    def _option_pairs(cls, value: Any) -> Any:
        """Options as Spark returns them — ``{"key": ..., "value": ...}``
        structs — or as pairs."""
        if value is None:
            return ()
        return tuple(
            (item["key"], item["value"]) if isinstance(item, Mapping) else tuple(item)
            for item in value
        )

    @field_validator("sas_targets", mode="before")
    @classmethod
    def _no_targets(cls, value: Any) -> Any:
        return () if value is None else value


# ---------------------------------------------------------------------------
# Building rows from the chunker's records
# ---------------------------------------------------------------------------

_REDACTED = "<redacted>"

#: Connection options whose values are kept: where a connection goes and as
#: whom — what the planner and its readers use (``path``, ``server``,
#: ``schema``, ``user``) and what a person reading the inventory needs to
#: recognise it. Every other value is redacted (see the module docstring).
_KEPT_OPTIONS = frozenset(
    {
        "access", "account", "authdomain", "catalog", "connection", "database",
        "databasename", "datasrc", "db", "dbname", "driver", "dsn", "host",
        "hostname", "mode", "path", "port", "project", "role", "schema",
        "server", "service", "service_name", "sid", "tdpid", "uid", "user",
        "warehouse",
    }
)  # fmt: skip
#: Options whose value is a connection string of its own ``key=value;`` pairs,
#: redacted pair by pair.
_CONNECTION_STRINGS = frozenset(
    {"complete", "conopts", "init_string", "noprompt", "prompt", "required", "uri", "url"}
)
# One pair of a connection string; a braced ODBC value may hold ';' and spaces.
_PAIR_RE = re.compile(r"([^;=]+)=(\{[^}]*\}|[^;]*)")
# Credentials written into a URL: scheme://user:password@host.
_URL_PASSWORD_RE = re.compile(r"(//[^/:@;]+:)[^@/;]+(@)")


def _is_macro_ref(value: str) -> bool:
    return value.strip("'\" ").startswith("&")


def _kept_value(key: str, value: str) -> str:
    """*value* of option *key*, or ``<redacted>`` (see :data:`_KEPT_OPTIONS`)."""
    key = key.strip().lower()
    if _is_macro_ref(value) or key in _KEPT_OPTIONS:
        return value
    if key in _CONNECTION_STRINGS:
        return _URL_PASSWORD_RE.sub(
            rf"\g<1>{_REDACTED}\g<2>",
            _PAIR_RE.sub(lambda m: f"{m.group(1)}={_kept_value(m.group(1), m.group(2))}", value),
        )
    return _REDACTED


def _redacted(options: Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """*options* with every value but a connection coordinate's redacted."""
    return tuple((key, _kept_value(key, value)) for key, value in options)


def _statement(binds: str, engine: str, options: Iterable[tuple[str, str]]) -> str:
    """A database LIBNAME as a statement, from its (redacted) options: what the
    inventory keeps for ``raw``, since the statement as written carries every
    secret its options do."""
    words = [f"libname {binds} {engine}"]
    for key, value in options:
        quoted = any(c in value for c in " ;'\"") and not value.startswith(("(", '"', "'"))
        words.append(f'{key}="{value}"' if quoted else f"{key}={value}")
    return " ".join(words) + ";"


def _dataset_parts(name: str) -> tuple[str | None, str | None]:
    """``(libref, member)`` of a canonical dataset name. A physical path
    (``'/x.sas7bdat'``) has neither, nor has a name whose libref is still a
    macro reference (``&lib..x``); a parameter (``&ds``) neither."""
    if name.startswith("'") or "." not in name or name.startswith("&"):
        return None, None
    libref, member = name.split(".", 1)
    return libref, member


def new_run_id(now: datetime | None = None) -> str:
    """A run id: the UTC instant, path-safe, and 8 hex digits for uniqueness —
    ``20261009T132000Z-1f2e3d4c``. Ids sort in time order."""
    now = now or datetime.now(timezone.utc)
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def inventory_rows(
    file_results: Iterable[Any],
    *,
    corpus: str = "",
    run_id: str | None = None,
    recorded_at: datetime | None = None,
) -> list[InventoryRow]:
    """Every reference *file_results* make, as rows, in corpus order.

    *file_results* are :class:`chunker.SasChunkResult` objects (read by
    attribute), resolved across the corpus first with
    ``chunker.resolve_corpus_datasets``: a ``%LET`` or LIBNAME in one file
    reaches the references in the next, ``_LAST_`` is made concrete, and each
    macro call carries the datasets its macro reads through its parameters —
    without that last, ``%load(raw.c)`` names nothing here. Only top-level
    chunks are read; a reference a chunk names twice is one row. *corpus*
    names the corpus on every row (see :attr:`InventoryRow.corpus`).
    """
    recorded_at = recorded_at or datetime.now(timezone.utc)
    run_id = run_id or new_run_id(recorded_at)
    rows: list[InventoryRow] = []
    for file_order, result in enumerate(file_results):
        source_id = result.source_id or ""
        ref_order = 0
        for chunk in result.chunks:
            if chunk.parent_id is not None:
                continue
            where = {
                "run_id": run_id,
                "corpus": corpus,
                "recorded_at": recorded_at,
                "file_order": file_order,
                "source_id": source_id,
                "chunk_id": chunk.chunk_id,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            }
            seen: set[tuple[Any, ...]] = set()
            for fields in _chunk_refs(chunk.metadata):
                key = tuple(sorted((k, str(v)) for k, v in fields.items()))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(InventoryRow(**where, ref_order=ref_order, **fields))
                ref_order += 1
    logger.info(
        f"inventory_rows: {len(rows)} reference(s) across "
        f"{len({r.source_id for r in rows})} file(s), run {run_id}"
    )
    return rows


def _chunk_refs(meta: Any) -> Iterable[dict[str, Any]]:
    """The row fields of every reference on one chunk's metadata."""
    for ref in meta.engine_refs:
        options = _redacted(ref.options)
        yield {
            "kind": RefKind.ENGINE,
            "statement": ref.statement,
            "role": "bind",
            "name": ref.binds,
            "raw": _statement(ref.binds, ref.engine, options),
            "value": ref.binds,
            "resolved": not ref.has_macro_ref,
            "has_macro_ref": ref.has_macro_ref,
            "libref": ref.binds,
            "engine": ref.engine,
            "options": options,
        }
    for ref in meta.external_refs:
        location = str(ref.location)
        yield {
            "kind": RefKind.PATH,
            "statement": ref.statement,
            "role": _PATH_ROLES.get(ref.statement, "read"),
            "name": ref.path,
            "raw": ref.raw,
            "value": ref.effective_path,
            "resolved": not ref.has_macro_ref and location != "fileref",
            "has_macro_ref": ref.has_macro_ref,
            "libref": ref.binds,
            "location": location,
            "engine": ref.engine,
            "device": ref.device,
            "concat_index": ref.concat_index,
        }
    for ref in meta.db_tables:
        yield {
            "kind": RefKind.DB_TABLE,
            "statement": str(ref.via),
            "role": str(ref.access),
            "name": ref.qualified,
            "raw": ref.raw or ref.qualified,
            "value": ref.qualified,
            "resolved": not ref.has_macro_ref and not ref.parameterised,
            "has_macro_ref": ref.has_macro_ref,
            "libref": ref.connection,
            "member": ref.table,
            "db_schema": ref.db_schema,
            "engine": ref.engine,
            "dblink": ref.dblink,
            "options": _redacted(ref.options),
            "sas_targets": tuple(ref.sas_targets),
            "pattern": ref.table.endswith(":"),
            "macro": ref.macro,
            "parameterised": ref.parameterised,
        }
    for ref in meta.dataset_refs:
        libref, member = _dataset_parts(ref.name)
        unresolved = "&" in ref.name
        yield {
            "kind": RefKind.DATASET,
            "statement": ref.via,
            "role": str(ref.role),
            "name": ref.name,
            "raw": ref.raw or ref.name,
            "value": ref.name,
            "resolved": not unresolved,
            "has_macro_ref": unresolved,
            "libref": libref,
            "member": member,
            "pattern": ref.pattern,
            "in_macro_body": ref.in_macro_body,
            "macro_param": ref.param,
            "macro_param_pos": ref.param_pos,
        }


# ---------------------------------------------------------------------------
# Planning from rows
# ---------------------------------------------------------------------------
#
# The planner reads chunker records by attribute; these carry the same
# attributes, built from a row, so a plan needs neither the chunker nor the
# SAS source.


@dataclass(frozen=True, slots=True)
class _PathRecord:
    statement: str
    location: str
    path: str
    raw: str
    binds: str | None
    device: str | None
    engine: str | None
    has_macro_ref: bool
    resolved_path: str | None
    #: Where the reference stands in its file (``ref_order``): a LIBNAME binds
    #: its libref for the reads after it, not the ones before.
    order: int | None = None
    concat_index: int | None = None

    @property
    def effective_path(self) -> str:
        return self.resolved_path or self.raw

    @classmethod
    def of(cls, row: InventoryRow) -> "_PathRecord":
        return cls(
            statement=row.statement,
            location=row.location or "",
            path=row.name,
            raw=row.raw,
            binds=row.libref,
            device=row.device,
            engine=row.engine,
            has_macro_ref=row.has_macro_ref,
            resolved_path=row.value if row.value != row.raw else None,
            order=row.ref_order,
            concat_index=row.concat_index,
        )


@dataclass(frozen=True, slots=True)
class _EngineRecord:
    engine: str
    binds: str
    options: tuple[tuple[str, str], ...]
    has_macro_ref: bool
    raw: str
    statement: str = "libname"

    @property
    def option_map(self) -> dict[str, str]:
        return dict(self.options)

    @classmethod
    def of(cls, row: InventoryRow) -> "_EngineRecord":
        return cls(
            engine=row.engine or "",
            binds=row.name,
            options=row.options,
            has_macro_ref=row.has_macro_ref,
            raw=row.raw,
            statement=row.statement,
        )


@dataclass(frozen=True, slots=True)
class _DbTableRecord:
    engine: str | None
    connection: str
    db_schema: str | None
    table: str
    access: str
    via: str
    sas_targets: tuple[str, ...]
    dblink: str | None
    options: tuple[tuple[str, str], ...]
    has_macro_ref: bool
    raw: str
    macro: str | None
    parameterised: bool

    @property
    def qualified(self) -> str:
        return f"{self.db_schema}.{self.table}" if self.db_schema else self.table

    @property
    def option_map(self) -> dict[str, str]:
        return dict(self.options)

    @classmethod
    def of(cls, row: InventoryRow) -> "_DbTableRecord":
        return cls(
            engine=row.engine,
            connection=row.libref or "",
            db_schema=row.db_schema,
            table=row.member or "",
            access=row.role,
            via=row.statement,
            sas_targets=row.sas_targets,
            dblink=row.dblink,
            options=row.options,
            has_macro_ref=row.has_macro_ref,
            raw=row.raw,
            macro=row.macro,
            parameterised=row.parameterised,
        )


@dataclass(frozen=True, slots=True)
class _DatasetRecord:
    name: str
    role: str
    raw: str
    via: str
    in_macro_body: bool
    param: str | None
    param_pos: int | None
    pattern: bool
    order: int | None = None  # as _PathRecord.order

    @classmethod
    def of(cls, row: InventoryRow) -> "_DatasetRecord":
        return cls(
            name=row.name,
            role=row.role,
            raw=row.raw,
            via=row.statement,
            in_macro_body=row.in_macro_body,
            param=row.macro_param,
            param_pos=row.macro_param_pos,
            pattern=row.pattern,
            order=row.ref_order,
        )


def plan_from_inventory(
    rows: Iterable[InventoryRow],
    *,
    config: "HydrationConfig | None" = None,
    probe: "SourceProbe | None" = None,
    only: Iterable[str] = (),
) -> "HydrationPlan":
    """The hydration plan for one inventory run's *rows*.

    The rows are read in corpus order (``file_order``, ``ref_order``) and
    handed to :func:`~data_hydration.planner.build_corpus_plan` as the records
    the chunks would have given it: paths, database LIBNAMEs and tables, and the
    datasets a directory LIBNAME's members are found by. *only* keeps the
    references made through those librefs (or connections), as the CLI's
    ``--only`` does.
    """
    from .planner import build_corpus_plan

    wanted = {libref.lower() for libref in only}
    by_source: dict[str, tuple[list[Any], list[Any]]] = {}
    db_tables: dict[str, list[Any]] = {}
    # Each file's datasets chunk by chunk: a step that reads a table and
    # rewrites it needs it from outside, one after the step that created it
    # does not (see planner._library_members).
    datasets: dict[str, list[list[Any]]] = {}
    chunk_of: dict[str, str] = {}
    for row in sorted(rows, key=lambda r: (r.file_order, r.ref_order)):
        engine_refs, path_refs = by_source.setdefault(row.source_id, ([], []))
        if wanted and (row.libref or "") not in wanted:
            continue
        if row.kind is RefKind.ENGINE:
            engine_refs.append(_EngineRecord.of(row))
        elif row.kind is RefKind.PATH:
            path_refs.append(_PathRecord.of(row))
        elif row.kind is RefKind.DB_TABLE:
            db_tables.setdefault(row.source_id, []).append(_DbTableRecord.of(row))
        else:
            chunks = datasets.setdefault(row.source_id, [])
            if not chunks or chunk_of[row.source_id] != row.chunk_id:
                chunks.append([])
                chunk_of[row.source_id] = row.chunk_id
            chunks[-1].append(_DatasetRecord.of(row))
    return build_corpus_plan(
        by_source, db_tables=db_tables, datasets=datasets, config=config, probe=probe
    )


# ---------------------------------------------------------------------------
# The Delta table
# ---------------------------------------------------------------------------

#: The table's columns, in order: name, Spark SQL type, and what it holds.
#: :class:`InventoryRow` documents each.
_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("run_id", "STRING", "inventory run; the latest is the greatest"),
    ("corpus", "STRING", "the corpus the run took: a name, or its directory"),
    ("recorded_at", "TIMESTAMP", "when the run was taken"),
    ("file_order", "INT", "place of the file in the corpus"),
    ("ref_order", "INT", "place of the reference in its file"),
    ("source_id", "STRING", "the SAS file"),
    ("chunk_id", "STRING", "the chunk that names it"),
    ("start_line", "INT", "first line of that chunk"),
    ("end_line", "INT", "last line of that chunk"),
    ("kind", "STRING", "path, dataset, db_table or engine"),
    ("statement", "STRING", "the statement or option that names it"),
    ("role", "STRING", "read, write, update, drop, mention or bind"),
    ("name", "STRING", "the comparison key"),
    ("raw", "STRING", "as written"),
    ("value", "STRING", "best-known value, macro variables expanded"),
    ("resolved", "BOOLEAN", "names a definite place or object"),
    ("has_macro_ref", "BOOLEAN", "an unresolved macro reference remains"),
    ("libref", "STRING", "libref, fileref or connection"),
    ("member", "STRING", "dataset member or database table"),
    ("db_schema", "STRING", "database schema"),
    ("location", "STRING", "filesystem, remote, email, pipe, device or fileref"),
    ("engine", "STRING", "LIBNAME or database engine"),
    ("device", "STRING", "FILENAME device"),
    ("dblink", "STRING", "Oracle database link"),
    (
        "options",
        "ARRAY<STRUCT<key: STRING, value: STRING>>",
        "connection options, passwords redacted",
    ),
    ("sas_targets", "ARRAY<STRING>", "SAS datasets a database read lands in"),
    ("pattern", "BOOLEAN", "a list of datasets or tables"),
    ("in_macro_body", "BOOLEAN", "named inside a %MACRO body"),
    ("macro_param", "STRING", "the macro parameter that spells it"),
    ("macro_param_pos", "INT", "position of that parameter; -1 for a keyword"),
    ("macro", "STRING", "the macro whose call reads the table"),
    ("parameterised", "BOOLEAN", "a template built from the parameters of its macro"),
    ("found_local", "ARRAY<STRING>", "an included script found in the local corpus"),
    ("found_sharepoint", "ARRAY<STRING>", "an included script found in SharePoint"),
    # Columns added after the table's first release go last: an existing table
    # gains them on the next write (write_inventory), null in earlier runs.
    ("concat_index", "INT", "place of a path in a concatenation, from 0"),
)

# Unity Catalog identifiers: one to three parts of letters, digits and _.
_TABLE_PART_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _quoted_table(table: str) -> str:
    """*table* as a backquoted SQL identifier, or :class:`ValueError`.

    A table name is configuration, not data, but it is spliced into SQL, so it
    is held to ordinary identifier parts.
    """
    parts = table.split(".") if table else []
    if not 1 <= len(parts) <= 3 or not all(_TABLE_PART_RE.match(p) for p in parts):
        raise ValueError(
            f"inventory table {table!r} must be one to three identifier parts "
            f"of letters, digits and underscores"
        )
    return ".".join(f"`{part}`" for part in parts)


def _session(spark: Any) -> Any:
    """*spark*, else the active session or a new one (see
    :func:`app_config.spark.active_or_new_session`)."""
    if spark is not None:
        return spark
    from app_config.spark import active_or_new_session

    return active_or_new_session("sas-parser-hydration")


def ensure_inventory_table(table: str, *, spark: Any = None) -> None:
    """Create the inventory table when it does not exist. Its schema, never
    its catalog: see :func:`data_hydration.sinks.delta._ensure_schema`."""
    spark = _session(spark)
    quoted = _quoted_table(table)
    if "." in quoted:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {quoted.rsplit('.', 1)[0]}")
    columns = ",\n  ".join(
        f"`{name}` {sql_type} COMMENT '{comment}'" for name, sql_type, comment in _COLUMNS
    )
    spark.sql(
        f"CREATE TABLE IF NOT EXISTS {quoted} (\n  {columns}\n) USING DELTA "
        f"COMMENT 'Every path and dataset a SAS corpus names, one run per write'"
    )


def _values(row: InventoryRow) -> tuple[Any, ...]:
    """*row* as a tuple in :data:`_COLUMNS` order, in the types Spark takes."""
    fields = row.model_dump()
    fields["kind"] = str(row.kind)
    fields["options"] = [list(pair) for pair in row.options]
    fields["sas_targets"] = list(row.sas_targets)
    for name in ("found_local", "found_sharepoint"):
        found = getattr(row, name)
        fields[name] = None if found is None else list(found)
    return tuple(fields[name] for name, _, _ in _COLUMNS)


def write_inventory(rows: Sequence[InventoryRow], table: str, *, spark: Any = None) -> int:
    """Append *rows* to the Delta table *table* as one commit; return how many.

    One run per call: the rows are one corpus's references at one instant, and
    earlier runs stay as they were — the table is the history, and
    :func:`read_inventory` reads the latest. The table is created when missing,
    and a table created before a column was added gains it here (Delta schema
    merge), the earlier runs reading null in it.
    """
    if not rows:
        logger.warning(f"write_inventory: no references to write to {table}")
        return 0
    spark = _session(spark)
    ensure_inventory_table(table, spark=spark)
    schema = ", ".join(f"`{name}` {sql_type}" for name, sql_type, _ in _COLUMNS)
    frame = spark.createDataFrame([_values(row) for row in rows], schema=schema)
    frame.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table)
    logger.info(f"write_inventory: {len(rows)} row(s) -> {table} (run {rows[0].run_id})")
    return len(rows)


def read_inventory(
    table: str,
    *,
    spark: Any = None,
    run_id: str | None = None,
    corpus: str | None = None,
) -> list[InventoryRow]:
    """The rows of one run of the inventory *table*, in corpus order: *run_id*;
    else the latest run of *corpus*; else the table's latest run, whichever
    corpus it took. Empty when there is no such run."""
    spark = _session(spark)
    from pyspark.sql import functions as F

    frame = spark.table(_quoted_table(table))
    if run_id is None:
        runs = frame if corpus is None else frame.filter(F.col("corpus") == corpus)
        run_id = runs.agg(F.max("run_id")).first()[0]
        if run_id is None:
            which = "no run" if corpus is None else f"no run of corpus {corpus!r}"
            logger.warning(f"read_inventory: {table} holds {which}")
            return []
    selected = frame.filter(F.col("run_id") == run_id).orderBy("file_order", "ref_order")
    rows = [InventoryRow.model_validate(row.asDict(recursive=True)) for row in selected.collect()]
    logger.info(f"read_inventory: {len(rows)} row(s) of run {run_id} from {table}")
    return rows
