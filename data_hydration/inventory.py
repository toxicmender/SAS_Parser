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
* **Credentials are redacted.** A connection option whose key names a password
  (``pass=``, ``password=``, ``pwd=``, ...) is stored as ``<redacted>``, and so
  is a ``PWD=`` inside a connection string. A macro reference (``&user_pass.``)
  is kept: it is no secret, and the planner's blocker names it. No reader takes
  a password from these options — :mod:`data_hydration.secrets` resolves it.

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
#: name to it, an INFILE or %INCLUDE reads it, a FILE or ODS destination writes.
_PATH_ROLES = {
    "libname": "bind",
    "filename": "bind",
    "infile": "read",
    "include": "read",
    "proc_import": "read",
    "sasautos": "read",
    "file": "write",
    "proc_export": "write",
    "ods": "write",
    "printto": "write",
}


class InventoryRow(BaseModel, frozen=True):
    """One reference a SAS corpus makes, as a row of the inventory table.

    Attributes
    ----------
    run_id
        The inventory run: ``<UTC stamp>-<8 hex>``, so the latest run of a
        table is its greatest ``run_id``.
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
        ``libname`` for a database LIBNAME.
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
        Connection options, password values redacted.
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
    """

    run_id: str
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
# Option keys that hold a password: pass, password, pw, pwd, dbpass, orapw, ...
_SECRET_KEY_RE = re.compile(r"pass|pw", re.IGNORECASE)
# A password inside a connection string: `noprompt="DSN=x;UID=u;PWD=secret;"`.
# A macro reference (`PWD=&pw.`) is no secret and stays.
_SECRET_IN_VALUE_RE = re.compile(r"\b((?:pwd|password)\s*=\s*)(?!&)[^;\"'\s]+", re.IGNORECASE)
# A password option as a statement spells it: `pass="secret"`, `pw=secret`.
_SECRET_OPTION_RE = re.compile(
    r"""(\b\w*(?:pass|pw)\w*\s*=\s*)("[^"]*"|'[^']*'|[^\s;)]+)""", re.IGNORECASE
)


def _is_macro_ref(value: str) -> bool:
    return value.strip("'\" ").startswith("&")


def _redacted(options: Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """*options* with every literal password replaced by ``<redacted>``."""
    kept: list[tuple[str, str]] = []
    for key, value in options:
        if _SECRET_KEY_RE.search(key) and not _is_macro_ref(value):
            value = _REDACTED
        else:
            value = _SECRET_IN_VALUE_RE.sub(rf"\g<1>{_REDACTED}", value)
        kept.append((key, value))
    return tuple(kept)


def _redacted_statement(text: str) -> str:
    """*text*, a statement as written, with its literal passwords redacted by
    :func:`_redacted`'s rules."""

    def hide(match: re.Match[str]) -> str:
        if _is_macro_ref(match.group(2)):
            return match.group(0)
        return f"{match.group(1)}{_REDACTED}"

    text = _SECRET_OPTION_RE.sub(hide, text)
    return _SECRET_IN_VALUE_RE.sub(rf"\g<1>{_REDACTED}", text)


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
    run_id: str | None = None,
    recorded_at: datetime | None = None,
) -> list[InventoryRow]:
    """Every reference *file_results* make, as rows, in corpus order.

    *file_results* are :class:`chunker.SasChunkResult` objects (read by
    attribute), resolved across the corpus first —
    ``chunker.resolve_corpus_references`` — so a ``%LET`` or LIBNAME in one
    file reaches the references in the next. Only top-level chunks are read;
    a reference a chunk names twice is one row.
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
        # The statement as written carries its options, passwords included.
        raw = _redacted_statement(ref.raw) or " ".join(
            ["libname", ref.binds, ref.engine, *(f"{k}={v}" for k, v in options)]
        )
        yield {
            "kind": RefKind.ENGINE,
            "statement": "libname",
            "role": "bind",
            "name": ref.binds,
            "raw": raw,
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
        )


@dataclass(frozen=True, slots=True)
class _EngineRecord:
    engine: str
    binds: str
    options: tuple[tuple[str, str], ...]
    has_macro_ref: bool
    raw: str

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
    datasets: dict[str, list[Any]] = {}
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
            datasets.setdefault(row.source_id, []).append(_DatasetRecord.of(row))
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
    :func:`read_inventory` reads the latest. The table is created when missing.
    """
    if not rows:
        logger.warning(f"write_inventory: no references to write to {table}")
        return 0
    spark = _session(spark)
    ensure_inventory_table(table, spark=spark)
    schema = ", ".join(f"`{name}` {sql_type}" for name, sql_type, _ in _COLUMNS)
    frame = spark.createDataFrame([_values(row) for row in rows], schema=schema)
    frame.write.format("delta").mode("append").saveAsTable(table)
    logger.info(f"write_inventory: {len(rows)} row(s) -> {table} (run {rows[0].run_id})")
    return len(rows)


def read_inventory(
    table: str, *, spark: Any = None, run_id: str | None = None
) -> list[InventoryRow]:
    """The rows of one run of the inventory *table*, in corpus order: *run_id*,
    or the latest run when it is ``None``. Empty when the table holds none."""
    spark = _session(spark)
    from pyspark.sql import functions as F

    frame = spark.table(_quoted_table(table))
    if run_id is None:
        run_id = frame.agg(F.max("run_id")).first()[0]
        if run_id is None:
            logger.warning(f"read_inventory: {table} holds no run")
            return []
    selected = frame.filter(F.col("run_id") == run_id).orderBy("file_order", "ref_order")
    rows = [InventoryRow.model_validate(row.asDict(recursive=True)) for row in selected.collect()]
    logger.info(f"read_inventory: {len(rows)} row(s) of run {run_id} from {table}")
    return rows
