"""Pydantic models for the SAS semantic chunker and batcher. See chunker/README.md."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, computed_field, model_validator

logger = logging.getLogger(__name__)


def _is_automatic_macro_var(name: str) -> bool:
    """
    True if *name* (without the leading ``&`` or trailing ``.``) is one of
    SAS's automatic macro variables.

    Per *SAS Macro Language: Reference* (Ch. 12, "Automatic Macro
    Variables"), every automatic macro variable's name begins with the
    reserved ``SYS`` prefix — confirmed across all ~60 of them (SYSDATE,
    SYSLAST, SYSPARM, …).  A simple prefix check is sufficient; no
    enumerated lookup table is needed or maintained.
    """
    return name.lower().startswith("sys")


class SasChunkKind(StrEnum):
    """Semantic unit types recognised by the chunker."""

    DATA_STEP = "DATA_STEP"
    PROC_STEP = "PROC_STEP"
    MACRO_DEFINITION = "MACRO_DEFINITION"
    MACRO_CALL = "MACRO_CALL"
    MACRO_CONTROL_FLOW = "MACRO_CONTROL_FLOW"
    INCLUDE = "INCLUDE"
    GLOBAL_STATEMENT = "GLOBAL_STATEMENT"
    STEP_BOUNDARY = "STEP_BOUNDARY"
    COMMENT_BLOCK = "COMMENT_BLOCK"
    OPTIONS = "OPTIONS"
    FORMAT_OR_INFORMAT = "FORMAT_OR_INFORMAT"
    UNKNOWN_STATEMENT_GROUP = "UNKNOWN_STATEMENT_GROUP"
    UNKNOWN_BLOCK = "UNKNOWN_BLOCK"


class SasDiagnosticSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class PathLocation(StrEnum):
    """What kind of place a :class:`SasPathRef` points at.

    A SAS statement that takes a quoted argument does not always take a
    *filesystem* path: ``FILENAME`` accepts a device keyword that redirects the
    same syntax at an FTP server, a mailbox, or a shell pipe. Each is a real
    external dependency of the corpus and each needs a different answer on the
    target, so they are classified rather than collapsed — or, worse, silently
    treated as directories somebody then tries to mount.

    ``DEVICE`` is the deliberate catch-all: a device keyword this module does
    not know must land somewhere visible instead of defaulting to
    :attr:`FILESYSTEM`.
    """

    FILESYSTEM = "filesystem"
    REMOTE = "remote"
    EMAIL = "email"
    PIPE = "pipe"
    DEVICE = "device"


class SasPathRef(BaseModel, frozen=True):
    """One external reference a statement names, with its provenance.

    Frozen so it is hashable: ``_merge_meta`` merges the containing list by set
    union, which a mutable model could not do. (Same reason
    :class:`prompt_builder.ConstructKey` is frozen.)

    Attributes
    ----------
    statement
        Which statement named it — ``libname``, ``filename``, ``infile``,
        ``file``, ``include``, ``proc_import``, ``proc_export``, ``printto``,
        ``ods``, ``sasautos``.
    location
        See :class:`PathLocation`.
    path
        Normalised for comparison: lowercased, backslashes turned to forward
        slashes. Unlike the dataset vocabulary's quoted-path keys it carries no
        quote wrapper — nothing here shares a namespace with identifiers.
    raw
        Exactly as written, before normalisation. A consumer that has to
        *rewrite* the source needs the original spelling; a consumer that has to
        *match* wants ``path``.
    binds
        The libref or fileref the statement assigns, when it assigns one.
    device
        The device keyword as written, when there was one.
    engine
        The LIBNAME engine as written, when the statement named one —
        ``spde``, ``xport``, ``v9``, ... An engine changes what the quoted
        directory *is*: ``libname x spde '/p'`` is a partitioned SPD Engine
        library, not the ordinary directory ``libname x '/p'`` names, and
        nothing downstream could tell them apart while this went unrecorded.
        Engines that carry no path at all are :class:`SasEngineRef` instead.
    has_macro_ref
        The value contains a ``&`` reference, so its real value is not knowable
        without running SAS. Recorded rather than dropped: a path that cannot be
        resolved is exactly what a migration needs told about.
    """

    statement: str
    location: PathLocation
    path: str
    raw: str
    binds: str | None = None
    device: str | None = None
    engine: str | None = None
    has_macro_ref: bool = False

    def __str__(self) -> str:
        bound = f" {self.binds}" if self.binds else ""
        via = f" via {self.engine}" if self.engine else ""
        return f"{self.statement}{bound}{via} [{self.location}] {self.raw}"


def _path_ref_sort_key(ref: SasPathRef) -> tuple[str, str, str, str]:
    """Total order over :class:`SasPathRef`, defined once.

    Both places that deduplicate these records through a set —
    ``chunker.metadata._merge_meta`` and :attr:`SasBatch.external_refs` — sort
    the result with this, because set iteration order is not stable across runs
    and batch output is pinned by tests (invariant 9).
    """
    return (str(ref.location), ref.statement, ref.path, ref.binds or "")


class SasEngineRef(BaseModel, frozen=True):
    """A ``LIBNAME`` bound to a database engine, with its connection options.

    The sibling of :class:`SasPathRef`, for the LIBNAME form that names no path
    at all::

        libname edwprod oracle path=EDWPRO_READ_ONLY schema=fr_dm_pro
                               user="&username." pass="&user_pass.";

    That statement has no quoted directory, so :data:`chunker.paths.PATH_STATEMENTS`
    — every entry of which ends in a quoted value — never matched it, and the
    connection a migration has to reproduce went unrecorded. It is a *foreign
    system*, not a place: whether it is federated or copied into the lakehouse is
    a decision somebody makes downstream, and this records what the SAS declared
    so that decision can be made at all.

    Frozen so it is hashable, for the same reason :class:`SasPathRef` is —
    ``_merge_meta`` merges the containing list by set union.

    Attributes
    ----------
    engine
        The engine keyword, lowercased: ``oracle``, ``odbc``, ``teradata``, ...
        One of :data:`chunker.paths.ENGINE_LIBNAMES`.
    binds
        The libref the statement assigns, lowercased. Always present — a LIBNAME
        without one does not parse.
    options
        The statement's ``key=value`` options, keys lowercased and values exactly
        as written. A tuple of pairs rather than a ``dict`` because this model is
        frozen *and hashable*, which a dict field would break; use
        :attr:`option_map` to read it.

        Values are **not** resolved: ``user="&username."`` is stored with the
        macro reference intact. Credentials are the hydrating consumer's problem,
        and the chunker has no way to resolve a macro variable anyway.
    has_macro_ref
        Some option value contains a ``&`` reference, so the connection is not
        fully knowable without running SAS. Same meaning as the flag of the same
        name on :class:`SasPathRef`.
    raw
        The statement as written, whitespace-collapsed — what a human needs to
        recognise it in their own source.
    """

    engine: str
    binds: str
    options: tuple[tuple[str, str], ...] = ()
    has_macro_ref: bool = False
    raw: str = ""

    @property
    def option_map(self) -> dict[str, str]:
        """:attr:`options` as a mapping. Later duplicates win, as SAS does."""
        return dict(self.options)

    def __str__(self) -> str:
        opts = " ".join(f"{k}={v}" for k, v in self.options)
        return f"libname {self.binds} {self.engine}{' ' + opts if opts else ''}"


def _engine_ref_sort_key(ref: SasEngineRef) -> tuple[str, str]:
    """Total order over :class:`SasEngineRef`, defined once.

    The counterpart of :func:`_path_ref_sort_key`, and it exists for the same
    reason: ``_merge_meta`` and :attr:`SasBatch.engine_refs` both deduplicate
    through a set, whose iteration order is not stable across runs.
    """
    return (ref.engine, ref.binds)


class DbTableAccess(StrEnum):
    """Which way a :class:`SasDbTableRef`'s rows move, seen from the database."""

    READ = "read"
    WRITE = "write"


class DbTableVia(StrEnum):
    """How the SAS reached a :class:`SasDbTableRef`.

    ``CONNECTION_TO`` and ``EXECUTE`` are explicit SQL pass-through — native SQL
    SAS hands to the database untouched, recognised by
    :mod:`chunker.passthrough`. ``LIBNAME`` is a SAS two-level name whose libref
    a database-engine LIBNAME bound (``set edw.accounts;`` after
    ``libname edw oracle ...``), recognised by
    :func:`chunker.metadata.resolve_db_librefs`.
    """

    CONNECTION_TO = "connection_to"
    EXECUTE = "execute"
    LIBNAME = "libname"


class SasDbTableRef(BaseModel, frozen=True):
    """A table inside a database that a chunk reads or writes, in the database's terms.

    The third sibling of :class:`SasPathRef` and :class:`SasEngineRef`.
    :class:`SasEngineRef` records that a job *connects* to Oracle; this records
    *which tables* it touches there — ``edw_export.current_nonip`` — which is
    what a migration has to hydrate, and which no SAS dataset name says::

        create table nonip as select * from connection to oracle
        (select cov_month from edw_export.current_nonip);

    The SAS copy that read lands in (``work.nonip``) stays registered as an
    ordinary SAS dataset in ``output_datasets``; :attr:`sas_targets` is the link
    between the two. Frozen so it is hashable, for the same reason the siblings
    are.

    Attributes
    ----------
    engine
        The SAS/ACCESS engine, lowercased: ``oracle``, ``teradata``, ... ``None``
        when the connection cannot be traced — a ``CONNECTION TO edw`` whose
        ``CONNECT`` was made by a macro call, under an alias that names no engine.
    connection
        The name the SAS used for the connection: the pass-through alias
        (``CONNECT TO oracle AS edw`` → ``edw``; the engine itself when there was
        no ``AS``), or the libref for :attr:`DbTableVia.LIBNAME` and
        ``CONNECT USING``.
    db_schema
        The schema (Oracle owner) the table lives in, lowercased. ``None`` when
        the reference is unqualified: the database then resolves it against the
        connecting account's default schema, which static analysis cannot know.
        One ending in an unresolved reference keeps its delimiter dot
        (``&sch.``), so :attr:`qualified` reads ``&sch..t`` as SAS would.
        Named ``db_schema`` because ``schema`` shadows a pydantic attribute.
    table
        The table name, lowercased, quotes stripped.
    access
        Read or write, from the database's side. A pass-through ``SELECT`` reads;
        ``EXECUTE (create table ...)`` writes; ``data edw.x;`` writes.
    via
        See :class:`DbTableVia`.
    sas_targets
        The SAS datasets a read is copied into — ``create table nonip as select
        * from connection to oracle (...)`` → ``("work.nonip",)``. Canonical SAS
        names, as in ``output_datasets``. Empty for writes, and for a read whose
        rows only print or feed ``INTO :macro_var``.
    dblink
        An Oracle ``@link`` suffix, lowercased: the table lives in the *linked*
        database, not the one the connection reaches.
    options
        The connection's ``key=value`` options exactly as written — the
        ``CONNECT TO`` arguments, or the LIBNAME's — so a consumer has the whole
        connection on the one record rather than joining by alias (two
        ``PROC SQL`` blocks may reuse one alias with different ``path=``).
        Macro references stay unresolved, as on :class:`SasEngineRef`.
    has_macro_ref
        The table's *name* still holds an unresolved ``&`` reference, so
        :attr:`db_schema` / :attr:`table` are not the names SAS would send.
    raw
        The reference exactly as written: ``"EDW"."T"``, ``&schema..current_nonip``,
        or the SAS name ``edw.accounts`` for :attr:`DbTableVia.LIBNAME`.
    macro
        Set on a record attributed to a **macro call**: the name of the macro
        whose ``%MACRO`` body holds the SQL. The record sits on the call's
        chunk, resolved with that call's arguments — ``%pull(tbl=current_nonip)``
        reads ``edw_export.current_nonip`` even though the body only says
        ``&schema..&tbl``. ``None`` for a table named where the record sits.
    parameterised
        Inside a ``%MACRO`` body, the name is built from that macro's *own
        parameters* — ``&schema..&tbl`` in ``%macro pull(schema=, tbl=)`` — so
        it is a template, not a table: each call's reading is recorded on the
        call's chunk (see :attr:`macro`), and a macro the corpus never calls
        reads nothing. Reported, never hydrated.
    """

    engine: str | None = None
    connection: str
    db_schema: str | None = None
    table: str
    access: DbTableAccess
    via: DbTableVia
    sas_targets: tuple[str, ...] = ()
    dblink: str | None = None
    options: tuple[tuple[str, str], ...] = ()
    has_macro_ref: bool = False
    raw: str = ""
    macro: str | None = None
    parameterised: bool = False

    @property
    def qualified(self) -> str:
        """``db_schema.table``, or just ``table`` when unqualified.

        Unresolved names read as SAS code would write them: ``&sch..t``, since a
        schema ending in a reference carries its delimiter — ``&sch.t`` would be
        one name, the value of ``sch`` followed by ``t``.
        """
        return f"{self.db_schema}.{self.table}" if self.db_schema else self.table

    @property
    def option_map(self) -> dict[str, str]:
        """:attr:`options` as a mapping. Later duplicates win, as SAS does."""
        return dict(self.options)

    def __str__(self) -> str:
        # Options are deliberately left out: this string reaches the LLM prompt,
        # and connection options are where credentials live.
        link = f"@{self.dblink}" if self.dblink else ""
        copies = f" → {', '.join(self.sas_targets)}" if self.sas_targets else ""
        called = f" in %{self.macro}" if self.macro else ""
        return (
            f"{self.engine or self.connection}:{self.qualified}{link}{copies} "
            f"({self.access} via {self.via} {self.connection}{called})"
        )


def _db_table_sort_key(
    ref: SasDbTableRef,
) -> tuple[
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    tuple[str, ...],
    tuple[tuple[str, str], ...],
    str,
    str,
]:
    """Total order over :class:`SasDbTableRef`, defined once.

    Same reason as :func:`_path_ref_sort_key`: these records are deduplicated
    through sets, and batch output is pinned by tests (invariant 9). Every field
    that can tell two records apart takes part, so the order is total.
    """
    return (
        ref.engine or "",
        ref.db_schema or "",
        ref.table,
        ref.dblink or "",
        str(ref.access),
        str(ref.via),
        ref.connection,
        ref.macro or "",
        ref.sas_targets,
        ref.options,
        ref.raw,
        str(ref.parameterised),
    )


class DatasetRole(StrEnum):
    """What a statement does to a SAS dataset it names."""

    READ = "read"
    WRITE = "write"
    # Read and rewritten in place (MODIFY, APPEND BASE=, SQL INSERT/UPDATE/
    # DELETE): an input and an output at once.
    UPDATE = "update"
    # Deleted (PROC DATASETS DELETE, SQL DROP TABLE): neither read nor written.
    DROP = "drop"
    # Named, not used: a %LET value written like a dataset reference
    # (``%let t = edw.accounts;``). The step that uses &t reads or writes it.
    MENTION = "mention"

    @property
    def reads(self) -> bool:
        return self in (DatasetRole.READ, DatasetRole.UPDATE)

    @property
    def writes(self) -> bool:
        return self in (DatasetRole.WRITE, DatasetRole.UPDATE)


@dataclass(frozen=True, slots=True)
class SasDatasetRef:
    """One SAS dataset a chunk names, and what the chunk does with it.

    The stored source of a chunk's dataset metadata: ``input_datasets``,
    ``output_datasets``, ``dropped_datasets``, ``referenced_datasets``,
    ``referenced_librefs`` and the ``body_*`` lists are views of
    :attr:`SasChunkMetadata.dataset_refs`. Every rewrite — macro
    variables resolved, ``_LAST_`` and ``_DATA_`` made concrete, the Databricks
    mapping — goes through :meth:`SasChunkMetadata.map_dataset_names`, so the
    views cannot disagree.

    Frozen so it is hashable, like its siblings, but a slotted dataclass
    rather than a model: a corpus holds several per chunk, and the lighter
    record keeps chunking and batching fast. Pydantic still validates and
    serialises it as a field of :class:`SasChunkMetadata`.

    Attributes
    ----------
    name
        Canonical, as the batcher matches producers to consumers: ``work.x``
        for a one-level name, ``lib.x``, ``'/path'`` for a physical path, and
        the reference as written while it holds an unresolved ``&``. For a
        parameter reference (:attr:`param`), the parameter: ``&ds``.
    role
        See :class:`DatasetRole`.
    raw
        The name as the statement spelled it; empty when nothing recorded it.
    via
        The statement or option that named it (``set``, ``data``, ``out=``,
        ``from``, ``%let``, …), or where the reference came from (``macro_call``: a
        call's argument, resolved through the macro's body; ``_last_``: the
        dataset SAS's ``_LAST_`` held); empty when nothing recorded it.
    in_macro_body
        Named inside a ``%MACRO`` body, so read or written when the macro is
        called, not where it is defined: the ``body_*`` views.
    param, param_pos
        A body reference spelled through one of the macro's own parameters,
        and that parameter's position (-1 for a keyword parameter): each call
        site supplies the dataset (``body_param_*``).
    pattern
        A name list rather than one dataset: ``lib.sales_:`` is every member of
        ``lib`` whose name starts ``sales_``.
    """

    name: str
    role: DatasetRole
    raw: str = ""
    via: str = ""
    in_macro_body: bool = False
    param: str | None = None
    param_pos: int | None = None
    pattern: bool = False

    def __post_init__(self) -> None:
        # Built directly, a dataclass is not validated: take "read" as READ, so
        # the views' identity tests hold for every caller.
        if type(self.role) is not DatasetRole:
            object.__setattr__(self, "role", DatasetRole(self.role))

    def __str__(self) -> str:
        notes = [n for n in (self.via, "body" if self.in_macro_body else "") if n]
        if self.param is not None:
            notes.append(f"param {self.param}#{self.param_pos}")
        return f"{self.role} {self.name}" + (f" ({', '.join(notes)})" if notes else "")


# The dataset lists SasChunkMetadata computes from its dataset_refs. Also what
# the metadata accepts in their place as input: JSON written before
# dataset_refs, and callers that build metadata from lists (see
# SasChunkMetadata._dataset_lists_as_refs).
_DATASET_VIEWS = frozenset(
    {
        "input_datasets",
        "output_datasets",
        "dropped_datasets",
        "referenced_datasets",
        "referenced_librefs",
        "body_literal_inputs",
        "body_literal_outputs",
        "body_param_inputs",
        "body_param_outputs",
    }
)


_NO_VIEWS: dict[str, tuple[Any, ...]] = dict.fromkeys(_DATASET_VIEWS, ())


def _libref_of(name: str) -> str | None:
    """The libref of a two-level SAS name, or ``None`` (one-level, quoted path)."""
    return name.split(".", 1)[0] if "." in name and not name.startswith("'") else None


def _dataset_views(refs: tuple[SasDatasetRef, ...]) -> dict[str, tuple[Any, ...]]:
    """Every view of *refs* in one pass: names (or a parameter's ``(param,
    pos)``) in first-seen order, each once.

    A chunk's own references fill ``input_datasets``, ``output_datasets`` and
    ``dropped_datasets``; a macro body's fill the ``body_literal_*`` lists, or
    ``body_param_*`` when spelled through a parameter. UPDATE reads and writes.
    ``referenced_datasets`` is every name, sorted, whatever its role, and
    ``referenced_librefs`` the librefs those names hold (without the
    chunk's ``defines_librefs``, which the property adds).
    """
    if not refs:
        return _NO_VIEWS  # most chunks name no dataset; never mutated
    found: dict[str, dict[Any, None]] = {}
    for ref in refs:
        role = ref.role
        if ref.param is not None:
            key: Any = (ref.param, ref.param_pos)
            into = ("body_param_inputs", "body_param_outputs", None)
        elif ref.in_macro_body:
            key = ref.name
            into = ("body_literal_inputs", "body_literal_outputs", None)
        else:
            key = ref.name
            into = ("input_datasets", "output_datasets", "dropped_datasets")
        if role is DatasetRole.READ or role is DatasetRole.UPDATE:
            found.setdefault(into[0], {})[key] = None
        if role is DatasetRole.WRITE or role is DatasetRole.UPDATE:
            found.setdefault(into[1], {})[key] = None
        if role is DatasetRole.DROP and into[2] is not None:
            found.setdefault(into[2], {})[key] = None
    views = dict(_NO_VIEWS)
    for view, keys in found.items():
        views[view] = tuple(keys)
    names = sorted({ref.name for ref in refs})
    views["referenced_datasets"] = tuple(names)
    views["referenced_librefs"] = tuple(
        sorted({lib for name in names if (lib := _libref_of(name)) is not None})
    )
    return views


def _refs_from_lists(
    inputs: Iterable[str] = (),
    outputs: Iterable[str] = (),
    dropped: Iterable[str] = (),
    body_inputs: Iterable[str] = (),
    body_outputs: Iterable[str] = (),
    param_inputs: Iterable[Mapping[str, Any]] = (),
    param_outputs: Iterable[Mapping[str, Any]] = (),
    referenced: Iterable[str] = (),
) -> tuple[SasDatasetRef, ...]:
    """The dataset references behind dataset lists, in the order the views
    read them back: the chunk's inputs, outputs and drops, then the macro
    body's. Parameter entries are ``{"param": name, "pos": n}``. A
    *referenced* name no other list holds is a MENTION."""
    read, write = DatasetRole.READ, DatasetRole.WRITE
    refs = [SasDatasetRef(name, read) for name in inputs]
    refs += [SasDatasetRef(name, write) for name in outputs]
    refs += [SasDatasetRef(name, DatasetRole.DROP) for name in dropped]
    refs += [SasDatasetRef(name, read, in_macro_body=True) for name in body_inputs]
    refs += [SasDatasetRef(name, write, in_macro_body=True) for name in body_outputs]
    for entries, role in ((param_inputs, read), (param_outputs, write)):
        for entry in entries:
            param = str(entry["param"])
            refs.append(
                SasDatasetRef(
                    f"&{param}",
                    role,
                    in_macro_body=True,
                    param=param,
                    param_pos=int(entry["pos"]),
                )
            )
    named = {ref.name for ref in refs}
    refs += [
        SasDatasetRef(name, DatasetRole.MENTION)
        for name in dict.fromkeys(referenced)
        if name not in named
    ]
    return tuple(refs)


class SasDiagnostic(BaseModel):
    """A recoverable parsing or classification issue."""

    code: str
    message: str
    severity: SasDiagnosticSeverity = SasDiagnosticSeverity.WARNING
    start_line: int
    end_line: int | None = None
    source_id: str | None = None

    def model_post_init(self, __context: object) -> None:  # noqa: ANN001
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"SasDiagnostic  code={self.code}  severity={self.severity}  line={self.start_line}  source={self.source_id or '<inline>'}"
            )

    def __str__(self) -> str:
        span = (
            f"line {self.start_line}"
            if self.end_line is None or self.end_line == self.start_line
            else f"lines {self.start_line}-{self.end_line}"
        )
        source = f" [{self.source_id}]" if self.source_id else ""
        return f"[{self.severity}] {self.code} ({span}){source}: {self.message}"


class SasChunkMetadata(BaseModel):
    """Lightweight semantic metadata extracted from a chunk."""

    step_name: str | None = None
    proc_name: str | None = None
    macro_name: str | None = None
    labels: list[str] = Field(default_factory=list)
    defines_librefs: list[str] = Field(default_factory=list)
    includes: list[str] = Field(default_factory=list)
    options: list[str] = Field(default_factory=list)
    has_unclosed_block: bool = False

    macro_var_op: str | None = None
    global_statement_keyword: str | None = None

    declared_macro_vars: list[str] = Field(default_factory=list)
    referenced_macro_vars: list[str] = Field(default_factory=list)

    # The *values* the chunk's ``%LET`` statements assign, name → value, both
    # lowercased. Only values that could name a dataset, a library or part of
    # one are kept (``chunker.macro_vars.let_values``); any other value maps to
    # ``""``, meaning *unknown from here on*, so an earlier value cannot keep
    # answering for the variable. This is the symbol table
    # ``chunker.metadata.resolve_macro_var_refs`` expands ``&name`` references
    # against, not a record of every string the job holds.
    macro_var_values: dict[str, str] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def referenced_automatic_vars(self) -> list[str]:
        return [n for n in self.referenced_macro_vars if _is_automatic_macro_var(n)]

    recognized_functions: list[str] = Field(default_factory=list)
    recognized_call_routines: list[str] = Field(default_factory=list)
    # DATA step component objects the chunk declares (hash, hiter, javaobj,
    # logger, appender) — via DECLARE/DCL or the _NEW_ operator.
    component_objects: list[str] = Field(default_factory=list)
    # DATA step statements present in the chunk (``merge``, ``by``, ``retain``,
    # ``array``, ``output``, ...), plus the three keyword-less constructs the
    # scan derives: ``retain`` for a sum statement, ``subsetting_if`` for an
    # ``if <expr>;`` that drops rows, and ``dataset_option`` for ``keep=`` and
    # friends. Each names a distinct translation problem, so guidance can be
    # scoped to the steps that actually raise it instead of to every DATA step.
    data_step_statements: list[str] = Field(default_factory=list)

    # Every SAS dataset the chunk names and what it does with each — see
    # :class:`SasDatasetRef`. The dataset lists below are views of it. A tuple,
    # so it changes only by being replaced, which keeps the views' cache sound.
    dataset_refs: tuple[SasDatasetRef, ...] = ()
    defines_macros: list[str] = Field(default_factory=list)
    invokes_macros: list[str] = Field(default_factory=list)
    macro_param_names: list[str] = Field(default_factory=list)

    def _views(self) -> dict[str, tuple[Any, ...]]:
        """:func:`_dataset_views` of :attr:`dataset_refs`, computed once per
        tuple: the batcher reads the views of every chunk many times over.
        Kept outside the fields, so equality and dumps never see it."""
        if not self.dataset_refs:
            return _NO_VIEWS
        cached = self.__dict__.get("_dataset_views_cache")
        if cached is not None and cached[0] is self.dataset_refs:
            return cached[1]
        views = _dataset_views(self.dataset_refs)
        self.__dict__["_dataset_views_cache"] = (self.dataset_refs, views)
        return views

    @computed_field  # type: ignore[prop-decorator]
    @property
    def input_datasets(self) -> list[str]:
        """Datasets the chunk reads (READ and UPDATE), first-seen order."""
        return list(self._views()["input_datasets"])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def output_datasets(self) -> list[str]:
        """Datasets the chunk writes (WRITE and UPDATE), in the order the
        source names them — load-bearing: the last is what ``_LAST_`` holds
        after the chunk runs."""
        return list(self._views()["output_datasets"])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dropped_datasets(self) -> list[str]:
        """Datasets the chunk deletes."""
        return list(self._views()["dropped_datasets"])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def referenced_datasets(self) -> list[str]:
        """Every dataset the chunk names, sorted: what it reads, writes or
        deletes, what its macro body does (a parameter as ``&param``), and
        each ``%LET`` value written like a dataset (a MENTION)."""
        return list(self._views()["referenced_datasets"])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def referenced_librefs(self) -> list[str]:
        """The librefs of :attr:`referenced_datasets`' two-level names, and
        the ones the chunk assigns (``defines_librefs``), sorted. A libref
        still spelled through a macro variable is reported as written."""
        named = self._views()["referenced_librefs"]
        if not self.defines_librefs:
            return list(named)
        return sorted({*named, *self.defines_librefs})

    @computed_field  # type: ignore[prop-decorator]
    @property
    def body_literal_inputs(self) -> list[str]:
        """Datasets a ``%MACRO`` body reads under a name of its own."""
        return list(self._views()["body_literal_inputs"])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def body_literal_outputs(self) -> list[str]:
        """Datasets a ``%MACRO`` body writes under a name of its own."""
        return list(self._views()["body_literal_outputs"])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def body_param_inputs(self) -> list[dict[str, object]]:
        """``{"param": name, "pos": n}`` for each parameter a ``%MACRO`` body
        reads a dataset through; ``pos`` >= 0 positional, -1 keyword."""
        return [{"param": p, "pos": n} for p, n in self._views()["body_param_inputs"]]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def body_param_outputs(self) -> list[dict[str, object]]:
        """As :attr:`body_param_inputs`, for the datasets the body writes."""
        return [{"param": p, "pos": n} for p, n in self._views()["body_param_outputs"]]

    @model_validator(mode="before")
    @classmethod
    def _dataset_lists_as_refs(cls, data: Any) -> Any:
        """Accept the dataset lists in place of ``dataset_refs``: JSON written
        before they existed loads, and ``SasChunkMetadata(input_datasets=[…])``
        builds the references behind it. Alongside ``dataset_refs`` the lists
        are its views, serialised with it, and give way to it."""
        if type(data) is not dict or data.keys().isdisjoint(_DATASET_VIEWS):
            return data
        lists = {k: data[k] or () for k in _DATASET_VIEWS if k in data}
        data = {k: v for k, v in data.items() if k not in _DATASET_VIEWS}
        # referenced_librefs is not read: every libref it held comes back from
        # the names and defines_librefs.
        data.setdefault(
            "dataset_refs",
            _refs_from_lists(
                inputs=lists.get("input_datasets", ()),
                outputs=lists.get("output_datasets", ()),
                dropped=lists.get("dropped_datasets", ()),
                body_inputs=lists.get("body_literal_inputs", ()),
                body_outputs=lists.get("body_literal_outputs", ()),
                param_inputs=lists.get("body_param_inputs", ()),
                param_outputs=lists.get("body_param_outputs", ()),
                referenced=lists.get("referenced_datasets", ()),
            ),
        )
        return data

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        # A view named in `update` would be dropped without a word: its value
        # comes from dataset_refs. Say so instead.
        if update and (views := _DATASET_VIEWS & update.keys()):
            raise ValueError(
                f"{sorted(views)} are views of dataset_refs; rewrite the "
                f"references (map_dataset_names, add_dataset_refs) instead"
            )
        return super().model_copy(update=update, deep=deep)

    def map_dataset_names(
        self, rename: Callable[[SasDatasetRef], str | None]
    ) -> SasChunkMetadata:
        """This metadata with each dataset reference renamed to what *rename*
        returns for it, or dropped where that is ``None``: the one way to
        rewrite dataset names, so every view moves together.

        A parameter reference passes through untouched — each call site
        supplies its dataset. Returns ``self`` when nothing changes.
        """
        refs = self.dataset_refs
        names = [ref.name if ref.param is not None else rename(ref) for ref in refs]
        if all(new == ref.name for new, ref in zip(names, refs)):
            return self  # the common case, decided before any reference is hashed
        renamed = tuple(
            dict.fromkeys(
                ref if new == ref.name else replace(ref, name=new)
                for new, ref in zip(names, refs)
                if new is not None
            )
        )
        return self.model_copy(update={"dataset_refs": renamed})

    def add_dataset_refs(self, refs: Iterable[SasDatasetRef]) -> SasChunkMetadata:
        """This metadata with *refs* after its own; ``self`` when it has them all."""
        merged = tuple(dict.fromkeys([*self.dataset_refs, *refs]))
        if merged == self.dataset_refs:
            return self
        return self.model_copy(update={"dataset_refs": merged})

    produces_macrovars: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def consumes_macrovars(self) -> list[str]:
        own_params = set(self.macro_param_names)
        return [
            n
            for n in self.referenced_macro_vars
            if not _is_automatic_macro_var(n) and n not in own_params
        ]

    symput_scope_hazard: bool = False
    symput_hazard_vars: list[str] = Field(default_factory=list)

    control_flow_op: str | None = None
    contains_abort: bool = False
    contains_computed_goto: bool = False

    # Every external reference the chunk names, whatever kind of place it points
    # at — see :class:`SasPathRef`. One stored list rather than one per kind, so
    # there is one scan to keep correct and one merge rule to keep honest; the
    # per-kind views below are computed from it.
    external_refs: list[SasPathRef] = Field(default_factory=list)

    # Database-engine LIBNAMEs — see :class:`SasEngineRef`. Kept separate from
    # ``external_refs`` rather than folded in: those records answer "where is
    # this file", these answer "what system is this, and how did the job log in
    # to it", and the two have no field in common beyond the libref.
    engine_refs: list[SasEngineRef] = Field(default_factory=list)

    # Tables inside a database the chunk reads or writes — see
    # :class:`SasDbTableRef`. Separate from the SAS dataset lists above because
    # it is a different namespace: ``edw_export.current_nonip`` is an Oracle
    # owner and table, not a SAS libref and member, and filing it with them is
    # how an Oracle schema used to be reported as a missing LIBNAME. Sorted by
    # ``_db_table_sort_key``.
    db_tables: list[SasDbTableRef] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unresolved_dataset_refs(self) -> list[str]:
        """Dataset names the chunk still spells through a macro variable.

        A name reaches this list when no ``%LET`` in the corpus gave its
        reference a value — ``&lib_out_spd..cia_hso_excl`` — so its real
        library and member are only knowable by running SAS. They stay in
        ``referenced_datasets`` and the I/O lists exactly as written, because
        a dependency that cannot be resolved is still a dependency; this view
        is how a consumer tells those apart from the resolved names without
        re-scanning for ``&``. A macro body's parameter (``&ds``) is one too:
        only a call names its dataset.
        """
        return [d for d in self._views()["referenced_datasets"] if "&" in d]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def physical_paths(self) -> list[SasPathRef]:
        """Refs pointing at a filesystem location — the ones a target has to
        map to a volume or external location."""
        return [r for r in self.external_refs if r.location is PathLocation.FILESYSTEM]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def remote_paths(self) -> list[SasPathRef]:
        """Refs reaching a remote service (FTP, URL, ...) — network egress, not
        storage."""
        return [r for r in self.external_refs if r.location is PathLocation.REMOTE]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def email_refs(self) -> list[SasPathRef]:
        """Refs addressing a mailbox."""
        return [r for r in self.external_refs if r.location is PathLocation.EMAIL]

    def __str__(self) -> str:
        # Show only populated fields, so empty defaults don't drown out the rest.
        populated = ", ".join(
            f"{name}={value!r}"
            for name in type(self).model_fields
            if (value := getattr(self, name))
        )
        return f"SasChunkMetadata({populated or '<empty>'})"


class SasChunk(BaseModel):
    """A source-preserving semantic chunk with line/char offsets."""

    chunk_id: str
    source_id: str | None = None
    text: str
    kind: SasChunkKind
    title: str | None = None
    start_line: int
    end_line: int
    start_char: int
    end_char: int
    parent_id: str | None = None
    metadata: SasChunkMetadata = Field(default_factory=SasChunkMetadata)

    def model_post_init(self, __context: object) -> None:  # noqa: ANN001
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"SasChunk  id={self.chunk_id}  kind={self.kind.value}  lines={self.start_line}-{self.end_line}  source={self.source_id or '<inline>'}  parent={self.parent_id or 'none'}"
            )

    def __str__(self) -> str:
        title = f" '{self.title}'" if self.title else ""
        source = f" [{self.source_id}]" if self.source_id else ""
        return (
            f"SasChunk {self.chunk_id} [{self.kind.value}]{title} "
            f"lines {self.start_line}-{self.end_line}{source}"
        )


class SasChunkResult(BaseModel):
    """Output of SasSemanticChunker for a single file or text string."""

    source_id: str | None = None
    chunks: list[SasChunk] = Field(default_factory=list)
    diagnostics: list[SasDiagnostic] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:  # noqa: ANN001
        logger.info(
            f"SasChunkResult  source='{self.source_id or '<inline>'}'  chunks={len(self.chunks)}  diagnostics={len(self.diagnostics)}"
        )

    def __str__(self) -> str:
        return (
            f"SasChunkResult(source='{self.source_id or '<inline>'}', "
            f"chunks={len(self.chunks)}, diagnostics={len(self.diagnostics)})"
        )


class SasCorpus(BaseModel):
    """
    A named collection of :class:`SasChunkResult` objects, one per SAS file.

    This is the entry point for multi-file batching.  Build it by chunking
    each file independently and passing the results to
    :class:`~chunker.batcher.MultiFileBatcher`.

    Attributes
    ----------
    file_results
        Ordered list of per-file chunk results.  Order determines the
        default execution order when inter-file dependencies are absent
        (i.e. the order in which files would be submitted to SAS).
    """

    file_results: list[SasChunkResult] = Field(default_factory=list)

    @property
    def source_ids(self) -> list[str]:
        """Canonical source_id for every file in the corpus."""
        return [r.source_id or "<inline>" for r in self.file_results]

    @property
    def all_chunks(self) -> list[SasChunk]:
        """Flat list of every chunk across all files, in corpus order."""
        return [c for r in self.file_results for c in r.chunks]

    @property
    def all_diagnostics(self) -> list[SasDiagnostic]:
        """Flat list of every diagnostic across all files."""
        return [d for r in self.file_results for d in r.diagnostics]

    def model_post_init(self, __context: object) -> None:  # noqa: ANN001
        total_chunks = sum(len(r.chunks) for r in self.file_results)
        logger.info(
            f"SasCorpus  files={len(self.file_results)}  total_chunks={total_chunks}  source_ids={self.source_ids}"
        )

    def __str__(self) -> str:
        return (
            f"SasCorpus(files={len(self.file_results)}, "
            f"total_chunks={len(self.all_chunks)}, source_ids={self.source_ids})"
        )


class SasBatch(BaseModel):
    """
    An ordered group of inter-dependent :class:`SasChunk` objects that must
    be sent to the LLM together.

    Cross-file batches are possible: if ``File_A.sas`` produces a dataset
    that ``File_B.sas`` consumes, those chunks will appear in the same batch
    with ``source_files`` listing both files.

    Fields
    ------
    batch_id
        Zero-padded sequential id, e.g. ``"batch-001"``.
    is_global_context
        True for the (at most one) global-context batch: chunks whose
        outputs — macro definitions, %LET/%GLOBAL declarations, datasets —
        are consumed by two or more otherwise-independent batches.  It is
        always emitted first in the batch list so downstream consumers can
        process the shared context before any batch that depends on it,
        and it may legitimately contain a single chunk.
    chunks
        Member chunks in dependency-respecting, source-order sequence.
        Chunks from different files are interleaved so that producers always
        appear before their consumers.
    reason
        Human-readable explanation of every dependency edge that caused
        these chunks to be grouped.
    source_files
        Distinct ``source_id`` values of all member chunks, in the order
        they first appear.  Single-file batches have exactly one entry.
    input_datasets
        Datasets consumed by this batch but produced *outside* it.
    output_datasets
        Datasets produced by this batch (may feed later batches/singletons).
    required_macros
        Macro names invoked inside but not defined inside this batch.
    required_librefs
        Librefs referenced by this batch's dataset I/O but not assigned by
        a LIBNAME statement inside the batch, excluding the SAS-supplied
        default libraries (work, user, sashelp, sasuser, maps, mapssas).
        A non-empty list means the batch is not self-contained: it relies
        on LIBNAME assignments that live outside it (mirrors
        ``required_macros`` for the library namespace).
    defined_macros
        Macro names whose full definitions live inside this batch.
    produced_macrovars
        Macro variable names created inside this batch — via CALL SYMPUT/
        SYMPUTX or PROC SQL INTO, or declared with ``%LET`` /
        ``%GLOBAL`` / ``%LOCAL`` (mirrors ``output_datasets`` for the
        macro-variable namespace).
    required_macrovars
        Macro variable names referenced inside this batch (via ``&name``)
        but not produced inside it (mirrors ``input_datasets``).
        Automatic/system variables are never included here.
    standard_autocall_macros
        Names of well-known, SAS-provided autocall macros (``%left``,
        ``%trim``, ``%cmpres``, ...) invoked inside this batch.  These are
        deliberately excluded from ``required_macros`` — they ship with
        every SAS installation, so a call to one is never a missing
        dependency the user needs to locate, but the information is still
        surfaced here rather than silently dropped.
    """

    batch_id: str
    chunks: list[SasChunk] = Field(default_factory=list)
    reason: str = ""
    is_global_context: bool = False
    source_files: list[str] = Field(default_factory=list)
    input_datasets: list[str] = Field(default_factory=list)
    output_datasets: list[str] = Field(default_factory=list)
    required_macros: list[str] = Field(default_factory=list)
    required_librefs: list[str] = Field(default_factory=list)
    defined_macros: list[str] = Field(default_factory=list)
    produced_macrovars: list[str] = Field(default_factory=list)
    required_macrovars: list[str] = Field(default_factory=list)
    standard_autocall_macros: list[str] = Field(default_factory=list)

    @property
    def chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]

    @property
    def start_line(self) -> int:
        return self.chunks[0].start_line if self.chunks else 0

    @property
    def end_line(self) -> int:
        return self.chunks[-1].end_line if self.chunks else 0

    @property
    def is_cross_file(self) -> bool:
        """True when this batch spans more than one source file."""
        return len(self.source_files) > 1

    # ------------------------------------------------------------------
    # Aggregated construct metadata (deduplicated sets over member chunks).
    #
    # These roll up the per-chunk identifiers a batch actually uses so a
    # consumer can answer "does this batch use INTCK / a hash object / PROC
    # SQL?" with an O(1) hashed set membership test, instead of re-scanning
    # every chunk's metadata. The instruction-guidance layer keys targeted
    # reference/user instructions off exactly these sets (via the pipeline's
    # metadata -> ConstructKey mapping), so an instruction for a construct is
    # injected only when the construct is present in the batch.
    # ------------------------------------------------------------------

    @property
    def recognized_functions(self) -> set[str]:
        """SAS functions recognised across member chunks (e.g. ``intnx``)."""
        return {fn for c in self.chunks for fn in c.metadata.recognized_functions}

    @property
    def recognized_call_routines(self) -> set[str]:
        """CALL routines recognised across member chunks (e.g. ``symput``)."""
        return {
            r for c in self.chunks for r in c.metadata.recognized_call_routines
        }

    @property
    def component_objects(self) -> set[str]:
        """DATA-step component objects declared in the batch (``hash``, ...)."""
        return {o for c in self.chunks for o in c.metadata.component_objects}

    @property
    def data_step_statements(self) -> set[str]:
        """DATA-step statements used across member chunks (``merge``, ...)."""
        return {
            s for c in self.chunks for s in c.metadata.data_step_statements
        }

    @property
    def proc_names(self) -> set[str]:
        """Names of the PROCs the batch runs (e.g. ``sql``, ``means``)."""
        return {
            c.metadata.proc_name
            for c in self.chunks
            if c.kind is SasChunkKind.PROC_STEP and c.metadata.proc_name
        }

    @property
    def global_statement_keywords(self) -> set[str]:
        """Global-statement keywords present in the batch (``libname``, ...)."""
        return {
            c.metadata.global_statement_keyword
            for c in self.chunks
            if c.metadata.global_statement_keyword
        }

    @property
    def external_refs(self) -> list[SasPathRef]:
        """Every external reference the batch's chunks name, deduplicated.

        A list rather than a set like its neighbours above: the records are
        hashable, but a stable order is what makes batch output reproducible
        (invariant 9), and a consumer rendering a report wants them ordered.
        """
        return sorted(
            {r for c in self.chunks for r in c.metadata.external_refs},
            key=_path_ref_sort_key,
        )

    @property
    def engine_refs(self) -> list[SasEngineRef]:
        """Every database-engine LIBNAME the batch's chunks declare.

        Deduplicated and ordered on the same grounds as :attr:`external_refs`.
        """
        return sorted(
            {r for c in self.chunks for r in c.metadata.engine_refs},
            key=_engine_ref_sort_key,
        )

    @property
    def db_tables(self) -> list[SasDbTableRef]:
        """Every database table the batch's chunks read or write.

        Deduplicated and ordered on the same grounds as :attr:`external_refs`.
        """
        return sorted(
            {r for c in self.chunks for r in c.metadata.db_tables},
            key=_db_table_sort_key,
        )

    @property
    def physical_paths(self) -> list[SasPathRef]:
        """:attr:`external_refs` narrowed to filesystem locations."""
        return [r for r in self.external_refs if r.location is PathLocation.FILESYSTEM]

    @property
    def remote_paths(self) -> list[SasPathRef]:
        """:attr:`external_refs` narrowed to remote services."""
        return [r for r in self.external_refs if r.location is PathLocation.REMOTE]

    @property
    def unresolved_dataset_refs(self) -> list[str]:
        """Dataset names the batch spells through an unresolved macro variable.

        The batch-level view of
        :attr:`SasChunkMetadata.unresolved_dataset_refs`: every name whose
        library or member no ``%LET`` in the corpus supplied. Non-empty means
        part of this batch's data flow could not be traced statically, which a
        consumer translating it needs to say out loud rather than discover.
        """
        return sorted(
            {r for c in self.chunks for r in c.metadata.unresolved_dataset_refs}
        )

    @property
    def has_symput_scope_hazard(self) -> bool:
        """True if any member chunk carries a CALL SYMPUT scope hazard."""
        return any(c.metadata.symput_scope_hazard for c in self.chunks)

    @property
    def has_abort(self) -> bool:
        """True if any member chunk contains a macro %ABORT."""
        return any(c.metadata.contains_abort for c in self.chunks)

    @property
    def has_computed_goto(self) -> bool:
        """True if any member chunk contains a computed %GOTO."""
        return any(c.metadata.contains_computed_goto for c in self.chunks)

    def model_post_init(self, __context: object) -> None:  # noqa: ANN001
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"SasBatch  id={self.batch_id}  chunks={len(self.chunks)}  source_files={self.source_files}  cross_file={self.is_cross_file}  inputs={self.input_datasets}  outputs={self.output_datasets}"
            )

    def __str__(self) -> str:
        scope = "cross-file" if self.is_cross_file else "single-file"
        if self.is_global_context:
            scope += ", global-context"
        return (
            f"SasBatch {self.batch_id} ({scope}) chunks={len(self.chunks)} "
            f"lines {self.start_line}-{self.end_line} "
            f"source_files={self.source_files} "
            f"inputs={self.input_datasets} outputs={self.output_datasets} "
            f"required_librefs={self.required_librefs}"
        )


class SasBatchResult(BaseModel):
    """
    Output of :class:`~chunker.batcher.SasChunkBatcher` and
    :class:`~chunker.batcher.MultiFileBatcher`.

    One model serves both workflows: ``source_ids`` lists every file in the
    corpus (exactly one entry for a single-file run, ``"<inline>"`` for
    string input), and ``cross_file_batches`` is empty when only one file
    was batched.

    Attributes
    ----------
    source_ids
        Ordered list of all source file identifiers in the corpus.
    batches
        All multi-chunk dependency groups, including cross-file ones.
    singletons
        All independent chunks (no cross-chunk dependency edges).
    """

    source_ids: list[str] = Field(default_factory=list)
    batches: list[SasBatch] = Field(default_factory=list)
    singletons: list[SasChunk] = Field(default_factory=list)

    @property
    def source_id(self) -> str | None:
        """The lone source id of a single-file result (``"<inline>"`` for
        string input), or ``None`` when the corpus holds several files."""
        return self.source_ids[0] if len(self.source_ids) == 1 else None

    @property
    def cross_file_batches(self) -> list[SasBatch]:
        """Batches that span more than one source file."""
        return [b for b in self.batches if b.is_cross_file]

    @property
    def all_ordered_items(self) -> list[SasBatch | SasChunk]:
        """
        All items ordered by (file_index, start_line) so that the sequence
        respects both inter-file corpus order and intra-file source order.
        For a single-file result this reduces to plain start_line order.

        For cross-file batches the position is determined by the earliest
        chunk in the batch (i.e. the producing chunk).
        """
        file_rank = {sid: i for i, sid in enumerate(self.source_ids)}

        def _key(item: SasBatch | SasChunk) -> tuple[int, int]:
            if isinstance(item, SasBatch):
                first = item.chunks[0]
            else:
                first = item
            fid = first.source_id or "<inline>"
            return (file_rank.get(fid, 999), first.start_line)

        tagged = list(self.batches) + list(self.singletons)
        return sorted(tagged, key=_key)

    def model_post_init(self, __context: object) -> None:  # noqa: ANN001
        cf = sum(1 for b in self.batches if b.is_cross_file)
        logger.info(
            f"SasBatchResult  source_ids={self.source_ids}  batches={len(self.batches)}  cross_file_batches={cf}  singletons={len(self.singletons)}"
        )

    def __str__(self) -> str:
        return (
            f"SasBatchResult(source_ids={self.source_ids}, "
            f"batches={len(self.batches)}, "
            f"cross_file_batches={len(self.cross_file_batches)}, "
            f"singletons={len(self.singletons)})"
        )
