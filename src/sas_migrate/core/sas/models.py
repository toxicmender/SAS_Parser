"""Pydantic models for the SAS semantic chunker and batcher. See chunker/README.md."""

from __future__ import annotations

import logging
from enum import StrEnum

from pydantic import BaseModel, Field, computed_field

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
        ``file``, ``include``, ``proc_import``, ``proc_export``, ``ods``,
        ``sasautos``.
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


class SasDiagnostic(BaseModel):
    """A recoverable parsing or classification issue."""

    code: str
    message: str
    severity: SasDiagnosticSeverity = SasDiagnosticSeverity.WARNING
    start_line: int
    end_line: int | None = None
    source_id: str | None = None

    def model_post_init(self, __context: object, /) -> None:
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
    referenced_librefs: list[str] = Field(default_factory=list)
    referenced_datasets: list[str] = Field(default_factory=list)
    defines_librefs: list[str] = Field(default_factory=list)
    includes: list[str] = Field(default_factory=list)
    options: list[str] = Field(default_factory=list)
    has_unclosed_block: bool = False

    macro_var_op: str | None = None
    global_statement_keyword: str | None = None

    declared_macro_vars: list[str] = Field(default_factory=list)
    referenced_macro_vars: list[str] = Field(default_factory=list)

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

    input_datasets: list[str] = Field(default_factory=list)
    output_datasets: list[str] = Field(default_factory=list)
    defines_macros: list[str] = Field(default_factory=list)
    invokes_macros: list[str] = Field(default_factory=list)

    body_literal_inputs: list[str] = Field(default_factory=list)
    body_literal_outputs: list[str] = Field(default_factory=list)
    # Each entry: {"param": "<name>", "pos": <int>} — pos >= 0 positional, -1 keyword.
    body_param_inputs: list[dict[str, object]] = Field(default_factory=list)
    body_param_outputs: list[dict[str, object]] = Field(default_factory=list)
    macro_param_names: list[str] = Field(default_factory=list)

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
            f"{name}={value!r}" for name, value in self.__dict__.items() if value
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

    def model_post_init(self, __context: object, /) -> None:
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

    def model_post_init(self, __context: object, /) -> None:
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

    def model_post_init(self, __context: object, /) -> None:
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
    def physical_paths(self) -> list[SasPathRef]:
        """:attr:`external_refs` narrowed to filesystem locations."""
        return [r for r in self.external_refs if r.location is PathLocation.FILESYSTEM]

    @property
    def remote_paths(self) -> list[SasPathRef]:
        """:attr:`external_refs` narrowed to remote services."""
        return [r for r in self.external_refs if r.location is PathLocation.REMOTE]

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

    def model_post_init(self, __context: object, /) -> None:
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

    def model_post_init(self, __context: object, /) -> None:
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
