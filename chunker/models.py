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
