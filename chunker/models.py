"""
models.py — Pydantic models for the SAS semantic chunker and batcher.

Single-file models
------------------
SasChunkKind, SasDiagnosticSeverity, SasDiagnostic,
SasChunkMetadata, SasChunk, SasChunkResult

Single-file batcher models
--------------------------
SasBatch, SasBatchResult

Multi-file models
-----------------
SasCorpus            — collection of SasChunkResult objects (one per file)
SasMultiBatchResult  — cross-file batcher output; superset of SasBatchResult
"""

from __future__ import annotations

import logging
from enum import StrEnum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunk classification
# ---------------------------------------------------------------------------


class SasChunkKind(StrEnum):
    """Semantic unit types recognised by the chunker."""

    DATA_STEP = "DATA_STEP"
    PROC_STEP = "PROC_STEP"
    MACRO_DEFINITION = "MACRO_DEFINITION"
    MACRO_CALL = "MACRO_CALL"
    # %if/%then/%else/%do/%end/%return/%goto/%abort appearing as a
    # standalone statement *outside* any macro definition (legal, per the
    # SAS Macro Language Reference Ch. 12, for %if/%then/%else only; the
    # others are macro-definition-only and would represent malformed
    # source if seen here — recognised defensively regardless).  The SAME
    # constructs appearing *inside* a %macro...%mend body remain part of
    # that single MACRO_DEFINITION chunk's text, exactly as before — this
    # kind only applies to open-code occurrences (ROADMAP Phase 3).
    MACRO_CONTROL_FLOW = "MACRO_CONTROL_FLOW"
    INCLUDE = "INCLUDE"
    GLOBAL_STATEMENT = "GLOBAL_STATEMENT"
    # A standalone RUN;/QUIT; that is *not* closing a DATA or PROC step —
    # e.g. a stray RUN; written after a global LIBNAME/FILENAME/TITLE
    # statement (a no-op in SAS, but common in hand-written code).  Inside a
    # DATA/PROC block a RUN;/QUIT; still terminates that block and stays part
    # of its chunk; this kind only applies to open-code occurrences that
    # would otherwise fall through to UNKNOWN_STATEMENT_GROUP and raise a
    # spurious UNRECOGNIZED_SOURCE_REGION diagnostic.
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


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class SasDiagnostic(BaseModel):
    """A recoverable parsing or classification issue."""

    code: str
    message: str
    severity: SasDiagnosticSeverity = SasDiagnosticSeverity.WARNING
    start_line: int
    end_line: int | None = None
    # Which file this diagnostic came from (populated in multi-file mode)
    source_id: str | None = None

    def model_post_init(self, __context: object) -> None:  # noqa: ANN001
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


# ---------------------------------------------------------------------------
# Chunk metadata
# ---------------------------------------------------------------------------


class SasChunkMetadata(BaseModel):
    """Lightweight semantic metadata extracted from a chunk."""

    # ── legacy general fields (used by prompts and reporting) ────────────────
    step_name: str | None = None
    proc_name: str | None = None
    macro_name: str | None = None
    labels: list[str] = Field(default_factory=list)
    referenced_librefs: list[str] = Field(default_factory=list)
    referenced_datasets: list[str] = Field(default_factory=list)
    # Librefs this chunk *assigns* via a LIBNAME statement (lowercased).
    # A LIBNAME may legally appear inside a DATA/PROC body, so this is
    # extracted for every chunk kind, not just GLOBAL_STATEMENT.  Caveat:
    # ``libname x clear;`` (deassignment) still reports ``x`` here — the
    # extraction is positional, not temporal.  The batcher subtracts these
    # from a batch's used librefs to derive ``SasBatch.required_librefs``.
    defines_librefs: list[str] = Field(default_factory=list)
    defined_macros: list[str] = Field(default_factory=list)
    called_macros: list[str] = Field(default_factory=list)
    includes: list[str] = Field(default_factory=list)
    options: list[str] = Field(default_factory=list)
    has_unclosed_block: bool = False

    # ``%let``/``%global``/``%local``/``%put`` all currently share the same
    # SasChunkKind (GLOBAL_STATEMENT), alongside libname/filename/title/
    # footnote/ods, with no way to tell them apart.  This field distinguishes
    # the four macro-variable-related statements from each other and from
    # the non-macro GLOBAL_STATEMENT members, without changing the chunk's
    # kind (which several tests already pin to GLOBAL_STATEMENT for these
    # four statements).  ``None`` for every chunk that isn't one of these
    # four — including the other GLOBAL_STATEMENT-classified statements.
    macro_var_op: str | None = None

    # The leading statement keyword of a GLOBAL_STATEMENT chunk, normalised
    # to lowercase with any trailing occurrence digits removed (``title2`` ->
    # ``title``).  One of ``let``/``put``/``global``/``local`` (the four
    # macro-variable statements, mirroring ``macro_var_op``) or one of the
    # non-macro global statements the chunker folds into GLOBAL_STATEMENT:
    # ``libname``/``filename``/``title``/``footnote``/``ods``.  ``None`` for
    # every chunk whose kind is not GLOBAL_STATEMENT.  Gives a consumer the
    # specific statement type without having to re-parse the chunk text.
    global_statement_keyword: str | None = None

    # ── automatic (system) macro variable references ────────────────────────
    # Every automatic macro variable SAS itself provides (&sysdate, &syslast,
    # &sysparm, ...) begins with the reserved "SYS" prefix (SAS Macro
    # Language: Reference, Ch. 12 "Automatic Macro Variables" — confirmed
    # across all ~60 of them).  Rather than maintaining an enumerated lookup
    # table, a chunk's text is scanned for any "&sys..." reference and the
    # matched names are recorded here.  These are read-mostly, system-
    # provided values — never a corpus-local dependency — so this field lets
    # a future macro-variable dependency graph (see ROADMAP Phase 2) exclude
    # them from "unresolved external variable" treatment for free.
    referenced_automatic_vars: list[str] = Field(default_factory=list)

    # ── macro-variable declarations and references (macro-language level) ────
    #
    # These two fields describe macro variables at the *source-text* level —
    # what this chunk explicitly declares, and every ``&name`` it references —
    # and are deliberately distinct from the Phase-2 producer/consumer edges
    # below (``produces_macrovars`` / ``consumes_macrovars``), which track the
    # narrower CALL SYMPUT/SYMPUTX and PROC SQL INTO data-flow used by the
    # batcher.
    #
    # ``declared_macro_vars`` — names introduced by ``%LET name = ...`` and by
    # ``%GLOBAL``/``%LOCAL`` declaration lists.  These are the macro variables
    # this chunk brings into existence via the macro language itself (as
    # opposed to the DATA-step side effects tracked in ``produces_macrovars``).
    declared_macro_vars: list[str] = Field(default_factory=list)

    # ``referenced_macro_vars`` — every ``&name`` reference in this chunk's
    # text, including automatic (``&sys*``) variables.  This is the complete
    # reference set; the filtered, dependency-oriented view lives in
    # ``consumes_macrovars`` (automatics and own-parameters excluded).
    referenced_macro_vars: list[str] = Field(default_factory=list)

    # ── recognised SAS functions and CALL routines ──────────────────────────
    #
    # Names of DATA-step functions (called as ``name(...)``) and CALL routines
    # (invoked as ``CALL name(...)``) recognised in this chunk against the
    # published SAS 9.4 Functions and CALL Routines: Reference dictionary
    # (see ``chunker._SAS_FUNCTIONS`` / ``_SAS_CALL_ROUTINES``).  These give an
    # LLM translator an at-a-glance inventory of the built-ins a chunk relies
    # on — several of which have no direct one-to-one target-language
    # equivalent and need explicit handling.
    recognized_functions: list[str] = Field(default_factory=list)
    recognized_call_routines: list[str] = Field(default_factory=list)

    # ── directed I/O edges — used by the batcher ─────────────────────────────
    input_datasets: list[str] = Field(default_factory=list)
    output_datasets: list[str] = Field(default_factory=list)
    defines_macros: list[str] = Field(default_factory=list)
    invokes_macros: list[str] = Field(default_factory=list)

    # ── macro body I/O — populated for MACRO_DEFINITION chunks only ────────
    #
    # The body of a %MACRO block may reference datasets in two ways:
    #
    #   Literal   — a hard-coded name, e.g. ``data work.base;``
    #               Resolvable purely from the macro source text.
    #               Stored in ``body_literal_inputs`` / ``body_literal_outputs``.
    #
    #   Parameterised — a macro variable reference, e.g. ``data &ds.;``
    #               Only resolvable at the call site where the argument value
    #               is known.  Stored as (param_name, role) in
    #               ``body_param_inputs`` / ``body_param_outputs`` where
    #               *param_name* is the macro parameter name (without &)
    #               and *role* is the positional index (0-based) of that
    #               parameter in the macro signature, or -1 if it is a
    #               keyword parameter.
    #
    # The batcher uses these fields in two complementary passes:
    #   Pass A (literal)      — adds literal body outputs to ``produces_ds``
    #                           for the MACRO_DEFINITION chunk itself.
    #   Pass B (parameterised) — at each MACRO_CALL site, substitutes the
    #                            actual argument values into the param names
    #                            and adds the resolved dataset names to
    #                            ``produces_ds`` for that call-site chunk.
    body_literal_inputs: list[str] = Field(default_factory=list)
    body_literal_outputs: list[str] = Field(default_factory=list)
    # Each entry: {"param": "<name>", "pos": <int|-1>}
    # pos >= 0  → positional parameter at that index
    # pos == -1 → keyword parameter (has a default value)
    body_param_inputs: list[dict[str, object]] = Field(default_factory=list)
    body_param_outputs: list[dict[str, object]] = Field(default_factory=list)
    # Ordered list of macro parameter names as they appear in the signature.
    # Used by the batcher to build the positional arg→param mapping.
    macro_param_names: list[str] = Field(default_factory=list)

    # ── macro-variable producer/consumer edges (ROADMAP Phase 2) ────────────
    #
    # Three SAS constructs create a macro variable as a *side effect* of
    # DATA-step or PROC-step execution, rather than via %LET:
    #   - CALL SYMPUT(name, value) / CALL SYMPUTX(name, value <, scope>)
    #     inside a DATA step (or a DATA step embedded in a macro's body —
    #     which is the *same* chunk as the enclosing MACRO_DEFINITION, since
    #     the chunker does not split macro bodies into nested chunks).
    #   - PROC SQL ... INTO :var [, :var2 ...] | :var1-:varN | :var SEPARATED BY ...
    #
    # ``produces_macrovars`` is the producer side (mirrors ``output_datasets``);
    # only populated when the macro-variable *name* is statically resolvable
    # (a literal quoted string, or an explicit ``:name`` in a SQL INTO
    # clause) — a name built from a DATA step expression or variable is left
    # unresolved rather than guessed.
    produces_macrovars: list[str] = Field(default_factory=list)

    # ``consumes_macrovars`` is the consumer side (mirrors ``input_datasets``):
    # every "&name" reference in this chunk's text, *excluding* automatic
    # variables (already tracked separately in ``referenced_automatic_vars``)
    # and, for MACRO_DEFINITION chunks, the macro's own declared parameters
    # (already tracked via ``macro_param_names`` / ``body_param_*`` — those
    # are call-site-resolved, not a corpus-level macro-variable dependency).
    consumes_macrovars: list[str] = Field(default_factory=list)

    # ── CALL SYMPUT/SYMPUTX local-scope hazard (ROADMAP Phase 2, C5c) ───────
    #
    # Per SAS Macro Language: Reference, Ch. 5 "Special Cases of Scope":
    # CALL SYMPUT/SYMPUTX stores the macro variable in the *current* symbol
    # table if that table is not empty (i.e. the enclosing macro has at
    # least one local variable — a declared parameter or an explicit
    # %LOCAL) — otherwise it walks up to the closest non-empty table.  A
    # macro with parameters therefore silently creates a *local* variable
    # via CALL SYMPUT, even when the author's evident intent (set once,
    # read later from outside the macro) requires a *global* one.  The
    # downstream `&var` reference then resolves to a bare
    # ``WARNING: Apparent symbolic reference ... not resolved`` rather than
    # the intended value — a genuine, well-documented, easy-to-miss defect
    # pattern worth surfacing explicitly to an LLM translator.
    #
    # This flag is suppressed when CALL SYMPUTX's optional third argument
    # explicitly forces global scope (a literal starting with 'G'), since
    # the author has then made the scope choice deliberately.
    symput_scope_hazard: bool = False
    # Names of the macro variables at risk (only the ones whose name is
    # statically resolvable to a literal string); empty if the hazard flag
    # is True only because of an unresolvable/dynamic name.
    symput_hazard_vars: list[str] = Field(default_factory=list)

    # ── control-flow recognition (ROADMAP Phase 3) ───────────────────────────
    #
    # Which specific control-flow statement this chunk is, when its kind is
    # MACRO_CONTROL_FLOW — one of "if", "else", "do", "end", "return",
    # "goto", "abort".  ``None`` for every other chunk kind.  Mirrors the
    # ``macro_var_op`` pattern from Phase 1 exactly.
    control_flow_op: str | None = None

    # True if "%abort" appears anywhere in a MACRO_DEFINITION chunk's body.
    # %ABORT is macro-definition-only (Ch. 19) and stops not just the
    # macro but the current DATA step / session / job — high-severity
    # enough to surface distinctly regardless of how deeply nested inside
    # %if/%do blocks it is, rather than requiring a consumer to re-scan
    # the chunk's raw text themselves.
    contains_abort: bool = False

    # True if a *computed* %GOTO (Ch. 5: a %goto whose label contains "&"
    # or "%") appears anywhere in a MACRO_DEFINITION chunk's body.  This is
    # one of the three documented conditions (alongside a non-empty local
    # symbol table) that forces CALL SYMPUT/SYMPUTX into local scope —
    # surfaced here for the same reason as the scope hazard fields above,
    # and folded into `_macro_has_local_scope()`'s own check (see chunker.py).
    contains_computed_goto: bool = False

    def __str__(self) -> str:
        # Show only the fields that carry information, so the many empty
        # defaults don't drown out the handful that are actually populated.
        populated = ", ".join(
            f"{name}={value!r}" for name, value in self.__dict__.items() if value
        )
        return f"SasChunkMetadata({populated or '<empty>'})"


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


class SasChunk(BaseModel):
    """A source-preserving semantic chunk with line/char offsets."""

    chunk_id: str
    # Identifies which file this chunk came from.
    # In single-file mode this equals the file path or "<inline>".
    # In multi-file mode it is set to the canonical file path so that
    # cross-file dependency edges can be resolved unambiguously.
    source_id: str | None = None
    text: str
    kind: SasChunkKind
    title: str | None = None
    start_line: int
    end_line: int
    start_char: int
    end_char: int
    # Non-None for child chunks produced by the oversized-split pass.
    parent_id: str | None = None
    metadata: SasChunkMetadata = Field(default_factory=SasChunkMetadata)

    def model_post_init(self, __context: object) -> None:  # noqa: ANN001
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


# ---------------------------------------------------------------------------
# Single-file chunker result
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Multi-file corpus
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Batch (shared by single-file and multi-file batcher)
# ---------------------------------------------------------------------------


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
        macro-variable namespace; ROADMAP Phase 2).
    required_macrovars
        Macro variable names referenced inside this batch (via ``&name``)
        but not produced inside it (mirrors ``input_datasets``; ROADMAP
        Phase 2).  Automatic/system variables are never included here.
    standard_autocall_macros
        Names of well-known, SAS-provided autocall macros (``%left``,
        ``%trim``, ``%cmpres``, ... — ROADMAP Phase 5, F2b) invoked inside
        this batch.  These are deliberately excluded from
        ``required_macros`` — they ship with every SAS installation, so a
        call to one is never a missing dependency the user needs to
        locate, but the information is still surfaced here rather than
        silently dropped.
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


# ---------------------------------------------------------------------------
# Single-file batcher result
# ---------------------------------------------------------------------------


class SasBatchResult(BaseModel):
    """
    Output of :class:`~chunker.batcher.SasChunkBatcher` (single file).
    """

    source_id: str | None = None
    batches: list[SasBatch] = Field(default_factory=list)
    singletons: list[SasChunk] = Field(default_factory=list)

    @property
    def all_ordered_items(self) -> list[SasBatch | SasChunk]:
        """
        Batches and singletons merged back into original source order,
        sorted by the start_line of the first chunk in each item.
        """
        tagged: list[tuple[int, SasBatch | SasChunk]] = [
            (b.start_line, b) for b in self.batches
        ] + [(c.start_line, c) for c in self.singletons]
        return [item for _, item in sorted(tagged, key=lambda t: t[0])]

    def model_post_init(self, __context: object) -> None:  # noqa: ANN001
        logger.info(
            f"SasBatchResult  source='{self.source_id or '<inline>'}'  batches={len(self.batches)}  singletons={len(self.singletons)}"
        )

    def __str__(self) -> str:
        return (
            f"SasBatchResult(source='{self.source_id or '<inline>'}', "
            f"batches={len(self.batches)}, singletons={len(self.singletons)})"
        )


# ---------------------------------------------------------------------------
# Multi-file batcher result
# ---------------------------------------------------------------------------


class SasMultiBatchResult(BaseModel):
    """
    Output of :class:`~chunker.batcher.MultiFileBatcher`.

    Structurally identical to :class:`SasBatchResult` but carries the full
    set of source files and exposes cross-file batch statistics.

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
    def cross_file_batches(self) -> list[SasBatch]:
        """Batches that span more than one source file."""
        return [b for b in self.batches if b.is_cross_file]

    @property
    def all_ordered_items(self) -> list[SasBatch | SasChunk]:
        """
        All items ordered by (file_index, start_line) so that the sequence
        respects both inter-file corpus order and intra-file source order.

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
            f"SasMultiBatchResult  source_ids={self.source_ids}  batches={len(self.batches)}  cross_file_batches={cf}  singletons={len(self.singletons)}"
        )

    def __str__(self) -> str:
        return (
            f"SasMultiBatchResult(source_ids={self.source_ids}, "
            f"batches={len(self.batches)}, "
            f"cross_file_batches={len(self.cross_file_batches)}, "
            f"singletons={len(self.singletons)})"
        )
