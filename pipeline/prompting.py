"""Item → prompt mapping and formatting for the SAS LLM pipeline.

Two things live here, both pure functions over chunker models:

- the SAS-metadata → prompt_builder mapping (retrieval query, construct keys,
  kind and metadata-flag scope tokens) — this is the sole such mapping, and
  keeping it here (not in ``prompt_builder``) is what lets that package stay
  free of any ``chunker`` import;
- the batch → user-message formatting over the templates in
  :mod:`pipeline.constants`.

Items are :class:`SasBatch` only: ``coalesce_into_batches`` wraps every
singleton chunk before the pipeline prompts anything, so no bare
:class:`SasChunk` ever reaches the LLM (per-chunk helpers here exist only as
ingredients of the batch-level functions).

Logger name: ``pipeline.prompting``.
"""

from __future__ import annotations

import logging

from chunker.models import (
    SasBatch,
    SasChunk,
    SasDiagnostic,
)
from prompt_builder import ConstructKey

from .constants import (
    _BATCH_CONTEXT_TEMPLATE,
    _BATCH_MEMBER_TEMPLATE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------


def _fmt_list(xs: list[str] | None) -> str:
    return ", ".join(xs) if xs else "none"


# ---------------------------------------------------------------------------
# Item -> instruction-retrieval query / constructs
# ---------------------------------------------------------------------------


def _query_for_chunk(chunk: SasChunk) -> str:
    """Free-text retrieval query from a chunk's *constructs*, not its source.

    Dataset names and literal source text are retrieval noise for reference
    guidance, which is organised by construct; kind, title, and the recognised
    functions / routines / statements are the signal.
    """
    m = chunk.metadata
    tokens = [chunk.kind.value.replace("_", " "), chunk.title or ""]
    if m.proc_name:
        tokens.append(m.proc_name)
    tokens.extend(m.recognized_functions)
    tokens.extend(m.recognized_call_routines)
    # "hash object", "hiter object", ... — the reference guides discuss
    # component objects by that phrasing ("hash object", "hash table").
    tokens.extend(f"{obj} object" for obj in m.component_objects)
    if m.global_statement_keyword:
        tokens.append(m.global_statement_keyword)
    if m.control_flow_op:
        tokens.append(m.control_flow_op)
    tokens.extend(m.invokes_macros)
    return " ".join(t for t in tokens if t)


def _query_for_item(item: SasBatch) -> str:
    return " ".join(_query_for_chunk(c) for c in item.chunks)


def _constructs_for_item(item: SasBatch) -> list[ConstructKey]:
    """The SAS constructs an item uses, as reference-lookup keys.

    Driven off the batch's aggregated identifier *sets* (its set-valued
    properties): each name becomes a frozen — therefore hashable —
    :class:`ConstructKey`, deduplicated through a hashed ``seen`` set, so the
    selector's construct lookup (also hash-based) fires an instruction for a
    construct only when that construct is actually present in the batch. Sets
    are iterated in sorted order to keep the key sequence deterministic.

    Hazard flags add their canonical construct even when the name extractor
    missed it, so the selector still pulls the SYMPUT / %GOTO / %ABORT section.
    """
    keys: list[ConstructKey] = []
    seen: set[ConstructKey] = set()

    def add(kind: str, name: str | None) -> None:
        if not name:
            return
        key = ConstructKey(kind=kind, name=name.lower())
        if key not in seen:
            seen.add(key)
            keys.append(key)

    for name in sorted(item.proc_names):
        add("proc", name)
    for name in sorted(item.recognized_functions):
        add("function", name)
    for name in sorted(item.recognized_call_routines):
        add("call_routine", name)
    for name in sorted(item.component_objects):
        add("component_object", name)
    for name in sorted(item.global_statement_keywords):
        add("global_statement", name)
    if item.has_symput_scope_hazard:
        add("call_routine", "symput")
    if item.has_computed_goto:
        add("macro_statement", "goto")
    if item.has_abort:
        add("macro_statement", "abort")
    return keys


def _kinds_for_item(item: SasBatch) -> set[str]:
    """The SasChunkKind values an item uses, as ``[kind: ...]`` scope tokens."""
    return {c.kind.value for c in item.chunks}


# Metadata predicate flags, keyed by the ``[meta: ...]`` token an instruction
# scopes on. Each maps to a metadata attribute that is truthy when the flag
# holds; the pipeline owns this vocabulary so prompt_builder treats the tokens
# as opaque. Kept in sync with the docstring in user_instructions.py.
_META_FLAG_ATTRS: tuple[tuple[str, str], ...] = (
    ("symput_hazard", "symput_scope_hazard"),
    ("abort", "contains_abort"),
    ("computed_goto", "contains_computed_goto"),
    ("component_object", "component_objects"),
    ("unclosed_block", "has_unclosed_block"),
    ("includes", "includes"),
    ("defines_macros", "defines_macros"),
    ("invokes_macros", "invokes_macros"),
    ("produces_macrovars", "produces_macrovars"),
    ("automatic_vars", "referenced_automatic_vars"),
)


def _meta_flags_for_item(item: SasBatch) -> set[str]:
    """The metadata predicate flags an item raises, as ``[meta: ...]`` tokens.

    Unioned over member chunks (a batch flag holds if any member raises it),
    so an instruction scoped ``[meta: symput_hazard]`` fires for a batch that
    contains a SYMPUT scope hazard anywhere inside it.
    """
    flags: set[str] = set()
    for chunk in item.chunks:
        m = chunk.metadata
        for token, attr in _META_FLAG_ATTRS:
            if token not in flags and getattr(m, attr):
                flags.add(token)
    return flags


def _diagnostics_for_chunk(
    chunk: SasChunk, diagnostics: list[SasDiagnostic]
) -> list[SasDiagnostic]:
    return [
        d
        for d in diagnostics
        if d.source_id in (None, chunk.source_id)
        and chunk.start_line <= d.start_line <= chunk.end_line
    ]


def _diagnostics_for_batch(
    batch: SasBatch, diagnostics: list[SasDiagnostic]
) -> list[SasDiagnostic]:
    seen: list[SasDiagnostic] = []
    seen_keys: set[tuple[str, int, str | None]] = set()
    for member in batch.chunks:
        for d in _diagnostics_for_chunk(member, diagnostics):
            key = (d.code, d.start_line, d.source_id)
            if key not in seen_keys:
                seen.append(d)
                seen_keys.add(key)
    return seen


def _format_batch_message(
    batch: SasBatch,
    index: int,
    total: int,
    diagnostics: list[SasDiagnostic],
) -> str:
    diags = _diagnostics_for_batch(batch, diagnostics)
    members = "\n".join(
        _BATCH_MEMBER_TEMPLATE.format(
            chunk_id=c.chunk_id,
            kind=c.kind.value,
            source_id=c.source_id or "unknown",
            start_line=c.start_line,
            end_line=c.end_line,
            title=c.title or "—",
            text=c.text,
        )
        for c in batch.chunks
    )
    msg = _BATCH_CONTEXT_TEMPLATE.format(
        batch_id=batch.batch_id,
        index=index,
        total_items=total,
        is_cross_file="yes" if batch.is_cross_file else "no",
        source_files=_fmt_list(batch.source_files),
        chunk_count=len(batch.chunks),
        start_line=batch.start_line,
        end_line=batch.end_line,
        reason=batch.reason or "none",
        input_datasets=_fmt_list(batch.input_datasets),
        output_datasets=_fmt_list(batch.output_datasets),
        required_macros=_fmt_list(batch.required_macros),
        defined_macros=_fmt_list(batch.defined_macros),
        required_librefs=_fmt_list(batch.required_librefs),
        standard_autocall_macros=_fmt_list(batch.standard_autocall_macros),
        required_macrovars=_fmt_list(batch.required_macrovars),
        produced_macrovars=_fmt_list(batch.produced_macrovars),
        sas_functions=_fmt_list(sorted(batch.recognized_functions)),
        call_routines=_fmt_list(sorted(batch.recognized_call_routines)),
        component_objects=_fmt_list(sorted(batch.component_objects)),
        global_statement_keywords=_fmt_list(sorted(batch.global_statement_keywords)),
        symput_hazard="yes" if batch.has_symput_scope_hazard else "no",
        contains_abort="yes" if batch.has_abort else "no",
        contains_computed_goto="yes" if batch.has_computed_goto else "no",
        diagnostics="; ".join(f"[{d.code}] {d.message}" for d in diags) or "none",
        members=members,
    )
    logger.debug(
        f"_format_batch_message: batch={batch.batch_id}  members={len(batch.chunks)}  prompt_chars={len(msg)}"
    )
    return msg
