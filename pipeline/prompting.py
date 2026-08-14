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
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from chunker.keywords import SAS_FUNCTION_CATEGORIES
from chunker.models import (
    SasBatch,
    SasChunk,
    SasDiagnostic,
)
import token_budget as tokens
from prompt_builder import ConstructKey
from target_language import TargetLanguage

if TYPE_CHECKING:  # complexity is loaded lazily — see target_for_item
    from complexity.fallback import TargetChoice

from .constants import (
    _BATCH_CONTEXT_TEMPLATE,
    _BATCH_MEMBER_TEMPLATE,
    _TARGET_OVERRIDE_TEMPLATE,
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

    Every recognised function/routine also contributes a ``category`` key
    (:data:`~chunker.keywords.SAS_FUNCTION_CATEGORIES`), so an instruction can
    be scoped to a whole family — ``[category: date_time]`` — instead of
    enumerating its members. Category keys are emitted after the specific ones
    so the more precise match is offered first, and they never hit the
    reference corpus: PDF sections are titled per function, so no reference
    chunk carries a ``category`` key. They exist for user instructions.
    """
    return _construct_keys(
        procs=item.proc_names,
        functions=item.recognized_functions,
        call_routines=item.recognized_call_routines,
        component_objects=item.component_objects,
        statements=item.data_step_statements,
        global_statements=item.global_statement_keywords,
        symput_hazard=item.has_symput_scope_hazard,
        computed_goto=item.has_computed_goto,
        abort=item.has_abort,
    )


def _profile_constructs_for_item(item: SasBatch) -> list[tuple[str, str]]:
    """The item's constructs in the *complexity profiles'* vocabulary.

    :func:`_constructs_for_item` names constructs for the instruction selector,
    whose catalogue is titled per construct; the profiles' catalogue is keyed
    ``{construct_kind: {name: spec}}``. The two vocabularies agree on ``proc``,
    ``function``, ``call_routine``, ``component_object`` and
    ``global_statement``, and :data:`complexity.fallback.COMPARED_KINDS` ignores
    the rest.

    Two kinds have no counterpart in the selector's vocabulary and are added
    here:

    ``kind``
        The profiles rate a chunk *kind* (``MACRO_DEFINITION``) and
        ``_construct_keys`` never emits one. It matters more than everything
        else put together, because ``%MACRO`` is the construct pure SQL most
        obviously cannot host.
    ``detector``
        The DATA step's imperative constructs — ``DO`` loops, ``LINK``/
        ``RETURN``, ``MERGE`` without ``BY`` — which
        :class:`~chunker.models.SasChunkMetadata` does not report because they
        are found by scanning source text, not by naming an identifier. They are
        why :mod:`complexity.detectors` exists, and two of them (``do_loop`` and
        ``link_return``, both ``HARD`` against Spark SQL and ``PARTIAL`` against
        PySpark) are exactly the "no SQL answer, but a Python one" case this
        routing is for.

    The detector scan is a sanitise pass plus gated regexes over each member
    chunk. It runs only on the fallback path — :func:`target_for_item` returns
    before calling this when there is nowhere to fall back to — so a PySpark run
    and a run with ``pipeline.sql_fallback`` off pay nothing for it.
    """
    from complexity.detectors import detect_constructs

    keys = [(k.kind, k.name) for k in _constructs_for_item(item)]
    keys.extend(("kind", chunk.kind.value) for chunk in item.chunks)
    # Deduplicated: one DO loop and five are the same routing decision, and the
    # comparison is a dictionary lookup per entry.
    detected = {
        found.name for chunk in item.chunks for found in detect_constructs(chunk.text)
    }
    keys.extend(("detector", name) for name in sorted(detected))
    return keys


def target_for_item(
    item: SasBatch,
    run_target: TargetLanguage,
    *,
    fallback_to: TargetLanguage | None = None,
) -> "TargetChoice":
    """Which target *item* should be translated into — see
    :func:`complexity.fallback.choose_target`.

    Imported lazily: :mod:`complexity` is a sibling package the pipeline
    otherwise does not need, and a run with the fallback disabled should not
    pay to load its rule sets at all.
    """
    if fallback_to is None:
        from complexity.fallback import TargetChoice

        return TargetChoice(target=run_target)
    from complexity.fallback import choose_target

    return choose_target(
        _profile_constructs_for_item(item),
        run_target=run_target,
        fallback_to=fallback_to,
    )


def _constructs_for_chunk(chunk: SasChunk) -> list[ConstructKey]:
    """:func:`_constructs_for_item`'s keys for a single member chunk.

    Same primitive, same emission order — the difference is only the scope of
    the identifier sets. Used to attribute a batch's guidance back to the
    member whose constructs pulled it in (:func:`_attribution_for_item`); the
    batch-level function stays driven off the batch rollups so its output is
    unaffected by how members are visited.
    """
    m = chunk.metadata
    return _construct_keys(
        procs={m.proc_name} if m.proc_name else set(),
        functions=m.recognized_functions,
        call_routines=m.recognized_call_routines,
        component_objects=m.component_objects,
        statements=m.data_step_statements,
        global_statements=(
            {m.global_statement_keyword} if m.global_statement_keyword else set()
        ),
        symput_hazard=m.symput_scope_hazard,
        computed_goto=m.contains_computed_goto,
        abort=m.contains_abort,
    )


def _construct_keys(
    *,
    procs: Iterable[str],
    functions: Iterable[str],
    call_routines: Iterable[str],
    component_objects: Iterable[str],
    statements: Iterable[str],
    global_statements: Iterable[str],
    symput_hazard: bool,
    computed_goto: bool,
    abort: bool,
) -> list[ConstructKey]:
    """Build the ordered, deduplicated construct keys for one scope.

    The single place the metadata -> :class:`ConstructKey` mapping lives, so a
    batch and one of its members can never disagree about what a construct is
    called. Emission order is part of the contract: the selector treats the
    caller's order as meaningful (it reports the first matching key as a pick's
    provenance and interleaves construct hits in it), so specific kinds come
    before ``category``, and the whole sequence is deterministic.
    """
    keys: list[ConstructKey] = []
    seen: set[ConstructKey] = set()
    categories: set[str] = set()

    def add(kind: str, name: str | None) -> None:
        if not name:
            return
        key = ConstructKey(kind=kind, name=name.lower())
        if key not in seen:
            seen.add(key)
            keys.append(key)

    def note_category(name: str) -> None:
        category = SAS_FUNCTION_CATEGORIES.get(name.lower())
        if category:
            categories.add(category)

    for name in sorted(procs):
        add("proc", name)
    for name in sorted(functions):
        add("function", name)
        note_category(name)
    for name in sorted(call_routines):
        add("call_routine", name)
        note_category(name)
    for name in sorted(component_objects):
        add("component_object", name)
    for name in sorted(statements):
        add("statement", name)
    for name in sorted(global_statements):
        add("global_statement", name)
    if symput_hazard:
        add("call_routine", "symput")
    if computed_goto:
        add("macro_statement", "goto")
    if abort:
        add("macro_statement", "abort")
    for category in sorted(categories):
        add("category", category)
    return keys


def _attribution_for_item(item: SasBatch) -> dict[ConstructKey, list[str]]:
    """Which member chunk(s) each of the batch's constructs came from.

    Lets the guidance block say *why* a section is present in a multi-member
    batch — a MERGE rule names the member that merges, rather than leaving the
    model to re-derive the mapping from the member bodies. Chunk ids are in
    batch order and deduplicated; a construct several members share lists them
    all. Keys with no member (only the batch rollup can produce those, and it
    cannot) simply do not appear.
    """
    attribution: dict[ConstructKey, list[str]] = {}
    for chunk in item.chunks:
        for key in _constructs_for_chunk(chunk):
            ids = attribution.setdefault(key, [])
            if chunk.chunk_id not in ids:
                ids.append(chunk.chunk_id)
    return attribution


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
    # An item that names a filesystem location or reaches a remote service has
    # a translation problem no construct key describes — the path itself has to
    # become something on the target — so it gets its own scope token.
    ("physical_paths", "physical_paths"),
    ("remote_paths", "remote_paths"),
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


def prompt_cost_estimator(
    model: str | None = None,
) -> Callable[[SasBatch | SasChunk], int]:
    """A per-item prompt-token cost function under *model*'s encoding, for
    token-budgeted packing (``coalesce_into_batches(max_tokens=...)``).

    The template overheads are measured **once**, from the template text
    itself (the format placeholders count roughly like the short values that
    replace them), rather than formatting every candidate window — packing
    needs an estimate that tracks the real prompt cost, not the exact
    formatted count. Member text and title are counted with the real
    tokenizer (:mod:`llm_client.tokens`), which degrades to chars//4 offline.
    """
    member_overhead = tokens.count_text(_BATCH_MEMBER_TEMPLATE, model=model)
    context_overhead = tokens.count_text(_BATCH_CONTEXT_TEMPLATE, model=model)

    def cost(item: SasBatch | SasChunk) -> int:
        chunks = item.chunks if isinstance(item, SasBatch) else [item]
        total = context_overhead
        for c in chunks:
            total += member_overhead + tokens.count_text(c.text, model=model)
            if c.title:
                total += tokens.count_text(c.title, model=model)
        return total

    return cost


def _format_target_directive(choice: "TargetChoice", run_target: TargetLanguage) -> str:
    """The per-item target override, or ``""`` when the item keeps the run's.

    Empty for every item of a run that never falls back, so the batch context is
    byte-identical to what it was before the fallback existed.
    """
    if not choice.fell_back:
        return ""
    # The three worst reasons: enough for the model to recognise the construct
    # it is looking at, short enough not to crowd the item itself.
    named = ", ".join(f"`{r.name}`" for r in choice.reasons[:3])
    if len(choice.reasons) > 3:
        named += f" (and {len(choice.reasons) - 3} more)"
    return _TARGET_OVERRIDE_TEMPLATE.format(
        output_language=choice.target.display_name,
        run_language=run_target.display_name,
        reasons=named,
        fence_info=choice.target.default_fence,
        cell_language=choice.target.cell_language,
        comment_prefix=choice.target.comment_prefix,
    )


def _format_batch_message(
    batch: SasBatch,
    index: int,
    total: int,
    diagnostics: list[SasDiagnostic],
    target_directive: str = "",
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
        proc_names=_fmt_list(sorted(batch.proc_names)),
        data_step_statements=_fmt_list(sorted(batch.data_step_statements)),
        sas_functions=_fmt_list(sorted(batch.recognized_functions)),
        call_routines=_fmt_list(sorted(batch.recognized_call_routines)),
        component_objects=_fmt_list(sorted(batch.component_objects)),
        global_statement_keywords=_fmt_list(sorted(batch.global_statement_keywords)),
        symput_hazard="yes" if batch.has_symput_scope_hazard else "no",
        contains_abort="yes" if batch.has_abort else "no",
        contains_computed_goto="yes" if batch.has_computed_goto else "no",
        diagnostics="; ".join(f"[{d.code}] {d.message}" for d in diags) or "none",
        target_directive=target_directive,
        members=members,
    )
    logger.debug(
        f"_format_batch_message: batch={batch.batch_id}  members={len(batch.chunks)}  prompt_chars={len(msg)}"
    )
    return msg
