"""
batcher.py — dependency-graph-driven batching of SAS semantic chunks.

Two public classes
------------------
SasChunkBatcher   — single-file batching (accepts one SasChunkResult)
MultiFileBatcher  — multi-file batching  (accepts a SasCorpus)

Both share the same core algorithm; MultiFileBatcher extends it with
cross-file edge discovery and globally-unique chunk indexing.

Algorithm (shared)
------------------
1. Build a producer index: dataset-name → [(file_rank, chunk_index)]
                           macro-name   → (file_rank, chunk_index)
2. Walk all chunks in corpus order (file_rank, chunk_index).  For each
   chunk look up its input_datasets and invokes_macros.  Any hit → edge
   + Union-Find union(producer_global_idx, consumer_global_idx).
3. For MACRO_CALL chunks also scan argument tokens for dataset names
   (macro_arg_dataset edges).
4. Optionally absorb OPTIONS/GLOBAL_STATEMENT/COMMENT_BLOCK chunks into
   the component of the nearest following substantive chunk.
5. Extract UF components → SasBatch (size ≥ 2) or singleton (size 1).
6. Each batch derives external I/O by set-difference.

Cross-file specifics
---------------------
- Chunk indices are global: file 0 chunks are 0..n0-1, file 1 chunks are
  n0..n0+n1-1, etc.  Union-Find runs over this flat index space.
- Ordering inside a cross-file batch: chunks are sorted by
  (file_rank, start_line) so producers always precede consumers.
- SasBatch.source_files lists the distinct source_ids of member chunks.

Logging
-------
Logger: ``sas_chunker.batcher``

  Level    When emitted
  -------  ---------------------------------------------------------------
  DEBUG    Every edge (or skip); UF merges; component membership;
           batch I/O derivation; context absorption
  INFO     Start/finish; edge totals; per-batch summary; cross-file count
  WARNING  Macro redefinition across files; unresolvable dependencies
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass

from .chunker import _STANDARD_AUTOCALL_MACROS
from .models import (
    SasBatch,
    SasBatchResult,
    SasChunk,
    SasChunkKind,
    SasChunkResult,
    SasCorpus,
    SasMultiBatchResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Union-Find (path-halving + union-by-rank)
# ---------------------------------------------------------------------------


class _UF:
    """Path-compressed union-find over integer indices."""

    def __init__(self, n: int) -> None:
        self._p = list(range(n))
        self._r = [0] * n

    def find(self, x: int) -> int:
        while self._p[x] != x:
            self._p[x] = self._p[self._p[x]]
            x = self._p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        """Merge components; return True if they were distinct."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self._r[ra] < self._r[rb]:
            ra, rb = rb, ra
        self._p[rb] = ra
        if self._r[ra] == self._r[rb]:
            self._r[ra] += 1
        return True

    def components(self, n: int) -> dict[int, list[int]]:
        """Return {root: [member_global_indices]} for all n elements."""
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            groups[self.find(i)].append(i)
        return dict(groups)


# ---------------------------------------------------------------------------
# Dependency-edge record
# ---------------------------------------------------------------------------


@dataclass
class _Edge:
    kind: str  # dataset_flow | macro_invocation | macro_arg_dataset | macro_body_dataset | *_context
    from_id: str  # chunk_id of producer/predecessor
    to_id: str  # chunk_id of consumer/successor
    via: str  # dataset name, %macro, or [context_reason]
    from_global_idx: int
    to_global_idx: int
    cross_file: bool = False


# Splits a call-site argument list on commas, respecting nested parens
_CALL_ARG_SPLIT_RE = re.compile(r",(?![^(]*\))")

# Extracts the call's argument list: %macroname( ... )
_CALL_ARGS_RE = re.compile(r"%\s*\w+\s*\(([^)]*)\)", re.DOTALL)


def _parse_call_args(call_text: str) -> tuple[list[str], dict[str, str]]:
    """
    Parse a MACRO_CALL chunk's raw text into (positional_args, keyword_args).

    Used by Fix B (parameterised macro output/input resolution): the
    definition's ``body_param_outputs``/``body_param_inputs`` reference a
    parameter by name and positional index, and this function recovers the
    actual values supplied at the call site so those references can be
    resolved to concrete dataset names.

    Quoting and trailing dots are stripped from each value so that
    ``work.orders``, ``'work.orders'``, and ``work.orders.`` all normalise
    to the same lowercase dataset key.
    """
    m = _CALL_ARGS_RE.search(call_text)
    if not m:
        return [], {}

    raw_args = m.group(1)
    parts = [p.strip() for p in _CALL_ARG_SPLIT_RE.split(raw_args) if p.strip()]

    positional: list[str] = []
    keyword: dict[str, str] = {}

    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            keyword[k.strip().lower()] = _clean_arg_value(v)
        else:
            positional.append(_clean_arg_value(part))

    return positional, keyword


def _clean_arg_value(value: str) -> str:
    """Strip quotes and a single trailing dot from a macro call argument."""
    v = value.strip()
    if v.startswith(("'", '"')) and v.endswith(("'", '"')):
        v = v[1:-1]
    return v.rstrip(".").lower()


def _file_of_map(file_offsets: list[int], n: int) -> list[int]:
    """
    Build a ``global_index → file_rank`` lookup list from ``file_offsets``.

    ``file_offsets[i]`` is the global index of the first chunk of file *i*;
    this expands that into a flat list of length *n* where each position
    holds the file rank that chunk belongs to.  Used by every function that
    needs to know whether two global indices fall in the same file
    (``_discover_edges``, ``_absorb_context``, ``_make_batch``).
    """
    file_of: list[int] = []
    for fi, off in enumerate(file_offsets):
        nxt = file_offsets[fi + 1] if fi + 1 < len(file_offsets) else n
        file_of.extend([fi] * (nxt - off))
    return file_of


# ---------------------------------------------------------------------------
# Chunk-kind sets for context absorption
# ---------------------------------------------------------------------------

_OPTION_KINDS = frozenset(
    {
        SasChunkKind.OPTIONS,
        SasChunkKind.GLOBAL_STATEMENT,
        SasChunkKind.FORMAT_OR_INFORMAT,
    }
)
_COMMENT_KINDS = frozenset({SasChunkKind.COMMENT_BLOCK})
_CONTEXT_KINDS = _OPTION_KINDS | _COMMENT_KINDS


# ---------------------------------------------------------------------------
# Shared core: build flat index, discover edges, extract batches
# ---------------------------------------------------------------------------


def _build_flat_index(
    file_results: list[SasChunkResult],
) -> tuple[list[SasChunk], list[tuple[int, int]], list[int]]:
    """
    Flatten all files into a single ordered list and re-stamp chunk IDs
    so they are globally unique across files.

    Original per-file IDs (``chunk-0001``) are replaced with
    ``f{file_rank+1}-chunk-{local_seq}`` (e.g. ``f2-chunk-0003``).
    This is done on *copies* of the SasChunk objects so the original
    SasChunkResult objects are not mutated.

    Returns
    -------
    flat_chunks : list[SasChunk]
        Every chunk across all files in corpus order, with global IDs.
    file_chunk_ranges : list[(start, end)]
        Inclusive global index range for each file.
    file_offsets : list[int]
        Global index of the first chunk of each file.
    """
    flat: list[SasChunk] = []
    ranges: list[tuple[int, int]] = []
    offsets: list[int] = []

    for fi, fr in enumerate(file_results):
        start = len(flat)
        for chunk in fr.chunks:
            # Build a globally-unique ID: f<file_rank>-<original_id>
            # e.g. "chunk-0003" in file 2 → "f2-chunk-0003"
            new_id = f"f{fi + 1}-{chunk.chunk_id}"
            # model_copy avoids mutating the original chunk
            stamped = chunk.model_copy(update={"chunk_id": new_id})
            # Also update parent_id if it references a sibling chunk
            if stamped.parent_id:
                stamped = stamped.model_copy(
                    update={"parent_id": f"f{fi + 1}-{chunk.parent_id}"}
                )
            flat.append(stamped)
        end = len(flat) - 1
        ranges.append((start, end))
        offsets.append(start)

    return flat, ranges, offsets


def _add_edge(
    edges: list[_Edge],
    uf: _UF,
    flat_chunks: list[SasChunk],
    file_of: list[int],
    *,
    kind: str,
    from_idx: int,
    to_idx: int,
    via: str,
) -> bool:
    """
    Create a dependency edge between two global chunk indices, union them
    in the Union-Find structure, log the result, and return whether the
    union actually merged two previously-distinct components.

    Shared by every edge-discovery pass (dataset_flow, macro_invocation,
    macro_arg_dataset, macro_body_dataset) so the "build edge, append,
    union, log" sequence exists in exactly one place.
    """
    cf = file_of[from_idx] != file_of[to_idx]
    e = _Edge(
        kind=kind,
        from_id=flat_chunks[from_idx].chunk_id,
        to_id=flat_chunks[to_idx].chunk_id,
        via=via,
        from_global_idx=from_idx,
        to_global_idx=to_idx,
        cross_file=cf,
    )
    edges.append(e)
    merged = uf.union(from_idx, to_idx)
    logger.debug(
        f"edge[{kind}] {e.from_id}(g{from_idx}) → {e.to_id}(g{to_idx}) via={via!r} cross_file={cf} uf_merged={merged}"
    )
    return merged


def _discover_edges(
    flat_chunks: list[SasChunk],
    uf: _UF,
    *,
    file_offsets: list[int],
) -> list[_Edge]:
    """
    Walk flat_chunks once, building producer indices and emitting edges.

    file_offsets is used to determine when two chunks belong to different
    files (for logging the cross_file flag on edges).
    """
    n = len(flat_chunks)

    # ── build producer indices ─────────────────────────────────────────────
    # dataset → list of global indices of chunks that write it
    produces_ds: dict[str, list[int]] = defaultdict(list)
    # macro name → global index of defining chunk (last definition wins)
    defines_macro: dict[str, int] = {}
    # macro VARIABLE name → list of global indices of chunks that create it
    # via CALL SYMPUT/SYMPUTX or PROC SQL INTO (ROADMAP Phase 2)
    produces_macrovar: dict[str, list[int]] = defaultdict(list)

    # reverse-map: global_index → file_rank
    file_of = _file_of_map(file_offsets, n)

    for gidx, chunk in enumerate(flat_chunks):
        meta = chunk.metadata
        for ds in meta.output_datasets:
            produces_ds[ds].append(gidx)
            logger.debug(
                f"index[ds]:    chunk {chunk.chunk_id} (g{gidx}) writes '{ds}'"
            )
        for mac in meta.defines_macros:
            if mac in defines_macro:
                prev = flat_chunks[defines_macro[mac]]
                prev_file = file_of[defines_macro[mac]]
                cur_file = file_of[gidx]
                logger.warning(
                    f"index[macro]: macro '%{mac}' redefined — chunk {chunk.chunk_id} (file {cur_file}) overrides chunk {prev.chunk_id} (file {prev_file})"
                )
            defines_macro[mac] = gidx
            logger.debug(
                f"index[macro]: chunk {chunk.chunk_id} (g{gidx}) defines %{mac}"
            )

        for mvar in meta.produces_macrovars:
            produces_macrovar[mvar].append(gidx)
            logger.debug(
                f"index[macrovar]: chunk {chunk.chunk_id} (g{gidx}) creates &{mvar}"
            )

        # ── Fix A: literal macro-body outputs ────────────────────────────
        # A %MACRO body may contain hard-coded dataset names, e.g.
        #     %macro setup; data work.base; set mylib.raw; run; %mend;
        # These datasets are produced whenever the macro is *invoked*, but
        # for indexing purposes we register them as if the MACRO_DEFINITION
        # chunk itself were the producer.  The macro_invocation edge (added
        # below) already links every call site back to this chunk, so any
        # consumer of 'work.base' transitively joins the same component as
        # every call site of %setup.
        if meta.body_literal_outputs:
            for ds in meta.body_literal_outputs:
                produces_ds[ds].append(gidx)
                logger.debug(
                    f"index[ds-literal-body]: chunk {chunk.chunk_id} (g{gidx}) macro body writes '{ds}'"
                )

    logger.debug(
        f"index built  unique_output_datasets={len(produces_ds)}  unique_macro_definitions={len(defines_macro)}"
    )

    # ── discover edges ─────────────────────────────────────────────────────
    edges: list[_Edge] = []

    for cidx, chunk in enumerate(flat_chunks):
        meta = chunk.metadata

        # — dataset-flow —
        for ds in meta.input_datasets:
            if ds not in produces_ds:
                logger.debug(
                    f"edge-skip[ds]: '{ds}' read by {chunk.chunk_id} — no producer in corpus"
                )
                continue
            for pidx in produces_ds[ds]:
                if pidx == cidx:
                    logger.debug(f"edge-skip[ds]: self-ref '{ds}' in {chunk.chunk_id}")
                    continue
                _add_edge(
                    edges,
                    uf,
                    flat_chunks,
                    file_of,
                    kind="dataset_flow",
                    from_idx=pidx,
                    to_idx=cidx,
                    via=ds,
                )

        # — macro-variable flow (ROADMAP Phase 2) —
        # Mirrors dataset_flow exactly, but for the macro-variable
        # namespace: a chunk that creates &cutoff via CALL SYMPUT/SYMPUTX
        # or PROC SQL INTO links to any chunk that references &cutoff.
        for mvar in meta.consumes_macrovars:
            if mvar not in produces_macrovar:
                logger.debug(
                    f"edge-skip[macrovar]: '&{mvar}' read by {chunk.chunk_id} — no producer in corpus"
                )
                continue
            for pidx in produces_macrovar[mvar]:
                if pidx == cidx:
                    logger.debug(
                        f"edge-skip[macrovar]: self-ref '&{mvar}' in {chunk.chunk_id}"
                    )
                    continue
                _add_edge(
                    edges,
                    uf,
                    flat_chunks,
                    file_of,
                    kind="macro_var_flow",
                    from_idx=pidx,
                    to_idx=cidx,
                    via=f"&{mvar}",
                )

        # — macro-invocation —
        for mac in meta.invokes_macros:
            if mac not in defines_macro:
                logger.debug(
                    f"edge-skip[macro]: '%{mac}' in {chunk.chunk_id} — no definition in corpus"
                )
                continue
            didx = defines_macro[mac]
            if didx == cidx:
                continue
            _add_edge(
                edges,
                uf,
                flat_chunks,
                file_of,
                kind="macro_invocation",
                from_idx=didx,
                to_idx=cidx,
                via=f"%{mac}",
            )

        # — macro-argument dataset —
        if chunk.kind == SasChunkKind.MACRO_CALL:
            arg_tokens = re.findall(
                r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?",
                chunk.text,
            )
            for tok in arg_tokens:
                ds_norm = tok.lower()
                if ds_norm not in produces_ds:
                    continue
                for pidx in produces_ds[ds_norm]:
                    if pidx == cidx:
                        continue
                    _add_edge(
                        edges,
                        uf,
                        flat_chunks,
                        file_of,
                        kind="macro_arg_dataset",
                        from_idx=pidx,
                        to_idx=cidx,
                        via=ds_norm,
                    )

        # — macro-body dataset (Fix B: parameterised resolution) —
        # The MACRO_CALL invokes a macro whose body references a dataset
        # only through a parameter, e.g. "data &ds.;" inside %clean(ds).
        # We resolve &ds. using the actual argument supplied at this call
        # site, then treat the resolved name as if this call-site chunk
        # itself produced/consumed that dataset — letting normal
        # dataset_flow edges (computed in the next pass-through of this
        # same loop, since produces_ds is mutated in place) link it to
        # whatever else reads/writes that name elsewhere in the corpus.
        if chunk.kind == SasChunkKind.MACRO_CALL:
            for mac in meta.invokes_macros:
                if mac not in defines_macro:
                    continue
                didx = defines_macro[mac]
                def_meta = flat_chunks[didx].metadata
                if not (def_meta.body_param_outputs or def_meta.body_param_inputs):
                    continue

                pos_args, kw_args = _parse_call_args(chunk.text)
                logger.debug(
                    f"macro_body_dataset: call {chunk.chunk_id} invokes %{mac}  pos_args={pos_args}  kw_args={kw_args}"
                )

                def _resolve(entry: dict) -> str | None:
                    pname = entry["param"]
                    pos = entry["pos"]
                    if pos >= 0:
                        if pos < len(pos_args):
                            return pos_args[pos]
                        if pname in kw_args:
                            return kw_args[pname]
                        return None
                    # keyword-only parameter
                    return kw_args.get(pname)

                resolved_outputs: list[str] = []
                for entry in def_meta.body_param_outputs:
                    val = _resolve(entry)
                    if val:
                        resolved_outputs.append(val)
                        logger.debug(
                            f"macro_body_dataset: resolved param '{entry['param']}' → output '{val}'  (call {chunk.chunk_id})"
                        )

                resolved_inputs: list[str] = []
                for entry in def_meta.body_param_inputs:
                    val = _resolve(entry)
                    if val:
                        resolved_inputs.append(val)
                        logger.debug(
                            f"macro_body_dataset: resolved param '{entry['param']}' → input '{val}'  (call {chunk.chunk_id})"
                        )

                # Register this call site as a producer of its resolved
                # outputs so later chunks (in any file) that read those
                # names get linked via the normal dataset_flow pass.
                for ds in resolved_outputs:
                    if cidx not in produces_ds[ds]:
                        produces_ds[ds].append(cidx)
                        logger.debug(
                            f"index[ds-call-resolved]: chunk {chunk.chunk_id} (g{cidx}) resolved-produces '{ds}' via %{mac}"
                        )

                # Persist resolved outputs/inputs back onto the chunk's own
                # metadata (output_datasets / input_datasets) so that
                # _make_batch — and any other downstream consumer — sees
                # them through the normal metadata fields, with no special
                # casing required.  This is what makes
                # SasBatch.output_datasets correctly report 'work.first',
                # 'work.second', etc. for resolved parameterised calls.
                if resolved_outputs or resolved_inputs:
                    new_out = sorted(
                        set(chunk.metadata.output_datasets) | set(resolved_outputs)
                    )
                    new_in = sorted(
                        set(chunk.metadata.input_datasets) | set(resolved_inputs)
                    )
                    updated_meta = chunk.metadata.model_copy(
                        update={"output_datasets": new_out, "input_datasets": new_in},
                    )
                    chunk = chunk.model_copy(update={"metadata": updated_meta})
                    flat_chunks[cidx] = chunk
                    meta = updated_meta
                    logger.debug(
                        f"macro_body_dataset: chunk {chunk.chunk_id} metadata updated  output_datasets={new_out}  input_datasets={new_in}"
                    )

                # Link this call site to whatever already produces its
                # resolved inputs (covers the case where the macro reads
                # a dataset whose actual name is only known at this site).
                for ds in resolved_inputs:
                    if ds not in produces_ds:
                        logger.debug(
                            f"macro_body_dataset: resolved input '{ds}' has no producer in corpus (call {chunk.chunk_id})"
                        )
                        continue
                    for pidx in produces_ds[ds]:
                        if pidx == cidx:
                            continue
                        _add_edge(
                            edges,
                            uf,
                            flat_chunks,
                            file_of,
                            kind="macro_body_dataset",
                            from_idx=pidx,
                            to_idx=cidx,
                            via=ds,
                        )

    cf_count = sum(1 for e in edges if e.cross_file)
    logger.info(
        f"edges discovered  total={len(edges)}  dataset_flow={sum(1 for e in edges if e.kind == 'dataset_flow')}  macro_invocation={sum(1 for e in edges if e.kind == 'macro_invocation')}  macro_arg_dataset={sum(1 for e in edges if e.kind == 'macro_arg_dataset')}  macro_body_dataset={sum(1 for e in edges if e.kind == 'macro_body_dataset')}  macro_var_flow={sum(1 for e in edges if e.kind == 'macro_var_flow')}  cross_file={cf_count}"
    )
    if cf_count:
        logger.info(
            f"cross-file edges: {[(e.from_id, '→', e.to_id, 'via', e.via) for e in edges if e.cross_file]}"
        )
    return edges


def _absorb_context(
    flat_chunks: list[SasChunk],
    uf: _UF,
    edges: list[_Edge],
    *,
    include_options: bool,
    include_comments: bool,
    file_offsets: list[int],
) -> None:
    """
    Pull OPTIONS/GLOBAL_STATEMENT and/or COMMENT_BLOCK chunks into the
    component of the first substantive chunk that follows them,
    *within the same file only*.  We never absorb across a file boundary.
    """
    n = len(flat_chunks)
    file_of = _file_of_map(file_offsets, n)

    for idx, chunk in enumerate(flat_chunks):
        kind = chunk.kind
        is_option = kind in _OPTION_KINDS and include_options
        is_comment = kind in _COMMENT_KINDS and include_comments
        if not (is_option or is_comment):
            continue

        my_file = file_of[idx]
        for nxt_idx in range(idx + 1, n):
            if file_of[nxt_idx] != my_file:
                break  # don't cross file boundaries
            nxt_chunk = flat_chunks[nxt_idx]
            if nxt_chunk.kind in _CONTEXT_KINDS:
                continue  # skip consecutive context chunks
            reason = "options_context" if is_option else "comment_context"
            merged = uf.union(idx, nxt_idx)
            logger.debug(
                f"absorb[{reason}]: {chunk.chunk_id} ← {nxt_chunk.chunk_id}  merged={merged}"
            )
            if merged:
                edges.append(
                    _Edge(
                        kind=reason,
                        from_id=chunk.chunk_id,
                        to_id=nxt_chunk.chunk_id,
                        via=f"[{reason}]",
                        from_global_idx=idx,
                        to_global_idx=nxt_idx,
                        cross_file=False,
                    )
                )
            break


def _make_batch(
    global_indices: list[int],
    flat_chunks: list[SasChunk],
    edges_by_pair: dict[tuple[int, int], list[_Edge]],
    batch_number: int,
    file_offsets: list[int],
) -> SasBatch:
    """
    Build a SasBatch from a set of global chunk indices.

    Chunks are ordered by (file_rank, start_line) so producers always
    appear before consumers in the batch text.
    """
    n = len(flat_chunks)
    file_of = _file_of_map(file_offsets, n)

    bid = f"batch-{batch_number:03d}"

    # Sort by (file_rank, start_line) — producers before consumers
    ordered = sorted(
        global_indices,
        key=lambda gi: (file_of[gi], flat_chunks[gi].start_line),
    )
    member_chunks = [flat_chunks[gi] for gi in ordered]
    member_ids = {c.chunk_id for c in member_chunks}

    # ── source_files: distinct file ids in appearance order ───────────────
    seen_files: list[str] = []
    seen_set: set[str] = set()
    for c in member_chunks:
        fid = c.source_id or "<inline>"
        if fid not in seen_set:
            seen_files.append(fid)
            seen_set.add(fid)

    # ── intra-batch production sets ────────────────────────────────────────
    intra_outputs: set[str] = set()
    intra_macros: set[str] = set()
    intra_macrovars: set[str] = set()
    for c in member_chunks:
        intra_outputs.update(c.metadata.output_datasets)
        intra_macros.update(c.metadata.defines_macros)
        intra_macrovars.update(c.metadata.produces_macrovars)
        # A %MACRO definition's literal body outputs are produced whenever
        # the macro is invoked.  If any call site is also in this batch
        # (the usual case, since macro_invocation edges put them together),
        # report the literal body output as a batch-level output too.
        if c.kind == SasChunkKind.MACRO_DEFINITION:
            intra_outputs.update(c.metadata.body_literal_outputs)

    logger.debug(
        f"_make_batch {bid}: intra_outputs={sorted(intra_outputs)}  intra_macros={sorted(intra_macros)}  source_files={seen_files}"
    )

    # ── external inputs (not satisfied intra-batch) ────────────────────────
    ext_inputs: list[str] = []
    seen_ei: set[str] = set()
    for c in member_chunks:
        for ds in c.metadata.input_datasets:
            if ds not in intra_outputs and ds not in seen_ei:
                ext_inputs.append(ds)
                seen_ei.add(ds)
                logger.debug(f"_make_batch {bid}: ext input '{ds}'")

    # ── external macro requirements ────────────────────────────────────────
    ext_macros: list[str] = []
    seen_em: set[str] = set()
    # Standard, SAS-provided autocall macros (ROADMAP Phase 5, F2b) — never
    # a missing dependency the user needs to locate, so excluded from
    # ext_macros, but still tracked separately rather than silently dropped.
    standard_autocall: list[str] = []
    seen_sa: set[str] = set()
    for c in member_chunks:
        for mac in c.metadata.invokes_macros:
            if mac in intra_macros:
                continue
            if mac in _STANDARD_AUTOCALL_MACROS:
                if mac not in seen_sa:
                    standard_autocall.append(mac)
                    seen_sa.add(mac)
                    logger.debug(f"_make_batch {bid}: standard autocall macro '%{mac}'")
                continue
            if mac not in seen_em:
                ext_macros.append(mac)
                seen_em.add(mac)
                logger.debug(f"_make_batch {bid}: ext macro '%{mac}'")

    # ── external macro-variable requirements (ROADMAP Phase 2) ──────────────
    ext_macrovars: list[str] = []
    seen_emv: set[str] = set()
    for c in member_chunks:
        for mvar in c.metadata.consumes_macrovars:
            if mvar not in intra_macrovars and mvar not in seen_emv:
                ext_macrovars.append(mvar)
                seen_emv.add(mvar)
                logger.debug(f"_make_batch {bid}: ext macrovar '&{mvar}'")

    # ── reason string ──────────────────────────────────────────────────────
    reason_parts: list[str] = []
    seen_r: set[str] = set()
    for (fi, ti), edge_list in edges_by_pair.items():
        fid = edge_list[0].from_id
        tid = edge_list[0].to_id
        if fid not in member_ids and tid not in member_ids:
            continue
        for e in edge_list:
            cf_tag = " [cross-file]" if e.cross_file else ""
            desc = f"{e.kind}({e.via}){cf_tag}: {e.from_id} → {e.to_id}"
            if desc not in seen_r:
                reason_parts.append(desc)
                seen_r.add(desc)

    reason = "; ".join(reason_parts) or "[context absorption only]"
    logger.debug(f"_make_batch {bid}: reason={reason!r}")

    return SasBatch(
        batch_id=bid,
        chunks=member_chunks,
        reason=reason,
        source_files=seen_files,
        input_datasets=ext_inputs,
        output_datasets=sorted(intra_outputs),
        required_macros=ext_macros,
        defined_macros=sorted(intra_macros),
        produced_macrovars=sorted(intra_macrovars),
        required_macrovars=ext_macrovars,
        standard_autocall_macros=sorted(standard_autocall),
    )


def _extract_result(
    uf: _UF,
    flat_chunks: list[SasChunk],
    edges: list[_Edge],
    file_offsets: list[int],
) -> tuple[list[SasBatch], list[SasChunk]]:
    """Convert UF components → (batches, singletons)."""
    n = len(flat_chunks)
    components = uf.components(n)

    edges_by_pair: dict[tuple[int, int], list[_Edge]] = defaultdict(list)
    for e in edges:
        edges_by_pair[(e.from_global_idx, e.to_global_idx)].append(e)

    batches: list[SasBatch] = []
    singletons: list[SasChunk] = []

    for root, members in sorted(components.items()):
        logger.debug(
            f"component root={root}  size={len(members)}  members={[flat_chunks[i].chunk_id for i in members]}"
        )
        if len(members) == 1:
            solo = flat_chunks[members[0]]
            singletons.append(solo)
            logger.debug(
                f"singleton: {solo.chunk_id}  kind={solo.kind.value}  source={solo.source_id}"
            )
            continue

        batch = _make_batch(
            global_indices=members,
            flat_chunks=flat_chunks,
            edges_by_pair=edges_by_pair,
            batch_number=len(batches) + 1,
            file_offsets=file_offsets,
        )
        batches.append(batch)
        logger.info(
            f"batch {batch.batch_id}  chunks={len(batch.chunks)}  source_files={batch.source_files}  cross_file={batch.is_cross_file}  inputs={batch.input_datasets}  outputs={batch.output_datasets}  req_macros={batch.required_macros}  def_macros={batch.defined_macros}"
        )

    return batches, singletons


# ---------------------------------------------------------------------------
# Single-file batcher  (unchanged public API)
# ---------------------------------------------------------------------------


class SasChunkBatcher:
    """
    Group the chunks of a *single* :class:`SasChunkResult` into
    dependency-aware :class:`SasBatch` objects.

    For multi-file batching use :class:`MultiFileBatcher`.

    Parameters
    ----------
    include_comment_chunks : bool
        Pull adjacent COMMENT_BLOCK chunks into the batch of the following
        substantive chunk.  Default: ``False``.
    include_options_chunks : bool
        Pull OPTIONS / GLOBAL_STATEMENT chunks that immediately precede a
        substantive chunk into that chunk's batch.  Default: ``True``.
    """

    def __init__(
        self,
        *,
        include_comment_chunks: bool = False,
        include_options_chunks: bool = True,
    ) -> None:
        self.include_comment_chunks = include_comment_chunks
        self.include_options_chunks = include_options_chunks
        logger.debug(
            f"SasChunkBatcher  include_comment={include_comment_chunks}  include_options={include_options_chunks}"
        )

    def batch(self, chunk_result: SasChunkResult) -> SasBatchResult:
        """Compute dependency-driven batches for all chunks in *chunk_result*."""
        label = chunk_result.source_id or "<inline>"
        logger.info(
            f"SasChunkBatcher.batch: start  source='{label}'  chunks={len(chunk_result.chunks)}"
        )
        t0 = time.perf_counter()

        if not chunk_result.chunks:
            logger.warning(f"batch: no chunks for source='{label}'")
            return SasBatchResult(source_id=chunk_result.source_id)

        # Wrap as a single-file corpus and delegate to the shared core
        corpus = SasCorpus(file_results=[chunk_result])
        multi = MultiFileBatcher(
            include_comment_chunks=self.include_comment_chunks,
            include_options_chunks=self.include_options_chunks,
        ).batch(corpus)

        # Unwrap back to SasBatchResult
        result = SasBatchResult(
            source_id=chunk_result.source_id,
            batches=multi.batches,
            singletons=multi.singletons,
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            f"SasChunkBatcher.batch: done  source='{label}'  batches={len(result.batches)}  singletons={len(result.singletons)}  elapsed={elapsed:.3f}s"
        )
        return result


# ---------------------------------------------------------------------------
# Multi-file batcher
# ---------------------------------------------------------------------------


class MultiFileBatcher:
    """
    Group chunks from a **multi-file** :class:`SasCorpus` into
    dependency-aware :class:`SasBatch` objects, resolving cross-file
    dataset-flow and macro-invocation edges.

    A cross-file edge arises when:
    - ``File_A.sas`` contains a DATA step that writes ``work.base``, and
    - ``File_B.sas`` contains a PROC step that reads ``work.base``.

    In this case the DATA step chunk from File_A and the PROC step chunk
    from File_B are placed in the **same batch**, with ``source_files``
    listing both files and ``is_cross_file = True``.

    Parameters
    ----------
    include_comment_chunks : bool
        Pull adjacent COMMENT_BLOCK chunks into the batch of the following
        substantive chunk within the same file.  Default: ``False``.
    include_options_chunks : bool
        Pull OPTIONS / GLOBAL_STATEMENT chunks that immediately precede a
        substantive chunk into that chunk's batch (same-file only).
        Default: ``True``.

    Usage
    -----
    ::

        from sas_chunker import SasSemanticChunker, SasCorpus
        from sas_chunker.batcher import MultiFileBatcher

        chunker = SasSemanticChunker()
        corpus  = SasCorpus(file_results=[
            chunker.chunk_file("macros.sas"),
            chunker.chunk_file("etl.sas"),
            chunker.chunk_file("reports.sas"),
        ])
        result = MultiFileBatcher().batch(corpus)

        for item in result.all_ordered_items:
            ...  # SasBatch or SasChunk
    """

    def __init__(
        self,
        *,
        include_comment_chunks: bool = False,
        include_options_chunks: bool = True,
    ) -> None:
        self.include_comment_chunks = include_comment_chunks
        self.include_options_chunks = include_options_chunks
        logger.debug(
            f"MultiFileBatcher  include_comment={include_comment_chunks}  include_options={include_options_chunks}"
        )

    # ------------------------------------------------------------------
    # Factory helpers: build a corpus without pre-chunking separately
    # ------------------------------------------------------------------

    @classmethod
    def from_files(
        cls,
        paths: list[str],
        *,
        chunker_kwargs: dict | None = None,
        **batcher_kwargs,
    ) -> "tuple[SasCorpus, SasMultiBatchResult]":
        """
        Convenience: chunk each file in *paths* and batch the resulting corpus.

        Returns ``(corpus, result)`` so callers can inspect both.

        Parameters
        ----------
        paths
            Ordered list of ``.sas`` file paths.  Order establishes the
            default execution sequence for resolving tie-breaks.
        chunker_kwargs
            Forwarded to :class:`~sas_chunker.chunker.SasSemanticChunker`.
        **batcher_kwargs
            Forwarded to :class:`MultiFileBatcher`.
        """
        from .chunker import SasSemanticChunker  # avoid circular at module level

        chunker = SasSemanticChunker(**(chunker_kwargs or {}))
        results: list[SasChunkResult] = []
        for path in paths:
            logger.info(f"MultiFileBatcher.from_files: chunking '{path}'")
            results.append(chunker.chunk_file(path))

        corpus = SasCorpus(file_results=results)
        batcher = cls(**batcher_kwargs)
        result = batcher.batch(corpus)
        return corpus, result

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def batch(self, corpus: SasCorpus) -> SasMultiBatchResult:
        """
        Compute cross-file dependency batches for all files in *corpus*.

        Returns a :class:`SasMultiBatchResult` containing all batches
        (including cross-file ones) and standalone singletons.
        """
        source_ids = corpus.source_ids
        logger.info(
            f"MultiFileBatcher.batch: start  files={len(corpus.file_results)}  source_ids={source_ids}  total_chunks={sum(len(r.chunks) for r in corpus.file_results)}"
        )
        t0 = time.perf_counter()

        if not corpus.file_results or not corpus.all_chunks:
            logger.warning("MultiFileBatcher.batch: corpus is empty")
            return SasMultiBatchResult(source_ids=source_ids)

        # ── flatten corpus ──────────────────────────────────────────────────
        flat_chunks, _, file_offsets = _build_flat_index(corpus.file_results)
        n = len(flat_chunks)
        uf = _UF(n)

        logger.debug(
            f"MultiFileBatcher.batch: flat index  total={n}  file_offsets={file_offsets}"
        )

        # ── edge discovery ──────────────────────────────────────────────────
        edges = _discover_edges(flat_chunks, uf, file_offsets=file_offsets)

        # ── context absorption (same-file only) ─────────────────────────────
        if self.include_options_chunks or self.include_comment_chunks:
            _absorb_context(
                flat_chunks,
                uf,
                edges,
                include_options=self.include_options_chunks,
                include_comments=self.include_comment_chunks,
                file_offsets=file_offsets,
            )

        # ── extract batches + singletons ────────────────────────────────────
        batches, singletons = _extract_result(uf, flat_chunks, edges, file_offsets)

        cf_count = sum(1 for b in batches if b.is_cross_file)
        elapsed = time.perf_counter() - t0
        logger.info(
            f"MultiFileBatcher.batch: done  files={len(corpus.file_results)}  batches={len(batches)}  cross_file_batches={cf_count}  singletons={len(singletons)}  elapsed={elapsed:.3f}s"
        )

        return SasMultiBatchResult(
            source_ids=source_ids,
            batches=batches,
            singletons=singletons,
        )
