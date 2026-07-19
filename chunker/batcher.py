"""Dependency-graph-driven batching of SAS semantic chunks. See chunker/README.md.

`SasChunkBatcher` batches one `SasChunkResult`; `MultiFileBatcher` batches a
`SasCorpus`, extending the same core with cross-file edge discovery and
globally-unique chunk indexing.

Logger name: ``chunker.batcher``.
"""

from __future__ import annotations

import logging
import re
import time
from bisect import bisect_left, insort
from collections import defaultdict
from dataclasses import dataclass

from .keywords import _STANDARD_AUTOCALL_MACROS
from .metadata import _canon_ds
from .models import (
    SasBatch,
    SasBatchResult,
    SasChunk,
    SasChunkKind,
    SasChunkMetadata,
    SasChunkResult,
    SasCorpus,
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
    kind: str  # strong: dataset_flow|macro_body_dataset; weak: macro_invocation|
    # macro_var_flow|macro_arg_dataset; context: options_context|comment_context
    from_id: str  # chunk_id of producer/predecessor
    to_id: str  # chunk_id of consumer/successor
    via: str  # dataset name, %macro, or [context_reason]
    from_global_idx: int
    to_global_idx: int
    cross_file: bool = False

    def __str__(self) -> str:
        scope = " (cross-file)" if self.cross_file else ""
        return f"_Edge {self.from_id} -> {self.to_id} [{self.kind} via {self.via}]{scope}"


# Locates the opening of a macro call's argument list: %macroname( . The
# balanced closing paren is found by _extract_call_arg_text below.
_CALL_OPEN_RE = re.compile(r"%\s*[A-Za-z_]\w*\s*\(")

# A keyword argument is name= at the start of the (stripped) argument, so a
# positional value like f(x=1) is not mistaken for keyword 'f(x'.
_KW_ARG_RE = re.compile(r"([A-Za-z_]\w*)\s*=(.*)$", re.DOTALL)

# A libref.member (or bare member) dataset token, for scanning a MACRO_CALL's
# argument text against the producer index.
_ARG_DS_TOKEN_RE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?")


def _extract_call_arg_text(call_text: str) -> str | None:
    """Return the text between the call's balanced outer parens, or None.

    Walks characters with a paren-depth counter, treating single- and
    double-quoted spans as opaque, so nested constructs like
    ``%clean(%str(a,b), out=f(x))`` yield the full argument text instead
    of stopping at the first ``)``.  An unbalanced call (truncated chunk)
    falls back to everything after the opening paren.
    """
    m = _CALL_OPEN_RE.search(call_text)
    if not m:
        return None
    start = m.end()
    depth = 1
    quote: str | None = None
    for i in range(start, len(call_text)):
        ch = call_text[i]
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return call_text[start:i]
    return call_text[start:]


def _split_call_args(raw_args: str) -> list[str]:
    """Split an argument list on top-level commas (quote- and paren-aware)."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    quote: str | None = None
    for ch in raw_args:
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


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
    raw_args = _extract_call_arg_text(call_text)
    if raw_args is None:
        return [], {}

    positional: list[str] = []
    keyword: dict[str, str] = {}

    for part in _split_call_args(raw_args):
        kw = _KW_ARG_RE.match(part)
        if kw:
            keyword[kw.group(1).lower()] = _clean_arg_value(kw.group(2))
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
# Library handling
# ---------------------------------------------------------------------------

# Libraries SAS supplies without a LIBNAME statement — never a missing libref a
# batch needs to locate, so excluded from SasBatch.required_librefs.
_DEFAULT_LIBREFS = frozenset(
    {"work", "user", "sashelp", "sasuser", "maps", "mapssas"}
)

# PROC steps that read the most recently created dataset (_LAST_) when no DATA=
# is given. Kept to common, unambiguous consumers; anything else gets no
# implicit input rather than a guessed edge.
_LAST_CONSUMING_PROCS = frozenset(
    {
        "chart",
        "contents",
        "corr",
        "freq",
        "means",
        "plot",
        "print",
        "rank",
        "report",
        "sgplot",
        "sort",
        "summary",
        "tabulate",
        "transpose",
        "univariate",
    }
)

# A bare ``set;`` statement (no dataset named) reads _LAST_. Only consulted for
# DATA_STEP chunks that have no recorded input at all.
_BARE_SET_RE = re.compile(r"\bset\s*;", re.IGNORECASE)


def _resolve_implicit_datasets(flat_chunks: list[SasChunk]) -> None:
    """Resolve implicit dataset references in corpus order (finding 3).

    SAS maintains a session-wide "most recently created data set" (_LAST_)
    that several constructs read implicitly, and the reserved name
    ``_DATA_`` names its outputs via the DATAn convention (DATA1, DATA2, …
    — guide Ch. 3 "Special Data Set Names").  This pass walks the flat
    corpus once, replacing those placeholders with concrete canonical
    names so the ordinary dataset_flow pass links them with no special
    cases:

    - ``_data_`` outputs become ``work.data<n>`` (corpus-wide counter,
      mirroring the session-wide DATAn counter);
    - ``_last_`` inputs become the tracked last-created dataset (dropped
      with a debug log when nothing has been created yet);
    - a whitelisted PROC step with no input at all, or a DATA step whose
      only SET statement is a bare ``set;``, gains the last-created
      dataset as input.

    ``last_created`` deliberately persists across file boundaries —
    consistent with the same-session assumption that already links
    ``work.*`` datasets across files.  Known limitations: a MACRO_CALL
    whose outputs are only resolved later (Fix B) does not advance
    ``last_created``, and MACRO_DEFINITION chunks never do (defining a
    macro executes nothing — their ``output_datasets`` are empty).
    """
    last_created: str | None = None
    datan = 0

    for idx, chunk in enumerate(flat_chunks):
        meta = chunk.metadata
        new_in = list(meta.input_datasets)
        new_out = list(meta.output_datasets)
        changed = False

        if "_data_" in new_out:
            resolved_out: list[str] = []
            for ds in new_out:
                if ds == "_data_":
                    datan += 1
                    ds = f"work.data{datan}"
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"implicit[_data_]: chunk {chunk.chunk_id} output resolved to '{ds}'"
                        )
                resolved_out.append(ds)
            new_out = resolved_out
            changed = True

        if "_last_" in new_in:
            if last_created is not None:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"implicit[_last_]: chunk {chunk.chunk_id} input resolved to '{last_created}'"
                    )
                new_in = [last_created if ds == "_last_" else ds for ds in new_in]
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"implicit[_last_]: chunk {chunk.chunk_id} references _LAST_ "
                        f"before any dataset was created — dropped"
                    )
                new_in = [ds for ds in new_in if ds != "_last_"]
            changed = True

        if (
            not new_in
            and last_created is not None
            and (
                (
                    chunk.kind == SasChunkKind.PROC_STEP
                    and (meta.proc_name or "") in _LAST_CONSUMING_PROCS
                )
                or (
                    chunk.kind == SasChunkKind.DATA_STEP
                    and _BARE_SET_RE.search(chunk.text)
                )
            )
        ):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"implicit[no-data=]: chunk {chunk.chunk_id} ({chunk.kind.value}) "
                    f"gains implicit input '{last_created}'"
                )
            new_in = [last_created]
            changed = True

        if changed:
            updated_meta = meta.model_copy(
                update={"input_datasets": new_in, "output_datasets": new_out}
            )
            flat_chunks[idx] = chunk.model_copy(update={"metadata": updated_meta})

        if new_out:
            # SAS sets _LAST_ to the most recently created data set; for a
            # multi-dataset DATA statement that is the last one named.
            last_created = new_out[-1]


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
# Edge tiers
# ---------------------------------------------------------------------------

# Strong edges (confirmed data flow) union their components immediately.
_STRONG_EDGE_KINDS = frozenset({"dataset_flow", "macro_body_dataset"})

# Weak edges (shared context) are resolved after discovery at component
# granularity by _resolve_weak_edges, so one widely-used %LET or utility macro
# cannot fuse the whole corpus into a single batch.
_WEAK_EDGE_KINDS = frozenset({"macro_invocation", "macro_var_flow", "macro_arg_dataset"})

# Soft safety net: warn past this many chunks (no hard cap — a long pipeline is
# legitimately one batch).
_WARN_BATCH_CHUNKS = 50

# Maximum edge descriptions included in SasBatch.reason before truncation.
_MAX_REASON_PARTS = 30


# ---------------------------------------------------------------------------
# Shared core: build flat index, discover edges, extract batches
# ---------------------------------------------------------------------------


def _build_flat_index(
    file_results: list[SasChunkResult],
) -> tuple[list[SasChunk], list[int]]:
    """
    Flatten all files into a single ordered list and re-stamp chunk IDs
    so they are globally unique across files.

    Original per-file IDs (``chunk-0001``) are replaced with
    ``f{file_rank+1}-chunk-{local_seq}`` (e.g. ``f2-chunk-0003``).
    This is done on *copies* of the SasChunk objects so the original
    SasChunkResult objects are not mutated.

    A single-file corpus skips the re-stamping entirely: per-file IDs are
    already unique, and preserving them keeps SasChunkBatcher output IDs
    identical to the input SasChunkResult IDs.

    Returns
    -------
    flat_chunks : list[SasChunk]
        Every chunk across all files in corpus order, with global IDs.
    file_offsets : list[int]
        Global index of the first chunk of each file.
    """
    if len(file_results) == 1:
        return list(file_results[0].chunks), [0]

    flat: list[SasChunk] = []
    offsets: list[int] = []

    for fi, fr in enumerate(file_results):
        offsets.append(len(flat))
        for chunk in fr.chunks:
            new_id = f"f{fi + 1}-{chunk.chunk_id}"
            stamped = chunk.model_copy(update={"chunk_id": new_id})
            if stamped.parent_id:
                stamped = stamped.model_copy(
                    update={"parent_id": f"f{fi + 1}-{chunk.parent_id}"}
                )
            flat.append(stamped)

    return flat, offsets


class _EdgeDiscovery:
    """
    Single-pass dependency-edge discovery over the flattened corpus.

    Holds the state every pass shares — the producer indices, the
    Union-Find, and the growing edge list — so each edge family reads as
    its own small method instead of one interleaved loop body.

    INVARIANT — one walk, in corpus order.  :meth:`discover` must remain a
    single pass over ``flat_chunks`` in global-index order, with the edge
    families applied per chunk in the order written there, because
    :meth:`_resolve_macro_body` mutates ``produces_ds`` mid-walk: a macro
    call site's resolved outputs are registered as producers at the moment
    the call is visited, which is exactly what implements "a macro's output
    exists only once the call has executed" under the
    nearest-preceding-producer bisection used by :meth:`_dataset_flow` and
    :meth:`_macro_arg_datasets`.  Splitting the families into separate
    corpus walks would let a consumer link to a producer that, in
    sequential SAS execution, does not exist yet at its position — or miss
    one that does.
    """

    def __init__(
        self,
        flat_chunks: list[SasChunk],
        uf: _UF,
        file_of: list[int],
    ) -> None:
        self.flat_chunks = flat_chunks
        self.uf = uf
        self.file_of = file_of  # global_index → file_rank (cross_file edge flag)
        self.edges: list[_Edge] = []

        # ── producer indices (filled by _build_indices) ────────────────────
        # dataset → global indices of chunks that write it
        self.produces_ds: dict[str, list[int]] = defaultdict(list)
        # macro name → global index of defining chunk (last definition wins)
        self.defines_macro: dict[str, int] = {}
        # macro variable name → global indices of chunks that create it, via
        # SYMPUT/SQL-INTO side effects or %LET/%GLOBAL/%LOCAL declarations
        # (both treated identically).
        self.produces_macrovar: dict[str, list[int]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Shared edge constructor
    # ------------------------------------------------------------------

    def _add_edge(self, *, kind: str, from_idx: int, to_idx: int, via: str) -> bool:
        """
        Create a dependency edge between two global chunk indices, log it,
        and — for strong edge kinds only — union the endpoints in the
        Union-Find structure.  Weak edges are recorded but left to
        _resolve_weak_edges.

        Returns whether a union actually merged two previously-distinct
        components (always False for weak kinds).

        Shared by every edge-discovery pass so the "build edge, append,
        union, log" sequence exists in exactly one place.
        """
        cf = self.file_of[from_idx] != self.file_of[to_idx]
        e = _Edge(
            kind=kind,
            from_id=self.flat_chunks[from_idx].chunk_id,
            to_id=self.flat_chunks[to_idx].chunk_id,
            via=via,
            from_global_idx=from_idx,
            to_global_idx=to_idx,
            cross_file=cf,
        )
        self.edges.append(e)
        merged = (
            self.uf.union(from_idx, to_idx) if kind in _STRONG_EDGE_KINDS else False
        )
        if logger.isEnabledFor(logging.DEBUG):
            tier = "strong" if kind in _STRONG_EDGE_KINDS else "weak"
            logger.debug(
                f"edge[{kind}/{tier}] {e.from_id}(g{from_idx}) → {e.to_id}(g{to_idx}) via={via!r} cross_file={cf} uf_merged={merged}"
            )
        return merged

    # ------------------------------------------------------------------
    # Pass 0 — producer indices
    # ------------------------------------------------------------------

    def _build_indices(self) -> None:
        for gidx, chunk in enumerate(self.flat_chunks):
            meta = chunk.metadata
            for ds in meta.output_datasets:
                self.produces_ds[ds].append(gidx)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"index[ds]:    chunk {chunk.chunk_id} (g{gidx}) writes '{ds}'"
                    )
            for mac in meta.defines_macros:
                if mac in self.defines_macro:
                    prev = self.flat_chunks[self.defines_macro[mac]]
                    prev_file = self.file_of[self.defines_macro[mac]]
                    cur_file = self.file_of[gidx]
                    logger.warning(
                        f"index[macro]: macro '%{mac}' redefined — chunk {chunk.chunk_id} (file {cur_file}) overrides chunk {prev.chunk_id} (file {prev_file})"
                    )
                self.defines_macro[mac] = gidx
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"index[macro]: chunk {chunk.chunk_id} (g{gidx}) defines %{mac}"
                    )

            for mvar in meta.produces_macrovars:
                self.produces_macrovar[mvar].append(gidx)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"index[macrovar]: chunk {chunk.chunk_id} (g{gidx}) creates &{mvar}"
                    )

            # %LET / %GLOBAL / %LOCAL declarations register in the same
            # namespace as the SYMPUT/SQL-INTO producers above. Guard against
            # double-registering a name a chunk both declares and produces.
            for mvar in meta.declared_macro_vars:
                if gidx not in self.produces_macrovar[mvar]:
                    self.produces_macrovar[mvar].append(gidx)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"index[macrovar-decl]: chunk {chunk.chunk_id} (g{gidx}) declares &{mvar}"
                        )

            # Literal macro-body outputs (hard-coded dataset names in a %MACRO
            # body) are registered as if the MACRO_DEFINITION chunk itself were
            # the producer; macro_invocation edges then link every call site in.
            if meta.body_literal_outputs:
                for ds in meta.body_literal_outputs:
                    self.produces_ds[ds].append(gidx)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"index[ds-literal-body]: chunk {chunk.chunk_id} (g{gidx}) macro body writes '{ds}'"
                        )

        logger.debug(
            f"index built  unique_output_datasets={len(self.produces_ds)}  unique_macro_definitions={len(self.defines_macro)}"
        )

    # ------------------------------------------------------------------
    # Main walk
    # ------------------------------------------------------------------

    def discover(self) -> list[_Edge]:
        """Build the producer indices, then walk the corpus once emitting
        edges — see the class docstring for why this stays one pass."""
        self._build_indices()

        for cidx, chunk in enumerate(self.flat_chunks):
            meta = chunk.metadata
            self._dataset_flow(cidx, chunk, meta)
            self._macrovar_flow(cidx, chunk, meta)
            self._macro_invocation(cidx, chunk, meta)
            if chunk.kind == SasChunkKind.MACRO_CALL:
                # Parsed once per call chunk; the text never changes mid-walk,
                # and both passes below need the same (positional, keyword) split.
                call_args = _parse_call_args(chunk.text)
                self._macro_arg_datasets(cidx, chunk, call_args)
                self._resolve_macro_body(cidx, chunk, meta, call_args)

        cf_count = sum(1 for e in self.edges if e.cross_file)
        logger.info(
            f"edges discovered  total={len(self.edges)}  dataset_flow={sum(1 for e in self.edges if e.kind == 'dataset_flow')}  macro_invocation={sum(1 for e in self.edges if e.kind == 'macro_invocation')}  macro_arg_dataset={sum(1 for e in self.edges if e.kind == 'macro_arg_dataset')}  macro_body_dataset={sum(1 for e in self.edges if e.kind == 'macro_body_dataset')}  macro_var_flow={sum(1 for e in self.edges if e.kind == 'macro_var_flow')}  cross_file={cf_count}"
        )
        if cf_count:
            logger.info(
                f"cross-file edges: {[(e.from_id, '→', e.to_id, 'via', e.via) for e in self.edges if e.cross_file]}"
            )
        return self.edges

    # ------------------------------------------------------------------
    # Edge families — one method per kind, called per chunk by discover
    # ------------------------------------------------------------------

    def _dataset_flow(
        self, cidx: int, chunk: SasChunk, meta: SasChunkMetadata
    ) -> None:
        # A consumer links only to the nearest preceding producer in corpus
        # order — the state a sequential SAS session would read — so independent
        # jobs reusing a scratch name like work.tmp stay in separate components.
        for ds in meta.input_datasets:
            plist = self.produces_ds.get(ds)
            if not plist:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"edge-skip[ds]: '{ds}' read by {chunk.chunk_id} — no producer in corpus"
                    )
                continue
            pos = bisect_left(plist, cidx) - 1
            if pos < 0:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"edge-skip[ds]: '{ds}' read by {chunk.chunk_id} — no preceding producer"
                    )
                continue
            self._add_edge(
                kind="dataset_flow",
                from_idx=plist[pos],
                to_idx=cidx,
                via=ds,
            )

    def _macrovar_flow(
        self, cidx: int, chunk: SasChunk, meta: SasChunkMetadata
    ) -> None:
        # Mirrors dataset_flow for the macro-variable namespace: a chunk that
        # creates &cutoff links to any chunk that references &cutoff.
        for mvar in meta.consumes_macrovars:
            if mvar not in self.produces_macrovar:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"edge-skip[macrovar]: '&{mvar}' read by {chunk.chunk_id} — no producer in corpus"
                    )
                continue
            for pidx in self.produces_macrovar[mvar]:
                if pidx == cidx:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"edge-skip[macrovar]: self-ref '&{mvar}' in {chunk.chunk_id}"
                        )
                    continue
                self._add_edge(
                    kind="macro_var_flow",
                    from_idx=pidx,
                    to_idx=cidx,
                    via=f"&{mvar}",
                )

    def _macro_invocation(
        self, cidx: int, chunk: SasChunk, meta: SasChunkMetadata
    ) -> None:
        for mac in meta.invokes_macros:
            if mac not in self.defines_macro:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"edge-skip[macro]: '%{mac}' in {chunk.chunk_id} — no definition in corpus"
                    )
                continue
            didx = self.defines_macro[mac]
            if didx == cidx:
                continue
            self._add_edge(
                kind="macro_invocation",
                from_idx=didx,
                to_idx=cidx,
                via=f"%{mac}",
            )

    def _macro_arg_datasets(
        self,
        cidx: int,
        chunk: SasChunk,
        call_args: tuple[list[str], dict[str, str]],
    ) -> None:
        # Scan only the parsed argument values, deduped so one dataset name
        # yields at most one edge; link to the nearest preceding producer.
        arg_pos, arg_kw = call_args
        seen_arg_ds: set[str] = set()
        for val in (*arg_pos, *arg_kw.values()):
            for tok in _ARG_DS_TOKEN_RE.findall(val):
                # Canonicalise so a bare one-level argument matches its
                # work.-qualified producer (index keys are canonical).
                ds_norm = _canon_ds(tok.lower())
                if ds_norm in seen_arg_ds:
                    continue
                seen_arg_ds.add(ds_norm)
                plist = self.produces_ds.get(ds_norm)
                if not plist:
                    continue
                pos = bisect_left(plist, cidx) - 1
                if pos < 0:
                    continue
                self._add_edge(
                    kind="macro_arg_dataset",
                    from_idx=plist[pos],
                    to_idx=cidx,
                    via=ds_norm,
                )

    def _resolve_macro_body(
        self,
        cidx: int,
        chunk: SasChunk,
        meta: SasChunkMetadata,
        call_args: tuple[list[str], dict[str, str]],
    ) -> None:
        # Parameterised resolution: the call invokes a macro whose body
        # references a dataset only through a parameter (e.g. "data &ds.;" in
        # %clean(ds)). Resolve using the actual argument at this call site, then
        # treat the resolved name as if this chunk produced/consumed it. Under
        # nearest-preceding-producer semantics this mid-walk registration is
        # correct: the output exists only once the call has executed.
        pos_args, kw_args = call_args
        for mac in meta.invokes_macros:
            if mac not in self.defines_macro:
                continue
            didx = self.defines_macro[mac]
            def_meta = self.flat_chunks[didx].metadata
            if not (def_meta.body_param_outputs or def_meta.body_param_inputs):
                continue

            if logger.isEnabledFor(logging.DEBUG):
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

            # Resolved values are canonicalised (one-level → work.) to match the
            # producer index; a value still containing ``&`` is unresolvable and
            # kept verbatim, matching nothing rather than a guessed name.
            resolved_outputs: list[str] = []
            for entry in def_meta.body_param_outputs:
                val = _resolve(entry)
                if val:
                    val = val if "&" in val else _canon_ds(val)
                    resolved_outputs.append(val)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"macro_body_dataset: resolved param '{entry['param']}' → output '{val}'  (call {chunk.chunk_id})"
                        )

            resolved_inputs: list[str] = []
            for entry in def_meta.body_param_inputs:
                val = _resolve(entry)
                if val:
                    val = val if "&" in val else _canon_ds(val)
                    resolved_inputs.append(val)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"macro_body_dataset: resolved param '{entry['param']}' → input '{val}'  (call {chunk.chunk_id})"
                        )

            # Register this call site as a producer of its resolved outputs so
            # later chunks get linked via the normal dataset_flow pass. insort
            # (not append) keeps produces_ds sorted, the invariant the
            # nearest-preceding bisect lookups rely on.
            for ds in resolved_outputs:
                if cidx not in self.produces_ds[ds]:
                    insort(self.produces_ds[ds], cidx)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"index[ds-call-resolved]: chunk {chunk.chunk_id} (g{cidx}) resolved-produces '{ds}' via %{mac}"
                        )

            # Persist resolved outputs/inputs back onto the chunk's metadata so
            # _make_batch and other consumers see them through the normal fields.
            if resolved_outputs or resolved_inputs:
                # Insertion-order dedupe, NOT sorted(): _resolve_implicit_datasets
                # treats output_datasets[-1] as "the last dataset named".
                new_out = list(
                    dict.fromkeys(
                        [*chunk.metadata.output_datasets, *resolved_outputs]
                    )
                )
                new_in = list(
                    dict.fromkeys(
                        [*chunk.metadata.input_datasets, *resolved_inputs]
                    )
                )
                updated_meta = chunk.metadata.model_copy(
                    update={"output_datasets": new_out, "input_datasets": new_in},
                )
                chunk = chunk.model_copy(update={"metadata": updated_meta})
                self.flat_chunks[cidx] = chunk
                meta = updated_meta
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"macro_body_dataset: chunk {chunk.chunk_id} metadata updated  output_datasets={new_out}  input_datasets={new_in}"
                    )

            # Link this call site to the nearest preceding producer of each
            # resolved input (whose actual name is only known at this site).
            for ds in resolved_inputs:
                plist = self.produces_ds.get(ds)
                preceding = None
                if plist:
                    pos = bisect_left(plist, cidx) - 1
                    if pos >= 0:
                        preceding = plist[pos]
                if preceding is None:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"macro_body_dataset: resolved input '{ds}' has no preceding producer in corpus (call {chunk.chunk_id})"
                        )
                    continue
                self._add_edge(
                    kind="macro_body_dataset",
                    from_idx=preceding,
                    to_idx=cidx,
                    via=ds,
                )


def _discover_edges(
    flat_chunks: list[SasChunk],
    uf: _UF,
    *,
    file_of: list[int],
) -> list[_Edge]:
    """
    Walk flat_chunks once, building producer indices and emitting edges.

    Thin wrapper over :class:`_EdgeDiscovery` — see its docstring for the
    single-pass corpus-order invariant.  ``file_of`` (global_index →
    file_rank) is computed once by the caller and shared across all passes.
    """
    return _EdgeDiscovery(flat_chunks, uf, file_of).discover()


def _absorb_context(
    flat_chunks: list[SasChunk],
    uf: _UF,
    edges: list[_Edge],
    *,
    include_options: bool,
    include_comments: bool,
    file_of: list[int],
    globals_root: int | None = None,
) -> None:
    """
    Pull OPTIONS/GLOBAL_STATEMENT and/or COMMENT_BLOCK chunks into the
    component of the first substantive chunk that follows them,
    *within the same file only*.  We never absorb across a file boundary.

    Runs AFTER weak-edge resolution: a context chunk that was promoted to
    the global-context component (e.g. a ``%let`` consumed by two
    independent pipelines) is skipped here — absorbing it into the chunk
    that happens to follow it would drag the entire globals component into
    one pipeline's batch, recreating exactly the mega-batch the promotion
    exists to prevent.

    ``file_of`` (global_index → file_rank) is computed once by the caller and
    shared across all passes.
    """
    n = len(flat_chunks)

    # next_substantive[i]: nearest following same-file chunk that is not a
    # context kind, or None. One backward pass replaces the per-chunk forward
    # rescan, which was quadratic on long runs of consecutive context chunks.
    next_substantive: list[int | None] = [None] * n
    nxt: int | None = None
    for idx in range(n - 1, -1, -1):
        if idx + 1 < n and file_of[idx + 1] != file_of[idx]:
            nxt = None  # don't cross file boundaries
        next_substantive[idx] = nxt
        if flat_chunks[idx].kind not in _CONTEXT_KINDS:
            nxt = idx

    for idx, chunk in enumerate(flat_chunks):
        kind = chunk.kind
        is_option = kind in _OPTION_KINDS and include_options
        is_comment = kind in _COMMENT_KINDS and include_comments
        if not (is_option or is_comment):
            continue
        if globals_root is not None and uf.find(idx) == globals_root:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"absorb-skip[globals]: {chunk.chunk_id} is in the global-context component"
                )
            continue

        nxt_idx = next_substantive[idx]
        if nxt_idx is None:
            continue
        nxt_chunk = flat_chunks[nxt_idx]
        reason = "options_context" if is_option else "comment_context"
        merged = uf.union(idx, nxt_idx)
        if logger.isEnabledFor(logging.DEBUG):
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


def _resolve_weak_edges(
    uf: _UF,
    edges: list[_Edge],
    flat_chunks: list[SasChunk],
) -> int | None:
    """
    Resolve weak edges (macro_invocation, macro_var_flow, macro_arg_dataset)
    at component granularity, after strong-edge discovery has shaped the
    core components (context absorption runs after this pass, so that a
    globally-consumed ``%let`` is promoted here before absorption could
    merge it into whichever pipeline happens to follow it in the source).

    For each producer component (the from-side of weak edges):

    - exactly ONE distinct consumer component → the producer is absorbed
      into it (a macro defined for a single caller batches with that
      caller, exactly as before);
    - TWO OR MORE distinct consumer components → the producer is globally
      affecting.  All such producers are unioned together into one
      global-context component, which the caller emits as the FIRST batch.

    The producer→consumers map is a snapshot of component roots taken
    before any weak union is applied, so the outcome does not depend on
    iteration order beyond the deterministic ascending-root processing.
    (A producer absorbed into its lone consumer can in principle leave
    another producer with fewer distinct consumers than the snapshot saw;
    this single-pass resolution accepts that slack rather than iterating
    to a fixpoint, keeping the result deterministic and cheap.)

    Returns the root of the global-context component, or None when no
    producer spans multiple components.
    """
    consumers_by_producer: dict[int, set[int]] = defaultdict(set)
    for e in edges:
        if e.kind not in _WEAK_EDGE_KINDS:
            continue
        pr = uf.find(e.from_global_idx)
        cr = uf.find(e.to_global_idx)
        if pr != cr:
            consumers_by_producer[pr].add(cr)

    global_roots: list[int] = []
    for pr in sorted(consumers_by_producer):
        crs = consumers_by_producer[pr]
        if len(crs) == 1:
            cr = next(iter(crs))
            uf.union(pr, cr)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"weak-resolve[absorb]: producer component of {flat_chunks[pr].chunk_id} "
                    f"absorbed into its single consumer component ({flat_chunks[cr].chunk_id})"
                )
        else:
            global_roots.append(pr)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"weak-resolve[global]: producer component of {flat_chunks[pr].chunk_id} "
                    f"feeds {len(crs)} distinct components — promoted to global context"
                )

    if not global_roots:
        return None

    anchor = global_roots[0]
    for r in global_roots[1:]:
        uf.union(anchor, r)
    root = uf.find(anchor)
    logger.info(
        f"weak-resolve: {len(global_roots)} globally-affecting producer component(s) "
        f"collected into global-context component (root=g{root})"
    )
    return root


def _make_batch(
    global_indices: list[int],
    flat_chunks: list[SasChunk],
    component_edges: list[_Edge],
    batch_number: int,
    file_of: list[int],
    *,
    is_global_context: bool = False,
) -> SasBatch:
    """
    Build a SasBatch from a set of global chunk indices.

    Chunks are ordered by (file_rank, start_line) so producers always
    appear before consumers in the batch text.

    ``component_edges`` are exactly the edges internal to this batch's
    Union-Find component (every edge's endpoints share a component root,
    since each edge unions them), pre-grouped by the caller so the reason
    string is built without re-scanning the whole corpus edge list.
    ``file_of`` (global_index → file_rank) is computed once by the caller.
    """
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
        # %LET / %GLOBAL / %LOCAL declarations also satisfy a name intra-batch
        # (mirrors the producer index in _discover_edges).
        intra_macrovars.update(c.metadata.declared_macro_vars)
        # A %MACRO definition's literal body outputs count as batch outputs when
        # a call site is also in this batch (usually the case).
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
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"_make_batch {bid}: ext input '{ds}'")

    # ── external macro requirements ────────────────────────────────────────
    ext_macros: list[str] = []
    seen_em: set[str] = set()
    # Standard SAS-provided autocall macros — never a missing dependency, so
    # excluded from ext_macros but still tracked separately.
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
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"_make_batch {bid}: standard autocall macro '%{mac}'"
                        )
                continue
            if mac not in seen_em:
                ext_macros.append(mac)
                seen_em.add(mac)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"_make_batch {bid}: ext macro '%{mac}'")

    # ── external libref requirements ────────────────────────────────────────
    # Librefs used by this batch's dataset I/O but not assigned inside it,
    # excluding the SAS-supplied default libraries.
    defined_librefs: set[str] = set()
    used_librefs: set[str] = set()
    for c in member_chunks:
        m = c.metadata
        defined_librefs.update(m.defines_librefs)
        for ds in (
            *m.input_datasets,
            *m.output_datasets,
            *m.body_literal_inputs,
            *m.body_literal_outputs,
        ):
            # Quoted physical paths ('c:/tmp/x') address a file directly, not
            # a library member; special _name_ tokens have no libref either.
            if ds.startswith("'") or "." not in ds:
                continue
            used_librefs.add(ds.split(".", 1)[0])
    ext_librefs = sorted(used_librefs - defined_librefs - _DEFAULT_LIBREFS)
    if ext_librefs:
        logger.debug(f"_make_batch {bid}: ext librefs {ext_librefs}")

    # ── external macro-variable requirements (ROADMAP Phase 2) ──────────────
    ext_macrovars: list[str] = []
    seen_emv: set[str] = set()
    for c in member_chunks:
        for mvar in c.metadata.consumes_macrovars:
            if mvar not in intra_macrovars and mvar not in seen_emv:
                ext_macrovars.append(mvar)
                seen_emv.add(mvar)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"_make_batch {bid}: ext macrovar '&{mvar}'")

    # ── reason string ──────────────────────────────────────────────────────
    # Group this component's edges by (from, to) pair, in discovery order.
    edges_by_pair: dict[tuple[int, int], list[_Edge]] = defaultdict(list)
    for e in component_edges:
        edges_by_pair[(e.from_global_idx, e.to_global_idx)].append(e)

    reason_parts: list[str] = []
    seen_r: set[str] = set()
    for edge_list in edges_by_pair.values():
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

    # Cap the reason string — a large component can carry hundreds of
    # edges, and an unbounded reason bloats every serialised result.
    if len(reason_parts) > _MAX_REASON_PARTS:
        omitted = len(reason_parts) - _MAX_REASON_PARTS
        reason_parts = reason_parts[:_MAX_REASON_PARTS]
        reason_parts.append(f"… +{omitted} more edges")

    reason = "; ".join(reason_parts) or "[context absorption only]"
    if is_global_context:
        reason = f"[global context — consumed by multiple batches] {reason}"
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"_make_batch {bid}: reason={reason!r}")

    if len(member_chunks) > _WARN_BATCH_CHUNKS:
        logger.warning(
            f"_make_batch {bid}: batch has {len(member_chunks)} chunks "
            f"(> {_WARN_BATCH_CHUNKS}) — downstream consumers may want to split it"
        )

    return SasBatch(
        batch_id=bid,
        chunks=member_chunks,
        reason=reason,
        is_global_context=is_global_context,
        source_files=seen_files,
        input_datasets=ext_inputs,
        output_datasets=sorted(intra_outputs),
        required_macros=ext_macros,
        required_librefs=ext_librefs,
        defined_macros=sorted(intra_macros),
        produced_macrovars=sorted(intra_macrovars),
        required_macrovars=ext_macrovars,
        standard_autocall_macros=sorted(standard_autocall),
    )


def _extract_result(
    uf: _UF,
    flat_chunks: list[SasChunk],
    edges: list[_Edge],
    file_of: list[int],
    globals_root: int | None = None,
) -> tuple[list[SasBatch], list[SasChunk]]:
    """Convert UF components → (batches, singletons).

    When ``globals_root`` names a component (see :func:`_resolve_weak_edges`),
    that component is emitted FIRST as ``batch-001`` with
    ``is_global_context=True`` — even at size 1, since "process these
    globally-consumed definitions before everything else" is meaningful for
    a lone macro definition too.
    """
    n = len(flat_chunks)
    components = uf.components(n)

    # Group edges by component root, keyed on the from-endpoint so a weak edge
    # that spans components buckets with its producer.
    edges_by_component: dict[int, list[_Edge]] = defaultdict(list)
    for e in edges:
        edges_by_component[uf.find(e.from_global_idx)].append(e)

    batches: list[SasBatch] = []
    singletons: list[SasChunk] = []

    if globals_root is not None:
        members = components.pop(globals_root)
        batch = _make_batch(
            global_indices=members,
            flat_chunks=flat_chunks,
            component_edges=edges_by_component.get(globals_root, []),
            batch_number=1,
            file_of=file_of,
            is_global_context=True,
        )
        batches.append(batch)
        logger.info(
            f"batch {batch.batch_id} [GLOBAL CONTEXT]  chunks={len(batch.chunks)}  source_files={batch.source_files}  def_macros={batch.defined_macros}  produced_macrovars={batch.produced_macrovars}"
        )

    for root, members in sorted(components.items()):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"component root={root}  size={len(members)}  members={[flat_chunks[i].chunk_id for i in members]}"
            )
        if len(members) == 1:
            solo = flat_chunks[members[0]]
            singletons.append(solo)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"singleton: {solo.chunk_id}  kind={solo.kind.value}  source={solo.source_id}"
                )
            continue

        batch = _make_batch(
            global_indices=members,
            flat_chunks=flat_chunks,
            component_edges=edges_by_component.get(root, []),
            batch_number=len(batches) + 1,
            file_of=file_of,
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
    dependency-aware :class:`SasBatch` objects — a single-file convenience
    over :class:`MultiFileBatcher`, which does all the work.

    Chunk IDs in the result match the input SasChunkResult IDs (single-file
    corpora skip the multi-file ID re-stamping).  When globally-consumed
    definitions exist (see :class:`MultiFileBatcher`), the first batch is a
    global-context batch with ``is_global_context=True``.

    For multi-file batching use :class:`MultiFileBatcher` directly.

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
        self._delegate = MultiFileBatcher(
            include_comment_chunks=include_comment_chunks,
            include_options_chunks=include_options_chunks,
        )
        logger.debug(
            f"SasChunkBatcher  include_comment={include_comment_chunks}  include_options={include_options_chunks}"
        )

    def batch(self, chunk_result: SasChunkResult) -> SasBatchResult:
        """Compute dependency-driven batches for all chunks in *chunk_result*.

        Wraps the result as a one-file corpus and delegates to
        :class:`MultiFileBatcher`; the returned :class:`SasBatchResult` has
        exactly one entry in ``source_ids``.
        """
        return self._delegate.batch(SasCorpus(file_results=[chunk_result]))


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

    Edges are tiered: confirmed dataset flow merges components directly,
    while shared-context links (macro definitions, macro variables, call
    arguments) merge only when unambiguous — a producer consumed by two or
    more otherwise-independent components is instead promoted to a single
    **global-context batch**, emitted first with ``is_global_context=True``.
    Dataset consumers link to the *nearest preceding* producer in corpus
    order, so unrelated jobs reusing a scratch name (``work.tmp``) no
    longer fuse into one batch.

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

        from chunker import SasSemanticChunker, SasCorpus
        from chunker.batcher import MultiFileBatcher

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
    ) -> "tuple[SasCorpus, SasBatchResult]":
        """
        Convenience: chunk each file in *paths* and batch the resulting corpus.

        Returns ``(corpus, result)`` so callers can inspect both.

        Parameters
        ----------
        paths
            Ordered list of ``.sas`` file paths.  Order establishes the
            default execution sequence for resolving tie-breaks.
        chunker_kwargs
            Forwarded to :class:`~chunker.chunker.SasSemanticChunker`.
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

    def batch(self, corpus: SasCorpus) -> SasBatchResult:
        """
        Compute cross-file dependency batches for all files in *corpus*.

        Returns a :class:`SasBatchResult` containing all batches
        (including cross-file ones) and standalone singletons.
        """
        source_ids = corpus.source_ids
        logger.info(
            f"MultiFileBatcher.batch: start  files={len(corpus.file_results)}  source_ids={source_ids}  total_chunks={sum(len(r.chunks) for r in corpus.file_results)}"
        )
        t0 = time.perf_counter()

        if not corpus.file_results or not corpus.all_chunks:
            logger.warning("MultiFileBatcher.batch: corpus is empty")
            return SasBatchResult(source_ids=source_ids)

        # ── flatten corpus ──────────────────────────────────────────────────
        flat_chunks, file_offsets = _build_flat_index(corpus.file_results)
        n = len(flat_chunks)
        uf = _UF(n)
        # global_index → file_rank, computed once and shared by every pass.
        file_of = _file_of_map(file_offsets, n)

        logger.debug(
            f"MultiFileBatcher.batch: flat index  total={n}  file_offsets={file_offsets}"
        )

        # ── implicit dataset resolution (_LAST_ / _DATA_ / no DATA=) ────────
        _resolve_implicit_datasets(flat_chunks)

        # ── edge discovery ──────────────────────────────────────────────────
        edges = _discover_edges(flat_chunks, uf, file_of=file_of)

        # ── weak-edge resolution (absorb or promote to global context) ──────
        globals_root = _resolve_weak_edges(uf, edges, flat_chunks)

        # ── context absorption (same-file only; skips globals members) ──────
        if self.include_options_chunks or self.include_comment_chunks:
            _absorb_context(
                flat_chunks,
                uf,
                edges,
                include_options=self.include_options_chunks,
                include_comments=self.include_comment_chunks,
                file_of=file_of,
                globals_root=globals_root,
            )

        # ── extract batches + singletons ────────────────────────────────────
        batches, singletons = _extract_result(
            uf, flat_chunks, edges, file_of, globals_root
        )

        cf_count = sum(1 for b in batches if b.is_cross_file)
        elapsed = time.perf_counter() - t0
        logger.info(
            f"MultiFileBatcher.batch: done  files={len(corpus.file_results)}  batches={len(batches)}  cross_file_batches={cf_count}  singletons={len(singletons)}  elapsed={elapsed:.3f}s"
        )

        return SasBatchResult(
            source_ids=source_ids,
            batches=batches,
            singletons=singletons,
        )
