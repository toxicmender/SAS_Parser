"""Public deterministic dependency facts from the batcher's single walk."""

from __future__ import annotations

from ..batching import (
    _UF,
    _build_flat_index,
    _discover_edges,
    _Edge,
    _file_of_map,
    _resolve_implicit_datasets,
)
from ..models import SasCorpus
from .models import DependencyEdge, DependencyEdgeFamily

_MACRO_KINDS = frozenset(
    {"macro_invocation", "macro_arg_dataset", "macro_body_dataset"}
)


def _family(edge: _Edge) -> DependencyEdgeFamily:
    if edge.kind == "dataset_flow":
        return DependencyEdgeFamily.DATASET
    if edge.kind == "macro_var_flow":
        return DependencyEdgeFamily.MACRO_VARIABLE
    if edge.kind in _MACRO_KINDS:
        return DependencyEdgeFamily.MACRO
    return DependencyEdgeFamily.CONTEXT


def discover_dependency_edges(corpus: SasCorpus) -> tuple[DependencyEdge, ...]:
    """Return edges in discovery order without mutating the supplied corpus."""

    copied_results = [result.model_copy(deep=True) for result in corpus.file_results]
    flat_chunks, file_offsets = _build_flat_index(copied_results)
    if not flat_chunks:
        return ()
    _resolve_implicit_datasets(flat_chunks)
    file_of = _file_of_map(file_offsets, len(flat_chunks))
    edges = _discover_edges(flat_chunks, _UF(len(flat_chunks)), file_of=file_of)
    return tuple(
        DependencyEdge(
            family=_family(edge),
            kind=edge.kind,
            from_chunk_id=edge.from_id,
            to_chunk_id=edge.to_id,
            reason=edge.via,
            cross_file=edge.cross_file,
        )
        for edge in edges
    )
