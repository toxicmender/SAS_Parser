"""Dataset edge-family view."""

from __future__ import annotations

from collections.abc import Iterable

from .models import DependencyEdge, DependencyEdgeFamily


def dataset_edges(edges: Iterable[DependencyEdge]) -> tuple[DependencyEdge, ...]:
    return tuple(edge for edge in edges if edge.family is DependencyEdgeFamily.DATASET)
