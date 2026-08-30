"""Macro and macro-variable edge-family views."""

from __future__ import annotations

from collections.abc import Iterable

from .models import DependencyEdge, DependencyEdgeFamily


def macro_edges(edges: Iterable[DependencyEdge]) -> tuple[DependencyEdge, ...]:
    families = {DependencyEdgeFamily.MACRO, DependencyEdgeFamily.MACRO_VARIABLE}
    return tuple(edge for edge in edges if edge.family in families)
