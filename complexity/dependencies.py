"""The corpus dependency graph: which file must be migrated before which.

:class:`DependencyEdge` and :class:`DependencyGraph`, split out of
:mod:`complexity.models` because they answer a different question from the
rest of it. Every other model there scores *one* unit — a chunk, a batch, a
file — while these two describe the relationships *between* files, which is
what turns a set of verdicts into a migration order.

The rendering side lives in :mod:`complexity.graph` (the edge table and the
image); this module is the data alone.

Pure data — no logging, and no imports from the rest of this package.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field

# Singular labels for a dependency edge's subjects, keyed by the field holding
# them. Kept beside the model rather than in complexity.graph so that
# `DependencyEdge.label` stays self-contained data.
_KIND_LABELS: dict[str, str] = {
    "datasets": "dataset",
    "macros": "macro",
    "macrovars": "macro var",
    "librefs": "libref",
}


class DependencyEdge(BaseModel):
    """One directed dependency: *upstream* must be migrated before *downstream*.

    The subject lists are why the edge exists, kept apart by kind because they
    do not carry the same weight — a shared dataset is a data dependency that
    ordering alone resolves, while a shared macro means the two files cannot be
    translated by different people without agreeing on the macro first.
    """

    upstream: str
    downstream: str
    datasets: list[str] = Field(default_factory=list)
    macros: list[str] = Field(default_factory=list)
    macrovars: list[str] = Field(default_factory=list)
    librefs: list[str] = Field(default_factory=list)

    @property
    def subjects(self) -> list[tuple[str, str]]:
        """``(kind label, name)`` for everything on this edge, datasets first."""
        return [
            (_KIND_LABELS[field], name)
            for field in ("datasets", "macros", "macrovars", "librefs")
            for name in getattr(self, field)
        ]

    @property
    def label(self) -> str:
        """The edge's causes as one line: ``dataset raw.customers, macro fmt``."""
        return ", ".join(f"{kind} {name}" for kind, name in self.subjects) or "—"

    def __str__(self) -> str:
        return f"{self.upstream} -> {self.downstream} ({self.label})"


class DependencyGraph(BaseModel):
    """File-level dependency structure across one analysed corpus.

    Built by :func:`complexity.graph.build_graph` and rendered by that module's
    ``render_*`` functions; everything here is derivation from the edges alone.

    **A "DAG" is an aspiration, not a guarantee.** Two SAS jobs can each read a
    dataset the other writes; a corpus with a cycle is unusual but not invalid,
    and silently emitting a topological order for one would be a confident lie.
    So :attr:`cycles` is reported explicitly, cycle members are parked in a
    final :attr:`layers` entry rather than interleaved, and :attr:`is_acyclic`
    states which case the reader is looking at.
    """

    #: Every analysed file, including the isolated ones. Sorted, so a corpus
    #: with no edges still reports its members rather than an empty graph.
    nodes: list[str] = Field(default_factory=list)
    edges: list[DependencyEdge] = Field(default_factory=list)

    def _adjacency(self) -> tuple[dict[str, set[str]], dict[str, int]]:
        """``upstream -> downstreams`` and each node's in-degree."""
        out: dict[str, set[str]] = {n: set() for n in self.nodes}
        indegree: dict[str, int] = {n: 0 for n in self.nodes}
        for edge in self.edges:
            # An edge naming a file `nodes` does not list would otherwise
            # KeyError; tolerate it by admitting the node.
            for node in (edge.upstream, edge.downstream):
                out.setdefault(node, set())
                indegree.setdefault(node, 0)
            if edge.downstream not in out[edge.upstream]:
                out[edge.upstream].add(edge.downstream)
                indegree[edge.downstream] += 1
        return out, indegree

    @computed_field  # type: ignore[prop-decorator]
    @property
    def layers(self) -> list[list[str]]:
        """Files grouped into migration waves, earliest first (Kahn levelling).

        Everything in one layer can be migrated in parallel once every earlier
        layer is done. Nodes caught in a cycle cannot be levelled at all, so
        they are appended as a final layer rather than given a position the
        graph does not support — see :attr:`cycles`.
        """
        out, indegree = self._adjacency()
        frontier = sorted(n for n, deg in indegree.items() if deg == 0)
        levels: list[list[str]] = []
        placed = 0
        while frontier:
            levels.append(frontier)
            placed += len(frontier)
            nxt: set[str] = set()
            for node in frontier:
                for child in out[node]:
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        nxt.add(child)
            frontier = sorted(nxt)
        if placed < len(indegree):
            levels.append(sorted(n for n, deg in indegree.items() if deg > 0))
        return levels

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cycles(self) -> list[list[str]]:
        """Groups of files that depend on themselves in a loop, each sorted.

        Strongly connected components of more than one file, by an iterative
        Tarjan — a deep corpus should not be able to exhaust the recursion
        limit while drawing its own dependency graph.
        """
        out, _ = self._adjacency()
        index_of: dict[str, int] = {}
        low: dict[str, int] = {}
        on_stack: set[str] = set()
        stack: list[str] = []
        counter = 0
        found: list[list[str]] = []

        for root in sorted(out):
            if root in index_of:
                continue
            # Each frame is (node, its not-yet-visited children).
            work: list[tuple[str, list[str]]] = [(root, sorted(out[root]))]
            index_of[root] = low[root] = counter
            counter += 1
            stack.append(root)
            on_stack.add(root)
            while work:
                node, children = work[-1]
                if children:
                    child = children.pop(0)
                    if child not in index_of:
                        index_of[child] = low[child] = counter
                        counter += 1
                        stack.append(child)
                        on_stack.add(child)
                        work.append((child, sorted(out[child])))
                    elif child in on_stack:
                        low[node] = min(low[node], index_of[child])
                    continue
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == index_of[node]:
                    component: list[str] = []
                    while True:
                        member = stack.pop()
                        on_stack.discard(member)
                        component.append(member)
                        if member == node:
                            break
                    if len(component) > 1:
                        found.append(sorted(component))
        return sorted(found)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_acyclic(self) -> bool:
        """Whether the graph really is a DAG — false when any cycle was found."""
        return not self.cycles

    @property
    def roots(self) -> list[str]:
        """Files nothing else has to be migrated before: where to start."""
        _, indegree = self._adjacency()
        return sorted(n for n, deg in indegree.items() if deg == 0)

    @property
    def leaves(self) -> list[str]:
        """Files nothing depends on: safe to leave until last."""
        out, _ = self._adjacency()
        return sorted(n for n, children in out.items() if not children)

    @property
    def is_empty(self) -> bool:
        """Whether any file depends on any other at all."""
        return not self.edges

    def __str__(self) -> str:
        return (
            f"DependencyGraph(nodes={len(self.nodes)}, edges={len(self.edges)}, "
            f"acyclic={self.is_acyclic})"
        )

