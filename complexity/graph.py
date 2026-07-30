"""Building and drawing the corpus dependency graph.

:mod:`complexity.crossfile` answers "what does *this* file depend on?" one file
at a time. That is the right shape for a file report and the wrong shape for
planning: a migration is ordered work, and the order is a property of the whole
corpus, not of any file in it. :func:`build_graph` folds every resolved
reference into file-to-file edges; the structure read off them — the roots to
start from, the waves, the cycles — lives on
:class:`~complexity.models.DependencyGraph`, with the rest of the data model.

Imports and exports are the same dependency seen from both ends: an import in
file B naming peer A and an export in A naming peer B both mean "A before B",
so both are folded into one ``A -> B`` edge carrying every name that caused it.

Rendering comes in two forms, and the Markdown one is not a fallback for the
image — it is the primary artefact. :func:`render_markdown` works in every
viewer, in a terminal, and in the ``--out``/stdout path where there is nowhere
to put a file. :func:`render_png` adds the picture when matplotlib is installed
(the ``graph`` extra) and an output directory exists, and returns ``None``
rather than raising when it is not: a missing optional dependency must not fail
an analysis run that is otherwise complete.

Logger name: ``complexity.graph``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from .models import DependencyEdge, DependencyGraph

if TYPE_CHECKING:  # pragma: no cover - typing only, and avoids an import cycle
    from .crossfile import CrossFileIndex

logger = logging.getLogger(__name__)

# Beyond this many files the image is a hairball — every node illegible, no
# structure readable — and the edge table is strictly the better artefact. The
# Markdown always renders; only the picture is capped.
MAX_GRAPH_NODES = 60

#: Construct-name prefix -> the :class:`~complexity.models.DependencyEdge`
#: field it accumulates into. Mirrors
#: :data:`complexity.crossfile.CROSS_FILE_CONSTRUCTS`; a prefix missing here
#: contributes an unlabelled edge rather than no edge.
_EDGE_KINDS: dict[str, str] = {
    "dataset": "datasets",
    "macro": "macros",
    "macrovar": "macrovars",
    "libref": "librefs",
}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def build_graph(
    index: "CrossFileIndex", *, nodes: Iterable[str] | None = None
) -> DependencyGraph:
    """Fold a resolved :class:`~complexity.crossfile.CrossFileIndex` into a graph.

    Imports and exports are mirror images of one dependency, so both are walked
    and merged: an import in B naming A and an export in A naming B produce the
    single edge ``A -> B``, not two.

    *nodes* overrides the file list, which otherwise comes from the index. Pass
    the corpus's own when it is wider — the index only knows the files it was
    built from, so a file whose every chunk was excluded from the analysis
    would otherwise be missing from the graph rather than shown as isolated.
    """
    merged: dict[tuple[str, str], dict[str, list[str]]] = {}

    def _add(upstream: str, downstream: str, kind: str, subject: str) -> None:
        # A file depending on itself is not an ordering constraint, and the
        # resolvers already exclude the owning file from every peer list — this
        # is belt and braces against a future one that does not.
        if upstream == downstream:
            return
        names = merged.setdefault((upstream, downstream), {}).setdefault(kind, [])
        if subject and subject.lower() not in {n.lower() for n in names}:
            names.append(subject)

    for source, refs in index.refs_by_source().items():
        for ref in refs:
            if not (ref.name.endswith("_import") or ref.name.endswith("_export")):
                continue  # unresolved/external: no peer, so no edge
            kind = _EDGE_KINDS.get(ref.name.rsplit("_", 1)[0])
            if kind is None:
                logger.warning(
                    f"build_graph: construct {ref.name!r} has no edge kind in "
                    f"_EDGE_KINDS; recording its edge as a dataset dependency"
                )
            inbound = ref.name.endswith("_import")
            for peer in ref.peers:
                upstream, downstream = (peer, source) if inbound else (source, peer)
                _add(upstream, downstream, kind or "datasets", ref.subject)

    edges = [
        DependencyEdge(upstream=up, downstream=down, **buckets)
        for (up, down), buckets in sorted(merged.items())
    ]
    known = index.sources if nodes is None else nodes
    graph = DependencyGraph(nodes=sorted(set(known)), edges=edges)
    logger.info(f"build_graph: {graph}")
    return graph


# ---------------------------------------------------------------------------
# Rendering — Markdown
# ---------------------------------------------------------------------------


def render_markdown(
    graph: DependencyGraph, *, image: str | None = None
) -> list[str]:
    """The ``## Dependency graph`` section, as Markdown lines.

    *image* is a path to a rendered PNG, relative to wherever the report will be
    written; when given it is linked above the table. Returns ``[]`` for a
    corpus whose files are all independent — a section asserting "no edges" adds
    nothing that the absent section does not already say.
    """
    if graph.is_empty:
        return []

    lines = ["", "## Dependency graph", ""]
    if graph.is_acyclic:
        lines.append(
            f"{len(graph.edges)} dependency edge(s) across {len(graph.nodes)} "
            f"file(s). An upstream file has to be migrated before the files "
            f"below it, so the layers are migration waves: everything in one "
            f"can be done in parallel once the previous one is done."
        )
    else:
        lines.append(
            f"{len(graph.edges)} dependency edge(s) across {len(graph.nodes)} "
            f"file(s). **This corpus is not acyclic** — see the cycles below. "
            f"Files caught in one cannot be ordered and are reported as a final "
            f"layer rather than given a position the graph does not support."
        )
    if image:
        lines += ["", f"![Dependency graph]({image})"]

    lines += [
        "",
        "| Upstream (migrate first) | Downstream | Via |",
        "| --- | --- | --- |",
    ]
    for edge in graph.edges:
        # Pipes inside a dataset name would split the row into extra columns.
        via = edge.label.replace("|", "\\|")
        lines.append(f"| {edge.upstream} | {edge.downstream} | {via} |")

    layers = graph.layers
    if layers:
        lines += ["", "### Migration order", ""]
        trailing_cycle = not graph.is_acyclic
        for position, layer in enumerate(layers, start=1):
            if trailing_cycle and position == len(layers):
                lines.append(f"- **Unordered** (in a cycle): {', '.join(layer)}")
            else:
                lines.append(f"- **Wave {position}**: {', '.join(layer)}")

    if graph.cycles:
        lines += ["", f"### Cycles ({len(graph.cycles)})", ""]
        lines.append(
            "Each of these groups of files depends on itself in a loop. Break "
            "the loop — usually by splitting whichever file both reads and "
            "writes the shared dataset — before ordering the migration."
        )
        lines.append("")
        for cycle in graph.cycles:
            lines.append(f"- {' → '.join(cycle)} → {cycle[0]}")

    return lines


# ---------------------------------------------------------------------------
# Rendering — PNG
# ---------------------------------------------------------------------------


def _box_edge(
    origin: tuple[float, float],
    towards: tuple[float, float],
    half_width: float,
    half_height: float,
) -> tuple[float, float]:
    """Where the line from *origin* to *towards* leaves *origin*'s node box.

    Offsetting horizontally would be enough for edges that run between
    columns, and wrong for the ones that do not: the members of a cycle all
    share the trailing layer, so their edges are vertical, and a horizontal
    offset would start each arrow on the far side of its own box and point it
    backwards through the node.
    """
    dx = towards[0] - origin[0]
    dy = towards[1] - origin[1]
    if not dx and not dy:
        return origin
    # Scale the direction until it first crosses one of the box's two axes.
    reach = min(
        half_width / abs(dx) if dx else float("inf"),
        half_height / abs(dy) if dy else float("inf"),
    )
    return origin[0] + dx * reach, origin[1] + dy * reach


def render_png(
    graph: DependencyGraph,
    path: Path | str,
    *,
    max_nodes: int = MAX_GRAPH_NODES,
) -> Path | None:
    """Draw *graph* left-to-right by migration wave and write a PNG to *path*.

    Returns the path written, or ``None`` when the image was not produced — no
    edges to draw, matplotlib not installed (it is the optional ``graph``
    extra), too many nodes to be legible, or a rendering failure. Never raises:
    an analysis run that produced every other artefact must not fail on the
    picture, and :func:`render_markdown` has already carried the same
    information in a form that always renders.
    """
    if graph.is_empty:
        return None
    if len(graph.nodes) > max_nodes:
        logger.info(
            f"render_png: {len(graph.nodes)} files exceeds the {max_nodes}-node "
            f"cap; the edge table is the legible artefact at this size"
        )
        return None

    try:
        import matplotlib

        # Set before pyplot is imported: report generation is headless, and the
        # default backend would try to find a display.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    except ImportError:
        logger.info(
            "render_png: matplotlib is not installed, so no dependency image "
            "was drawn (install the 'graph' extra); the edge table in the "
            "report carries the same edges"
        )
        return None

    destination = Path(path)
    layers = graph.layers
    cycle_nodes = {n for cycle in graph.cycles for n in cycle}

    # Position: one column per wave, nodes spread evenly down each column.
    positions: dict[str, tuple[float, float]] = {}
    tallest = max(len(layer) for layer in layers)
    for column, layer in enumerate(layers):
        for row, node in enumerate(layer):
            # Centre each column vertically against the tallest one, so a
            # two-node wave sits beside the middle of a six-node wave rather
            # than at its top.
            offset = (tallest - len(layer)) / 2
            positions[node] = (float(column), float(row) + offset)

    node_width, node_height = 0.78, 0.34
    figure_width = max(6.0, len(layers) * 3.0)
    figure_height = max(3.0, tallest * 0.85)

    try:
        figure, axes = plt.subplots(figsize=(figure_width, figure_height))
        for edge in graph.edges:
            if edge.upstream not in positions or edge.downstream not in positions:
                continue
            start, end = positions[edge.upstream], positions[edge.downstream]
            in_cycle = edge.upstream in cycle_nodes and edge.downstream in cycle_nodes
            axes.add_patch(
                FancyArrowPatch(
                    _box_edge(start, end, node_width / 2, node_height / 2),
                    _box_edge(end, start, node_width / 2, node_height / 2),
                    arrowstyle="-|>",
                    mutation_scale=13,
                    # Curved, so two edges between the same columns stay apart
                    # and an arrow never runs straight through a third node.
                    connectionstyle="arc3,rad=0.12",
                    color="#c1121f" if in_cycle else "#8d99ae",
                    linewidth=1.5 if in_cycle else 1.1,
                    zorder=1,
                )
            )
        for node, (x, y) in positions.items():
            in_cycle = node in cycle_nodes
            axes.add_patch(
                FancyBboxPatch(
                    (x - node_width / 2, y - node_height / 2),
                    node_width,
                    node_height,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor="#fdd5d5" if in_cycle else "#e8eef7",
                    edgecolor="#c1121f" if in_cycle else "#4a6fa5",
                    linewidth=1.4,
                    zorder=2,
                )
            )
            axes.text(
                x,
                y,
                Path(node).name,
                ha="center",
                va="center",
                fontsize=8,
                zorder=3,
            )

        axes.set_xlim(-0.8, len(layers) - 1 + 0.8)
        axes.set_ylim(-0.8, tallest - 1 + 0.8)
        # Waves read left to right; matplotlib's y axis runs the wrong way for
        # a top-down reading of each column.
        axes.invert_yaxis()
        axes.set_axis_off()
        title = "SAS file dependencies — migrate left to right"
        if not graph.is_acyclic:
            title += "  (red: cycle)"
        axes.set_title(title, fontsize=10)
        figure.tight_layout()

        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150, bbox_inches="tight")
        plt.close(figure)
    except Exception:
        # Deliberately broad: matplotlib raises a wide and version-dependent
        # range, and none of it is worth failing an otherwise complete run for.
        logger.warning(
            f"render_png: could not draw the dependency graph to {destination}; "
            f"the report's edge table still carries every edge",
            exc_info=True,
        )
        return None

    logger.info(f"render_png: wrote {destination}")
    return destination
