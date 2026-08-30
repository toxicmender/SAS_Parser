"""Enforce the v2 import direction and top-level acyclic architecture graph."""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "sas_migrate"
TOP_LEVEL_AREAS = frozenset(
    {"core", "application", "adapters", "config", "observability", "cli"}
)
SAS_CORE_FORBIDDEN_IMPORTS = (
    "app_config",
    "conversion.sharepoint",
    "langchain",
    "llm_client",
    "memory",
    "msgraph",
    "openai",
    "sharepoint",
)


def _module_name(path: pathlib.Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(("sas_migrate", *parts))


def _imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text("utf-8"), filename=str(path))
    module_parts = _module_name(path).split(".")
    package_parts = module_parts if path.name == "__init__.py" else module_parts[:-1]
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package_parts) - node.level + 1
                if keep < 1:
                    continue
                imported_parts = package_parts[:keep]
                if node.module:
                    imported_parts.extend(node.module.split("."))
                imports.add(".".join(imported_parts))
            elif node.module:
                imports.add(node.module)
    return imports


def _absolute_imports(path: pathlib.Path) -> set[str]:
    return {name for name in _imports(path) if name.startswith("sas_migrate.")}


def architecture_graph() -> dict[str, set[str]]:
    graph = {area: set() for area in TOP_LEVEL_AREAS}
    for path in PACKAGE_ROOT.rglob("*.py"):
        source_parts = _module_name(path).split(".")
        if len(source_parts) < 2 or source_parts[1] not in TOP_LEVEL_AREAS:
            continue
        source = source_parts[1]
        for imported in _absolute_imports(path):
            parts = imported.split(".")
            if len(parts) >= 2 and parts[1] in TOP_LEVEL_AREAS and parts[1] != source:
                graph[source].add(parts[1])
    return graph


def _cycles(graph: dict[str, set[str]]) -> list[tuple[str, ...]]:
    found: set[tuple[str, ...]] = set()

    def visit(node: str, path: tuple[str, ...]) -> None:
        if node in path:
            cycle = path[path.index(node) :] + (node,)
            rotations = [cycle[index:-1] + cycle[: index + 1] for index in range(len(cycle) - 1)]
            found.add(min(rotations))
            return
        for neighbour in graph[node]:
            visit(neighbour, (*path, node))

    for area in graph:
        visit(area, ())
    return sorted(found)


def violations() -> list[str]:
    failures: list[str] = []
    graph = architecture_graph()
    for dependency in sorted(graph["core"] - {"core"}):
        failures.append(f"core imports forbidden top-level area {dependency}")
    if "adapters" in graph["application"]:
        failures.append("application imports concrete adapters")

    for path in (PACKAGE_ROOT / "core" / "sas").rglob("*.py"):
        for imported in sorted(_imports(path)):
            if imported.startswith(SAS_CORE_FORBIDDEN_IMPORTS):
                relative = path.relative_to(ROOT)
                failures.append(f"{relative} imports forbidden SAS-core dependency {imported}")

    adapter_edges: defaultdict[str, set[str]] = defaultdict(set)
    for path in (PACKAGE_ROOT / "adapters").rglob("*.py"):
        source_parts = _module_name(path).split(".")
        source_family = source_parts[2] if len(source_parts) > 2 else None
        for imported in _absolute_imports(path):
            parts = imported.split(".")
            if len(parts) > 2 and parts[1] == "adapters":
                target_family = parts[2]
                if source_family and target_family != source_family:
                    adapter_edges[source_family].add(target_family)
    for source, targets in sorted(adapter_edges.items()):
        for target in sorted(targets):
            failures.append(f"adapter family {source} imports adapter family {target}")

    for cycle in _cycles(graph):
        failures.append(f"top-level import cycle: {' -> '.join(cycle)}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the graph as JSON")
    args = parser.parse_args()

    graph = architecture_graph()
    failures = violations()
    if args.json:
        print(json.dumps({area: sorted(edges) for area, edges in sorted(graph.items())}))
    if failures:
        for failure in failures:
            print(f"architecture violation: {failure}")
        return 1
    if not args.json:
        edge_count = sum(len(edges) for edges in graph.values())
        print(f"v2 architecture check passed: {len(graph)} areas, {edge_count} edges, no cycles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
