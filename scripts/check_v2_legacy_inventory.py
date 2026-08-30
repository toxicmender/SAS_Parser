"""Verify the authoritative v2 gap and legacy inventory matches the tree."""

from __future__ import annotations

import ast
import json
import pathlib
import subprocess
import tomllib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "docs" / "plans" / "v2-gap-legacy-inventory.json"
V2_ROOT = ROOT / "src" / "sas_migrate"


def _load_inventory() -> dict[str, Any]:
    return json.loads(INVENTORY_PATH.read_text("utf-8"))


def _legacy_imports(packages: set[str]) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for path in V2_ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text("utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules = (node.module,)
            else:
                continue
            for module in modules:
                if module.split(".", 1)[0] in packages:
                    found.add((relative, module))
    return found


def violations() -> list[str]:
    inventory = _load_inventory()
    failures: list[str] = []
    entries = inventory["legacy_packages"]
    declared = {entry["name"] for entry in entries}
    actual = {
        path.name
        for path in ROOT.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }
    if declared != actual:
        failures.append(
            "legacy package inventory drift: "
            f"missing={sorted(actual - declared)}, stale={sorted(declared - actual)}"
        )

    for entry in entries:
        package = ROOT / entry["name"]
        count = sum(1 for path in package.rglob("*.py") if path.is_file())
        if count != entry["python_files"]:
            failures.append(
                f"{entry['name']} Python file count is {count}, "
                f"inventory says {entry['python_files']}"
            )
        tracked = subprocess.run(
            ["git", "ls-files", f"{entry['name']}/**"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if len(tracked) != entry["tracked_files"]:
            failures.append(
                f"{entry['name']} tracked file count is {len(tracked)}, "
                f"inventory says {entry['tracked_files']}"
            )

    expected_imports = {
        (entry["file"], entry["module"])
        for entry in inventory["allowed_v2_legacy_imports"]
    }
    actual_imports = _legacy_imports(declared)
    if expected_imports != actual_imports:
        failures.append(
            "v2 legacy import allowlist drift: "
            f"new={sorted(actual_imports - expected_imports)}, "
            f"stale={sorted(expected_imports - actual_imports)}"
        )

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    packaged = set(pyproject["tool"]["setuptools"]["packages"]["find"]["include"])
    coverage = set(pyproject["tool"]["coverage"]["run"]["source"])
    typed = set(pyproject["tool"]["pyright"]["include"])
    for package in sorted(declared):
        if f"{package}*" not in packaged:
            failures.append(f"legacy package {package} is shipped but not explicit in packaging")
        if package not in coverage:
            failures.append(f"legacy package {package} is absent from coverage sources")
        if package not in typed:
            failures.append(f"legacy package {package} is absent from Pyright inputs")

    reference_paths = {
        value
        for values in inventory["reference_surfaces"].values()
        for value in values
    }
    for value in sorted(reference_paths):
        if not (ROOT / value).exists():
            failures.append(f"declared legacy reference surface does not exist: {value}")

    gaps = inventory["gaps"]
    gap_ids = [gap["id"] for gap in gaps]
    if len(gap_ids) != len(set(gap_ids)):
        failures.append("gap ids are not unique")
    for gap in gaps:
        missing = {
            field
            for field in ("id", "area", "owner_phase", "status", "summary", "exit_gate")
            if not gap.get(field)
        }
        if missing:
            failures.append(f"gap {gap.get('id', '<unknown>')} lacks {sorted(missing)}")

    return failures


def main() -> int:
    inventory = _load_inventory()
    failures = violations()
    if failures:
        for failure in failures:
            print(f"legacy inventory violation: {failure}")
        return 1
    print(
        "v2 legacy inventory passed: "
        f"{len(inventory['legacy_packages'])} packages, "
        f"{len(inventory['allowed_v2_legacy_imports'])} allowed v2 import, "
        f"{sum(gap['status'] == 'open' for gap in inventory['gaps'])} open gaps, "
        f"{sum(gap['status'] == 'closed' for gap in inventory['gaps'])} closed gap(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
