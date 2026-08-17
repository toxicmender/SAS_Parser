"""Verify an installed SAS Parser wheel without importing the source tree.

Run this script with the Python interpreter from a clean environment containing
the built wheel.  It intentionally checks distribution metadata and module
origins in addition to importing modules: a source checkout can otherwise hide
missing packages and package data.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import pathlib
import subprocess
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
REQUIRED_MODULES = (
    "main",
    "sas_migrate",
    "sas_migrate.core",
    "pipeline",
    "prompt_builder",
    "complexity",
    "token_budget",
    "validation",
)
REQUIRED_DISTRIBUTION_FILES = (
    "main.py",
    "token_budget/__init__.py",
    "validation/__init__.py",
    "complexity/profiles/pyspark.json",
    "complexity/profiles/sparksql.json",
    "prompt_builder/instructions/_common/source_fidelity.md",
    "prompt_builder/instructions/pyspark/overview.md",
    "prompt_builder/instructions/sparksql/overview.md",
    "sas_migrate/resources/contracts/schema-v2.json",
)


def _is_inside_repository(path: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT)
    except ValueError:
        return False
    return True


def main() -> int:
    distribution = importlib.metadata.distribution("sas-parser")
    installed_files = {str(path).replace("\\", "/") for path in distribution.files or ()}

    missing_files = sorted(set(REQUIRED_DISTRIBUTION_FILES) - installed_files)
    if missing_files:
        raise RuntimeError(f"wheel is missing runtime files: {missing_files}")

    entry_points = {
        entry.name: entry
        for entry in distribution.entry_points
        if entry.group == "console_scripts"
    }
    entry_point = entry_points.get("sas-parser")
    if entry_point is None:
        raise RuntimeError("wheel does not define the sas-parser console script")
    if not callable(entry_point.load()):
        raise TypeError("sas-parser console entry point is not callable")

    for module_name in REQUIRED_MODULES:
        module = importlib.import_module(module_name)
        origin = getattr(module, "__file__", None)
        if origin is None:
            raise RuntimeError(f"{module_name} has no import origin")
        if _is_inside_repository(pathlib.Path(origin)):
            raise RuntimeError(
                f"{module_name} imported from the source checkout instead of the wheel: {origin}"
            )

    help_result = subprocess.run(
        [sys.executable, "-m", "main", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    if help_result.returncode != 0:
        raise RuntimeError(
            "installed CLI failed: "
            f"stdout={help_result.stdout!r}, stderr={help_result.stderr!r}"
        )
    if "sas-parser" not in help_result.stdout.lower():
        raise RuntimeError("installed CLI help did not identify sas-parser")

    v2_help_result = subprocess.run(
        [sys.executable, "-m", "sas_migrate.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    if v2_help_result.returncode != 0:
        raise RuntimeError(
            "installed v2 CLI failed: "
            f"stdout={v2_help_result.stdout!r}, stderr={v2_help_result.stderr!r}"
        )
    if "sas-migrate" not in v2_help_result.stdout.lower():
        raise RuntimeError("installed v2 CLI help did not identify sas-migrate")

    print(
        f"installed-wheel smoke passed for sas-parser {distribution.version} "
        f"({len(installed_files)} files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
