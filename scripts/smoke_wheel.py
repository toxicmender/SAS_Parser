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
    "sas_migrate.application",
    "sas_migrate.application.knowledge",
    "sas_migrate.application.memory",
    "sas_migrate.application.assessment",
    "sas_migrate.application.translation",
    "sas_migrate.application.validation",
    "sas_migrate.application.xref",
    "sas_migrate.adapters.memory",
    "sas_migrate.adapters.assessment",
    "sas_migrate.adapters.validation",
    "sas_migrate.adapters.xref",
    "sas_migrate.core",
    "sas_migrate.core.responses",
    "sas_migrate.core.sas",
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
    "sas_migrate/resources/assessment/pyspark.json",
    "sas_migrate/resources/assessment/sparksql.json",
    "prompt_builder/instructions/_common/source_fidelity.md",
    "prompt_builder/instructions/pyspark/overview.md",
    "prompt_builder/instructions/sparksql/overview.md",
    "sas_migrate/resources/contracts/schema-v2.json",
    "sas_migrate/application/response_acceptance.py",
    "sas_migrate/application/translation/attempts.py",
    "sas_migrate/application/translation/artifacts.py",
    "sas_migrate/application/translation/budgeting.py",
    "sas_migrate/application/translation/models.py",
    "sas_migrate/application/translation/orchestration.py",
    "sas_migrate/application/translation/prompt_assembly.py",
    "sas_migrate/application/translation/prompting.py",
    "sas_migrate/application/translation/run_state.py",
    "sas_migrate/application/translation/token_accounting.py",
    "sas_migrate/application/translation/token_audit.py",
    "sas_migrate/core/responses/normalization.py",
    "sas_migrate/core/responses/validation.py",
    "sas_migrate/core/sas/chunking.py",
    "sas_migrate/core/sas/metadata/extraction.py",
    "sas_migrate/core/sas/dependencies/discovery.py",
    "sas_migrate/core/tokens/audit.py",
    "sas_migrate/core/tokens/counting.py",
    "sas_migrate/application/ports/run_events.py",
    "sas_migrate/application/ports/knowledge.py",
    "sas_migrate/application/ports/conversation_memory.py",
    "sas_migrate/application/knowledge/ingestion.py",
    "sas_migrate/application/knowledge/retrieval.py",
    "sas_migrate/application/memory/services.py",
    "sas_migrate/application/assessment/service.py",
    "sas_migrate/application/assessment/profiles.py",
    "sas_migrate/application/validation/service.py",
    "sas_migrate/application/validation/judged.py",
    "sas_migrate/application/validation/reporting.py",
    "sas_migrate/application/xref/service.py",
    "sas_migrate/application/xref/sas_rewriter.py",
    "sas_migrate/application/xref/target_rewriters/pyspark.py",
    "sas_migrate/application/xref/target_rewriters/sql.py",
    "sas_migrate/adapters/knowledge/pdf.py",
    "sas_migrate/adapters/memory/in_memory.py",
    "sas_migrate/adapters/memory/delta.py",
    "sas_migrate/adapters/memory/delta_operations.py",
    "sas_migrate/adapters/assessment/profiles.py",
    "sas_migrate/adapters/assessment/pdf.py",
    "sas_migrate/adapters/validation/tracking.py",
    "sas_migrate/adapters/validation/pdf.py",
    "sas_migrate/adapters/xref/csv.py",
    "sas_migrate/adapters/xref/sharepoint.py",
)


def _is_inside_repository(path: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT)
    except ValueError:
        return False
    return True


def main() -> int:
    distribution = importlib.metadata.distribution("sas-parser")
    installed_files: set[str] = {
        str(path).replace("\\", "/") for path in distribution.files or ()
    }

    required_files: set[str] = set(REQUIRED_DISTRIBUTION_FILES)
    missing_files = sorted(required_files - installed_files)
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
