"""Verify an installed SAS Parser wheel without importing the source tree.

Run this script with the Python interpreter from a clean environment containing
the built wheel.  It intentionally checks distribution metadata and module
origins in addition to importing modules: a source checkout can otherwise hide
missing packages and package data.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import pathlib
import subprocess
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
REQUIRED_MODULES = (
    "main",
    "sas_migrate",
    "sas_migrate.application",
    "sas_migrate.application.conversion",
    "sas_migrate.application.hydration",
    "sas_migrate.application.deployment",
    "sas_migrate.application.knowledge",
    "sas_migrate.application.memory",
    "sas_migrate.application.assessment",
    "sas_migrate.application.translation",
    "sas_migrate.application.validation",
    "sas_migrate.application.xref",
    "sas_migrate.adapters.memory",
    "sas_migrate.adapters.conversion",
    "sas_migrate.adapters.hydration",
    "sas_migrate.adapters.assessment",
    "sas_migrate.adapters.auth",
    "sas_migrate.adapters.credentials",
    "sas_migrate.adapters.sharepoint",
    "sas_migrate.adapters.validation",
    "sas_migrate.adapters.xref",
    "sas_migrate.core",
    "sas_migrate.core.responses",
    "sas_migrate.core.sas",
    "sas_migrate.config",
    "sas_migrate.observability",
    "sas_migrate.cli",
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
    "sas_migrate/application/conversion/models.py",
    "sas_migrate/application/conversion/service.py",
    "sas_migrate/application/ports/conversion.py",
    "sas_migrate/adapters/conversion/local.py",
    "sas_migrate/adapters/conversion/sharepoint.py",
    "sas_migrate/application/hydration/models.py",
    "sas_migrate/application/hydration/planner.py",
    "sas_migrate/application/hydration/service.py",
    "sas_migrate/application/ports/hydration.py",
    "sas_migrate/adapters/hydration/drivers.py",
    "sas_migrate/adapters/hydration/delta.py",
    "sas_migrate/adapters/hydration/ranged_io.py",
    "sas_migrate/application/deployment.py",
    "sas_migrate/cli/__init__.py",
    "sas_migrate/cli/__main__.py",
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
    "sas_migrate/adapters/auth/azure.py",
    "sas_migrate/adapters/credentials/chain.py",
    "sas_migrate/adapters/credentials/databricks.py",
    "sas_migrate/adapters/credentials/environment.py",
    "sas_migrate/adapters/credentials/vault.py",
    "sas_migrate/adapters/sharepoint/graph.py",
    "sas_migrate/adapters/sharepoint/preflight.py",
    "sas_migrate/adapters/sharepoint/worker.py",
    "sas_migrate/application/ports/access_token.py",
    "sas_migrate/config/loader.py",
    "sas_migrate/config/models.py",
    "sas_migrate/observability/logging.py",
    "sas_migrate/observability/redaction.py",
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
    expected_entry_points = {"sas-parser", "sas-migrate"}
    missing_entry_points = sorted(expected_entry_points - entry_points.keys())
    if missing_entry_points:
        raise RuntimeError(
            f"wheel does not define console scripts: {missing_entry_points}"
        )
    for name in sorted(expected_entry_points):
        if not callable(entry_points[name].load()):
            raise TypeError(f"{name} console entry point is not callable")

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

    v2_smoke_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sas_migrate.cli",
            "smoke",
            "--require-wheel",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if v2_smoke_result.returncode != 0:
        raise RuntimeError(
            "installed v2 deployment smoke failed: "
            f"stdout={v2_smoke_result.stdout!r}, stderr={v2_smoke_result.stderr!r}"
        )
    smoke_report = json.loads(v2_smoke_result.stdout)
    if not smoke_report.get("passed"):
        raise RuntimeError("installed v2 deployment smoke reported a failure")
    if smoke_report.get("sqlglot_dialect") != "databricks":
        raise RuntimeError("installed v2 deployment smoke used the wrong SQL dialect")

    print(
        f"installed-wheel smoke passed for sas-parser {distribution.version} "
        f"({len(installed_files)} files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
