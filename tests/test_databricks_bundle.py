"""Static deployment contracts for the Databricks Lakeflow Job bundle.

These tests intentionally require no workspace credentials. A live bundle
validation still belongs in the deployment environment, where the existing
general-purpose cluster ID and workspace profile are available.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

import main as application_main
from app_config import databricks_check

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "databricks.yml"


def _bundle() -> dict:
    return yaml.safe_load(BUNDLE_PATH.read_text(encoding="utf-8"))


def test_bundle_uses_existing_general_purpose_compute_for_every_task() -> None:
    bundle = _bundle()
    tasks = bundle["resources"]["jobs"]["sas_parser"]["tasks"]

    assert tasks
    assert {task["existing_cluster_id"] for task in tasks} == {
        "${var.general_purpose_cluster_id}"
    }
    for task in tasks:
        assert "new_cluster" not in task
        assert "job_cluster_key" not in task
        assert "environment_key" not in task


def test_bundle_blocks_conversion_until_dbr_18_preflight_passes() -> None:
    bundle = _bundle()
    assert bundle["variables"]["databricks_runtime_family"]["default"] == "18"

    tasks = {
        task["task_key"]: task
        for task in bundle["resources"]["jobs"]["sas_parser"]["tasks"]
    }
    preflight = tasks["runtime_preflight"]["python_wheel_task"]
    assert preflight["entry_point"] == "databricks_preflight"
    assert preflight["parameters"] == [
        "--expected-runtime",
        "${var.databricks_runtime_family}",
        "--json",
    ]
    assert tasks["convert_pending_requests"]["depends_on"] == [
        {"task_key": "runtime_preflight"}
    ]


def test_bundle_builds_and_installs_the_wheel_with_dbr_18_requirements() -> None:
    bundle = _bundle()
    artifact = bundle["artifacts"]["default"]
    assert artifact == {
        "type": "whl",
        "build": "uv build --wheel",
        "dynamic_version": True,
    }

    tasks = bundle["resources"]["jobs"]["sas_parser"]["tasks"]
    expected_libraries = [
        {"whl": "./dist/*.whl"},
        {"requirements": "./databricks/requirements.txt"},
    ]
    assert all(task["libraries"] == expected_libraries for task in tasks)

    requirements = (ROOT / "databricks" / "requirements.txt").read_text("utf-8")
    assert "Databricks Runtime 18 LTS" in requirements
    assert "Runtime 19" not in requirements
    assert (ROOT / "databricks" / "constraints-dbr18.txt").is_file()
    assert not (ROOT / "databricks" / "constraints-dbr19.txt").exists()


def test_wheel_exposes_both_lakeflow_entry_points() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    entry_points = pyproject["project"]["entry-points"]["databricks_jobs"]
    assert entry_points == {
        "databricks_preflight": "app_config.databricks_check:lakeflow_main",
        "sas_parser": "main:lakeflow_main",
    }


def test_lakeflow_entry_points_propagate_nonzero_status(monkeypatch) -> None:
    monkeypatch.setattr(databricks_check, "main", lambda: 3)
    with pytest.raises(SystemExit, match="3"):
        databricks_check.lakeflow_main()

    monkeypatch.setattr(application_main, "main", lambda: 4)
    with pytest.raises(SystemExit, match="4"):
        application_main.lakeflow_main()


def test_lakeflow_entry_points_allow_success(monkeypatch) -> None:
    monkeypatch.setattr(databricks_check, "main", lambda: 0)
    monkeypatch.setattr(application_main, "main", lambda: 0)
    assert databricks_check.lakeflow_main() is None
    assert application_main.lakeflow_main() is None
