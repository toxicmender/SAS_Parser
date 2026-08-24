"""Deployment-level contracts for the wheel-only v2 runtime."""

from __future__ import annotations

import json
from pathlib import Path

from sas_migrate.application import deployment
from sas_migrate.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_offline_deployment_flow_crosses_v2_boundaries() -> None:
    report = deployment.run_deployment_smoke()

    assert report.schema_version == 2
    assert report.passed
    assert report.target.value == "spark_sql"
    assert report.sqlglot_dialect == "databricks"
    assert {check.name for check in report.checks} == {
        "installed_distribution",
        "runtime_identity",
        "packaged_schema",
        "v2_application_flow",
    }
    flow = next(check for check in report.checks if check.name == "v2_application_flow")
    assert "validation score=1.000" in flow.details
    assert "tokens=36" in flow.details
    assert deployment.DeploymentSmokeReport.from_json(report.to_json()) == report


def test_deployment_requirements_reject_editable_root_runtime(monkeypatch) -> None:
    monkeypatch.setattr(deployment, "_installation", lambda: ("0.1.0", "editable"))
    monkeypatch.setattr(deployment, "_runtime_identity", lambda: (False, "uid=0"))

    report = deployment.run_deployment_smoke(
        require_wheel=True,
        require_non_root=True,
    )

    assert not report.passed
    failed = {check.name for check in report.checks if not check.passed}
    assert failed == {"installed_distribution", "runtime_identity"}


def test_cli_emits_versioned_json_report(capsys) -> None:
    assert main(["smoke", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema_version"] == 2
    assert payload["passed"] is True
    assert payload["sqlglot_dialect"] == "databricks"


def test_v2_deployment_image_is_wheel_only_non_root_and_ci_gated() -> None:
    dockerfile = (ROOT / "docker" / "v2.Dockerfile").read_text("utf-8")
    dockerignore = (ROOT / "docker" / "v2.Dockerfile.dockerignore").read_text("utf-8")
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8")

    assert "ARG UV_VERSION=0.9.21" in dockerfile
    assert "ghcr.io/astral-sh/uv:latest" not in dockerfile
    assert "uv sync --locked --no-install-project" in dockerfile
    assert "uv build --wheel" in dockerfile
    assert "uv pip install --python /opt/venv --no-deps" in dockerfile
    assert "COPY --from=builder /opt/venv /opt/venv" in dockerfile
    runtime = dockerfile.split(" AS runtime", maxsplit=1)[1]
    assert "COPY . ." not in runtime
    assert "USER 10001:10001" in runtime
    assert "--require-wheel" in dockerfile
    assert "--require-non-root" in dockerfile
    assert dockerignore.splitlines()[0] == "**"
    assert "!src/**" in dockerignore
    assert "file: docker/v2.Dockerfile" in workflow
    assert "name: V2 deployment smoke" in workflow
