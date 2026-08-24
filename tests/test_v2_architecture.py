"""V2 import graph and migrated CLI-command gates."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_v2_architecture_graph_is_acyclic_and_obeys_boundaries() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_v2_architecture.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no cycles" in result.stdout

    graph_result = subprocess.run(
        [sys.executable, "scripts/check_v2_architecture.py", "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert graph_result.returncode == 0, graph_result.stdout + graph_result.stderr
    graph = json.loads(graph_result.stdout)
    assert graph["core"] == []
    assert graph["application"] == ["core"]


def test_v2_cli_exposes_smoke_but_not_unmigrated_commands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "sas_migrate.cli", "--help"],
        cwd=SRC,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "smoke" in result.stdout
    assert "convert" not in result.stdout
