"""Contract tests for the consolidated gap and legacy inventory."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs" / "plans" / "v2-gap-legacy-inventory.json"
CHECKER = ROOT / "scripts" / "check_v2_legacy_inventory.py"


def test_inventory_checker_passes_for_the_committed_tree() -> None:
    result = subprocess.run(
        [sys.executable, str(CHECKER)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_inventory_has_complete_ownership_and_exit_gates() -> None:
    inventory = json.loads(INVENTORY.read_text("utf-8"))
    assert inventory["schema_version"] == 1
    assert len(inventory["legacy_packages"]) == 14
    assert sum(entry["python_files"] for entry in inventory["legacy_packages"]) == 118
    assert sum(entry["tracked_files"] for entry in inventory["legacy_packages"]) == 168
    assert {entry["owner_phase"] for entry in inventory["legacy_packages"]} <= {
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    }
    assert all(gap["status"] == "open" for gap in inventory["gaps"])
    assert all(gap["exit_gate"] for gap in inventory["gaps"])
    assert importlib.util.find_spec("sas_migrate") is not None
