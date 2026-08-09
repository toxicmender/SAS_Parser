"""Regression tests for the CI type-check ratchet policy."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "pyright_ratchet.py"


def _load_ratchet() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pyright_ratchet_under_test", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_fixture(
    module: ModuleType,
    tmp_path: Path,
    monkeypatch,
    diagnostics: list[dict[str, object]],
) -> None:
    (tmp_path / "existing.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "new_module.py").write_text("value = 2\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pyright]\ninclude = [\"existing.py\", \"new_module.py\"]\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"files": {"existing.py": 0}}), encoding="utf-8")
    report = tmp_path / "pyright.json"
    report.write_text(json.dumps({"generalDiagnostics": diagnostics}), encoding="utf-8")

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "BASELINE_PATH", baseline)
    monkeypatch.setattr(sys, "argv", ["pyright_ratchet.py", "--input", str(report)])


def test_ratchet_allows_errors_in_new_modules(tmp_path: Path, monkeypatch) -> None:
    """New source files are reported but must not block the CI ratchet."""
    module = _load_ratchet()
    _configure_fixture(
        module,
        tmp_path,
        monkeypatch,
        [{"severity": "error", "file": str(tmp_path / "new_module.py")}],
    )

    assert module.main() == 0


def test_ratchet_blocks_errors_in_baselined_modules(tmp_path: Path, monkeypatch) -> None:
    """Existing source files remain a blocking type-quality contract."""
    module = _load_ratchet()
    _configure_fixture(
        module,
        tmp_path,
        monkeypatch,
        [{"severity": "error", "file": str(tmp_path / "existing.py")}],
    )

    assert module.main() == 1


def test_ci_workflow_defers_the_gate_to_the_ratchet() -> None:
    """Keep GitHub Actions from stopping before the policy can be evaluated."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8")

    pyright_step = workflow.split("      - name: Pyright\n", 1)[1].split(
        "      - name: Ratchet against baseline\n", 1
    )[0]
    ratchet_step = workflow.split("      - name: Ratchet against baseline\n", 1)[1].split(
        "      # Advisory:", 1
    )[0]

    assert "continue-on-error: true" in pyright_step
    assert "if: always()" in ratchet_step
