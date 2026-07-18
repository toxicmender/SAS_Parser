#!/usr/bin/env python3
"""Per-file pyright ratchet: fail CI when an existing file gains type errors.

The repo's shipped code is pyright-clean today, but a hard "0 errors" gate
would also block any *new* module that isn't clean yet. This ratchet instead
locks in the files that exist now (recorded in the baseline with their current
error counts, all 0) and:

* fails when a file already in the baseline reports MORE errors than recorded;
* ignores files absent from the baseline (newly added modules) — they are not
  gated, matching the "don't block new files/modules" policy;
* reports files that improved, so the baseline can be tightened with --update.

It consumes ``pyright --outputjson`` (pyright only lists files that have
diagnostics, so the set of existing files is enumerated separately from the
[tool.pyright] include list in pyproject.toml).

Usage:
    pyright --outputjson > pyright.json
    python scripts/pyright_ratchet.py --input pyright.json            # check
    python scripts/pyright_ratchet.py --input pyright.json --update   # rewrite baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "scripts" / "pyright_baseline.json"


def _source_files() -> set[str]:
    """Repo-relative .py paths covered by [tool.pyright].include, POSIX-style."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    include = pyproject["tool"]["pyright"]["include"]
    files: set[str] = set()
    for entry in include:
        target = REPO_ROOT / entry
        if target.is_dir():
            for path in target.rglob("*.py"):
                if "__pycache__" in path.parts or "build" in path.parts:
                    continue
                files.add(path.relative_to(REPO_ROOT).as_posix())
        elif target.is_file():
            files.add(target.relative_to(REPO_ROOT).as_posix())
    return files


def _error_counts(pyright_json: Path) -> dict[str, int]:
    """Repo-relative file -> number of error-severity diagnostics."""
    data = json.loads(pyright_json.read_text("utf-8"))
    counts: dict[str, int] = {}
    for diag in data.get("generalDiagnostics", []):
        if diag.get("severity") != "error":
            continue
        rel = Path(diag["file"]).resolve().relative_to(REPO_ROOT).as_posix()
        counts[rel] = counts.get(rel, 0) + 1
    return counts


def _load_baseline() -> dict[str, int]:
    if not BASELINE_PATH.exists():
        return {}
    return json.loads(BASELINE_PATH.read_text("utf-8"))["files"]


def _write_baseline(counts: dict[str, int], files: set[str]) -> None:
    # Every existing source file is recorded (0 when clean) so its presence,
    # not just a non-zero count, marks it as "already gated".
    payload = {
        "_comment": (
            "Per-file pyright error ceilings. Regenerate with: "
            "python scripts/pyright_ratchet.py --input pyright.json --update"
        ),
        "files": {name: counts.get(name, 0) for name in sorted(files)},
    }
    BASELINE_PATH.write_text(json.dumps(payload, indent=2) + "\n", "utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="pyright --outputjson report to evaluate",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite the baseline from --input instead of checking against it",
    )
    args = parser.parse_args()

    counts = _error_counts(args.input)
    source_files = _source_files()

    if args.update:
        _write_baseline(counts, source_files)
        total = sum(counts.get(f, 0) for f in source_files)
        print(f"Wrote {BASELINE_PATH.relative_to(REPO_ROOT)}: "
              f"{len(source_files)} files, {total} baseline error(s).")
        return 0

    baseline = _load_baseline()
    if not baseline:
        print("No baseline found; run with --update first.", file=sys.stderr)
        return 2

    regressions: list[tuple[str, int, int]] = []
    new_file_errors: list[tuple[str, int]] = []
    improvements: list[tuple[str, int, int]] = []

    for file, current in sorted(counts.items()):
        if file not in baseline:
            # Not gated: a module added after the baseline was taken.
            new_file_errors.append((file, current))
        elif current > baseline[file]:
            regressions.append((file, baseline[file], current))

    for file, allowed in baseline.items():
        current = counts.get(file, 0)
        if current < allowed:
            improvements.append((file, allowed, current))

    if new_file_errors:
        print("Type errors in new (un-gated) files - not blocking, please clean up:")
        for file, current in new_file_errors:
            print(f"  {file}: {current} error(s)")
        print()

    if improvements:
        print("Files now below their baseline — tighten with --update:")
        for file, allowed, current in improvements:
            print(f"  {file}: {allowed} -> {current}")
        print()

    if regressions:
        print("Pyright ratchet FAILED - existing files gained type errors:")
        for file, allowed, current in regressions:
            print(f"  {file}: {allowed} -> {current} (+{current - allowed})")
        print("\nFix the new errors, or if intentional, run:")
        print("  python scripts/pyright_ratchet.py --input pyright.json --update")
        return 1

    print("Pyright ratchet passed: no existing file exceeded its baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
