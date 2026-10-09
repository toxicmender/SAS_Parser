#!/usr/bin/env python3
"""Time SasSemanticChunker.chunk_text on synthetic corpora.

The performance gate of docs/plans/chunker-sas-coverage.md: a phase may slow
full chunking by at most 20%. Timings on a shared machine drift by about 15%
between sessions, so compare two commits by running this on both in the same
session (a ``git worktree`` for the older one), never against numbers recorded
earlier.

Usage:
    uv run python scripts/bench_chunker.py            # best of 3
    uv run python scripts/bench_chunker.py --runs 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from chunker import SasSemanticChunker  # noqa: E402  (needs REPO_ROOT on sys.path)


def corpora() -> dict[str, str]:
    """The workloads: a long ordinary job written with and without semicolons
    after its macro calls, and a long run of back-to-back calls."""
    steps = range(10_000)
    return {
        "30k statements, calls without ;": "".join(
            f"%let v{i} = lib.t{i};\n%pull(tbl=t{i}, out=o{i})\n"
            f"data o{i}b; set o{i}; x = {i}; run;\n"
            for i in steps
        ),
        "30k statements, calls with ;": "".join(
            f"%let v{i} = lib.t{i};\n%pull(tbl=t{i}, out=o{i});\n"
            f"data o{i}b; set o{i}; x = {i}; run;\n"
            for i in steps
        ),
        "5,000 back-to-back calls": "".join(f"%pull(tbl=t{i})\n" for i in range(5_000))
        + "data x; set y; run;\n",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Time the chunker on synthetic corpora.")
    parser.add_argument("--runs", type=int, default=3, help="best of N runs (default 3)")
    args = parser.parse_args()
    logging.disable(logging.WARNING)
    chunker = SasSemanticChunker(min_words=1, max_words=700, timeout=None)
    for label, source in corpora().items():
        runs: list[float] = []
        chunks = 0
        for _ in range(args.runs):
            start = time.perf_counter()
            chunks = len(chunker.chunk_text(source).chunks)
            runs.append(time.perf_counter() - start)
        print(
            f"{label:<32} {len(source) / 1024:5.0f} KB  best {min(runs):5.2f}s  "
            f"runs {' '.join(f'{r:.2f}' for r in runs)}  chunks {chunks}"
        )


if __name__ == "__main__":
    main()
