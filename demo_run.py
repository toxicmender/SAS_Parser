"""Demo: run the SAS -> target pipeline over a directory of local .sas files.

End-to-end wiring the pipeline is built for:

    reference_docs/*.pdf ── PromptBuilder.from_reference_dir ─┐
                                                              ├─> SasLLMPipeline
    <sas_dir>/**/*.sas ───────── discovered here ────────────┘
                                                              │
                                    run_files() ── MultiFileBatcher ── LLM

`SasLLMPipeline.run_files` chunks every file, batches the whole corpus with
`MultiFileBatcher` (so cross-file dataset-flow / macro edges are resolved into
shared batches), and feeds every batch + singleton through the LLM on one
thread. Per-item reference guidance is retrieved from the `reference_docs`
corpus and injected ephemerally.

Usage
-----
    # needs ANTHROPIC_API_KEY and the `anthropic` extra installed:
    #   uv pip install -e ".[anthropic]"
    python demo_run.py path/to/sas_dir
    python demo_run.py path/to/sas_dir --model claude-haiku-4-5-20251001 --debug

Run from the repo root so the default ``reference_docs`` path resolves.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from chunker import SasLLMPipeline
from prompt_builder import PromptBuilder

logger = logging.getLogger("demo_run")


def _discover_sas_files(sas_dir: Path, pattern: str) -> list[str]:
    """Recursively find .sas files under *sas_dir*, sorted for deterministic order.

    File order establishes the default execution sequence MultiFileBatcher uses
    to resolve cross-file producer/consumer tie-breaks, so a stable sort keeps
    batching reproducible across runs.
    """
    paths = sorted(sas_dir.rglob(pattern))
    return [str(p) for p in paths]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "sas_dir",
        type=Path,
        help="Directory containing local .sas files (searched recursively).",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("reference_docs"),
        help="Directory of reference PDFs for instruction chunking "
        "(default: ./reference_docs).",
    )
    parser.add_argument(
        "--pattern",
        default="*.sas",
        help="Glob for SAS files within sas_dir (default: *.sas).",
    )
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="LangChain chat-model string (default: claude-haiku-4-5-20251001).",
    )
    parser.add_argument(
        "--output-language",
        default="PySpark",
        help="Target language named in the system prompt (default: PySpark).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="If set, write each item's LLM response to a file under this dir.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging for the whole pipeline.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if not args.sas_dir.is_dir():
        logger.error(f"sas_dir is not a directory: {args.sas_dir}")
        return 2
    if not args.reference_dir.is_dir():
        logger.error(f"reference_dir is not a directory: {args.reference_dir}")
        return 2

    sas_files = _discover_sas_files(args.sas_dir, args.pattern)
    if not sas_files:
        logger.error(f"no files matching {args.pattern!r} under {args.sas_dir}")
        return 1
    logger.info(f"discovered {len(sas_files)} SAS file(s) under {args.sas_dir}")
    for path in sas_files:
        logger.info(f"  - {path}")

    # Load + chunk + index the reference corpus once (cached on disk after the
    # first run). This is the "document chunking for instructions" half.
    logger.info(f"building instruction corpus from {args.reference_dir}")
    builder = PromptBuilder.from_reference_dir(str(args.reference_dir))

    # In-memory message store (delta_table=None) — no Spark/JVM is booted.
    pipeline = SasLLMPipeline(
        model=args.model,
        output_language=args.output_language,
        prompt_builder=builder,
    )

    # run_files chunks every file and batches the corpus via MultiFileBatcher,
    # then runs every batch/singleton through the LLM on one shared thread.
    logger.info(f"running pipeline over {len(sas_files)} file(s) with model={args.model}")
    outputs = pipeline.run_files(sas_files)

    logger.info(f"pipeline produced {len(outputs)} item response(s)")
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for out in outputs:
        header = (
            f"=== {out['item_id']} "
            f"({'batch' if out['is_batch'] else out['kind']}) "
            f"files={out['source_files']} ==="
        )
        print(f"\n{header}")
        print(out["response"])

        if args.out_dir:
            dest = args.out_dir / f"{out['item_id']}.txt"
            dest.write_text(
                f"{header}\n\n{out['response']}\n", encoding="utf-8"
            )
            logger.debug(f"wrote {dest}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
