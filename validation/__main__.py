"""CLI: run the validation suite over JSON case files or an existing thread.

Usage
-----
    # deterministic metrics against a live model
    # (needs ANTHROPIC_API_KEY and the `anthropic` extra):
    python -m validation validation/cases --model claude-sonnet-4-5

    # additionally grade each translation with an LLM judge:
    python -m validation validation/cases --judge-model claude-sonnet-4-5

    # append the run to the local Spark-parquet history (./validation_runs):
    python -m validation validation/cases --track

    # or straight into a Databricks Delta table:
    python -m validation validation/cases --track --table main.qa.validation_runs

    # post-hoc: score a conversation thread already in a Delta-backed
    # memory store, without re-running the pipeline:
    python -m validation --thread run::job1.sas --delta-table main.ml.memory

Exit code 0 when every case (or the thread) passes, 1 otherwise — so the
command gates CI.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from chunker import SasLLMPipeline
from llm_client import LLMClient, LLMClientConfig

from .conversation import validate_thread
from .dataset import load_cases
from .judge import LLMJudgeMetric
from .metrics import ValidationMetric, default_metrics
from .models import ValidationReport
from .runner import ValidationRunner
from .tracking import DEFAULT_PATH, log_report

logger = logging.getLogger("validation.__main__")


def _validate_thread(
    args: argparse.Namespace, metrics: list[ValidationMetric]
) -> ValidationReport:
    """Post-hoc mode: score one existing thread from a Delta-backed store."""
    from memory.store import MemoryHub
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("validation_thread")
        .getOrCreate()
    )
    hub = MemoryHub(spark=spark, table=args.delta_table)
    result = validate_thread(hub, args.thread, metrics=metrics)
    # The pipeline model that produced the thread is not recorded in the
    # store, so the report is labelled as post-hoc rather than guessing.
    return ValidationReport(model="post-hoc", results=[result])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # __doc__ is None under `python -OO`, which strips docstrings.
    parser = argparse.ArgumentParser(
        prog="python -m validation",
        description=__doc__.splitlines()[0] if __doc__ else None,
    )
    parser.add_argument(
        "cases",
        type=Path,
        nargs="?",
        default=None,
        help="Directory of *.json case files (or a single case file).",
    )
    parser.add_argument(
        "--thread",
        default=None,
        help="Post-hoc mode: validate this existing thread id instead of "
        "running cases (requires --delta-table).",
    )
    parser.add_argument(
        "--delta-table",
        default=None,
        help="Delta table backing the memory store the thread lives in, "
        "e.g. main.ml.langchain_memory (post-hoc mode only).",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-5",
        help="LangChain chat-model string for the pipeline under test.",
    )
    parser.add_argument(
        "--output-language",
        default="PySpark",
        help="Target language named in the system prompt (default: PySpark).",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="If set, also grade each translation with this judge model.",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Append the report to the Spark-backed run history.",
    )
    parser.add_argument(
        "--table",
        default=None,
        help="Spark table target, e.g. catalog.schema.validation_runs "
        "(default: config.json validation.table).",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Parquet-directory target (default: config.json validation.path, "
        f"then ./{DEFAULT_PATH}).",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable DEBUG logging."
    )
    args = parser.parse_args(argv)
    if (args.cases is None) == (args.thread is None):
        parser.error("give exactly one of: a cases path, or --thread")
    if args.thread is not None and args.delta_table is None:
        parser.error(
            "--thread needs --delta-table (an in-memory store from a past "
            "run is gone; post-hoc validation reads a persistent store)"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    metrics = default_metrics()
    if args.judge_model:
        logger.info(f"main: adding LLM judge  model={args.judge_model}")
        metrics.append(
            LLMJudgeMetric(
                llm=LLMClient(LLMClientConfig(model=args.judge_model)),
                output_language=args.output_language,
            )
        )

    if args.thread is not None:
        report = _validate_thread(args, metrics)
    else:
        cases = load_cases(args.cases)
        pipeline = SasLLMPipeline(
            model=args.model, output_language=args.output_language
        )
        report = ValidationRunner(pipeline, metrics=metrics).run(cases)
    print(report.to_markdown())

    if args.track:
        run_id = log_report(report, table=args.table, path=args.path)
        print(f"logged validation run: {run_id}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
