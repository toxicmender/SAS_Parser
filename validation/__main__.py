"""CLI: run the validation suite over a directory of JSON case files.

Usage
-----
    # deterministic metrics against a live model
    # (needs ANTHROPIC_API_KEY and the `anthropic` extra):
    python -m validation validation/cases --model claude-haiku-4-5-20251001

    # additionally grade each translation with an LLM judge:
    python -m validation validation/cases --judge-model claude-haiku-4-5-20251001

    # append the run to the local Spark-parquet history (./validation_runs):
    python -m validation validation/cases --track

    # or straight into a Databricks Delta table:
    python -m validation validation/cases --track --table main.qa.validation_runs

Exit code 0 when every case passes, 1 otherwise — so the command gates CI.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from chunker import SasLLMPipeline
from llm_client import LLMClient, LLMClientConfig

from .dataset import load_cases
from .judge import LLMJudgeMetric
from .metrics import default_metrics
from .runner import ValidationRunner
from .tracking import DEFAULT_PATH, log_report

logger = logging.getLogger("validation.__main__")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m validation", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "cases",
        type=Path,
        help="Directory of *.json case files (or a single case file).",
    )
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    cases = load_cases(args.cases)
    pipeline = SasLLMPipeline(
        model=args.model, output_language=args.output_language
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

    report = ValidationRunner(pipeline, metrics=metrics).run(cases)
    print(report.to_markdown())

    if args.track:
        run_id = log_report(report, table=args.table, path=args.path)
        print(f"logged validation run: {run_id}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
