"""CLI: run the validation suite over JSON case files or an existing thread.

Usage
-----
    # deterministic metrics against a live model (needs OPENAI_API_KEY —
    # the gateway is OpenAI-compatible for every model it fronts):
    python -m validation validation/cases --model claude-sonnet-4-5

    # additionally grade each translation with an LLM judge:
    python -m validation validation/cases --judge-model claude-sonnet-4-5

    # the full LLM-judged suite (deepeval's metrics, implemented natively) —
    # or a named subset; costs ~15 judge calls per item plus 5 per run:
    python -m validation validation/cases --judge-model claude-sonnet-4-5 \
        --judge-metrics all
    python -m validation validation/cases --judge-model claude-sonnet-4-5 \
        --judge-metrics faithfulness,contextual_relevancy

    # append the run to the local Spark-parquet history (./validation_runs):
    python -m validation validation/cases --track

    # or straight into a Databricks Delta table:
    python -m validation validation/cases --track --table main.qa.validation_runs

    # render the report to a local PDF, and/or upload it to SharePoint:
    python -m validation validation/cases --pdf report.pdf
    python -m validation validation/cases --pdf-sharepoint Reports/Validation

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

from pipeline import SasLLMPipeline
from llm_client import LLMClient, LLMClientConfig, TokenUsage
from target_language import resolve_target_language

from .conversation import validate_thread
from .dataset import load_cases
from .metrics import (
    JUDGED_METRIC_NAMES,
    ValidationMetric,
    default_metrics,
    judged_metrics,
)
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
    # store, so the report is labelled as post-hoc rather than guessing —
    # and for the same reason there is no pipeline token figure here: the
    # translation was billed in some earlier run. Judging it, however,
    # happens now, and is billed now.
    judge_usage = _judge_token_usage(metrics)
    return ValidationReport(
        model="post-hoc",
        results=[result],
        judge_token_usage=judge_usage if judge_usage.calls else None,
    )


def _judge_token_usage(metrics: list[ValidationMetric]) -> TokenUsage:
    """Tokens billed by LLM-backed metrics, summed across *metrics*.

    Mirrors :attr:`~validation.runner.ValidationRunner.judge_token_usage` for
    the thread mode, which scores without a runner.
    """
    total = TokenUsage()
    for metric in metrics:
        usage = getattr(metric, "token_usage", None)
        if isinstance(usage, TokenUsage):
            total = total + usage
    return total


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
        default="gpt-5.4",
        help="Model id (as the AI Gateway names it) for the pipeline under test.",
    )
    parser.add_argument(
        "--output-language",
        default=None,
        help=(
            "Target language the pipeline under test translates into, and "
            "the one the deterministic metrics score its code against. Omit "
            "to use config.json pipeline.output_language, then the code "
            "default."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="If set, also grade each translation with this judge model.",
    )
    parser.add_argument(
        "--judge-metrics",
        default=None,
        metavar="NAMES",
        help="Comma-separated LLM-judged metrics to add (needs --judge-model), "
        "or 'all'. Known: " + ", ".join(JUDGED_METRIC_NAMES) + ". Omit to add "
        "only the llm_judge grade. The full suite costs roughly 15 judge calls "
        "per item plus 5 per run.",
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
        "--md",
        type=Path,
        default=None,
        help="Also write the report as Markdown to this local file path.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Also render the report to this local PDF file path.",
    )
    parser.add_argument(
        "--pdf-sharepoint",
        nargs="?",
        const="",
        default=None,
        metavar="DEST",
        help="Render the report to PDF and upload it to a SharePoint document "
        "library. DEST is a folder (a timestamped filename is appended) or an "
        "exact '*.pdf' path; omit DEST to use config.json "
        "validation.report_sharepoint_path (then the library root).",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable DEBUG logging."
    )
    args = parser.parse_args(argv)
    if (args.cases is None) == (args.thread is None):
        parser.error("give exactly one of: a cases path, or --thread")
    if args.judge_metrics is not None and args.judge_model is None:
        parser.error("--judge-metrics needs --judge-model (the model to judge with)")
    if args.judge_metrics not in (None, "all"):
        unknown = [
            name.strip()
            for name in args.judge_metrics.split(",")
            if name.strip() not in JUDGED_METRIC_NAMES
        ]
        if unknown:
            parser.error(
                f"unknown --judge-metrics: {', '.join(unknown)}; "
                f"known: {', '.join(JUDGED_METRIC_NAMES)}, or 'all'"
            )
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

    # One resolution for the whole invocation: the deterministic suite, the
    # judged suite, and the pipeline under test must all mean the same target.
    target = resolve_target_language(args.output_language)
    metrics = default_metrics(target)
    if args.judge_model:
        # No --judge-metrics keeps the historical meaning of --judge-model:
        # the single llm_judge grade, one call per item.
        include = (
            None
            if args.judge_metrics == "all"
            else [name.strip() for name in (args.judge_metrics or "llm_judge").split(",")]
        )
        logger.info(f"main: adding judged metrics  model={args.judge_model}")
        metrics.extend(
            judged_metrics(
                # The "validator" role overlay: the reference gives the
                # validator its own (much longer) gateway timeout, and
                # --judge-model still wins over whatever it configures.
                LLMClient(
                    LLMClientConfig.for_role("validator", model=args.judge_model)
                ),
                output_language=target.display_name,
                include=include,
            )
        )

    if args.thread is not None:
        report = _validate_thread(args, metrics)
    else:
        cases = load_cases(args.cases)
        pipeline = SasLLMPipeline(
            model=args.model, output_language=target.display_name
        )
        report = ValidationRunner(pipeline, metrics=metrics).run(cases)
    print(report.to_markdown())

    if args.track:
        run_id = log_report(report, table=args.table, path=args.path)
        print(f"logged validation run: {run_id}")

    if args.md is not None:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(report.to_markdown(), encoding="utf-8")
        print(f"wrote Markdown report: {args.md}")

    if args.pdf is not None:
        from .pdf import report_to_pdf

        args.pdf.write_bytes(report_to_pdf(report))
        print(f"wrote PDF report: {args.pdf}")

    if args.pdf_sharepoint is not None:
        from app_config.sharepoint import SharePointError
        from .pdf import publish_report_pdf

        try:
            item = publish_report_pdf(report, args.pdf_sharepoint or None)
            print(
                "uploaded PDF report to SharePoint: "
                f"{item.get('web_url') or item.get('name')}"
            )
        except SharePointError as exc:
            # The validation verdict is already printed and gates the exit code;
            # a failed upload is surfaced loudly but does not mask that verdict.
            logger.error(f"main: SharePoint upload failed: {exc}")
            print(f"SharePoint upload failed: {exc}", file=sys.stderr)

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
