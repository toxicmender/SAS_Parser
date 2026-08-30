"""Composition root and command shell for the v2 application."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sas_migrate import __version__
from sas_migrate.application.deployment import run_deployment_smoke
from sas_migrate.core.targets import TargetId

from .commands import (
    CommandError,
    ReportFormat,
    run_assess,
    run_check_sharepoint,
    run_convert_local,
    run_validate,
)


def _report_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=tuple(ReportFormat),
        default=ReportFormat.JSON.value,
        dest="output_format",
        help="report representation (default: json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the report to PATH instead of stdout; required for PDF",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sas-migrate",
        description="SAS Migrate v2 application.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    smoke = commands.add_parser(
        "smoke",
        help="run the credential-free v2 deployment smoke",
    )
    smoke.add_argument(
        "--json",
        action="store_true",
        help="emit the versioned machine-readable smoke report",
    )

    convert = commands.add_parser("convert", help="run a v2 conversion workflow")
    convert_commands = convert.add_subparsers(dest="convert_command", metavar="SOURCE")
    convert_local = convert_commands.add_parser(
        "local",
        help="translate SAS files from a local directory",
    )
    convert_local.add_argument("source_dir", type=Path)
    convert_local.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sas-migrate-output"),
        help="artifact directory (default: sas-migrate-output)",
    )
    convert_local.add_argument(
        "--target",
        choices=tuple(target.value for target in TargetId),
        default=TargetId.SPARK_SQL.value,
        help="migration target (default: spark_sql)",
    )
    convert_local.add_argument("--model", default="gpt-5.4")
    convert_local.add_argument("--application-name")
    convert_local.add_argument("--request-id", default="local")
    convert_local.add_argument("--gateway-base-url")
    convert_local.add_argument(
        "--api-key-env",
        default="SAS_MIGRATE_GATEWAY_API_KEY",
        help="environment variable containing the gateway credential",
    )
    convert_local.add_argument("--gateway-version")
    convert_local.add_argument("--max-input-tokens", type=int, default=128_000)
    convert_local.add_argument("--reserved-output-tokens", type=int, default=16_000)
    convert_local.add_argument("--safety-margin-tokens", type=int, default=2_000)
    convert_local.add_argument("--max-run-tokens", type=int)
    convert_local.add_argument("--max-attempts", type=int, default=2)
    convert_local.add_argument(
        "--dry-run",
        action="store_true",
        help="parse inputs and write a plan without invoking the gateway",
    )

    check = commands.add_parser("check", help="run a read-only deployment check")
    check_commands = check.add_subparsers(dest="check_command", metavar="RESOURCE")
    check_sharepoint = check_commands.add_parser(
        "sharepoint",
        help="validate SharePoint configuration and read access",
    )
    check_sharepoint.add_argument(
        "--config",
        type=Path,
        help="optional v2 JSON settings document; environment values override it",
    )
    check_sharepoint.add_argument(
        "--offline",
        action="store_true",
        help="check configuration and installed SDKs without credentials or network I/O",
    )
    check_sharepoint.add_argument(
        "--output",
        type=Path,
        help="write the versioned JSON report to PATH instead of stdout",
    )
    smoke.add_argument(
        "--quiet",
        action="store_true",
        help="suppress a successful report (for image health checks)",
    )
    smoke.add_argument(
        "--require-wheel",
        action="store_true",
        help="fail when the distribution is installed editable or is unavailable",
    )
    smoke.add_argument(
        "--require-non-root",
        action="store_true",
        help="fail unless the runtime has a non-root effective user id",
    )

    assess = commands.add_parser(
        "assess",
        help="assess versioned SAS analysis units offline",
    )
    assess.add_argument("input", type=Path, help="JSON array of AssessmentUnit objects")
    assess.add_argument(
        "--target",
        choices=tuple(target.value for target in TargetId),
        default=TargetId.SPARK_SQL.value,
        help="migration target (default: spark_sql)",
    )
    assess.add_argument(
        "--profiles",
        type=Path,
        help="optional directory containing pyspark.json and sparksql.json profiles",
    )
    _report_arguments(assess)

    validate = commands.add_parser(
        "validate",
        help="evaluate a versioned offline EvaluationRun",
    )
    validate.add_argument("input", type=Path, help="EvaluationRun JSON document")
    validate.add_argument(
        "--model",
        default="offline",
        help="model label recorded in the report (default: offline)",
    )
    validate.add_argument("--translation-ledger", type=Path)
    validate.add_argument("--judge-ledger", type=Path)
    validate.add_argument("--translation-policy", type=Path)
    validate.add_argument("--judge-policy", type=Path)
    _report_arguments(validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "smoke":
            report = run_deployment_smoke(
                require_wheel=args.require_wheel,
                require_non_root=args.require_non_root,
            )
            if not (args.quiet and report.passed):
                if args.json:
                    print(report.to_json())
                else:
                    status = "PASSED" if report.passed else "FAILED"
                    print(f"v2 deployment smoke: {status}")
                    for check in report.checks:
                        outcome = "pass" if check.passed else "FAIL"
                        print(f"- {check.name}: {outcome} — {check.details}")
            return 0 if report.passed else 1
        if args.command == "assess":
            return run_assess(args)
        if args.command == "validate":
            return run_validate(args)
        if args.command == "convert" and args.convert_command == "local":
            return run_convert_local(args)
        if args.command == "check" and args.check_command == "sharepoint":
            return run_check_sharepoint(args)
    except CommandError as exc:
        parser.print_usage(sys.stderr)
        print(f"sas-migrate: error: {exc}", file=sys.stderr)
        return 2
    parser.print_help()
    return 0


__all__ = ["build_parser", "main"]
