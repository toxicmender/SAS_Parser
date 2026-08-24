"""Composition root and command shell for the v2 application."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from sas_migrate import __version__
from sas_migrate.application.deployment import run_deployment_smoke


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
    parser.print_help()
    return 0


__all__ = ["build_parser", "main"]
