"""Composition root and command shell for the v2 application."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from sas_migrate import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sas-migrate",
        description="SAS Migrate v2 command shell (operational commands are not enabled yet).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


__all__ = ["build_parser", "main"]
