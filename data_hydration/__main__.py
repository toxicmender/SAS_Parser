"""``python -m data_hydration`` — plan a corpus's data loads, and optionally run them.

A separate entry point from ``main.py`` on purpose. Architecture.md invariant 12
keeps ``main.py`` as the one conversion flow; hydration is a different job with a
different cadence — it moves data once, where conversion runs per request — and
folding it in would give the conversion CLI a set of flags that never apply to a
conversion. The shape follows ``python -m complexity``: ``parse_args``, a
separate ``_argument_error`` validation pass, then ``main`` returning an exit
code.

``--dry-run`` is the important mode: it prints exactly what a real run would do,
opening no connection and needing no driver installed, because the planner does
no I/O.

Logger name: ``data_hydration.__main__``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app_config.logging_setup import configure_logging

from .config import HydrationConfig
from .models import HydrationPlan

logger = logging.getLogger("data_hydration")

EXIT_OK = 0
EXIT_ARGS = 2
EXIT_FAILED = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m data_hydration",
        description=(
            "Plan and run the data loads a SAS corpus implies: read the "
            "LIBNAMEs and paths the chunker finds, and land each source as a "
            "managed Delta table."
        ),
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory of SAS files to plan loads for.",
    )
    parser.add_argument(
        "--pattern",
        default="*.sas",
        help="Glob for SAS files under the source directory (default: *.sas).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and stop. Opens no connection.",
    )
    parser.add_argument(
        "--stage",
        help="Value for the <stage> placeholder in the table template.",
    )
    parser.add_argument("--catalog", help="Target Unity Catalog catalog.")
    parser.add_argument("--schema", help="Target schema (default: the SAS libref).")
    parser.add_argument(
        "--table-template",
        help=(
            "Target-name template, e.g. "
            "'<catalog_name>.<schema_name>.<table_name>_<stage>_<date>'."
        ),
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="LIBREF",
        help="Hydrate only these librefs. Repeatable.",
    )
    parser.add_argument("--debug", action="store_true", help="Debug logging.")
    parser.add_argument("--log-file", type=Path, help="Also write logs here.")
    return parser.parse_args(argv)


def _argument_error(args: argparse.Namespace) -> str | None:
    """The first thing wrong with *args*, or ``None``.

    Validation before work, so a bad path or an unusable template is reported
    immediately rather than after the corpus has been chunked.
    """
    if not args.source_dir.is_dir():
        return f"source directory not found: {args.source_dir}"
    if args.table_template:
        from .naming import TableNameError, validate_template

        try:
            validate_template(args.table_template)
        except TableNameError as exc:
            return str(exc)
    return None


def _config_for(args: argparse.Namespace) -> HydrationConfig:
    """The run's config, with CLI values overriding the resolved ones."""
    config = HydrationConfig.from_env()
    for attribute, value in (
        ("catalog", args.catalog),
        ("schema", args.schema),
        ("stage", args.stage),
        ("table_template", args.table_template),
    ):
        if value:
            setattr(config, attribute, value)
    return config


def _build_plan(args: argparse.Namespace, config: HydrationConfig) -> HydrationPlan:
    """Chunk the corpus and plan every load its references imply.

    :mod:`chunker` is imported *here* rather than at module scope: the package
    itself must not depend on it (see the README's decoupling contract), and
    this entry point is a caller like any other.
    """
    from chunker import SasSemanticChunker

    from .planner import build_corpus_plan

    chunker = SasSemanticChunker()
    by_source: dict[str, tuple[list, list]] = {}
    for path in sorted(args.source_dir.rglob(args.pattern)):
        result = chunker.chunk_file(str(path))
        engine_refs = [r for c in result.chunks for r in c.metadata.engine_refs]
        path_refs = [r for c in result.chunks for r in c.metadata.external_refs]
        if args.only:
            wanted = {libref.lower() for libref in args.only}
            engine_refs = [r for r in engine_refs if r.binds in wanted]
            path_refs = [r for r in path_refs if (r.binds or "") in wanted]
        if engine_refs or path_refs:
            by_source[str(path)] = (engine_refs, path_refs)
    logger.info(f"_build_plan: {len(by_source)} file(s) name external data")
    return build_corpus_plan(by_source, config=config, probe=None)


def _print_plan(plan: HydrationPlan) -> None:
    """The plan as a table, which is what a dry run is for."""
    if not plan.items:
        print("No external data sources found.")
        return
    print(f"\nHydration plan — {len(plan.items)} item(s), date {plan.run_date}\n")
    width = max(len(str(i.source)) for i in plan.items)
    for item in plan.items:
        flag = "  ** needs operator input" if item.blockers else ""
        print(f"  {str(item.source):<{width}}  ->  {item.target_table}{flag}")
        print(f"  {'':<{width}}      {item.strategy}: {item.strategy_reason}")
        for note in item.notes:
            print(f"  {'':<{width}}      - {note}")
        for blocker in item.blockers:
            print(f"  {'':<{width}}      ! {blocker}")
    print(
        f"\n{len(plan.target_tables)} target table(s); "
        f"{plan.blocked_count} item(s) need operator input.\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(debug=args.debug, log_file=args.log_file)

    problem = _argument_error(args)
    if problem:
        logger.error(problem)
        return EXIT_ARGS

    from .naming import TableNameError

    config = _config_for(args)
    try:
        plan = _build_plan(args, config)
    except TableNameError as exc:
        # A template problem is a configuration error, not a crash: report it
        # the way the argument errors above are reported.
        logger.error(f"table template: {exc}")
        return EXIT_ARGS

    _print_plan(plan)
    if args.dry_run:
        return EXIT_OK

    from .runner import execute

    report = execute(plan, config=config)
    for outcome in report.outcomes:
        if outcome.error:
            logger.warning(str(outcome))
    logger.info(str(report))
    return EXIT_OK if report.ok else EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
