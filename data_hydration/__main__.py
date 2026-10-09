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

The plan is built from the corpus's reference inventory
(:mod:`data_hydration.inventory`): every path and dataset it names, resolved or
not. ``--inventory-table`` keeps that inventory in a Delta table, one run per
write; ``--from-inventory`` plans from the table's latest run instead of a
source directory, so a job that has the table needs neither the SAS source nor
the chunker.

Logger name: ``data_hydration.__main__``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from app_config.logging_setup import configure_logging

from .config import HydrationConfig
from .models import HydrationPlan

if TYPE_CHECKING:
    from .inventory import InventoryRow

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
        nargs="?",
        help="Directory of SAS files to plan loads for (not with --from-inventory).",
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
    parser.add_argument(
        "--inventory-table",
        metavar="TABLE",
        help=(
            "Delta table (catalog.schema.table) for the reference inventory: "
            "every path and dataset the corpus names, resolved or not. A run "
            "appends its references to it, even with --dry-run. "
            "Default: data_hydration.inventory_table."
        ),
    )
    parser.add_argument(
        "--from-inventory",
        action="store_true",
        help="Plan from the latest run in the inventory table instead of a source directory.",
    )
    parser.add_argument(
        "--run-id",
        help="With --from-inventory: plan from this inventory run, not the latest.",
    )
    parser.add_argument(
        "--check-includes",
        action="store_true",
        help=(
            "Look for every %%INCLUDEd script by file name in the source "
            "directory, and report the ones found nowhere."
        ),
    )
    parser.add_argument(
        "--sharepoint-app",
        metavar="APPLICATION",
        help=(
            "Also look for them in APPLICATION's SharePoint scripts folder "
            "({base}/APPLICATION/scripts_original). Implies --check-includes."
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Debug logging.")
    parser.add_argument("--log-file", type=Path, help="Also write logs here.")
    return parser.parse_args(argv)


def _argument_error(args: argparse.Namespace) -> str | None:
    """The first thing wrong with *args*, or ``None``.

    Validation before work, so a bad path or an unusable template is reported
    immediately rather than after the corpus has been chunked.
    """
    if args.from_inventory:
        if args.source_dir is not None:
            return "--from-inventory plans from the inventory table, not a source directory"
    elif args.source_dir is None:
        return "a source directory is required (or --from-inventory)"
    elif not args.source_dir.is_dir():
        return f"source directory not found: {args.source_dir}"
    if args.run_id and not args.from_inventory:
        return "--run-id selects an inventory run, so it needs --from-inventory"
    if args.inventory_table:
        from .inventory import _quoted_table

        try:
            _quoted_table(args.inventory_table)
        except ValueError as exc:
            return str(exc)
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
        ("inventory_table", args.inventory_table),
    ):
        if value:
            setattr(config, attribute, value)
    return config


def _corpus_inventory(args: argparse.Namespace) -> list[InventoryRow]:
    """Chunk the corpus and take its reference inventory.

    :mod:`chunker` is imported *here* rather than at module scope: the package
    itself must not depend on it (see the README's decoupling contract), and
    this entry point is a caller like any other.
    """
    from chunker import SasCorpus, SasSemanticChunker, resolve_corpus_references

    from .inventory import inventory_rows

    chunker = SasSemanticChunker()
    results = [
        chunker.chunk_file(str(path))
        for path in sorted(args.source_dir.rglob(args.pattern))
    ]
    # Resolved as one corpus, so a database LIBNAME (or a %LET) in a setup file
    # reaches the reads in the files after it — chunk_file sees one file alone.
    corpus = resolve_corpus_references(SasCorpus(file_results=results))
    return inventory_rows(corpus.file_results)


def _build_plan(
    args: argparse.Namespace,
    config: HydrationConfig,
    rows: list[InventoryRow] | None = None,
) -> HydrationPlan:
    """Plan every load the corpus's references imply, from its inventory:
    *rows*, or the inventory of ``args.source_dir``.

    Every file that names external data takes its place in corpus order, so
    the first file to read a shared table owns its item.
    """
    from .inventory import plan_from_inventory

    if rows is None:
        rows = _corpus_inventory(args)
    logger.info(
        f"_build_plan: {len({r.source_id for r in rows})} file(s) name "
        f"{len(rows)} reference(s)"
    )
    return plan_from_inventory(rows, config=config, only=args.only or ())


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


def _inventory(
    args: argparse.Namespace, config: HydrationConfig
) -> tuple[list[InventoryRow] | None, bool]:
    """The inventory to plan from, and whether keeping it failed.

    With ``--from-inventory``, one run read from the table — ``None`` when
    there is no table to read or nothing in it. Otherwise the source
    directory's, appended to the table when one is configured. A table that
    cannot be written costs the run its exit status, never its plan: the
    inventory is in hand either way.
    """
    from .inventory import read_inventory, write_inventory

    table = config.inventory_table
    if args.from_inventory:
        if not table:
            logger.error(
                "--from-inventory needs an inventory table: pass "
                "--inventory-table or set data_hydration.inventory_table"
            )
            return None, False
        try:
            rows = read_inventory(table, run_id=args.run_id)
        except Exception as exc:
            logger.error(
                f"could not read the reference inventory from {table} — "
                f"{type(exc).__name__}: {exc}"
            )
            return None, False
        if not rows:
            logger.error(f"no inventory run to plan from in {table}")
            return None, False
        return _match_includes(args, rows)
    rows, failed = _match_includes(args, _corpus_inventory(args))
    if not table:
        return rows, failed
    try:
        write_inventory(rows, table)
    except Exception as exc:
        logger.error(
            f"could not write the reference inventory to {table} — "
            f"{type(exc).__name__}: {exc}"
        )
        return rows, True
    return rows, failed


def _match_includes(
    args: argparse.Namespace, rows: list[InventoryRow]
) -> tuple[list[InventoryRow], bool]:
    """*rows* with every ``%INCLUDE`` looked for, when asked; and whether a
    place could not be searched. The source directory is searched when there
    is one, and ``--sharepoint-app``'s scripts folder when given. A folder
    SharePoint cannot list is reported and left unsearched."""
    if not (args.check_includes or args.sharepoint_app):
        return rows, False
    from .includes import local_index, match_includes, sharepoint_index

    local = local_index(args.source_dir) if args.source_dir is not None else None
    sharepoint = None
    failed = False
    if args.sharepoint_app:
        # The folder convention is conversion's, stated once there; this entry
        # point imports it the way it imports the chunker.
        from conversion.paths import original_scripts

        folder = original_scripts(args.sharepoint_app)
        try:
            sharepoint = sharepoint_index(folder)
        except Exception as exc:
            logger.error(
                f"could not list SharePoint folder {folder!r} — "
                f"{type(exc).__name__}: {exc}"
            )
            failed = True
    return match_includes(rows, local=local, sharepoint=sharepoint), failed


def _print_includes(rows: list[InventoryRow]) -> None:
    """Where each %INCLUDEd script was found, when anyone looked."""
    from .includes import include_checks

    checks = [c for c in include_checks(rows) if c.status != "unchecked"]
    if not checks:
        return
    missing = [c for c in checks if c.status != "found"]
    print(f"Included scripts — {len(checks)}, {len(missing)} not found\n")
    for check in checks:
        where = [
            f"{place}: {', '.join(found)}"
            for place, found in (("local", check.local), ("SharePoint", check.sharepoint))
            if found
        ]
        name = check.file_name or check.spelled
        status = "; ".join(where) if where else f"** {check.status}"
        print(f"  {name}  ->  {status}")
        for source_id, line in check.included_by:
            print(f"      included by {source_id}:{line}")
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(debug=args.debug, log_file=args.log_file)

    problem = _argument_error(args)
    if problem:
        logger.error(problem)
        return EXIT_ARGS

    from .naming import TableNameError

    config = _config_for(args)
    rows, inventory_failed = _inventory(args, config)
    if rows is None:
        return EXIT_ARGS if not config.inventory_table else EXIT_FAILED
    try:
        plan = _build_plan(args, config, rows)
    except TableNameError as exc:
        # A template problem is a configuration error, not a crash: report it
        # the way the argument errors above are reported.
        logger.error(f"table template: {exc}")
        return EXIT_ARGS

    _print_plan(plan)
    _print_includes(rows)
    if args.dry_run:
        return EXIT_FAILED if inventory_failed else EXIT_OK

    from .runner import execute

    report = execute(plan, config=config)
    for outcome in report.outcomes:
        if outcome.error:
            logger.warning(str(outcome))
    logger.info(str(report))
    return EXIT_OK if report.ok and not inventory_failed else EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
