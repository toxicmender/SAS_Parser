"""Executing a plan, one item at a time, without letting one failure end the run.

The runner is deliberately thin. Every decision — what to read, what to call the
target, how to slice it — was made by :mod:`data_hydration.planner` and is on the
plan already; this walks the items and moves the bytes.

Two rules it exists to enforce:

* **A failed item is recorded, not raised.** One unreachable host must not cost
  the operator the other forty tables. Each item is wrapped, the error text lands
  on its :class:`~data_hydration.models.ItemOutcome`, and the run carries on —
  the "degrade, don't crash" rule this repo applies from ``app_config`` outwards.
  ``data_hydration.on_error = "stop"`` opts out.
* **A blocked item never runs.** An item the planner marked
  ``needs_operator_input`` is skipped with its blockers intact, because the
  coordinates it holds are known to be wrong — an unresolved ``&macro`` password
  is not a credential to try, it is a question for a human.

Logger name: ``data_hydration.runner``.
"""

from __future__ import annotations

import logging

from .config import HydrationConfig
from .models import (
    HydrationItem,
    HydrationPlan,
    HydrationReport,
    ItemOutcome,
    ItemStatus,
)

logger = logging.getLogger(__name__)


def execute(
    plan: HydrationPlan,
    *,
    config: HydrationConfig | None = None,
    dry_run: bool = False,
) -> HydrationReport:
    """Run *plan*, returning what happened to every item.

    Parameters
    ----------
    plan
        Built by :func:`data_hydration.planner.build_corpus_plan`.
    config
        ``None`` builds one with :meth:`HydrationConfig.from_env`. Supplies
        ``on_error``, the secret scope, and the driver settings.
    dry_run
        Skip every item without connecting to anything, so the report shows
        exactly what a real run would attempt. The offline check.

    The return value is always a report — an exception escapes only if
    ``on_error`` is ``"stop"``, and even then the failure is on the report first.
    """
    config = config or HydrationConfig.from_env()
    outcomes: list[ItemOutcome] = []

    for item in plan.items:
        if dry_run:
            outcomes.append(
                ItemOutcome(item=item, status=ItemStatus.SKIPPED, error="dry run")
            )
            continue
        if item.blockers:
            logger.warning(
                f"execute: skipping {item.target_table} — {item.blockers[0]}"
            )
            outcomes.append(
                ItemOutcome(
                    item=item,
                    status=ItemStatus.SKIPPED,
                    error="; ".join(item.blockers),
                )
            )
            continue
        outcome = _run_item(item, config)
        outcomes.append(outcome)
        if outcome.status is ItemStatus.FAILED and config.on_error == "stop":
            logger.error(
                f"execute: stopping after {item.target_table} failed "
                f"(data_hydration.on_error='stop')"
            )
            break

    return HydrationReport(plan=plan, outcomes=outcomes, dry_run=dry_run)


def _run_item(item: HydrationItem, config: HydrationConfig) -> ItemOutcome:
    """One item, with its failure captured rather than propagated."""
    try:
        from .sinks.delta import write_item

        rows = write_item(item, config)
    except Exception as exc:
        # The type is named because these read alike once stringified and need
        # opposite fixes: a credential error is a secret-scope problem, an
        # ImportError is a missing driver extra, an OSError is the network.
        message = f"{type(exc).__name__}: {exc}"
        logger.error(f"execute: {item.target_table} failed — {message}")
        return ItemOutcome(item=item, status=ItemStatus.FAILED, error=message)
    logger.info(f"execute: wrote {rows} row(s) to {item.target_table}")
    return ItemOutcome(item=item, status=ItemStatus.WRITTEN, rows=rows)
