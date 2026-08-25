"""Failure-isolated hydration plan execution through v2 ports."""

from __future__ import annotations

from sas_migrate.application.ports.hydration import (
    HydrationDriverRegistry,
    HydrationSink,
)

from .models import HydrationItemOutcome, HydrationPlan, HydrationReport, ItemStatus


class HydrationWorkflow:
    def __init__(self, *, drivers: HydrationDriverRegistry, sink: HydrationSink) -> None:
        self._drivers = drivers
        self._sink = sink

    def run(
        self,
        plan: HydrationPlan,
        *,
        dry_run: bool = False,
        on_error: str = "continue",
    ) -> HydrationReport:
        if on_error not in {"continue", "stop"}:
            raise ValueError("on_error must be 'continue' or 'stop'")
        outcomes: list[HydrationItemOutcome] = []
        for item in plan.items:
            if dry_run:
                outcomes.append(HydrationItemOutcome(item=item, status=ItemStatus.SKIPPED, error="dry run"))
                continue
            if item.blockers:
                outcomes.append(
                    HydrationItemOutcome(item=item, status=ItemStatus.SKIPPED, error="; ".join(item.blockers))
                )
                continue

            driver = None
            try:
                driver = self._drivers.driver_for(item.source.kind)
                rows = self._sink.write(item, driver.batches(item))
                outcome = HydrationItemOutcome(item=item, status=ItemStatus.WRITTEN, rows=rows)
            except Exception as exc:  # noqa: BLE001 - isolate source items
                outcome = HydrationItemOutcome(
                    item=item,
                    status=ItemStatus.FAILED,
                    error=f"{type(exc).__name__}: {exc}",
                )
            finally:
                if driver is not None:
                    driver.close()
            outcomes.append(outcome)
            if outcome.status is ItemStatus.FAILED and on_error == "stop":
                break
        return HydrationReport(plan=plan, outcomes=tuple(outcomes), dry_run=dry_run)


__all__ = ["HydrationWorkflow"]
