"""Append-only persistence port for v2 run events."""

from __future__ import annotations

from typing import Protocol

from sas_migrate.core.ids import RunId, ThreadId
from sas_migrate.core.runs import RunEvent


class RunEventRepository(Protocol):
    async def append(self, event: RunEvent) -> None: ...

    async def events(
        self,
        run_id: RunId,
        thread_id: ThreadId,
    ) -> tuple[RunEvent, ...]: ...


__all__ = ["RunEventRepository"]
