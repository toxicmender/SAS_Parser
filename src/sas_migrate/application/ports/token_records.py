"""Persistence port for attempt-level token facts."""

from __future__ import annotations

from typing import Protocol

from sas_migrate.core.ids import RunId, ThreadId
from sas_migrate.core.tokens import CallTokenRecord


class TokenRecordRepository(Protocol):
    async def append(self, record: CallTokenRecord) -> None: ...

    async def records(
        self,
        run_id: RunId,
        thread_id: ThreadId,
    ) -> tuple[CallTokenRecord, ...]: ...


__all__ = ["TokenRecordRepository"]
