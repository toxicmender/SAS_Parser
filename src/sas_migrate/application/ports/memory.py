"""Accepted-response memory port."""

from __future__ import annotations

from typing import Protocol

from sas_migrate.core.ids import ItemId, RunId, ThreadId
from sas_migrate.core.responses import ResponseEnvelope


class MemoryPort(Protocol):
    async def accepted_response(
        self,
        run_id: RunId,
        thread_id: ThreadId,
        item_id: ItemId,
    ) -> ResponseEnvelope | None: ...

    async def remember_accepted(
        self,
        run_id: RunId,
        thread_id: ThreadId,
        item_id: ItemId,
        response: ResponseEnvelope,
    ) -> None: ...
