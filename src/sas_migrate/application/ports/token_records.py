"""Persistence port for attempt-level token facts."""

from __future__ import annotations

from typing import Protocol

from sas_migrate.core.tokens import CallTokenRecord


class TokenRecordRepository(Protocol):
    async def append(self, record: CallTokenRecord) -> None: ...


__all__ = ["TokenRecordRepository"]
