"""Append-only run events suitable for ledgers and resume."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from ..ids import ItemId, RunId, ThreadId
from ..models import VersionedContract


class RunEventType(StrEnum):
    RUN_STARTED = "run_started"
    ITEM_STARTED = "item_started"
    ATTEMPT_COMPLETED = "attempt_completed"
    ITEM_ACCEPTED = "item_accepted"
    ITEM_FAILED = "item_failed"
    ITEM_REWOUND = "item_rewound"
    RUN_FORKED = "run_forked"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


class RunEvent(VersionedContract):
    event_id: str = Field(min_length=1)
    event_type: RunEventType
    occurred_at: datetime
    run_id: RunId
    thread_id: ThreadId | None = None
    item_id: ItemId | None = None
    attempt: int | None = Field(default=None, ge=1)
    payload: dict[str, Any] = Field(default_factory=dict)
