"""Versioned run and item state snapshots."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from ..ids import ItemId, RunId
from ..models import ContractModel, VersionedContract
from ..targets.models import ResolvedTarget


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ItemStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    ACCEPTED = "accepted"
    FAILED = "failed"
    SKIPPED = "skipped"


class ItemState(ContractModel):
    item_id: ItemId
    status: ItemStatus
    attempt: int = Field(default=0, ge=0)
    error: str | None = None


class RunState(VersionedContract):
    run_id: RunId
    status: RunStatus
    resolved_target: ResolvedTarget
    created_at: datetime
    updated_at: datetime
    items: tuple[ItemState, ...] = Field(default_factory=tuple)
