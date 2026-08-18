"""Event-sourced run state, resume, rewind, and fork control."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any
from uuid import uuid4

from sas_migrate.application.ports import (
    Clock,
    MemoryPort,
    RunEventRepository,
    TokenRecordRepository,
)
from sas_migrate.core.errors import ContractError
from sas_migrate.core.ids import ItemId, RunId, ThreadId
from sas_migrate.core.runs import (
    ItemState,
    ItemStatus,
    RunEvent,
    RunEventType,
    RunState,
    RunStatus,
)
from sas_migrate.core.targets import ResolvedTarget


class RunStateService:
    def __init__(
        self,
        *,
        events: RunEventRepository,
        memory: MemoryPort,
        token_records: TokenRecordRepository,
        clock: Clock,
        event_id: Callable[[], str] | None = None,
    ) -> None:
        self._events = events
        self._memory = memory
        self._token_records = token_records
        self._clock = clock
        self._event_id = event_id or (lambda: uuid4().hex)

    async def _append(
        self,
        event_type: RunEventType,
        *,
        run_id: RunId,
        thread_id: ThreadId,
        item_id: ItemId | None = None,
        attempt: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> RunEvent:
        event = RunEvent(
            event_id=self._event_id(),
            event_type=event_type,
            occurred_at=self._clock.now(),
            run_id=run_id,
            thread_id=thread_id,
            item_id=item_id,
            attempt=attempt,
            payload=payload or {},
        )
        await self._events.append(event)
        return event

    async def start(
        self,
        run_id: RunId,
        thread_id: ThreadId,
        target: ResolvedTarget,
        *,
        allow_existing: bool = False,
    ) -> RunState:
        existing = await self.state(run_id, thread_id)
        if existing is not None:
            if not allow_existing:
                raise ContractError(f"run {run_id!r} already exists")
            if existing.resolved_target != target:
                raise ContractError("resumed run target differs from stored target")
            return existing
        await self._append(
            RunEventType.RUN_STARTED,
            run_id=run_id,
            thread_id=thread_id,
            payload={"resolved_target": target.model_dump(mode="json")},
        )
        state = await self.state(run_id, thread_id)
        if state is None:
            raise RuntimeError("run start event was not persisted")
        return state

    async def item_started(
        self,
        run_id: RunId,
        thread_id: ThreadId,
        item_id: ItemId,
        attempt: int,
    ) -> None:
        await self._append(
            RunEventType.ITEM_STARTED,
            run_id=run_id,
            thread_id=thread_id,
            item_id=item_id,
            attempt=attempt,
        )

    async def attempt_completed(
        self,
        run_id: RunId,
        thread_id: ThreadId,
        item_id: ItemId,
        attempt: int,
        *,
        valid: bool,
        sent: bool,
    ) -> None:
        await self._append(
            RunEventType.ATTEMPT_COMPLETED,
            run_id=run_id,
            thread_id=thread_id,
            item_id=item_id,
            attempt=attempt,
            payload={"valid": valid, "sent": sent},
        )

    async def item_accepted(
        self,
        run_id: RunId,
        thread_id: ThreadId,
        item_id: ItemId,
        attempt: int,
        *,
        recovered: bool = False,
    ) -> None:
        await self._append(
            RunEventType.ITEM_ACCEPTED,
            run_id=run_id,
            thread_id=thread_id,
            item_id=item_id,
            attempt=attempt,
            payload={"recovered": recovered},
        )

    async def item_failed(
        self,
        run_id: RunId,
        thread_id: ThreadId,
        item_id: ItemId,
        attempt: int,
        error: str,
    ) -> None:
        await self._append(
            RunEventType.ITEM_FAILED,
            run_id=run_id,
            thread_id=thread_id,
            item_id=item_id,
            attempt=attempt,
            payload={"error": error},
        )

    async def completed(self, run_id: RunId, thread_id: ThreadId) -> None:
        await self._append(
            RunEventType.RUN_COMPLETED,
            run_id=run_id,
            thread_id=thread_id,
        )

    async def failed(
        self,
        run_id: RunId,
        thread_id: ThreadId,
        error: str,
    ) -> None:
        await self._append(
            RunEventType.RUN_FAILED,
            run_id=run_id,
            thread_id=thread_id,
            payload={"error": error},
        )

    async def state(self, run_id: RunId, thread_id: ThreadId) -> RunState | None:
        events = await self._events.events(run_id, thread_id)
        if not events:
            return None
        first = events[0]
        if first.event_type is not RunEventType.RUN_STARTED:
            raise ContractError("run event stream does not start with run_started")
        target_value = first.payload.get("resolved_target")
        if not isinstance(target_value, dict):
            raise ContractError("run_started event has no resolved target")
        target = ResolvedTarget.model_validate(target_value)
        status = RunStatus.RUNNING
        items: dict[str, ItemState] = {}
        order: list[str] = []
        for event in events[1:]:
            item_id = event.item_id
            if item_id is not None and item_id not in items:
                order.append(item_id)
                items[item_id] = ItemState(item_id=item_id, status=ItemStatus.PENDING)
            if event.event_type is RunEventType.ITEM_STARTED and item_id is not None:
                status = RunStatus.RUNNING
                items[item_id] = ItemState(
                    item_id=item_id,
                    status=ItemStatus.RUNNING,
                    attempt=event.attempt or items[item_id].attempt,
                )
            elif (
                event.event_type is RunEventType.ATTEMPT_COMPLETED
                and item_id is not None
            ):
                items[item_id] = items[item_id].model_copy(
                    update={"attempt": event.attempt or items[item_id].attempt}
                )
            elif event.event_type is RunEventType.ITEM_ACCEPTED and item_id is not None:
                items[item_id] = ItemState(
                    item_id=item_id,
                    status=ItemStatus.ACCEPTED,
                    attempt=event.attempt or items[item_id].attempt,
                )
            elif event.event_type is RunEventType.ITEM_FAILED and item_id is not None:
                items[item_id] = ItemState(
                    item_id=item_id,
                    status=ItemStatus.FAILED,
                    attempt=event.attempt or items[item_id].attempt,
                    error=str(event.payload.get("error") or "item failed"),
                )
            elif event.event_type is RunEventType.ITEM_REWOUND and item_id is not None:
                status = RunStatus.RUNNING
                items[item_id] = ItemState(
                    item_id=item_id,
                    status=ItemStatus.PENDING,
                    attempt=event.attempt or items[item_id].attempt,
                )
            elif event.event_type is RunEventType.RUN_COMPLETED:
                status = RunStatus.COMPLETED
            elif event.event_type is RunEventType.RUN_FAILED:
                status = RunStatus.FAILED
        return RunState(
            run_id=run_id,
            status=status,
            resolved_target=target,
            created_at=first.occurred_at,
            updated_at=events[-1].occurred_at,
            items=tuple(items[item_id] for item_id in order),
        )

    async def rewind(
        self,
        run_id: RunId,
        thread_id: ThreadId,
        ordered_item_ids: Sequence[ItemId],
        start_item_id: ItemId,
    ) -> tuple[ItemId, ...]:
        try:
            start = ordered_item_ids.index(start_item_id)
        except ValueError as exc:
            raise ContractError("rewind start item is not in the corpus") from exc
        state = await self.state(run_id, thread_id)
        if state is None:
            raise ContractError("cannot rewind an unknown run")
        by_id = {item.item_id: item for item in state.items}
        affected = tuple(ordered_item_ids[start:])
        await self._memory.forget_accepted(run_id, thread_id, affected)
        for item_id in affected:
            previous = by_id.get(item_id)
            await self._append(
                RunEventType.ITEM_REWOUND,
                run_id=run_id,
                thread_id=thread_id,
                item_id=item_id,
                attempt=previous.attempt if previous else None,
            )
        return affected

    async def fork(
        self,
        *,
        source_run_id: RunId,
        source_thread_id: ThreadId,
        destination_run_id: RunId,
        destination_thread_id: ThreadId,
        ordered_item_ids: Sequence[ItemId],
        upto_items: int | None = None,
    ) -> tuple[ItemId, ...]:
        source = await self.state(source_run_id, source_thread_id)
        if source is None:
            raise ContractError("cannot fork an unknown run")
        accepted = {
            item.item_id: item
            for item in source.items
            if item.status is ItemStatus.ACCEPTED
        }
        copied = tuple(
            item_id
            for item_id in ordered_item_ids[:upto_items]
            if item_id in accepted
        )
        await self.start(
            destination_run_id,
            destination_thread_id,
            source.resolved_target,
        )
        await self._append(
            RunEventType.RUN_FORKED,
            run_id=destination_run_id,
            thread_id=destination_thread_id,
            payload={
                "source_run_id": source_run_id,
                "source_thread_id": source_thread_id,
                "copied_items": list(copied),
            },
        )
        await self._memory.fork_accepted(
            source_run_id,
            source_thread_id,
            destination_run_id,
            destination_thread_id,
            copied,
        )
        source_records = await self._token_records.records(
            source_run_id,
            source_thread_id,
        )
        for record in source_records:
            if record.item_id not in copied:
                continue
            await self._token_records.append(
                record.model_copy(
                    update={
                        "run_id": destination_run_id,
                        "thread_id": destination_thread_id,
                        "recovered": True,
                    }
                )
            )
        for item_id in copied:
            item = accepted[item_id]
            await self.item_started(
                destination_run_id,
                destination_thread_id,
                item_id,
                item.attempt,
            )
            await self.item_accepted(
                destination_run_id,
                destination_thread_id,
                item_id,
                item.attempt,
                recovered=True,
            )
        return copied


__all__ = ["RunStateService"]
