"""Delta-backed implementation of the v2 conversation-memory repository."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel

from sas_migrate.application.memory.models import (
    ChatMessage,
    MemoryAuditEvent,
    MemorySnapshot,
    PolicyProposal,
    RollingSummary,
    TaskPolicySnapshot,
    ThreadNote,
)
from sas_migrate.application.ports import Clock
from sas_migrate.core.ids import ItemId, RunId, ThreadId
from sas_migrate.core.responses import ResponseEnvelope

from .delta_store import DeltaKVStore


class MemoryKVStore(Protocol):
    """Small synchronous surface supplied by the existing Delta KV engine."""

    def set(
        self,
        key: str,
        value: Any,
        tags: list[str] | None = None,
        source: str | None = None,
    ) -> None: ...

    def get(self, key: str, default: Any = None) -> Any: ...

    def delete(self, key: str) -> bool: ...

    def delete_many(self, keys: list[str]) -> int: ...

    def keys(self, prefix: str = "") -> list[str]: ...

    def all_records(self, prefix: str = "") -> list[tuple[str, dict[str, Any]]]: ...

    def sync_cdf(self, consumer_id: str) -> Any: ...


def _payload(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


class DeltaMemoryRepository:
    """Persist every v2 memory entity in one CDF-enabled Delta KV table."""

    def __init__(
        self,
        store: MemoryKVStore,
        clock: Clock,
        *,
        identifier: Callable[[], str] | None = None,
    ) -> None:
        self._store = store
        self._clock = clock
        self._identifier = identifier or (lambda: uuid4().hex)

    @classmethod
    def from_delta(
        cls,
        spark: Any,
        table: str,
        clock: Clock,
        *,
        audit_table: str,
        max_write_retries: int = 3,
        identifier: Callable[[], str] | None = None,
    ) -> DeltaMemoryRepository:
        """Create the adapter without importing Spark on in-memory code paths."""

        store = DeltaKVStore(
            spark,
            table,
            audit_table=audit_table,
            max_write_retries=max_write_retries,
        )
        return cls(store, clock, identifier=identifier)

    @staticmethod
    def _message_prefix(thread_id: str) -> str:
        return f"v2::message::{thread_id}::"

    @classmethod
    def _message_key(cls, message: ChatMessage) -> str:
        return f"{cls._message_prefix(message.thread_id)}{message.sequence:020d}"

    @staticmethod
    def _note_prefix(thread_id: str) -> str:
        return f"v2::note::{thread_id}::"

    @classmethod
    def _note_key(cls, note: ThreadNote) -> str:
        return f"{cls._note_prefix(note.thread_id)}{note.note_id}"

    @staticmethod
    def _policy_key(task_id: str) -> str:
        return f"v2::policy::{task_id}"

    @staticmethod
    def _summary_key(thread_id: str) -> str:
        return f"v2::summary::{thread_id}"

    @staticmethod
    def _proposal_key(proposal_id: str) -> str:
        return f"v2::proposal::{proposal_id}"

    @staticmethod
    def _accepted_key(run_id: str, thread_id: str, item_id: str) -> str:
        return f"v2::accepted::{run_id}::{thread_id}::{item_id}"

    def _record(
        self,
        operation: str,
        *,
        thread_id: str | None = None,
        entity_id: str | None = None,
        **details: str,
    ) -> None:
        event = MemoryAuditEvent(
            event_id=self._identifier(),
            operation=operation,
            occurred_at=self._clock.now(),
            thread_id=thread_id,
            entity_id=entity_id,
            details=details,
        )
        timestamp = event.occurred_at.isoformat()
        key = f"v2::audit::{timestamp}::{event.event_id}"
        self._store.set(key, _payload(event), tags=[operation], source="v2-memory")

    async def append_message(self, message: ChatMessage) -> None:
        messages = await self.messages(message.thread_id)
        expected = messages[-1].sequence + 1 if messages else 1
        if message.sequence != expected:
            raise ValueError(
                f"message sequence {message.sequence} does not follow {expected - 1}"
            )
        if any(existing.message_id == message.message_id for existing in messages):
            raise ValueError("message id already exists in thread")
        self._store.set(
            self._message_key(message),
            _payload(message),
            tags=["message", message.thread_id],
            source="v2-memory",
        )
        self._record(
            "message_appended",
            thread_id=message.thread_id,
            entity_id=message.message_id,
            sequence=str(message.sequence),
        )

    async def messages(
        self,
        thread_id: str,
        *,
        after_sequence: int = 0,
    ) -> tuple[ChatMessage, ...]:
        return tuple(
            message
            for _, record in self._store.all_records(self._message_prefix(thread_id))
            for message in (ChatMessage.model_validate(record["value"]),)
            if message.sequence > after_sequence
        )

    async def rewind_messages(self, thread_id: str, *, after_sequence: int) -> int:
        messages = await self.messages(thread_id)
        targets = [
            self._message_key(message)
            for message in messages
            if message.sequence > after_sequence
        ]
        removed = self._store.delete_many(targets)
        summary = await self.summary(thread_id)
        if summary is not None and summary.through_sequence > after_sequence:
            self._store.delete(self._summary_key(thread_id))
        self._record(
            "messages_rewound",
            thread_id=thread_id,
            removed=str(removed),
            after_sequence=str(after_sequence),
        )
        return removed

    async def policy(self, task_id: str) -> TaskPolicySnapshot | None:
        value = self._store.get(self._policy_key(task_id))
        return None if value is None else TaskPolicySnapshot.model_validate(value)

    async def put_policy(self, policy: TaskPolicySnapshot) -> None:
        previous = await self.policy(policy.task_id)
        if previous is not None and policy.version <= previous.version:
            raise ValueError("policy version must increase")
        self._store.set(
            self._policy_key(policy.task_id),
            _payload(policy),
            tags=["policy"],
            source="v2-memory",
        )
        self._record(
            "policy_written", entity_id=policy.task_id, version=str(policy.version)
        )

    async def notes(
        self, thread_id: str, *, now: datetime
    ) -> tuple[ThreadNote, ...]:
        pairs = [
            (key, ThreadNote.model_validate(record["value"]))
            for key, record in self._store.all_records(self._note_prefix(thread_id))
        ]
        expired = [key for key, note in pairs if not note.live_at(now)]
        self._store.delete_many(expired)
        return tuple(
            sorted(
                (note for _, note in pairs if note.live_at(now)),
                key=lambda note: (note.created_at, note.note_id),
            )
        )

    async def put_note(self, note: ThreadNote) -> None:
        self._store.set(
            self._note_key(note),
            _payload(note),
            tags=["note", note.thread_id, note.kind],
            source="v2-memory",
        )
        self._record("note_written", thread_id=note.thread_id, entity_id=note.note_id)

    async def delete_note(self, thread_id: str, note_id: str) -> bool:
        removed = self._store.delete(f"{self._note_prefix(thread_id)}{note_id}")
        if removed:
            self._record("note_deleted", thread_id=thread_id, entity_id=note_id)
        return removed

    async def summary(self, thread_id: str) -> RollingSummary | None:
        value = self._store.get(self._summary_key(thread_id))
        return None if value is None else RollingSummary.model_validate(value)

    async def put_summary(self, summary: RollingSummary) -> None:
        self._store.set(
            self._summary_key(summary.thread_id),
            _payload(summary),
            tags=["summary", summary.thread_id],
            source="v2-memory",
        )
        self._record(
            "summary_written",
            thread_id=summary.thread_id,
            entity_id=str(summary.through_sequence),
        )

    async def proposal(self, proposal_id: str) -> PolicyProposal | None:
        value = self._store.get(self._proposal_key(proposal_id))
        return None if value is None else PolicyProposal.model_validate(value)

    async def put_proposal(self, proposal: PolicyProposal) -> None:
        self._store.set(
            self._proposal_key(proposal.proposal_id),
            _payload(proposal),
            tags=["proposal", proposal.thread_id, proposal.status.value],
            source="v2-memory",
        )
        self._record(
            "proposal_written",
            thread_id=proposal.thread_id,
            entity_id=proposal.proposal_id,
            status=proposal.status.value,
        )

    async def proposals(
        self, thread_id: str | None = None
    ) -> tuple[PolicyProposal, ...]:
        values = (
            PolicyProposal.model_validate(record["value"])
            for _, record in self._store.all_records("v2::proposal::")
        )
        return tuple(
            proposal
            for proposal in values
            if thread_id is None or proposal.thread_id == thread_id
        )

    async def snapshot(self, thread_id: str) -> MemorySnapshot:
        snapshot = MemorySnapshot(
            snapshot_id=self._identifier(),
            thread_id=thread_id,
            created_at=self._clock.now(),
            messages=await self.messages(thread_id),
            notes=await self.notes(thread_id, now=self._clock.now()),
            summary=await self.summary(thread_id),
        )
        self._record(
            "snapshot_created", thread_id=thread_id, entity_id=snapshot.snapshot_id
        )
        return snapshot

    async def restore(self, snapshot: MemorySnapshot) -> None:
        keys = self._store.keys(self._message_prefix(snapshot.thread_id))
        keys.extend(self._store.keys(self._note_prefix(snapshot.thread_id)))
        keys.append(self._summary_key(snapshot.thread_id))
        self._store.delete_many(keys)
        for message in snapshot.messages:
            self._store.set(self._message_key(message), _payload(message))
        for note in snapshot.notes:
            self._store.set(self._note_key(note), _payload(note))
        if snapshot.summary is not None:
            self._store.set(
                self._summary_key(snapshot.thread_id), _payload(snapshot.summary)
            )
        self._record(
            "snapshot_restored", thread_id=snapshot.thread_id, entity_id=snapshot.snapshot_id
        )

    async def fork_thread(
        self, source_thread_id: str, destination_thread_id: str
    ) -> int:
        if self._store.keys(self._message_prefix(destination_thread_id)) or self._store.keys(
            self._note_prefix(destination_thread_id)
        ):
            raise ValueError("destination thread already exists")
        messages = await self.messages(source_thread_id)
        notes = await self.notes(source_thread_id, now=self._clock.now())
        for message in messages:
            copy = message.model_copy(
                update={
                    "message_id": self._identifier(),
                    "thread_id": destination_thread_id,
                }
            )
            self._store.set(self._message_key(copy), _payload(copy))
        for note in notes:
            copy = note.model_copy(
                update={
                    "note_id": self._identifier(),
                    "thread_id": destination_thread_id,
                    "inherited_from": source_thread_id,
                }
            )
            self._store.set(self._note_key(copy), _payload(copy))
        summary = await self.summary(source_thread_id)
        if summary is not None:
            copy = summary.model_copy(update={"thread_id": destination_thread_id})
            self._store.set(self._summary_key(destination_thread_id), _payload(copy))
        self._record(
            "thread_forked",
            thread_id=destination_thread_id,
            entity_id=source_thread_id,
            messages=str(len(messages)),
            notes=str(len(notes)),
        )
        return len(messages) + len(notes)

    async def prune(self, *, before: datetime) -> int:
        targets: list[str] = []
        for key, record in self._store.all_records("v2::message::"):
            if ChatMessage.model_validate(record["value"]).created_at < before:
                targets.append(key)
        for key, record in self._store.all_records("v2::note::"):
            note = ThreadNote.model_validate(record["value"])
            if note.expires_at is not None and note.expires_at < before:
                targets.append(key)
        removed = self._store.delete_many(targets)
        self._record("retention_pruned", removed=str(removed))
        return removed

    async def audit_events(
        self, thread_id: str | None = None
    ) -> tuple[MemoryAuditEvent, ...]:
        values = (
            MemoryAuditEvent.model_validate(record["value"])
            for _, record in self._store.all_records("v2::audit::")
        )
        return tuple(
            event for event in values if thread_id is None or event.thread_id == thread_id
        )

    async def accepted_response(
        self,
        run_id: RunId,
        thread_id: ThreadId,
        item_id: ItemId,
    ) -> ResponseEnvelope | None:
        value = self._store.get(self._accepted_key(run_id, thread_id, item_id))
        return None if value is None else ResponseEnvelope.model_validate(value)

    async def remember_accepted(
        self,
        run_id: RunId,
        thread_id: ThreadId,
        item_id: ItemId,
        response: ResponseEnvelope,
    ) -> None:
        self._store.set(
            self._accepted_key(run_id, thread_id, item_id),
            _payload(response),
            tags=["accepted-response", thread_id],
            source="v2-memory",
        )
        self._record(
            "accepted_response_written",
            thread_id=thread_id,
            entity_id=item_id,
            run_id=run_id,
        )

    async def forget_accepted(
        self,
        run_id: RunId,
        thread_id: ThreadId,
        item_ids: tuple[ItemId, ...],
    ) -> None:
        self._store.delete_many(
            [self._accepted_key(run_id, thread_id, item_id) for item_id in item_ids]
        )

    async def fork_accepted(
        self,
        source_run_id: RunId,
        source_thread_id: ThreadId,
        destination_run_id: RunId,
        destination_thread_id: ThreadId,
        item_ids: tuple[ItemId, ...],
    ) -> None:
        for item_id in item_ids:
            response = await self.accepted_response(
                source_run_id, source_thread_id, item_id
            )
            if response is not None:
                self._store.set(
                    self._accepted_key(
                        destination_run_id, destination_thread_id, item_id
                    ),
                    _payload(response),
                )

    def sync_cdf(self, consumer_id: str) -> Any:
        """Consume the durable Delta CDF tail using the configured audit table."""

        return self._store.sync_cdf(consumer_id)


__all__ = ["DeltaMemoryRepository", "MemoryKVStore"]
