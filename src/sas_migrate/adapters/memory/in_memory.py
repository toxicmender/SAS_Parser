"""Spark-free in-memory conversation and accepted-response repository."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from uuid import uuid4

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


class InMemoryMemoryRepository:
    def __init__(
        self,
        clock: Clock,
        *,
        identifier: Callable[[], str] | None = None,
    ) -> None:
        self._clock = clock
        self._identifier = identifier or (lambda: uuid4().hex)
        self._messages: dict[str, list[ChatMessage]] = {}
        self._policies: dict[str, TaskPolicySnapshot] = {}
        self._notes: dict[str, dict[str, ThreadNote]] = {}
        self._summaries: dict[str, RollingSummary] = {}
        self._proposals: dict[str, PolicyProposal] = {}
        self._accepted: dict[tuple[str, str, str], ResponseEnvelope] = {}
        self._audit: list[MemoryAuditEvent] = []

    def _record(
        self,
        operation: str,
        *,
        thread_id: str | None = None,
        entity_id: str | None = None,
        **details: str,
    ) -> None:
        self._audit.append(
            MemoryAuditEvent(
                event_id=self._identifier(),
                operation=operation,
                occurred_at=self._clock.now(),
                thread_id=thread_id,
                entity_id=entity_id,
                details=details,
            )
        )

    async def append_message(self, message: ChatMessage) -> None:
        messages = self._messages.setdefault(message.thread_id, [])
        expected = messages[-1].sequence + 1 if messages else 1
        if message.sequence != expected:
            raise ValueError(
                f"message sequence {message.sequence} does not follow {expected - 1}"
            )
        if any(existing.message_id == message.message_id for existing in messages):
            raise ValueError("message id already exists in thread")
        messages.append(message)
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
            for message in self._messages.get(thread_id, [])
            if message.sequence > after_sequence
        )

    async def rewind_messages(self, thread_id: str, *, after_sequence: int) -> int:
        messages = self._messages.get(thread_id, [])
        retained = [
            message for message in messages if message.sequence <= after_sequence
        ]
        removed = len(messages) - len(retained)
        self._messages[thread_id] = retained
        summary = self._summaries.get(thread_id)
        if summary is not None and summary.through_sequence > after_sequence:
            self._summaries.pop(thread_id, None)
        self._record(
            "messages_rewound",
            thread_id=thread_id,
            removed=str(removed),
            after_sequence=str(after_sequence),
        )
        return removed

    async def policy(self, task_id: str) -> TaskPolicySnapshot | None:
        return self._policies.get(task_id)

    async def put_policy(self, policy: TaskPolicySnapshot) -> None:
        previous = self._policies.get(policy.task_id)
        if previous is not None and policy.version <= previous.version:
            raise ValueError("policy version must increase")
        self._policies[policy.task_id] = policy
        self._record(
            "policy_written",
            entity_id=policy.task_id,
            version=str(policy.version),
        )

    async def notes(
        self,
        thread_id: str,
        *,
        now: datetime,
    ) -> tuple[ThreadNote, ...]:
        notes = self._notes.get(thread_id, {})
        expired = [note_id for note_id, note in notes.items() if not note.live_at(now)]
        for note_id in expired:
            notes.pop(note_id, None)
        return tuple(
            sorted(notes.values(), key=lambda note: (note.created_at, note.note_id))
        )

    async def put_note(self, note: ThreadNote) -> None:
        self._notes.setdefault(note.thread_id, {})[note.note_id] = note
        self._record(
            "note_written",
            thread_id=note.thread_id,
            entity_id=note.note_id,
        )

    async def delete_note(self, thread_id: str, note_id: str) -> bool:
        removed = self._notes.get(thread_id, {}).pop(note_id, None) is not None
        if removed:
            self._record("note_deleted", thread_id=thread_id, entity_id=note_id)
        return removed

    async def summary(self, thread_id: str) -> RollingSummary | None:
        return self._summaries.get(thread_id)

    async def put_summary(self, summary: RollingSummary) -> None:
        self._summaries[summary.thread_id] = summary
        self._record(
            "summary_written",
            thread_id=summary.thread_id,
            entity_id=str(summary.through_sequence),
        )

    async def proposal(self, proposal_id: str) -> PolicyProposal | None:
        return self._proposals.get(proposal_id)

    async def put_proposal(self, proposal: PolicyProposal) -> None:
        self._proposals[proposal.proposal_id] = proposal
        self._record(
            "proposal_written",
            thread_id=proposal.thread_id,
            entity_id=proposal.proposal_id,
            status=proposal.status.value,
        )

    async def proposals(
        self, thread_id: str | None = None
    ) -> tuple[PolicyProposal, ...]:
        return tuple(
            proposal
            for proposal in self._proposals.values()
            if thread_id is None or proposal.thread_id == thread_id
        )

    async def snapshot(self, thread_id: str) -> MemorySnapshot:
        snapshot = MemorySnapshot(
            snapshot_id=self._identifier(),
            thread_id=thread_id,
            created_at=self._clock.now(),
            messages=tuple(self._messages.get(thread_id, [])),
            notes=tuple(self._notes.get(thread_id, {}).values()),
            summary=self._summaries.get(thread_id),
        )
        self._record(
            "snapshot_created",
            thread_id=thread_id,
            entity_id=snapshot.snapshot_id,
        )
        return snapshot

    async def restore(self, snapshot: MemorySnapshot) -> None:
        self._messages[snapshot.thread_id] = list(snapshot.messages)
        self._notes[snapshot.thread_id] = {
            note.note_id: note for note in snapshot.notes
        }
        if snapshot.summary is None:
            self._summaries.pop(snapshot.thread_id, None)
        else:
            self._summaries[snapshot.thread_id] = snapshot.summary
        self._record(
            "snapshot_restored",
            thread_id=snapshot.thread_id,
            entity_id=snapshot.snapshot_id,
        )

    async def fork_thread(
        self, source_thread_id: str, destination_thread_id: str
    ) -> int:
        if (
            destination_thread_id in self._messages
            or destination_thread_id in self._notes
        ):
            raise ValueError("destination thread already exists")
        messages = self._messages.get(source_thread_id, [])
        copied_messages = [
            message.model_copy(
                update={
                    "message_id": self._identifier(),
                    "thread_id": destination_thread_id,
                }
            )
            for message in messages
        ]
        self._messages[destination_thread_id] = copied_messages
        source_notes = self._notes.get(source_thread_id, {})
        copied_notes = {
            copy.note_id: copy
            for note in source_notes.values()
            for copy in (
                note.model_copy(
                    update={
                        "note_id": self._identifier(),
                        "thread_id": destination_thread_id,
                        "inherited_from": source_thread_id,
                    }
                ),
            )
        }
        self._notes[destination_thread_id] = copied_notes
        summary = self._summaries.get(source_thread_id)
        if summary is not None:
            self._summaries[destination_thread_id] = summary.model_copy(
                update={"thread_id": destination_thread_id}
            )
        self._record(
            "thread_forked",
            thread_id=destination_thread_id,
            entity_id=source_thread_id,
            messages=str(len(copied_messages)),
            notes=str(len(copied_notes)),
        )
        return len(copied_messages) + len(copied_notes)

    async def prune(self, *, before: datetime) -> int:
        removed = 0
        for thread_id, messages in self._messages.items():
            retained = [message for message in messages if message.created_at >= before]
            removed += len(messages) - len(retained)
            self._messages[thread_id] = retained
        for thread_id, notes in self._notes.items():
            expired = [
                note_id
                for note_id, note in notes.items()
                if note.expires_at is not None and note.expires_at < before
            ]
            for note_id in expired:
                notes.pop(note_id, None)
            removed += len(expired)
        self._record("retention_pruned", removed=str(removed))
        return removed

    async def audit_events(
        self, thread_id: str | None = None
    ) -> tuple[MemoryAuditEvent, ...]:
        return tuple(
            event
            for event in self._audit
            if thread_id is None or event.thread_id == thread_id
        )

    async def accepted_response(
        self,
        run_id: RunId,
        thread_id: ThreadId,
        item_id: ItemId,
    ) -> ResponseEnvelope | None:
        return self._accepted.get((run_id, thread_id, item_id))

    async def remember_accepted(
        self,
        run_id: RunId,
        thread_id: ThreadId,
        item_id: ItemId,
        response: ResponseEnvelope,
    ) -> None:
        self._accepted[(run_id, thread_id, item_id)] = response
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
        for item_id in item_ids:
            self._accepted.pop((run_id, thread_id, item_id), None)

    async def fork_accepted(
        self,
        source_run_id: RunId,
        source_thread_id: ThreadId,
        destination_run_id: RunId,
        destination_thread_id: ThreadId,
        item_ids: tuple[ItemId, ...],
    ) -> None:
        for item_id in item_ids:
            response = self._accepted.get((source_run_id, source_thread_id, item_id))
            if response is not None:
                self._accepted[(destination_run_id, destination_thread_id, item_id)] = (
                    response
                )


__all__ = ["InMemoryMemoryRepository"]
