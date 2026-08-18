"""Conversation, policy, note, summary, proposal, and audit persistence ports."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from .memory import MemoryPort

if TYPE_CHECKING:
    from sas_migrate.application.memory.models import (
        ChatMessage,
        MemoryAuditEvent,
        MemoryCandidate,
        MemorySnapshot,
        PolicyProposal,
        RollingSummary,
        TaskPolicySnapshot,
        ThreadNote,
    )


class ConversationMemoryRepository(MemoryPort, Protocol):
    async def append_message(self, message: ChatMessage) -> None: ...

    async def messages(
        self,
        thread_id: str,
        *,
        after_sequence: int = 0,
    ) -> tuple[ChatMessage, ...]: ...

    async def rewind_messages(self, thread_id: str, *, after_sequence: int) -> int: ...

    async def policy(self, task_id: str) -> TaskPolicySnapshot | None: ...

    async def put_policy(self, policy: TaskPolicySnapshot) -> None: ...

    async def notes(
        self, thread_id: str, *, now: datetime
    ) -> tuple[ThreadNote, ...]: ...

    async def put_note(self, note: ThreadNote) -> None: ...

    async def delete_note(self, thread_id: str, note_id: str) -> bool: ...

    async def summary(self, thread_id: str) -> RollingSummary | None: ...

    async def put_summary(self, summary: RollingSummary) -> None: ...

    async def proposal(self, proposal_id: str) -> PolicyProposal | None: ...

    async def put_proposal(self, proposal: PolicyProposal) -> None: ...

    async def proposals(
        self, thread_id: str | None = None
    ) -> tuple[PolicyProposal, ...]: ...

    async def snapshot(self, thread_id: str) -> MemorySnapshot: ...

    async def restore(self, snapshot: MemorySnapshot) -> None: ...

    async def fork_thread(
        self, source_thread_id: str, destination_thread_id: str
    ) -> int: ...

    async def prune(self, *, before: datetime) -> int: ...

    async def audit_events(
        self, thread_id: str | None = None
    ) -> tuple[MemoryAuditEvent, ...]: ...


class MemoryClassifier(Protocol):
    async def extract(
        self,
        user_content: str,
        assistant_content: str,
    ) -> tuple[MemoryCandidate, ...]: ...


__all__ = ["ConversationMemoryRepository", "MemoryClassifier"]
