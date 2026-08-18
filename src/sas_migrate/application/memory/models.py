"""Versioned memory, policy, note, summary, snapshot, and audit contracts."""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from sas_migrate.core.ids import ItemId, ThreadId
from sas_migrate.core.models import ContractModel, VersionedContract
from sas_migrate.core.tokens import PromptComponentDraft


class ChatRole(StrEnum):
    HUMAN = "human"
    ASSISTANT = "assistant"


class ProposalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class MemoryScope(StrEnum):
    TEMPORARY = "temporary"
    PERMANENT = "permanent"


class ChatMessage(VersionedContract):
    message_id: str = Field(min_length=1)
    thread_id: ThreadId
    chat_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    role: ChatRole
    content: str = Field(min_length=1)
    created_at: datetime
    item_id: ItemId | None = None
    ephemeral: bool = False

    @model_validator(mode="after")
    def reject_ephemeral_history(self) -> ChatMessage:
        if self.ephemeral:
            raise ValueError("ephemeral content cannot be persisted as chat history")
        return self


class PolicyInstruction(ContractModel):
    instruction_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    overridable: bool = True
    source: str = "operator"


class TaskPolicySnapshot(VersionedContract):
    task_id: str = Field(min_length=1)
    version: int = Field(ge=0)
    instructions: tuple[PolicyInstruction, ...] = Field(default_factory=tuple)
    updated_at: datetime

    @property
    def fingerprint(self) -> str:
        rendered = "\n".join(
            f"{instruction.text}\0{instruction.overridable}"
            for instruction in self.instructions
        )
        return hashlib.sha256(rendered.encode()).hexdigest() if rendered else ""


class ThreadNote(VersionedContract):
    note_id: str = Field(min_length=1)
    thread_id: ThreadId
    text: str = Field(min_length=1)
    kind: str = "note"
    source: str = "operator"
    created_at: datetime
    expires_at: datetime | None = None
    inherited_from: ThreadId | None = None

    def live_at(self, now: datetime) -> bool:
        return self.expires_at is None or self.expires_at > now


class RollingSummary(VersionedContract):
    thread_id: ThreadId
    content: str = Field(min_length=1)
    through_sequence: int = Field(ge=0)
    token_count: int = Field(ge=0)
    updated_at: datetime


class PolicyProposal(VersionedContract):
    proposal_id: str = Field(min_length=1)
    thread_id: ThreadId
    task_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    reason: str = ""
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: datetime
    resolved_at: datetime | None = None

    @model_validator(mode="after")
    def validate_resolution(self) -> PolicyProposal:
        if self.status is ProposalStatus.PENDING and self.resolved_at is not None:
            raise ValueError("pending proposal cannot have resolved_at")
        if self.status is not ProposalStatus.PENDING and self.resolved_at is None:
            raise ValueError("resolved proposal requires resolved_at")
        return self


class MemoryCandidate(ContractModel):
    text: str = Field(min_length=1)
    scope: MemoryScope
    kind: str = "note"
    reason: str = ""
    ttl_seconds: int | None = Field(default=None, gt=0)


class MemoryAuditEvent(VersionedContract):
    event_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    occurred_at: datetime
    thread_id: ThreadId | None = None
    entity_id: str | None = None
    details: dict[str, str] = Field(default_factory=dict)


class MemorySnapshot(VersionedContract):
    snapshot_id: str = Field(min_length=1)
    thread_id: ThreadId
    created_at: datetime
    messages: tuple[ChatMessage, ...] = Field(default_factory=tuple)
    notes: tuple[ThreadNote, ...] = Field(default_factory=tuple)
    summary: RollingSummary | None = None


class MemoryContextResult(VersionedContract):
    thread_id: ThreadId
    components: tuple[PromptComponentDraft, ...]
    selected_message_ids: tuple[str, ...] = Field(default_factory=tuple)
    selected_history_tokens: int = Field(ge=0)
    policy_fingerprint: str = ""
    note_count: int = Field(ge=0)


class ExtractionResult(VersionedContract):
    applied_notes: tuple[ThreadNote, ...] = Field(default_factory=tuple)
    pending_proposals: tuple[PolicyProposal, ...] = Field(default_factory=tuple)


__all__ = [
    "ChatMessage",
    "ChatRole",
    "ExtractionResult",
    "MemoryAuditEvent",
    "MemoryCandidate",
    "MemoryContextResult",
    "MemoryScope",
    "MemorySnapshot",
    "PolicyInstruction",
    "PolicyProposal",
    "ProposalStatus",
    "RollingSummary",
    "TaskPolicySnapshot",
    "ThreadNote",
]
