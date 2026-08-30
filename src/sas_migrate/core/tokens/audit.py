"""Redacted token-audit wire contracts."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, StringConstraints, model_validator

from ..ids import ItemId, RunId, ThreadId
from ..models import ContractModel, VersionedContract
from ..targets import TargetId
from .models import CallTokenRecord, MessageRole, TokenCategory
from .policy import TokenBudgetIssue

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class PromptComponentAudit(ContractModel):
    category: TokenCategory
    message_role: MessageRole
    token_count: int = Field(ge=0)
    text_sha256: Sha256
    source_id_sha256: Sha256 | None = None
    cacheable: bool = False
    ephemeral: bool = False


class TokenBudgetAudit(ContractModel):
    allowed: bool
    original_input_tokens: int = Field(ge=0)
    final_input_tokens: int = Field(ge=0)
    run_tokens_before: int = Field(ge=0)
    projected_run_tokens: int = Field(ge=0)
    violations: tuple[TokenBudgetIssue, ...] = Field(default_factory=tuple)
    warnings: tuple[TokenBudgetIssue, ...] = Field(default_factory=tuple)
    removed_component_count: int = Field(default=0, ge=0)
    summary_compressed: bool = False

    @model_validator(mode="after")
    def validate_allowed_state(self) -> TokenBudgetAudit:
        if self.allowed == bool(self.violations):
            raise ValueError("budget audit allowed state must match its violations")
        return self


class TokenAuditArtifact(VersionedContract):
    run_id: RunId
    thread_id: ThreadId
    item_id: ItemId
    attempt: int = Field(ge=1)
    target: TargetId
    budget: TokenBudgetAudit
    components: tuple[PromptComponentAudit, ...]
    call_record: CallTokenRecord | None = None

    @model_validator(mode="after")
    def validate_call_record(self) -> TokenAuditArtifact:
        if not self.budget.allowed and self.call_record is not None:
            raise ValueError("a rejected preflight cannot contain provider usage")
        if self.call_record is None:
            return self
        key = (
            self.call_record.run_id,
            self.call_record.thread_id,
            self.call_record.item_id,
            self.call_record.attempt,
            self.call_record.target,
        )
        expected = (
            self.run_id,
            self.thread_id,
            self.item_id,
            self.attempt,
            self.target,
        )
        if key != expected:
            raise ValueError("token audit and call record attempt keys differ")
        return self


__all__ = [
    "PromptComponentAudit",
    "TokenAuditArtifact",
    "TokenBudgetAudit",
]
