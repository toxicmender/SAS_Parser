"""Shared token budget used by both packing and invocation."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from ..models import VersionedContract
from .models import PromptAssembly, TokenCategory


class BudgetExceededAction(StrEnum):
    REJECT = "reject"
    SHRINK_OPTIONAL_CONTEXT = "shrink_optional_context"


class TokenBudgetIssueCode(StrEnum):
    INPUT_LIMIT = "input_limit"
    SAS_SOURCE_LIMIT = "sas_source_limit"
    INSTRUCTION_LIMIT = "instruction_limit"
    HISTORY_LIMIT = "history_limit"
    RUN_LIMIT = "run_limit"
    INSTRUCTION_SHARE = "instruction_share"


class TokenBudgetIssue(VersionedContract):
    code: TokenBudgetIssueCode
    actual_tokens: int = Field(ge=0)
    limit_tokens: int = Field(ge=0)
    categories: tuple[TokenCategory, ...] = Field(default_factory=tuple)
    message: str


class PromptBudgetDecision(VersionedContract):
    """Preflight result retaining the final prompt but no provider usage."""

    prompt: PromptAssembly
    original_input_tokens: int = Field(ge=0)
    run_tokens_before: int = Field(ge=0)
    projected_run_tokens: int = Field(ge=0)
    violations: tuple[TokenBudgetIssue, ...] = Field(default_factory=tuple)
    warnings: tuple[TokenBudgetIssue, ...] = Field(default_factory=tuple)
    removed_source_ids: tuple[str, ...] = Field(default_factory=tuple)
    summary_compressed: bool = False

    @property
    def allowed(self) -> bool:
        return not self.violations

    @property
    def final_input_tokens(self) -> int:
        return self.prompt.estimated_input_total


class TokenBudgetPolicy(VersionedContract):
    max_input_tokens: int = Field(gt=0)
    reserved_output_tokens: int = Field(ge=0)
    safety_margin_tokens: int = Field(ge=0)
    max_run_tokens: int | None = Field(default=None, gt=0)
    max_sas_source_tokens: int | None = Field(default=None, gt=0)
    max_instruction_tokens: int | None = Field(default=None, gt=0)
    max_history_tokens: int | None = Field(default=None, gt=0)
    instruction_warning_share: float | None = Field(default=None, gt=0, le=1)
    reconciliation_warning_tolerance: int | None = Field(default=None, ge=0)
    on_exceeded: BudgetExceededAction = BudgetExceededAction.REJECT

    @model_validator(mode="after")
    def validate_reserved_capacity(self) -> TokenBudgetPolicy:
        if self.reserved_output_tokens + self.safety_margin_tokens >= self.max_input_tokens:
            raise ValueError("reserved output and safety margin leave no input capacity")
        return self

    @property
    def available_input_tokens(self) -> int:
        return self.max_input_tokens - self.reserved_output_tokens - self.safety_margin_tokens
