"""Versioned prompt composition and provider-usage records."""

from __future__ import annotations

from collections import defaultdict
from enum import StrEnum

from pydantic import Field, model_validator

from ..ids import ItemId, RunId, ThreadId
from ..models import ContractModel, VersionedContract
from ..targets.models import TargetId


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class TokenCategory(StrEnum):
    SYSTEM_STATIC = "system_static"
    STRUCTURED_SCHEMA = "structured_schema"
    TARGET_DIRECTIVE = "target_directive"
    SAS_SOURCE = "sas_source"
    BATCH_CONTEXT = "batch_context"
    REFERENCE_GUIDANCE = "reference_guidance"
    PROJECT_INSTRUCTIONS = "project_instructions"
    TASK_POLICY = "task_policy"
    THREAD_NOTES = "thread_notes"
    ROLLING_SUMMARY = "rolling_summary"
    SELECTED_HISTORY = "selected_history"
    RETRY_FEEDBACK = "retry_feedback"
    CHAT_FRAMING = "chat_framing"
    ANALYSIS_OUTPUT = "analysis_output"
    MAPPING_OUTPUT = "mapping_output"
    CODE_OUTPUT = "code_output"
    MARKDOWN_OUTPUT = "markdown_output"
    RISK_OUTPUT = "risk_output"
    RAW_OUTPUT_OVERHEAD = "raw_output_overhead"


class PromptComponent(ContractModel):
    category: TokenCategory
    text: str
    message_role: MessageRole
    token_count: int = Field(ge=0)
    source_id: str | None = None
    cacheable: bool = False
    ephemeral: bool = False


class PromptAssembly(VersionedContract):
    components: tuple[PromptComponent, ...]
    estimator: str
    encoding: str
    approximate: bool = False

    def input_by_category(self) -> dict[TokenCategory, int]:
        totals: defaultdict[TokenCategory, int] = defaultdict(int)
        for component in self.components:
            totals[component.category] += component.token_count
        return dict(totals)

    @property
    def estimated_input_total(self) -> int:
        return sum(component.token_count for component in self.components)


class CallTokenRecord(VersionedContract):
    run_id: RunId
    thread_id: ThreadId
    item_id: ItemId
    attempt: int = Field(ge=1)
    target: TargetId
    estimator: str
    encoding: str
    approximate: bool = False
    estimated_input_by_category: dict[TokenCategory, int]
    estimated_input_total: int = Field(ge=0)
    provider_input_tokens: int | None = Field(default=None, ge=0)
    provider_output_tokens: int | None = Field(default=None, ge=0)
    provider_cache_read_tokens: int | None = Field(default=None, ge=0)
    provider_cache_write_tokens: int | None = Field(default=None, ge=0)
    provider_total_tokens: int | None = Field(default=None, ge=0)
    provider_input_delta: int | None = None
    estimated_output_by_category: dict[TokenCategory, int] = Field(default_factory=dict)
    accepted_attempt: bool = False
    recovered: bool = False

    @model_validator(mode="after")
    def validate_totals(self) -> CallTokenRecord:
        if sum(self.estimated_input_by_category.values()) != self.estimated_input_total:
            raise ValueError("estimated input category counts must sum to the total")
        expected_delta = (
            None
            if self.provider_input_tokens is None
            else self.provider_input_tokens - self.estimated_input_total
        )
        if self.provider_input_delta != expected_delta:
            raise ValueError("provider_input_delta must reconcile provider and estimated input")
        if self.provider_total_tokens is not None:
            if self.provider_input_tokens is None or self.provider_output_tokens is None:
                raise ValueError("provider total requires provider input and output counts")
            expected_total = self.provider_input_tokens + self.provider_output_tokens
            if self.provider_total_tokens != expected_total:
                raise ValueError("provider total must equal provider input plus output")
        return self
