"""Provider-neutral contracts for validation runs and reports."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import Field, computed_field, model_validator

from sas_migrate.core.models import ContractModel, VersionedContract
from sas_migrate.core.targets import ResponseValidationResult, TargetId
from sas_migrate.core.tokens import TokenCallLedger


class ValidationUnit(ContractModel):
    """One response and the evidence needed to score it."""

    unit_id: str = Field(min_length=1)
    source: str = ""
    prompt: str = ""
    response: str = ""
    input_datasets: tuple[str, ...] = ()
    output_datasets: tuple[str, ...] = ()
    retrieval_context: tuple[str, ...] = ()
    target_validation: ResponseValidationResult | None = None


class ValidationCase(ContractModel):
    case_id: str = Field(min_length=1)
    target: TargetId
    sas_source: str
    description: str = ""
    reference_translation: str | None = None
    required_terms: tuple[str, ...] = ()
    prompt_instructions: tuple[str, ...] = ()


class EvaluationRun(VersionedContract):
    """Conversation-sized validation input, independent of its producer."""

    run_id: str = Field(min_length=1)
    target: TargetId
    units: tuple[ValidationUnit, ...]
    expected_units: int | None = Field(default=None, ge=0)
    required_terms: tuple[str, ...] = ()
    reference_translation: str | None = None
    prompt_instructions: tuple[str, ...] = ()
    summary: str | None = None
    summary_source: str | None = None
    task_policy: tuple[str, ...] = ()
    thread_notes: tuple[str, ...] = ()
    foreign_notes: tuple[str, ...] = ()
    expected_memories: tuple[str, ...] = ()
    extracted_memories: tuple[str, ...] = ()

    @property
    def unit_count(self) -> int:
        return self.expected_units if self.expected_units is not None else len(self.units)

    @property
    def joined_responses(self) -> str:
        return "\n\n".join(unit.response for unit in self.units)


class MetricResult(ContractModel):
    metric: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    passed: bool
    skipped: bool = False
    details: str = ""

    @model_validator(mode="after")
    def validate_skip(self) -> MetricResult:
        if self.skipped and not self.passed:
            raise ValueError("a skipped metric must pass")
        return self


class CaseResult(ContractModel):
    case_id: str
    item_count: int = Field(ge=0)
    metrics: tuple[MetricResult, ...]

    @computed_field
    @property
    def score(self) -> float:
        scored = [metric.score for metric in self.metrics if not metric.skipped]
        return sum(scored) / len(scored) if scored else 1.0

    @computed_field
    @property
    def passed(self) -> bool:
        return all(metric.passed for metric in self.metrics)


class TokenBudgetPolicy(ContractModel):
    max_input_tokens_per_call: int | None = Field(default=None, gt=0)
    max_output_tokens_per_call: int | None = Field(default=None, gt=0)
    max_run_tokens: int | None = Field(default=None, gt=0)


class TokenBudgetReport(ContractModel):
    input_by_category: dict[str, int]
    output_by_category: dict[str, int]
    estimated_input_tokens: int = Field(ge=0)
    estimated_output_tokens: int = Field(ge=0)
    current_run_tokens: int = Field(ge=0)
    recovered_tokens: int = Field(ge=0)
    retry_overhead_tokens: int = Field(ge=0)
    compliant: bool
    violations: tuple[str, ...] = ()


class ValidationReport(VersionedContract):
    model: str
    target: TargetId
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    results: tuple[CaseResult, ...]
    target_results: tuple[ResponseValidationResult, ...] = ()
    translation_tokens: TokenBudgetReport | None = None
    judge_tokens: TokenBudgetReport | None = None
    translation_ledger: TokenCallLedger = Field(default_factory=TokenCallLedger)
    judge_ledger: TokenCallLedger = Field(default_factory=TokenCallLedger)

    @computed_field
    @property
    def score(self) -> float:
        return (
            sum(result.score for result in self.results) / len(self.results)
            if self.results
            else 0.0
        )

    @computed_field
    @property
    def passed(self) -> bool:
        budgets = tuple(
            budget
            for budget in (self.translation_tokens, self.judge_tokens)
            if budget is not None
        )
        return (
            bool(self.results)
            and all(result.passed for result in self.results)
            and all(result.valid for result in self.target_results)
            and all(budget.compliant for budget in budgets)
        )

    def to_json(self) -> str:
        """Serialize stored fields only; computed presentation fields re-derive."""

        return self.model_dump_json(exclude_computed_fields=True)


__all__ = [
    "CaseResult",
    "EvaluationRun",
    "MetricResult",
    "TokenBudgetPolicy",
    "TokenBudgetReport",
    "ValidationCase",
    "ValidationReport",
    "ValidationUnit",
]
