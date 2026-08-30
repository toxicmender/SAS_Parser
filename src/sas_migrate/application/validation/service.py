"""Validation use case composed only from application and core contracts."""

from __future__ import annotations

from collections.abc import Sequence

from sas_migrate.core.tokens import TokenCallLedger

from .budgeting import build_token_budget_report, token_budget_compliance
from .evaluator import Evaluator
from .metrics import ValidationMetric
from .models import EvaluationRun, TokenBudgetPolicy, ValidationReport


class ValidationService:
    def __init__(self, metrics: Sequence[ValidationMetric] | None = None) -> None:
        self._evaluator = Evaluator(metrics)

    def validate(
        self,
        run: EvaluationRun,
        *,
        model: str,
        translation_ledger: TokenCallLedger | None = None,
        judge_ledger: TokenCallLedger | None = None,
        translation_policy: TokenBudgetPolicy | None = None,
        judge_policy: TokenBudgetPolicy | None = None,
    ) -> ValidationReport:
        translation_ledger = translation_ledger or TokenCallLedger()
        judge_ledger = judge_ledger or TokenCallLedger()
        translation_budget = (
            build_token_budget_report(translation_ledger, translation_policy)
            if translation_policy is not None
            else None
        )
        judge_budget = (
            build_token_budget_report(judge_ledger, judge_policy)
            if judge_policy is not None
            else None
        )
        result = self._evaluator.evaluate(run)
        budget_metrics = tuple(
            token_budget_compliance(budget)
            for budget in (translation_budget, judge_budget)
            if budget is not None
        )
        if budget_metrics:
            result = result.model_copy(
                update={"metrics": (*result.metrics, *budget_metrics)}
            )
        return ValidationReport(
            model=model,
            target=run.target,
            results=(result,),
            target_results=tuple(
                unit.target_validation
                for unit in run.units
                if unit.target_validation is not None
            ),
            translation_tokens=translation_budget,
            judge_tokens=judge_budget,
            translation_ledger=translation_ledger,
            judge_ledger=judge_ledger,
        )


__all__ = ["ValidationService"]
