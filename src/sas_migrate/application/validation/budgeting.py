"""Token-budget validation over immutable attempt-level ledgers."""

from __future__ import annotations

from sas_migrate.core.tokens import TokenCallLedger

from .models import MetricResult, TokenBudgetPolicy, TokenBudgetReport


def build_token_budget_report(
    ledger: TokenCallLedger,
    policy: TokenBudgetPolicy,
) -> TokenBudgetReport:
    violations: list[str] = []
    for record in ledger.records:
        key = f"{record.item_id}/attempt-{record.attempt}"
        if (
            policy.max_input_tokens_per_call is not None
            and record.estimated_input_total > policy.max_input_tokens_per_call
        ):
            violations.append(
                f"{key} input {record.estimated_input_total} exceeds "
                f"{policy.max_input_tokens_per_call}"
            )
        if (
            policy.max_output_tokens_per_call is not None
            and record.estimated_output_total > policy.max_output_tokens_per_call
        ):
            violations.append(
                f"{key} output {record.estimated_output_total} exceeds "
                f"{policy.max_output_tokens_per_call}"
            )
    if (
        policy.max_run_tokens is not None
        and ledger.current_run_total_tokens > policy.max_run_tokens
    ):
        violations.append(
            f"run total {ledger.current_run_total_tokens} exceeds {policy.max_run_tokens}"
        )
    input_totals = ledger.input_by_category()
    output_totals = ledger.output_by_category()
    return TokenBudgetReport(
        input_by_category={key.value: value for key, value in input_totals.items()},
        output_by_category={key.value: value for key, value in output_totals.items()},
        estimated_input_tokens=sum(input_totals.values()),
        estimated_output_tokens=sum(output_totals.values()),
        current_run_tokens=ledger.current_run_total_tokens,
        recovered_tokens=ledger.recovered_total_tokens,
        retry_overhead_tokens=ledger.retry_overhead_tokens,
        compliant=not violations,
        violations=tuple(violations),
    )


def token_budget_compliance(report: TokenBudgetReport) -> MetricResult:
    return MetricResult(
        metric="token_budget_compliance",
        score=1.0 if report.compliant else 0.0,
        threshold=1.0,
        passed=report.compliant,
        details=(
            "all configured token limits satisfied"
            if report.compliant
            else "; ".join(report.violations)
        ),
    )


__all__ = ["build_token_budget_report", "token_budget_compliance"]
