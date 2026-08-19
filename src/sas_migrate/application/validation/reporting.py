"""JSON and Markdown validation presenters."""

from __future__ import annotations

from .models import TokenBudgetReport, ValidationReport


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _budget_lines(label: str, budget: TokenBudgetReport) -> list[str]:
    status = "PASS" if budget.compliant else "FAIL"
    lines = [
        f"## {label} token budget",
        "",
        f"- token_budget_compliance: **{status}**",
        f"- current run tokens: {budget.current_run_tokens}",
        f"- recovered tokens: {budget.recovered_tokens}",
        f"- retry overhead tokens: {budget.retry_overhead_tokens}",
        "",
        "| direction | component | tokens |",
        "|---|---|---:|",
    ]
    lines.extend(
        f"| input | {name} | {count} |"
        for name, count in sorted(budget.input_by_category.items())
    )
    lines.extend(
        f"| output | {name} | {count} |"
        for name, count in sorted(budget.output_by_category.items())
    )
    if budget.violations:
        lines.extend(("", "Violations:", ""))
        lines.extend(f"- {_escape(value)}" for value in budget.violations)
    return lines


def render_markdown(report: ValidationReport) -> str:
    lines = [
        f"# Validation report — `{report.model}`",
        "",
        f"- target: `{report.target.value}`",
        f"- run at: {report.created_at.isoformat()}",
        f"- cases: {len(report.results)}",
        f"- aggregate score: **{report.score:.3f}**",
        f"- overall: **{'PASSED' if report.passed else 'FAILED'}**",
        "",
        "| case | metric | score | threshold | status | details |",
        "|---|---|---:|---:|---|---|",
    ]
    for result in report.results:
        for metric in result.metrics:
            status = "skipped" if metric.skipped else ("pass" if metric.passed else "FAIL")
            lines.append(
                f"| {_escape(result.case_id)} | {metric.metric} | {metric.score:.3f} "
                f"| {metric.threshold:.2f} | {status} | {_escape(metric.details)} |"
            )
    lines.extend(("", "## Target resolution validation", ""))
    lines.extend(("| response | resolved target | status | issues |", "|---:|---|---|---|"))
    for index, result in enumerate(report.target_results, start=1):
        issues = "; ".join(issue.message for issue in result.issues) or "none"
        lines.append(
            f"| {index} | {result.resolved_target.value} | "
            f"{'pass' if result.valid else 'FAIL'} | {_escape(issues)} |"
        )
    if not report.target_results:
        lines.append("| - | - | skipped | no structured target result |")
    if report.translation_tokens is not None:
        lines.extend(("", *_budget_lines("Translation", report.translation_tokens)))
    if report.judge_tokens is not None:
        lines.extend(("", *_budget_lines("Judge", report.judge_tokens)))
    return "\n".join(lines) + "\n"


def render_json(report: ValidationReport) -> str:
    return report.to_json()


__all__ = ["render_json", "render_markdown"]
