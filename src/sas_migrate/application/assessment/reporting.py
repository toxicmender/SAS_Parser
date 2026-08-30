"""Assessment JSON and Markdown presenters."""

from __future__ import annotations

from .models import AssessmentReport


def render_json(report: AssessmentReport) -> str:
    return report.to_json()


def render_markdown(report: AssessmentReport) -> str:
    lines = [
        f"# Migration assessment — `{report.target.value}`",
        "",
        f"- profile: `{report.profile}`",
        f"- files: {len(report.files)}",
        f"- total story points: {report.total_story_points:g}",
        "",
        "| file | size | points | tier | parity | raw score |",
        "|---|---|---:|---|---|---:|",
    ]
    lines.extend(
        f"| {file.source_id} | {file.size.value} | {file.story_points:g} | "
        f"{file.tier.value} | {file.parity.value} | {file.raw_score:g} |"
        for file in report.files
    )
    lines.extend(("", "## Cross-file dependencies", "", "| producer | consumer | dataset |", "|---|---|---|"))
    lines.extend(
        f"| {edge.producer} | {edge.consumer} | {edge.dataset} |"
        for edge in report.dependencies
    )
    if not report.dependencies:
        lines.append("| - | - | none |")
    if report.review is not None:
        lines.extend(("", "## LLM review", "", report.review))
    return "\n".join(lines) + "\n"


__all__ = ["render_json", "render_markdown"]
