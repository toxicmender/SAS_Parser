"""Operational command handlers for the v2 CLI composition root."""

from __future__ import annotations

import argparse
import sys
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from sas_migrate.adapters.assessment import (
    DirectoryAssessmentProfileRepository,
    PackageAssessmentProfileRepository,
)
from sas_migrate.application.assessment import (
    AssessmentService,
    AssessmentUnit,
)
from sas_migrate.application.assessment import (
    render_json as render_assessment_json,
)
from sas_migrate.application.assessment import (
    render_markdown as render_assessment_markdown,
)
from sas_migrate.application.validation import (
    EvaluationRun,
    TokenBudgetPolicy,
    ValidationService,
)
from sas_migrate.application.validation import (
    render_json as render_validation_json,
)
from sas_migrate.application.validation import (
    render_markdown as render_validation_markdown,
)
from sas_migrate.core.targets import TargetId
from sas_migrate.core.tokens import TokenCallLedger


class CommandError(ValueError):
    """Expected command-input or output error suitable for an operator."""


class ReportFormat(StrEnum):
    JSON = "json"
    MARKDOWN = "markdown"
    PDF = "pdf"


_ASSESSMENT_UNITS = TypeAdapter(tuple[AssessmentUnit, ...])


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CommandError(f"could not read {path}: {exc}") from exc


def _optional_model[ModelT: BaseModel](
    path: Path | None,
    model: type[ModelT],
) -> ModelT | None:
    if path is None:
        return None
    try:
        return model.model_validate_json(_read(path))
    except ValueError as exc:
        raise CommandError(f"invalid {model.__name__} in {path}: {exc}") from exc


def _emit(content: str | bytes, output: Path | None, report_format: ReportFormat) -> None:
    if isinstance(content, bytes) and output is None:
        raise CommandError("PDF output requires --output PATH")
    try:
        if output is not None:
            if isinstance(content, bytes):
                output.write_bytes(content)
            else:
                output.write_text(content, encoding="utf-8")
        elif report_format is ReportFormat.JSON:
            print(content)
        else:
            assert isinstance(content, str)
            sys.stdout.write(content)
    except OSError as exc:
        raise CommandError(f"could not write report: {exc}") from exc


def run_assess(args: argparse.Namespace) -> int:
    try:
        units = _ASSESSMENT_UNITS.validate_json(_read(args.input))
    except ValueError as exc:
        raise CommandError(f"invalid assessment units in {args.input}: {exc}") from exc
    if not units:
        raise CommandError("assessment input must contain at least one unit")

    profiles = (
        DirectoryAssessmentProfileRepository(args.profiles)
        if args.profiles is not None
        else PackageAssessmentProfileRepository()
    )
    report = AssessmentService(profiles).assess(units, TargetId(args.target))
    report_format = ReportFormat(args.output_format)
    if report_format is ReportFormat.JSON:
        content: str | bytes = render_assessment_json(report)
    elif report_format is ReportFormat.MARKDOWN:
        content = render_assessment_markdown(report)
    else:
        from sas_migrate.adapters.assessment import render_pdf

        content = render_pdf(report)
    _emit(content, args.output, report_format)
    return 0


def run_validate(args: argparse.Namespace) -> int:
    try:
        run = EvaluationRun.model_validate_json(_read(args.input))
    except ValueError as exc:
        raise CommandError(f"invalid EvaluationRun in {args.input}: {exc}") from exc

    translation_ledger = _optional_model(args.translation_ledger, TokenCallLedger)
    judge_ledger = _optional_model(args.judge_ledger, TokenCallLedger)
    translation_policy = _optional_model(args.translation_policy, TokenBudgetPolicy)
    judge_policy = _optional_model(args.judge_policy, TokenBudgetPolicy)
    report = ValidationService().validate(
        run,
        model=args.model,
        translation_ledger=translation_ledger,
        judge_ledger=judge_ledger,
        translation_policy=translation_policy,
        judge_policy=judge_policy,
    )
    report_format = ReportFormat(args.output_format)
    if report_format is ReportFormat.JSON:
        content: str | bytes = render_validation_json(report)
    elif report_format is ReportFormat.MARKDOWN:
        content = render_validation_markdown(report)
    else:
        from sas_migrate.adapters.validation import render_pdf

        content = render_pdf(report)
    _emit(content, args.output, report_format)
    return 0 if report.passed else 1


__all__ = [
    "CommandError",
    "ReportFormat",
    "run_assess",
    "run_validate",
]
