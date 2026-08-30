"""Operational command handlers for the v2 CLI composition root."""

from __future__ import annotations

import argparse
import asyncio
import os
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
from sas_migrate.core.tokens import TokenBudgetPolicy as TranslationTokenBudgetPolicy
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


def run_convert_local(args: argparse.Namespace) -> int:
    from sas_migrate.adapters.ai import OpenAICompatibleLLM
    from sas_migrate.adapters.conversion import (
        LocalConversionRequestRepository,
        LocalConversionSourceRepository,
        LocalConversionTranslator,
    )
    from sas_migrate.adapters.credentials import EnvironmentCredentialProvider
    from sas_migrate.application.conversion import ConversionRequest, ConversionWorkflow
    from sas_migrate.config import GatewaySettings

    if not args.source_dir.is_dir():
        raise CommandError(f"source directory does not exist: {args.source_dir}")
    try:
        policy = TranslationTokenBudgetPolicy(
            max_input_tokens=args.max_input_tokens,
            reserved_output_tokens=args.reserved_output_tokens,
            safety_margin_tokens=args.safety_margin_tokens,
            max_run_tokens=args.max_run_tokens,
        )
        settings = GatewaySettings(
            base_url=args.gateway_base_url
            or os.environ.get("SAS_MIGRATE_GATEWAY_BASE_URL"),
            api_key_env=args.api_key_env,
            gateway_version=args.gateway_version,
        )
    except ValueError as exc:
        raise CommandError(f"invalid local conversion settings: {exc}") from exc

    async def execute() -> int:
        credential = None
        if not args.dry_run:
            credentials = EnvironmentCredentialProvider(
                {"gateway": settings.api_key_env}
            )
            credential = await credentials.get("gateway")
            if credential is None:
                raise CommandError(
                    f"gateway credential is not set in {settings.api_key_env}"
                )

        def llm_factory(model: str) -> OpenAICompatibleLLM:
            if credential is None:
                raise RuntimeError("dry-run unexpectedly requested an LLM client")
            return OpenAICompatibleLLM(
                settings=settings,
                credential=credential,
                model=model,
            )

        request = ConversionRequest(
            request_id=args.request_id,
            application_name=args.application_name or args.source_dir.name,
            output_language=args.target,
            status="New",
        )
        outcome = await ConversionWorkflow(
            requests=LocalConversionRequestRepository(request),
            sources=LocalConversionSourceRepository(args.source_dir),
            translator=LocalConversionTranslator(
                output_dir=args.output_dir,
                llm_factory=llm_factory,
                policy=policy,
                max_attempts=args.max_attempts,
            ),
            default_model=args.model,
        ).run(dry_run=args.dry_run)
        print(outcome.model_dump_json(indent=2))
        return outcome.exit_code

    return asyncio.run(execute())


def run_hydrate(args: argparse.Namespace) -> int:
    from sas_migrate.application.hydration import HydrationPlan, HydrationWorkflow

    from .hydration import hydration_delta_sink, hydration_driver_registry

    try:
        plan = HydrationPlan.model_validate_json(_read(args.input))
    except ValueError as exc:
        raise CommandError(f"invalid HydrationPlan in {args.input}: {exc}") from exc
    if not plan.items:
        raise CommandError("hydration plan must contain at least one item")
    if args.batch_rows < 1:
        raise CommandError("--batch-rows must be at least 1")

    report = HydrationWorkflow(
        drivers=hydration_driver_registry(batch_rows=args.batch_rows),
        sink=hydration_delta_sink(
            apply_index_clustering=args.apply_index_clustering,
        ),
    ).run(
        plan,
        dry_run=args.dry_run,
        on_error=args.on_error,
    )
    _emit(
        report.model_dump_json(indent=2) + "\n",
        args.output,
        ReportFormat.JSON,
    )
    return 0 if report.ok else 1


def run_check_sharepoint(args: argparse.Namespace) -> int:
    from sas_migrate.adapters.sharepoint import (
        SharePointGraphTransport,
        SharePointPreflight,
    )
    from sas_migrate.config import (
        ConfigurationError,
        load_settings,
        load_settings_file,
    )
    from sas_migrate.observability import redact_text

    from .sharepoint import sharepoint_token_provider

    try:
        settings = (
            load_settings_file(args.config)
            if args.config is not None
            else load_settings()
        )
    except ConfigurationError as exc:
        raise CommandError(f"invalid v2 settings: {exc}") from exc

    if args.offline:
        report = SharePointPreflight(settings.sharepoint).run(offline=True)
    else:
        try:
            with SharePointGraphTransport(
                settings.sharepoint,
                sharepoint_token_provider(settings),
            ) as transport:
                report = SharePointPreflight(
                    settings.sharepoint,
                    transport,
                ).run()
        except Exception as exc:
            raise CommandError(
                "could not run SharePoint preflight: " + redact_text(str(exc))
            ) from exc

    _emit(
        report.model_dump_json(indent=2) + "\n",
        args.output,
        ReportFormat.JSON,
    )
    return report.exit_code


__all__ = [
    "CommandError",
    "ReportFormat",
    "run_assess",
    "run_check_sharepoint",
    "run_convert_local",
    "run_hydrate",
    "run_validate",
]
