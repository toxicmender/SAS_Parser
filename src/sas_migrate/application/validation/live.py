"""Inline validation entry point for an already-produced response."""

from __future__ import annotations

from sas_migrate.core.targets import ResponseValidationResult, TargetId

from .models import EvaluationRun, ValidationReport, ValidationUnit
from .service import ValidationService


def validate_response(
    *,
    run_id: str,
    unit_id: str,
    target: TargetId,
    model: str,
    response: str,
    prompt: str = "",
    source: str = "",
    target_validation: ResponseValidationResult | None = None,
    service: ValidationService | None = None,
) -> ValidationReport:
    run = EvaluationRun(
        run_id=run_id,
        target=target,
        units=(
            ValidationUnit(
                unit_id=unit_id,
                prompt=prompt,
                response=response,
                source=source,
                target_validation=target_validation,
            ),
        ),
    )
    return (service or ValidationService()).validate(run, model=model)


__all__ = ["validate_response"]
