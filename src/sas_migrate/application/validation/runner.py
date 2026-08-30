"""Offline-case runner composed through a run-producer port."""

from __future__ import annotations

from collections.abc import Sequence

from sas_migrate.application.ports.validation import ValidationRunProducer

from .models import TokenBudgetPolicy, ValidationCase, ValidationReport
from .service import ValidationService


class ValidationRunner:
    def __init__(
        self,
        producer: ValidationRunProducer,
        service: ValidationService | None = None,
    ) -> None:
        self._producer = producer
        self._service = service or ValidationService()

    async def run(
        self,
        cases: Sequence[ValidationCase],
        *,
        model: str,
        translation_policy: TokenBudgetPolicy | None = None,
    ) -> ValidationReport:
        if not cases:
            raise ValueError("at least one validation case is required")
        reports = []
        for case in cases:
            produced = await self._producer.produce(case)
            if produced.target is not case.target or produced.run_id != case.case_id:
                raise ValueError("validation producer changed the case identity or target")
            reports.append(
                self._service.validate(
                    produced,
                    model=model,
                    translation_policy=translation_policy,
                )
            )
        first = reports[0]
        return ValidationReport(
            model=model,
            target=first.target,
            results=tuple(result for report in reports for result in report.results),
            target_results=tuple(result for report in reports for result in report.target_results),
            translation_tokens=first.translation_tokens,
            judge_tokens=first.judge_tokens,
            translation_ledger=first.translation_ledger,
            judge_ledger=first.judge_ledger,
        )


__all__ = ["ValidationRunner"]
