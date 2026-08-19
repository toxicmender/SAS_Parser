"""Mandatory response-target validation port."""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Protocol

from pydantic import Field

from sas_migrate.core.models import ContractModel
from sas_migrate.core.responses import TranslationDocument
from sas_migrate.core.targets import ResolvedTarget
from sas_migrate.core.targets.validation import ResponseValidationResult

if TYPE_CHECKING:
    from sas_migrate.application.validation.models import (
        EvaluationRun,
        ValidationCase,
        ValidationReport,
    )


class ResponseValidator(Protocol):
    def validate(
        self,
        document: TranslationDocument,
        target: ResolvedTarget,
        *,
        known_chunk_ids: Collection[str],
    ) -> ResponseValidationResult: ...


class ValidationReportRepository(Protocol):
    async def append(self, report: ValidationReport) -> str: ...

    async def load(self) -> tuple[ValidationReport, ...]: ...


class JudgeRequest(ContractModel):
    metric: str
    prompt: str = ""
    response: str
    contexts: tuple[str, ...] = ()
    instructions: tuple[str, ...] = ()


class JudgeVerdict(ContractModel):
    score: float = Field(ge=0.0, le=1.0)
    details: str = ""


class ValidationJudge(Protocol):
    def evaluate(self, request: JudgeRequest) -> JudgeVerdict: ...


class ValidationRunProducer(Protocol):
    async def produce(self, case: ValidationCase) -> EvaluationRun: ...
