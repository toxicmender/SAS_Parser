"""Mandatory response-target validation port."""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Protocol

from sas_migrate.core.responses import TranslationDocument
from sas_migrate.core.targets import ResolvedTarget
from sas_migrate.core.targets.validation import ResponseValidationResult

if TYPE_CHECKING:
    from sas_migrate.application.validation.models import ValidationReport


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
