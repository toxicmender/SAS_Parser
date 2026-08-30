"""Normalize, validate, retry, and gate one provider translation response."""

from __future__ import annotations

from collections.abc import Collection
from typing import Protocol

from pydantic import Field, model_validator

from sas_migrate.application.ports import ProviderResponse
from sas_migrate.core.errors import ResponseContractError
from sas_migrate.core.ids import ItemId
from sas_migrate.core.models import ContractModel, VersionedContract
from sas_migrate.core.responses import (
    ResponseEnvelope,
    ResponseMode,
    ResponseTargetValidator,
    TranslationCell,
    TranslationDocument,
    normalize_raw_response,
)
from sas_migrate.core.runs import ItemStatus
from sas_migrate.core.targets import ResolvedTarget
from sas_migrate.core.targets.validation import TargetValidationIssue


class AttemptProvider(Protocol):
    async def __call__(
        self,
        attempt: int,
        feedback: tuple[TargetValidationIssue, ...],
        /,
    ) -> ProviderResponse: ...


class ResponseAttempt(ContractModel):
    attempt: int = Field(ge=1)
    envelope: ResponseEnvelope


class ResponseAcceptanceOutcome(VersionedContract):
    """Auditable item result; only accepted results expose runnable code."""

    item_id: ItemId
    status: ItemStatus
    attempts: tuple[ResponseAttempt, ...]
    accepted_document: TranslationDocument | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> ResponseAcceptanceOutcome:
        if not self.attempts:
            raise ValueError("response acceptance outcome requires at least one attempt")
        last = self.attempts[-1].envelope
        if self.status is ItemStatus.ACCEPTED:
            if self.accepted_document is None or not last.validation.valid:
                raise ValueError("accepted outcome requires a valid final document")
            if self.error is not None:
                raise ValueError("accepted outcome cannot carry an error")
        elif self.status is ItemStatus.FAILED:
            if self.accepted_document is not None or last.validation.valid:
                raise ValueError("failed outcome cannot publish a valid document")
            if not self.error:
                raise ValueError("failed outcome must explain the failure")
        else:
            raise ValueError("response outcome status must be accepted or failed")
        return self

    @property
    def runnable_code_cells(self) -> tuple[TranslationCell, ...]:
        if self.accepted_document is None:
            return ()
        return self.accepted_document.code_cells

    def canonical_markdown(self) -> str:
        if self.accepted_document is None:
            raise ResponseContractError("failed response has no publishable Markdown")
        return self.accepted_document.to_markdown()


class ResponseAcceptanceService:
    """Apply the same normalization and validation to every response attempt."""

    def __init__(self, validator: ResponseTargetValidator | None = None) -> None:
        self._validator = validator or ResponseTargetValidator()

    def envelope(
        self,
        response: ProviderResponse,
        target: ResolvedTarget,
        *,
        known_chunk_ids: Collection[str],
    ) -> ResponseEnvelope:
        if response.structured_document is not None:
            document = response.structured_document
            mode = ResponseMode.STRUCTURED
            structured_error = None
            normalization_issues: tuple[TargetValidationIssue, ...] = ()
        else:
            normalized = normalize_raw_response(response.raw_message, target)
            document = normalized.document
            mode = ResponseMode.RAW_FALLBACK
            structured_error = (
                response.structured_error
                or "provider returned no target-bearing structured document"
            )
            normalization_issues = normalized.issues

        validation = self._validator.validate(
            document,
            target,
            known_chunk_ids=known_chunk_ids,
            normalization_issues=normalization_issues,
        )
        return ResponseEnvelope(
            mode=mode,
            raw_message=response.raw_message,
            document=document,
            structured_error=structured_error,
            resolved_target=target,
            validation=validation,
        )

    async def accept(
        self,
        *,
        item_id: ItemId,
        target: ResolvedTarget,
        known_chunk_ids: Collection[str],
        request_attempt: AttemptProvider,
        max_retries: int,
    ) -> ResponseAcceptanceOutcome:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        attempts: list[ResponseAttempt] = []
        feedback: tuple[TargetValidationIssue, ...] = ()
        for attempt in range(1, max_retries + 2):
            response = await request_attempt(attempt, feedback)
            envelope = self.envelope(
                response,
                target,
                known_chunk_ids=known_chunk_ids,
            )
            attempts.append(ResponseAttempt(attempt=attempt, envelope=envelope))
            if envelope.validation.valid:
                return ResponseAcceptanceOutcome(
                    item_id=item_id,
                    status=ItemStatus.ACCEPTED,
                    attempts=tuple(attempts),
                    accepted_document=envelope.document,
                )
            feedback = envelope.validation.issues

        return ResponseAcceptanceOutcome(
            item_id=item_id,
            status=ItemStatus.FAILED,
            attempts=tuple(attempts),
            error=(
                "response target validation failed after "
                f"{len(attempts)} attempt(s)"
            ),
        )


__all__ = [
    "AttemptProvider",
    "ResponseAcceptanceOutcome",
    "ResponseAcceptanceService",
    "ResponseAttempt",
]
