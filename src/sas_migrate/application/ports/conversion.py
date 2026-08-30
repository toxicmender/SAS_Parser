"""Ports used by local and remote conversion workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sas_migrate.application.conversion.models import (
        ConversionModelPreference,
        ConversionRequest,
        ConversionStatus,
        ConversionTranslationCommand,
        ConversionTranslationResult,
    )
    from sas_migrate.application.ports.source_repository import SourceObject


class ConversionRequestRepository(Protocol):
    async def list_requests(self) -> tuple[ConversionRequest, ...]: ...

    async def model_preferences(self) -> tuple[ConversionModelPreference, ...]: ...

    async def set_status(
        self,
        request_id: str,
        status: ConversionStatus,
    ) -> None: ...


class ConversionSourceRepository(Protocol):
    async def sources_for(
        self,
        request: ConversionRequest,
    ) -> tuple[SourceObject, ...]: ...


class ConversionTranslationPort(Protocol):
    async def translate(
        self,
        command: ConversionTranslationCommand,
    ) -> ConversionTranslationResult: ...


__all__ = [
    "ConversionRequestRepository",
    "ConversionSourceRepository",
    "ConversionTranslationPort",
]
