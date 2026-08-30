"""Application orchestration for isolated conversion request processing."""

from __future__ import annotations

import logging

from sas_migrate.application.ports.conversion import (
    ConversionRequestRepository,
    ConversionSourceRepository,
    ConversionTranslationPort,
)
from sas_migrate.core.targets import resolve_sharepoint_target

from .models import (
    ConversionBatchOutcome,
    ConversionModelPreference,
    ConversionOutcome,
    ConversionRequest,
    ConversionStatus,
    ConversionTranslationCommand,
)

LOGGER = logging.getLogger(__name__)


def select_requests(
    requests: tuple[ConversionRequest, ...],
    *,
    include_completed: bool = False,
    request_id: str | None = None,
    application_name: str | None = None,
) -> tuple[ConversionRequest, ...]:
    selected = requests if include_completed else tuple(item for item in requests if item.pending)
    if request_id is not None:
        selected = tuple(item for item in selected if item.request_id == request_id.strip())
    if application_name is not None:
        wanted = application_name.strip().casefold()
        selected = tuple(
            item for item in selected if item.application_name.strip().casefold() == wanted
        )
    return selected


def model_for(
    request: ConversionRequest,
    preferences: tuple[ConversionModelPreference, ...],
    default_model: str,
) -> str:
    for preference in preferences:
        if preference.request_id == request.request_id and preference.model:
            model = preference.model.strip()
            if model:
                return model
    return default_model


class ConversionWorkflow:
    def __init__(
        self,
        *,
        requests: ConversionRequestRepository,
        sources: ConversionSourceRepository,
        translator: ConversionTranslationPort,
        default_model: str,
        explicit_target_fallback: str | None = None,
        configured_target: str | None = None,
    ) -> None:
        if not default_model.strip():
            raise ValueError("default conversion model cannot be blank")
        self._requests = requests
        self._sources = sources
        self._translator = translator
        self._default_model = default_model.strip()
        self._explicit_target_fallback = explicit_target_fallback
        self._configured_target = configured_target

    async def run(
        self,
        *,
        include_completed: bool = False,
        request_id: str | None = None,
        application_name: str | None = None,
        dry_run: bool = False,
    ) -> ConversionBatchOutcome:
        requests = select_requests(
            await self._requests.list_requests(),
            include_completed=include_completed,
            request_id=request_id,
            application_name=application_name,
        )
        preferences = await self._requests.model_preferences()
        outcomes = []
        for request in requests:
            outcomes.append(
                await self._run_one(
                    request,
                    preferences=preferences,
                    dry_run=dry_run,
                )
            )
        return ConversionBatchOutcome(outcomes=tuple(outcomes))

    async def _run_one(
        self,
        request: ConversionRequest,
        *,
        preferences: tuple[ConversionModelPreference, ...],
        dry_run: bool,
    ) -> ConversionOutcome:
        model = model_for(request, preferences, self._default_model)
        target = None
        if not dry_run:
            try:
                await self._requests.set_status(
                    request.request_id,
                    ConversionStatus.IN_PROGRESS,
                )
            except Exception as exc:  # noqa: BLE001 - isolate repository rows
                return ConversionOutcome(
                    request_id=request.request_id,
                    application_name=request.application_name,
                    status=ConversionStatus.FAILED,
                    model=model,
                    error=f"could not mark request in progress: {exc}",
                )

        try:
            target = resolve_sharepoint_target(
                request.output_language,
                explicit_fallback=self._explicit_target_fallback,
                configured=self._configured_target,
            )
            sources = await self._sources.sources_for(request)
            if not sources:
                raise ValueError("no supported source files found")
            result = await self._translator.translate(
                ConversionTranslationCommand(
                    request=request,
                    target=target,
                    model=model,
                    sources=sources,
                    dry_run=dry_run,
                )
            )
            if not result.ok:
                raise RuntimeError(result.error)
            if not dry_run:
                await self._requests.set_status(
                    request.request_id,
                    ConversionStatus.COMPLETED,
                )
            return ConversionOutcome(
                request_id=request.request_id,
                application_name=request.application_name,
                status=ConversionStatus.COMPLETED,
                model=model,
                target=target,
                artifacts=result.artifacts,
            )
        except Exception as exc:  # noqa: BLE001 - one bad request must not abort the queue
            LOGGER.warning(
                "conversion request %s failed: %s",
                request.request_id,
                exc,
            )
            if not dry_run:
                try:
                    await self._requests.set_status(
                        request.request_id,
                        ConversionStatus.FAILED,
                    )
                except Exception as status_exc:  # noqa: BLE001 - preserve original failure
                    LOGGER.error(
                        "could not mark conversion request %s failed: %s",
                        request.request_id,
                        status_exc,
                    )
            return ConversionOutcome(
                request_id=request.request_id,
                application_name=request.application_name,
                status=ConversionStatus.FAILED,
                model=model,
                target=target,
                error=str(exc),
            )


__all__ = ["ConversionWorkflow", "model_for", "select_requests"]
