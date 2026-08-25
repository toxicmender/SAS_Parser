"""SharePoint request and source adapters over a narrow transport boundary."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from pydantic import Field

from sas_migrate.application.conversion import (
    ConversionModelPreference,
    ConversionRequest,
    ConversionStatus,
)
from sas_migrate.application.ports import SourceObject
from sas_migrate.core.models import ContractModel

LOGGER = logging.getLogger(__name__)


class SharePointConversionConfig(ContractModel):
    request_list_id: str = Field(min_length=1)
    conversion_list_id: str | None = None
    base_path: str = Field(min_length=1)


class SharePointConversionTransport(Protocol):
    def list_items(self, list_id: str, **options: Any) -> list[dict[str, Any]]: ...

    def update_list_item(
        self,
        list_id: str,
        item_id: str,
        fields: dict[str, Any],
    ) -> Any: ...

    def list_files(
        self,
        folder: str,
        extensions: set[str] | None = None,
    ) -> list[dict[str, Any]]: ...

    def download_file_as_text(self, path: str, *, encoding: str = "utf-8") -> str: ...


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"yes", "true", "1"}
    return bool(value)


def request_from_row(row: dict[str, Any]) -> ConversionRequest:
    fields = row.get("fields") or {}
    application_name = str(fields.get("Application Name") or "").strip()
    if not application_name:
        raise ValueError("SharePoint request row requires Application Name")
    request_id = str(row.get("id") or "").strip()
    if not request_id:
        raise ValueError("SharePoint request row requires an id")
    output = str(fields.get("Destination Language") or "").strip() or None
    macro = str(fields.get("Macro File Name x003f ") or "").strip() or None
    return ConversionRequest(
        request_id=request_id,
        application_name=application_name,
        input_language=str(fields.get("Source Language") or "SAS").strip() or "SAS",
        output_language=output,
        macro_file_name=macro,
        validation_required=_truthy(fields.get("Validation x0020 Documents x0020 ")),
        status=str(fields.get("Status") or ""),
    )


def preference_from_row(row: dict[str, Any]) -> ConversionModelPreference:
    fields = row.get("fields") or {}
    request_id = str(fields.get("Request_ID") or "").strip() or None
    return ConversionModelPreference(
        request_id=request_id,
        script_name=str(fields.get("Script_Name") or ""),
        model=str(fields.get("Model") or "").strip() or None,
        status=str(fields.get("Status") or ""),
    )


class SharePointConversionRequestRepository:
    def __init__(
        self,
        transport: SharePointConversionTransport,
        config: SharePointConversionConfig,
    ) -> None:
        self._transport = transport
        self._config = config

    async def list_requests(self) -> tuple[ConversionRequest, ...]:
        requests = []
        for row in self._transport.list_items(self._config.request_list_id):
            try:
                requests.append(request_from_row(row))
            except (TypeError, ValueError) as exc:
                LOGGER.warning("skipping malformed SharePoint request row: %s", exc)
        return tuple(requests)

    async def model_preferences(self) -> tuple[ConversionModelPreference, ...]:
        if self._config.conversion_list_id is None:
            return ()
        return tuple(
            preference_from_row(row)
            for row in self._transport.list_items(self._config.conversion_list_id)
        )

    async def set_status(self, request_id: str, status: ConversionStatus) -> None:
        self._transport.update_list_item(
            self._config.request_list_id,
            request_id,
            {"Status": status.value},
        )


class SharePointConversionSourceRepository:
    def __init__(
        self,
        transport: SharePointConversionTransport,
        config: SharePointConversionConfig,
    ) -> None:
        self._transport = transport
        self._config = config

    async def sources_for(
        self,
        request: ConversionRequest,
    ) -> tuple[SourceObject, ...]:
        if request.input_language.strip().casefold() != "sas":
            raise ValueError(
                f"no source extensions known for {request.input_language!r}"
            )
        folder = "/".join(
            (
                self._config.base_path.strip("/"),
                request.application_name.strip("/"),
                "scripts_original",
            )
        )
        rows = sorted(
            self._transport.list_files(folder, extensions={"sas", "txt"}),
            key=lambda row: str(row.get("name") or "").casefold(),
        )
        sources = []
        for row in rows:
            if row.get("is_folder"):
                continue
            path = str(row.get("path") or f"{folder}/{row.get('name', '')}")
            try:
                text = self._transport.download_file_as_text(path)
            except Exception as exc:  # noqa: BLE001 - skip one unreadable remote file
                LOGGER.warning("skipping unreadable SharePoint source %s: %s", path, exc)
                continue
            sources.append(
                SourceObject(
                    source_id=path,
                    name=str(row.get("name") or path.rsplit("/", 1)[-1]),
                    content=text.encode("utf-8"),
                    metadata={"origin": "sharepoint", "folder": folder},
                )
            )
        return tuple(sources)


__all__ = [
    "SharePointConversionConfig",
    "SharePointConversionRequestRepository",
    "SharePointConversionSourceRepository",
    "SharePointConversionTransport",
    "preference_from_row",
    "request_from_row",
]
