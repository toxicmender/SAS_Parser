"""Filesystem and in-process adapters for local conversion runs."""

from __future__ import annotations

from pathlib import Path

from sas_migrate.application.conversion import (
    ConversionModelPreference,
    ConversionRequest,
    ConversionStatus,
)
from sas_migrate.application.ports import SourceObject


class LocalConversionRequestRepository:
    def __init__(
        self,
        request: ConversionRequest,
        *,
        preferences: tuple[ConversionModelPreference, ...] = (),
    ) -> None:
        self._request = request
        self._preferences = preferences
        self.statuses: list[ConversionStatus] = []

    async def list_requests(self) -> tuple[ConversionRequest, ...]:
        return (self._request,)

    async def model_preferences(self) -> tuple[ConversionModelPreference, ...]:
        return self._preferences

    async def set_status(self, request_id: str, status: ConversionStatus) -> None:
        if request_id != self._request.request_id:
            raise KeyError(f"unknown local conversion request {request_id!r}")
        self.statuses.append(status)
        self._request = self._request.model_copy(update={"status": status.value})


class LocalConversionSourceRepository:
    def __init__(
        self,
        source_dir: str | Path,
        *,
        extensions: frozenset[str] = frozenset({"sas", "txt"}),
    ) -> None:
        self._source_dir = Path(source_dir)
        self._extensions = frozenset(value.casefold().lstrip(".") for value in extensions)

    async def sources_for(
        self,
        request: ConversionRequest,
    ) -> tuple[SourceObject, ...]:
        del request
        if not self._source_dir.is_dir():
            raise FileNotFoundError(f"source directory does not exist: {self._source_dir}")
        paths = sorted(
            (
                path
                for path in self._source_dir.iterdir()
                if path.is_file() and path.suffix.casefold().lstrip(".") in self._extensions
            ),
            key=lambda path: path.name.casefold(),
        )
        return tuple(
            SourceObject(
                source_id=path.as_posix(),
                name=path.name,
                content=path.read_bytes(),
                metadata={"origin": "local"},
            )
            for path in paths
        )


__all__ = ["LocalConversionRequestRepository", "LocalConversionSourceRepository"]
