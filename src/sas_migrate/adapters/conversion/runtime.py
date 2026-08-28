"""Process-local state and filesystem artifacts for local conversion runs."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from sas_migrate.application.ports import ArtifactWrite
from sas_migrate.core.responses import ResponseEnvelope
from sas_migrate.core.runs import RunEvent
from sas_migrate.core.tokens import CallTokenRecord


def _safe_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "run"


def _artifact_path(artifact_id: str) -> PurePosixPath:
    value = PurePosixPath(artifact_id)
    unsafe = (
        value.is_absolute()
        or not value.parts
        or any(part in {"", ".", ".."} for part in value.parts)
    )
    if unsafe:
        raise ValueError(f"unsafe artifact id: {artifact_id!r}")
    return value


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class InMemoryRunEventRepository:
    def __init__(self) -> None:
        self._events: list[RunEvent] = []

    async def append(self, event: RunEvent) -> None:
        self._events.append(event)

    async def events(self, run_id: str, thread_id: str) -> tuple[RunEvent, ...]:
        return tuple(
            event
            for event in self._events
            if event.run_id == run_id and event.thread_id == thread_id
        )


class InMemoryTokenRecordRepository:
    def __init__(self) -> None:
        self._records: list[CallTokenRecord] = []

    async def append(self, record: CallTokenRecord) -> None:
        self._records.append(record)

    async def records(
        self,
        run_id: str,
        thread_id: str,
    ) -> tuple[CallTokenRecord, ...]:
        return tuple(
            record
            for record in self._records
            if record.run_id == run_id and record.thread_id == thread_id
        )


class InMemoryAcceptedResponseRepository:
    def __init__(self) -> None:
        self._responses: dict[tuple[str, str, str], ResponseEnvelope] = {}

    async def accepted_response(
        self,
        run_id: str,
        thread_id: str,
        item_id: str,
    ) -> ResponseEnvelope | None:
        return self._responses.get((run_id, thread_id, item_id))

    async def remember_accepted(
        self,
        run_id: str,
        thread_id: str,
        item_id: str,
        response: ResponseEnvelope,
    ) -> None:
        self._responses[(run_id, thread_id, item_id)] = response

    async def forget_accepted(
        self,
        run_id: str,
        thread_id: str,
        item_ids: tuple[str, ...],
    ) -> None:
        for item_id in item_ids:
            self._responses.pop((run_id, thread_id, item_id), None)

    async def fork_accepted(
        self,
        source_run_id: str,
        source_thread_id: str,
        destination_run_id: str,
        destination_thread_id: str,
        item_ids: tuple[str, ...],
    ) -> None:
        for item_id in item_ids:
            response = self._responses.get(
                (source_run_id, source_thread_id, item_id)
            )
            if response is not None:
                self._responses[
                    (destination_run_id, destination_thread_id, item_id)
                ] = response


class DirectoryArtifactRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    async def write(self, run_id: str, artifact: ArtifactWrite) -> str:
        relative = _artifact_path(artifact.artifact_id)
        destination = self._root / _safe_segment(run_id) / Path(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_bytes(artifact.content)
        temporary.replace(destination)
        return str(destination.resolve())


__all__ = [
    "DirectoryArtifactRepository",
    "InMemoryAcceptedResponseRepository",
    "InMemoryRunEventRepository",
    "InMemoryTokenRecordRepository",
    "SystemClock",
]
