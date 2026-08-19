"""Append-only JSONL validation report repository."""

from __future__ import annotations

import asyncio
from pathlib import Path

from sas_migrate.application.validation import ValidationReport


class JsonlValidationReportRepository:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = asyncio.Lock()

    async def append(self, report: ValidationReport) -> str:
        async with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as stream:
                stream.write(report.to_json())
                stream.write("\n")
        return str(self._path)

    async def load(self) -> tuple[ValidationReport, ...]:
        if not self._path.exists():
            return ()
        async with self._lock:
            lines = self._path.read_text("utf-8").splitlines()
        return tuple(ValidationReport.from_json(line) for line in lines if line.strip())


__all__ = ["JsonlValidationReportRepository"]
