"""Packaged and filesystem assessment-profile repositories."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path


class PackageAssessmentProfileRepository:
    def load(self, name: str) -> dict[str, object]:
        resource = files("sas_migrate") / "resources" / "assessment" / f"{name}.json"
        return _object(json.loads(resource.read_text(encoding="utf-8")), name)

    def names(self) -> tuple[str, ...]:
        root = files("sas_migrate") / "resources" / "assessment"
        return tuple(sorted(path.name.removesuffix(".json") for path in root.iterdir() if path.name.endswith(".json")))


class DirectoryAssessmentProfileRepository:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def load(self, name: str) -> dict[str, object]:
        path = self._path / f"{name}.json"
        return _object(json.loads(path.read_text("utf-8")), name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(path.stem for path in self._path.glob("*.json")))


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"assessment profile {name!r} is not an object")
    return value


__all__ = ["DirectoryAssessmentProfileRepository", "PackageAssessmentProfileRepository"]
