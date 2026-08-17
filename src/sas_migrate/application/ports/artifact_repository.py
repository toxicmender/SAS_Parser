"""Audit and deliverable persistence port."""

from __future__ import annotations

from typing import Protocol

from pydantic import Field

from sas_migrate.core.ids import ArtifactId, RunId
from sas_migrate.core.models import ContractModel


class ArtifactWrite(ContractModel):
    artifact_id: ArtifactId
    media_type: str
    content: bytes
    metadata: dict[str, str] = Field(default_factory=dict)


class ArtifactRepository(Protocol):
    async def write(self, run_id: RunId, artifact: ArtifactWrite) -> str: ...
