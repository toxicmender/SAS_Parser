"""Source discovery and retrieval port."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from pydantic import Field

from sas_migrate.core.models import ContractModel


class SourceObject(ContractModel):
    source_id: str = Field(min_length=1)
    name: str
    content: bytes
    metadata: dict[str, str] = Field(default_factory=dict)


class SourceRepository(Protocol):
    def iter_sources(self) -> AsyncIterator[SourceObject]: ...
