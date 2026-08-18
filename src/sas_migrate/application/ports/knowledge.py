"""Knowledge corpus persistence boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sas_migrate.application.knowledge.models import KnowledgeChunk, KnowledgeSource


class KnowledgeRepository(Protocol):
    async def source_fingerprint(self, source_id: str) -> str | None: ...

    async def replace_source(
        self,
        source: KnowledgeSource,
        chunks: tuple[KnowledgeChunk, ...],
    ) -> None: ...

    async def chunks(
        self,
        *,
        source_ids: frozenset[str] | None = None,
    ) -> tuple[KnowledgeChunk, ...]: ...


__all__ = ["KnowledgeRepository"]
