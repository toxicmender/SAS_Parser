"""Spark-free in-memory knowledge repository."""

from __future__ import annotations

from asyncio import Lock

from sas_migrate.application.knowledge.models import KnowledgeChunk, KnowledgeSource


class InMemoryKnowledgeRepository:
    def __init__(self) -> None:
        self._sources: dict[str, KnowledgeSource] = {}
        self._chunks: dict[str, tuple[KnowledgeChunk, ...]] = {}
        self._lock = Lock()
        self.write_count = 0

    async def source_fingerprint(self, source_id: str) -> str | None:
        source = self._sources.get(source_id)
        return source.extraction_fingerprint if source is not None else None

    async def replace_source(
        self,
        source: KnowledgeSource,
        chunks: tuple[KnowledgeChunk, ...],
    ) -> None:
        if any(chunk.source_id != source.source_id for chunk in chunks):
            raise ValueError("knowledge chunk source does not match replacement source")
        async with self._lock:
            self._sources[source.source_id] = source
            self._chunks[source.source_id] = chunks
            self.write_count += 1

    async def chunks(
        self,
        *,
        source_ids: frozenset[str] | None = None,
    ) -> tuple[KnowledgeChunk, ...]:
        selected = source_ids or frozenset(self._chunks)
        return tuple(
            chunk
            for source_id in self._sources
            if source_id in selected
            for chunk in self._chunks.get(source_id, ())
        )


__all__ = ["InMemoryKnowledgeRepository"]
