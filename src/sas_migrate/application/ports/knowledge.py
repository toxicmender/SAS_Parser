"""Knowledge corpus persistence boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sas_migrate.application.knowledge.models import (
        KnowledgeChunk,
        KnowledgeRanking,
        KnowledgeSource,
    )

type EmbeddingVector = tuple[float, ...]


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: tuple[str, ...]) -> tuple[EmbeddingVector, ...]: ...

    def embed_query(self, text: str) -> EmbeddingVector: ...


class EmbeddingCache(Protocol):
    def get_many(self, keys: tuple[str, ...]) -> dict[str, EmbeddingVector]: ...

    def put_many(self, values: dict[str, EmbeddingVector]) -> None: ...


class KnowledgeReranker(Protocol):
    def score(self, query: str, documents: tuple[str, ...]) -> tuple[float, ...]: ...


class KnowledgeRanker(Protocol):
    def rank(
        self,
        query: str,
        chunks: tuple[KnowledgeChunk, ...],
        *,
        limit: int | None = None,
    ) -> tuple[KnowledgeRanking, ...]: ...


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


__all__ = [
    "EmbeddingCache",
    "EmbeddingProvider",
    "EmbeddingVector",
    "KnowledgeRanker",
    "KnowledgeRepository",
    "KnowledgeReranker",
]
