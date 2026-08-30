"""Lazy BM25/FAISS hybrid ranker for the v2 knowledge contract."""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from sas_migrate.application.knowledge.models import (
    KnowledgeChunk,
    KnowledgeRanking,
    RetrievalSignal,
)
from sas_migrate.application.ports.knowledge import (
    EmbeddingCache,
    EmbeddingProvider,
    EmbeddingVector,
    KnowledgeReranker,
)

from .cache import InMemoryEmbeddingCache

_WORD = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


class CallableKnowledgeReranker:
    def __init__(
        self,
        scorer: Callable[[str, tuple[str, ...]], tuple[float, ...]],
    ) -> None:
        self._scorer = scorer

    def score(self, query: str, documents: tuple[str, ...]) -> tuple[float, ...]:
        return self._scorer(query, documents)


class HybridKnowledgeRanker:
    """Index a supplied corpus with lexical and optional dense retrieval."""

    def __init__(
        self,
        *,
        embeddings: EmbeddingProvider | None = None,
        embedding_cache: EmbeddingCache | None = None,
        embedding_namespace: str | None = None,
        reranker: KnowledgeReranker | None = None,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least one")
        if embeddings is not None and not embedding_namespace:
            raise ValueError("dense retrieval requires an embedding namespace")
        self._embeddings = embeddings
        self._cache = embedding_cache or InMemoryEmbeddingCache()
        self._namespace = embedding_namespace or ""
        self._reranker = reranker
        self._rrf_k = rrf_k
        self._lexical_documents: tuple[str, ...] | None = None
        self._lexical_index: Any | None = None
        self._dense_documents: tuple[str, ...] | None = None
        self._dense_index: Any | None = None

    def rank(
        self,
        query: str,
        chunks: tuple[KnowledgeChunk, ...],
        *,
        limit: int | None = None,
    ) -> tuple[KnowledgeRanking, ...]:
        if limit is not None and limit < 0:
            raise ValueError("ranking limit cannot be negative")
        if not chunks or limit == 0:
            return ()

        documents = tuple(f"{chunk.section_path} {chunk.text}" for chunk in chunks)
        rankings: list[list[int]] = []
        lexical = self._lexical_ranking(query, documents)
        if lexical is not None:
            rankings.append(lexical)
        dense = self._dense_ranking(query, documents)
        if dense is not None:
            rankings.append(dense)
        if not rankings:
            return ()

        lexical_positions = self._positions(lexical)
        dense_positions = self._positions(dense)
        fused_scores = self._fusion_scores(rankings)
        order = sorted(
            fused_scores,
            key=lambda index: (
                -fused_scores[index],
                chunks[index].ordinal,
                chunks[index].chunk_id,
            ),
        )
        reranker_scores: dict[int, float] = {}
        if self._reranker is not None:
            window_size = min(len(order), max(4 * (limit or len(order)), limit or 0))
            window = order[:window_size]
            raw_scores = self._reranker.score(
                query,
                tuple(documents[index] for index in window),
            )
            if len(raw_scores) != len(window):
                raise ValueError("reranker must return one score per document")
            if any(not math.isfinite(score) for score in raw_scores):
                raise ValueError("reranker scores must be finite")
            reranker_scores = dict(zip(window, raw_scores, strict=True))
            window.sort(
                key=lambda index: (
                    -reranker_scores[index],
                    chunks[index].ordinal,
                    chunks[index].chunk_id,
                )
            )
            order = window + order[window_size:]

        if limit is not None:
            order = order[:limit]
        return tuple(
            KnowledgeRanking(
                chunk_id=chunks[index].chunk_id,
                score=1.0 / position,
                reciprocal_rank_score=fused_scores[index],
                lexical_rank=lexical_positions.get(index),
                dense_rank=dense_positions.get(index),
                reranker_score=reranker_scores.get(index),
                signals=tuple(
                    signal
                    for signal, positions in (
                        (RetrievalSignal.LEXICAL, lexical_positions),
                        (RetrievalSignal.DENSE, dense_positions),
                        (RetrievalSignal.RERANKER, reranker_scores),
                    )
                    if index in positions
                ),
            )
            for position, index in enumerate(order, start=1)
        )

    @staticmethod
    def _positions(ranking: list[int] | None) -> dict[int, int]:
        return (
            {index: position for position, index in enumerate(ranking, start=1)}
            if ranking is not None
            else {}
        )

    def _fusion_scores(self, rankings: list[list[int]]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for ranking in rankings:
            for position, index in enumerate(ranking, start=1):
                scores[index] += 1.0 / (self._rrf_k + position)
        return scores

    def _lexical_ranking(
        self,
        query: str,
        documents: tuple[str, ...],
    ) -> list[int] | None:
        query_tokens = _tokens(query)
        if not query_tokens:
            return None

        import bm25s

        if documents != self._lexical_documents:
            tokenized = [_tokens(document) or ["_"] for document in documents]
            self._lexical_index = bm25s.BM25()
            self._lexical_index.index(tokenized, show_progress=False)
            self._lexical_documents = documents
        assert self._lexical_index is not None
        scores = self._lexical_index.get_scores(query_tokens)
        if float(scores.max()) <= 1e-12:
            return None
        return sorted(range(len(documents)), key=lambda item: (-scores[item], item))

    def _dense_ranking(
        self,
        query: str,
        documents: tuple[str, ...],
    ) -> list[int] | None:
        if self._embeddings is None:
            return None

        import faiss
        import numpy as np

        vectors = np.asarray(self._document_vectors(documents), dtype=np.float32)
        query_vector = np.asarray(self._embeddings.embed_query(query), dtype=np.float32)
        if vectors.ndim != 2 or query_vector.ndim != 1 or not query_vector.size:
            raise ValueError("embeddings must be non-empty one-dimensional vectors")
        if vectors.shape[1] != query_vector.shape[0]:
            raise ValueError("document and query embedding dimensions must match")
        if not np.isfinite(vectors).all() or not np.isfinite(query_vector).all():
            raise ValueError("embeddings must contain only finite values")
        faiss.normalize_L2(vectors)
        query_matrix = query_vector.reshape(1, -1)
        faiss.normalize_L2(query_matrix)
        if not np.any(vectors) or not np.any(query_matrix):
            return None
        if documents != self._dense_documents:
            self._dense_index = faiss.IndexFlatIP(vectors.shape[1])
            self._dense_index.add(vectors)
            self._dense_documents = documents
        assert self._dense_index is not None
        similarities, order = self._dense_index.search(query_matrix, len(documents))
        if float(similarities.max()) - float(similarities.min()) < 1e-9:
            return None
        return [int(item) for item in order[0] if item >= 0]

    def _document_vectors(
        self,
        documents: tuple[str, ...],
    ) -> tuple[EmbeddingVector, ...]:
        if self._embeddings is None:
            return ()
        keys = tuple(
            hashlib.sha256(f"{self._namespace}\0{text}".encode()).hexdigest()
            for text in documents
        )
        cached = self._cache.get_many(keys)
        missing = tuple(
            (key, text)
            for key, text in zip(keys, documents, strict=True)
            if key not in cached
        )
        if missing:
            fresh = self._embeddings.embed_documents(tuple(text for _, text in missing))
            if len(fresh) != len(missing):
                raise ValueError("embedding provider must return one vector per document")
            additions = {
                key: tuple(float(value) for value in vector)
                for (key, _), vector in zip(missing, fresh, strict=True)
            }
            self._cache.put_many(additions)
            cached.update(additions)
        dimensions = {len(cached[key]) for key in keys}
        if len(dimensions) != 1 or 0 in dimensions:
            raise ValueError("document embeddings must share a non-zero dimension")
        return tuple(cached[key] for key in keys)


__all__ = ["CallableKnowledgeReranker", "HybridKnowledgeRanker"]
