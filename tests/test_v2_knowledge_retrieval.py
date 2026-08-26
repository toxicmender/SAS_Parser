"""Phase 9 advanced knowledge retrieval and embedding-cache contracts."""

from __future__ import annotations

import asyncio
import hashlib
import pathlib
import subprocess
import sys

import pytest
from pydantic import ValidationError

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from sas_migrate.adapters.knowledge import (
    CallableKnowledgeReranker,
    HybridKnowledgeRanker,
    InMemoryEmbeddingCache,
    InMemoryKnowledgeRepository,
    NpzEmbeddingCache,
)
from sas_migrate.application.knowledge import (
    KnowledgeChunk,
    KnowledgeRanking,
    KnowledgeRetriever,
    KnowledgeRole,
    KnowledgeSource,
    RetrievalQuery,
    RetrievalSignal,
)
from sas_migrate.core.targets import TargetId


def _chunk(chunk_id: str, text: str, ordinal: int) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        source_id="guide",
        role=KnowledgeRole.TARGET_GUIDE,
        section_path=chunk_id,
        text=text,
        page_start=1,
        page_end=1,
        ordinal=ordinal,
        token_count=len(text.split()),
        target=TargetId.PYSPARK,
    )


def _corpus() -> tuple[KnowledgeChunk, ...]:
    return (
        _chunk("revenue", "quarterly revenue tables", 0),
        _chunk("dates", "calendar interval arithmetic", 1),
        _chunk("joins", "merge customer dimensions", 2),
    )


class _Embeddings:
    def __init__(self) -> None:
        self.document_calls: list[tuple[str, ...]] = []
        self.query_calls: list[str] = []

    @staticmethod
    def _vector(text: str) -> tuple[float, ...]:
        lowered = text.casefold()
        if "revenue" in lowered or "sales" in lowered:
            return (1.0, 0.0, 0.0)
        if "calendar" in lowered or "date" in lowered:
            return (0.0, 1.0, 0.0)
        return (0.0, 0.0, 1.0)

    def embed_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        self.document_calls.append(texts)
        return tuple(self._vector(text) for text in texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        self.query_calls.append(text)
        return self._vector(text)


def _dense_ranker(
    embeddings: _Embeddings | None = None,
    *,
    cache: InMemoryEmbeddingCache | NpzEmbeddingCache | None = None,
    namespace: str = "fake-v1",
    reranker: CallableKnowledgeReranker | None = None,
) -> HybridKnowledgeRanker:
    return HybridKnowledgeRanker(
        embeddings=embeddings or _Embeddings(),
        embedding_cache=cache,
        embedding_namespace=namespace,
        reranker=reranker,
    )


def test_adapter_import_does_not_load_optional_ranking_libraries() -> None:
    code = """
import sys
before = set(sys.modules)
import sas_migrate.adapters.knowledge
loaded = set(sys.modules) - before
assert not ({'bm25s', 'faiss', 'numpy'} & loaded), sorted(loaded)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=SRC,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_lexical_ranker_reports_signal_and_stable_order() -> None:
    rankings = HybridKnowledgeRanker().rank("calendar arithmetic", _corpus())
    assert rankings[0].chunk_id == "dates"
    assert rankings[0].lexical_rank == 1
    assert rankings[0].signals == (RetrievalSignal.LEXICAL,)
    assert rankings[0].score == 1.0
    assert HybridKnowledgeRanker().rank("!!!", _corpus()) == ()


def test_dense_ranker_recovers_semantic_match_and_fuses_signals() -> None:
    ranker = _dense_ranker()
    semantic = ranker.rank("sales figures", _corpus(), limit=1)
    assert semantic[0].chunk_id == "revenue"
    assert semantic[0].signals == (RetrievalSignal.DENSE,)

    fused = ranker.rank("revenue", _corpus(), limit=1)
    assert fused[0].signals == (
        RetrievalSignal.LEXICAL,
        RetrievalSignal.DENSE,
    )
    assert fused[0].reciprocal_rank_score == pytest.approx(2 / 61)


def test_reranker_reorders_fused_window_and_validates_scores() -> None:
    reranker = CallableKnowledgeReranker(
        lambda _query, documents: tuple(
            10.0 if "calendar" in document else 0.0 for document in documents
        )
    )
    rankings = _dense_ranker(reranker=reranker).rank("revenue", _corpus(), limit=2)
    assert [ranking.chunk_id for ranking in rankings] == ["dates", "revenue"]
    assert rankings[0].signals[-1] is RetrievalSignal.RERANKER
    assert rankings[0].reranker_score == 10.0
    assert rankings[0].score > rankings[1].score

    short = CallableKnowledgeReranker(lambda _query, _documents: ())
    with pytest.raises(ValueError, match="one score per document"):
        _dense_ranker(reranker=short).rank("revenue", _corpus())
    non_finite = CallableKnowledgeReranker(
        lambda _query, documents: tuple(float("nan") for _ in documents)
    )
    with pytest.raises(ValueError, match="finite"):
        _dense_ranker(reranker=non_finite).rank("revenue", _corpus())


def test_ranker_reuses_indexes_and_namespaces_cached_documents(monkeypatch) -> None:
    import bm25s
    import faiss

    bm25_builds = 0
    faiss_builds = 0
    real_bm25 = bm25s.BM25
    real_faiss = faiss.IndexFlatIP

    def counted_bm25(*args, **kwargs):
        nonlocal bm25_builds
        bm25_builds += 1
        return real_bm25(*args, **kwargs)

    def counted_faiss(*args, **kwargs):
        nonlocal faiss_builds
        faiss_builds += 1
        return real_faiss(*args, **kwargs)

    monkeypatch.setattr(bm25s, "BM25", counted_bm25)
    monkeypatch.setattr(faiss, "IndexFlatIP", counted_faiss)
    cache = InMemoryEmbeddingCache()
    first = _Embeddings()
    ranker = _dense_ranker(first, cache=cache, namespace="model-a")
    ranker.rank("sales", _corpus())
    ranker.rank("revenue", _corpus())
    assert len(first.document_calls) == 1
    assert len(first.query_calls) == 2
    assert bm25_builds == 1
    assert faiss_builds == 1

    changed = (*_corpus(), _chunk("extra", "new guidance", 3))
    ranker.rank("revenue", changed)
    assert bm25_builds == 2
    assert faiss_builds == 2

    second = _Embeddings()
    _dense_ranker(second, cache=cache, namespace="model-b").rank("sales", _corpus())
    assert len(second.document_calls) == 1


def test_npz_cache_round_trips_across_rankers_and_recovers_corruption(tmp_path) -> None:
    path = tmp_path / "embeddings.npz"
    first = _Embeddings()
    _dense_ranker(first, cache=NpzEmbeddingCache(path)).rank("sales", _corpus())
    assert path.exists()
    assert len(first.document_calls) == 1

    second = _Embeddings()
    _dense_ranker(second, cache=NpzEmbeddingCache(path)).rank("sales", _corpus())
    assert second.document_calls == []

    path.write_text("not an npz", encoding="utf-8")
    third = _Embeddings()
    _dense_ranker(third, cache=NpzEmbeddingCache(path)).rank("sales", _corpus())
    assert len(third.document_calls) == 1


def test_cache_and_embedding_contract_reject_invalid_shapes(tmp_path) -> None:
    cache = NpzEmbeddingCache(tmp_path / "mixed.npz")
    with pytest.raises(ValueError, match="share a non-zero dimension"):
        cache.put_many({"a": (1.0,), "b": (1.0, 2.0)})
    cache.put_many({})

    class TooFew(_Embeddings):
        def embed_documents(
            self, texts: tuple[str, ...]
        ) -> tuple[tuple[float, ...], ...]:
            return super().embed_documents(texts)[:-1]

    with pytest.raises(ValueError, match="one vector per document"):
        _dense_ranker(TooFew()).rank("sales", _corpus())

    class WrongQuery(_Embeddings):
        def embed_query(self, text: str) -> tuple[float, ...]:
            return (1.0, 0.0)

    with pytest.raises(ValueError, match="dimensions must match"):
        _dense_ranker(WrongQuery()).rank("sales", _corpus())


def test_ranker_validates_configuration_limits_and_zero_vectors() -> None:
    with pytest.raises(ValueError, match="rrf_k"):
        HybridKnowledgeRanker(rrf_k=0)
    with pytest.raises(ValueError, match="namespace"):
        HybridKnowledgeRanker(embeddings=_Embeddings())
    with pytest.raises(ValueError, match="limit"):
        HybridKnowledgeRanker().rank("query", _corpus(), limit=-1)
    assert HybridKnowledgeRanker().rank("query", (), limit=2) == ()
    assert HybridKnowledgeRanker().rank("query", _corpus(), limit=0) == ()

    class Zero(_Embeddings):
        @staticmethod
        def _vector(text: str) -> tuple[float, ...]:
            return (0.0, 0.0, 0.0)

    assert _dense_ranker(Zero()).rank("unmatched", _corpus()) == ()


def test_application_retriever_accepts_advanced_ranker() -> None:
    async def scenario() -> None:
        repository = InMemoryKnowledgeRepository()
        chunks = _corpus()
        await repository.replace_source(
            KnowledgeSource(
                source_id="guide",
                role=KnowledgeRole.TARGET_GUIDE,
                content_sha256=hashlib.sha256(b"guide").hexdigest(),
                extraction_fingerprint=hashlib.sha256(b"extraction").hexdigest(),
                target=TargetId.PYSPARK,
            ),
            chunks,
        )
        selection = await KnowledgeRetriever(
            repository,
            topical_ranker=_dense_ranker(),
        ).select(
            RetrievalQuery(
                text="sales figures",
                target=TargetId.PYSPARK,
                max_results=1,
                max_tokens=100,
            )
        )
        assert selection.results[0].chunk.chunk_id == "revenue"
        assert selection.results[0].reasons == ("topical dense match",)

    asyncio.run(scenario())


def test_ranking_contract_rejects_non_finite_scores() -> None:
    with pytest.raises(ValidationError, match="finite"):
        KnowledgeRanking(
            chunk_id="chunk",
            score=1.0,
            reciprocal_rank_score=float("inf"),
            signals=(RetrievalSignal.LEXICAL,),
        )
