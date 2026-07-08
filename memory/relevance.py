"""Relevance-based selection of prompted chat history. See memory/README.md.

Logger name: ``memory.relevance``.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from typing import Any, Callable

import bm25s
import faiss
import numpy as np
from langchain_core.messages import BaseMessage, HumanMessage

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _group_turns(history: list[BaseMessage]) -> list[list[BaseMessage]]:
    """
    Group a chronological message list into turns: each HumanMessage opens
    a new turn and every following non-human message (AI, tool, …) joins
    it. Leading non-human messages form a turn of their own, so no message
    is ever dropped by grouping.
    """
    turns: list[list[BaseMessage]] = []
    for msg in history:
        if isinstance(msg, HumanMessage) or not turns:
            turns.append([msg])
        else:
            turns[-1].append(msg)
    return turns


def _turn_text(turn: list[BaseMessage]) -> str:
    return "\n".join(str(m.content) for m in turn)


class RelevantHistorySelector:
    """
    Select the history turn pairs most relevant to the current request.

    Parameters
    ----------
    top_k : int
        Total turn pairs to keep per call (including the always-kept tail).
    always_keep_last : int
        Most recent pairs kept unconditionally, for continuity.
    embeddings : Any | None
        LangChain ``Embeddings`` instance, or a provider string forwarded
        to ``langchain.embeddings.init_embeddings`` (e.g.
        ``"openai:text-embedding-3-small"``). ``None`` (default) disables
        the dense stage — selection is BM25-only, fully offline.
    rrf_k : int
        Reciprocal Rank Fusion constant; 60 is the standard choice.
        Larger values flatten the difference between rank positions.
    reranker : Callable[[str, list[str]], list[float]] | None
        Optional second-stage reranker: given ``(query, docs)`` returns
        one relevance score per doc (higher = better). Applied to the
        RRF-fused shortlist; use for a cross-encoder or LLM judge.
    """

    def __init__(
        self,
        *,
        top_k: int = 6,
        always_keep_last: int = 1,
        embeddings: Any | None = None,
        rrf_k: int = 60,
        reranker: Callable[[str, list[str]], list[float]] | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if always_keep_last < 0:
            raise ValueError(f"always_keep_last must be >= 0, got {always_keep_last}")
        if isinstance(embeddings, str):
            from langchain.embeddings import init_embeddings

            embeddings = init_embeddings(embeddings)
        self.top_k = top_k
        self.always_keep_last = always_keep_last
        self.rrf_k = rrf_k
        self._embeddings = embeddings
        self._reranker = reranker
        self._embedding_cache: dict[str, np.ndarray] = {}
        logger.info(
            f"RelevantHistorySelector: top_k={top_k}  "
            f"always_keep_last={always_keep_last}  "
            f"dense={'on' if embeddings is not None else 'off'}  "
            f"reranker={'on' if reranker is not None else 'off'}  rrf_k={rrf_k}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self, history: list[BaseMessage], query: str) -> list[BaseMessage]:
        """
        Return the subset of *history* to prompt for *query*: the
        ``top_k`` most relevant turn pairs (always including the last
        ``always_keep_last``), flattened back to messages in their
        original chronological order.
        """
        if not history:
            return []
        turns = _group_turns(history)
        if len(turns) <= self.top_k:
            return list(history)

        keep_last = min(self.always_keep_last, self.top_k)
        tail = list(range(len(turns) - keep_last, len(turns)))
        candidates = list(range(len(turns) - keep_last))
        slots = self.top_k - keep_last

        chosen: list[int] = []
        if slots > 0:
            docs = [_turn_text(t) for t in turns]
            maybe_rankings = [self._bm25_ranking(docs, candidates, query)]
            if self._embeddings is not None:
                maybe_rankings.append(self._dense_ranking(docs, candidates, query))
            rankings = [r for r in maybe_rankings if r is not None]
            if rankings:
                fused = self._rrf_fuse(rankings)
            else:
                logger.debug("select: no scorer has signal; recency fallback")
                fused = sorted(candidates, reverse=True)
            if self._reranker is not None:
                fused = self._rerank(fused, docs, query)
            chosen = fused[:slots]

        selected = sorted(set(chosen) | set(tail))
        logger.debug(
            f"select: kept {len(selected)}/{len(turns)} turn pair(s)  "
            f"relevant={sorted(chosen)}  tail={tail}"
        )
        return [m for i in selected for m in turns[i]]

    # ------------------------------------------------------------------
    # Stage 1 — retrieval rankings (best-first candidate index lists)
    # ------------------------------------------------------------------

    def _bm25_ranking(
        self, docs: list[str], candidates: list[int], query: str
    ) -> list[int] | None:
        """Best-first candidate ranking, or ``None`` when BM25 has no signal."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            logger.warning("_bm25_ranking: query produced no tokens; no signal")
            return None
        # "_" placeholder keeps a pathological empty doc from breaking the
        # index; it can never match a real query token.
        corpus = [_tokenize(docs[i]) or ["_"] for i in candidates]
        retriever = bm25s.BM25()
        retriever.index(corpus, show_progress=False)
        scores = retriever.get_scores(query_tokens)
        if float(scores.max()) - float(scores.min()) < 1e-12:
            return None  # every doc tied — ordering would be arbitrary
        # Remaining ties break toward recency.
        order = sorted(
            range(len(candidates)), key=lambda j: (-scores[j], -candidates[j])
        )
        return [candidates[j] for j in order]

    def _dense_ranking(
        self, docs: list[str], candidates: list[int], query: str
    ) -> list[int] | None:
        """Best-first candidate ranking, or ``None`` when cosine has no signal."""
        vectors = self._embed_cached([docs[i] for i in candidates])
        query_vec = self._normalize(
            np.asarray(self._embeddings.embed_query(query), dtype=np.float32)
        )
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        similarities, order = index.search(query_vec[None, :], len(candidates))
        if float(similarities.max()) - float(similarities.min()) < 1e-9:
            return None  # every doc tied — ordering would be arbitrary
        return [candidates[j] for j in order[0] if j != -1]

    # ------------------------------------------------------------------
    # Stage 2 — rerank
    # ------------------------------------------------------------------

    def _rrf_fuse(self, rankings: list[list[int]]) -> list[int]:
        scores: dict[int, float] = defaultdict(float)
        for ranking in rankings:
            for position, idx in enumerate(ranking):
                scores[idx] += 1.0 / (self.rrf_k + position + 1)
        # Ties break toward recency (higher turn index first).
        return sorted(scores, key=lambda i: (-scores[i], -i))

    def _rerank(self, fused: list[int], docs: list[str], query: str) -> list[int]:
        window = fused[: max(4 * self.top_k, self.top_k)]
        scores = self._reranker(query, [docs[i] for i in window])
        order = sorted(range(len(window)), key=lambda j: (-scores[j], -window[j]))
        return [window[j] for j in order] + fused[len(window) :]

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        return vec / max(float(np.linalg.norm(vec)), 1e-12)

    def _embed_cached(self, texts: list[str]) -> np.ndarray:
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        missing = [
            (k, t) for k, t in zip(keys, texts) if k not in self._embedding_cache
        ]
        if missing:
            logger.debug(
                f"_embed_cached: embedding {len(missing)} new turn pair(s) "
                f"({len(texts) - len(missing)} cached)"
            )
            new_vectors = self._embeddings.embed_documents([t for _, t in missing])
            for (key, _), vec in zip(missing, new_vectors):
                self._embedding_cache[key] = self._normalize(
                    np.asarray(vec, dtype=np.float32)
                )
        return np.stack([self._embedding_cache[k] for k in keys])
