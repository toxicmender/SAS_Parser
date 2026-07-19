"""Relevance-based selection of prompted chat history. See memory/README.md.

Logger name: ``memory.relevance``.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import bm25s
import numpy as np
from langchain_core.messages import BaseMessage

# faiss is imported lazily inside HybridRanker.index() — the only remaining
# consumer — so BM25-only pipelines never pay its import cost.

from .turns import approx_token_count, group_turns, turn_text

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# Turn helpers live in memory.turns (shared with memory.summarize, which
# must not import this module's bm25s/faiss stack); the old private names
# are kept as aliases for existing importers.
_group_turns = group_turns
_turn_text = turn_text


class DiskCachedEmbeddings:
    """
    Wrap a LangChain ``Embeddings`` with an on-disk document-embedding cache.

    Embedding a large fixed corpus is the one genuinely expensive step of
    turning dense retrieval on; the vectors never change between runs. Document
    embeddings are memoised to an ``.npz`` keyed by content SHA-1 (queries,
    which vary every call, are passed straight through). Sits under
    :class:`HybridRanker`'s in-process cache, so a warm disk cache means no
    model call at all on subsequent runs.
    """

    def __init__(self, embeddings: Any, cache_path: str) -> None:
        self._embeddings = embeddings
        self._cache_path = Path(cache_path)
        self._cache: dict[str, np.ndarray] = self._load()

    def _load(self) -> dict[str, np.ndarray]:
        if not self._cache_path.exists():
            return {}
        try:
            with np.load(self._cache_path) as data:
                keys = data["keys"]
                vecs = data["vecs"]
        except (OSError, ValueError, KeyError) as exc:
            logger.warning(f"DiskCachedEmbeddings: unreadable cache: {exc}")
            return {}
        return {str(k): vecs[i] for i, k in enumerate(keys)}

    def _save(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._cache)
        vecs = np.stack([self._cache[k] for k in keys]).astype("float32")
        np.savez(self._cache_path, keys=np.array(keys, dtype="U40"), vecs=vecs)

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        missing = [i for i, t in enumerate(texts) if _sha(t) not in self._cache]
        if missing:
            logger.info(
                f"DiskCachedEmbeddings: embedding {len(missing)} new / "
                f"{len(texts)} document(s) ({len(texts) - len(missing)} cached)"
            )
            fresh = self._embeddings.embed_documents([texts[i] for i in missing])
            for i, vec in zip(missing, fresh):
                self._cache[_sha(texts[i])] = np.asarray(vec, dtype=np.float32)
            self._save()
        return [self._cache[_sha(t)] for t in texts]

    def embed_query(self, text: str) -> Any:
        return self._embeddings.embed_query(text)


class HybridRanker:
    """
    Hybrid lexical + dense retrieval with RRF fusion and an optional reranker.

    Bundles the scoring primitives shared by two callers with opposite corpus
    lifetimes:

    * :class:`RelevantHistorySelector`, whose corpus (one chat thread's turns)
      changes every call, ranks an arbitrary doc list afresh each time via the
      stateless :meth:`bm25_ranking` / :meth:`dense_ranking` / :meth:`rrf_fuse`
      / :meth:`rerank` primitives.
    * A fixed instruction corpus calls :meth:`index` once and then
      :meth:`query` many times, reusing one BM25 index and one FAISS index
      rather than rebuilding them per query — the corpus is thousands of
      chunks, so a per-query rebuild would dominate runtime.

    Both paths share the RRF fusion, reranker hook, and content-hashed
    embedding cache, so lexical/dense/fusion behaviour is identical whichever
    entry point is used.

    Parameters
    ----------
    embeddings : Any | None
        LangChain ``Embeddings`` instance, or a provider string forwarded to
        ``langchain.embeddings.init_embeddings`` (e.g.
        ``"openai:text-embedding-3-small"``). ``None`` (default) disables the
        dense stage — ranking is BM25-only and fully offline.
    rrf_k : int
        Reciprocal Rank Fusion constant; 60 is the standard choice. Larger
        values flatten the difference between rank positions.
    reranker : Callable[[str, list[str]], list[float]] | None
        Optional second-stage reranker: given ``(query, docs)`` returns one
        relevance score per doc (higher = better). Applied to the RRF-fused
        shortlist; use for a cross-encoder or LLM judge.
    """

    def __init__(
        self,
        *,
        embeddings: Any | None = None,
        rrf_k: int = 60,
        reranker: Callable[[str, list[str]], list[float]] | None = None,
    ) -> None:
        if isinstance(embeddings, str):
            from langchain.embeddings import init_embeddings

            embeddings = init_embeddings(embeddings)
        self.rrf_k = rrf_k
        self._embeddings = embeddings
        self._reranker = reranker
        self._embedding_cache: dict[str, np.ndarray] = {}
        # Content-hashed tokenization cache (mirrors _embedding_cache): a
        # chat thread's turns are append-only during a run, so per-call BM25
        # re-tokenization only pays for documents not seen before.
        self._token_cache: dict[str, list[str]] = {}
        # Static-corpus state, populated by index(); unused in per-call mode.
        self._corpus: list[str] = []
        self._bm25: Any | None = None
        self._faiss: Any | None = None

    @property
    def has_dense(self) -> bool:
        return self._embeddings is not None

    @property
    def has_reranker(self) -> bool:
        return self._reranker is not None

    def _require_embeddings(self) -> Any:
        """The embeddings backend, or a clear error. Guarded by :attr:`has_dense`."""
        if self._embeddings is None:
            raise RuntimeError(
                "dense retrieval needs embeddings: construct HybridRanker with "
                "embeddings=, or check has_dense first"
            )
        return self._embeddings

    def _require_reranker(self) -> Callable[[str, list[str]], list[float]]:
        """The reranker hook, or a clear error. Guarded by :attr:`has_reranker`."""
        if self._reranker is None:
            raise RuntimeError(
                "rerank() needs a reranker: construct HybridRanker with "
                "reranker=, or check has_reranker first"
            )
        return self._reranker

    # ------------------------------------------------------------------
    # Stateless per-call ranking (corpus differs every call)
    # ------------------------------------------------------------------

    def _tokenize_cached(self, text: str) -> list[str]:
        """Tokenize *text*, memoised by content hash (see ``_token_cache``)."""
        key = _sha(text)
        tokens = self._token_cache.get(key)
        if tokens is None:
            tokens = _tokenize(text)
            self._token_cache[key] = tokens
        return tokens

    def bm25_ranking(
        self, docs: list[str], candidates: list[int], query: str
    ) -> list[int] | None:
        """Best-first candidate ranking, or ``None`` when BM25 has no signal."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            logger.warning("bm25_ranking: query produced no tokens; no signal")
            return None
        # "_" placeholder keeps a pathological empty doc from breaking the
        # index; it can never match a real query token.
        corpus = [self._tokenize_cached(docs[i]) or ["_"] for i in candidates]
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

    def dense_ranking(
        self, docs: list[str], candidates: list[int], query: str
    ) -> list[int] | None:
        """Best-first candidate ranking, or ``None`` when cosine has no signal."""
        vectors = self.embed_cached([docs[i] for i in candidates])
        query_vec = self._normalize(
            np.asarray(self._require_embeddings().embed_query(query), dtype=np.float32)
        )
        # One matrix-vector product replaces the FAISS index this method used
        # to build per call: vectors are already L2-normalised, so the inner
        # products are the same cosine scores IndexFlatIP produced. Remaining
        # ties break toward recency, matching bm25_ranking (FAISS left tie
        # order unspecified).
        sims = vectors @ query_vec
        if float(sims.max()) - float(sims.min()) < 1e-9:
            return None  # every doc tied — ordering would be arbitrary
        order = sorted(
            range(len(candidates)), key=lambda j: (-float(sims[j]), -candidates[j])
        )
        return [candidates[j] for j in order]

    def rrf_fuse(self, rankings: list[list[int]]) -> list[int]:
        scores: dict[int, float] = defaultdict(float)
        for ranking in rankings:
            for position, idx in enumerate(ranking):
                scores[idx] += 1.0 / (self.rrf_k + position + 1)
        # Ties break toward recency (higher turn index first).
        return sorted(scores, key=lambda i: (-scores[i], -i))

    def rerank(
        self, fused: list[int], docs: list[str], query: str, *, window_size: int
    ) -> list[int]:
        """Re-order the top ``window_size`` of *fused* with the reranker hook."""
        window = fused[:window_size]
        scores = self._require_reranker()(query, [docs[i] for i in window])
        order = sorted(range(len(window)), key=lambda j: (-scores[j], -window[j]))
        return [window[j] for j in order] + fused[len(window) :]

    # ------------------------------------------------------------------
    # Static-corpus ranking (index once, query many)
    # ------------------------------------------------------------------

    def index(self, docs: list[str]) -> None:
        """
        Build reusable BM25 (and dense, if enabled) indexes over a fixed
        corpus so repeated :meth:`query` calls do not rebuild them.
        """
        self._corpus = list(docs)
        logger.info(
            f"HybridRanker.index: {len(self._corpus)} document(s)  "
            f"dense={'on' if self.has_dense else 'off'}"
        )
        if not self._corpus:
            self._bm25 = None
            self._faiss = None
            return
        tokenized = [self._tokenize_cached(d) or ["_"] for d in self._corpus]
        self._bm25 = bm25s.BM25()
        self._bm25.index(tokenized, show_progress=False)
        if self._embeddings is not None:
            import faiss  # deferred: only the index-once/query-many path needs it

            vectors = self.embed_cached(self._corpus)
            self._faiss = faiss.IndexFlatIP(vectors.shape[1])
            self._faiss.add(vectors)
        else:
            self._faiss = None

    def query(self, text: str, *, top_k: int | None = None) -> list[int]:
        """
        Best-first corpus indices for *text*, reusing the indexes built by
        :meth:`index`. Returns an empty list when no scorer has signal —
        a fixed instruction corpus has no recency to fall back to, so
        irrelevant matches are simply dropped. Truncated to ``top_k`` when
        given.
        """
        if not self._corpus:
            return []
        if self._bm25 is None:
            raise RuntimeError("HybridRanker.index() must be called before query()")
        n = len(self._corpus)
        candidates = list(range(n))
        rankings: list[list[int]] = []

        query_tokens = _tokenize(text)
        if not query_tokens:
            logger.warning("HybridRanker.query: query produced no tokens; BM25 skipped")
        else:
            scores = self._bm25.get_scores(query_tokens)
            # Signal = any positive score. BM25 (Lucene idf) scores are >= 0
            # with 0 meaning "no term matched", so a max of 0 is genuinely no
            # signal — but tied *positive* scores (or a single-doc corpus,
            # where max always equals min) are real matches and must rank.
            # This differs from the per-call spread check in bm25_ranking,
            # where a history selector can fall back to recency instead.
            if float(scores.max()) > 1e-12:
                # No recency to prefer in a static corpus: break ties toward the
                # lower (earlier) index for a stable, deterministic order.
                rankings.append(sorted(candidates, key=lambda i: (-scores[i], i)))

        if self._faiss is not None:
            query_vec = self._normalize(
                np.asarray(
                    self._require_embeddings().embed_query(text), dtype=np.float32
                )
            )
            similarities, order = self._faiss.search(query_vec[None, :], n)
            if float(similarities.max()) - float(similarities.min()) >= 1e-9:
                rankings.append([int(j) for j in order[0] if j != -1])

        if not rankings:
            logger.debug("HybridRanker.query: no scorer has signal; empty result")
            return []
        fused = self.rrf_fuse(rankings)
        if self._reranker is not None:
            window = max(4 * top_k, top_k) if top_k else len(fused)
            fused = self.rerank(fused, self._corpus, text, window_size=window)
        return fused[:top_k] if top_k is not None else fused

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        return vec / max(float(np.linalg.norm(vec)), 1e-12)

    def embed_cached(self, texts: list[str]) -> np.ndarray:
        keys = [_sha(t) for t in texts]
        missing = [
            (k, t) for k, t in zip(keys, texts) if k not in self._embedding_cache
        ]
        if missing:
            logger.debug(
                f"embed_cached: embedding {len(missing)} new text(s) "
                f"({len(texts) - len(missing)} cached)"
            )
            new_vectors = self._require_embeddings().embed_documents(
                [t for _, t in missing]
            )
            for (key, _), vec in zip(missing, new_vectors):
                self._embedding_cache[key] = self._normalize(
                    np.asarray(vec, dtype=np.float32)
                )
        return np.stack([self._embedding_cache[k] for k in keys])


class RelevantHistorySelector:
    """
    Select the history turn pairs most relevant to the current request.

    Thin orchestration over a :class:`HybridRanker`: it groups the thread into
    turns, ranks the candidate turns with the shared BM25/dense/RRF stack, and
    layers the history-specific policy on top — an always-kept recency tail, a
    recency fallback when no scorer has signal, and chronological-order output.

    Parameters
    ----------
    top_k : int
        Total turn pairs to keep per call (including the always-kept tail).
    always_keep_last : int
        Most recent pairs kept unconditionally, for continuity.
    max_tokens : int | None
        Optional token envelope for the selected history. Relevance-ranked
        pairs are packed best-first while they fit; a pair that would
        overflow is skipped so smaller relevant pairs can still fill the
        remaining budget. The always-kept tail is exempt — it is included
        even when it alone exceeds the budget. ``None`` (default) keeps
        pure ``top_k`` counting. When set, selection also runs on short
        histories that ``top_k`` alone would have passed through whole.
    token_counter : Callable[[str], int] | None
        Counts tokens for the ``max_tokens`` budget. ``None`` (default)
        uses the offline ~4-chars/token estimate
        (:func:`memory.turns.approx_token_count`).
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
        max_tokens: int | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if always_keep_last < 0:
            raise ValueError(f"always_keep_last must be >= 0, got {always_keep_last}")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        self.top_k = top_k
        self.always_keep_last = always_keep_last
        self.rrf_k = rrf_k
        self.max_tokens = max_tokens
        self._count_tokens = token_counter or approx_token_count
        self._ranker = HybridRanker(
            embeddings=embeddings, rrf_k=rrf_k, reranker=reranker
        )
        logger.info(
            f"RelevantHistorySelector: top_k={top_k}  "
            f"always_keep_last={always_keep_last}  "
            f"max_tokens={max_tokens}  "
            f"dense={'on' if self._ranker.has_dense else 'off'}  "
            f"reranker={'on' if self._ranker.has_reranker else 'off'}  rrf_k={rrf_k}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self, history: list[BaseMessage], query: str) -> list[BaseMessage]:
        """
        Return the subset of *history* to prompt for *query*: the
        ``top_k`` most relevant turn pairs (always including the last
        ``always_keep_last``), flattened back to messages in their
        original chronological order. With ``max_tokens`` set, ranked
        pairs are additionally packed into that token envelope.
        """
        if not history:
            return []
        turns = group_turns(history)
        if self.max_tokens is None and len(turns) <= self.top_k:
            return list(history)

        # With a token budget the selection body also runs on short
        # histories, so the tail must never exceed the turn count.
        keep_last = min(self.always_keep_last, self.top_k, len(turns))
        tail = list(range(len(turns) - keep_last, len(turns)))
        candidates = list(range(len(turns) - keep_last))
        slots = self.top_k - keep_last

        docs = [turn_text(t) for t in turns]
        chosen: list[int] = []
        if slots > 0 and candidates:
            fused = self._fuse(docs, candidates, query)
            if fused is None:
                logger.debug("select: no scorer has signal; recency fallback")
                fused = sorted(candidates, reverse=True)
            if self._ranker.has_reranker:
                fused = self._ranker.rerank(
                    fused,
                    docs,
                    query,
                    window_size=max(4 * self.top_k, self.top_k),
                )
            chosen = self._pack(fused, docs, tail, slots)

        selected = sorted(set(chosen) | set(tail))
        logger.debug(
            f"select: kept {len(selected)}/{len(turns)} turn pair(s)  "
            f"relevant={sorted(chosen)}  tail={tail}"
        )
        return [m for i in selected for m in turns[i]]

    def _pack(
        self, fused: list[int], docs: list[str], tail: list[int], slots: int
    ) -> list[int]:
        """Take up to *slots* pairs from *fused*, best-first.

        Without a budget this is a plain slice. With one, the always-kept
        tail is charged first (but included regardless — it is a floor
        guarantee, not a candidate), then ranked pairs are packed while
        they fit; an oversized pair is skipped, not a stopping point, so
        smaller relevant pairs behind it can still use the budget.
        """
        if self.max_tokens is None:
            return fused[:slots]
        budget = self.max_tokens - sum(self._count_tokens(docs[i]) for i in tail)
        chosen: list[int] = []
        skipped = 0
        for idx in fused:
            if len(chosen) >= slots:
                break
            cost = self._count_tokens(docs[idx])
            if cost <= budget:
                chosen.append(idx)
                budget -= cost
            else:
                skipped += 1
        if skipped:
            logger.debug(
                f"_pack: skipped {skipped} ranked pair(s) over the "
                f"max_tokens={self.max_tokens} budget"
            )
        return chosen

    # ------------------------------------------------------------------
    # Internals — delegate scoring to the shared HybridRanker
    # ------------------------------------------------------------------

    def _fuse(
        self, docs: list[str], candidates: list[int], query: str
    ) -> list[int] | None:
        """RRF-fused candidate ranking, or ``None`` when no scorer has signal."""
        maybe_rankings = [self._ranker.bm25_ranking(docs, candidates, query)]
        if self._ranker.has_dense:
            maybe_rankings.append(self._ranker.dense_ranking(docs, candidates, query))
        rankings = [r for r in maybe_rankings if r is not None]
        if not rankings:
            return None
        return self._ranker.rrf_fuse(rankings)

    # Kept for direct unit-test access; each forwards to the shared ranker.
    def _bm25_ranking(
        self, docs: list[str], candidates: list[int], query: str
    ) -> list[int] | None:
        return self._ranker.bm25_ranking(docs, candidates, query)

    def _dense_ranking(
        self, docs: list[str], candidates: list[int], query: str
    ) -> list[int] | None:
        return self._ranker.dense_ranking(docs, candidates, query)

    def _rrf_fuse(self, rankings: list[list[int]]) -> list[int]:
        return self._ranker.rrf_fuse(rankings)
