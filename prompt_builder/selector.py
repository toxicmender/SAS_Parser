"""Instruction retrieval: pipeline item -> relevant instruction chunks.

See prompt_builder/README.md.

Two-stage selection per pipeline item:

1. **Construct lookup (deterministic).** The item's metadata already names its
   constructs (functions, CALL routines, PROCs, global/macro statements). Those
   map straight to the reference section documenting each — an exact hit no
   ranker can beat. Hazard-linked constructs (SYMPUT/SYMGET, %GOTO, %ABORT) are
   fetched first and never stop-listed; a stop-list keeps trivial ubiquitous
   functions (PUT, INPUT, SUM, …) from flooding the budget.
2. **Hybrid ranking (topical).** :class:`~memory.relevance.HybridRanker` (BM25
   always, dense optional) over the whole chunk corpus surfaces guidance no
   title lookup can find — target-platform sections ("DataFrames and SQL",
   "Structured Streaming") keyed off a free-text query built from the item.

Results fill a word budget in priority order (pinned -> hazard constructs ->
other constructs -> topical), dropping whole chunks at the tail rather than
truncating. Nothing relevant -> an empty list, so the prompt carries no
guidance block (irrelevant reference pages are worse than none).

Logger name: ``prompt_builder.selector``.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from memory.relevance import HybridRanker

from .models import ConstructKey, InstructionChunk

logger = logging.getLogger(__name__)


# Ubiquitous, self-evident functions whose reference section carries no
# translation insight worth spending budget on. Never stop-listed if also a
# hazard construct (below), but none of these are.
_STOP_FUNCTION_NAMES = frozenset(
    {
        "put", "input", "sum", "min", "max", "mean", "n", "nmiss", "abs", "int",
        "ceil", "floor", "round", "length", "trim", "trimn", "strip", "left",
        "right", "upcase", "lowcase", "propcase", "compress", "cat", "cats",
        "catx", "catt", "catq", "coalesce", "coalescec", "missing",
    }
)
DEFAULT_STOP_CONSTRUCTS: frozenset[ConstructKey] = frozenset(
    ConstructKey(kind="function", name=n) for n in _STOP_FUNCTION_NAMES
)

# Constructs with silent-error potential the system prompt already flags:
# always pull their reference section, always ahead of ordinary hits.
_HAZARD_CONSTRUCTS: tuple[tuple[str, str], ...] = (
    ("call_routine", "symput"),
    ("call_routine", "symputx"),
    ("call_routine", "symget"),
    ("call_routine", "execute"),
    ("macro_statement", "goto"),
    ("macro_statement", "abort"),
    ("macro_function", "sysfunc"),
)
DEFAULT_HAZARD_CONSTRUCTS: frozenset[ConstructKey] = frozenset(
    ConstructKey(kind=k, name=n) for k, n in _HAZARD_CONSTRUCTS
)


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class DiskCachedEmbeddings:
    """
    Wrap a LangChain ``Embeddings`` with an on-disk document-embedding cache.

    Embedding the 6–9k-chunk corpus is the one genuinely expensive step of
    turning dense retrieval on; the vectors never change between runs. Document
    embeddings are memoised to an ``.npz`` keyed by content SHA-1 (queries,
    which vary every call, are passed straight through). Sits under
    :class:`~memory.relevance.HybridRanker`'s in-process cache, so a warm disk
    cache means no model call at all on subsequent runs.
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


class InstructionSelector:
    """
    Select the instruction chunks most relevant to a pipeline item.

    Parameters
    ----------
    chunks : Iterable[InstructionChunk]
        The full instruction corpus (from :class:`CorpusLoader`).
    embeddings : Any | None
        LangChain ``Embeddings`` (or provider string) to enable dense topical
        retrieval; ``None`` (default) is BM25-only and fully offline.
    embedding_cache_path : str | None
        Where to persist document embeddings when dense retrieval is on.
    rrf_k, reranker :
        Forwarded to :class:`~memory.relevance.HybridRanker`.
    pinned_sections : Iterable[str]
        Section-path substrings (case-insensitive) whose chunks are always
        injected first, within budget.
    stop_constructs, hazard_constructs : Iterable[ConstructKey]
        Override the default stop-list / hazard set.
    """

    def __init__(
        self,
        chunks: Iterable[InstructionChunk],
        *,
        embeddings: Any | None = None,
        embedding_cache_path: str | None = None,
        rrf_k: int = 60,
        reranker: Callable[[str, list[str]], list[float]] | None = None,
        pinned_sections: Iterable[str] = (),
        stop_constructs: Iterable[ConstructKey] = DEFAULT_STOP_CONSTRUCTS,
        hazard_constructs: Iterable[ConstructKey] = DEFAULT_HAZARD_CONSTRUCTS,
    ) -> None:
        self._chunks = list(chunks)
        self._wc = [len(c.text.split()) for c in self._chunks]
        self._stop = frozenset(stop_constructs)
        self._hazard = frozenset(hazard_constructs)

        self._by_construct: dict[ConstructKey, list[int]] = defaultdict(list)
        for i, chunk in enumerate(self._chunks):
            for key in chunk.construct_keys:
                self._by_construct[key].append(i)

        pins = [p.lower() for p in pinned_sections]
        self._pinned = [
            i
            for i, c in enumerate(self._chunks)
            if any(p in c.section_path.lower() for p in pins)
        ]

        if embeddings is not None and embedding_cache_path is not None:
            embeddings = DiskCachedEmbeddings(embeddings, embedding_cache_path)
        self._ranker = HybridRanker(
            embeddings=embeddings, rrf_k=rrf_k, reranker=reranker
        )
        self._ranker.index([c.text for c in self._chunks])
        logger.info(
            f"InstructionSelector: {len(self._chunks)} chunk(s)  "
            f"{len(self._by_construct)} construct key(s)  "
            f"{len(self._pinned)} pinned  "
            f"dense={'on' if embeddings is not None else 'off'}"
        )

    def select(
        self,
        query: str,
        constructs: Iterable[ConstructKey] = (),
        *,
        max_words: int = 1500,
        top_k: int = 6,
    ) -> list[InstructionChunk]:
        """
        Chunks to inject for one item, in priority order (pinned -> hazard
        constructs -> other constructs -> topical), filling ``max_words`` and
        taking at most ``top_k`` topical chunks. Empty when nothing is relevant.
        """
        chosen: list[int] = []
        chosen_set: set[int] = set()
        used = 0

        def add(idx: int) -> bool:
            nonlocal used
            if idx in chosen_set:
                return False
            if used + self._wc[idx] > max_words:
                return False
            chosen.append(idx)
            chosen_set.add(idx)
            used += self._wc[idx]
            return True

        for idx in self._pinned:
            add(idx)

        hazard, normal = self._construct_hits(constructs)
        for idx in hazard:
            add(idx)
        for idx in normal:
            add(idx)

        topical_added = 0
        if top_k > 0 and query.strip():
            for idx in self._ranker.query(query):
                if topical_added >= top_k:
                    break
                if add(idx):
                    topical_added += 1

        logger.debug(
            f"select: {len(chosen)} chunk(s)  words={used}/{max_words}  "
            f"pinned={len(self._pinned)}  hazard={len(hazard)}  "
            f"construct={len(normal)}  topical={topical_added}"
        )
        return [self._chunks[i] for i in chosen]

    def _construct_hits(
        self, constructs: Iterable[ConstructKey]
    ) -> tuple[list[int], list[int]]:
        """Primary-chunk indices for the item's constructs, hazard set apart."""
        hazard: list[int] = []
        normal: list[int] = []
        seen: set[ConstructKey] = set()
        for key in constructs:
            if key in seen:
                continue
            seen.add(key)
            idxs = self._by_construct.get(key)
            if not idxs:
                continue
            if key in self._hazard:
                hazard.append(idxs[0])
            elif key in self._stop:
                continue
            else:
                normal.append(idxs[0])
        return hazard, normal
