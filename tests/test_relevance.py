"""
Tests for memory.relevance — BM25 + dense retrieval, RRF fusion, and
relevance-based history selection, plus its wiring into SasLLMPipeline.

Fully offline: the dense stage is exercised with a deterministic
vocabulary-group embedding fake (synonyms share a dimension), and the
pipeline test injects a recording chat-model stub. bm25s / faiss / numpy
are project dependencies, so no test is conditionally skipped.
"""

from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from memory.relevance import HybridRanker, RelevantHistorySelector, _group_turns


# ---------------------------------------------------------------------------
# Test doubles / helpers
# ---------------------------------------------------------------------------


def _mk_history(*pair_texts: str) -> list:
    """One (human, AI) pair per text; the AI reply echoes the pair index."""
    history: list = []
    for i, text in enumerate(pair_texts):
        history.append(HumanMessage(text))
        history.append(AIMessage(f"reply {i}"))
    return history


class _VocabEmbeddings:
    """
    Deterministic embeddings: one dimension per synonym group, so 'revenue'
    and 'sales' land on the same axis while sharing no BM25 token.
    """

    groups = [
        ("sales", "revenue"),
        ("customer", "client"),
        ("inventory", "stock"),
        ("forecast", "prediction"),
    ]

    def __init__(self) -> None:
        self.documents_embedded: list[str] = []

    def _vec(self, text: str) -> list[float]:
        tokens = set(re.findall(r"[a-z0-9_]+", text.lower()))
        return [float(sum(t in tokens for t in group)) for group in self.groups]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents_embedded.extend(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


# ---------------------------------------------------------------------------
# Turn grouping
# ---------------------------------------------------------------------------


def test_group_turns_pairs_human_with_following_ai():
    history = _mk_history("a", "b")
    turns = _group_turns(history)
    assert len(turns) == 2
    assert [type(m).__name__ for m in turns[0]] == ["HumanMessage", "AIMessage"]


def test_group_turns_never_drops_messages():
    # Leading AI message and a double-AI turn are grouped, not lost.
    history = [
        AIMessage("stray leading"),
        HumanMessage("q1"),
        AIMessage("a1"),
        AIMessage("a1-continued"),
        HumanMessage("q2"),
    ]
    turns = _group_turns(history)
    assert sum(len(t) for t in turns) == len(history)
    assert len(turns) == 3  # stray, q1-turn, q2-turn


# ---------------------------------------------------------------------------
# select() — BM25-only mode
# ---------------------------------------------------------------------------


def test_short_history_returned_unchanged():
    history = _mk_history("data work.a; run;", "proc print; run;")
    selector = RelevantHistorySelector(top_k=6)
    assert selector.select(history, "anything") == history


def test_bm25_selects_lexical_match_over_recency():
    history = _mk_history(
        "data work.sales_summary_xq; set raw.orders; run;",  # pair 0 <- relevant
        "proc means data=work.inventory; run;",  # pair 1
        "proc freq data=work.customers; run;",  # pair 2
        "libname raw '/mnt/raw';",  # pair 3 <- tail
    )
    selector = RelevantHistorySelector(top_k=2, always_keep_last=1)
    query = "translate: proc print data=work.sales_summary_xq; run;"

    selected = selector.select(history, query)

    texts = [str(m.content) for m in selected]
    assert len(selected) == 4  # 2 pairs
    assert "sales_summary_xq" in texts[0]  # relevant pair kept...
    assert "libname raw" in texts[2]  # ...plus the always-kept tail
    # chronological order preserved: pair 0 before pair 3
    assert texts == [
        history[0].content,
        history[1].content,
        history[6].content,
        history[7].content,
    ]


def test_no_lexical_overlap_falls_back_to_recency_ties():
    history = _mk_history("alpha one", "beta two", "gamma three", "delta four")
    selector = RelevantHistorySelector(top_k=2, always_keep_last=1)
    selected = selector.select(history, "zzz completely unrelated qqq")

    # All BM25 scores tie at 0 -> recency wins: pairs 2 and 3 kept.
    texts = [str(m.content) for m in selected]
    assert texts[0] == "gamma three"
    assert texts[2] == "delta four"


def test_empty_query_tokens_fall_back_to_recency():
    history = _mk_history("alpha", "beta", "gamma")
    selector = RelevantHistorySelector(top_k=2, always_keep_last=1)
    selected = selector.select(history, "!!! ???")  # tokenizes to nothing

    texts = [str(m.content) for m in selected]
    assert texts[0] == "beta" and texts[2] == "gamma"


def test_always_keep_last_zero_is_pure_relevance():
    history = _mk_history(
        "data work.match_me_xq; run;",
        "proc means; run;",
        "proc freq; run;",
    )
    selector = RelevantHistorySelector(top_k=1, always_keep_last=0)
    selected = selector.select(history, "work.match_me_xq")

    assert len(selected) == 2
    assert "match_me_xq" in str(selected[0].content)


# ---------------------------------------------------------------------------
# Dense stage (FAISS) and RRF fusion
# ---------------------------------------------------------------------------


def test_dense_ranking_finds_semantic_match():
    docs = [
        "quarterly revenue tables by region",  # semantic match for 'sales'
        "warehouse shelving layout notes",
        "employee onboarding checklist",
    ]
    selector = RelevantHistorySelector(top_k=1, embeddings=_VocabEmbeddings())
    ranking = selector._dense_ranking(docs, [0, 1, 2], "sales figures please")
    assert ranking[0] == 0


def test_hybrid_selection_keeps_semantic_match_bm25_misses():
    history = _mk_history(
        "update the quarterly revenue tables",  # pair 0: dense match only
        "warehouse shelving layout",  # pair 1
        "employee onboarding checklist",  # pair 2
        "libname raw '/mnt/raw';",  # pair 3 <- tail
    )
    # 'sales' ~ 'revenue' via embeddings; no token (not even a stopword)
    # shared with pair 0, so BM25 alone cannot find it.
    query = "recompute sales figures"

    lexical_only = RelevantHistorySelector(top_k=2, always_keep_last=1)
    hybrid = RelevantHistorySelector(
        top_k=2, always_keep_last=1, embeddings=_VocabEmbeddings()
    )

    # BM25 alone ties at 0 and falls back to recency (pair 2)...
    assert "revenue" not in str(lexical_only.select(history, query)[0].content)
    # ...the dense stage recovers the synonym pair.
    assert "revenue" in str(hybrid.select(history, query)[0].content)


def test_bm25_ranking_reports_no_signal_on_zero_overlap():
    selector = RelevantHistorySelector(top_k=2)
    docs = ["alpha one", "beta two", "gamma three"]
    assert selector._bm25_ranking(docs, [0, 1, 2], "zzz unrelated") is None
    assert selector._bm25_ranking(docs, [0, 1, 2], "beta") is not None


def test_rrf_fuse_prefers_doc_ranked_well_by_both():
    selector = RelevantHistorySelector(top_k=2)
    # doc 5 is #1 in one ranking and #2 in the other; doc 9 only leads one.
    fused = selector._rrf_fuse([[5, 9, 3], [9, 5, 3]])
    assert fused[0] in (5, 9)  # both lead one list -> tie on RRF score...
    assert fused[0] == 9  # ...broken toward recency (higher index)
    assert fused[-1] == 3


def test_embedding_cache_embeds_each_pair_once():
    embeddings = _VocabEmbeddings()
    history = _mk_history(
        "sales report", "stock levels", "client emails", "forecast model"
    )
    selector = RelevantHistorySelector(
        top_k=2, always_keep_last=1, embeddings=embeddings
    )

    selector.select(history, "revenue query one")
    first_pass = len(embeddings.documents_embedded)
    selector.select(history, "prediction query two")

    assert first_pass == 3  # candidates only (tail pair is never embedded)
    assert len(embeddings.documents_embedded) == first_pass  # all cached


# ---------------------------------------------------------------------------
# HybridRanker — static-corpus mode (index once, query many)
# ---------------------------------------------------------------------------


def test_hybrid_ranker_query_ranks_lexical_match_first():
    ranker = HybridRanker()
    ranker.index(
        [
            "proc means computes descriptive statistics",  # 0
            "the intnx function advances a sas date",  # 1 <- match
            "libname assigns a library reference",  # 2
        ]
    )
    ranking = ranker.query("how does intnx advance a date")
    assert ranking[0] == 1


def test_hybrid_ranker_query_no_signal_returns_empty():
    ranker = HybridRanker()
    ranker.index(["alpha one", "beta two", "gamma three"])
    assert ranker.query("zzz completely unrelated qqq") == []


def test_hybrid_ranker_empty_query_returns_empty():
    ranker = HybridRanker()
    ranker.index(["alpha one", "beta two"])
    assert ranker.query("!!! ???") == []  # tokenizes to nothing, no dense stage


def test_hybrid_ranker_top_k_truncates():
    ranker = HybridRanker()
    ranker.index(["sas macro %let statement", "macro variable resolution", "macro"])
    ranking = ranker.query("macro", top_k=2)
    assert len(ranking) == 2


def test_hybrid_ranker_empty_corpus_queries_empty():
    ranker = HybridRanker()
    ranker.index([])
    assert ranker.query("anything") == []


def test_hybrid_ranker_single_doc_corpus_matches():
    # max == min always holds with one doc; a positive score is still signal.
    ranker = HybridRanker()
    ranker.index(["the intnx function advances a sas date"])
    assert ranker.query("advance a date with intnx") == [0]
    assert ranker.query("zzz unrelated") == []  # zero score stays no-signal


def test_hybrid_ranker_tied_positive_scores_rank_stably():
    ranker = HybridRanker()
    ranker.index(["macro alpha", "macro beta"])  # identical BM25 for "macro"
    assert ranker.query("macro") == [0, 1]  # tie broken toward earlier index


def test_hybrid_ranker_query_before_index_raises():
    ranker = HybridRanker()
    ranker._corpus = ["never indexed"]  # simulate corpus set without index()
    with pytest.raises(RuntimeError):
        ranker.query("anything")


def test_hybrid_ranker_dense_recovers_synonym():
    ranker = HybridRanker(embeddings=_VocabEmbeddings())
    ranker.index(
        [
            "warehouse shelving layout notes",  # 0
            "quarterly revenue tables by region",  # 1 <- synonym of 'sales'
            "employee onboarding checklist",  # 2
        ]
    )
    ranking = ranker.query("sales figures please")
    assert ranking[0] == 1


def test_hybrid_ranker_index_reuses_embedding_cache():
    embeddings = _VocabEmbeddings()
    ranker = HybridRanker(embeddings=embeddings)
    docs = ["sales report", "stock levels", "client emails"]
    ranker.index(docs)
    embedded_after_index = len(embeddings.documents_embedded)
    ranker.query("revenue query")  # query embeds only the query, not docs
    assert embedded_after_index == 3
    assert len(embeddings.documents_embedded) == 3


# ---------------------------------------------------------------------------
# Custom reranker hook
# ---------------------------------------------------------------------------


def test_reranker_overrides_fused_order():
    history = _mk_history("alpha one", "beta two", "gamma three", "delta tail")

    def prefer_beta(query: str, docs: list[str]) -> list[float]:
        return [1.0 if "beta" in d else 0.0 for d in docs]

    selector = RelevantHistorySelector(
        top_k=2, always_keep_last=1, reranker=prefer_beta
    )
    selected = selector.select(history, "zzz no lexical signal")

    assert str(selected[0].content) == "beta two"
    assert str(selected[2].content) == "delta tail"


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"always_keep_last": -1}])
def test_invalid_construction_raises(kwargs):
    with pytest.raises(ValueError):
        RelevantHistorySelector(**kwargs)


# ---------------------------------------------------------------------------
# Pipeline wiring — relevance selection replaces window trimming
# ---------------------------------------------------------------------------


class _RecordingChatModel:
    """Chat-model stub that records every prompted message list."""

    def __init__(self) -> None:
        self.prompts: list[list] = []

    def invoke(self, messages, config=None):
        self.prompts.append(list(messages))
        return AIMessage(f"resp {len(self.prompts)}")


def test_pipeline_prompts_relevant_pair_not_recency_window():
    from chunker.models import SasChunk, SasChunkKind, SasChunkMetadata
    from chunker.pipeline import SasLLMPipeline
    from memory.short_mem import DatabricksMemory

    def _mk_chunk(chunk_id: str, text: str) -> SasChunk:
        return SasChunk(
            chunk_id=chunk_id,
            source_id="etl.sas",
            text=text,
            kind=SasChunkKind.DATA_STEP,
            title=f"Step {chunk_id}",
            start_line=1,
            end_line=3,
            start_char=0,
            end_char=len(text),
            metadata=SasChunkMetadata(),
        )

    llm = _RecordingChatModel()
    pipeline = SasLLMPipeline(
        model="unused-because-llm-injected",
        memory=DatabricksMemory(),
        llm=llm,
        window_k=None,
        history_selector=RelevantHistorySelector(top_k=2, always_keep_last=1),
    )
    chunks = [
        _mk_chunk("c1", "data work.zzunique_first_xq; run;"),  # relevant to c4
        _mk_chunk("c2", "proc means data=work.midstream_a; run;"),
        _mk_chunk("c3", "proc freq data=work.midstream_b; run;"),
        _mk_chunk("c4", "proc print data=work.zzunique_first_xq; run;"),
    ]
    pipeline._process(items=chunks, diagnostics=[], thread_id="run::etl.sas")

    # 4th call: system + 2 selected pairs (4 msgs) + current human = 6.
    final_prompt = llm.prompts[3]
    assert len(final_prompt) == 6
    prompt_text = "\n".join(str(m.content) for m in final_prompt)
    assert "zzunique_first_xq" in prompt_text  # pair for c1 retrieved by BM25
    assert "midstream_b" in prompt_text  # c3 pair kept as recency tail
    assert "midstream_a" not in prompt_text  # c2 pair dropped

    # Storage is untouched by prompt-side selection: all 8 messages kept.
    assert len(pipeline.get_thread_messages("run::etl.sas")) == 8
