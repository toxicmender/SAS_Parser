"""
Tests for prompt_builder.selector — construct lookup (hazard-first, stop-list),
topical HybridRanker retrieval, budget/priority filling, pinned sections, and
the on-disk embedding cache.

Fully offline: BM25 is a project dependency; the dense stage uses a
deterministic fake embeddings model.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from memory.relevance import DiskCachedEmbeddings
from prompt_builder.models import (
    ConstructKey,
    DocRole,
    InstructionChunk,
    SelectionTier,
)
from prompt_builder.selector import InstructionSelector
from prompt_builder.user_instructions import UserInstructionSet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    chunk_id: str,
    section_path: str,
    text: str,
    *,
    keys: list[ConstructKey] | None = None,
    role: DocRole = DocRole.SAS_REFERENCE,
) -> InstructionChunk:
    return InstructionChunk(
        chunk_id=chunk_id,
        doc_id="doc",
        section_path=section_path,
        text=f"{section_path}\n\n{text}",
        page_start=1,
        page_end=1,
        role=role,
        construct_keys=keys or [],
    )


def _words(n: int, token: str = "w") -> str:
    return " ".join(f"{token}{i}" for i in range(n))


INTNX = ConstructKey(kind="function", name="intnx")
SQL = ConstructKey(kind="proc", name="sql")
SYMPUT = ConstructKey(kind="call_routine", name="symput")
PUT = ConstructKey(kind="function", name="put")


def _corpus() -> list[InstructionChunk]:
    return [
        _chunk("c0", "Funcs > INTNX Function", "advances a sas date " + _words(40), keys=[INTNX]),
        _chunk("c1", "Procs > SQL Procedure", "ansi sql queries " + _words(40), keys=[SQL]),
        _chunk("c2", "CALL Routines > SYMPUT Routine", "macro symbol scope hazard " + _words(40), keys=[SYMPUT]),
        _chunk("c3", "Funcs > PUT Function", "writes a value " + _words(40), keys=[PUT]),
        _chunk("c4", "Spark > DataFrames and SQL", "dataframe join merge across partitions " + _words(40), role=DocRole.TARGET_GUIDE),
        _chunk("c5", "Guidelines > Output Format", "always return structured markdown " + _words(40)),
    ]


# ---------------------------------------------------------------------------
# Construct lookup
# ---------------------------------------------------------------------------


def test_construct_lookup_returns_exact_section():
    sel = InstructionSelector(_corpus())
    out = sel.select("zzz no topical overlap qqq", [INTNX])
    assert [c.chunk_id for c in out] == ["c0"]


def test_stop_listed_construct_is_skipped():
    sel = InstructionSelector(_corpus())
    # PUT is stop-listed and the query has no topical signal -> nothing.
    out = sel.select("zzz unrelated qqq", [PUT])
    assert out == []


def test_hazard_construct_ordered_before_ordinary():
    sel = InstructionSelector(_corpus())
    out = sel.select("zzz", [INTNX, SYMPUT])
    ids = [c.chunk_id for c in out]
    assert ids.index("c2") < ids.index("c0")  # SYMPUT (hazard) before INTNX


def test_hazard_construct_wins_tight_budget():
    sel = InstructionSelector(_corpus())
    # Each chunk ~43 words; budget fits only one.
    out = sel.select("zzz", [INTNX, SYMPUT], max_words=50)
    assert [c.chunk_id for c in out] == ["c2"]


def test_multi_window_construct_returns_all_windows_breadth_first():
    big = ConstructKey(kind="function", name="bigfn")
    corpus = _corpus() + [
        _chunk("w0", "Funcs > BIGFN Function", "window one " + _words(30), keys=[big]),
        _chunk("w1", "Funcs > BIGFN Function", "window two " + _words(30), keys=[big]),
    ]
    sel = InstructionSelector(corpus)
    out = sel.select("zzz", [big, INTNX], max_words=10_000, top_k=0)
    ids = [c.chunk_id for c in out]
    # All windows of BIGFN present, but INTNX's primary section comes before
    # BIGFN's second window (breadth-first interleave).
    assert set(ids) == {"w0", "w1", "c0"}
    assert ids.index("c0") < ids.index("w1")


# ---------------------------------------------------------------------------
# Topical ranking
# ---------------------------------------------------------------------------


def test_topical_surfaces_target_guide_without_construct():
    sel = InstructionSelector(_corpus())
    out = sel.select("dataframe join merge partitions", [])
    assert "c4" in [c.chunk_id for c in out]


def test_no_signal_returns_empty():
    sel = InstructionSelector(_corpus())
    assert sel.select("zzzq unrelated gibberish", []) == []


def test_top_k_caps_topical_results():
    sel = InstructionSelector(_corpus())
    # A query that hits many chunks lexically ("sql" in c1 and c4).
    out = sel.select("sql", [], top_k=1)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Pinned sections
# ---------------------------------------------------------------------------


def test_pinned_section_always_included_first():
    sel = InstructionSelector(_corpus(), pinned_sections=["Output Format"])
    out = sel.select("zzz nothing matches", [])
    assert out and out[0].chunk_id == "c5"


def test_pinned_plus_construct():
    sel = InstructionSelector(_corpus(), pinned_sections=["Output Format"])
    out = sel.select("zzz", [INTNX])
    ids = [c.chunk_id for c in out]
    assert ids[0] == "c5"  # pinned first
    assert "c0" in ids  # construct hit still present


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def test_budget_drops_whole_chunks_without_truncating():
    sel = InstructionSelector(_corpus())
    out = sel.select("zzz", [INTNX, SQL, SYMPUT], max_words=50)
    assert len(out) == 1  # only one ~43-word chunk fits
    # returned chunk is intact (not truncated)
    assert len(out[0].text.split()) > 40


# ---------------------------------------------------------------------------
# User-instruction tiers
# ---------------------------------------------------------------------------

USER_RULES = """\
## Output rules
Return fenced pyspark blocks with a risk table.

## [when: proc:sql] SQL rules
Prefer broadcast joins qqxyzzy.

## [topic] Partitioning guidance
Repartition dataframe join merge partitions guidance.
"""


def _user_selector(**kwargs) -> InstructionSelector:
    return InstructionSelector(
        _corpus(),
        user_instructions=UserInstructionSet.from_text(USER_RULES),
        **kwargs,
    )


def test_user_always_injected_even_with_no_signal():
    sel = _user_selector()
    out = sel.select("zzz nothing matches", [])
    assert [c.chunk_id for c in out] == ["user::c0000"]


def test_user_conditional_included_iff_constructs_match():
    sel = _user_selector()
    with_sql = [c.chunk_id for c in sel.select("zzz", [SQL])]
    without = [c.chunk_id for c in sel.select("zzz", [INTNX])]
    assert "user::c0001" in with_sql
    assert "user::c0001" not in without


def test_unmatched_conditional_never_surfaces_topically():
    sel = _user_selector()
    # The query lexically nails the conditional chunk's body ("qqxyzzy"),
    # but its scope is when:proc:sql and the item has no matching construct.
    out = sel.select("broadcast joins qqxyzzy", [])
    assert "user::c0001" not in [c.chunk_id for c in out]


def test_full_tier_ordering():
    sel = _user_selector(pinned_sections=["Output Format"])
    out = sel.select(
        "dataframe join merge partitions", [SQL, SYMPUT], top_k=2
    )
    assert [c.chunk_id for c in out] == [
        "user::c0000",  # 1. user always
        "user::c0001",  # 2. user conditional (proc:sql matched)
        "c5",  # 3. reference pinned (Guidelines > Output Format)
        "c2",  # 4. hazard construct (SYMPUT)
        "c1",  # 5. other construct (SQL Procedure)
        "user::c0002",  # 6. user topical
        "c4",  # 7. reference topical (Spark > DataFrames and SQL)
    ]


def test_user_topic_partition_beats_higher_scoring_reference():
    sel = _user_selector()
    # "across w0 w1" pushes the reference chunk's BM25 score above the user
    # topic chunk's; the partition must still order the user chunk first.
    out = sel.select("dataframe join merge across partitions w0 w1", [], top_k=2)
    ids = [c.chunk_id for c in out]
    assert ids.index("user::c0002") < ids.index("c4")


def test_top_k_caps_user_and_reference_topical_together():
    sel = _user_selector()
    out = sel.select("dataframe join merge partitions", [], top_k=1)
    ids = [c.chunk_id for c in out]
    assert "user::c0002" in ids  # the one topical slot goes to the user chunk
    assert "c4" not in ids


def test_user_max_words_caps_user_block_only(caplog):
    import logging

    two_rules = (
        "## Rule one\nalpha beta gamma delta epsilon.\n\n"
        "## Rule two\nzeta eta theta iota kappa.\n"
    )
    sel = InstructionSelector(
        _corpus(),
        user_instructions=UserInstructionSet.from_text(two_rules),
        user_max_words=9,  # each rule chunk is ~7 words; only one fits
    )
    with caplog.at_level(logging.WARNING, logger="prompt_builder.selector"):
        out = sel.select("zzz", [INTNX], max_words=10_000)
    ids = [c.chunk_id for c in out]
    assert "user::c0000" in ids  # first rule within the user cap
    assert "user::c0001" not in ids  # second rule over the user cap
    assert "c0" in ids  # reference chunks unaffected by the user cap
    assert "user_max_words=9" in caplog.text


def test_user_always_overflow_warns(caplog):
    import logging

    sel = _user_selector()
    with caplog.at_level(logging.WARNING, logger="prompt_builder.selector"):
        out = sel.select("zzz", [], max_words=5)
    assert out == []
    assert "does not fit budget" in caplog.text


# ---------------------------------------------------------------------------
# select_detailed — tier/construct provenance
# ---------------------------------------------------------------------------


def test_select_detailed_tags_every_tier():
    sel = _user_selector(pinned_sections=["Output Format"])
    picks = sel.select_detailed(
        "dataframe join merge partitions", [SQL, SYMPUT], top_k=2
    )
    tiers = {p.chunk.chunk_id: p.tier for p in picks}
    assert tiers == {
        "user::c0000": SelectionTier.USER_ALWAYS,
        "user::c0001": SelectionTier.USER_WHEN,
        "c5": SelectionTier.PINNED,
        "c2": SelectionTier.HAZARD,
        "c1": SelectionTier.CONSTRUCT,
        "user::c0002": SelectionTier.USER_TOPIC,
        "c4": SelectionTier.TOPICAL,
    }


def test_select_detailed_carries_matched_construct():
    sel = InstructionSelector(_corpus())
    picks = sel.select_detailed("zzz", [INTNX, SYMPUT])
    by_id = {p.chunk.chunk_id: p for p in picks}
    assert by_id["c0"].construct_key == INTNX
    assert by_id["c2"].construct_key == SYMPUT


def test_select_detailed_topical_has_no_construct():
    sel = InstructionSelector(_corpus())
    picks = sel.select_detailed("dataframe join merge partitions", [])
    topical = [p for p in picks if p.chunk.chunk_id == "c4"]
    assert topical and topical[0].tier is SelectionTier.TOPICAL
    assert topical[0].construct_key is None


def test_select_matches_select_detailed_chunks():
    sel = _user_selector(pinned_sections=["Output Format"])
    args = ("dataframe join merge partitions", [SQL, SYMPUT])
    assert [c.chunk_id for c in sel.select(*args)] == [
        p.chunk.chunk_id for p in sel.select_detailed(*args)
    ]


# ---------------------------------------------------------------------------
# Dense embedding disk cache
# ---------------------------------------------------------------------------


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.doc_calls = 0

    def _vec(self, text: str) -> list[float]:
        return [float(len(text) % 7) + 1.0, float(text.count("a")) + 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.doc_calls += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def test_disk_cached_embeddings_round_trip(tmp_path):
    cache = str(tmp_path / "emb.npz")
    fake1 = _FakeEmbeddings()
    wrapped1 = DiskCachedEmbeddings(fake1, cache)
    texts = ["alpha text", "beta text", "gamma text"]
    v1 = wrapped1.embed_documents(texts)
    assert fake1.doc_calls == 3
    assert (tmp_path / "emb.npz").exists()

    # A fresh wrapper over the same file serves every vector from disk.
    fake2 = _FakeEmbeddings()
    wrapped2 = DiskCachedEmbeddings(fake2, cache)
    v2 = wrapped2.embed_documents(texts)
    assert fake2.doc_calls == 0
    assert [list(map(float, v)) for v in v1] == [list(map(float, v)) for v in v2]


def test_selector_dense_uses_disk_cache_across_instances(tmp_path):
    cache = str(tmp_path / "emb.npz")
    corpus = _corpus()

    fake1 = _FakeEmbeddings()
    InstructionSelector(corpus, embeddings=fake1, embedding_cache_path=cache)
    assert fake1.doc_calls == len(corpus)  # cold: embedded every chunk

    fake2 = _FakeEmbeddings()
    sel2 = InstructionSelector(corpus, embeddings=fake2, embedding_cache_path=cache)
    assert fake2.doc_calls == 0  # warm: all chunk vectors from disk
    # dense path still functions
    out = sel2.select("dataframe join merge", [])
    assert out
