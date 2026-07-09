"""
Tests for prompt_builder.doc_chunker — word-budget chunking of reader sections:
same-parent merge of undersized sections, oversized paragraph-window splitting
with overlap, breadcrumb prefixing, and construct-key aggregation.

Fully offline: DocSections are built directly, no PDF needed.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from prompt_builder.doc_chunker import InstructionChunker, _split_overlapping
from prompt_builder.models import ConstructKey, DocRole, DocSection


def _section(
    path: str,
    text: str,
    *,
    doc_id: str = "doc",
    page_start: int = 1,
    page_end: int = 1,
    construct_key: ConstructKey | None = None,
) -> DocSection:
    return DocSection(
        doc_id=doc_id,
        section_path=path,
        title=path.split(" > ")[-1],
        text=text,
        page_start=page_start,
        page_end=page_end,
        construct_key=construct_key,
    )


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


# ---------------------------------------------------------------------------
# Single section
# ---------------------------------------------------------------------------


def test_single_section_becomes_one_chunk_with_breadcrumb():
    key = ConstructKey(kind="function", name="intnx")
    sections = [
        _section("Functions > INTNX Function", _words(200), construct_key=key)
    ]
    chunks = InstructionChunker(min_words=120, max_words=900).chunk(sections)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.section_path == "Functions > INTNX Function"
    assert chunk.text.startswith("Functions > INTNX Function\n\n")
    assert chunk.construct_keys == [key]
    assert chunk.chunk_id == "doc::c0000"


def test_role_is_propagated():
    sections = [_section("Guide > Datasets", _words(50))]
    chunks = InstructionChunker(min_words=10).chunk(
        sections, role=DocRole.TARGET_GUIDE
    )
    assert chunks[0].role is DocRole.TARGET_GUIDE


# ---------------------------------------------------------------------------
# Merge of undersized siblings
# ---------------------------------------------------------------------------


def test_small_siblings_under_same_parent_merge():
    sections = [
        _section(
            "Funcs > A", _words(40), construct_key=ConstructKey(kind="function", name="a")
        ),
        _section(
            "Funcs > B", _words(40), construct_key=ConstructKey(kind="function", name="b")
        ),
        _section(
            "Funcs > C", _words(40), construct_key=ConstructKey(kind="function", name="c")
        ),
    ]
    chunks = InstructionChunker(min_words=100, max_words=900).chunk(sections)
    assert len(chunks) == 1
    chunk = chunks[0]
    # Merged group collapses to the shared parent breadcrumb...
    assert chunk.section_path == "Funcs"
    # ...and aggregates every member's construct key, in order.
    assert [k.name for k in chunk.construct_keys] == ["a", "b", "c"]


def test_different_parent_breaks_merge():
    sections = [
        _section("Funcs > A", _words(40)),
        _section("Procs > SORT", _words(40)),
    ]
    chunks = InstructionChunker(min_words=100).chunk(sections)
    assert len(chunks) == 2
    assert chunks[0].section_path == "Funcs > A"
    assert chunks[1].section_path == "Procs > SORT"


def test_large_section_stands_alone_not_merged_with_neighbor():
    sections = [
        _section("Funcs > BIG", _words(300)),
        _section("Funcs > SMALL", _words(20)),
    ]
    chunks = InstructionChunker(min_words=120, max_words=900).chunk(sections)
    assert len(chunks) == 2
    assert chunks[0].section_path == "Funcs > BIG"
    assert chunks[1].section_path == "Funcs > SMALL"


def test_page_span_covers_all_merged_members():
    sections = [
        _section("Funcs > A", _words(30), page_start=10, page_end=10),
        _section("Funcs > B", _words(30), page_start=11, page_end=12),
    ]
    chunks = InstructionChunker(min_words=100).chunk(sections)
    assert len(chunks) == 1
    assert chunks[0].page_start == 10
    assert chunks[0].page_end == 12


# ---------------------------------------------------------------------------
# Oversized split
# ---------------------------------------------------------------------------


def test_oversized_section_splits_into_windows():
    big = "\n\n".join(_words(200) for _ in range(5))  # 1000 words, 5 paragraphs
    sections = [_section("Guide > Long", big)]
    chunks = InstructionChunker(min_words=120, max_words=300).chunk(sections)
    assert len(chunks) > 1
    # every window respects the budget (breadcrumb line excluded from the count)
    for chunk in chunks:
        body = chunk.text.split("\n\n", 1)[1]
        assert _wc_body(body) <= 300
        assert chunk.section_path == "Guide > Long"
        assert chunk.text.startswith("Guide > Long\n\n")


def test_split_windows_are_uniquely_ided():
    big = "\n\n".join(_words(100) for _ in range(6))
    sections = [_section("Guide > Long", big)]
    chunks = InstructionChunker(max_words=150).chunk(sections)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_split_overlapping_carries_overlap_between_windows():
    paras = [f"para{i} {_words(4)}" for i in range(4)]  # 5 words each
    text = "\n\n".join(paras)
    windows = _split_overlapping(text, max_words=12, overlap_words=6)
    assert len(windows) >= 2
    # Consecutive windows share a paragraph (the carried overlap).
    for earlier, later in zip(windows, windows[1:]):
        earlier_paras = set(earlier.split("\n\n"))
        later_paras = set(later.split("\n\n"))
        assert earlier_paras & later_paras


def test_giant_single_paragraph_is_hard_split():
    sections = [_section("Guide > Wall", _words(500))]  # one 500-word paragraph
    chunks = InstructionChunker(max_words=100).chunk(sections)
    assert len(chunks) >= 5
    for chunk in chunks:
        body = chunk.text.split("\n\n", 1)[1]
        assert _wc_body(body) <= 100


def _wc_body(text: str) -> int:
    return len(text.split())
