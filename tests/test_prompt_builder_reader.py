"""
Tests for prompt_builder.pdf_reader — TOC and font extraction strategies,
text cleanup, construct-key parsing, and graceful diagnostics.

Fully offline: fixture PDFs are generated in-process with pymupdf (a project
dependency), so no binary fixtures live in the repo and no test is skipped.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pymupdf
import pytest

from prompt_builder.models import (
    ConstructKey,
    DocRole,
    ExtractionStrategy,
)
from prompt_builder.pdf_reader import (
    PdfReader,
    _normalize_text,
    parse_construct_key,
)


# ---------------------------------------------------------------------------
# Fixture-PDF helper
# ---------------------------------------------------------------------------


def _write_pdf(path: pathlib.Path, pages_lines, toc=None) -> str:
    """
    Build a PDF at *path*. ``pages_lines`` is a list of pages, each a list of
    ``(text, fontsize)`` lines laid out top-to-bottom. An empty page list makes
    a blank page. ``toc`` (if given) is a pymupdf ``[level, title, page]`` list.
    """
    doc = pymupdf.open()
    for lines in pages_lines:
        page = doc.new_page(width=612, height=792)
        y = 72.0
        for text, size in lines:
            page.insert_text((72, y), text, fontsize=size, fontname="helv")
            y += size * 1.6 + 6
    if toc:
        doc.set_toc(toc)
    doc.save(str(path))
    doc.close()
    return str(path)


# ---------------------------------------------------------------------------
# Construct-key parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "title,expected",
    [
        ("INTNX Function", ("function", "intnx")),
        ("SUBSTR Function (Left of =)", ("function", "substr")),
        ("%SYSFUNC Function", ("macro_function", "sysfunc")),
        ("CALL SYMPUT Routine", ("call_routine", "symput")),
        ("SYMPUTX Routine", ("call_routine", "symputx")),
        ("%LET Statement", ("macro_statement", "let")),
        ("LIBNAME Statement", ("global_statement", "libname")),
        ("The SQL Procedure", ("proc", "sql")),
        ("MEANS Procedure", ("proc", "means")),
        ("Hash Object", ("component_object", "hash")),
        ("HITER Component Object", ("component_object", "hiter")),
        ("Introduction", None),
        ("Dictionary of SAS Functions", None),
    ],
)
def test_parse_construct_key(title, expected):
    key = parse_construct_key(title)
    if expected is None:
        assert key is None
    else:
        assert key is not None
        assert (key.kind, key.name) == expected


# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------


def test_normalize_text_dehyphenates_and_straightens():
    out = _normalize_text("inter-\nval\nSpark’s “tool”")
    assert "interval" in out
    assert "Spark's" in out
    assert '"tool"' in out
    assert "’" not in out and "“" not in out


# ---------------------------------------------------------------------------
# TOC strategy
# ---------------------------------------------------------------------------


def _funcs_pdf(tmp_path: pathlib.Path) -> str:
    pages = [
        [("Contents", 14), ("Dictionary of SAS Functions .... 3", 11)],
        [("About This Book", 14), ("This book covers the SAS language.", 11)],
        [
            ("Dictionary of SAS Functions", 16),
            ("This section documents SAS functions.", 11),
            ("INTCK Function", 16),
            ("Counts interval boundaries between two SAS dates.", 11),
        ],
        [
            ("INTNX Function", 16),
            ("Advances a SAS date by a number of intervals.", 11),
            ("Syntax INTNX interval start increment", 11),
        ],
        [
            ("SUBSTR Function", 16),
            ("Extracts a substring from a character value.", 11),
        ],
        [
            ("The SQL Procedure", 16),
            ("PROC SQL runs ANSI SQL queries against SAS tables.", 11),
        ],
    ]
    toc = [
        [1, "Contents", 1],
        [1, "About This Book", 2],
        [1, "Dictionary of SAS Functions", 3],
        [2, "INTCK Function", 3],
        [2, "INTNX Function", 4],
        [2, "SUBSTR Function", 5],
        [1, "The SQL Procedure", 6],
    ]
    return _write_pdf(tmp_path / "funcs.pdf", pages, toc)


def test_toc_skips_front_matter_and_keeps_content(tmp_path):
    reader = PdfReader()
    summary, sections = reader.read(
        _funcs_pdf(tmp_path), doc_id="funcs", section_level=2
    )
    assert summary.strategy is ExtractionStrategy.TOC
    titles = [s.title for s in sections]
    assert "Contents" not in titles
    assert "About This Book" not in titles
    assert "INTNX Function" in titles
    assert "The SQL Procedure" in titles


def test_toc_shared_page_split_and_breadcrumb(tmp_path):
    reader = PdfReader()
    _, sections = reader.read(_funcs_pdf(tmp_path), doc_id="funcs", section_level=2)
    intnx = next(s for s in sections if s.title == "INTNX Function")
    # Sliced at the heading, so it holds INTNX body and not the preceding INTCK.
    assert "Advances a SAS date" in intnx.text
    assert "interval boundaries" not in intnx.text
    assert intnx.section_path == "Dictionary of SAS Functions > INTNX Function"
    assert intnx.page_start == 4


def test_toc_attaches_construct_keys(tmp_path):
    reader = PdfReader()
    _, sections = reader.read(_funcs_pdf(tmp_path), doc_id="funcs", section_level=2)
    by_title = {s.title: s for s in sections}
    assert by_title["INTNX Function"].construct_key == ConstructKey(
        kind="function", name="intnx"
    )
    assert by_title["The SQL Procedure"].construct_key == ConstructKey(
        kind="proc", name="sql"
    )
    # A chapter heading names no single construct.
    assert by_title["Dictionary of SAS Functions"].construct_key is None


# ---------------------------------------------------------------------------
# Font strategy
# ---------------------------------------------------------------------------


def test_font_segments_by_size_tiers(tmp_path):
    pages = [
        [
            ("A Tour of Spark", 26),
            ("Spark is a unified engine for large-scale data processing.", 11),
            ("Datasets", 20),
            ("Datasets are a type-safe structured API over the JVM.", 11),
        ],
        [
            ("Structured Streaming", 20),
            ("Structured Streaming is a stream processing engine on Spark.", 11),
        ],
    ]
    path = _write_pdf(tmp_path / "spark.pdf", pages)  # no TOC
    reader = PdfReader()
    summary, sections = reader.read(
        path, doc_id="spark", role=DocRole.TARGET_GUIDE, strategy="auto"
    )
    assert summary.strategy is ExtractionStrategy.FONT
    assert [s.title for s in sections] == [
        "A Tour of Spark",
        "Datasets",
        "Structured Streaming",
    ]
    tour = next(s for s in sections if s.title == "A Tour of Spark")
    datasets = next(s for s in sections if s.title == "Datasets")
    assert tour.level == 1 and datasets.level == 2
    assert datasets.section_path == "A Tour of Spark > Datasets"
    assert "type-safe" in datasets.text


def test_requested_toc_without_toc_falls_back_to_font(tmp_path):
    pages = [[("Overview", 22), ("Some body text about the platform.", 11)]]
    path = _write_pdf(tmp_path / "notoc.pdf", pages)
    reader = PdfReader()
    summary, sections = reader.read(path, doc_id="notoc", strategy="toc")
    assert summary.strategy is ExtractionStrategy.FONT
    assert any(d.code == "NO_TOC" for d in summary.diagnostics)
    assert [s.title for s in sections] == ["Overview"]


# ---------------------------------------------------------------------------
# Graceful degradation — running headers, empty pages, no headings
# ---------------------------------------------------------------------------


def test_running_footer_stripped_and_empty_page_skipped(tmp_path):
    footer = "SAS Institute Inc. Confidential Draft"
    pages = [
        [("Getting started with migration.", 11), (footer, 11)],
        [("Chapter two body content here.", 11), (footer, 11)],
        [],  # empty page
        [("Final page content about datasets.", 11), (footer, 11)],
    ]
    path = _write_pdf(tmp_path / "plain.pdf", pages)  # no TOC, no headings
    reader = PdfReader()
    summary, sections = reader.read(path, doc_id="plain", strategy="auto")

    assert summary.strategy is ExtractionStrategy.PAGE
    assert any(d.code == "NO_HEADINGS_DETECTED" for d in summary.diagnostics)
    full_text = "\n".join(s.text for s in sections)
    assert "SAS Institute Inc." not in full_text  # running footer removed
    # Blank page 3 yields no section; pages 1, 2, 4 do.
    assert [s.page_start for s in sections] == [1, 2, 4]


def test_reader_never_raises_on_blank_document(tmp_path):
    path = _write_pdf(tmp_path / "blank.pdf", [[], []])  # two empty pages
    reader = PdfReader()
    summary, sections = reader.read(path, doc_id="blank", strategy="auto")
    assert sections == []
    assert any(d.code == "NO_TEXT_LAYER" for d in summary.diagnostics)
