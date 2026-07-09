"""
End-to-end integration test over the real reference_docs/ corpus.

Marked `slow` and skipped when the PDFs are absent (CI, fresh clones), because
the manuals are user-provided and gitignored. It catches TOC/structure-drift
assumptions no synthetic fixture can — e.g. that INTNX still lands in an "INTNX
Function" leaf. Uses the repo's on-disk cache, so the first run is slow and
subsequent runs are fast.

Run explicitly with:  pytest -m slow
Skip elsewhere with:  pytest -m "not slow"
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_REF_DIR = _ROOT / "reference_docs"
_CACHE_DIR = _ROOT / ".prompt_builder_cache"


def _has_reference_pdfs() -> bool:
    return _REF_DIR.is_dir() and any(_REF_DIR.glob("*.pdf"))


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _has_reference_pdfs(),
        reason="reference_docs/ PDFs not present (user-provided, gitignored)",
    ),
]


def test_real_catalog_loads_and_finds_intnx():
    from prompt_builder import ConstructKey, PromptBuilder
    from prompt_builder.catalog import CorpusLoader, default_catalog

    specs = default_catalog(str(_REF_DIR))
    assert specs, "expected at least one bundled reference PDF present"

    chunks = CorpusLoader(cache_dir=str(_CACHE_DIR)).load(specs)
    assert len(chunks) > 500  # the real corpus is thousands of chunks
    # Every chunk is attributed and page-located.
    assert all(c.doc_id and c.page_start >= 1 for c in chunks)

    builder = PromptBuilder(chunks)

    # Deterministic construct lookup still lands on the INTNX leaf section.
    block = builder.build(
        "advance a sas date to the next month interval",
        [ConstructKey(kind="function", name="intnx")],
    )
    assert block is not None
    assert "INTNX Function" in block

    # Pure topical retrieval surfaces target-platform guidance when present.
    if any(s.doc_id == "spark_guide" for s in specs):
        spark_block = builder.build(
            "convert a dataframe join and aggregation to spark sql", []
        )
        assert spark_block is not None
