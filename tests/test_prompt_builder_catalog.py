"""
Tests for prompt_builder.catalog — DocumentSpec, default_catalog, and the
CorpusLoader on-disk extraction cache (hit, invalidation, round-trip).

Fully offline: fixture PDFs are generated in-process with pymupdf.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pymupdf

from prompt_builder.catalog import (
    CorpusLoader,
    DocumentSpec,
    default_catalog,
)
from prompt_builder.doc_chunker import InstructionChunker
from prompt_builder.models import ConstructKey, DocRole
from prompt_builder.pdf_reader import PdfReader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_funcs_pdf(path: pathlib.Path) -> str:
    pages = [
        [
            ("Dictionary of Functions", 16),
            ("This section documents SAS functions.", 11),
            ("INTNX Function", 16),
            ("Advances a SAS date by a number of intervals. " * 8, 11),
        ],
        [
            ("SUBSTR Function", 16),
            ("Extracts a substring from a character value. " * 8, 11),
        ],
    ]
    toc = [
        [1, "Dictionary of Functions", 1],
        [2, "INTNX Function", 1],
        [2, "SUBSTR Function", 2],
    ]
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page(width=612, height=792)
        y = 72.0
        for text, size in lines:
            page.insert_text((72, y), text, fontsize=size, fontname="helv")
            y += size * 1.6 + 6
    doc.set_toc(toc)
    doc.save(str(path))
    doc.close()
    return str(path)


class _CountingReader(PdfReader):
    """PdfReader that records how many times read() runs (to detect cache hits)."""

    def __init__(self) -> None:
        super().__init__()
        self.read_calls = 0

    def read(self, *args, **kwargs):
        self.read_calls += 1
        return super().read(*args, **kwargs)


def _spec(path: str) -> DocumentSpec:
    return DocumentSpec(
        path=path, doc_id="funcs", role=DocRole.SAS_REFERENCE, section_level=2
    )


# ---------------------------------------------------------------------------
# DocumentSpec / default_catalog
# ---------------------------------------------------------------------------


def test_document_spec_defaults():
    spec = DocumentSpec(path="x.pdf", doc_id="x")
    assert spec.role is DocRole.SAS_REFERENCE
    assert spec.strategy == "auto"
    assert spec.section_level is None
    assert spec.pinned_sections == []


def test_default_catalog_only_lists_present_files(tmp_path):
    # An empty reference dir yields no specs.
    assert default_catalog(str(tmp_path)) == []

    # Drop in one known filename; only that spec comes back, with its pinned level.
    _write_funcs_pdf(tmp_path / "SAS_Functions_and_Call_Routines.pdf")
    specs = default_catalog(str(tmp_path))
    assert [s.doc_id for s in specs] == ["functions"]
    assert specs[0].section_level == 4
    assert specs[0].role is DocRole.SAS_REFERENCE


def test_default_catalog_include_unknown_indexes_extra_pdfs(tmp_path):
    _write_funcs_pdf(tmp_path / "SAS_Functions_and_Call_Routines.pdf")
    _write_funcs_pdf(tmp_path / "My Custom Style-Guide (v2).pdf")

    # Off by default: the unknown PDF is ignored.
    assert [s.doc_id for s in default_catalog(str(tmp_path))] == ["functions"]

    specs = default_catalog(str(tmp_path), include_unknown=True)
    by_id = {s.doc_id: s for s in specs}
    assert set(by_id) == {"functions", "my_custom_style_guide_v2"}
    unknown = by_id["my_custom_style_guide_v2"]
    assert unknown.strategy == "auto"
    assert unknown.section_level is None
    assert unknown.role is DocRole.SAS_REFERENCE


# ---------------------------------------------------------------------------
# CorpusLoader extraction + cache
# ---------------------------------------------------------------------------


def test_load_one_extracts_and_writes_cache(tmp_path):
    path = _write_funcs_pdf(tmp_path / "funcs.pdf")
    cache_dir = tmp_path / "cache"
    loader = CorpusLoader(cache_dir=str(cache_dir))
    chunks = loader.load_one(_spec(path))

    assert chunks
    assert all(c.doc_id == "funcs" for c in chunks)
    assert (cache_dir / "funcs.json").exists()
    # INTNX construct key survived reader -> chunker.
    intnx = ConstructKey(kind="function", name="intnx")
    assert any(intnx in c.construct_keys for c in chunks)


def test_second_load_hits_cache_without_reading_pdf(tmp_path):
    path = _write_funcs_pdf(tmp_path / "funcs.pdf")
    reader = _CountingReader()
    loader = CorpusLoader(reader=reader, cache_dir=str(tmp_path / "cache"))

    first = loader.load_one(_spec(path))
    assert reader.read_calls == 1

    second = loader.load_one(_spec(path))
    assert reader.read_calls == 1  # served from cache, PDF not re-read
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]


def test_cache_round_trips_construct_keys_and_pages(tmp_path):
    path = _write_funcs_pdf(tmp_path / "funcs.pdf")
    loader = CorpusLoader(cache_dir=str(tmp_path / "cache"))
    fresh = loader.load_one(_spec(path))
    # Force a second, cache-served load and compare full fidelity.
    cached = CorpusLoader(cache_dir=str(tmp_path / "cache")).load_one(_spec(path))
    assert [c.model_dump() for c in fresh] == [c.model_dump() for c in cached]


def test_changed_chunker_params_invalidate_cache(tmp_path):
    path = _write_funcs_pdf(tmp_path / "funcs.pdf")
    cache_dir = str(tmp_path / "cache")

    reader_a = _CountingReader()
    CorpusLoader(reader=reader_a, cache_dir=cache_dir).load_one(_spec(path))
    assert reader_a.read_calls == 1

    # Different chunker budget -> different signature -> cache miss -> re-read.
    reader_b = _CountingReader()
    CorpusLoader(
        reader=reader_b,
        chunker=InstructionChunker(min_words=999, max_words=1000),
        cache_dir=cache_dir,
    ).load_one(_spec(path))
    assert reader_b.read_calls == 1


def test_changed_source_bytes_invalidate_cache(tmp_path):
    pdf = tmp_path / "funcs.pdf"
    _write_funcs_pdf(pdf)
    cache_dir = str(tmp_path / "cache")

    reader_a = _CountingReader()
    CorpusLoader(reader=reader_a, cache_dir=cache_dir).load_one(_spec(str(pdf)))
    assert reader_a.read_calls == 1

    # Rewrite the file with different content: SHA changes -> cache miss.
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "COMPLETELY DIFFERENT CONTENT", fontsize=14)
    doc.set_toc([[1, "COMPLETELY DIFFERENT CONTENT", 1]])
    doc.save(str(pdf))
    doc.close()

    reader_b = _CountingReader()
    CorpusLoader(reader=reader_b, cache_dir=cache_dir).load_one(_spec(str(pdf)))
    assert reader_b.read_calls == 1


def test_use_cache_false_never_writes(tmp_path):
    path = _write_funcs_pdf(tmp_path / "funcs.pdf")
    cache_dir = tmp_path / "cache"
    loader = CorpusLoader(cache_dir=str(cache_dir), use_cache=False)
    loader.load_one(_spec(path))
    assert not cache_dir.exists()


# ---------------------------------------------------------------------------
# Freshness API — check_freshness / freshness_report / prune_stale
# ---------------------------------------------------------------------------


def test_freshness_lifecycle(tmp_path):
    pdf = tmp_path / "funcs.pdf"
    _write_funcs_pdf(pdf)
    loader = CorpusLoader(cache_dir=str(tmp_path / "cache"))
    spec = _spec(str(pdf))

    assert loader.check_freshness(spec) == "uncached"
    loader.load_one(spec)
    assert loader.check_freshness(spec) == "fresh"

    # Rewriting the source with different bytes makes the entry stale.
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "changed content", fontsize=14)
    doc.set_toc([[1, "changed content", 1]])
    doc.save(str(pdf))
    doc.close()
    assert loader.check_freshness(spec) == "stale"

    pdf.unlink()
    assert loader.check_freshness(spec) == "missing"


def test_freshness_report_covers_all_specs(tmp_path):
    pdf = tmp_path / "funcs.pdf"
    _write_funcs_pdf(pdf)
    loader = CorpusLoader(cache_dir=str(tmp_path / "cache"))
    present = _spec(str(pdf))
    absent = DocumentSpec(path=str(tmp_path / "absent.pdf"), doc_id="absent")
    loader.load_one(present)

    report = loader.freshness_report([present, absent])
    assert report == {"funcs": "fresh", "absent": "missing"}


def test_stat_fast_path_skips_rehash(tmp_path, monkeypatch):
    import prompt_builder.catalog as catalog_mod

    pdf = tmp_path / "funcs.pdf"
    _write_funcs_pdf(pdf)
    loader = CorpusLoader(cache_dir=str(tmp_path / "cache"))
    spec = _spec(str(pdf))
    loader.load_one(spec)  # populates cache incl. size/mtime/sha

    calls = {"n": 0}
    real = catalog_mod._file_sha256

    def counting_sha(path):
        calls["n"] += 1
        return real(path)

    monkeypatch.setattr(catalog_mod, "_file_sha256", counting_sha)
    loader.load_one(spec)  # warm load: stat matches, no rehash
    assert calls["n"] == 0

    # Touch the mtime without changing bytes: rehash happens, entry stays fresh.
    stat = pdf.stat()
    import os

    os.utime(pdf, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert loader.check_freshness(spec) == "fresh"
    assert calls["n"] == 1


def test_extractor_code_change_invalidates_cache(tmp_path, monkeypatch):
    import prompt_builder.catalog as catalog_mod

    pdf = tmp_path / "funcs.pdf"
    _write_funcs_pdf(pdf)
    loader = CorpusLoader(cache_dir=str(tmp_path / "cache"))
    spec = _spec(str(pdf))
    loader.load_one(spec)
    assert loader.check_freshness(spec) == "fresh"

    # Simulate an edit to pdf_reader.py / doc_chunker.py: the fingerprint
    # changes, so the signature no longer matches.
    monkeypatch.setattr(catalog_mod, "_CODE_HASH", "deadbeefdead")
    assert loader.check_freshness(spec) == "stale"


def test_prune_stale_removes_orphans_and_stale_keeps_fresh(tmp_path):
    pdf = tmp_path / "funcs.pdf"
    _write_funcs_pdf(pdf)
    cache_dir = tmp_path / "cache"
    loader = CorpusLoader(cache_dir=str(cache_dir))
    spec = _spec(str(pdf))
    loader.load_one(spec)

    # An orphaned entry no spec refers to, and a stale one for a removed file.
    (cache_dir / "orphan.json").write_text('{"signature": "x", "chunks": []}')
    gone_pdf = tmp_path / "gone.pdf"
    _write_funcs_pdf(gone_pdf)
    gone_spec = DocumentSpec(path=str(gone_pdf), doc_id="gone", section_level=2)
    loader.load_one(gone_spec)
    gone_pdf.unlink()

    removed = loader.prune_stale([spec, gone_spec])
    assert sorted(removed) == ["gone.json", "orphan.json"]
    assert (cache_dir / "funcs.json").exists()  # fresh entry kept
    assert loader.check_freshness(spec) == "fresh"


def test_load_concatenates_specs_and_skips_missing(tmp_path):
    path = _write_funcs_pdf(tmp_path / "funcs.pdf")
    loader = CorpusLoader(cache_dir=str(tmp_path / "cache"))
    specs = [
        _spec(path),
        DocumentSpec(path=str(tmp_path / "absent.pdf"), doc_id="absent"),
    ]
    chunks = loader.load(specs)
    assert chunks  # the present spec's chunks
    assert all(c.doc_id == "funcs" for c in chunks)  # missing spec contributed none
