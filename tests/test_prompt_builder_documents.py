"""
Tests for the LangChain Document interop — InstructionChunk.to_document /
from_document round-trip and CorpusLoader.load_documents.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document

from prompt_builder.models import ConstructKey, DocRole, InstructionChunk


def _chunk() -> InstructionChunk:
    return InstructionChunk(
        chunk_id="functions::c0042",
        doc_id="functions",
        section_path="Dictionary > INTNX Function",
        text="Dictionary > INTNX Function\n\nAdvances a SAS date.",
        page_start=1109,
        page_end=1118,
        role=DocRole.SAS_REFERENCE,
        construct_keys=[ConstructKey(kind="function", name="intnx")],
        tags=["date"],
    )


def test_to_document_maps_text_and_metadata():
    doc = _chunk().to_document()
    assert isinstance(doc, Document)
    assert doc.id == "functions::c0042"
    assert doc.page_content.startswith("Dictionary > INTNX Function")
    assert doc.metadata["doc_id"] == "functions"
    assert doc.metadata["page_start"] == 1109
    assert doc.metadata["role"] == "sas_reference"
    # Construct keys flatten to vector-store-safe strings.
    assert doc.metadata["construct_keys"] == ["function:intnx"]


def test_document_round_trip_is_lossless():
    original = _chunk()
    rebuilt = InstructionChunk.from_document(original.to_document())
    assert rebuilt.model_dump() == original.model_dump()


def test_from_document_defaults_missing_optionals():
    doc = Document(
        id="x::c0",
        page_content="body",
        metadata={
            "doc_id": "x",
            "section_path": "S",
            "page_start": 1,
            "page_end": 2,
        },
    )
    chunk = InstructionChunk.from_document(doc)
    assert chunk.chunk_id == "x::c0"  # falls back to Document.id
    assert chunk.role is DocRole.SAS_REFERENCE
    assert chunk.construct_keys == []
    assert chunk.tags == []


def test_load_documents_returns_documents(tmp_path):
    import pymupdf

    from prompt_builder.catalog import CorpusLoader, DocumentSpec

    pdf = tmp_path / "funcs.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "INTNX Function", fontsize=16)
    page.insert_text((72, 110), "Advances a SAS date by intervals. " * 6, fontsize=11)
    doc.set_toc([[1, "INTNX Function", 1]])
    doc.save(str(pdf))
    doc.close()

    loader = CorpusLoader(cache_dir=str(tmp_path / "cache"))
    docs = loader.load_documents(
        [DocumentSpec(path=str(pdf), doc_id="funcs", section_level=1)]
    )
    assert docs
    assert all(isinstance(d, Document) for d in docs)
    assert all(d.metadata["doc_id"] == "funcs" for d in docs)
