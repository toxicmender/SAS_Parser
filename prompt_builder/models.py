"""Pydantic models for the reference-PDF instruction layer. See prompt_builder/README.md."""

from __future__ import annotations

import logging
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class DocRole(StrEnum):
    """What a reference document is *for*, which drives how it is retrieved."""

    SAS_REFERENCE = "sas_reference"  # SAS language manuals (source-side guidance)
    TARGET_GUIDE = "target_guide"  # target-platform guide, e.g. Spark (target-side)
    CHEAT_SHEET = "cheat_sheet"  # short quick-reference, optionally pinned


class ExtractionStrategy(StrEnum):
    """How :class:`~prompt_builder.pdf_reader.PdfReader` segmented a document."""

    TOC = "toc"  # segmented on the PDF's own table of contents
    FONT = "font"  # segmented on font-size heading heuristics
    PAGE = "page"  # fallback: one section per page (no TOC, no headings)


class InstructionDiagnostic(BaseModel):
    """A recoverable extraction issue — emitted, never raised."""

    code: str
    message: str
    doc_id: str | None = None
    page: int | None = None  # 1-based page number, when page-specific

    def __str__(self) -> str:
        where = f" [{self.doc_id}]" if self.doc_id else ""
        page = f" p{self.page}" if self.page is not None else ""
        return f"[{self.code}]{where}{page}: {self.message}"


class ConstructKey(BaseModel):
    """
    A normalised SAS construct a reference section documents, e.g.
    ``ConstructKey(kind="function", name="intnx")`` for "INTNX Function".

    Frozen so it is hashable and can be matched, in a set, against the
    constructs a pipeline item's metadata reports. ``name`` is always
    lowercased at construction by the parser.
    """

    model_config = ConfigDict(frozen=True)

    kind: str  # function | call_routine | macro_function | macro_statement |
    #            global_statement | proc | format | informat | option | system_option
    name: str  # lowercased identifier, e.g. "intnx", "symput", "let", "sql"

    def __str__(self) -> str:
        return f"{self.kind}:{self.name}"


class DocSection(BaseModel):
    """
    One section extracted from a reference PDF: the raw retrieval-preserving
    unit the chunker (Phase 3) turns into :class:`InstructionChunk`s.
    """

    doc_id: str
    section_path: str  # breadcrumb, e.g. "Dictionary of SAS Functions > INTNX Function"
    title: str  # leaf heading only, e.g. "INTNX Function"
    text: str
    page_start: int  # 1-based, inclusive
    page_end: int  # 1-based, inclusive
    level: int = 0  # TOC/heading depth (1 = top); 0 when not meaningful
    construct_key: ConstructKey | None = None

    def __str__(self) -> str:
        span = (
            f"p{self.page_start}"
            if self.page_start == self.page_end
            else f"pp{self.page_start}-{self.page_end}"
        )
        return f"{self.section_path} ({span}, {len(self.text)} chars)"


class InstructionDoc(BaseModel):
    """Per-document extraction summary returned alongside its sections."""

    doc_id: str
    path: str
    role: DocRole = DocRole.SAS_REFERENCE
    page_count: int = 0
    strategy: ExtractionStrategy = ExtractionStrategy.TOC
    section_count: int = 0
    diagnostics: list[InstructionDiagnostic] = Field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"InstructionDoc({self.doc_id}: {self.section_count} section(s) via "
            f"{self.strategy}, {len(self.diagnostics)} diagnostic(s))"
        )


class InstructionChunk(BaseModel):
    """
    A word-budgeted, retrieval-ready instruction unit (produced in Phase 3
    from one or more :class:`DocSection`s). Defined here so the whole data
    model lives in one place; the reader itself emits :class:`DocSection`s.
    """

    chunk_id: str
    doc_id: str
    section_path: str
    text: str
    page_start: int
    page_end: int
    role: DocRole = DocRole.SAS_REFERENCE
    construct_keys: list[ConstructKey] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    def __str__(self) -> str:
        return f"InstructionChunk({self.chunk_id}: {self.section_path})"
