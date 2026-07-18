"""Pydantic models for the reference-PDF instruction layer. See prompt_builder/README.md."""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocRole(StrEnum):
    """What a reference document is *for*, which drives how it is retrieved."""

    SAS_REFERENCE = "sas_reference"  # SAS language manuals (source-side guidance)
    TARGET_GUIDE = "target_guide"  # target-platform guide, e.g. Spark (target-side)
    CHEAT_SHEET = "cheat_sheet"  # short quick-reference, optionally pinned
    USER_INSTRUCTION = "user_instruction"  # operator-supplied project rules


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


class ConstructKey(BaseModel, frozen=True):
    """
    A normalised SAS construct a reference section documents, e.g.
    ``ConstructKey(kind="function", name="intnx")`` for "INTNX Function".

    Frozen so it is hashable and can be matched, in a set, against the
    constructs a pipeline item's metadata reports. ``name`` is always
    lowercased at construction by the parser.

    Frozen via the class keyword rather than ``model_config`` so type
    checkers see the generated ``__hash__`` and accept it as a dict key.
    """

    kind: str  # function | call_routine | macro_function | macro_statement |
    #            global_statement | proc | format | informat | option |
    #            system_option | component_object
    name: str  # lowercased identifier, e.g. "intnx", "symput", "let", "sql"

    def __str__(self) -> str:
        return f"{self.kind}:{self.name}"


class SelectionTier(StrEnum):
    """
    Why the selector picked a chunk — its priority tier, in selection order.
    Carried on :class:`SelectedInstruction` so downstream formatting can treat
    picks by provenance (e.g. render hazard hits as explicit focus hints).
    """

    USER_ALWAYS = "user_always"  # operator rule, always-on
    USER_WHEN = "user_when"  # operator rule matched via [when: ...] constructs
    USER_EXAMPLE = "user_example"  # operator few-shot example ([example: ...])
    PINNED = "pinned"  # reference section pinned by section-path substring
    HAZARD = "hazard"  # construct lookup hit on a hazard construct
    CONSTRUCT = "construct"  # construct lookup hit (non-hazard)
    USER_TOPIC = "user_topic"  # operator [topic] chunk surfaced by ranking
    TOPICAL = "topical"  # reference chunk surfaced by hybrid ranking


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

    # ------------------------------------------------------------------
    # LangChain Document interop — so the corpus can feed any LangChain
    # indexing pipeline (vector stores, retrievers, LangChain's index API)
    # without a custom adapter. Metadata values are kept to str/int so
    # every vector-store backend accepts them; construct keys flatten to
    # their "kind:name" string form.
    # ------------------------------------------------------------------

    def to_document(self) -> "Document":
        """This chunk as a ``langchain_core.documents.Document``."""
        from langchain_core.documents import Document

        return Document(
            id=self.chunk_id,
            page_content=self.text,
            metadata={
                "chunk_id": self.chunk_id,
                "doc_id": self.doc_id,
                "section_path": self.section_path,
                "page_start": self.page_start,
                "page_end": self.page_end,
                "role": self.role.value,
                "construct_keys": [str(k) for k in self.construct_keys],
                "tags": list(self.tags),
            },
        )

    @classmethod
    def from_document(cls, document: "Document") -> "InstructionChunk":
        """Rebuild a chunk from a :meth:`to_document` round-trip."""
        meta = document.metadata
        keys = []
        for raw in meta.get("construct_keys", []):
            kind, _, name = raw.partition(":")
            keys.append(ConstructKey(kind=kind, name=name))
        return cls(
            chunk_id=meta.get("chunk_id") or document.id or "",
            doc_id=meta["doc_id"],
            section_path=meta["section_path"],
            text=document.page_content,
            page_start=meta["page_start"],
            page_end=meta["page_end"],
            role=DocRole(meta.get("role", DocRole.SAS_REFERENCE.value)),
            construct_keys=keys,
            tags=list(meta.get("tags", [])),
        )


class SelectedInstruction(BaseModel):
    """
    One selector pick with its provenance: the chunk, the tier that claimed
    it, and — for construct-lookup tiers — the construct key that matched.
    """

    chunk: InstructionChunk
    tier: SelectionTier
    construct_key: ConstructKey | None = None

    def __str__(self) -> str:
        via = f" via {self.construct_key}" if self.construct_key else ""
        return f"SelectedInstruction({self.chunk.chunk_id}: {self.tier}{via})"
