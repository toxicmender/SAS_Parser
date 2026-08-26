"""Versioned contracts for instruction ingestion and retrieval."""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from sas_migrate.core.models import ContractModel, VersionedContract
from sas_migrate.core.targets import TargetId
from sas_migrate.core.tokens import MessageRole, PromptComponentDraft, TokenCategory


class KnowledgeRole(StrEnum):
    SAS_REFERENCE = "sas_reference"
    TARGET_GUIDE = "target_guide"
    CHEAT_SHEET = "cheat_sheet"
    USER_INSTRUCTION = "user_instruction"


class RetrievalTier(StrEnum):
    USER_ALWAYS = "user_always"
    USER_WHEN = "user_when"
    PINNED = "pinned"
    HAZARD = "hazard"
    CONSTRUCT = "construct"
    USER_TOPIC = "user_topic"
    TOPICAL = "topical"


class RetrievalSignal(StrEnum):
    LEXICAL = "lexical"
    DENSE = "dense"
    RERANKER = "reranker"


class RuleScope(StrEnum):
    ALWAYS = "always"
    CONDITIONAL = "conditional"
    TOPIC = "topic"


class ConstructKey(ContractModel):
    kind: str = Field(min_length=1)
    name: str = Field(min_length=1)

    @field_validator("kind", "name", mode="before")
    @classmethod
    def normalize(cls, value: object) -> str:
        return str(value).strip().casefold().replace("-", "_").replace(" ", "_")

    def __str__(self) -> str:
        return f"{self.kind}:{self.name}"


class DocumentSection(ContractModel):
    source_id: str = Field(min_length=1)
    section_path: str = Field(min_length=1)
    text: str
    page_start: int = Field(default=1, ge=1)
    page_end: int = Field(default=1, ge=1)
    construct_keys: tuple[ConstructKey, ...] = Field(default_factory=tuple)
    tags: frozenset[str] = Field(default_factory=frozenset)

    @model_validator(mode="after")
    def validate_pages(self) -> DocumentSection:
        if self.page_end < self.page_start:
            raise ValueError("section page_end cannot precede page_start")
        return self


class ExtractionDiagnostic(ContractModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)


class DocumentExtraction(VersionedContract):
    source_id: str = Field(min_length=1)
    sections: tuple[DocumentSection, ...]
    strategy: str = Field(min_length=1)
    page_count: int = Field(ge=0)
    diagnostics: tuple[ExtractionDiagnostic, ...] = Field(default_factory=tuple)


class KnowledgeSource(VersionedContract):
    source_id: str = Field(min_length=1)
    role: KnowledgeRole
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    extraction_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    target: TargetId | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class KnowledgeChunk(VersionedContract):
    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    role: KnowledgeRole
    section_path: str = Field(min_length=1)
    text: str = Field(min_length=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    ordinal: int = Field(ge=0)
    token_count: int = Field(ge=0)
    construct_keys: tuple[ConstructKey, ...] = Field(default_factory=tuple)
    tags: frozenset[str] = Field(default_factory=frozenset)
    target: TargetId | None = None


class UserRule(VersionedContract):
    rule_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    scope: RuleScope = RuleScope.ALWAYS
    constructs: frozenset[str] = Field(default_factory=frozenset)
    chunk_kinds: frozenset[str] = Field(default_factory=frozenset)
    metadata_flags: frozenset[str] = Field(default_factory=frozenset)
    topics: frozenset[str] = Field(default_factory=frozenset)
    target: TargetId | None = None
    priority: int = 0
    enabled: bool = True

    @model_validator(mode="after")
    def validate_scope(self) -> UserRule:
        if self.scope is RuleScope.CONDITIONAL and not (
            self.constructs or self.chunk_kinds or self.metadata_flags
        ):
            raise ValueError("conditional rule requires a condition")
        if self.scope is RuleScope.TOPIC and not self.topics:
            raise ValueError("topic rule requires at least one topic")
        return self


class RetrievalQuery(VersionedContract):
    text: str
    target: TargetId
    constructs: frozenset[str] = Field(default_factory=frozenset)
    hazards: frozenset[str] = Field(default_factory=frozenset)
    chunk_kinds: frozenset[str] = Field(default_factory=frozenset)
    metadata_flags: frozenset[str] = Field(default_factory=frozenset)
    topics: frozenset[str] = Field(default_factory=frozenset)
    pinned_sections: tuple[str, ...] = Field(default_factory=tuple)
    max_results: int = Field(default=8, ge=0)
    max_tokens: int = Field(default=4_000, ge=0)


class RetrievedKnowledge(VersionedContract):
    chunk: KnowledgeChunk
    tier: RetrievalTier
    score: float = Field(ge=0)
    matched_construct: str | None = None
    reasons: tuple[str, ...] = Field(default_factory=tuple)

    def to_prompt_component(self) -> PromptComponentDraft:
        category = (
            TokenCategory.PROJECT_INSTRUCTIONS
            if self.chunk.role is KnowledgeRole.USER_INSTRUCTION
            else TokenCategory.REFERENCE_GUIDANCE
        )
        pages = (
            f"p{self.chunk.page_start}"
            if self.chunk.page_start == self.chunk.page_end
            else f"pp{self.chunk.page_start}-{self.chunk.page_end}"
        )
        return PromptComponentDraft(
            category=category,
            text=(
                f"[{self.chunk.source_id} · {self.chunk.section_path} · {pages}]\n"
                f"{self.chunk.text}"
            ),
            message_role=MessageRole.SYSTEM,
            source_id=self.chunk.chunk_id,
            cacheable=True,
        )


class KnowledgeRanking(VersionedContract):
    chunk_id: str = Field(min_length=1)
    score: float = Field(ge=0)
    reciprocal_rank_score: float = Field(ge=0)
    lexical_rank: int | None = Field(default=None, ge=1)
    dense_rank: int | None = Field(default=None, ge=1)
    reranker_score: float | None = None
    signals: tuple[RetrievalSignal, ...] = Field(min_length=1)

    @field_validator("score", "reciprocal_rank_score", "reranker_score")
    @classmethod
    def finite_scores(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("knowledge ranking scores must be finite")
        return value


class KnowledgeSelection(VersionedContract):
    query: RetrievalQuery
    results: tuple[RetrievedKnowledge, ...]
    components: tuple[PromptComponentDraft, ...]
    selected_tokens: int = Field(ge=0)


__all__ = [
    "ConstructKey",
    "DocumentExtraction",
    "DocumentSection",
    "ExtractionDiagnostic",
    "KnowledgeChunk",
    "KnowledgeRanking",
    "KnowledgeRole",
    "KnowledgeSelection",
    "KnowledgeSource",
    "RetrievalQuery",
    "RetrievalSignal",
    "RetrievalTier",
    "RetrievedKnowledge",
    "RuleScope",
    "UserRule",
]
