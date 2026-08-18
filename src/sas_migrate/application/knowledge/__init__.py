"""Instruction ingestion, scoped rules, and attributed retrieval."""

from .ingestion import InstructionChunker, KnowledgeIngestionService
from .models import (
    ConstructKey,
    DocumentExtraction,
    DocumentSection,
    ExtractionDiagnostic,
    KnowledgeChunk,
    KnowledgeRole,
    KnowledgeSelection,
    KnowledgeSource,
    RetrievalQuery,
    RetrievalTier,
    RetrievedKnowledge,
    RuleScope,
    UserRule,
)
from .retrieval import KnowledgeRetriever
from .rules import UserRuleSet

__all__ = [
    "ConstructKey",
    "DocumentExtraction",
    "DocumentSection",
    "ExtractionDiagnostic",
    "InstructionChunker",
    "KnowledgeChunk",
    "KnowledgeIngestionService",
    "KnowledgeRetriever",
    "KnowledgeRole",
    "KnowledgeSelection",
    "KnowledgeSource",
    "RetrievalQuery",
    "RetrievalTier",
    "RetrievedKnowledge",
    "RuleScope",
    "UserRule",
    "UserRuleSet",
]
