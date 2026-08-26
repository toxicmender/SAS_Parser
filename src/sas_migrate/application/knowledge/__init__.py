"""Instruction ingestion, scoped rules, and attributed retrieval."""

from .ingestion import InstructionChunker, KnowledgeIngestionService
from .models import (
    ConstructKey,
    DocumentExtraction,
    DocumentSection,
    ExtractionDiagnostic,
    KnowledgeChunk,
    KnowledgeRanking,
    KnowledgeRole,
    KnowledgeSelection,
    KnowledgeSource,
    RetrievalQuery,
    RetrievalSignal,
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
    "KnowledgeRanking",
    "KnowledgeRetriever",
    "KnowledgeRole",
    "KnowledgeSelection",
    "KnowledgeSource",
    "RetrievalQuery",
    "RetrievalSignal",
    "RetrievalTier",
    "RetrievedKnowledge",
    "RuleScope",
    "UserRule",
    "UserRuleSet",
]
