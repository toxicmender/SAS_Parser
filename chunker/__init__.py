"""SAS semantic chunker and dependency batcher. See chunker/README.md.

The LLM orchestration layer, structured response models, and notebook
renderer live in the top-level ``pipeline`` package. They were once
re-exported here; those bridges are gone, so import them from ``pipeline``.
"""

from .batcher import (
    MultiFileBatcher,
    SasChunkBatcher,
    parse_databricks_mapping_csv,
    replace_dataset_names,
)
from .chunker import SasSemanticChunker
from .models import (
    SasBatch,
    SasBatchResult,
    SasChunk,
    SasChunkKind,
    SasChunkMetadata,
    SasChunkResult,
    SasCorpus,
    SasDiagnostic,
    SasDiagnosticSeverity,
)

__all__ = [
    # chunker
    "SasSemanticChunker",
    # single-file batcher
    "SasChunkBatcher",
    # multi-file batcher
    "MultiFileBatcher",
    # Databricks dataset-name mapping post-pass
    "replace_dataset_names",
    "parse_databricks_mapping_csv",
    # models — single-file
    "SasChunk",
    "SasChunkKind",
    "SasChunkMetadata",
    "SasChunkResult",
    "SasDiagnostic",
    "SasDiagnosticSeverity",
    # models — batcher (single- and multi-file)
    "SasBatch",
    "SasBatchResult",
    # models — multi-file input
    "SasCorpus",
]
