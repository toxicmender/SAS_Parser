"""SAS semantic chunking and dependency batching. See ``chunker/README.md``."""

from .batcher import (
    MultiFileBatcher,
    SasChunkBatcher,
    parse_databricks_mapping_csv,
    replace_dataset_names,
)
from .chunker import SasSemanticChunker
from .models import (
    PathLocation,
    SasBatch,
    SasBatchResult,
    SasChunk,
    SasChunkKind,
    SasChunkMetadata,
    SasChunkResult,
    SasCorpus,
    SasDiagnostic,
    SasDiagnosticSeverity,
    SasPathRef,
)
from .paths import PATH_STATEMENTS, classify_location, extract_paths

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
    # physical/remote path recognition — the grammar xref.pre also reads
    "SasPathRef",
    "PathLocation",
    "PATH_STATEMENTS",
    "classify_location",
    "extract_paths",
]
