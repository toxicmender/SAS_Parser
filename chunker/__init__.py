"""SAS semantic chunking and dependency batching. See ``chunker/README.md``."""

from .batcher import (
    MultiFileBatcher,
    SasChunkBatcher,
    parse_databricks_mapping_csv,
    replace_dataset_names,
)
from .chunker import SasSemanticChunker
from .metadata import resolve_corpus_references
from .models import (
    DatasetRole,
    DbTableAccess,
    DbTableVia,
    PathLocation,
    SasBatch,
    SasBatchResult,
    SasChunk,
    SasChunkKind,
    SasChunkMetadata,
    SasChunkResult,
    SasCorpus,
    SasDatasetRef,
    SasDbTableRef,
    SasDiagnostic,
    SasDiagnosticSeverity,
    SasEngineRef,
    SasIncludeFile,
    SasPathRef,
)
from .paths import (
    ENGINE_LIBNAMES,
    PATH_STATEMENTS,
    classify_location,
    extract_engine_refs,
    extract_paths,
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
    "SasDatasetRef",
    "DatasetRole",
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
    "SasIncludeFile",
    "PathLocation",
    "PATH_STATEMENTS",
    "classify_location",
    "extract_paths",
    # database-engine LIBNAME recognition — what data_hydration connects with
    "SasEngineRef",
    "ENGINE_LIBNAMES",
    "extract_engine_refs",
    # database tables (SQL pass-through and engine LIBNAMEs) — what it reads
    "SasDbTableRef",
    "DbTableAccess",
    "DbTableVia",
    # cross-file name resolution for callers that do not batch
    "resolve_corpus_references",
]
