"""SAS semantic chunker, dependency batcher, and LangChain pipeline. See chunker/README.md."""

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
from .notebook import (
    build_notebook,
    document_to_cells,
    markdown_to_cells,
    notebook_to_json,
    notebooks_from_outputs,
    write_notebooks,
)
from .response_models import (
    MappingEntry,
    RiskNote,
    TranslationCell,
    TranslationDocument,
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
    # models — structured LLM response
    "TranslationDocument",
    "TranslationCell",
    "MappingEntry",
    "RiskNote",
    # notebook rendering
    "write_notebooks",
    "notebooks_from_outputs",
    "build_notebook",
    "notebook_to_json",
    "document_to_cells",
    "markdown_to_cells",
]


def __getattr__(name: str):
    if name == "SasLLMPipeline":
        from .pipeline import SasLLMPipeline

        return SasLLMPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*__all__, "SasLLMPipeline"])
