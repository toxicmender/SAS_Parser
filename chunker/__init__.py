"""SAS semantic chunker and dependency batcher. See chunker/README.md.

The LLM orchestration layer, structured response models, and notebook
renderer moved to the top-level ``pipeline`` package; their names still
resolve here for backward compatibility (with a DeprecationWarning) but new
code should import them from ``pipeline``.
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

# Names that moved to the top-level `pipeline` package, still resolvable here
# for backward compatibility: name -> (new module, attribute).
_MOVED_TO_PIPELINE = {
    "SasLLMPipeline": ("pipeline", "SasLLMPipeline"),
    "TranslationDocument": ("pipeline.response_models", "TranslationDocument"),
    "TranslationCell": ("pipeline.response_models", "TranslationCell"),
    "MappingEntry": ("pipeline.response_models", "MappingEntry"),
    "RiskNote": ("pipeline.response_models", "RiskNote"),
    "write_notebooks": ("pipeline.notebook", "write_notebooks"),
    "notebooks_from_outputs": ("pipeline.notebook", "notebooks_from_outputs"),
    "build_notebook": ("pipeline.notebook", "build_notebook"),
    "notebook_to_json": ("pipeline.notebook", "notebook_to_json"),
    "document_to_cells": ("pipeline.notebook", "document_to_cells"),
    "markdown_to_cells": ("pipeline.notebook", "markdown_to_cells"),
}


def __getattr__(name: str):
    moved = _MOVED_TO_PIPELINE.get(name)
    if moved is not None:
        import importlib
        import warnings

        module, attr = moved
        warnings.warn(
            f"importing {name!r} from 'chunker' is deprecated; import it "
            f"from {module!r} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*__all__, *_MOVED_TO_PIPELINE])
