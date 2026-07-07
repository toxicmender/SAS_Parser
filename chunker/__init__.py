"""
chunker — SAS semantic chunker, dependency batcher, and LangChain pipeline.

Single-file workflow
--------------------
    from chunker import SasSemanticChunker, SasChunkBatcher

    chunker = SasSemanticChunker()
    result  = chunker.chunk_file("program.sas")

    batcher = SasChunkBatcher()
    batches = batcher.batch(result)

Multi-file workflow
-------------------
    from chunker import SasSemanticChunker, SasCorpus
    from chunker.batcher import MultiFileBatcher

    chunker = SasSemanticChunker()
    corpus  = SasCorpus(file_results=[
        chunker.chunk_file("macros.sas"),
        chunker.chunk_file("etl.sas"),
        chunker.chunk_file("reports.sas"),
    ])
    result = MultiFileBatcher().batch(corpus)

    # Or use the convenience factory:
    corpus, result = MultiFileBatcher.from_files([
        "macros.sas", "etl.sas", "reports.sas",
    ])

    for item in result.all_ordered_items:
        ...  # SasBatch or SasChunk, cross-file batches included
"""

from .batcher import MultiFileBatcher, SasChunkBatcher
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


def __getattr__(name: str):
    if name == "SasLLMPipeline":
        from .pipeline import SasLLMPipeline

        return SasLLMPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
