"""LLM conversion pipeline and notebook renderer. See ``pipeline/README.md``."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Keep lazy exports precise for type checkers.
    from .engine import SasLLMPipeline as SasLLMPipeline
    from .setup import MemorySetup as MemorySetup

from .notebook import (
    build_notebook,
    document_to_cells,
    item_cells,
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
    # orchestration (lazy)
    "SasLLMPipeline",
    "MemorySetup",
    # structured LLM response
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
    "item_cells",
]


def __getattr__(name: str):
    if name == "SasLLMPipeline":
        from .engine import SasLLMPipeline

        return SasLLMPipeline
    if name == "MemorySetup":
        from .setup import MemorySetup

        return MemorySetup
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
