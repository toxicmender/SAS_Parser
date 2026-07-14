"""Reference-PDF instruction retrieval for the SAS-to-target migration prompt.

Reads reference PDFs (SAS language manuals, target-platform guides) once,
segments them into instruction sections, and — in later phases — chunks,
indexes, and retrieves the sections most relevant to each pipeline item so
targeted guidance can be injected into the LLM prompt.

A regular package (not a PEP-420 namespace package) so packaging tools and
import machinery treat it uniformly. Imports nothing from ``chunker`` or
``llm_client``; the pipeline remains the sole integration point.

See prompt_builder/README.md.
"""

from .builder import PromptBuilder
from .catalog import CorpusLoader, DocumentSpec, default_catalog
from .models import (
    ConstructKey,
    DocRole,
    InstructionChunk,
    SelectedInstruction,
    SelectionTier,
)
from .selector import InstructionSelector
from .user_instructions import UserInstructionSet

__all__ = [
    "PromptBuilder",
    "InstructionSelector",
    "CorpusLoader",
    "DocumentSpec",
    "default_catalog",
    "ConstructKey",
    "DocRole",
    "InstructionChunk",
    "SelectedInstruction",
    "SelectionTier",
    "UserInstructionSet",
]
