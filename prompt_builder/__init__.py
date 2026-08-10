"""Reference-guidance retrieval for conversion prompts. See ``prompt_builder/README.md``."""

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
