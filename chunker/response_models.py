"""Deprecated location: moved to ``pipeline.response_models``.

This shim will be removed in a future release.
"""

import warnings

from pipeline.response_models import (
    MappingEntry,
    RiskNote,
    TranslationCell,
    TranslationDocument,
)

warnings.warn(
    "chunker.response_models is deprecated; import from "
    "pipeline.response_models instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["MappingEntry", "RiskNote", "TranslationCell", "TranslationDocument"]
