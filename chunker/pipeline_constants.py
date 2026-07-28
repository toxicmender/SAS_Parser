"""Deprecated location: moved to ``pipeline.constants``.

This shim will be removed in a future release.
"""

import warnings

from pipeline.constants import (
    _BATCH_CONTEXT_TEMPLATE,
    _BATCH_MEMBER_TEMPLATE,
    _STRUCTURED_SYSTEM_PROMPT_TEMPLATE,
    _SYSTEM_PROMPT_TEMPLATE,
)

warnings.warn(
    "chunker.pipeline_constants is deprecated; import from "
    "pipeline.constants instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "_BATCH_CONTEXT_TEMPLATE",
    "_BATCH_MEMBER_TEMPLATE",
    "_STRUCTURED_SYSTEM_PROMPT_TEMPLATE",
    "_SYSTEM_PROMPT_TEMPLATE",
]
