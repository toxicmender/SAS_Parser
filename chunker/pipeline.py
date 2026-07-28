"""Deprecated location: ``chunker.pipeline`` moved to the ``pipeline`` package.

Import :class:`SasLLMPipeline` from ``pipeline`` (orchestration lives in
``pipeline.engine``; the item → prompt mapping and message formatting in
``pipeline.prompting``). This shim re-exports the public class plus the
helper functions external callers and older tests reached in here for, and
will be removed in a future release.
"""

import warnings

from pipeline.engine import SasLLMPipeline
from pipeline.prompting import (
    _constructs_for_item,
    _format_batch_message,
    _kinds_for_item,
    _meta_flags_for_item,
    _query_for_chunk,
    _query_for_item,
)

warnings.warn(
    "chunker.pipeline is deprecated; import from the 'pipeline' package "
    "instead (pipeline.SasLLMPipeline)",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "SasLLMPipeline",
    "_constructs_for_item",
    "_format_batch_message",
    "_kinds_for_item",
    "_meta_flags_for_item",
    "_query_for_chunk",
    "_query_for_item",
]
