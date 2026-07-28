"""Turn grouping and light text stats shared across the memory package.

Deliberately dependency-light (langchain_core only; tiktoken lazily):
``memory.summarize`` imports these helpers without dragging in the bm25s /
faiss stack that ``memory.relevance`` needs.

:func:`token_count` is the package's default text-token counter — a real
tiktoken run under ``o200k_base`` when the encoding is available, else the
:func:`approx_token_count` estimate. The o200k choice deliberately matches
``llm_client.tokens`` (which owns the model-id → encoding map; this module
cannot import it — ``memory`` and ``llm_client`` never import each other),
so the summarizer trigger, history packing, and the input-token budget all
count under one vocabulary by default.

Logger name: ``memory.turns``.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

logger = logging.getLogger(__name__)


def group_turns(history: list[BaseMessage]) -> list[list[BaseMessage]]:
    """
    Group a chronological message list into turns: each HumanMessage opens
    a new turn and every following non-human message (AI, tool, …) joins
    it. Leading non-human messages form a turn of their own, so no message
    is ever dropped by grouping.
    """
    turns: list[list[BaseMessage]] = []
    for msg in history:
        if isinstance(msg, HumanMessage) or not turns:
            turns.append([msg])
        else:
            turns[-1].append(msg)
    return turns


def turn_text(turn: list[BaseMessage]) -> str:
    return "\n".join(str(m.content) for m in turn)


def approx_token_count(text: str) -> int:
    """Cheap offline token estimate (~4 chars/token for English/code).

    The degradation target of :func:`token_count`, and still directly
    usable by callers that explicitly want the offline estimate.
    """
    return len(text) // 4 + 1


# tiktoken o200k_base encoding, loaded once on first use; False once loading
# has failed (offline, a blocking proxy), so the fetch attempt is paid for
# once, not per call.
_ENCODING_NAME = "o200k_base"
_encoding: Any = None


def token_count(text: str) -> int:
    """Default text-token counter: tiktoken ``o200k_base`` when available,
    else :func:`approx_token_count` (one-time WARNING on the first fallback).
    """
    global _encoding
    if _encoding is None:
        try:
            import tiktoken

            _encoding = tiktoken.get_encoding(_ENCODING_NAME)
        except Exception as exc:
            _encoding = False
            logger.warning(
                f"token_count: could not load tiktoken encoding "
                f"{_ENCODING_NAME!r} ({exc!r}); using the ~4-chars/token "
                f"approximation instead"
            )
    if _encoding is False:
        return approx_token_count(text)
    return len(_encoding.encode(text))
