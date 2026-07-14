"""Turn grouping and light text stats shared across the memory package.

Deliberately dependency-light (langchain_core only): ``memory.summarize``
imports these helpers without dragging in the bm25s / faiss stack that
``memory.relevance`` needs.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage


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

    Used wherever an exact tokenizer is not worth a network/model
    dependency; callers that have a real counter pass it instead.
    """
    return len(text) // 4 + 1
