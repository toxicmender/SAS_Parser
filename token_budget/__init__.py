"""Shared, offline-tolerant token estimation. See ``token_budget/README.md``."""

from __future__ import annotations

import logging
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Offline fallback: roughly four characters per token.
_FALLBACK_CHARS_PER_TOKEN = 4

# Approximate ChatML framing; suitable for budgeting, not billing.
_TOKENS_PER_MESSAGE = 3
_REPLY_PRIMER_TOKENS = 3

# Keep specific model prefixes ahead of their general family.
_ENCODING_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("gpt-5", "o200k_base"),
    ("gpt-4o", "o200k_base"),
    ("gpt-4.1", "o200k_base"),
    ("chatgpt-4o", "o200k_base"),
    ("o1", "o200k_base"),
    ("o3", "o200k_base"),
    ("o4", "o200k_base"),
    ("gpt-4", "cl100k_base"),
    ("gpt-3.5", "cl100k_base"),
)
_DEFAULT_ENCODING = "o200k_base"

# Cache failed loads too, avoiding repeated BPE fetch attempts.
_encodings: dict[str, Any] = {}
_warned_encodings: set[str] = set()


def encoding_name_for_model(model: str | None) -> str:
    """The tiktoken encoding name that estimates tokens for *model*.

    Tolerates a LangChain provider prefix (``"anthropic:claude-..."``);
    ``None`` or an unrecognised id resolves to the default (``o200k_base``).
    """
    if not model:
        return _DEFAULT_ENCODING
    bare = model.split(":", 1)[-1].lower()
    for prefix, encoding in _ENCODING_BY_PREFIX:
        if bare.startswith(prefix):
            return encoding
    return _DEFAULT_ENCODING


def _encoding(name: str) -> Any | None:
    """The loaded encoding, or ``None`` when it cannot be loaded (cached)."""
    if name in _encodings:
        return _encodings[name]
    try:
        import tiktoken

        encoding = tiktoken.get_encoding(name)
    except Exception as exc:
        encoding = None
        if name not in _warned_encodings:
            _warned_encodings.add(name)
            logger.warning(
                f"_encoding: could not load tiktoken encoding {name!r} "
                f"({exc!r}); falling back to the chars//{_FALLBACK_CHARS_PER_TOKEN} "
                f"approximation for it"
            )
    _encodings[name] = encoding
    return encoding


def count_text(text: str, *, model: str | None = None) -> int:
    """Estimated token count of *text* under *model*'s encoding.

    Degrades to ``len(text) // 4`` when the encoding is unavailable.
    """
    encoding = _encoding(encoding_name_for_model(model))
    if encoding is None:
        return len(text) // _FALLBACK_CHARS_PER_TOKEN
    return len(encoding.encode(text))


def _message_text(message: Any) -> str:
    """The countable text of one message: a plain string, an object carrying
    ``.content`` (a ``BaseMessage``, duck-typed), or a list of content parts
    whose ``text`` fields are concatenated (non-text parts count nothing)."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )
    return str(content)


def count_messages(messages: Iterable[Any], *, model: str | None = None) -> int:
    """Estimated prompt token count of *messages* under *model*'s encoding.

    Adds the ChatML per-message framing and reply-primer overheads, so the
    estimate tracks what a ``/chat/completions`` request is billed rather
    than the bare text. Message content degrades exactly as
    :func:`count_text` does.
    """
    total = _REPLY_PRIMER_TOKENS
    for message in messages:
        total += _TOKENS_PER_MESSAGE + count_text(
            _message_text(message), model=model
        )
    return total
