"""Offline-tolerant token estimation without importing provider clients."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

_FALLBACK_CHARS_PER_TOKEN = 4
_TOKENS_PER_MESSAGE = 3
_REPLY_PRIMER_TOKENS = 3
_DEFAULT_ENCODING = "o200k_base"
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


def encoding_name_for_model(model: str | None) -> str:
    """Resolve a stable estimator encoding without provider-specific imports."""

    if not model:
        return _DEFAULT_ENCODING
    bare = model.split(":", 1)[-1].lower()
    for prefix, encoding in _ENCODING_BY_PREFIX:
        if bare.startswith(prefix):
            return encoding
    return _DEFAULT_ENCODING


class TokenCounter(Protocol):
    estimator: str
    encoding: str
    approximate: bool

    def count_text(self, text: str) -> int: ...

    def framing_tokens(self, message_count: int) -> int: ...


class TokenEstimator:
    """Tiktoken estimator with an explicit, auditable approximation fallback."""

    def __init__(
        self,
        model: str | None = None,
        *,
        encoding: str | None = None,
        text_counter: Callable[[str], int] | None = None,
        estimator: str = "tiktoken",
        approximate: bool = False,
    ) -> None:
        self.encoding = encoding or encoding_name_for_model(model)
        self.estimator = estimator
        self.approximate = approximate
        self._text_counter = text_counter
        self._encoder: Any | None = None
        self._loaded = text_counter is not None

    @classmethod
    def approximate_for_model(cls, model: str | None = None) -> TokenEstimator:
        return cls(
            model,
            text_counter=lambda text: (
                0
                if not text
                else max(1, len(text) // _FALLBACK_CHARS_PER_TOKEN)
            ),
            estimator="character_approximation",
            approximate=True,
        )

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            import tiktoken

            self._encoder = tiktoken.get_encoding(self.encoding)
        except Exception:  # noqa: BLE001 - tokenizer loading has backend errors
            fallback = self.approximate_for_model()
            self._text_counter = fallback._text_counter
            self.estimator = fallback.estimator
            self.approximate = True

    def count_text(self, text: str) -> int:
        self._load()
        if self._text_counter is not None:
            return max(0, self._text_counter(text))
        if self._encoder is None:
            raise RuntimeError("token estimator failed to initialize")
        return len(self._encoder.encode(text))

    def framing_tokens(self, message_count: int) -> int:
        if message_count < 0:
            raise ValueError("message_count cannot be negative")
        return _REPLY_PRIMER_TOKENS + (_TOKENS_PER_MESSAGE * message_count)


__all__ = [
    "TokenCounter",
    "TokenEstimator",
    "encoding_name_for_model",
]
