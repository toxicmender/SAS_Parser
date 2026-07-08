"""LangChain chat-model construction and invocation. See llm_client/README.md.

Logger name: ``llm_client.client``.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable

from langchain.chat_models import init_chat_model
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage, HumanMessage, convert_to_messages
from langchain_core.prompt_values import PromptValue
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class InputTokenLimitError(ValueError):
    """Prompt exceeds the configured input-token budget; nothing was sent."""

    def __init__(self, token_count: int, max_input_tokens: int) -> None:
        self.token_count = token_count
        self.max_input_tokens = max_input_tokens
        super().__init__(
            f"prompt is {token_count} tokens, over the max_input_tokens "
            f"budget of {max_input_tokens}"
        )


class LLMClientConfig(BaseModel):
    """
    Declarative knobs for :class:`LLMClient`.

    Parameters
    ----------
    model : str
        LangChain chat-model string forwarded to ``init_chat_model``,
        e.g. ``"claude-haiku-4-5-20251001"``.
    temperature : float | None
        Sampling temperature. ``None`` (default) leaves the provider
        default untouched.
    max_output_tokens : int | None
        Cap on generated tokens (provider ``max_tokens``). ``None`` keeps
        the provider default.
    max_input_tokens : int | None
        Input-token budget per call. When set, the prompt is counted
        before invocation and :class:`InputTokenLimitError` is raised if
        it exceeds the budget. ``None`` (default) disables counting
        entirely — no token-counter is ever called.
    requests_per_second : float | None
        Proactive client-side throttle: at most this many request starts
        per second via ``InMemoryRateLimiter``. ``None`` disables it.
        Only applies to models built by the client (an injected ``llm``
        is used as-is; rate limiters attach at construction time).
    max_bucket_size : int
        Burst allowance for the rate limiter.
    max_retries : int
        Retries *after* the first attempt for rate-limit-shaped errors
        (HTTP 429 etc.). Non-rate-limit errors are never retried.
    retry_base_seconds, retry_max_seconds : float
        Exponential-backoff schedule: attempt *n* waits
        ``min(base * 2**(n-1), max)`` scaled by 0.5–1.5x jitter.
    token_counter : Callable[[list[BaseMessage]], int] | None
        Custom counter for the input-token budget. ``None`` uses the
        model's own ``get_num_tokens_from_messages``, falling back to a
        chars//4 approximation (with a one-time WARNING) if that raises —
        e.g. offline, or for fake models without a tokenizer.
    """

    model: str = "claude-haiku-4-5-20251001"
    temperature: float | None = Field(default=None, ge=0.0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    max_input_tokens: int | None = Field(default=None, gt=0)
    requests_per_second: float | None = Field(default=None, gt=0.0)
    max_bucket_size: int = Field(default=1, gt=0)
    max_retries: int = Field(default=3, ge=0)
    retry_base_seconds: float = Field(default=1.0, gt=0.0)
    retry_max_seconds: float = Field(default=30.0, gt=0.0)
    token_counter: Callable[[list[BaseMessage]], int] | None = None


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Provider-agnostic 'is this a rate limit?' check (429 / naming)."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 429:
        return True
    if "ratelimit" in type(exc).__name__.lower():
        return True
    text = str(exc).lower()
    return "rate limit" in text or "rate_limit" in text or "too many requests" in text


def _as_messages(input: LanguageModelInput) -> list[BaseMessage]:
    if isinstance(input, PromptValue):
        return input.to_messages()
    if isinstance(input, str):
        return [HumanMessage(input)]
    return convert_to_messages(input)


class LLMClient:
    """
    Thin invocation layer around a LangChain chat model.

    Use :meth:`invoke` directly, or :meth:`as_runnable` to drop the client
    into a LCEL chain in place of the raw model
    (``prompt | client.as_runnable()``).

    Parameters
    ----------
    config : LLMClientConfig | None
        Knobs; defaults to ``LLMClientConfig()``.
    llm : Any | None
        Pre-built chat model to use instead of constructing one from
        ``config.model`` (e.g. a fake in tests). Retry and the
        input-token budget still apply to an injected model;
        ``temperature`` / ``max_output_tokens`` / ``requests_per_second``
        do not, since those attach at construction time.
    """

    def __init__(
        self, config: LLMClientConfig | None = None, *, llm: Any | None = None
    ) -> None:
        self.config = config or LLMClientConfig()
        self._warned_approx_counter = False
        self._model = llm if llm is not None else self._build_model(self.config)

    @property
    def chat_model(self) -> Any:
        """The underlying LangChain chat model."""
        return self._model

    @staticmethod
    def _build_model(config: LLMClientConfig) -> Any:
        kwargs: dict[str, Any] = {}
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if config.max_output_tokens is not None:
            kwargs["max_tokens"] = config.max_output_tokens
        if config.requests_per_second is not None:
            kwargs["rate_limiter"] = InMemoryRateLimiter(
                requests_per_second=config.requests_per_second,
                max_bucket_size=config.max_bucket_size,
            )
        logger.info(
            f"LLMClient: building model '{config.model}'  "
            f"temperature={config.temperature}  "
            f"max_output_tokens={config.max_output_tokens}  "
            f"requests_per_second={config.requests_per_second}  "
            f"max_retries={config.max_retries}"
        )
        return init_chat_model(config.model, **kwargs)

    # ------------------------------------------------------------------
    # Input-token budget
    # ------------------------------------------------------------------

    def count_tokens(self, messages: list[BaseMessage]) -> int:
        """Count prompt tokens using the configured or model-native counter."""
        if self.config.token_counter is not None:
            return self.config.token_counter(messages)
        try:
            return self._model.get_num_tokens_from_messages(messages)
        except Exception as exc:
            if not self._warned_approx_counter:
                logger.warning(
                    f"count_tokens: model token counter unavailable ({exc!r}); "
                    f"falling back to chars//4 approximation"
                )
                self._warned_approx_counter = True
            return sum(len(str(m.content)) for m in messages) // 4

    def _enforce_input_limit(self, messages: list[BaseMessage]) -> None:
        limit = self.config.max_input_tokens
        if limit is None:
            return
        tokens = self.count_tokens(messages)
        if tokens > limit:
            logger.error(
                f"_enforce_input_limit: prompt of {tokens} tokens exceeds "
                f"max_input_tokens={limit}; request not sent"
            )
            raise InputTokenLimitError(tokens, limit)
        logger.debug(
            f"_enforce_input_limit: input_tokens={tokens} <= max_input_tokens={limit}"
        )

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    def invoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None
    ) -> BaseMessage:
        """
        Invoke the model with the input-token budget enforced and
        rate-limit errors retried with exponential backoff.
        """
        messages = _as_messages(input)
        self._enforce_input_limit(messages)

        attempts = self.config.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return self._model.invoke(messages, config)
            except Exception as exc:
                if attempt >= attempts or not _is_rate_limit_error(exc):
                    raise
                delay = min(
                    self.config.retry_base_seconds * 2 ** (attempt - 1),
                    self.config.retry_max_seconds,
                )
                delay *= 0.5 + random.random()  # 0.5x–1.5x jitter
                logger.warning(
                    f"invoke: rate limited (attempt {attempt}/{attempts}), "
                    f"retrying in {delay:.2f}s: {exc}"
                )
                time.sleep(delay)
        raise AssertionError("unreachable")  # loop always returns or raises

    def as_runnable(self) -> Runnable[LanguageModelInput, BaseMessage]:
        """The client as a LCEL Runnable, e.g. ``prompt | client.as_runnable()``."""
        return RunnableLambda(self.invoke, name="llm_client")
