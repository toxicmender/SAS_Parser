"""LangChain chat-model construction and invocation. See llm_client/README.md.

Logger name: ``llm_client.client``.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Callable

import app_config
from langchain.chat_models import init_chat_model
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage, HumanMessage, convert_to_messages
from langchain_core.prompt_values import PromptValue
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr

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

    Where noted, an *omitted* argument defaults from the ``llm_client``
    section of config.json (see the ``app_config`` package), following the
    repo-wide precedence rule: explicit argument > config.json > hard
    default. Passing an explicit value (including ``None``) always wins.
    File values are parsed through :func:`app_config.llm_client_value`,
    which type-checks them against the section schema — a wrong-typed
    entry is ignored with a WARNING and the hard default applies, instead
    of failing construction.

    Parameters
    ----------
    model : str
        LangChain chat-model string forwarded to ``init_chat_model``,
        e.g. ``"claude-sonnet-4-5"``. Also accepted under the
        alias ``model_name``. Config key: ``llm_client.model``.
    base_url : str | None
        Provider endpoint override (proxy / gateway URL), forwarded as
        ``base_url``. ``None`` keeps the provider default. Config key:
        ``llm_client.base_url``.
    api_key : SecretStr | str | None
        Explicit API key, forwarded as ``api_key``. ``None`` (default)
        defers to the provider's environment variable (e.g.
        ``ANTHROPIC_API_KEY``). Stored as a ``SecretStr`` so it is masked
        in ``repr`` and never logged. Deliberately NOT read from
        config.json — secrets do not belong in a committed file.
    url_headers : dict[str, str] | None
        Extra HTTP headers sent with every request (gateway auth,
        tracing, ...), forwarded as ``default_headers``. Config key:
        ``llm_client.url_headers``.
    timeout : float | None
        Per-request timeout in seconds, forwarded as ``timeout``.
        ``None`` keeps the provider default. Config key:
        ``llm_client.timeout``.
    temperature : float | None
        Sampling temperature. ``None`` (default) leaves the provider
        default untouched. Config key: ``llm_client.temperature``.
    max_output_tokens : int | None
        Cap on generated tokens (provider ``max_tokens``). ``None`` keeps
        the provider default. Config key: ``llm_client.max_output_tokens``.
    max_input_tokens : int | None
        Input-token budget per call. When set, the prompt is counted
        before invocation and :class:`InputTokenLimitError` is raised if
        it exceeds the budget. ``None`` disables counting entirely — no
        token-counter is ever called. Config key:
        ``llm_client.max_input_tokens``.
    requests_per_second : float | None
        Proactive client-side throttle: at most this many request starts
        per second via ``InMemoryRateLimiter``. ``None`` disables it.
        Only applies to models built by the client (an injected ``llm``
        is used as-is; rate limiters attach at construction time).
    max_bucket_size : int
        Burst allowance for the rate limiter.
    max_retries : int
        Retries *after* the first attempt for transient errors: rate
        limits (HTTP 429), overload / server errors (500, 502, 503, 504,
        529), timeouts, and connection drops. Non-transient errors are
        never retried. This is the only retry loop — the provider SDK's
        own retry layer is not configured here. Config key:
        ``llm_client.max_retries``.
    retry_base_seconds, retry_max_seconds : float
        Exponential-backoff schedule: attempt *n* waits
        ``min(base * 2**(n-1), max)`` scaled by 0.5–1.5x jitter.
    token_counter : Callable[[list[BaseMessage]], int] | None
        Custom counter for the input-token budget. ``None`` uses the
        model's own ``get_num_tokens_from_messages``, falling back to a
        chars//4 approximation (with a one-time WARNING) if that raises —
        e.g. offline, or for fake models without a tokenizer.
    model_kwargs : dict[str, Any] | None
        Provider-specific request-body extras forwarded as
        ``model_kwargs`` (e.g. ``{"top_k": 40}``). Config key:
        ``llm_client.model_kwargs``.
    kwargs : dict[str, Any] | None
        Escape hatch: arbitrary keyword arguments merged **last** into
        the ``init_chat_model`` call, overriding anything the named knobs
        produced. Constructor-only (not read from config.json).
    """

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    # Config-backed defaults go through app_config.llm_client_value, which
    # type-checks the file value against the section schema and degrades a
    # wrong-typed entry to the hard default with a WARNING.
    model: str = Field(
        default_factory=lambda: app_config.llm_client_value(
            "model", "claude-sonnet-4-5"
        ),
        validation_alias=AliasChoices("model", "model_name"),
    )
    base_url: str | None = Field(
        default_factory=lambda: app_config.llm_client_value("base_url")
    )
    api_key: SecretStr | None = None
    url_headers: dict[str, str] | None = Field(
        default_factory=lambda: app_config.llm_client_value("url_headers")
    )
    timeout: float | None = Field(
        default_factory=lambda: app_config.llm_client_value("timeout"),
        gt=0.0,
    )
    temperature: float | None = Field(
        default_factory=lambda: app_config.llm_client_value("temperature"),
        ge=0.0,
    )
    max_output_tokens: int | None = Field(
        default_factory=lambda: app_config.llm_client_value("max_output_tokens"),
        gt=0,
    )
    max_input_tokens: int | None = Field(
        default_factory=lambda: app_config.llm_client_value("max_input_tokens"),
        gt=0,
    )
    requests_per_second: float | None = Field(default=None, gt=0.0)
    max_bucket_size: int = Field(default=1, gt=0)
    max_retries: int = Field(
        default_factory=lambda: app_config.llm_client_value("max_retries", 3),
        ge=0,
    )
    retry_base_seconds: float = Field(default=1.0, gt=0.0)
    retry_max_seconds: float = Field(default=30.0, gt=0.0)
    token_counter: Callable[[list[BaseMessage]], int] | None = None
    model_kwargs: dict[str, Any] | None = Field(
        default_factory=lambda: app_config.llm_client_value("model_kwargs")
    )
    kwargs: dict[str, Any] | None = None


# Server-side statuses worth retrying: throttling (429), plain server
# errors (500/502/503/504), and Anthropic's "overloaded" (529).
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 529}


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Provider-agnostic 'is this a rate limit?' check (429 / naming)."""
    if _status_code(exc) == 429:
        return True
    if "ratelimit" in type(exc).__name__.lower():
        return True
    text = str(exc).lower()
    return "rate limit" in text or "rate_limit" in text or "too many requests" in text


def _is_transient_error(exc: BaseException) -> bool:
    """Worth retrying: rate limits, overload / 5xx, timeouts, connection drops.

    Type-based checks only (status codes, exception classes/names) apart
    from the rate-limit message match — a permanent error whose *message*
    merely mentions a connection must still fail fast.
    """
    if _is_rate_limit_error(exc):
        return True
    if _status_code(exc) in _RETRYABLE_STATUS_CODES:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__.lower()
    return "timeout" in name or "connect" in name or "overloaded" in name


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
        if config.base_url is not None:
            kwargs["base_url"] = config.base_url
        if config.api_key is not None:
            kwargs["api_key"] = config.api_key.get_secret_value()
        if config.url_headers:
            kwargs["default_headers"] = dict(config.url_headers)
        if config.timeout is not None:
            kwargs["timeout"] = config.timeout
        if config.model_kwargs:
            kwargs["model_kwargs"] = dict(config.model_kwargs)
        if config.requests_per_second is not None:
            kwargs["rate_limiter"] = InMemoryRateLimiter(
                requests_per_second=config.requests_per_second,
                max_bucket_size=config.max_bucket_size,
            )
        if config.kwargs:
            kwargs.update(config.kwargs)  # escape hatch wins over named knobs
        # Header VALUES may carry gateway credentials — log key names only.
        logger.info(
            f"LLMClient: building model '{config.model}'  "
            f"temperature={config.temperature}  "
            f"max_output_tokens={config.max_output_tokens}  "
            f"base_url={config.base_url}  "
            f"timeout={config.timeout}  "
            f"api_key={'set' if config.api_key else 'unset'}  "
            f"url_headers={sorted(config.url_headers) if config.url_headers else None}  "
            f"model_kwargs={config.model_kwargs}  "
            f"extra_kwargs={sorted(config.kwargs) if config.kwargs else None}  "
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

    def _retry_delay(
        self, exc: Exception, attempt: int, attempts: int, op: str
    ) -> float | None:
        """Backoff delay before the next attempt, or ``None`` when *exc*
        must propagate (non-transient, or attempts exhausted)."""
        if not _is_transient_error(exc):
            return None
        if attempt >= attempts:
            logger.error(
                f"{op}: transient error persisted through {attempts} "
                f"attempt(s); giving up: {exc!r}"
            )
            return None
        delay = min(
            self.config.retry_base_seconds * 2 ** (attempt - 1),
            self.config.retry_max_seconds,
        )
        delay *= 0.5 + random.random()  # 0.5x–1.5x jitter
        logger.warning(
            f"{op}: transient error (attempt {attempt}/{attempts}), "
            f"retrying in {delay:.2f}s: {exc}"
        )
        return delay

    def invoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None
    ) -> BaseMessage:
        """
        Invoke the model with the input-token budget enforced and
        transient errors (rate limits, overload / 5xx, timeouts,
        connection drops) retried with exponential backoff.
        """
        messages = _as_messages(input)
        self._enforce_input_limit(messages)

        attempts = self.config.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return self._model.invoke(messages, config)
            except Exception as exc:
                delay = self._retry_delay(exc, attempt, attempts, "invoke")
                if delay is None:
                    raise
                time.sleep(delay)
        raise AssertionError("unreachable")  # loop always returns or raises

    async def ainvoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None
    ) -> BaseMessage:
        """
        Async :meth:`invoke`: same input-token budget and transient-error
        retry, with non-blocking backoff. Token counting runs in a worker
        thread because model-native counters may call a synchronous HTTP
        endpoint.
        """
        messages = _as_messages(input)
        await asyncio.to_thread(self._enforce_input_limit, messages)

        attempts = self.config.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return await self._model.ainvoke(messages, config)
            except Exception as exc:
                delay = self._retry_delay(exc, attempt, attempts, "ainvoke")
                if delay is None:
                    raise
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")  # loop always returns or raises

    def as_runnable(self) -> Runnable[LanguageModelInput, BaseMessage]:
        """The client as a LCEL Runnable, e.g. ``prompt | client.as_runnable()``.

        Bound to both paths: ``chain.invoke`` uses :meth:`invoke` and
        ``chain.ainvoke`` uses :meth:`ainvoke`.
        """
        return RunnableLambda(self.invoke, afunc=self.ainvoke, name="llm_client")
