"""OpenAI-compatible chat-model construction and invocation. See
llm_client/README.md.

Every model is reached through the AI Gateway's OpenAI-compatible
``/chat/completions`` API — including the Anthropic- and Google-served ones,
which the gateway exposes under the same protocol. So the default transport is
one chat-model class (``langchain_openai.ChatOpenAI``, a thin LangChain wrapper
over the ``openai`` SDK) rather than a provider lookup: the model name selects
the model, not the transport. A provider prefix on the model id
(``"anthropic: claude-sonnet-4-5"``) is recorded, not routed on.

``LLMClientConfig.provider_client="native"`` is the escape hatch for a gateway
that will not accept the payload LangChain builds: it swaps in a raw
``openai.OpenAI`` call and forfeits everything that attaches at construction
time (rate limiter, ``model_kwargs``, structured output). It is not the
default, because the surrounding machinery — retry with ``Retry-After``, the
token budget, prompt-cache breakpoints, usage accounting — is worth more than
the payload difference until a gateway proves otherwise.

Logger name: ``llm_client.client``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable

import app_config
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    convert_to_messages,
)
from langchain_core.prompt_values import PromptValue
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr

import token_budget as tokens

logger = logging.getLogger(__name__)


class LLMClientError(RuntimeError):
    """The chat model could not be constructed from this configuration.

    Raised before any request is made — a misconfiguration, not a transport
    failure — so a caller can tell "this run was never going to work" from "the
    gateway said no".
    """


class InputTokenLimitError(ValueError):
    """Prompt exceeds the configured input-token budget; nothing was sent."""

    def __init__(self, token_count: int, max_input_tokens: int) -> None:
        self.token_count = token_count
        self.max_input_tokens = max_input_tokens
        super().__init__(
            f"prompt is {token_count} tokens, over the max_input_tokens "
            f"budget of {max_input_tokens}"
        )


# The llm_client keys a role overlay may carry — every schema key that names
# a field on LLMClientConfig. "roles" is structural and "prompt_caching"
# belongs to the pipeline, not to this config.
_ROLE_OVERLAY_KEYS: tuple[str, ...] = (
    "model",
    "model_provider",
    "gateway_version",
    "provider_client",
    "base_url",
    "url_headers",
    "timeout",
    "cert_file",
    "temperature",
    "max_retries",
    "model_kwargs",
    "max_input_tokens",
    "max_output_tokens",
    "requests_per_second",
    "max_bucket_size",
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
        Model id as the gateway names it, e.g. ``"gpt-5.4"`` (the default),
        ``"claude-sonnet-4-5"`` or ``"gemini-3.1-pro"`` — it is sent verbatim in the
        request body, so an Anthropic- or Google-served model is named
        exactly like an OpenAI one. A ``"provider: model"`` prefix
        (``"anthropic: claude-opus-4-6"``) is split off into
        :attr:`model_provider` and the remainder lowercased, matching how
        the reference gateway's own client parses the value it reads from
        SharePoint. Also accepted under the alias ``model_name``.
        Config key: ``llm_client.model``.
    model_provider : str | None
        The family the gateway routes to (``"anthropic"``, ``"openai"``,
        ...). Filled in from a ``"provider: model"`` prefix on
        :attr:`model` when unset. Informational for the default transport —
        the gateway routes on the model id — and the reason a deployment
        might pin :attr:`provider_client`. Config key:
        ``llm_client.model_provider``.
    gateway_version : str | None
        The gateway's protocol version, sent as the ``ai-gateway-version``
        request header on every call (the reference sends ``"v2"``). An
        ``ai-gateway-version`` supplied through :attr:`url_headers` wins.
        ``None`` (default) sends no such header. ``LLM_GATEWAY_VERSION`` /
        config key ``llm_client.gateway_version``;
        :meth:`from_ai_gateway` also picks it up from the Vault secret.
    provider_client : str | None
        How the chat model is built:

        ``"auto"``
            The default — :attr:`model_provider` decides, through
            ``app_config.PROVIDER_CLIENT_BY_PROVIDER``: ``anthropic`` routes
            to ``"native"``, ``openai`` and ``google`` to
            ``"openai_compatible"``. This is what the reference deployment
            does against this same gateway, so it is the behaviour a
            gateway-hosted model gets without configuring anything. A model
            id with no provider on it (and no ``model_provider`` set) has
            nothing to route on and takes ``"openai_compatible"``; a named
            but unrecognised provider raises :class:`LLMClientError` rather
            than being guessed at.
        ``"openai_compatible"``
            ``langchain_openai.ChatOpenAI``, which is what this whole class
            is written around. Set it explicitly to keep the LangChain
            transport for a provider that would otherwise route to native.
        ``"native"``
            A raw ``openai.OpenAI`` client wrapped to the minimal surface
            :class:`LLMClient` invokes, for a gateway that will not accept
            the LangChain payload. **Everything that attaches at
            construction time is forfeited**: the rate limiter
            (:attr:`requests_per_second`), :attr:`model_kwargs`, and
            structured output — for which ``pipeline.notebook`` already
            degrades to parsing the Markdown response. Retry with
            ``Retry-After``, the input-token budget, and usage accounting
            still apply, since those live in :class:`LLMClient` rather than
            in the model. Configuring a forfeited knob alongside it is
            logged at WARNING.

        An explicit value always wins over the provider, in either direction.
        Config key: ``llm_client.provider_client``.
    base_url : str | None
        Gateway endpoint (the OpenAI-compatible root, e.g.
        ``https://llm-gateway.example/v1``), forwarded as ``base_url``.
        A **trailing slash** marks a per-deployment route: the resolved
        model id is appended as the final path segment
        (``https://gw.example/openai/`` + ``gpt-5.4``), which is how a
        gateway that routes by URL rather than by request body is
        addressed. Without one the value is used verbatim. ``None`` falls
        back to ``OPENAI_BASE_URL`` and then to the public OpenAI
        endpoint — so in practice this, or the ``base_url`` carried by
        the Vault secret, must be set. Config key: ``llm_client.base_url``.
    api_key : SecretStr | str | None
        Explicit API key (the gateway token), forwarded as ``api_key``
        *and* mirrored into an ``api-key`` request header, since a
        gateway fronting Azure OpenAI authenticates on that header rather
        than the ``Authorization: Bearer`` the SDK sends. An ``api-key``
        supplied through :attr:`url_headers` wins over the mirror.
        ``None`` (default) defers to ``OPENAI_API_KEY`` in the
        environment. Stored as a ``SecretStr`` so it is masked in
        ``repr`` and never logged — the construction log records header
        *names* only, so neither copy leaks. Deliberately NOT read from
        config.json — secrets do not belong in a committed file.
    url_headers : dict[str, str] | None
        Extra HTTP headers sent with every request (gateway auth,
        tracing, ...), forwarded as ``default_headers``. Config key:
        ``llm_client.url_headers``.
    timeout : float | None
        Per-request timeout in seconds, forwarded as ``timeout``.
        ``None`` keeps the provider default. Config key:
        ``llm_client.timeout``.
    cert_file : str | None
        Path to a PEM certificate bundle (e.g. ``gateway.crt``) used to
        verify the endpoint's TLS certificate — needed when ``base_url``
        points at a gateway whose certificate is signed by an internal
        CA. Exported as the standard ``SSL_CERT_FILE`` environment
        variable (process-wide) before the model is built, which the
        ``openai`` SDK's httpx client honours, *and* passed as an explicit
        ``http_client`` / ``http_async_client`` pair pinned to that
        bundle — belt and braces, so verification survives anything else
        in the process rewriting the environment variable. Both clients
        are set; with only the sync one, ``ainvoke`` would fall back to
        the default trust store and fail against an internal CA. A
        missing file is skipped with a WARNING and the default trust
        store applies; if httpx cannot build the clients, the
        ``SSL_CERT_FILE`` export stands alone (also a WARNING). ``None``
        leaves the environment untouched. Config key:
        ``llm_client.cert_file``.
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
        per second via ``InMemoryRateLimiter``, so calls are paced *before*
        ever tripping a gateway 429. ``None`` (the default) disables it;
        :meth:`from_ai_gateway` turns it on (``2.0``) for the gateway
        credential path, since the gateway enforces a rate limit. Config
        key: ``llm_client.requests_per_second`` sets it for every path. Only
        applies to models built by the client (an injected ``llm`` is used
        as-is; rate limiters attach at construction time).
    max_bucket_size : int
        Burst allowance for the rate limiter. Config key:
        ``llm_client.max_bucket_size``.
    max_retries : int
        Retries *after* the first attempt for transient errors: rate
        limits (HTTP 429), overload / server errors (500, 502, 503, 504,
        529), timeouts, and connection drops. Non-transient errors are
        never retried. This is the *only* retry loop: the ``openai``
        SDK's own retry layer is explicitly disabled at construction
        (``max_retries=0``), so a 429 surfaces here with its
        ``Retry-After`` header intact instead of being silently re-sent
        underneath us. Config key: ``llm_client.max_retries``.
    retry_base_seconds, retry_max_seconds : float
        Exponential-backoff schedule used when the server gives no timing
        of its own: attempt *n* waits ``min(base * 2**(n-1), max)`` scaled
        by 0.5–1.5x jitter.
    retry_after_max_seconds : float
        Safety ceiling for a server-provided wait. On a 429 the gateway's
        ``Retry-After`` / ``retry-after-ms`` header is honored verbatim (no
        jitter, replacing the backoff) but never waited longer than this, so
        a pathological header value cannot hang the run. Default 300s.
    token_counter : Callable[[list[BaseMessage]], int] | None
        Custom counter for the input-token budget. ``None`` (default) uses
        the shared tiktoken counter (:mod:`llm_client.tokens`), which
        resolves the encoding from the model id by explicit prefix map:
        ``o200k_base`` for the modern GPT families **and** for every
        non-OpenAI id (Claude, Gemini — a real tokenizer run under a
        stand-in vocabulary; an estimate, though far closer than a guess),
        ``cl100k_base`` for the older GPT families. Pass a counter here
        when the budget has to be exact under the provider's own
        tokenization. When tiktoken cannot load its encoding data (offline,
        a blocking proxy) counting degrades to a chars//4 approximation
        with a one-time WARNING.
    model_kwargs : dict[str, Any] | None
        Request-body extras forwarded as ``model_kwargs`` (e.g.
        ``{"top_k": 40}``). These ride along in the ``/chat/completions``
        body, which is how the gateway takes model-specific knobs that
        are not part of the OpenAI schema. Config key:
        ``llm_client.model_kwargs``.
    kwargs : dict[str, Any] | None
        Escape hatch: arbitrary keyword arguments merged **last** into
        the ``ChatOpenAI(...)`` call, overriding anything the named knobs
        produced. Constructor-only (not read from config.json).
    """

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    # Config-backed defaults go through app_config.llm_client_value, which
    # type-checks the file value against the section schema and degrades a
    # wrong-typed entry to the hard default with a WARNING.
    model: str = Field(
        default_factory=lambda: app_config.llm_client_value(
            "model", "gpt-5.4"
        ),
        validation_alias=AliasChoices("model", "model_name"),
    )
    model_provider: str | None = Field(
        default_factory=lambda: app_config.llm_client_value("model_provider")
    )
    gateway_version: str | None = Field(
        default_factory=lambda: (
            os.environ.get("LLM_GATEWAY_VERSION")
            or app_config.llm_client_value("gateway_version")
        )
    )
    provider_client: str | None = Field(
        default_factory=lambda: app_config.llm_client_value("provider_client")
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
    cert_file: str | None = Field(
        default_factory=lambda: app_config.llm_client_value("cert_file")
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
    requests_per_second: float | None = Field(
        default_factory=lambda: app_config.llm_client_value("requests_per_second"),
        gt=0.0,
    )
    max_bucket_size: int = Field(
        default_factory=lambda: app_config.llm_client_value("max_bucket_size", 1),
        gt=0,
    )
    max_retries: int = Field(
        default_factory=lambda: app_config.llm_client_value("max_retries", 3),
        ge=0,
    )
    retry_base_seconds: float = Field(default=1.0, gt=0.0)
    retry_max_seconds: float = Field(default=30.0, gt=0.0)
    retry_after_max_seconds: float = Field(default=300.0, gt=0.0)
    token_counter: Callable[[list[BaseMessage]], int] | None = None
    model_kwargs: dict[str, Any] | None = Field(
        default_factory=lambda: app_config.llm_client_value("model_kwargs")
    )
    kwargs: dict[str, Any] | None = None

    @classmethod
    def for_role(cls, role: str, **overrides: Any) -> "LLMClientConfig":
        """
        A config resolved through the per-role overlay
        ``llm_client.roles.<role>`` in config.json.

        The reference deployment configures one gateway block per role — its
        validator gets a 12000-second timeout where the converter gets 6000 —
        and a role here is that same idea as a **sparse** overlay: a key the
        role does not mention resolves exactly as it would for
        ``LLMClientConfig()``. Roles in use: ``"validator"`` and
        ``"complexity"``. An unknown role name is not an error; it simply has
        no overrides.

        Parameters
        ----------
        role : str
            The overlay to apply.
        **overrides
            Ordinary constructor arguments, applied last — so an explicit
            ``model=`` from a CLI flag still beats the row, the role, and the
            base section, keeping the repo-wide precedence rule intact.
        """
        # Only keys the role actually *changes* are passed explicitly; every
        # other one falls through to the field's own default factory, which
        # stays the single source of truth for the hard defaults.
        values: dict[str, Any] = {}
        for key in _ROLE_OVERLAY_KEYS:
            if key in overrides:
                continue  # an explicit argument wins; do not read the file
            overlaid = app_config.role_value(role, key)
            if overlaid is not None and overlaid != app_config.llm_client_value(key):
                values[key] = overlaid
        applied = sorted(values)
        values.update(overrides)
        logger.info(
            f"LLMClientConfig.for_role: role={role!r} overrides from the file: "
            f"{applied or None} (explicit: {sorted(overrides) or None})"
        )
        return cls(**values)

    @classmethod
    def from_ai_gateway(cls, **overrides: Any) -> "LLMClientConfig":
        """
        A config whose :attr:`api_key` is the AI Gateway token fetched from
        Vault at :attr:`~app_config.vault.VaultConfig.resolved_ai_gateway_path`
        (``<app_name>/ai_gateway`` unless ``vault.ai_gateway_path`` says
        otherwise).

        Completes the credential chain: the Entra ID service principal (from
        the Databricks secret scope, or ``AZURE_*``) mints a JWT via ``msal``,
        that JWT logs in to Vault's ``jwt`` auth method, and the secret it
        unlocks is the gateway token set here. Nothing long-lived is stored on
        disk and no key is read from ``config.json``.

        If the secret also carries an endpoint (``base_url`` / ``endpoint`` /
        ``url``), it is used as :attr:`base_url`; likewise a
        ``gateway_version`` / ``ai_gateway_version`` / ``version`` becomes
        :attr:`gateway_version`. Otherwise the configured
        ``llm_client.base_url`` / ``llm_client.gateway_version`` stand.
        This path also turns the proactive
        throttle on by default (:attr:`requests_per_second` = ``2.0``), since
        the gateway enforces a rate limit — unless ``config.json`` or an
        explicit override already set it. Every other knob resolves exactly as
        it does for ``LLMClientConfig()``.

        Parameters
        ----------
        **overrides
            Ordinary constructor arguments, applied last — so an explicit
            ``base_url=`` or ``api_key=`` beats what Vault returned, keeping
            the repo-wide "explicit argument wins" rule intact.

        Raises
        ------
        app_config.vault.VaultError
            Vault is unreachable or unauthenticated, or the secret is missing
            a usable token field.

        Notes
        -----
        This is the one construction path that performs I/O, which is why it
        is an explicit classmethod rather than a default: ``LLMClientConfig()``
        must stay free of network calls.
        """
        # Imported here so importing llm_client never pulls in hvac/msal.
        from app_config import vault

        secret = vault.get_ai_gateway_secret()
        values: dict[str, Any] = {"api_key": SecretStr(vault.ai_gateway_token(secret))}
        base_url = vault.ai_gateway_base_url(secret)
        if base_url is not None:
            values["base_url"] = base_url
        gateway_version = vault.ai_gateway_version(secret)
        if gateway_version is not None:
            values["gateway_version"] = gateway_version
        # The gateway enforces a rate limit, so pace request starts by default
        # on this path — unless the operator already set it via config.json or
        # an explicit override (both of which must keep winning).
        if (
            "requests_per_second" not in overrides
            and app_config.llm_client_value("requests_per_second") is None
        ):
            values["requests_per_second"] = 2.0
        values.update(overrides)
        logger.info(
            f"LLMClientConfig.from_ai_gateway: got the gateway token from Vault "
            f"(base_url={values.get('base_url')}, "
            f"overrides={sorted(overrides) or None})"
        )
        return cls(**values)

    @classmethod
    def from_vault_secret(
        cls, path: str, key: str = "api_key", **overrides: Any
    ) -> "LLMClientConfig":
        """
        A config whose :attr:`api_key` is field *key* of the Vault secret at
        *path*, read over **AppRole** — the application's own identity
        (``VAULT_ROLE_ID`` / ``VAULT_SECRET_ID``) rather than a personal token.

        The escape hatch beside :meth:`from_ai_gateway`, for a deployment that
        files its LLM key somewhere other than the gateway secret. Any ambient
        ``VAULT_TOKEN`` is dropped so this path always exercises AppRole: the
        point of naming it is to use the app identity, and silently falling
        back to whatever token happens to be in the environment would read as
        success while proving nothing.

        Unlike :meth:`from_ai_gateway` this reads only the key — a secret of
        the operator's own choosing carries no ``base_url`` or
        ``gateway_version`` convention — so ``llm_client.base_url`` stands.

        Parameters
        ----------
        path : str
            KV path relative to the mount.
        key : str
            Field within that secret (default ``"api_key"``).
        **overrides
            Ordinary constructor arguments, applied last.

        Raises
        ------
        app_config.vault.VaultError
            No AppRole credentials are configured, Vault is unreachable, or
            the secret has no usable value at *key*.

        Notes
        -----
        Performs I/O, like :meth:`from_ai_gateway` and for the same reason it
        is an explicit classmethod rather than a default.
        """
        from dataclasses import replace as _replace

        from app_config.vault import VaultClient, VaultConfig, VaultError

        base = VaultConfig.from_env()
        if not (base.role_id and base.secret_id):
            raise VaultError(
                f"reading the LLM key from Vault secret {path!r} needs AppRole "
                f"credentials: set VAULT_ROLE_ID and VAULT_SECRET_ID (and "
                f"VAULT_ADDR)"
            )
        approle = _replace(base, token=None)  # force app identity, ignore any token
        value = VaultClient(approle).get_secret(path, key)
        if not isinstance(value, str) or not value:
            raise VaultError(
                f"Vault secret {path!r} field {key!r} is empty or not a string"
            )
        values: dict[str, Any] = {"api_key": SecretStr(value)}
        values.update(overrides)
        logger.info(
            f"LLMClientConfig.from_vault_secret: got the key from Vault secret "
            f"{path!r} (field {key!r}) via AppRole "
            f"(overrides={sorted(overrides) or None})"
        )
        return cls(**values)


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


# Prompt-cache breakpoints are an Anthropic-native content-part key that
# callers (pipeline.engine) set on the system block. Whether it survives the
# trip through an OpenAI-compatible gateway is a property of the *gateway*,
# not the model: LiteLLM- and OpenRouter-style proxies forward it to Anthropic,
# while a strict OpenAI schema rejects unknown content-part keys outright. So
# the support question is settled by asking — send it, and fall back once if
# the gateway says no (see LLMClient._invoke_messages).
_CACHE_CONTROL_KEY = "cache_control"
# Statuses a schema/validation rejection arrives under. 400 is the OpenAI
# shape; 422 is what several compatible proxies return instead.
_SCHEMA_REJECTION_STATUS_CODES = {400, 422}


def _is_cache_control_error(exc: BaseException) -> bool:
    """True when *exc* looks like the gateway rejecting a ``cache_control`` key.

    Deliberately narrow: a schema-rejection status *and* the key named in the
    error text. An unrelated 400 (a bad model id, an over-long prompt) must
    still fail fast — stripping the breakpoint would not help it, and a blind
    retry would double the cost of every malformed request.
    """
    if _status_code(exc) not in _SCHEMA_REJECTION_STATUS_CODES:
        return False
    return _CACHE_CONTROL_KEY in str(exc).lower()


def _strip_cache_control(messages: list[BaseMessage]) -> list[BaseMessage] | None:
    """*messages* with every ``cache_control`` breakpoint removed, or ``None``
    when there were none to remove.

    Only list-of-parts content can carry one. Messages are copied rather than
    mutated: the caller may hold the originals (a prompt template's cached
    output, say), and a retry must not edit them in place.
    """
    stripped: list[BaseMessage] = []
    found = False
    for message in messages:
        content = message.content
        if not isinstance(content, list):
            stripped.append(message)
            continue
        parts: list[Any] = []
        edited = False
        for part in content:
            if isinstance(part, dict) and _CACHE_CONTROL_KEY in part:
                part = {k: v for k, v in part.items() if k != _CACHE_CONTROL_KEY}
                edited = True
            parts.append(part)
        if not edited:
            stripped.append(message)
            continue
        found = True
        stripped.append(message.model_copy(update={"content": parts}))
    return stripped if found else None


def _response_headers(exc: BaseException) -> Any:
    """The HTTP response headers carried by a provider error, or ``None``.

    Provider SDKs (Anthropic/OpenAI httpx-based) hang the headers off
    ``exc.response.headers``; some also expose ``exc.headers`` directly.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers is None:
        headers = getattr(exc, "headers", None)
    return headers


def _header_lookup(headers: Any, name: str) -> Any:
    """Case-insensitive header read that works for httpx.Headers or a dict.

    Returns the raw header value (str for real HTTP headers, but ``Any`` since
    a dict may hold anything); the caller coerces it.
    """
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if callable(getter):
        # httpx.Headers.get is already case-insensitive; a plain dict.get is
        # exact, so a wrong-cased key falls through to the scan below.
        try:
            value = getter(name)
        except Exception:
            value = None
        if value is not None:
            return value
    try:
        low = name.lower()
        for key, value in headers.items():
            if isinstance(key, str) and key.lower() == low:
                return value
    except Exception:
        pass
    return None


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Seconds the gateway asked the caller to wait, or ``None``.

    Reads the standard rate-limit timing headers a 429 (or 503) response
    carries: ``retry-after-ms`` (milliseconds, checked first), then
    ``Retry-After`` as either a number of seconds or an HTTP-date. Returns
    ``None`` when no usable header is present, so the caller falls back to its
    own exponential backoff.
    """
    headers = _response_headers(exc)
    if headers is None:
        return None
    ms = _header_lookup(headers, "retry-after-ms")
    if ms is not None:
        try:
            return max(0.0, float(ms) / 1000.0)
        except (TypeError, ValueError):
            pass
    raw = _header_lookup(headers, "retry-after")
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))  # delta-seconds form
    except ValueError:
        pass
    try:  # HTTP-date form
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    now = datetime.now(when.tzinfo) if when.tzinfo else datetime.now()
    return max(0.0, (when - now).total_seconds())


class TokenUsage(BaseModel):
    """Tokens billed for one call, or accumulated across a client's calls.

    Populated from the response's ``usage_metadata`` when the gateway returns
    a usage block, which OpenAI-compatible ``/chat/completions`` responses do
    by default. Cache fields stay ``0`` on gateways that report no cache
    detail, so a zero there means "not reported", not "nothing was cached".
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    calls: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_creation_tokens=(
                self.cache_creation_tokens + other.cache_creation_tokens
            ),
            calls=self.calls + other.calls,
        )

    def __sub__(self, other: "TokenUsage") -> "TokenUsage":
        """Spend between two snapshots of the same client's running total.

        For attributing one run's cost when the client outlives it — take a
        snapshot before, subtract it after. Fields clamp at zero, so a
        subtraction of unrelated totals reads as "nothing" instead of going
        negative.
        """
        return TokenUsage(
            input_tokens=max(0, self.input_tokens - other.input_tokens),
            output_tokens=max(0, self.output_tokens - other.output_tokens),
            total_tokens=max(0, self.total_tokens - other.total_tokens),
            cache_read_tokens=max(0, self.cache_read_tokens - other.cache_read_tokens),
            cache_creation_tokens=max(
                0, self.cache_creation_tokens - other.cache_creation_tokens
            ),
            calls=max(0, self.calls - other.calls),
        )

    def summary(self) -> str:
        """One-line human rendering, e.g. for a log or a report header."""
        text = (
            f"{self.calls} call(s)  in={self.input_tokens:,}  "
            f"out={self.output_tokens:,}  total={self.total_tokens:,}"
        )
        if self.cache_read_tokens or self.cache_creation_tokens:
            text += (
                f"  cache_read={self.cache_read_tokens:,}  "
                f"cache_write={self.cache_creation_tokens:,}"
            )
        return text


def _int_field(source: Any, key: str) -> int:
    """``source[key]`` as a non-negative int, or 0 — usage blocks vary by
    gateway, and a missing or oddly-typed count must never fail a call that
    already succeeded."""
    if not isinstance(source, dict):
        return 0
    value = source.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def _token_usage(response: Any) -> TokenUsage | None:
    """Token counts carried by *response*, or ``None`` when it reports none.

    Reads LangChain's normalized ``usage_metadata`` first, then falls back to
    the raw ``response_metadata['token_usage']`` block that a gateway may
    return without LangChain recognizing it.
    """
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict) and usage:
        details = usage.get("input_token_details")
        return TokenUsage(
            input_tokens=_int_field(usage, "input_tokens"),
            output_tokens=_int_field(usage, "output_tokens"),
            total_tokens=_int_field(usage, "total_tokens"),
            cache_read_tokens=_int_field(details, "cache_read"),
            cache_creation_tokens=_int_field(details, "cache_creation"),
            calls=1,
        )
    raw = getattr(response, "response_metadata", None)
    raw = raw.get("token_usage") if isinstance(raw, dict) else None
    if not isinstance(raw, dict) or not raw:
        return None
    # OpenAI's own field names, for a gateway whose usage block LangChain
    # did not normalize into usage_metadata.
    return TokenUsage(
        input_tokens=_int_field(raw, "prompt_tokens"),
        output_tokens=_int_field(raw, "completion_tokens"),
        total_tokens=_int_field(raw, "total_tokens"),
        cache_read_tokens=_int_field(raw.get("prompt_tokens_details"), "cached_tokens"),
        calls=1,
    )


def _split_model(model: str) -> tuple[str | None, str]:
    """A ``"provider: model"`` string as ``(provider, model_id)``.

    Model strings may carry a provider prefix — a LangChain
    ``init_chat_model`` spelling (``"anthropic:claude-opus-4-6"``), or the
    reference gateway's own, read from a SharePoint ``Model`` column with a
    space after the colon. Either way the prefix names the *family*, which is
    what a deployment pins ``provider_client`` on, while the gateway itself
    routes on the model id alone — so the two are separated here rather than
    one being thrown away.

    The id is lowercased and both halves stripped, matching the reference's
    ``parse_preferred_llm``. An unprefixed string yields ``(None, model)``
    unchanged in case, since that is the id an operator wrote by hand.
    """
    prefix, sep, bare = model.partition(":")
    if not sep or not bare.strip():
        return None, model
    return prefix.strip(), bare.strip().lower()


# LangChain message types -> the OpenAI chat roles the wire format uses.
_OPENAI_ROLES = {"system": "system", "ai": "assistant", "human": "user"}


def _message_text(content: Any) -> str:
    """A message's content as plain text.

    Multi-part content (the list form a ``cache_control`` breakpoint rides on)
    is flattened to its text blocks: the native path speaks the plain
    ``{"role", "content"}`` shape, so anything richer has to be reduced to it.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "".join(parts)


def _to_openai_message(message: BaseMessage) -> dict[str, str]:
    """One LangChain message as an OpenAI ``{"role", "content"}`` dict.

    Unknown types map to ``user``, which is the safe default: a message the
    gateway cannot place is still the caller talking.
    """
    return {
        "role": _OPENAI_ROLES.get(message.type, "user"),
        "content": _message_text(message.content),
    }


def _native_usage(response: Any) -> dict[str, int] | None:
    """A raw OpenAI response's usage block in LangChain's normalized shape,
    so token accounting works the same on the native path as on the LangChain
    one. ``None`` when the gateway reported no usage."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens = _int_field({"v": getattr(usage, "prompt_tokens", 0)}, "v")
    output_tokens = _int_field({"v": getattr(usage, "completion_tokens", 0)}, "v")
    total = _int_field({"v": getattr(usage, "total_tokens", 0)}, "v")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total or input_tokens + output_tokens,
    }


def _as_messages(input: LanguageModelInput) -> list[BaseMessage]:
    if isinstance(input, PromptValue):
        return input.to_messages()
    if isinstance(input, str):
        return [HumanMessage(input)]
    return convert_to_messages(input)


class LLMClient:
    """
    Thin invocation layer around an OpenAI-compatible chat model.

    Use :meth:`invoke` directly, or :meth:`as_runnable` to drop the client
    into a LCEL chain in place of the raw model
    (``prompt | client.as_runnable()``).

    Parameters
    ----------
    config : LLMClientConfig | None
        Knobs; defaults to ``LLMClientConfig()``.
    llm : Any | None
        Pre-built chat model to use instead of constructing a
        ``ChatOpenAI`` from ``config.model`` (e.g. a fake in tests, or a
        provider-native client for an endpoint the gateway does not
        front). Retry and the
        input-token budget still apply to an injected model;
        ``temperature`` / ``max_output_tokens`` / ``requests_per_second``
        do not, since those attach at construction time.

    Attributes
    ----------
    usage : TokenUsage
        Tokens billed across this client's calls so far, accumulated from
        each response's usage block and logged per call at INFO. Stays at
        zero when the gateway reports no usage (logged once at WARNING).
        Plain attribute arithmetic, not synchronized — a client shared
        across threads can undercount.
    """

    def __init__(
        self, config: LLMClientConfig | None = None, *, llm: Any | None = None
    ) -> None:
        self.config = config or LLMClientConfig()
        self._warned_no_usage = False
        # Tri-state: None = not yet known, True = a cache hit was reported,
        # False = the gateway rejected the breakpoint (strip it from now on).
        self._cache_control_supported: bool | None = None
        self.usage = TokenUsage()
        self._model = llm if llm is not None else self._build_model(self.config)
        # schema name -> model bound to it; see structured_model().
        self._structured_models: dict[str, Any] = {}

    @property
    def chat_model(self) -> Any:
        """The underlying LangChain chat model (a ``ChatOpenAI`` unless one
        was injected)."""
        return self._model

    @staticmethod
    def _apply_cert_file(cert_file: str) -> str | None:
        """Export *cert_file* as ``SSL_CERT_FILE`` so the ``openai`` SDK's
        httpx client verifies TLS against it, and return the resolved path
        (``None`` when the file is missing).

        Process-wide on purpose: the export covers any code in the process
        reaching the gateway, not just this model. The returned path lets the
        caller *also* pin an explicit httpx client to the same bundle."""
        path = Path(cert_file)
        if not path.is_file():
            logger.warning(
                f"_apply_cert_file: cert_file '{cert_file}' does not exist; "
                f"ignoring it (the default TLS trust store applies)"
            )
            return None
        resolved = str(path.resolve())
        previous = os.environ.get("SSL_CERT_FILE")
        if previous is not None and previous != resolved:
            logger.warning(
                f"_apply_cert_file: overriding SSL_CERT_FILE='{previous}' "
                f"with configured cert_file '{resolved}'"
            )
        os.environ["SSL_CERT_FILE"] = resolved
        return resolved

    @staticmethod
    def _resolve_provider_client(
        provider_client: str | None, provider: str | None
    ) -> str:
        """
        Which construction strategy to use: an explicit *provider_client* wins,
        otherwise *provider* decides through
        :data:`app_config.PROVIDER_CLIENT_BY_PROVIDER`.

        With no provider to route on — a bare model id and no
        ``model_provider`` — the OpenAI-compatible path applies, which is what
        every deployment that never sets a provider has always got.

        Raises
        ------
        LLMClientError
            The provider is named but unknown. The reference raises
            ``"UNKNOWN model provider."`` here for the same reason: guessing a
            client for an unrecognised provider turns a one-line config typo
            into a failure deep inside the gateway.
        """
        if provider_client and provider_client != app_config.PROVIDER_CLIENT_AUTO:
            return provider_client
        if not provider:
            return app_config.PROVIDER_CLIENT_OPENAI_COMPATIBLE
        key = provider.strip().lower()
        resolved = app_config.PROVIDER_CLIENT_BY_PROVIDER.get(key)
        if resolved is None:
            known = "/".join(sorted(app_config.PROVIDER_CLIENT_BY_PROVIDER))
            raise LLMClientError(
                f"unknown model provider {provider!r}: it is not one of {known}, "
                f"so there is no client to build for it. Name a known provider "
                f"in llm_client.model_provider (or as a 'provider:model' prefix), "
                f"or pin llm_client.provider_client to "
                f"'{app_config.PROVIDER_CLIENT_OPENAI_COMPATIBLE}' or "
                f"'{app_config.PROVIDER_CLIENT_NATIVE}' outright"
            )
        logger.info(
            f"_resolve_provider_client: provider {key!r} -> {resolved!r} "
            f"(llm_client.provider_client is "
            f"{provider_client or app_config.PROVIDER_CLIENT_AUTO!r})"
        )
        return resolved

    @staticmethod
    def _build_model(config: LLMClientConfig) -> Any:
        provider, model = _split_model(config.model)
        provider = config.model_provider or provider
        strategy = LLMClient._resolve_provider_client(
            config.provider_client, provider
        )
        if strategy == app_config.PROVIDER_CLIENT_NATIVE:
            return LLMClient._build_native_model(config, provider, model)
        cert_path = (
            LLMClient._apply_cert_file(config.cert_file)
            if config.cert_file is not None
            else None
        )
        kwargs: dict[str, Any] = {
            # The SDK's own retry layer would re-send a 429 underneath us, out
            # of sight of the Retry-After handling in _retry_delay; this class
            # owns retries, so switch it off.
            "max_retries": 0,
            # An OpenAI-*compatible* gateway generally implements
            # /chat/completions only, so never let langchain-openai infer its
            # way to the Responses API from the invocation params.
            "use_responses_api": False,
        }  # both overridable through `kwargs` below
        if cert_path is not None:
            # Belt and braces with the SSL_CERT_FILE export: explicit httpx
            # clients pin verification for this model even if something else in
            # the process rewrites the environment variable. Both sync and
            # async are set — otherwise `ainvoke` would fall back to the
            # default trust store and fail against an internal CA. Only reached
            # when the bundle exists (_apply_cert_file returned a path), so
            # httpx is not handed a missing file.
            try:
                import ssl

                import httpx

                # An explicit SSLContext, not verify=<path>: the string form is
                # deprecated in modern httpx, and one context serves both
                # clients.
                ssl_context = ssl.create_default_context(cafile=cert_path)
                kwargs["http_client"] = httpx.Client(verify=ssl_context)
                kwargs["http_async_client"] = httpx.AsyncClient(verify=ssl_context)
            except Exception as exc:
                logger.warning(
                    f"_build_model: could not build an httpx client verifying "
                    f"against '{cert_path}' ({exc!r}); falling back to the "
                    f"SSL_CERT_FILE export alone"
                )
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if config.max_output_tokens is not None:
            kwargs["max_tokens"] = config.max_output_tokens
        base_url = config.base_url
        if base_url is not None:
            # A trailing slash marks a per-deployment route: the gateway wants
            # the model id as the last path segment rather than in the body
            # alone. Without one the URL is taken as the complete API root.
            if base_url.endswith("/"):
                base_url += model
            kwargs["base_url"] = base_url
        headers = dict(config.url_headers) if config.url_headers else {}
        if config.api_key is not None:
            key = config.api_key.get_secret_value()
            kwargs["api_key"] = key
            # Gateways fronting Azure OpenAI authenticate on an `api-key`
            # header rather than the Bearer the SDK sends, so the credential
            # goes out both ways. setdefault, not assignment: an operator who
            # set this header explicitly meant it.
            headers.setdefault("api-key", key)
        if config.gateway_version:
            # The gateway rejects (or silently mis-routes) a request that does
            # not declare which protocol version it speaks. setdefault, the
            # same rule as api-key above: an operator who set this header
            # through url_headers meant it.
            headers.setdefault("ai-gateway-version", config.gateway_version)
        if headers:
            kwargs["default_headers"] = headers
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
            f"LLMClient: building ChatOpenAI model '{model}'  "
            f"provider={provider}  "
            f"gateway_version={'set' if config.gateway_version else 'unset'}  "
            f"temperature={config.temperature}  "
            f"max_output_tokens={config.max_output_tokens}  "
            f"base_url={base_url}  "
            f"timeout={config.timeout}  "
            f"cert_file={config.cert_file}  "
            f"token_encoding={tokens.encoding_name_for_model(model)}  "
            f"api_key={'set' if config.api_key else 'unset'}  "
            f"url_headers={sorted(config.url_headers) if config.url_headers else None}  "
            f"model_kwargs={config.model_kwargs}  "
            f"extra_kwargs={sorted(config.kwargs) if config.kwargs else None}  "
            f"requests_per_second={config.requests_per_second}  "
            f"max_retries={config.max_retries}"
        )
        return ChatOpenAI(model=model, **kwargs)

    @staticmethod
    def _build_native_model(
        config: LLMClientConfig, provider: str | None, model: str
    ) -> Any:
        """A raw ``openai.OpenAI`` client wrapped to the surface
        :class:`LLMClient` invokes, for ``provider_client="native"``.

        The escape hatch for a gateway that will not take the payload
        ``ChatOpenAI`` builds. It is a ``RunnableLambda`` over one
        ``chat.completions.create`` call rather than a second model class, so
        nothing here has to be kept in step with the LangChain integration.

        What it costs, all of which attaches at construction time and so
        cannot survive the swap: the rate limiter, ``model_kwargs``, and
        structured output (``with_structured_output``, which
        :meth:`supports_structured_output` will now answer ``False`` for).
        Retry, the input-token budget and usage accounting are unaffected —
        they live in :class:`LLMClient`, above the model.
        """
        forfeited = [
            name
            for name, value in (
                ("requests_per_second", config.requests_per_second),
                ("model_kwargs", config.model_kwargs),
            )
            if value
        ]
        if forfeited:
            logger.warning(
                f"_build_native_model: provider_client='native' cannot apply "
                f"{', '.join(forfeited)} — they attach when a LangChain chat "
                f"model is constructed, and this path builds none. Structured "
                f"output is unavailable on it too. Set "
                f"llm_client.provider_client='openai_compatible' to go back to "
                f"the LangChain transport (this run reached the native one "
                f"because provider {provider!r} routes here)"
            )
        if config.cert_file is not None:
            LLMClient._apply_cert_file(config.cert_file)
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - openai is a hard dep
            raise RuntimeError(
                "provider_client='native' needs the openai SDK installed"
            ) from exc
        kwargs: dict[str, Any] = {"max_retries": 0}  # this class owns retries
        base_url = config.base_url
        if base_url is not None:
            if base_url.endswith("/"):
                base_url += model  # per-deployment route; see _build_model
            kwargs["base_url"] = base_url
        headers: dict[str, str] = dict(config.url_headers) if config.url_headers else {}
        if config.api_key is not None:
            key = config.api_key.get_secret_value()
            kwargs["api_key"] = key
            headers.setdefault("api-key", key)
        if config.gateway_version:
            headers.setdefault("ai-gateway-version", config.gateway_version)
        if headers:
            kwargs["default_headers"] = headers
        if config.timeout is not None:
            kwargs["timeout"] = config.timeout
        client = openai.OpenAI(**kwargs)

        body: dict[str, Any] = {"model": model}
        if config.max_output_tokens is not None:
            body["max_tokens"] = config.max_output_tokens
        if config.temperature is not None:
            body["temperature"] = config.temperature

        def _call(messages: list[BaseMessage]) -> BaseMessage:
            response = client.chat.completions.create(
                messages=[_to_openai_message(m) for m in messages],  # type: ignore[arg-type]
                **body,
            )
            return AIMessage(
                content=response.choices[0].message.content or "",
                # Carried through so LLMClient.usage keeps accumulating on this
                # path; _token_usage reads exactly these three names.
                usage_metadata=_native_usage(response),
            )

        # Header VALUES may carry gateway credentials — log key names only.
        logger.info(
            f"LLMClient: building native openai client for model '{model}'  "
            f"provider={provider}  "
            f"gateway_version={'set' if config.gateway_version else 'unset'}  "
            f"temperature={config.temperature}  "
            f"max_output_tokens={config.max_output_tokens}  "
            f"base_url={base_url}  "
            f"timeout={config.timeout}  "
            f"cert_file={config.cert_file}  "
            f"api_key={'set' if config.api_key else 'unset'}  "
            f"url_headers={sorted(headers) or None}  "
            f"max_retries={config.max_retries}"
        )
        return RunnableLambda(_call)

    # ------------------------------------------------------------------
    # Input-token budget
    # ------------------------------------------------------------------

    def count_tokens(self, messages: list[BaseMessage]) -> int:
        """Count prompt tokens with the configured counter, else the shared
        tiktoken counter (:mod:`llm_client.tokens`) under this model's
        encoding — client-owned, so an injected chat model needs no
        tokenizer of its own. Degrades to chars//4 when tiktoken cannot
        load its encoding data (one-time WARNING, in ``llm_client.tokens``).
        """
        if self.config.token_counter is not None:
            return self.config.token_counter(messages)
        return tokens.count_messages(messages, model=self.config.model)

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
    # Prompt-cache compatibility and token accounting
    # ------------------------------------------------------------------

    def _prepare(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """*messages* as they should go out, given what this gateway supports.

        Once a gateway has rejected a ``cache_control`` breakpoint, every later
        call strips it up front rather than spending a failed round trip
        rediscovering the same answer.
        """
        if self._cache_control_supported is not False:
            return messages
        return _strip_cache_control(messages) or messages

    def _demote_cache_control(
        self, messages: list[BaseMessage], exc: Exception, op: str
    ) -> list[BaseMessage] | None:
        """Messages to re-send without their cache breakpoints, or ``None``
        when *exc* is not the gateway refusing them.

        Records the answer on the client, so the fallback costs one failed
        request per process rather than one per call.
        """
        if not _is_cache_control_error(exc):
            return None
        retry = _strip_cache_control(messages)
        if retry is None:
            return None  # nothing to strip; the key was not the real problem
        self._cache_control_supported = False
        logger.warning(
            f"{op}: gateway rejected the prompt-cache breakpoint "
            f"(cache_control) with {_status_code(exc)}; re-sending without it "
            f"and dropping it from later calls — prompt caching is off for "
            f"this endpoint: {exc}"
        )
        return retry

    def _record_usage(self, response: Any, op: str) -> None:
        """Log this call's token usage and fold it into the running total."""
        # A structured call with include_raw=True answers
        # {"raw", "parsed", "parsing_error"}; the usage block rides on "raw".
        if isinstance(response, dict) and "raw" in response:
            response = response["raw"]
        usage = _token_usage(response)
        if usage is None:
            if not self._warned_no_usage:
                logger.warning(
                    f"{op}: response carried no token usage; the gateway is "
                    f"not reporting one, so LLMClient.usage stays empty"
                )
                self._warned_no_usage = True
            return
        self.usage = self.usage + usage
        if self._cache_control_supported is None and usage.cache_read_tokens:
            # A cache hit is the gateway telling us the breakpoint survived.
            self._cache_control_supported = True
        logger.info(
            f"{op}: tokens in={usage.input_tokens} out={usage.output_tokens} "
            f"total={usage.total_tokens} "
            f"cache_read={usage.cache_read_tokens} "
            f"cache_write={usage.cache_creation_tokens}  "
            f"(run total in={self.usage.input_tokens} "
            f"out={self.usage.output_tokens} over {self.usage.calls} call(s))"
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
        # Honor the gateway's own timing when it sends one: Retry-After /
        # retry-after-ms is authoritative, so it replaces the backoff (no
        # jitter), bounded only by a safety ceiling so a bad header can't hang.
        server_delay = _retry_after_seconds(exc)
        if server_delay is not None:
            delay = min(server_delay, self.config.retry_after_max_seconds)
            logger.warning(
                f"{op}: gateway asked to retry after {server_delay:.2f}s "
                f"(attempt {attempt}/{attempts}); honoring, waiting {delay:.2f}s "
                f"(capped at {self.config.retry_after_max_seconds:.0f}s): {exc}"
            )
            return delay
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

        A ``cache_control`` breakpoint the gateway will not accept is
        dropped and the call re-sent once (see :attr:`usage` and
        :meth:`_demote_cache_control`); token usage is logged and
        accumulated on every success.
        """
        return self._invoke_with(self._model, input, config, op="invoke")

    def _invoke_with(
        self,
        model: Any,
        input: LanguageModelInput,
        config: RunnableConfig | None,
        *,
        op: str,
    ) -> Any:
        """:meth:`invoke`'s budget / retry / usage machinery, over any *model*.

        Factored out so the structured-output path
        (:meth:`invoke_structured`) is the same call with a schema-bound model
        rather than a parallel implementation that would drift.
        """
        messages = self._prepare(_as_messages(input))
        self._enforce_input_limit(messages)

        attempts = self.config.max_retries + 1
        attempt = 0
        # `while True` rather than a bounded range: the loop exits through
        # _retry_delay, which returns None once the budget is spent (so the
        # exception propagates). That lets the cache_control probe below
        # re-send without spending an attempt — the fallback must work even
        # at max_retries=0, where there is no budget to borrow from.
        while True:
            attempt += 1
            try:
                response = model.invoke(messages, config)
            except Exception as exc:
                retry_messages = self._demote_cache_control(messages, exc, op)
                if retry_messages is not None:
                    # A schema rejection, not a transient failure: re-send at
                    # once, no backoff, no attempt spent. Bounded to one pass —
                    # the demotion is recorded on the client, so a second
                    # rejection would find nothing left to strip.
                    messages = retry_messages
                    attempt -= 1
                    continue
                delay = self._retry_delay(exc, attempt, attempts, op)
                if delay is None:
                    raise
                time.sleep(delay)
            else:
                self._record_usage(response, op)
                return response

    async def ainvoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None
    ) -> BaseMessage:
        """
        Async :meth:`invoke`: same input-token budget, transient-error
        retry, ``cache_control`` fallback and usage accounting, with
        non-blocking backoff. Token counting runs in a worker thread
        because model-native counters may call a synchronous HTTP
        endpoint.
        """
        messages = self._prepare(_as_messages(input))
        await asyncio.to_thread(self._enforce_input_limit, messages)

        attempts = self.config.max_retries + 1
        attempt = 0
        while True:  # exits via _retry_delay; see the note in invoke()
            attempt += 1
            try:
                response = await self._model.ainvoke(messages, config)
            except Exception as exc:
                retry_messages = self._demote_cache_control(messages, exc, "ainvoke")
                if retry_messages is not None:
                    messages = retry_messages
                    attempt -= 1
                    continue
                delay = self._retry_delay(exc, attempt, attempts, "ainvoke")
                if delay is None:
                    raise
                await asyncio.sleep(delay)
            else:
                self._record_usage(response, "ainvoke")
                return response

    def as_runnable(self) -> Runnable[LanguageModelInput, BaseMessage]:
        """The client as a LCEL Runnable, e.g. ``prompt | client.as_runnable()``.

        Bound to both paths: ``chain.invoke`` uses :meth:`invoke` and
        ``chain.ainvoke`` uses :meth:`ainvoke`.
        """
        return RunnableLambda(self.invoke, afunc=self.ainvoke, name="llm_client")

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

    def supports_structured_output(self, schema: Any) -> bool:
        """Whether this chat model can be asked for *schema* at all.

        Answers the binding question only — whether ``with_structured_output``
        exists and accepts the schema — not whether the gateway behind it will
        honour the request; that shows up per call as a ``parsing_error`` (or,
        at worst, a rejected request the caller demotes). Injected stubs and
        integrations without the method answer ``False`` here rather than
        raising mid-run.
        """
        try:
            return self.structured_model(schema) is not None
        except NotImplementedError:
            logger.info(
                f"supports_structured_output: model {self.config.model!r} does "
                "not implement with_structured_output"
            )
            return False
        except Exception as exc:
            logger.warning(
                f"supports_structured_output: binding the schema to model "
                f"{self.config.model!r} failed: {exc!r}"
            )
            return False

    def structured_model(self, schema: Any) -> Any:
        """The chat model bound to *schema*, built once per schema and cached.

        Always ``include_raw=True``: the caller needs the underlying
        ``AIMessage`` for token usage and response metadata, and needs a
        failure to arrive as ``parsing_error`` rather than an exception so it
        can fall back to reading the model's prose.
        """
        key = getattr(schema, "__name__", repr(schema))
        cached = self._structured_models.get(key)
        if cached is None:
            logger.debug(f"structured_model: binding schema {key!r}")
            cached = self._model.with_structured_output(schema, include_raw=True)
            self._structured_models[key] = cached
        return cached

    def invoke_structured(
        self,
        schema: Any,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """Invoke the model asking for *schema*, with the same guarantees as
        :meth:`invoke` (input budget, retries, ``cache_control`` fallback,
        usage accounting).

        Returns LangChain's ``include_raw`` envelope:
        ``{"raw": AIMessage, "parsed": schema | None, "parsing_error": ...}``.
        A model or gateway that cannot honour the schema surfaces here as
        ``parsed=None`` with a ``parsing_error``, which is the caller's cue to
        fall back — not an exception, because a usable prose answer is still in
        ``raw``.
        """
        result = self._invoke_with(
            self.structured_model(schema), input, config, op="invoke_structured"
        )
        if not isinstance(result, dict):
            # A model injected by a test (or a future LangChain shape) that
            # answers with the parsed object alone; normalize to the envelope.
            return {"raw": None, "parsed": result, "parsing_error": None}
        return result

    def as_structured_runnable(self, schema: Any) -> Runnable[LanguageModelInput, Any]:
        """:meth:`invoke_structured` as an LCEL Runnable bound to *schema*,
        e.g. ``prompt | client.as_structured_runnable(TranslationDocument)``."""
        return RunnableLambda(
            lambda value, config=None: self.invoke_structured(schema, value, config),
            name="llm_client_structured",
        )
