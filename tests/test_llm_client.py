"""
Tests for llm_client.client — construction knobs, input-token budget,
and rate-limit retry.

No network or API keys: model construction is exercised by monkeypatching
``llm_client.client.ChatOpenAI`` to capture kwargs, and invocation by
injecting fakes (FakeListChatModel or small stubs). Retry tests set
``retry_base_seconds`` tiny so backoff sleeps are negligible.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import sys
import time
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

import llm_client.client as client_mod
from llm_client import InputTokenLimitError, LLMClient, LLMClientConfig, TokenUsage


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeRateLimitError(Exception):
    """Mimics a provider rate-limit error via its status_code attribute."""

    status_code = 429


class _FlakyModel:
    """Chat-model stub that fails ``failures`` times before succeeding."""

    def __init__(self, failures: int, exc: Exception | None = None) -> None:
        self.failures = failures
        self.exc = exc or _FakeRateLimitError("throttled")
        self.calls = 0

    def invoke(self, messages, config=None):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return AIMessage("ok")

    async def ainvoke(self, messages, config=None):
        return self.invoke(messages, config)


def _fast_retry_config(**overrides) -> LLMClientConfig:
    return LLMClientConfig(
        retry_base_seconds=0.001, retry_max_seconds=0.002, **overrides
    )


# ---------------------------------------------------------------------------
# Construction — temperature / max_tokens / rate limiter forwarding
# ---------------------------------------------------------------------------


def _capture_init(monkeypatch):
    captured: dict = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return FakeListChatModel(responses=["built"])

    monkeypatch.setattr(client_mod, "ChatOpenAI", fake_chat_openai)
    return captured


def test_temperature_and_max_tokens_forwarded(monkeypatch):
    captured = _capture_init(monkeypatch)
    LLMClient(
        LLMClientConfig(model="some-model", temperature=0.2, max_output_tokens=512)
    )

    assert captured["model"] == "some-model"
    assert captured["temperature"] == 0.2
    assert captured["max_tokens"] == 512


def test_provider_defaults_left_alone_when_unset(monkeypatch):
    captured = _capture_init(monkeypatch)
    LLMClient(LLMClientConfig(model="some-model"))

    assert "temperature" not in captured
    assert "max_tokens" not in captured
    # Proactive throttle is off by default — only the gateway path turns it on.
    assert "rate_limiter" not in captured


def test_rate_limiter_disabled_when_requests_per_second_none(monkeypatch):
    captured = _capture_init(monkeypatch)
    LLMClient(LLMClientConfig(model="some-model", requests_per_second=None))

    assert "rate_limiter" not in captured


def test_requests_per_second_builds_rate_limiter(monkeypatch):
    captured = _capture_init(monkeypatch)
    LLMClient(
        LLMClientConfig(model="some-model", requests_per_second=2.0, max_bucket_size=4)
    )

    limiter = captured["rate_limiter"]
    assert isinstance(limiter, client_mod.InMemoryRateLimiter)
    assert limiter.requests_per_second == 2.0
    assert limiter.max_bucket_size == 4


@pytest.mark.parametrize(
    "configured, sent",
    [
        # Every model goes out over the same OpenAI-compatible transport, so
        # the name is passed through untouched...
        ("claude-sonnet-4-5", "claude-sonnet-4-5"),
        ("gpt-5.4", "gpt-5.4"),
        ("gemini-3.1-pro", "gemini-3.1-pro"),
        ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5-20250929"),
        # ...except for a LangChain provider prefix, which selected a client
        # class and would read as an unknown model id to the gateway.
        ("anthropic:claude-opus-4-6", "claude-opus-4-6"),
        ("openai:gpt-5.4", "gpt-5.4"),
    ],
)
def test_model_name_sent_to_the_gateway(monkeypatch, configured, sent):
    captured = _capture_init(monkeypatch)
    LLMClient(LLMClientConfig(model=configured))

    assert captured["model"] == sent


def test_sdk_retry_layer_is_disabled(monkeypatch):
    # The client's own loop owns retries; the openai SDK must not re-send a
    # 429 underneath it, out of sight of the Retry-After handling.
    captured = _capture_init(monkeypatch)
    LLMClient(LLMClientConfig(model="some-model", max_retries=5))

    assert captured["max_retries"] == 0


def test_responses_api_is_pinned_off(monkeypatch):
    # A compatible gateway generally implements /chat/completions only, so
    # langchain-openai must never infer its way to the Responses API.
    captured = _capture_init(monkeypatch)
    LLMClient(LLMClientConfig(model="some-model"))

    assert captured["use_responses_api"] is False


@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-4o", "claude-sonnet-4-5"])
def test_no_tiktoken_model_name_stand_in_is_passed(monkeypatch, model):
    # Counting is client-owned (llm_client.tokens) since the model-native
    # counter mis-resolved names like "gpt-5.4"; the stand-in kwarg is gone.
    captured = _capture_init(monkeypatch)
    LLMClient(LLMClientConfig(model=model))

    assert "tiktoken_model_name" not in captured


def _o200k_base_available() -> bool:
    """Whether tiktoken can load its encoding data here (cache or network)."""
    try:
        import tiktoken

        tiktoken.get_encoding("o200k_base")
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _o200k_base_available(),
    reason="tiktoken o200k_base encoding data unavailable (offline sandbox)",
)
def test_non_gpt_model_counts_tokens_without_the_chars_fallback(caplog):
    # Regression guard: a client named after a Claude model must produce a
    # tokenizer-backed count (o200k_base stand-in vocabulary), not trip the
    # approximation warning. Skips itself where the encoding data cannot be
    # loaded at all — CI, with network, always runs it.
    client = LLMClient(
        LLMClientConfig(
            model="claude-sonnet-4-5",
            api_key="sk-test",  # pyright: ignore[reportArgumentType]
            base_url="https://gateway.example/v1",
        )
    )

    with caplog.at_level(logging.WARNING, logger="llm_client.client"):
        tokens = client.count_tokens([HumanMessage("hello world " * 10)])

    assert tokens > 0
    assert not any("chars//4" in record.message for record in caplog.records)


def test_kwargs_can_reenable_sdk_retries(monkeypatch):
    captured = _capture_init(monkeypatch)
    LLMClient(
        LLMClientConfig(
            model="some-model", kwargs={"max_retries": 2, "use_responses_api": True}
        )
    )

    assert captured["max_retries"] == 2
    assert captured["use_responses_api"] is True


def test_model_name_alias_accepted():
    # model_name is a validation_alias for `model` (populate_by_name); pyright
    # doesn't synthesize it as a constructor kwarg, but pydantic accepts it.
    assert LLMClientConfig(model_name="alias-model").model == "alias-model"  # pyright: ignore[reportCallIssue]
    assert LLMClientConfig(model="plain-model").model == "plain-model"


def test_endpoint_overrides_forwarded(monkeypatch):
    captured = _capture_init(monkeypatch)
    LLMClient(
        LLMClientConfig(
            model="some-model",
            base_url="https://gateway.example/v1",
            # pydantic coerces a plain str into the SecretStr field at runtime.
            api_key="sk-secret",  # pyright: ignore[reportArgumentType]
            url_headers={"X-Team": "sas"},
            timeout=42.5,
            model_kwargs={"top_k": 40},
        )
    )

    assert captured["base_url"] == "https://gateway.example/v1"
    assert captured["api_key"] == "sk-secret"
    # The key also rides as an `api-key` header — gateways fronting Azure
    # OpenAI authenticate on that rather than the Bearer the SDK sends.
    assert captured["default_headers"] == {"X-Team": "sas", "api-key": "sk-secret"}
    assert captured["timeout"] == 42.5
    assert captured["model_kwargs"] == {"top_k": 40}


def test_api_key_header_does_not_override_an_explicit_one(monkeypatch):
    # An operator who set the header themselves means it: setdefault, not set.
    captured = _capture_init(monkeypatch)
    LLMClient(
        LLMClientConfig(
            model="some-model",
            api_key="sk-secret",  # pyright: ignore[reportArgumentType]
            url_headers={"api-key": "explicit-header-value"},
        )
    )

    assert captured["default_headers"]["api-key"] == "explicit-header-value"


def test_no_api_key_header_without_an_api_key(monkeypatch):
    captured = _capture_init(monkeypatch)
    LLMClient(LLMClientConfig(model="some-model", url_headers={"X-Team": "sas"}))

    assert captured["default_headers"] == {"X-Team": "sas"}


def test_api_key_value_is_never_logged(monkeypatch, caplog):
    # The key now appears in two places (api_key= and a header); neither may
    # reach the log, which records header names only.
    _capture_init(monkeypatch)

    with caplog.at_level(logging.INFO, logger="llm_client.client"):
        LLMClient(
            LLMClientConfig(
                model="some-model",
                api_key="sk-super-secret",  # pyright: ignore[reportArgumentType]
            )
        )

    assert not any("sk-super-secret" in record.message for record in caplog.records)


def test_trailing_slash_base_url_gets_the_model_appended(monkeypatch):
    # A per-deployment gateway route: the model name completes the path.
    captured = _capture_init(monkeypatch)
    LLMClient(
        LLMClientConfig(model="gpt-5.4", base_url="https://gateway.example/openai/")
    )

    assert captured["base_url"] == "https://gateway.example/openai/gpt-5.4"


def test_base_url_without_a_trailing_slash_is_left_alone(monkeypatch):
    captured = _capture_init(monkeypatch)
    LLMClient(
        LLMClientConfig(model="gpt-5.4", base_url="https://gateway.example/v1")
    )

    assert captured["base_url"] == "https://gateway.example/v1"


# ---------------------------------------------------------------------------
# from_ai_gateway — the Vault-backed credential path
# ---------------------------------------------------------------------------


def _fake_gateway_secret(monkeypatch, data):
    """Stub app_config.vault's AI Gateway read; llm_client imports it lazily."""
    from app_config import vault

    monkeypatch.setattr(vault, "get_ai_gateway_secret", lambda path=None: data)
    return data


def test_from_ai_gateway_sets_the_api_key(monkeypatch):
    _fake_gateway_secret(monkeypatch, {"token": "gw-token"})
    config = LLMClientConfig.from_ai_gateway(model="some-model")
    assert config.api_key is not None
    assert config.api_key.get_secret_value() == "gw-token"


def test_from_ai_gateway_token_never_in_repr(monkeypatch):
    _fake_gateway_secret(monkeypatch, {"token": "gw-SECRET"})
    assert "gw-SECRET" not in repr(LLMClientConfig.from_ai_gateway())


def test_from_ai_gateway_takes_base_url_from_the_secret(monkeypatch):
    _fake_gateway_secret(
        monkeypatch, {"token": "gw-token", "base_url": "https://gw.example/v1"}
    )
    assert LLMClientConfig.from_ai_gateway().base_url == "https://gw.example/v1"


def test_from_ai_gateway_overrides_win(monkeypatch):
    _fake_gateway_secret(
        monkeypatch, {"token": "gw-token", "base_url": "https://gw.example/v1"}
    )
    config = LLMClientConfig.from_ai_gateway(
        base_url="https://explicit/v1",
        api_key="sk-explicit",  # pyright: ignore[reportArgumentType]
    )
    # The repo-wide rule: an explicit argument beats what Vault returned.
    assert config.base_url == "https://explicit/v1"
    assert config.api_key is not None
    assert config.api_key.get_secret_value() == "sk-explicit"


def test_from_ai_gateway_leaves_other_knobs_at_their_defaults(monkeypatch):
    _fake_gateway_secret(monkeypatch, {"token": "gw-token"})
    assert LLMClientConfig.from_ai_gateway().max_retries == LLMClientConfig().max_retries


def test_from_ai_gateway_enables_proactive_throttle(monkeypatch):
    # The gateway path paces request starts by default, while a plain config
    # leaves the throttle off.
    _fake_gateway_secret(monkeypatch, {"token": "gw-token"})
    assert LLMClientConfig.from_ai_gateway().requests_per_second == 2.0
    assert LLMClientConfig(model="some-model").requests_per_second is None


def test_from_ai_gateway_throttle_override_wins(monkeypatch):
    _fake_gateway_secret(monkeypatch, {"token": "gw-token"})
    # Explicit override beats the gateway default (including turning it off).
    assert (
        LLMClientConfig.from_ai_gateway(requests_per_second=5.0).requests_per_second
        == 5.0
    )
    assert (
        LLMClientConfig.from_ai_gateway(requests_per_second=None).requests_per_second
        is None
    )


def test_from_ai_gateway_token_reaches_the_model(monkeypatch):
    captured = _capture_init(monkeypatch)
    _fake_gateway_secret(
        monkeypatch, {"token": "gw-token", "endpoint": "https://gw.example/v1"}
    )
    LLMClient(LLMClientConfig.from_ai_gateway(model="some-model"))

    assert captured["api_key"] == "gw-token"
    assert captured["base_url"] == "https://gw.example/v1"


def test_plain_config_does_no_vault_io(monkeypatch):
    # LLMClientConfig() must stay free of network calls; only the explicit
    # classmethod touches Vault.
    from app_config import vault

    def _boom(path=None):
        raise AssertionError("LLMClientConfig() must not read Vault")

    monkeypatch.setattr(vault, "get_ai_gateway_secret", _boom)
    assert LLMClientConfig(model="some-model").api_key is None


def test_cert_file_exported_as_ssl_cert_file(monkeypatch, tmp_path):
    _capture_init(monkeypatch)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    cert = tmp_path / "gateway.crt"
    cert.write_text("-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n")

    LLMClient(LLMClientConfig(model="some-model", cert_file=str(cert)))

    assert os.environ["SSL_CERT_FILE"] == str(cert.resolve())


def test_cert_file_pins_explicit_httpx_clients(monkeypatch, tmp_path):
    import ssl

    import httpx

    captured = _capture_init(monkeypatch)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    # Stub the SSL/httpx machinery so the test does not need a genuinely valid
    # CA bundle — only that a present file drives an explicit-client build.
    contexts: list[str] = []
    monkeypatch.setattr(
        ssl, "create_default_context", lambda cafile=None: contexts.append(cafile)
    )
    monkeypatch.setattr(httpx, "Client", lambda **kw: ("sync", kw))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: ("async", kw))
    cert = tmp_path / "gateway.crt"
    cert.write_text("-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n")

    LLMClient(LLMClientConfig(model="some-model", cert_file=str(cert)))

    # Both clients are pinned, so ainvoke verifies against the bundle too.
    assert captured["http_client"][0] == "sync"
    assert captured["http_async_client"][0] == "async"
    assert contexts == [str(cert.resolve())]


def test_missing_cert_file_builds_no_httpx_client(monkeypatch, tmp_path):
    # A missing bundle must not be handed to httpx (which would raise); the
    # SSL_CERT_FILE export path already declined it.
    captured = _capture_init(monkeypatch)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)

    LLMClient(
        LLMClientConfig(model="some-model", cert_file=str(tmp_path / "nope.crt"))
    )

    assert "http_client" not in captured
    assert "http_async_client" not in captured


def test_missing_cert_file_warns_and_leaves_env_alone(monkeypatch, tmp_path, caplog):
    _capture_init(monkeypatch)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)

    with caplog.at_level(logging.WARNING, logger="llm_client.client"):
        LLMClient(
            LLMClientConfig(model="some-model", cert_file=str(tmp_path / "nope.crt"))
        )

    assert "SSL_CERT_FILE" not in os.environ
    assert any("does not exist" in record.message for record in caplog.records)


def test_unset_cert_file_leaves_env_alone(monkeypatch):
    _capture_init(monkeypatch)
    monkeypatch.setenv("SSL_CERT_FILE", "preexisting.pem")

    LLMClient(LLMClientConfig(model="some-model", cert_file=None))

    assert os.environ["SSL_CERT_FILE"] == "preexisting.pem"


def test_api_key_masked_in_repr():
    # Plain str is coerced into the SecretStr field by pydantic at runtime.
    cfg = LLMClientConfig(api_key="sk-secret")  # pyright: ignore[reportArgumentType]
    assert "sk-secret" not in repr(cfg)
    assert "sk-secret" not in str(cfg)


def test_kwargs_escape_hatch_wins_over_named_knobs(monkeypatch):
    captured = _capture_init(monkeypatch)
    LLMClient(
        LLMClientConfig(
            model="some-model",
            temperature=0.1,
            kwargs={"temperature": 0.9, "stop": ["END"]},
        )
    )

    assert captured["temperature"] == 0.9
    assert captured["stop"] == ["END"]


def test_injected_llm_skips_construction(monkeypatch):
    def boom(*args, **kwargs):  # must never be called
        raise AssertionError("ChatOpenAI built despite injected llm")

    monkeypatch.setattr(client_mod, "ChatOpenAI", boom)
    fake = FakeListChatModel(responses=["hi"])
    client = LLMClient(llm=fake)

    assert client.chat_model is fake
    assert client.invoke("hello").content == "hi"


# ---------------------------------------------------------------------------
# Input-token budget
# ---------------------------------------------------------------------------


def test_over_budget_prompt_raises_and_never_calls_model():
    model = _FlakyModel(failures=0)
    client = LLMClient(
        LLMClientConfig(max_input_tokens=5, token_counter=lambda msgs: 100),
        llm=model,
    )

    with pytest.raises(InputTokenLimitError) as exc_info:
        client.invoke("way too long")

    assert exc_info.value.token_count == 100
    assert exc_info.value.max_input_tokens == 5
    assert model.calls == 0  # nothing was sent


def test_within_budget_prompt_goes_through():
    client = LLMClient(
        LLMClientConfig(max_input_tokens=100, token_counter=lambda msgs: 10),
        llm=FakeListChatModel(responses=["fine"]),
    )
    assert client.invoke("short").content == "fine"


def test_no_budget_means_counter_never_runs():
    def exploding_counter(msgs):
        raise AssertionError("token counter ran with max_input_tokens=None")

    client = LLMClient(
        LLMClientConfig(token_counter=exploding_counter),
        llm=FakeListChatModel(responses=["fine"]),
    )
    assert client.invoke("anything").content == "fine"


def test_counter_failure_falls_back_to_chars_over_four(monkeypatch):
    # An unloadable encoding (offline, blocking proxy) degrades to chars//4
    # plus the per-message framing constants, and the call still goes out.
    from llm_client import tokens as tokens_mod

    monkeypatch.setattr(tokens_mod, "_encoding", lambda name: None)
    client = LLMClient(
        LLMClientConfig(max_input_tokens=1_000),
        llm=FakeListChatModel(responses=["ok"]),
    )
    # 400 chars // 4 + 3 message framing + 3 reply primer
    assert client.count_tokens([HumanMessage("x" * 400)]) == 106
    assert client.invoke("hello").content == "ok"


def test_default_counter_needs_no_model_tokenizer():
    # Counting is client-owned: an injected fake model without a tokenizer
    # of its own (FakeListChatModel) must still count and invoke fine.
    client = LLMClient(
        LLMClientConfig(max_input_tokens=1_000_000),
        llm=FakeListChatModel(responses=["fine"]),
    )
    tokens = client.count_tokens([HumanMessage("x" * 400)])
    assert tokens > 0
    assert client.invoke("hello").content == "fine"


# ---------------------------------------------------------------------------
# Rate-limit retry
# ---------------------------------------------------------------------------


def test_rate_limit_errors_are_retried_until_success():
    model = _FlakyModel(failures=2)
    client = LLMClient(_fast_retry_config(max_retries=3), llm=model)

    assert client.invoke("hi").content == "ok"
    assert model.calls == 3  # 2 failures + 1 success


def test_retries_exhausted_reraises():
    model = _FlakyModel(failures=99)
    client = LLMClient(_fast_retry_config(max_retries=2), llm=model)

    with pytest.raises(_FakeRateLimitError):
        client.invoke("hi")
    assert model.calls == 3  # initial attempt + 2 retries


def test_non_rate_limit_error_is_not_retried():
    model = _FlakyModel(failures=99, exc=RuntimeError("connection reset"))
    client = LLMClient(_fast_retry_config(max_retries=5), llm=model)

    with pytest.raises(RuntimeError):
        client.invoke("hi")
    assert model.calls == 1


@pytest.mark.parametrize(
    "exc, expected",
    [
        (_FakeRateLimitError("throttled"), True),
        (type("RateLimitError", (Exception,), {})("nope"), True),
        (Exception("Too Many Requests"), True),
        (Exception("rate_limit_error: slow down"), True),
        (RuntimeError("connection reset"), False),
        (ValueError("bad prompt"), False),
    ],
)
def test_is_rate_limit_error_shapes(exc, expected):
    assert client_mod._is_rate_limit_error(exc) is expected


class _FakeOverloadedError(Exception):
    """Mimics Anthropic's overloaded_error via its status_code attribute."""

    status_code = 529


@pytest.mark.parametrize(
    "exc, expected",
    [
        (_FakeRateLimitError("throttled"), True),
        (_FakeOverloadedError("overloaded"), True),
        (type("ServerError", (Exception,), {"status_code": 503})("unavailable"), True),
        (TimeoutError("read timed out"), True),
        (ConnectionError("peer closed"), True),
        (type("APIConnectionError", (Exception,), {})("no route"), True),
        (type("ReadTimeout", (Exception,), {})("slow"), True),
        # Message-only mentions of transport words must NOT trigger a retry.
        (RuntimeError("connection reset"), False),
        (ValueError("bad prompt"), False),
        (type("BadRequestError", (Exception,), {"status_code": 400})("nope"), False),
    ],
)
def test_is_transient_error_shapes(exc, expected):
    assert client_mod._is_transient_error(exc) is expected


# ---------------------------------------------------------------------------
# Gateway Retry-After handling
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, headers: dict) -> None:
        self.headers = headers


class _RateLimitWithHeaders(Exception):
    """429 carrying rate-limit timing headers on ``exc.response.headers``."""

    status_code = 429

    def __init__(self, headers: dict, msg: str = "throttled") -> None:
        super().__init__(msg)
        self.response = _Resp(headers)


def test_retry_after_seconds_parses_delta_seconds():
    exc = _RateLimitWithHeaders({"retry-after": "7"})
    assert client_mod._retry_after_seconds(exc) == 7.0


def test_retry_after_ms_takes_precedence_over_retry_after():
    exc = _RateLimitWithHeaders({"retry-after-ms": "1500", "retry-after": "5"})
    assert client_mod._retry_after_seconds(exc) == 1.5


def test_retry_after_is_case_insensitive_for_plain_dict():
    exc = _RateLimitWithHeaders({"Retry-After": "3"})
    assert client_mod._retry_after_seconds(exc) == 3.0


def test_retry_after_http_date_form():
    from email.utils import formatdate

    exc = _RateLimitWithHeaders({"retry-after": formatdate(time.time() + 30, usegmt=True)})
    delay = client_mod._retry_after_seconds(exc)
    assert delay is not None
    # Whole-second HTTP-date precision, so allow slack around the 30s target.
    assert 20.0 <= delay <= 31.0


def test_retry_after_absent_returns_none():
    # A 429 without headers falls back to the client's own backoff.
    assert client_mod._retry_after_seconds(_FakeRateLimitError("throttled")) is None


def test_retry_delay_honors_server_value_capped():
    model = _FlakyModel(failures=0)
    client = LLMClient(
        _fast_retry_config(max_retries=3, retry_after_max_seconds=10.0), llm=model
    )
    exc = _RateLimitWithHeaders({"retry-after": "999"})
    # Server value honored but bounded by retry_after_max_seconds.
    assert client._retry_delay(exc, attempt=1, attempts=4, op="invoke") == 10.0


def test_retry_delay_uses_server_value_verbatim_within_cap():
    client = LLMClient(_fast_retry_config(max_retries=3), llm=_FlakyModel(failures=0))
    exc = _RateLimitWithHeaders({"retry-after-ms": "250"})
    assert client._retry_delay(exc, attempt=1, attempts=4, op="invoke") == 0.25


def test_retry_with_retry_after_header_eventually_succeeds():
    model = _FlakyModel(
        failures=1, exc=_RateLimitWithHeaders({"retry-after-ms": "1"})
    )
    client = LLMClient(_fast_retry_config(max_retries=2), llm=model)

    assert client.invoke("hi").content == "ok"
    assert model.calls == 2


def test_overloaded_5xx_errors_are_retried():
    model = _FlakyModel(failures=1, exc=_FakeOverloadedError("overloaded"))
    client = LLMClient(_fast_retry_config(max_retries=2), llm=model)

    assert client.invoke("hi").content == "ok"
    assert model.calls == 2


def test_timeout_errors_are_retried():
    model = _FlakyModel(failures=1, exc=TimeoutError("read timed out"))
    client = LLMClient(_fast_retry_config(max_retries=2), llm=model)

    assert client.invoke("hi").content == "ok"
    assert model.calls == 2


# ---------------------------------------------------------------------------
# Prompt-cache breakpoints — probe the gateway, fall back when refused
# ---------------------------------------------------------------------------


def _cached_system_message() -> SystemMessage:
    """A system prompt carrying an Anthropic-style cache breakpoint, the shape
    pipeline.engine builds when prompt_caching is on."""
    return SystemMessage(
        content=[
            {
                "type": "text",
                "text": "You translate SAS.",
                "cache_control": {"type": "ephemeral"},
            }
        ]
    )


class _SchemaRejection(Exception):
    """A gateway refusing an unknown content-part key, OpenAI 400 shape."""

    status_code = 400

    def __init__(self, msg: str = "Unrecognized key 'cache_control' in content part"):
        super().__init__(msg)


class _RecordingModel:
    """Records the messages of every call; optionally rejects cache_control."""

    def __init__(self, *, rejects_cache_control: bool = False) -> None:
        self.rejects_cache_control = rejects_cache_control
        self.calls: list[list] = []

    @staticmethod
    def _has_cache_control(messages) -> bool:
        return any(
            isinstance(part, dict) and "cache_control" in part
            for m in messages
            if isinstance(m.content, list)
            for part in m.content
        )

    def invoke(self, messages, config=None):
        self.calls.append(list(messages))
        if self.rejects_cache_control and self._has_cache_control(messages):
            raise _SchemaRejection()
        return AIMessage("ok")

    async def ainvoke(self, messages, config=None):
        return self.invoke(messages, config)


def test_cache_control_survives_on_a_gateway_that_accepts_it():
    model = _RecordingModel()
    client = LLMClient(_fast_retry_config(), llm=model)

    assert client.invoke([_cached_system_message()]).content == "ok"
    assert len(model.calls) == 1
    assert model._has_cache_control(model.calls[0])  # sent, untouched


def test_rejected_cache_control_is_stripped_and_the_call_retried(caplog):
    model = _RecordingModel(rejects_cache_control=True)
    client = LLMClient(_fast_retry_config(), llm=model)

    with caplog.at_level(logging.WARNING, logger="llm_client.client"):
        assert client.invoke([_cached_system_message()]).content == "ok"

    assert len(model.calls) == 2  # probe, then the stripped re-send
    assert model._has_cache_control(model.calls[0])
    assert not model._has_cache_control(model.calls[1])
    assert any("cache_control" in record.message for record in caplog.records)


def test_fallback_works_with_no_retry_budget():
    # The probe must not spend an attempt: at max_retries=0 there is none to
    # spend, and the call still has to succeed.
    model = _RecordingModel(rejects_cache_control=True)
    client = LLMClient(_fast_retry_config(max_retries=0), llm=model)

    assert client.invoke([_cached_system_message()]).content == "ok"
    assert len(model.calls) == 2


def test_fallback_leaves_the_retry_budget_intact():
    # One cache_control demotion plus a full run of transient failures: the
    # demotion must not have eaten one of the retries.
    model = _RecordingModel(rejects_cache_control=True)
    calls = {"n": 0}
    original_invoke = model.invoke

    def flaky_invoke(messages, config=None):
        result = original_invoke(messages, config)  # raises on cache_control
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _FakeRateLimitError("throttled")
        return result

    model.invoke = flaky_invoke
    client = LLMClient(_fast_retry_config(max_retries=2), llm=model)

    assert client.invoke([_cached_system_message()]).content == "ok"


def test_stripping_keeps_the_prompt_text_intact():
    model = _RecordingModel(rejects_cache_control=True)
    client = LLMClient(_fast_retry_config(), llm=model)
    client.invoke([_cached_system_message()])

    assert model.calls[1][0].content == [
        {"type": "text", "text": "You translate SAS."}
    ]


def test_the_gateway_is_only_asked_once():
    # After a refusal the answer is remembered: later calls go out stripped,
    # so the fallback costs one failed request per process, not per call.
    model = _RecordingModel(rejects_cache_control=True)
    client = LLMClient(_fast_retry_config(), llm=model)

    client.invoke([_cached_system_message()])
    client.invoke([_cached_system_message()])

    assert len(model.calls) == 3  # 1 refused + 1 retry + 1 pre-stripped
    assert not model._has_cache_control(model.calls[2])


def test_caller_messages_are_not_mutated_by_the_fallback():
    # The pipeline reuses its prompt template's system block across items; a
    # retry must copy rather than edit it in place.
    original = _cached_system_message()
    client = LLMClient(_fast_retry_config(), llm=_RecordingModel(rejects_cache_control=True))
    client.invoke([original])

    assert original.content[0]["cache_control"] == {"type": "ephemeral"}


def test_unrelated_400_is_not_treated_as_a_cache_control_refusal():
    # Stripping would not help a genuinely bad request, and a blind re-send
    # would double its cost.
    bad_request = type("BadRequestError", (Exception,), {"status_code": 400})(
        "model 'nope' does not exist"
    )
    model = _FlakyModel(failures=99, exc=bad_request)
    client = LLMClient(_fast_retry_config(), llm=model)

    with pytest.raises(Exception, match="does not exist"):
        client.invoke([_cached_system_message()])
    assert model.calls == 1


def test_cache_control_refusal_without_a_breakpoint_propagates():
    # Same status and wording, but nothing to strip — must not loop.
    model = _FlakyModel(failures=99, exc=_SchemaRejection())
    client = LLMClient(_fast_retry_config(), llm=model)

    with pytest.raises(_SchemaRejection):
        client.invoke("a plain string prompt")
    assert model.calls == 1


def test_async_path_also_falls_back():
    model = _RecordingModel(rejects_cache_control=True)
    client = LLMClient(_fast_retry_config(), llm=model)

    assert asyncio.run(client.ainvoke([_cached_system_message()])).content == "ok"
    assert len(model.calls) == 2
    assert not model._has_cache_control(model.calls[1])


# ---------------------------------------------------------------------------
# Token usage — logged per call, accumulated on the client
# ---------------------------------------------------------------------------


class _UsageModel:
    """Returns a response carrying *usage* as LangChain's usage_metadata."""

    def __init__(self, *usages) -> None:
        self.usages = list(usages)

    def invoke(self, messages, config=None):
        return AIMessage("ok", usage_metadata=self.usages.pop(0))

    async def ainvoke(self, messages, config=None):
        return self.invoke(messages, config)


def _usage(input_tokens, output_tokens, **details):
    block = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if details:
        block["input_token_details"] = details
    return block


def test_usage_is_read_from_the_response():
    client = LLMClient(llm=_UsageModel(_usage(120, 30)))
    client.invoke("hi")

    assert client.usage.input_tokens == 120
    assert client.usage.output_tokens == 30
    assert client.usage.total_tokens == 150
    assert client.usage.calls == 1


def test_usage_accumulates_across_calls():
    client = LLMClient(llm=_UsageModel(_usage(100, 10), _usage(200, 20)))
    client.invoke("one")
    client.invoke("two")

    assert client.usage.input_tokens == 300
    assert client.usage.output_tokens == 30
    assert client.usage.calls == 2


def test_usage_is_logged_per_call(caplog):
    client = LLMClient(llm=_UsageModel(_usage(120, 30)))

    with caplog.at_level(logging.INFO, logger="llm_client.client"):
        client.invoke("hi")

    assert any(
        "in=120" in record.message and "out=30" in record.message
        for record in caplog.records
    )


def test_cache_token_details_are_captured():
    client = LLMClient(llm=_UsageModel(_usage(120, 30, cache_read=100)))
    client.invoke("hi")

    assert client.usage.cache_read_tokens == 100


def test_usage_falls_back_to_the_raw_token_usage_block():
    # A gateway whose usage block LangChain did not normalize.
    class _RawUsageModel:
        def invoke(self, messages, config=None):
            return AIMessage(
                "ok",
                response_metadata={
                    "token_usage": {
                        "prompt_tokens": 80,
                        "completion_tokens": 12,
                        "total_tokens": 92,
                        "prompt_tokens_details": {"cached_tokens": 64},
                    }
                },
            )

    client = LLMClient(llm=_RawUsageModel())
    client.invoke("hi")

    assert client.usage.input_tokens == 80
    assert client.usage.output_tokens == 12
    assert client.usage.cache_read_tokens == 64


def test_missing_usage_warns_once_and_leaves_the_total_empty(caplog):
    client = LLMClient(llm=FakeListChatModel(responses=["a", "b"]))

    with caplog.at_level(logging.WARNING, logger="llm_client.client"):
        client.invoke("one")
        client.invoke("two")

    assert client.usage == TokenUsage()
    warnings = [r for r in caplog.records if "no token usage" in r.message]
    assert len(warnings) == 1


def test_malformed_usage_counts_never_break_a_successful_call():
    # AIMessage validates usage_metadata, so a junk block can only reach us
    # from a response object the SDK built loosely — but if one does, a call
    # that already succeeded must not fail over its accounting.
    class _JunkUsageResponse:
        content = "ok"
        usage_metadata = {"input_tokens": None, "output_tokens": "lots"}

    class _JunkUsageModel:
        def invoke(self, messages, config=None):
            return _JunkUsageResponse()

    client = LLMClient(llm=_JunkUsageModel())

    assert client.invoke("hi").content == "ok"
    assert client.usage.input_tokens == 0
    assert client.usage.calls == 1


def test_usage_is_recorded_on_the_async_path():
    client = LLMClient(llm=_UsageModel(_usage(50, 5)))
    asyncio.run(client.ainvoke("hi"))

    assert client.usage.input_tokens == 50


# ---------------------------------------------------------------------------
# Async — ainvoke mirrors invoke (budget, retry, LCEL)
# ---------------------------------------------------------------------------


def test_ainvoke_returns_response():
    client = LLMClient(llm=FakeListChatModel(responses=["hi"]))
    assert asyncio.run(client.ainvoke("hello")).content == "hi"


def test_ainvoke_retries_transient_errors():
    model = _FlakyModel(failures=2)
    client = LLMClient(_fast_retry_config(max_retries=3), llm=model)

    assert asyncio.run(client.ainvoke("hi")).content == "ok"
    assert model.calls == 3  # 2 failures + 1 success


def test_ainvoke_does_not_retry_non_transient_errors():
    model = _FlakyModel(failures=99, exc=ValueError("bad prompt"))
    client = LLMClient(_fast_retry_config(max_retries=5), llm=model)

    with pytest.raises(ValueError):
        asyncio.run(client.ainvoke("hi"))
    assert model.calls == 1


def test_ainvoke_enforces_input_token_budget():
    model = _FlakyModel(failures=0)
    client = LLMClient(
        LLMClientConfig(max_input_tokens=5, token_counter=lambda msgs: 100),
        llm=model,
    )

    with pytest.raises(InputTokenLimitError):
        asyncio.run(client.ainvoke("way too long"))
    assert model.calls == 0  # nothing was sent


def test_as_runnable_async_path_composes_with_prompt():
    client = LLMClient(llm=FakeListChatModel(responses=["translated"]))
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You translate SAS."), ("human", "{input}")]
    )
    chain = prompt | client.as_runnable()

    result = asyncio.run(chain.ainvoke({"input": "data work.a; run;"}))
    assert isinstance(result, AIMessage)
    assert result.content == "translated"


# ---------------------------------------------------------------------------
# LCEL integration — client in place of the raw model
# ---------------------------------------------------------------------------


def test_as_runnable_composes_with_prompt():
    client = LLMClient(llm=FakeListChatModel(responses=["translated"]))
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You translate SAS."), ("human", "{input}")]
    )
    chain = prompt | client.as_runnable()

    result = chain.invoke({"input": "data work.a; run;"})
    assert isinstance(result, AIMessage)
    assert result.content == "translated"


def test_pipeline_enforces_input_token_budget():
    # End-to-end: SasLLMPipeline forwards max_input_tokens into LLMClient;
    # the formatted batch prompt (hundreds of chars) blows a 1-token budget
    # under any counter, so the call must fail before reaching the fake LLM.
    from chunker.models import SasBatch, SasChunk, SasChunkKind, SasChunkMetadata
    from pipeline import SasLLMPipeline
    from pipeline.setup import MemorySetup
    from memory.store import MemoryHub

    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(
            model="unused-because-llm-injected", max_input_tokens=1
        ),
        memory_setup=MemorySetup(memory=MemoryHub()),
        llm=FakeListChatModel(responses=["never reached"]),
    )
    chunk = SasChunk(
        chunk_id="f1-chunk-0001",
        source_id="etl.sas",
        text="data work.a; run;",
        kind=SasChunkKind.DATA_STEP,
        title="Step",
        start_line=1,
        end_line=1,
        start_char=0,
        end_char=17,
        metadata=SasChunkMetadata(),
    )
    batch = SasBatch(batch_id="merged-001", chunks=[chunk], source_files=["etl.sas"])

    with pytest.raises(InputTokenLimitError):
        pipeline._process(items=[batch], diagnostics=[], thread_id="run::etl.sas")


# ---------------------------------------------------------------------------
# AI Gateway conformance: the ai-gateway-version header and provider splitting
# ---------------------------------------------------------------------------


@pytest.fixture
def _isolated_config(monkeypatch, tmp_path):
    """An empty config.json, so role overlays are only what a test sets."""
    import app_config

    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(app_config.ENV_VAR, str(cfg))
    monkeypatch.delenv("LLM_GATEWAY_VERSION", raising=False)
    app_config.clear_cache()
    yield cfg
    app_config.clear_cache()


def _write_config(cfg_path, mapping) -> None:
    import json

    import app_config

    cfg_path.write_text(json.dumps(mapping), encoding="utf-8")
    app_config.clear_cache()


def test_gateway_version_is_sent_as_a_header(monkeypatch):
    captured = _capture_init(monkeypatch)
    LLMClient(LLMClientConfig(model="some-model", gateway_version="v2"))

    assert captured["default_headers"]["ai-gateway-version"] == "v2"


def test_no_gateway_version_header_when_unset(monkeypatch, _isolated_config):
    captured = _capture_init(monkeypatch)
    LLMClient(LLMClientConfig(model="some-model", url_headers={"X-Team": "sas"}))

    assert captured["default_headers"] == {"X-Team": "sas"}


def test_explicit_gateway_version_header_wins(monkeypatch):
    # setdefault, the same rule as api-key: an operator who set the header
    # through url_headers meant it.
    captured = _capture_init(monkeypatch)
    LLMClient(
        LLMClientConfig(
            model="some-model",
            gateway_version="v2",
            url_headers={"ai-gateway-version": "v1"},
        )
    )

    assert captured["default_headers"]["ai-gateway-version"] == "v1"


def test_gateway_version_read_from_config_json(_isolated_config):
    _write_config(_isolated_config, {"llm_client": {"gateway_version": "v2"}})
    assert LLMClientConfig().gateway_version == "v2"


def test_gateway_version_env_beats_config_json(monkeypatch, _isolated_config):
    _write_config(_isolated_config, {"llm_client": {"gateway_version": "v2"}})
    monkeypatch.setenv("LLM_GATEWAY_VERSION", "v3")
    assert LLMClientConfig().gateway_version == "v3"


def test_split_model_separates_provider_and_lowercases():
    assert client_mod._split_model("anthropic:claude-sonnet-4-5") == (
        "anthropic",
        "claude-sonnet-4-5",
    )
    # The reference reads "provider: model" out of a SharePoint column, spaces
    # and mixed case included.
    assert client_mod._split_model("Anthropic: Claude-Sonnet-4-5") == (
        "Anthropic",
        "claude-sonnet-4-5",
    )


def test_split_model_leaves_an_unprefixed_id_alone():
    # No prefix means the id was written by hand; its case is the operator's.
    assert client_mod._split_model("GPT-5.4") == (None, "GPT-5.4")


def test_prefixed_model_sends_only_the_id_to_the_gateway(monkeypatch):
    captured = _capture_init(monkeypatch)
    LLMClient(LLMClientConfig(model="anthropic:claude-sonnet-4-5"))

    assert captured["model"] == "claude-sonnet-4-5"


def test_model_provider_is_recorded_not_dropped(monkeypatch, caplog):
    _capture_init(monkeypatch)
    with caplog.at_level(logging.INFO, logger="llm_client.client"):
        LLMClient(LLMClientConfig(model="anthropic:claude-sonnet-4-5"))

    assert "provider=anthropic" in caplog.text


def test_explicit_model_provider_beats_the_prefix(monkeypatch, caplog):
    _capture_init(monkeypatch)
    with caplog.at_level(logging.INFO, logger="llm_client.client"):
        LLMClient(
            LLMClientConfig(
                model="anthropic:claude-sonnet-4-5", model_provider="pinned"
            )
        )

    assert "provider=pinned" in caplog.text


# ---------------------------------------------------------------------------
# provider_client="native"
# ---------------------------------------------------------------------------


class _FakeUsage:
    prompt_tokens = 11
    completion_tokens = 7
    total_tokens = 18


class _FakeCompletion:
    def __init__(self, content):
        message: Any = type("M", (), {})()
        message.content = content
        choice: Any = type("Choice", (), {})()
        choice.message = message
        self.choices = [choice]
        self.usage = _FakeUsage()


class _FakeOpenAI:
    """Records construction kwargs and every create() body."""

    instances: list["_FakeOpenAI"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[dict] = []
        _FakeOpenAI.instances.append(self)
        completions: Any = type("Completions", (), {})()
        completions.create = self._create
        chat: Any = type("Chat", (), {})()
        chat.completions = completions
        self.chat = chat

    def _create(self, **body):
        self.calls.append(body)
        return _FakeCompletion("native answer")


@pytest.fixture
def _fake_openai(monkeypatch):
    import openai

    _FakeOpenAI.instances.clear()
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    return _FakeOpenAI


def _native_config(**overrides):
    values: dict[str, Any] = {
        "model": "anthropic:claude-sonnet-4-5",
        "provider_client": "native",
        "base_url": "https://gw.example/chat/",
        "gateway_version": "v2",
        "api_key": "sk-secret",
        "temperature": 0.0,
        "max_output_tokens": 64000,
    }
    values.update(overrides)
    return LLMClientConfig(**values)


def test_native_client_is_built_instead_of_chat_openai(monkeypatch, _fake_openai):
    def _explode(**kwargs):
        raise AssertionError("ChatOpenAI must not be built on the native path")

    monkeypatch.setattr(client_mod, "ChatOpenAI", _explode)
    LLMClient(_native_config())

    assert len(_fake_openai.instances) == 1
    built = _fake_openai.instances[0].kwargs
    # Trailing slash => per-deployment route, model as the last path segment.
    assert built["base_url"] == "https://gw.example/chat/claude-sonnet-4-5"
    assert built["api_key"] == "sk-secret"
    assert built["default_headers"]["ai-gateway-version"] == "v2"
    assert built["default_headers"]["api-key"] == "sk-secret"
    # This class owns retries; the SDK's own layer would hide Retry-After.
    assert built["max_retries"] == 0


def test_native_client_invokes_and_maps_the_roles(_fake_openai):
    llm = LLMClient(_native_config())
    reply = llm.invoke(
        [SystemMessage("be brief"), HumanMessage("hi"), AIMessage("hello")]
    )

    assert reply.content == "native answer"
    body = _fake_openai.instances[0].calls[0]
    assert body["model"] == "claude-sonnet-4-5"
    assert body["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert body["max_tokens"] == 64000
    assert body["temperature"] == 0.0


def test_native_client_still_accounts_for_tokens(_fake_openai):
    llm = LLMClient(_native_config())
    llm.invoke("hi")

    # Usage accounting lives in LLMClient, above the model, so it survives the
    # transport swap.
    assert llm.usage.input_tokens == 11
    assert llm.usage.output_tokens == 7
    assert llm.usage.calls == 1


def test_native_client_still_retries(_fake_openai):
    llm = LLMClient(_native_config(retry_base_seconds=0.001))
    calls = {"n": 0}

    def _flaky(**body):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _FakeRateLimitError("429")
        return _FakeCompletion("after retry")

    _fake_openai.instances[0].chat.completions.create = _flaky
    assert llm.invoke("hi").content == "after retry"
    assert calls["n"] == 2


def test_native_client_warns_about_what_it_forfeits(_fake_openai, caplog):
    with caplog.at_level(logging.WARNING, logger="llm_client.client"):
        LLMClient(_native_config(requests_per_second=2.0, model_kwargs={"top_k": 40}))

    assert "requests_per_second" in caplog.text
    assert "model_kwargs" in caplog.text


def test_native_client_reports_no_structured_output(_fake_openai):
    from pydantic import BaseModel as _PydanticModel

    class _Schema(_PydanticModel):
        answer: str

    llm = LLMClient(_native_config())
    # with_structured_output does not exist on the wrapped callable, and the
    # caller must learn that by asking rather than by a mid-run AttributeError.
    assert llm.supports_structured_output(_Schema) is False


def test_flattens_multipart_content_for_the_native_wire_format(_fake_openai):
    llm = LLMClient(_native_config())
    llm.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "part one "},
                    {"type": "text", "text": "part two"},
                ]
            )
        ]
    )

    body = _fake_openai.instances[0].calls[0]
    assert body["messages"] == [{"role": "user", "content": "part one part two"}]


# ---------------------------------------------------------------------------
# Per-role overlays (LLMClientConfig.for_role)
# ---------------------------------------------------------------------------


_ROLES_FILE = {
    "llm_client": {
        "timeout": 6000,
        "model": "claude-sonnet-4-5",
        "roles": {
            "validator": {"timeout": 12000},
            "complexity": {"model": "claude-opus-4-6"},
        },
    }
}


def test_for_role_applies_the_overlay(_isolated_config):
    _write_config(_isolated_config, _ROLES_FILE)
    validator = LLMClientConfig.for_role("validator")

    assert validator.timeout == 12000
    # A key the role does not mention resolves from the base section.
    assert validator.model == "claude-sonnet-4-5"


def test_for_role_leaves_unmentioned_keys_at_their_defaults(_isolated_config):
    _write_config(_isolated_config, _ROLES_FILE)
    assert LLMClientConfig.for_role("complexity").model == "claude-opus-4-6"
    assert LLMClientConfig.for_role("complexity").timeout == 6000
    assert LLMClientConfig.for_role("complexity").max_retries == 3


def test_explicit_argument_beats_the_role(_isolated_config):
    _write_config(_isolated_config, _ROLES_FILE)
    config = LLMClientConfig.for_role("validator", timeout=5.0)

    assert config.timeout == 5.0


def test_unknown_role_is_the_plain_config(_isolated_config):
    _write_config(_isolated_config, _ROLES_FILE)
    assert LLMClientConfig.for_role("no-such-role").timeout == 6000
    assert LLMClientConfig.for_role("no-such-role").model == "claude-sonnet-4-5"
