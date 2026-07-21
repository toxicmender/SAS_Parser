"""
Tests for llm_client.client — construction knobs, input-token budget,
and rate-limit retry.

No network or API keys: model construction is exercised by monkeypatching
``llm_client.client.init_chat_model`` to capture kwargs, and invocation by
injecting fakes (FakeListChatModel or small stubs). Retry tests set
``retry_base_seconds`` tiny so backoff sleeps are negligible.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

import llm_client.client as client_mod
from llm_client import InputTokenLimitError, LLMClient, LLMClientConfig


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

    def fake_init(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return FakeListChatModel(responses=["built"])

    monkeypatch.setattr(client_mod, "init_chat_model", fake_init)
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
    assert captured["default_headers"] == {"X-Team": "sas"}
    assert captured["timeout"] == 42.5
    assert captured["model_kwargs"] == {"top_k": 40}


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
        raise AssertionError("init_chat_model called despite injected llm")

    monkeypatch.setattr(client_mod, "init_chat_model", boom)
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


def test_counter_failure_falls_back_to_chars_over_four():
    class _NoTokenizerModel:
        def get_num_tokens_from_messages(self, messages):
            raise ImportError("no tokenizer installed")

        def invoke(self, messages, config=None):
            return AIMessage("ok")

    client = LLMClient(LLMClientConfig(max_input_tokens=1_000), llm=_NoTokenizerModel())
    assert client.count_tokens([HumanMessage("x" * 400)]) == 100  # 400 chars // 4
    assert client.invoke("hello").content == "ok"


def test_default_counter_falls_back_to_approximation():
    # FakeListChatModel's native counter needs the optional ``transformers``
    # GPT-2 tokenizer; whether or not it is installed, counting must succeed.
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
    # the formatted chunk prompt (hundreds of chars) blows a 1-token budget
    # under any counter, so the call must fail before reaching the fake LLM.
    from chunker.models import SasChunk, SasChunkKind, SasChunkMetadata
    from chunker.pipeline import SasLLMPipeline
    from memory.store import MemoryHub

    pipeline = SasLLMPipeline(
        model="unused-because-llm-injected",
        memory=MemoryHub(),
        llm=FakeListChatModel(responses=["never reached"]),
        max_input_tokens=1,
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

    with pytest.raises(InputTokenLimitError):
        pipeline._process(items=[chunk], diagnostics=[], thread_id="run::etl.sas")
