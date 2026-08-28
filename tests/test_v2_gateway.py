"""Gateway, local persistence, and operational conversion adapter contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from sas_migrate.adapters.ai import GatewayLLMError, OpenAICompatibleLLM
from sas_migrate.adapters.conversion import (
    DirectoryArtifactRepository,
    InMemoryRunEventRepository,
    InMemoryTokenRecordRepository,
    LocalConversionTranslator,
)
from sas_migrate.adapters.conversion.runtime import InMemoryAcceptedResponseRepository
from sas_migrate.application.conversion import (
    ConversionRequest,
    ConversionTranslationCommand,
)
from sas_migrate.application.ports import (
    ArtifactWrite,
    CredentialValue,
    ProviderResponse,
    ProviderTokenUsage,
    SourceObject,
)
from sas_migrate.config import GatewaySettings, InfrastructureSettings
from sas_migrate.core.responses import (
    ResponseEnvelope,
    ResponseMode,
    TranslationCell,
    TranslationCellKind,
    TranslationDocument,
)
from sas_migrate.core.runs import RunEvent, RunEventType
from sas_migrate.core.targets import TargetId, resolve_local_target
from sas_migrate.core.targets.validation import ResponseValidationResult
from sas_migrate.core.tokens import (
    CallTokenRecord,
    MessageRole,
    PromptAssembly,
    PromptComponent,
    TokenBudgetPolicy,
    TokenCategory,
)


class _Completions:
    def __init__(self, completion: Any) -> None:
        self.completion = completion
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return self.completion


def _client(completion: Any) -> tuple[Any, _Completions]:
    completions = _Completions(completion)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def _document(target: TargetId = TargetId.SPARK_SQL) -> TranslationDocument:
    sql = target is TargetId.SPARK_SQL
    return TranslationDocument(
        target=target,
        analysis="Preserve semantics.",
        cells=(
            TranslationCell(
                kind=TranslationCellKind.CODE,
                source="SELECT 1" if sql else "result = spark.range(1)",
                language="sql" if sql else "python",
            ),
        ),
    )


def _prompt() -> PromptAssembly:
    return PromptAssembly(
        components=(
            PromptComponent(
                category=TokenCategory.SAS_SOURCE,
                text="proc sql; select 1; quit;",
                message_role=MessageRole.USER,
                token_count=8,
                source_id="source-1",
            ),
        ),
        estimator="test",
        encoding="test",
    )


def _credential() -> CredentialValue:
    return CredentialValue(
        name="gateway",
        value=SecretStr("super-secret"),
        source="test",
    )


@pytest.mark.anyio
async def test_gateway_normalizes_structured_output_usage_and_request_shape() -> None:
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=_document().model_dump_json()))],
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=30,
            prompt_tokens_details=SimpleNamespace(cached_tokens=40),
            cache_creation_input_tokens=10,
        ),
    )
    client, requests = _client(completion)
    adapter = OpenAICompatibleLLM(
        settings=GatewaySettings(base_url="https://gateway.example/v1"),
        credential=_credential(),
        model="gpt-5.4",
        client=client,
    )

    response = await adapter.invoke(
        _prompt(),
        resolve_local_target("sql"),
        attempt=1,
    )

    assert response.structured_document == _document()
    assert response.structured_error is None
    assert response.usage == ProviderTokenUsage(
        input_tokens=120,
        output_tokens=30,
        cache_read_tokens=40,
        cache_write_tokens=10,
    )
    request = requests.requests[0]
    assert request["model"] == "gpt-5.4"
    assert request["messages"] == [
        {"role": "user", "content": "proc sql; select 1; quit;"}
    ]
    assert request["response_format"]["type"] == "json_schema"


@pytest.mark.anyio
async def test_gateway_retains_raw_fallback_and_rejects_empty_responses() -> None:
    raw = "```sql\nSELECT 1\n```"
    client, _ = _client(
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=raw))],
            usage=None,
        )
    )
    response = await OpenAICompatibleLLM(
        settings=GatewaySettings(),
        credential=_credential(),
        model="model",
        client=client,
    ).invoke(_prompt(), resolve_local_target("sql"), attempt=2)
    assert response.raw_message == raw
    assert response.structured_document is None
    assert response.structured_error
    assert response.usage == ProviderTokenUsage()

    for choices in ([], [SimpleNamespace(message=SimpleNamespace(content=[]))]):
        empty_client, _ = _client(SimpleNamespace(choices=choices, usage=None))
        with pytest.raises(GatewayLLMError, match="choice|content"):
            await OpenAICompatibleLLM(
                settings=GatewaySettings(),
                credential=_credential(),
                model="model",
                client=empty_client,
            ).invoke(_prompt(), resolve_local_target("sql"), attempt=1)


@pytest.mark.anyio
async def test_gateway_lazily_builds_client_with_secret_headers_and_list_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai

    document = _document()
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[
                        {"text": document.model_dump_json()},
                        SimpleNamespace(text=""),
                        {"ignored": "value"},
                    ]
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens="unknown",
            completion_tokens=-1,
            prompt_tokens_details=None,
        ),
    )
    client, requests = _client(completion)
    constructed: list[dict[str, Any]] = []

    def fake_client(**kwargs: Any) -> Any:
        constructed.append(kwargs)
        return client

    monkeypatch.setattr(openai, "AsyncOpenAI", fake_client)
    adapter = OpenAICompatibleLLM(
        settings=GatewaySettings(
            base_url="https://gateway.example/v1",
            gateway_version="2026-08",
            timeout=10,
            max_retries=1,
        ),
        credential=_credential(),
        model="model",
    )
    response = await adapter.invoke(
        _prompt(),
        resolve_local_target("sql"),
        attempt=1,
    )

    assert response.structured_document == document
    assert response.usage == ProviderTokenUsage()
    assert requests.requests
    assert constructed[0]["base_url"] == "https://gateway.example/v1"
    assert constructed[0]["default_headers"] == {
        "api-key": "super-secret",
        "ai-gateway-version": "2026-08",
    }

    invalid_client, _ = _client(
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=123))],
            usage=None,
        )
    )
    with pytest.raises(GatewayLLMError, match="content"):
        await OpenAICompatibleLLM(
            settings=GatewaySettings(),
            credential=_credential(),
            model="model",
            client=invalid_client,
        ).invoke(_prompt(), resolve_local_target("sql"), attempt=1)


@pytest.mark.anyio
async def test_local_repositories_write_safely_and_filter_ledgers(tmp_path: Path) -> None:
    artifacts = DirectoryArtifactRepository(tmp_path)
    location = await artifacts.write(
        "run one",
        ArtifactWrite(
            artifact_id="nested/report.json",
            media_type="application/json",
            content=b"{}\n",
        ),
    )
    assert Path(location).read_bytes() == b"{}\n"
    assert Path(location).is_relative_to(tmp_path)
    with pytest.raises(ValueError, match="unsafe artifact"):
        await artifacts.write(
            "run",
            ArtifactWrite(
                artifact_id="../escape.json",
                media_type="application/json",
                content=b"{}",
            ),
        )

    events = InMemoryRunEventRepository()
    event = RunEvent(
        event_id="event-1",
        event_type=RunEventType.RUN_COMPLETED,
        occurred_at="2026-08-28T00:00:00Z",
        run_id="run",
        thread_id="thread",
    )
    await events.append(event)
    assert await events.events("run", "thread") == (event,)
    assert await events.events("other", "thread") == ()

    tokens = InMemoryTokenRecordRepository()
    record = CallTokenRecord(
        run_id="run",
        thread_id="thread",
        item_id="item",
        attempt=1,
        target=TargetId.SPARK_SQL,
        estimator="test",
        encoding="test",
        estimated_input_by_category={TokenCategory.SAS_SOURCE: 1},
        estimated_input_total=1,
    )
    await tokens.append(record)
    assert await tokens.records("run", "thread") == (record,)
    assert await tokens.records("other", "thread") == ()

    accepted = InMemoryAcceptedResponseRepository()
    assert await accepted.accepted_response("run", "thread", "item") is None
    target = resolve_local_target("sql")
    envelope = ResponseEnvelope(
        mode=ResponseMode.STRUCTURED,
        raw_message="structured",
        document=_document(),
        resolved_target=target,
        validation=ResponseValidationResult.accepted(TargetId.SPARK_SQL),
    )
    await accepted.remember_accepted("run", "thread", "item", envelope)
    await accepted.fork_accepted(
        "run",
        "thread",
        "fork",
        "fork-thread",
        ("item", "missing"),
    )
    assert await accepted.accepted_response("fork", "fork-thread", "item") is envelope
    await accepted.forget_accepted("fork", "fork-thread", ("item",))
    assert await accepted.accepted_response("fork", "fork-thread", "item") is None


class _LLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, prompt, target, *, attempt: int) -> ProviderResponse:
        del prompt, attempt
        self.calls += 1
        document = _document(target.target)
        return ProviderResponse(
            raw_message=document.model_dump_json(),
            structured_document=document,
            usage=ProviderTokenUsage(input_tokens=100, output_tokens=20),
        )


def _command(*, dry_run: bool) -> ConversionTranslationCommand:
    return ConversionTranslationCommand(
        request=ConversionRequest(
            request_id="local-1",
            application_name="Local App",
            output_language="spark_sql",
        ),
        target=resolve_local_target("spark_sql"),
        model="gateway-model",
        sources=(
            SourceObject(
                source_id="program.sas",
                name="program.sas",
                content=b"proc sql; create table out as select 1 as id; quit;",
            ),
        ),
        dry_run=dry_run,
    )


@pytest.mark.anyio
async def test_local_conversion_translator_dry_run_and_live_artifacts(tmp_path: Path) -> None:
    llm = _LLM()
    policy = TokenBudgetPolicy(
        max_input_tokens=100_000,
        reserved_output_tokens=2_000,
        safety_margin_tokens=500,
    )
    translator = LocalConversionTranslator(
        output_dir=tmp_path,
        llm_factory=lambda _model: llm,
        policy=policy,
    )

    planned = await translator.translate(_command(dry_run=True))
    assert planned.ok
    assert llm.calls == 0
    plan = json.loads(Path(planned.artifacts[0].location).read_text("utf-8"))
    assert plan["target"] == "spark_sql"
    assert plan["sqlglot_dialect"] == "databricks"
    assert plan["sources"] == ["program.sas"]

    translated = await translator.translate(_command(dry_run=False))
    assert translated.ok
    assert llm.calls >= 1
    kinds = {artifact.kind for artifact in translated.artifacts}
    assert {"canonical_translation", "notebook", "conversion_run_summary"} <= kinds
    assert all(Path(artifact.location).is_file() for artifact in translated.artifacts)

    with pytest.raises(ValueError, match="max_attempts"):
        LocalConversionTranslator(
            output_dir=tmp_path,
            llm_factory=lambda _model: llm,
            policy=policy,
            max_attempts=0,
        )

    invalid_utf8 = _command(dry_run=False).model_copy(
        update={
            "sources": (
                SourceObject(
                    source_id="bad.sas",
                    name="bad.sas",
                    content=b"\xff",
                ),
            )
        }
    )
    failed = await translator.translate(invalid_utf8)
    assert not failed.ok
    assert "not valid UTF-8" in (failed.error or "")


def test_gateway_settings_are_secret_free_and_validate_transport_contract() -> None:
    settings = InfrastructureSettings(
        gateway=GatewaySettings(
            base_url="https://gateway.example/v1",
            api_key_env="CUSTOM_GATEWAY_TOKEN",
            gateway_version="2026-08",
        )
    )
    payload = settings.model_dump_json()
    assert settings.gateway.max_retries == 2
    assert "CUSTOM_GATEWAY_TOKEN" in payload
    assert "super-secret" not in payload
    with pytest.raises(ValueError, match="gateway model"):
        OpenAICompatibleLLM(
            settings=settings.gateway,
            credential=_credential(),
            model=" ",
        )
