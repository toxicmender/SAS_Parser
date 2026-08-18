"""Phase 5 translation item and event-sourced run-control tests."""

from __future__ import annotations

import asyncio
import pathlib
import sys
from datetime import UTC, datetime, timedelta

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from sas_migrate.application import (
    BudgetedResponseAttemptService,
    NotebookTranslation,
    PromptAssembler,
    PromptContext,
    RunStateService,
    TokenAccountingService,
    TokenAuditPersistenceService,
    TokenBudgetEnforcer,
    TranslateCorpus,
    TranslateCorpusRequest,
    TranslationArtifactService,
    TranslationItem,
    TranslationPromptBuilder,
    render_effective_prompt,
    render_notebooks,
)
from sas_migrate.application.ports import (
    ArtifactWrite,
    ProviderResponse,
    ProviderTokenUsage,
)
from sas_migrate.core.responses import (
    ResponseEnvelope,
    ResponseMode,
    TranslationCell,
    TranslationCellKind,
    TranslationDocument,
)
from sas_migrate.core.runs import ItemStatus, RunEvent, RunEventType, RunStatus
from sas_migrate.core.sas import SasBatch, SasChunk, SasChunkKind
from sas_migrate.core.targets import ResolvedTarget, TargetId, resolve_local_target
from sas_migrate.core.targets.validation import ResponseValidationResult
from sas_migrate.core.tokens import (
    CallTokenRecord,
    MessageRole,
    PromptAssembly,
    PromptComponentDraft,
    TokenBudgetPolicy,
    TokenCategory,
    TokenEstimator,
)


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 18, tzinfo=UTC)

    def now(self) -> datetime:
        result = self.value
        self.value += timedelta(seconds=1)
        return result


class _Events:
    def __init__(self) -> None:
        self.values: list[RunEvent] = []

    async def append(self, event: RunEvent) -> None:
        self.values.append(event)

    async def events(self, run_id: str, thread_id: str) -> tuple[RunEvent, ...]:
        return tuple(
            event
            for event in self.values
            if event.run_id == run_id and event.thread_id == thread_id
        )


class _Memory:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str, str], ResponseEnvelope] = {}
        self.remember_count = 0
        self.read_count = 0

    async def accepted_response(
        self, run_id: str, thread_id: str, item_id: str
    ) -> ResponseEnvelope | None:
        self.read_count += 1
        return self.values.get((run_id, thread_id, item_id))

    async def remember_accepted(
        self,
        run_id: str,
        thread_id: str,
        item_id: str,
        response: ResponseEnvelope,
    ) -> None:
        self.remember_count += 1
        self.values[(run_id, thread_id, item_id)] = response

    async def forget_accepted(
        self, run_id: str, thread_id: str, item_ids: tuple[str, ...]
    ) -> None:
        for item_id in item_ids:
            self.values.pop((run_id, thread_id, item_id), None)

    async def fork_accepted(
        self,
        source_run_id: str,
        source_thread_id: str,
        destination_run_id: str,
        destination_thread_id: str,
        item_ids: tuple[str, ...],
    ) -> None:
        for item_id in item_ids:
            value = self.values.get((source_run_id, source_thread_id, item_id))
            if value is not None:
                self.values[(destination_run_id, destination_thread_id, item_id)] = (
                    value
                )


class _TokenRecords:
    def __init__(self) -> None:
        self.values: list[CallTokenRecord] = []

    async def append(self, record: CallTokenRecord) -> None:
        self.values.append(record)

    async def records(self, run_id: str, thread_id: str) -> tuple[CallTokenRecord, ...]:
        return tuple(
            record
            for record in self.values
            if record.run_id == run_id and record.thread_id == thread_id
        )


class _Artifacts:
    def __init__(self) -> None:
        self.values: list[ArtifactWrite] = []

    async def write(self, run_id: str, artifact: ArtifactWrite) -> str:
        self.values.append(artifact)
        return f"memory://{run_id}/{artifact.artifact_id}"


class _LLM:
    def __init__(self, responses: list[ProviderResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[int, PromptAssembly, TargetId]] = []

    async def invoke(
        self,
        prompt: PromptAssembly,
        target: ResolvedTarget,
        *,
        attempt: int,
    ) -> ProviderResponse:
        self.calls.append((attempt, prompt, target.target))
        return self.responses.pop(0)


def _chunk(chunk_id: str, source_id: str) -> SasChunk:
    return SasChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        text="data output; run;",
        kind=SasChunkKind.DATA_STEP,
        start_line=1,
        end_line=1,
        start_char=0,
        end_char=17,
    )


def _envelope() -> ResponseEnvelope:
    target = resolve_local_target("sql")
    document = TranslationDocument(
        target=TargetId.SPARK_SQL,
        analysis="Preserve semantics.",
        cells=(
            TranslationCell(
                kind=TranslationCellKind.CODE,
                source="SELECT 1",
                language="sql",
                chunk_id="chunk-1",
            ),
        ),
    )
    return ResponseEnvelope(
        mode=ResponseMode.STRUCTURED,
        raw_message="structured",
        document=document,
        resolved_target=target,
        validation=ResponseValidationResult.accepted(TargetId.SPARK_SQL),
    )


def _record(run_id: str, thread_id: str, item_id: str) -> CallTokenRecord:
    return CallTokenRecord(
        run_id=run_id,
        thread_id=thread_id,
        item_id=item_id,
        attempt=1,
        target=TargetId.SPARK_SQL,
        estimator="test",
        encoding="test",
        estimated_input_by_category={TokenCategory.SAS_SOURCE: 10},
        estimated_input_total=10,
        accepted_attempt=True,
    )


def _translation_item(*, sources: tuple[str, ...] = ("one.sas",)) -> TranslationItem:
    chunks = tuple(
        _chunk(f"chunk-{index}", source)
        for index, source in enumerate(sources, start=1)
    )
    return TranslationItem.from_sas(
        SasBatch(
            batch_id="batch-001",
            chunks=list(chunks),
            source_files=list(sources),
        )
    )


def _prompt_builder() -> TranslationPromptBuilder:
    counter = TokenEstimator(
        encoding="characters",
        text_counter=len,
        estimator="test",
    )
    return TranslationPromptBuilder(PromptAssembler(counter))


def _document(target: TargetId) -> TranslationDocument:
    is_sql = target is TargetId.SPARK_SQL
    return TranslationDocument(
        target=target,
        analysis="Preserve row semantics.",
        cells=(
            TranslationCell(
                kind=TranslationCellKind.CODE,
                source="SELECT 1" if is_sql else "result = spark.range(1)",
                language="sql" if is_sql else "python",
                chunk_id="chunk-1",
            ),
        ),
    )


def _translation_service(
    llm: _LLM,
) -> tuple[TranslateCorpus, _Events, _Memory, _TokenRecords, _Artifacts]:
    events = _Events()
    memory = _Memory()
    records = _TokenRecords()
    artifacts = _Artifacts()
    clock = _Clock()
    counter = TokenEstimator(
        encoding="characters",
        text_counter=len,
        estimator="test",
    )
    assembler = PromptAssembler(counter)
    attempts = BudgetedResponseAttemptService(
        llm=llm,
        budgets=TokenBudgetEnforcer(assembler),
        accounting=TokenAccountingService(counter),
        audit=TokenAuditPersistenceService(records, artifacts),
    )
    service = TranslateCorpus(
        attempts=attempts,
        prompts=TranslationPromptBuilder(assembler),
        artifacts=TranslationArtifactService(artifacts),
        runs=RunStateService(
            events=events,
            memory=memory,
            token_records=records,
            clock=clock,
            event_id=iter(f"event-{index}" for index in range(100)).__next__,
        ),
        memory=memory,
        token_records=records,
    )
    return service, events, memory, records, artifacts


def _request(target: str, *, resume: bool = False) -> TranslateCorpusRequest:
    return TranslateCorpusRequest(
        run_id="run-e2e",
        thread_id="thread-e2e",
        target=resolve_local_target(target),
        items=(_translation_item(),),
        policy=TokenBudgetPolicy(
            max_input_tokens=100_000,
            reserved_output_tokens=2_000,
            safety_margin_tokens=500,
        ),
        max_attempts=2,
        resume=resume,
    )


def _service() -> tuple[RunStateService, _Events, _Memory, _TokenRecords]:
    events = _Events()
    memory = _Memory()
    records = _TokenRecords()
    service = RunStateService(
        events=events,
        memory=memory,
        token_records=records,
        clock=_Clock(),
        event_id=iter(f"event-{index}" for index in range(100)).__next__,
    )
    return service, events, memory, records


def test_translation_item_preserves_multi_source_attribution() -> None:
    item = TranslationItem.from_sas(
        SasBatch(
            batch_id="batch-001",
            chunks=[_chunk("chunk-1", "one.sas"), _chunk("chunk-2", "two.sas")],
            source_files=["one.sas", "two.sas"],
            reason="cross-file dataset dependency",
        )
    )
    assert item.source_files == ("one.sas", "two.sas")
    assert item.chunk_sources == {"chunk-1": "one.sas", "chunk-2": "two.sas"}
    assert item.known_chunk_ids == frozenset({"chunk-1", "chunk-2"})


def test_run_state_replays_attempt_and_completion_events() -> None:
    service, _, _, _ = _service()

    async def scenario() -> None:
        await service.start("run-1", "thread-1", resolve_local_target("sql"))
        await service.item_started("run-1", "thread-1", "item-1", 1)
        await service.attempt_completed(
            "run-1", "thread-1", "item-1", 1, valid=True, sent=True
        )
        await service.item_accepted("run-1", "thread-1", "item-1", 1)
        await service.completed("run-1", "thread-1")
        state = await service.state("run-1", "thread-1")
        assert state is not None
        assert state.status is RunStatus.COMPLETED
        assert state.items[0].status is ItemStatus.ACCEPTED
        assert state.items[0].attempt == 1

    asyncio.run(scenario())


def test_rewind_forgets_acceptance_and_reopens_completed_run() -> None:
    service, _, memory, _ = _service()

    async def scenario() -> None:
        await service.start("run-1", "thread-1", resolve_local_target("sql"))
        for item_id in ("item-1", "item-2"):
            await service.item_started("run-1", "thread-1", item_id, 1)
            await memory.remember_accepted("run-1", "thread-1", item_id, _envelope())
            await service.item_accepted("run-1", "thread-1", item_id, 1)
        await service.completed("run-1", "thread-1")
        affected = await service.rewind(
            "run-1", "thread-1", ("item-1", "item-2"), "item-2"
        )
        state = await service.state("run-1", "thread-1")
        assert affected == ("item-2",)
        assert state is not None and state.status is RunStatus.RUNNING
        assert [item.status for item in state.items] == [
            ItemStatus.ACCEPTED,
            ItemStatus.PENDING,
        ]
        assert await memory.accepted_response("run-1", "thread-1", "item-2") is None

    asyncio.run(scenario())


def test_fork_copies_accepted_prefix_and_marks_token_history_recovered() -> None:
    service, _, memory, records = _service()

    async def scenario() -> None:
        await service.start("run-1", "thread-1", resolve_local_target("sql"))
        for item_id in ("item-1", "item-2"):
            await service.item_started("run-1", "thread-1", item_id, 1)
            await memory.remember_accepted("run-1", "thread-1", item_id, _envelope())
            await records.append(_record("run-1", "thread-1", item_id))
            await service.item_accepted("run-1", "thread-1", item_id, 1)
        copied = await service.fork(
            source_run_id="run-1",
            source_thread_id="thread-1",
            destination_run_id="run-2",
            destination_thread_id="thread-2",
            ordered_item_ids=("item-1", "item-2"),
            upto_items=1,
        )
        state = await service.state("run-2", "thread-2")
        recovered = await records.records("run-2", "thread-2")
        assert copied == ("item-1",)
        assert state is not None and len(state.items) == 1
        assert state.items[0].status is ItemStatus.ACCEPTED
        assert len(recovered) == 1 and recovered[0].recovered
        assert await memory.accepted_response("run-2", "thread-2", "item-1")

    asyncio.run(scenario())


def test_prompt_builder_attributes_schema_context_sources_and_retry() -> None:
    item = _translation_item()
    target = resolve_local_target("sql")
    prompt = _prompt_builder().build(
        item,
        target,
        context=PromptContext(
            components=(
                PromptComponentDraft(
                    category=TokenCategory.PROJECT_INSTRUCTIONS,
                    text="Use catalog-qualified table names.",
                    message_role=MessageRole.SYSTEM,
                    source_id="project.md",
                ),
            )
        ),
        retry_feedback=("Return Spark SQL, not Python.",),
    )
    categories = [component.category for component in prompt.components]
    assert categories == [
        TokenCategory.SYSTEM_STATIC,
        TokenCategory.STRUCTURED_SCHEMA,
        TokenCategory.TARGET_DIRECTIVE,
        TokenCategory.PROJECT_INSTRUCTIONS,
        TokenCategory.BATCH_CONTEXT,
        TokenCategory.SAS_SOURCE,
        TokenCategory.RETRY_FEEDBACK,
        TokenCategory.CHAT_FRAMING,
    ]
    assert prompt.input_by_category()[TokenCategory.SAS_SOURCE] > 0
    assert "spark_sql" in prompt.render_messages()[0].content


def test_effective_prompt_golden_contains_attribution_and_provider_messages() -> None:
    item = _translation_item()
    target = resolve_local_target("sql")
    prompt = _prompt_builder().build(item, target)
    rendered = render_effective_prompt(item, target, 2, prompt)
    assert rendered.startswith("# Effective prompt: batch-001\n")
    assert "### 1. system_static" in rendered
    assert "### 4. batch_context" in rendered
    assert "## Provider messages" in rendered
    assert "- Attempt: `2`" in rendered


def test_notebooks_split_attributed_multi_source_translation() -> None:
    item = _translation_item(sources=("dir/one.sas", "other/two.sas"))
    document = TranslationDocument(
        target=TargetId.SPARK_SQL,
        analysis="Preserve joins.",
        cells=(
            TranslationCell(
                kind=TranslationCellKind.CODE,
                source="SELECT 1",
                language="sql",
                chunk_id="chunk-1",
            ),
            TranslationCell(
                kind=TranslationCellKind.MARKDOWN,
                source="Shared explanation.",
            ),
            TranslationCell(
                kind=TranslationCellKind.CODE,
                source="SELECT 2",
                language="sql",
                chunk_id="chunk-2",
            ),
        ),
    )
    notebooks = render_notebooks(
        (
            NotebookTranslation(
                item=item, target=resolve_local_target("sql"), document=document
            ),
        )
    )
    assert set(notebooks) == {"one", "two"}
    one_sources = [cell["source"] for cell in notebooks["one"]["cells"]]
    two_sources = [cell["source"] for cell in notebooks["two"]["cells"]]
    assert "SELECT 1" in one_sources and "SELECT 2" not in one_sources
    assert "SELECT 2" in two_sources and "SELECT 1" not in two_sources
    assert "Shared explanation." in one_sources and "Shared explanation." in two_sources


def test_notebooks_use_cross_file_fallback_and_python_host_for_mixed_code() -> None:
    item = _translation_item(sources=("one.sas", "two.sas"))
    document = TranslationDocument(
        target=TargetId.SPARK_SQL,
        analysis="Mixed fallback.",
        cells=(
            TranslationCell(
                kind=TranslationCellKind.CODE,
                source="SELECT 1",
                language="sql",
            ),
            TranslationCell(
                kind=TranslationCellKind.CODE,
                source="df.count()",
                language="python",
            ),
        ),
    )
    notebooks = render_notebooks(
        (
            NotebookTranslation(
                item=item, target=resolve_local_target("sql"), document=document
            ),
        )
    )
    assert set(notebooks) == {"_cross_file", "one", "two"}
    cross = notebooks["_cross_file"]
    assert cross["metadata"]["kernelspec"]["name"] == "python3"
    assert any(
        str(cell["source"]).startswith("%sql\nSELECT 1") for cell in cross["cells"]
    )
    assert "`_cross_file.ipynb`" in notebooks["one"]["cells"][0]["source"]


def test_artifact_service_persists_attempt_canonical_and_notebooks() -> None:
    repository = _Artifacts()
    service = TranslationArtifactService(repository)
    item = _translation_item()
    target = resolve_local_target("sql")
    prompt = _prompt_builder().build(item, target)
    envelope = _envelope()
    document = envelope.document
    assert document is not None

    async def scenario() -> None:
        attempt = await service.persist_attempt(
            "run-1", item, target, 1, prompt, envelope
        )
        canonical = await service.persist_canonical(
            "run-1", item.item_id, document
        )
        notebooks = await service.persist_notebooks(
            "run-1",
            (
                NotebookTranslation(
                    item=item,
                    target=target,
                    document=document,
                ),
            ),
        )
        assert len(attempt) == 2
        assert canonical.kind == "canonical_translation"
        assert len(notebooks) == 1

    asyncio.run(scenario())
    assert [artifact.metadata["kind"] for artifact in repository.values] == [
        "effective_prompt",
        "response_envelope",
        "canonical_translation",
        "notebook",
    ]


@pytest.mark.parametrize(
    ("requested", "target"),
    [("sql", TargetId.SPARK_SQL), ("pyspark", TargetId.PYSPARK)],
)
def test_translate_corpus_end_to_end_for_both_targets(
    requested: str,
    target: TargetId,
) -> None:
    llm = _LLM(
        [
            ProviderResponse(
                raw_message="structured",
                structured_document=_document(target),
                usage=ProviderTokenUsage(input_tokens=500, output_tokens=50),
            )
        ]
    )
    service, _, memory, records, artifacts = _translation_service(llm)
    outcome = asyncio.run(service.run(_request(requested)))
    assert outcome.state.status is RunStatus.COMPLETED
    assert outcome.items[0].status is ItemStatus.ACCEPTED
    assert outcome.items[0].target.target is target
    assert len([record for record in records.values if record.accepted_attempt]) == 1
    assert memory.remember_count == 1
    assert [call[2] for call in llm.calls] == [target]
    assert {artifact.metadata["kind"] for artifact in artifacts.values} >= {
        "effective_prompt",
        "response_envelope",
        "canonical_translation",
        "notebook",
        "token_call_audit",
    }


def test_translate_corpus_retries_with_feedback_and_accepts_raw_fallback() -> None:
    llm = _LLM(
        [
            ProviderResponse(
                raw_message="wrong target",
                structured_document=_document(TargetId.PYSPARK),
                usage=ProviderTokenUsage(input_tokens=500, output_tokens=50),
            ),
            ProviderResponse(
                raw_message=(
                    "## Analysis\n\nPreserve semantics.\n\n"
                    "## Mapping\n\n_No construct mapping reported._\n\n"
                    "## Translation\n\n```sql\nSELECT 1\n```\n\n"
                    "## Risks\n\n_No risks flagged._"
                ),
                structured_error="gateway does not support response schemas",
                usage=ProviderTokenUsage(input_tokens=550, output_tokens=60),
            ),
        ]
    )
    service, _, memory, records, _ = _translation_service(llm)
    outcome = asyncio.run(service.run(_request("sql")))
    assert outcome.state.status is RunStatus.COMPLETED
    assert outcome.items[0].attempts == 2
    assert outcome.items[0].document is not None
    assert [record.accepted_attempt for record in records.values] == [False, True]
    assert memory.remember_count == 1
    retry_categories = {
        component.category for component in llm.calls[1][1].components
    }
    assert TokenCategory.RETRY_FEEDBACK in retry_categories


def test_translate_corpus_resume_uses_memory_without_new_model_turn() -> None:
    llm = _LLM(
        [
            ProviderResponse(
                raw_message="structured",
                structured_document=_document(TargetId.SPARK_SQL),
                usage=ProviderTokenUsage(input_tokens=500, output_tokens=50),
            )
        ]
    )
    service, events, memory, records, _ = _translation_service(llm)
    first = asyncio.run(service.run(_request("sql")))
    resumed = asyncio.run(service.run(_request("sql", resume=True)))
    assert first.state.status is RunStatus.COMPLETED
    assert resumed.state.status is RunStatus.COMPLETED
    assert resumed.items[0].recovered
    assert len(llm.calls) == 1
    assert memory.remember_count == 1
    assert len(records.values) == 1
    assert resumed.tokens.records[0].recovered
    assert resumed.tokens.current_run_total_tokens == 0
    assert sum(
        event.event_type is RunEventType.RUN_COMPLETED for event in events.values
    ) == 1
