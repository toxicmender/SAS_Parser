"""Phase 5 translation item and event-sourced run-control tests."""

from __future__ import annotations

import asyncio
import pathlib
import sys
from datetime import UTC, datetime, timedelta

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from sas_migrate.application import RunStateService, TranslationItem
from sas_migrate.core.responses import (
    ResponseEnvelope,
    ResponseMode,
    TranslationCell,
    TranslationCellKind,
    TranslationDocument,
)
from sas_migrate.core.runs import ItemStatus, RunEvent, RunStatus
from sas_migrate.core.sas import SasBatch, SasChunk, SasChunkKind
from sas_migrate.core.targets import TargetId, resolve_local_target
from sas_migrate.core.targets.validation import ResponseValidationResult
from sas_migrate.core.tokens import CallTokenRecord, TokenCategory


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

    async def accepted_response(
        self, run_id: str, thread_id: str, item_id: str
    ) -> ResponseEnvelope | None:
        return self.values.get((run_id, thread_id, item_id))

    async def remember_accepted(
        self,
        run_id: str,
        thread_id: str,
        item_id: str,
        response: ResponseEnvelope,
    ) -> None:
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
                self.values[(destination_run_id, destination_thread_id, item_id)] = value


class _TokenRecords:
    def __init__(self) -> None:
        self.values: list[CallTokenRecord] = []

    async def append(self, record: CallTokenRecord) -> None:
        self.values.append(record)

    async def records(
        self, run_id: str, thread_id: str
    ) -> tuple[CallTokenRecord, ...]:
        return tuple(
            record
            for record in self.values
            if record.run_id == run_id and record.thread_id == thread_id
        )


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
            await memory.remember_accepted(
                "run-1", "thread-1", item_id, _envelope()
            )
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
            await memory.remember_accepted(
                "run-1", "thread-1", item_id, _envelope()
            )
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
