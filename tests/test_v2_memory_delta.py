"""Phase 6 Delta memory adapter and maintenance contract coverage."""

from __future__ import annotations

import asyncio
import os
import pathlib
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from sas_migrate.adapters.memory import (
    MIN_VACUUM_HOURS,
    DeltaMemoryMaintenance,
    DeltaMemoryRepository,
    VacuumPolicy,
    quoted_table_name,
)
from sas_migrate.application.memory import (
    ConversationMemoryService,
    PolicyInstruction,
    RollingSummary,
    TaskPolicySnapshot,
    ThreadNote,
)


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 19, tzinfo=UTC)

    def now(self) -> datetime:
        current = self.value
        self.value += timedelta(seconds=1)
        return current


class _Store:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.cdf_consumers: list[str] = []

    def set(
        self,
        key: str,
        value: Any,
        tags: list[str] | None = None,
        source: str | None = None,
    ) -> None:
        del tags, source
        self.values[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def delete(self, key: str) -> bool:
        return self.values.pop(key, None) is not None

    def delete_many(self, keys: list[str]) -> int:
        return sum(self.delete(key) for key in keys)

    def keys(self, prefix: str = "") -> list[str]:
        return sorted(key for key in self.values if key.startswith(prefix))

    def all_records(self, prefix: str = "") -> list[tuple[str, dict[str, Any]]]:
        return [
            (key, {"value": self.values[key]})
            for key in self.keys(prefix)
        ]

    def sync_cdf(self, consumer_id: str) -> Any:
        self.cdf_consumers.append(consumer_id)
        return SimpleNamespace(baseline=False, checkpoint_version=3, events=())


class _Spark:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def sql(self, command: str) -> Any:
        self.commands.append(command)
        if command.startswith("DESCRIBE HISTORY"):
            return SimpleNamespace(
                first=lambda: SimpleNamespace(version=7, operation="MERGE")
            )
        if command.startswith("DESCRIBE DETAIL"):
            return SimpleNamespace(
                first=lambda: SimpleNamespace(sizeInBytes=1024, numFiles=2)
            )
        return SimpleNamespace(collect=lambda: ["candidate"])


def _repository() -> tuple[_Clock, _Store, DeltaMemoryRepository]:
    clock = _Clock()
    store = _Store()
    identifiers = iter(f"id-{index}" for index in range(1_000))
    repository = DeltaMemoryRepository(
        store,
        clock,
        identifier=identifiers.__next__,
    )
    return clock, store, repository


def test_delta_repository_persists_contracts_across_adapter_instances() -> None:
    clock, store, repository = _repository()
    identifiers = iter(f"service-{index}" for index in range(100))
    history = ConversationMemoryService(
        repository,
        clock,
        identifier=identifiers.__next__,
    )

    async def scenario() -> None:
        await history.record_accepted_turn(
            thread_id="thread-1",
            chat_id="chat-1",
            item_id="item-1",
            user_content="convert proc sql",
            assistant_content="converted",
        )
        await repository.put_policy(
            TaskPolicySnapshot(
                task_id="task-1",
                version=1,
                instructions=(PolicyInstruction(instruction_id="p1", text="safe"),),
                updated_at=clock.now(),
            )
        )
        await repository.put_note(
            ThreadNote(
                note_id="note-1",
                thread_id="thread-1",
                text="remember source assumptions",
                created_at=clock.now(),
            )
        )
        await repository.put_summary(
            RollingSummary(
                thread_id="thread-1",
                content="one completed turn",
                through_sequence=2,
                token_count=4,
                updated_at=clock.now(),
            )
        )

        reopened = DeltaMemoryRepository(store, clock)
        assert len(await reopened.messages("thread-1")) == 2
        assert (await reopened.policy("task-1")).version == 1  # type: ignore[union-attr]
        assert len(await reopened.notes("thread-1", now=clock.now())) == 1
        assert (await reopened.summary("thread-1")).through_sequence == 2  # type: ignore[union-attr]
        assert len(await reopened.audit_events("thread-1")) >= 4

    asyncio.run(scenario())


def test_delta_repository_supports_snapshot_rewind_fork_prune_and_cdf() -> None:
    clock, store, repository = _repository()
    identifiers = iter(f"service-{index}" for index in range(100))
    history = ConversationMemoryService(
        repository,
        clock,
        identifier=identifiers.__next__,
    )

    async def scenario() -> None:
        await history.record_accepted_turn(
            thread_id="src",
            chat_id="chat",
            item_id="item",
            user_content="one",
            assistant_content="two",
        )
        snapshot = await repository.snapshot("src")
        assert await repository.fork_thread("src", "dst") == 2
        assert await repository.rewind_messages("src", after_sequence=0) == 2
        await repository.restore(snapshot)
        assert len(await repository.messages("src")) == 2
        assert await repository.prune(before=clock.now()) >= 4

    asyncio.run(scenario())
    result = repository.sync_cdf("memory-worker")
    assert result.checkpoint_version == 3
    assert store.cdf_consumers == ["memory-worker"]


def test_delta_maintenance_validates_identifiers_retention_and_commands() -> None:
    assert quoted_table_name("main.memory.history") == "`main`.`memory`.`history`"
    with pytest.raises(ValueError):
        quoted_table_name("main.memory; DROP TABLE history")
    with pytest.raises(ValueError, match="between"):
        VacuumPolicy(MIN_VACUUM_HOURS - 1)
    with pytest.raises(ValueError, match="must exceed"):
        VacuumPolicy(MIN_VACUUM_HOURS, MIN_VACUUM_HOURS)

    spark = _Spark()
    maintenance = DeltaMemoryMaintenance(
        spark,
        "main.memory.history",
        policy=VacuumPolicy(MIN_VACUUM_HOURS),
    )
    assert maintenance.status()["latest_version"] == 7
    maintenance.optimize()
    assert maintenance.vacuum() == ["candidate"]
    assert spark.commands[-2] == "OPTIMIZE `main`.`memory`.`history`"
    assert spark.commands[-1].endswith("168 HOURS DRY RUN")


def test_delta_adapter_import_does_not_eagerly_import_spark() -> None:
    code = """
import sys
from sas_migrate.adapters.memory.delta import DeltaMemoryRepository
assert DeltaMemoryRepository
assert 'pyspark' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=pathlib.Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": str(SRC)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_real_delta_memory_adapter_contract(delta_spark: Any) -> None:
    """Run in the dedicated CI job with delta-spark installed; never mocked."""

    suffix = uuid4().hex
    table = f"v2_memory_{suffix}"
    audit = f"v2_memory_audit_{suffix}"
    clock = _Clock()
    repository = DeltaMemoryRepository.from_delta(
        delta_spark,
        table,
        clock,
        audit_table=audit,
    )
    history = ConversationMemoryService(repository, clock)

    async def scenario() -> None:
        await history.record_accepted_turn(
            thread_id="delta-thread",
            chat_id="delta-chat",
            item_id="delta-item",
            user_content="source",
            assistant_content="translation",
        )
        reopened = DeltaMemoryRepository.from_delta(
            delta_spark,
            table,
            clock,
            audit_table=audit,
        )
        assert len(await reopened.messages("delta-thread")) == 2
        baseline = reopened.sync_cdf("phase-6-test")
        assert baseline.checkpoint_version is not None

    try:
        asyncio.run(scenario())
    finally:
        delta_spark.sql(f"DROP TABLE IF EXISTS `{table}`")
        delta_spark.sql(f"DROP TABLE IF EXISTS `{audit}`")
