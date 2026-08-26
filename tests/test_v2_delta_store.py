"""Offline failure-path contracts for native v2 Delta persistence."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from sas_migrate.adapters.memory.delta_store import DeltaKVStore


def _bare_store(**attributes: Any) -> DeltaKVStore:
    store = object.__new__(DeltaKVStore)
    for name, value in attributes.items():
        setattr(store, name, value)
    return store


def test_delta_store_validates_session_and_retry_count() -> None:
    with pytest.raises(ValueError, match="active SparkSession"):
        DeltaKVStore(None, "memory", audit_table="memory_audit")
    with pytest.raises(ValueError, match="max_write_retries"):
        DeltaKVStore(
            SimpleNamespace(),
            "memory",
            audit_table="memory_audit",
            max_write_retries=-1,
        )


def test_cdf_property_probe_is_explicit_and_failure_safe() -> None:
    unavailable = _bare_store(
        _spark=SimpleNamespace(sql=lambda _query: (_ for _ in ()).throw(RuntimeError("no metadata"))),
        _table="`memory`",
    )
    assert unavailable._cdf_enabled() is False

    class _Result:
        def collect(self) -> list[dict[str, str]]:
            return [
                {"key": "unrelated", "value": "true"},
                {"key": "delta.enableChangeDataFeed", "value": " TRUE "},
            ]

    enabled = _bare_store(
        _spark=SimpleNamespace(sql=lambda _query: _Result()),
        _table="`memory`",
    )
    assert enabled._cdf_enabled() is True


def test_schema_validation_reports_every_incompatible_field() -> None:
    string_type = SimpleNamespace(simpleString=lambda: "string")
    wrong_type = SimpleNamespace(simpleString=lambda: "bigint")
    schema = [
        SimpleNamespace(name="kv_key", dataType=string_type),
        SimpleNamespace(name="value", dataType=wrong_type),
    ]
    store = _bare_store(
        _spark=SimpleNamespace(table=lambda _table: SimpleNamespace(schema=schema)),
        _table="`memory`",
        _table_name="memory",
    )
    with pytest.raises(RuntimeError, match="incompatible memory schema"):
        store._validate_schema()


def test_idempotent_writes_retry_only_delta_conflicts() -> None:
    delays: list[float] = []
    store = _bare_store(
        _max_write_retries=2,
        _sleeper=delays.append,
    )
    attempts = 0

    def conflict_then_succeed() -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("concurrent Delta commit conflict")

    store._write("MERGE", conflict_then_succeed)
    assert attempts == 3
    assert delays == [0.05, 0.1]
    assert store._is_concurrent_write(RuntimeError("DELTA_X COMMIT failed"))
    assert not store._is_concurrent_write(RuntimeError("permission denied"))

    with pytest.raises(RuntimeError, match="permission denied"):
        store._write("MERGE", lambda: (_ for _ in ()).throw(RuntimeError("permission denied")))

    exhausted = _bare_store(_max_write_retries=0, _sleeper=delays.append)
    with pytest.raises(RuntimeError, match="conflict"):
        exhausted._write(
            "MERGE",
            lambda: (_ for _ in ()).throw(RuntimeError("concurrent conflict")),
        )


def test_empty_delete_and_invalid_consumer_do_not_touch_spark() -> None:
    store = _bare_store()
    assert store.delete_many([]) == 0
    with pytest.raises(ValueError, match="consumer_id"):
        store.sync_cdf(" ")


def test_history_and_operation_diagnostics_handle_empty_history() -> None:
    class _Query:
        def __init__(self, row: Any) -> None:
            self._row = row

        def first(self) -> Any:
            return self._row

    detail = SimpleNamespace(sizeInBytes=256, numFiles=2)

    def sql(query: str) -> _Query:
        return _Query(detail if "DETAIL" in query else None)

    store = _bare_store(
        _spark=SimpleNamespace(sql=sql),
        _table="`memory`",
        _table_name="memory",
        _audit_table_name="memory_audit",
    )
    assert store._latest_version() == 0
    assert store.operations() == {
        "table": "memory",
        "latest_version": None,
        "last_operation": None,
        "size_in_bytes": 256,
        "num_files": 2,
        "audit_table": "memory_audit",
    }
