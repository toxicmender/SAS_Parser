"""Offline tests for memory._delta — the shared retry, conflict
classification, temp-view, probe and status helpers (no Spark needed)."""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from memory import _delta


@pytest.mark.parametrize(
    "text",
    [
        "io.delta.exceptions.ConcurrentAppendException: Files were added",
        "[DELTA_CONCURRENT_APPEND] ConcurrentAppendException",
        "[DELTA_CONCURRENT_DELETE_READ] ...",
        "io.delta.exceptions.MetadataChangedException: The metadata ...",
        "[DELTA_PROTOCOL_CHANGED] ProtocolChangedException",
    ],
)
def test_delta_conflicts_are_retryable(text):
    assert _delta.is_retryable_conflict(RuntimeError(text))


@pytest.mark.parametrize(
    "text",
    [
        "Merge conflict in my_file.py",  # the old "CONFLICT" substring match
        "[PARSE_SYNTAX_ERROR] Syntax error at or near 'X'",
        "PERMISSION_DENIED: User does not have MODIFY",
        "[TABLE_OR_VIEW_ALREADY_EXISTS] Cannot create table",
    ],
)
def test_other_errors_are_not_write_conflicts(text):
    assert not _delta.is_retryable_conflict(RuntimeError(text))


def test_lost_create_races_are_retryable_ddl():
    assert _delta.is_retryable_ddl(RuntimeError("[TABLE_OR_VIEW_ALREADY_EXISTS] t"))
    assert _delta.is_retryable_ddl(RuntimeError("ProtocolChangedException"))
    assert not _delta.is_retryable_ddl(RuntimeError("PERMISSION_DENIED"))


def test_an_error_with_an_empty_message_is_classified_by_its_condition():
    class EmptyMessage(Exception):
        def __str__(self) -> str:
            return ""

        def getCondition(self) -> str:
            return "DELTA_CREATE_TABLE_WITH_NON_EMPTY_LOCATION"

    assert _delta.is_retryable_ddl(EmptyMessage())
    assert not _delta.is_retryable_conflict(EmptyMessage())


def test_with_retry_retries_conflicts_then_gives_up(monkeypatch):
    monkeypatch.setattr(_delta.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("ConcurrentAppendException")
        return "ok"

    assert _delta.with_retry("op", flaky, max_retries=3) == "ok"
    assert calls["n"] == 3

    calls["n"] = 0

    def always():
        calls["n"] += 1
        raise RuntimeError("ConcurrentAppendException")

    with pytest.raises(RuntimeError):
        _delta.with_retry("op", always, max_retries=2)
    assert calls["n"] == 3


def test_with_retry_does_not_retry_other_errors():
    calls = {"n": 0}

    def broken():
        calls["n"] += 1
        raise ValueError("schema mismatch")

    with pytest.raises(ValueError):
        _delta.with_retry("op", broken, max_retries=5)
    assert calls["n"] == 1


class _FakeSpark:
    def __init__(self, probe_ok: bool) -> None:
        self.probe_ok = probe_ok
        self.queries: list[str] = []
        self.dropped: list[str] = []
        self.sparkContext = type("SC", (), {"applicationId": f"app-{id(self)}"})()
        self.catalog = type("Cat", (), {"dropTempView": lambda _s, v: self.dropped.append(v)})()

    def sql(self, query: str):
        self.queries.append(query)
        if "isearch" in query and not self.probe_ok:
            raise RuntimeError("UNRESOLVED_ROUTINE isearch")
        return type("DF", (), {"collect": lambda _s: []})()


def test_search_probe_runs_once_per_application():
    spark = _FakeSpark(probe_ok=False)
    assert _delta.probe_search_functions(spark) is False
    assert _delta.probe_search_functions(spark) is False
    assert len(spark.queries) == 1


def test_temp_view_is_dropped_even_on_error():
    spark = _FakeSpark(probe_ok=True)
    registered: list[str] = []
    df = type("DF", (), {"createOrReplaceTempView": lambda _s, v: registered.append(v)})()
    with pytest.raises(RuntimeError):
        with _delta.temp_view(spark, df, "stem") as view:
            raise RuntimeError("boom")
    assert registered == [view] == spark.dropped
    assert view.startswith("_mem_stem_")
