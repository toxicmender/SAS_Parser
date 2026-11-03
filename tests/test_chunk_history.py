"""
Tests for memory.chunk_history (original-vs-converted chunk history) and the
SQL builders in memory.schema.

``ChunkHistoryContract`` runs against the in-memory backend always and against
a real local Delta table where pyspark + delta-spark + a JVM are present (the
shared ``delta_spark`` fixture in conftest.py). The Delta half also proves the
Databricks-only setup (predictive optimization, CREATE SEARCH INDEX) degrades
to a logged warning on OSS Delta, with searches answered by the LIKE fallback.
"""

from __future__ import annotations

import pathlib
import sys
import uuid

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chunker.models import SasBatch, SasChunk, SasChunkKind
from memory import schema
from memory.chunk_history import ChunkHistory, chunk_key, converted_text_for


def _chunk(chunk_id: str, text: str, source: str = "a.sas", start: int = 1) -> SasChunk:
    return SasChunk(
        chunk_id=chunk_id,
        source_id=source,
        text=text,
        kind=next(iter(SasChunkKind)),
        start_line=start,
        end_line=start,
        start_char=start * 10,
        end_char=start * 10 + len(text),
    )


def _batch(batch_id: str, *chunks: SasChunk) -> SasBatch:
    return SasBatch(batch_id=batch_id, chunks=list(chunks), reason="test")


SORT = "proc sort data=work.claims; by member_id; run;"
STEP = "data work.out; set work.claims; paid = amount * 2; run;"


def _document(*cells: tuple[str | None, str]) -> dict:
    return {
        "analysis": "x",
        "mapping": [],
        "risks": [],
        "cells": [
            {"kind": "code", "source": src, "chunk_id": cid, "comment": ""}
            for cid, src in cells
        ],
    }


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_chunk_key_is_content_addressed_per_source():
    assert chunk_key("a.sas", SORT) == chunk_key("a.sas", SORT)
    assert chunk_key("a.sas", SORT) != chunk_key("b.sas", SORT)
    assert chunk_key(None, SORT) == chunk_key("<inline>", SORT)


def test_converted_text_prefers_cells_attributed_to_the_chunk():
    doc = _document(("c1", "df1 = 1"), ("c2", "df2 = 2"), ("c1", "df1b = 3"))
    assert converted_text_for("c1", document=doc, response="", single_member=False) == (
        "df1 = 1\n\ndf1b = 3"
    )
    # Multi-member item, nothing attributed: no guess.
    assert converted_text_for("c9", document=doc, response="", single_member=False) is None


def test_converted_text_for_a_single_member_needs_no_attribution():
    doc = _document((None, "a = 1"), (None, "b = 2"))
    assert converted_text_for("c1", document=doc, response="", single_member=True) == (
        "a = 1\n\nb = 2"
    )
    md = "## Analysis\nx\n\n```python\nspark.sql('x')\n```\n"
    assert converted_text_for("c1", document=None, response=md, single_member=True) == (
        "spark.sql('x')"
    )
    assert converted_text_for("c1", document=None, response="plain", single_member=True) == (
        "plain"
    )


def test_sql_string_literal_cannot_escape_the_literal():
    assert schema.sql_string_literal("a'b") == "'a\\'b'"
    assert schema.sql_string_literal("a\\'b") == "'a\\\\\\'b'"
    assert schema.sql_string_literal("x\n;DROP") == "'x\\n;DROP'"


def test_search_sql_picks_the_function_and_mode():
    assert schema.search_sql(("value",), "Proc") == (
        "isearch(`value`, 'Proc', mode => 'substring')"
    )
    assert schema.search_sql(("a", "b"), "x y", case_sensitive=True, mode="word") == (
        "search(`a`, `b`, 'x y', mode => 'word')"
    )
    with pytest.raises(ValueError):
        schema.search_sql(("value; drop",), "x")
    with pytest.raises(ValueError):
        schema.search_sql(("value",), "x", mode="regex")  # type: ignore[arg-type]


def test_table_specs_respect_databricks_search_index_limits():
    for spec in schema.ALL_TABLES:
        assert len(spec.search_indexes) <= 4
        names = spec.column_names
        for index in spec.search_indexes:
            assert index.column in names
            assert dict(spec.columns)[index.column] == "STRING"
        ddl = spec.ddl("main.mem.llm")
        assert "'delta.enableRowTracking' = 'true'" in ddl
        assert "CLUSTER BY" in ddl
        assert ("enableChangeDataFeed" in ddl) == spec.cdf
    index_names = [
        spec.index_name("main.mem.llm", i.column)
        for spec in schema.ALL_TABLES
        for i in spec.search_indexes
    ]
    table_names = {f"`main`.`mem`.`{s.name('llm')}`" for s in schema.ALL_TABLES}
    assert len(set(index_names)) == len(index_names)
    assert not set(index_names) & table_names
    assert all(n.startswith("`main`.`mem`.") for n in index_names)


class _FakeSpark:
    """Records statements; rejects qualified index names like a runtime
    that only takes a bare one."""

    def __init__(self, reject_qualified: bool, error: str = "PARSE_SYNTAX_ERROR") -> None:
        self.sql_log: list[str] = []
        self.reject_qualified = reject_qualified
        self.error = error

    def sql(self, text: str):
        self.sql_log.append(text)
        if self.reject_qualified and "INDEX" in text and "`main`.`mem`.`llm_kv_" in text:
            raise RuntimeError(f"[{self.error}] syntax error at or near '.'")
        if text.startswith("SELECT current_catalog()"):
            row = {"c": "prev_cat", "s": "prev_schema"}
            return type("R", (), {"first": lambda self: row})()
        return "ran"


def test_index_sql_uses_the_qualified_name_when_accepted():
    spark = _FakeSpark(reject_qualified=False)
    result, style = schema.run_index_sql(spark, schema.KV, "main.mem.llm", "value", "DROP INDEX {name}")
    assert (result, style) == ("ran", "qualified")
    assert spark.sql_log == ["DROP INDEX `main`.`mem`.`llm_kv_value_fts`"]


def test_index_sql_falls_back_to_a_bare_name_in_a_restored_use_scope():
    spark = _FakeSpark(reject_qualified=True)
    _, style = schema.run_index_sql(spark, schema.KV, "main.mem.llm", "value", "DROP INDEX {name}")
    assert style == "bare"
    assert spark.sql_log[1:] == [
        "SELECT current_catalog() AS c, current_schema() AS s",
        "USE CATALOG `main`",
        "USE SCHEMA `mem`",
        "DROP INDEX `llm_kv_value_fts`",
        "USE CATALOG `prev_cat`",
        "USE SCHEMA `prev_schema`",
    ]


def test_index_sql_does_not_retry_other_errors():
    spark = _FakeSpark(reject_qualified=True, error="PERMISSION_DENIED")
    with pytest.raises(RuntimeError, match="PERMISSION_DENIED"):
        schema.run_index_sql(spark, schema.KV, "main.mem.llm", "value", "DROP INDEX {name}")
    assert len(spark.sql_log) == 1


def test_too_many_indexes_are_rejected():
    with pytest.raises(ValueError, match="at most four"):
        schema.TableSpec(
            suffix="x",
            columns=tuple((f"c{i}", "STRING") for i in range(5)),
            cluster_by=("c0",),
            search_indexes=tuple(schema.SearchIndex(f"c{i}") for i in range(5)),
        )


# ---------------------------------------------------------------------------
# Backend contract
# ---------------------------------------------------------------------------


class ChunkHistoryContract:
    def make(self) -> ChunkHistory:
        raise NotImplementedError

    def test_every_attempt_is_kept_and_only_the_last_is_final(self):
        h = self.make()
        item = _batch("batch-001", _chunk("c1", SORT, start=1), _chunk("c2", STEP, start=2))
        h.record_attempt(
            item, "first try", _document(("c1", "bad_sort()")),
            thread_id="t1", attempt=1, is_final=False,
            validation_score=0.2, validation_passed=False,
            target_language="pyspark", model="m",
        )
        h.record_attempt(
            item, "second try", _document(("c1", "df.orderBy('member_id')"), ("c2", "df2 = df")),
            thread_id="t1", attempt=2, is_final=True,
            validation_score=0.9, validation_passed=True,
            target_language="pyspark", model="m", chat_id="chat-1",
        )
        rows = h.for_item("t1", "batch-001")
        assert [(r["attempt"], r["chunk_id"]) for r in rows] == [
            (1, "c1"), (1, "c2"), (2, "c1"), (2, "c2"),
        ]
        assert [r["is_final"] for r in rows] == [False, False, True, True]
        # Original and converted chunk sit on the same row.
        final_c1 = [r for r in rows if r["is_final"] and r["chunk_id"] == "c1"][0]
        assert final_c1["original_text"] == SORT
        assert final_c1["converted_text"] == "df.orderBy('member_id')"
        assert final_c1["response"] == "second try"
        assert final_c1["document"]["cells"][0]["chunk_id"] == "c1"
        assert final_c1["validation_passed"] is True
        assert final_c1["chat_id"] == "chat-1"
        rejected_c2 = [r for r in rows if not r["is_final"] and r["chunk_id"] == "c2"][0]
        assert rejected_c2["converted_text"] is None
        assert rejected_c2["original_text"] == STEP

        finals = h.for_item("t1", "batch-001", final_only=True)
        assert {r["attempt"] for r in finals} == {2}

    def test_current_only_keeps_the_answer_that_stands_after_a_redo(self):
        h = self.make()
        item = _batch("batch-001", _chunk("c1", SORT))
        h.record_attempt(item, "first run", None, thread_id="t1", attempt=1, is_final=True)
        # A later resume re-translates the item; both answers were accepted.
        h.record_attempt(item, "redo rejected", None, thread_id="t1", attempt=1, is_final=False)
        h.record_attempt(item, "redo kept", None, thread_id="t1", attempt=2, is_final=True)
        h.record_attempt(_batch("batch-002", _chunk("c2", STEP, start=2)), "other", None,
                         thread_id="t1", attempt=1, is_final=True)
        assert [r["response"] for r in h.for_item("t1", "batch-001", final_only=True)] == [
            "first run", "redo kept",
        ]
        assert [r["response"] for r in h.for_item("t1", "batch-001", current_only=True)] == [
            "redo kept",
        ]
        assert sorted(r["response"] for r in h.history(thread_id="t1", current_only=True)) == [
            "other", "redo kept",
        ]

    def test_original_text_is_stored_once_per_distinct_chunk(self):
        h = self.make()
        item = _batch("batch-001", _chunk("c1", SORT))
        for attempt in (1, 2, 3):
            h.record_attempt(
                item, f"r{attempt}", None,
                thread_id="t1", attempt=attempt, is_final=attempt == 3,
            )
        # A rerun on another thread maps onto the same stored original.
        h.record_attempt(item, "again", None, thread_id="t2", attempt=1, is_final=True)
        assert len(h.sources()) == 1
        assert h.sources()[0]["original_text"] == SORT
        assert len(h.history(chunk_key=chunk_key("a.sas", SORT))) == 4
        assert len(h.history(chunk_id="c1", thread_id="t2")) == 1

    def test_search_original_converted_and_response(self):
        h = self.make()
        h.record_attempt(
            _batch("batch-001", _chunk("c1", SORT)),
            "Uses orderBy", _document((None, "claims.orderBy('member_id')")),
            thread_id="t1", attempt=1, is_final=True,
        )
        h.record_attempt(
            _batch("batch-002", _chunk("c2", STEP, start=2)),
            "Uses withColumn", _document((None, "out = claims.withColumn('paid', F.col('amount') * 2)")),
            thread_id="t1", attempt=1, is_final=True,
        )
        assert [r["chunk_id"] for r in h.search("PROC SORT")] == ["c1"]
        assert h.search("PROC SORT", case_sensitive=True) == []
        assert [r["chunk_id"] for r in h.search("withColumn", field="converted")] == ["c2"]
        assert [r["chunk_id"] for r in h.search("orderby", field="response")] == ["c1"]
        # Word mode: every word, any order.
        assert [r["chunk_id"] for r in h.search("run claims proc", mode="word")] == ["c1"]
        assert h.search("run claims missingword", mode="word") == []
        # Quotes in a pattern are data, not SQL.
        assert [r["chunk_id"] for r in h.search("('member_id')", field="converted")] == ["c1"]
        with pytest.raises(ValueError):
            h.search("x", field="nope")  # type: ignore[arg-type]

    # ---- feedback + buffering ----------------------------------------------

    def test_feedback_is_kept_on_the_attempt_it_prompted(self):
        h = self.make()
        item = _batch("b", _chunk("c1", SORT))
        h.record_attempt(item, "r1", None, thread_id="t", attempt=1, is_final=False)
        h.record_attempt(item, "r2", None, thread_id="t", attempt=2, is_final=True,
                         feedback="coverage failed: missing BY")
        rows = h.for_item("t", "b")
        assert [r["feedback"] for r in rows] == [None, "coverage failed: missing BY"]

    def test_buffered_attempts_land_together_on_the_final_write(self):
        h = self.make()
        item = _batch("b", _chunk("c1", SORT))
        h.record_attempt(item, "r1", None, thread_id="t", attempt=1, is_final=False, buffer=True)
        assert h.pending == 1
        assert h.for_item("t", "b") == []  # not readable until flushed
        h.record_attempt(item, "r2", None, thread_id="t", attempt=2, is_final=True)
        assert h.pending == 0
        assert [r["attempt"] for r in h.for_item("t", "b")] == [1, 2]
        h.record_attempt(item, "r3", None, thread_id="t2", attempt=1, is_final=False, buffer=True)
        assert h.flush() == 1 and h.flush() == 0

    # ---- archival ---------------------------------------------------------

    def _three_runs(self, h: ChunkHistory) -> None:
        item = _batch("b", _chunk("c1", SORT))
        h.record_attempt(item, "old kept", None, thread_id="t", attempt=1, is_final=True)
        h.record_attempt(item, "redo rejected", None, thread_id="t", attempt=1, is_final=False)
        h.record_attempt(item, "redo kept", None, thread_id="t", attempt=2, is_final=True)
        other = _batch("o", _chunk("c2", STEP, start=2))
        h.record_attempt(other, "other kept", None, thread_id="u", attempt=1, is_final=True)

    def test_archive_moves_stale_attempts_and_keeps_the_current_answer(self):
        import time as _time

        h = self.make()
        self._three_runs(h)
        result = h.archive(before_ts=_time.time() + 60)
        assert (result.attempts, result.conversions) == (2, 2)
        active = [r["response"] for r in h.history()]
        assert sorted(active) == ["other kept", "redo kept"]
        everything = h.history(include_archived=True)
        assert sorted(r["response"] for r in everything) == [
            "old kept", "other kept", "redo kept", "redo rejected",
        ]
        archived = [r for r in everything if r["archived_at"] is not None]
        assert sorted(r["response"] for r in archived) == ["old kept", "redo rejected"]
        assert all(r["original_text"] == SORT for r in archived)
        # Idempotent: nothing stale is left.
        assert h.archive(before_ts=_time.time() + 60).attempts == 0
        assert [r["chunk_id"] for r in h.search("PROC SORT", include_archived=True)] == [
            "c1", "c1", "c1",
        ]
        stats = h.stats()
        assert (stats["item_attempts"], stats["item_attempts_archive"]) == (2, 2)

    def test_archive_superseded_only_and_thread_scope(self):
        h = self.make()
        self._three_runs(h)
        assert h.archive(superseded_only=True, thread_id="u").attempts == 0
        result = h.archive(superseded_only=True, thread_id="t")
        assert result.attempts == 2
        assert [r["response"] for r in h.for_item("t", "b")] == ["redo kept"]
        with pytest.raises(ValueError, match="before_ts"):
            h.archive(thread_id="t")

    def test_archive_without_keep_current_can_take_the_current_answer(self):
        import time as _time

        h = self.make()
        self._three_runs(h)
        result = h.archive(before_ts=_time.time() + 60, keep_current=False, thread_id="u")
        assert result.attempts == 1
        assert h.for_item("u", "o") == []
        assert len(h.for_item("u", "o", include_archived=True)) == 1

    def test_stats_count_rows(self):
        h = self.make()
        h.record_attempt(
            _batch("b", _chunk("c1", SORT), _chunk("c2", STEP, start=2)),
            "r", None, thread_id="t", attempt=1, is_final=True,
        )
        stats = h.stats()
        assert (stats["chunk_sources"], stats["item_attempts"], stats["chunk_conversions"]) == (2, 1, 2)
        assert (stats["item_attempts_archive"], stats["chunk_conversions_archive"]) == (0, 0)


class TestInMemoryChunkHistory(ChunkHistoryContract):
    def make(self) -> ChunkHistory:
        return ChunkHistory()


class TestDeltaChunkHistory(ChunkHistoryContract):
    @pytest.fixture(autouse=True)
    def _bind_spark(self, delta_spark):
        self._spark = delta_spark
        self._bases: list[str] = []
        yield
        for base in self._bases:
            for archived in (False, True):
                delta_spark.sql(
                    f"DROP VIEW IF EXISTS {schema.chunk_history_view_name(base, archived=archived)}"
                )
            for spec in schema.HISTORY_TABLES + schema.ARCHIVE_TABLES:
                delta_spark.sql(f"DROP TABLE IF EXISTS {spec.name(base)}")

    def make(self, **kwargs) -> ChunkHistory:
        base = f"default.ch_{uuid.uuid4().hex[:10]}"
        self._bases.append(base)
        kwargs.setdefault("full_text_search", False)
        return ChunkHistory(self._spark, base, **kwargs)

    def test_databricks_features_degrade_to_a_warning_off_databricks(self, caplog):
        h = self.make(full_text_search=True)
        assert h.provision is not None
        assert h.provision.failures  # OSS Delta has neither feature
        assert not h.provision.full_text_search
        assert h.search_functions is False
        assert any("Databricks features unavailable" in r.message for r in caplog.records)
        h.record_attempt(
            _batch("b", _chunk("c1", SORT)), "r", None,
            thread_id="t", attempt=1, is_final=True,
        )
        assert [r["chunk_id"] for r in h.search("proc sort")] == ["c1"]

    def test_a_raced_duplicate_source_row_does_not_duplicate_history(self):
        h = self.make()
        h.record_attempt(
            _batch("b", _chunk("c1", SORT)), "r", None,
            thread_id="t", attempt=1, is_final=True,
        )
        base = self._bases[-1]
        sources = schema.CHUNK_SOURCES.name(base)
        # What two concurrent insert-only MERGEs of the same new chunk leave.
        self._spark.sql(f"INSERT INTO {sources} SELECT * FROM {sources}")
        assert self._spark.table(sources).count() == 2
        assert len(h.for_item("t", "b")) == 1

    def test_an_interrupted_archive_reads_once_and_a_rerun_finishes_it(self):
        import time as _time

        h = self.make()
        self._three_runs(h)
        base = self._bases[-1]
        real_retry = h._retry
        calls = {"n": 0}

        def crash_on_delete(operation, action):
            if "archive delete" in operation:
                raise RuntimeError("crash between copy and delete")
            calls["n"] += 1
            return real_retry(operation, action)

        h._retry = crash_on_delete  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="crash"):
            h.archive(before_ts=_time.time() + 60)
        h._retry = real_retry  # type: ignore[method-assign]
        # Copied but not deleted: both tables hold it, reads show it once.
        assert self._spark.table(schema.ITEM_ATTEMPTS_ARCHIVE.name(base)).count() == 2
        assert len(h.history(include_archived=True)) == 4
        assert h.archive(before_ts=_time.time() + 60).attempts == 2
        assert self._spark.table(schema.ITEM_ATTEMPTS_ARCHIVE.name(base)).count() == 2
        assert len(h.history()) == 2

    def test_an_item_with_retries_costs_three_commits(self):
        h = self.make()
        item = _batch("b", _chunk("c1", SORT))
        h.record_attempt(item, "r1", None, thread_id="t", attempt=1, is_final=False, buffer=True)
        h.record_attempt(item, "r2", None, thread_id="t", attempt=2, is_final=True)
        h.record_attempt(item, "r3", None, thread_id="t2", attempt=1, is_final=True)
        base = self._bases[-1]

        def writes(spec):
            return [
                r.operation
                for r in self._spark.sql(f"DESCRIBE HISTORY {spec.name(base)}").collect()
                if r.operation in ("WRITE", "MERGE")
            ]

        # One MERGE for the new chunk (the re-run on t2 skips it), one
        # INSERT per table per flush.
        assert writes(schema.CHUNK_SOURCES) == ["MERGE"]
        assert len(writes(schema.ITEM_ATTEMPTS)) == 2
        assert len(writes(schema.CHUNK_CONVERSIONS)) == 2

    def test_an_older_attempts_table_gains_the_feedback_column(self):
        base = f"default.ch_{uuid.uuid4().hex[:10]}"
        self._bases.append(base)
        cols = ", ".join(
            f"`{n}` {t}" for n, t in schema.ITEM_ATTEMPTS.columns if n != "feedback"
        )
        self._spark.sql(
            f"CREATE TABLE {schema.ITEM_ATTEMPTS.name(base)} ({cols}) USING DELTA"
        )
        h = ChunkHistory(self._spark, base, full_text_search=False)
        assert "feedback" in self._spark.table(schema.ITEM_ATTEMPTS.name(base)).columns
        h.record_attempt(_batch("b", _chunk("c1", SORT)), "r", None, thread_id="t",
                         attempt=2, is_final=True, feedback="why")
        assert h.for_item("t", "b")[0]["feedback"] == "why"

    def test_tables_carry_row_tracking_and_the_view_exists(self):
        h = self.make()
        h.record_attempt(
            _batch("b", _chunk("c1", SORT)), "r", None,
            thread_id="t", attempt=1, is_final=True,
        )
        base = self._bases[-1]
        for spec in schema.HISTORY_TABLES:
            assert schema.table_property(
                self._spark, spec.name(base), "delta.enableRowTracking"
            )
        row = self._spark.table(schema.chunk_history_view_name(base)).first()
        assert row.original_text == SORT
