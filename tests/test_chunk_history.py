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

    def test_stats_count_rows(self):
        h = self.make()
        h.record_attempt(
            _batch("b", _chunk("c1", SORT), _chunk("c2", STEP, start=2)),
            "r", None, thread_id="t", attempt=1, is_final=True,
        )
        stats = h.stats()
        assert (stats["chunk_sources"], stats["item_attempts"], stats["chunk_conversions"]) == (2, 1, 2)


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
            delta_spark.sql(f"DROP VIEW IF EXISTS {schema.chunk_history_view_name(base)}")
            for spec in schema.HISTORY_TABLES:
                delta_spark.sql(f"DROP TABLE IF EXISTS {spec.name(base)}")

    def make(self, **kwargs) -> ChunkHistory:
        base = f"default.ch_{uuid.uuid4().hex[:10]}"
        self._bases.append(base)
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
