"""
test_backend_contract.py — one behavioral contract, run against every
KVStore backend.

``KVStoreContract`` defines the store-level behaviors both backends must
share (upsert/COALESCE semantics, literal prefix matching, delete counting,
atomic restore, special characters in keys). ``TestInMemoryContract`` always
runs; ``TestDeltaContract`` runs the identical tests against a real local
Delta table and skips itself only when pyspark, delta-spark, or a JVM is
genuinely absent.

The ``delta_spark`` fixture those tests take lives in ``conftest.py``, shared
with the other suite that needs a real session
(``test_data_hydration_delta.py``) — there can only be one, since
``SparkSession.getOrCreate`` returns the process's existing session. Its
"an incompatible Delta is a failure, not a skip" rule is documented there.

To exercise the Delta side locally or in CI:

    pip install pyspark delta-spark   # matching major versions, JVM required
"""

import pathlib
import sys
import uuid

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from memory import schema
from memory.store import KVStore


class KVStoreContract:
    """Every KVStore backend must pass exactly these tests."""

    def make_store(self) -> KVStore:
        raise NotImplementedError

    # ---- CRUD ------------------------------------------------------------

    def test_set_get_roundtrip(self):
        s = self.make_store()
        s.set("kv::a", {"x": 1})
        assert s.get("kv::a") == {"x": 1}
        assert s.get("kv::missing") is None
        assert s.get("kv::missing", "default") == "default"

    def test_update_preserves_created_at_and_tags_when_omitted(self):
        s = self.make_store()
        s.set("kv::a", 1, tags=["orig"], source="src1")
        first = dict(s.all_records())["kv::a"]
        s.set("kv::a", 2)  # tags/source omitted → COALESCE keeps existing
        second = dict(s.all_records())["kv::a"]
        assert second["value"] == 2
        assert second["tags"] == ["orig"]
        assert second["source"] == "src1"
        assert second["created_at"] == first["created_at"]

    def test_update_replaces_tags_when_provided(self):
        s = self.make_store()
        s.set("kv::a", 1, tags=["old"])
        s.set("kv::a", 1, tags=["new"])
        assert dict(s.all_records())["kv::a"]["tags"] == ["new"]

    def test_set_many_batch_and_explicit_timestamps(self):
        s = self.make_store()
        s.set_many(
            [
                {"key": "kv::a", "value": 1, "tags": ["x"]},
                {"key": "kv::b", "value": {"n": 2}, "created_at": 100.0, "updated_at": 200.0},
            ]
        )
        records = dict(s.all_records())
        assert s.get("kv::a") == 1
        assert s.get("kv::b") == {"n": 2}
        assert records["kv::a"]["tags"] == ["x"]
        assert records["kv::b"]["created_at"] == 100.0
        assert records["kv::b"]["updated_at"] == 200.0

    # ---- Key queries -------------------------------------------------------

    def test_keys_prefix_filter_and_has_prefix(self):
        s = self.make_store()
        s.set("kv::a", 1)
        s.set("kv::b", 2)
        s.set("msg::t::0", 3)
        assert sorted(s.keys("kv::")) == ["kv::a", "kv::b"]
        assert sorted(s.keys()) == ["kv::a", "kv::b", "msg::t::0"]
        assert s.has_prefix("msg::")
        assert not s.has_prefix("idx::")

    def test_prefix_match_is_literal_not_wildcard(self):
        """'_' and '%' in a prefix must never act as LIKE wildcards."""
        s = self.make_store()
        s.set("msg::thread_1::0", "a")
        s.set("msg::threadX1::0", "b")
        s.set("kv::a%b::0", "c")
        s.set("kv::aXb::0", "d")
        assert s.keys("msg::thread_1::") == ["msg::thread_1::0"]
        assert s.clear_prefix("msg::thread_1::") == 1
        assert s.keys("msg::") == ["msg::threadX1::0"]
        assert s.clear_prefix("kv::a%b::") == 1
        assert sorted(s.keys("kv::")) == ["kv::aXb::0"]

    def test_prefix_containing_escape_char(self):
        s = self.make_store()
        s.set("kv::a~b::0", "kept-target")
        s.set("kv::a~~b::0", "other")
        assert s.clear_prefix("kv::a~b::") == 1
        assert s.keys("kv::") == ["kv::a~~b::0"]

    # ---- Deletes -----------------------------------------------------------

    def test_delete_and_delete_many_count_existing_only(self):
        s = self.make_store()
        s.set("kv::a", 1)
        s.set("kv::b", 2)
        assert s.delete("kv::a")
        assert not s.delete("kv::a")
        assert s.delete_many(["kv::b", "kv::missing"]) == 1
        assert s.keys() == []

    def test_keys_with_quotes_survive_write_and_delete(self):
        """Keys/values with SQL-hostile characters go through parameter
        markers, so they must round-trip and delete cleanly."""
        s = self.make_store()
        key = "kv::o'brien; DROP TABLE x --"
        s.set(key, "v'al")
        assert s.get(key) == "v'al"
        assert s.delete_many([key]) == 1
        assert s.keys() == []

    def test_records_after_returns_only_strictly_later_keys(self):
        s = self.make_store()
        s.set("msg::t::0001", "a")
        s.set("msg::t::0002", "b")
        s.set("msg::t::0003", "c")
        s.set("kv::other", "d")  # different prefix, never returned

        later = sorted(k for k, _ in s.records_after("msg::t::", "msg::t::0001"))
        assert later == ["msg::t::0002", "msg::t::0003"]
        everything = sorted(k for k, _ in s.records_after("msg::t::", ""))
        assert everything == ["msg::t::0001", "msg::t::0002", "msg::t::0003"]
        assert s.records_after("msg::t::", "msg::t::0003") == []

    def test_clear_removes_everything(self):
        s = self.make_store()
        s.set("kv::a", 1)
        s.set("msg::t::0", 2)
        s.clear()
        assert s.keys() == []
        assert s.stats()["total_keys"] == 0

    # ---- Snapshot / restore --------------------------------------------------

    def test_restore_preserves_timestamps(self):
        s = self.make_store()
        s.set("kv::a", 1, tags=["t"])
        before = dict(s.all_records())["kv::a"]

        target = self.make_store()
        target.restore(s.snapshot())

        after = dict(target.all_records())["kv::a"]
        assert after["created_at"] == before["created_at"]
        assert after["updated_at"] == before["updated_at"]
        assert after["value"] == 1
        assert after["tags"] == ["t"]

    def test_restore_replaces_existing_contents(self):
        source = self.make_store()
        source.set("kv::new", "n")
        target = self.make_store()
        target.set("kv::stale", "s")
        target.restore(source.snapshot())
        assert target.keys() == ["kv::new"]

    def test_restore_empty_snapshot_clears_store(self):
        s = self.make_store()
        s.set("kv::a", 1)
        s.restore({})
        assert s.keys() == []


class TestInMemoryContract(KVStoreContract):
    def make_store(self) -> KVStore:
        return KVStore(spark=None, table=None)


class TestDeltaContract(KVStoreContract):
    @pytest.fixture(autouse=True)
    def _bind_spark(self, delta_spark):
        self._spark = delta_spark
        self._tables: list[str] = []
        yield
        for table in self._tables:
            delta_spark.sql(f"DROP TABLE IF EXISTS {table}")

    def make_store(self) -> KVStore:
        return KVStore(spark=self._spark, table=self._base())

    # ---- Table family setup -----------------------------------------------

    def _base(self) -> str:
        base = f"default.kv_contract_{uuid.uuid4().hex[:10]}"
        self._tables.extend(spec.name(base) for spec in schema.KV_TABLES)
        return base

    def _properties(self, table: str) -> dict[str, str]:
        return {
            row["key"]: row["value"]
            for row in self._spark.sql(f"SHOW TBLPROPERTIES {table}").collect()
        }

    def test_tables_get_row_tracking_and_cdf_only_where_read(self):
        """Every table gets row tracking (a full-text search index
        prerequisite); only the tables a CDF consumer reads get CDF, so the
        hot fact tables do not pay its write cost. The CREATE sets both, so a
        fresh table never needs the ALTER (which a delta-spark built against
        an older Spark minor cannot execute)."""
        base = self._base()
        store = KVStore(spark=self._spark, table=base)
        for spec in schema.KV_TABLES:
            props = self._properties(spec.name(base))
            assert props.get("delta.enableRowTracking") == "true"
            assert (props.get("delta.enableChangeDataFeed") == "true") == spec.cdf
        assert store._backend._cdf_enabled("messages") is True

    def test_cdf_is_re_enabled_when_a_table_lacks_it(self):
        base = self._base()
        KVStore(spark=self._spark, table=base)
        messages = schema.MESSAGES.name(base)
        try:
            self._spark.sql(
                f"ALTER TABLE {messages} UNSET TBLPROPERTIES "
                "('delta.enableChangeDataFeed')"
            )
        except Exception as exc:
            pytest.skip(
                f"this pyspark/delta-spark pair cannot write table properties "
                f"({type(exc).__name__}); the upgrade path is unavailable here"
            )
        store = KVStore(spark=self._spark, table=base)
        assert store._backend._cdf_enabled("messages") is True

    def test_keys_are_routed_into_typed_tables(self):
        base = self._base()
        store = KVStore(spark=self._spark, table=base)
        store.set_many(
            [
                {"key": "msg::t1::0000000000000001-ab", "value": {
                    "message": {"type": "human", "data": {"content": "hello"}}, "ts": 1.0}},
                {"key": "chat::t1::c1", "value": {"chat_id": "c1", "message_count": 2}},
                {"key": "kv::run::t1::item::batch-001", "value": {
                    "status": "ok", "index": 1, "total": 3, "attempts": 2}},
                {"key": "kv::validation::t1::item::batch-001", "value": {
                    "score": 0.75, "passed": True, "index": 1}},
                {"key": "kv::policy::task-9", "value": {"instructions": []}},
            ]
        )
        msg = self._spark.table(schema.MESSAGES.name(base)).first()
        assert (msg.thread_id, msg.msg_tick, msg.role, msg.content) == ("t1", 1, "human", "hello")
        chat = self._spark.table(schema.CHATS.name(base)).first()
        assert (chat.thread_id, chat.chat_id, chat.message_count) == ("t1", "c1", 2)
        run = self._spark.table(schema.RUN_ITEMS.name(base)).first()
        assert (run.thread_id, run.item_id, run.status, run.idx, run.attempts) == (
            "t1", "batch-001", "ok", 1, 2,
        )
        val = self._spark.table(schema.VALIDATIONS.name(base)).first()
        assert (val.item_id, val.score, val.passed) == ("batch-001", 0.75, True)
        kv = self._spark.table(schema.KV.name(base)).first()
        assert (kv.namespace, kv.scope_id) == ("policy", "task-9")
        # The façade still sees one key space.
        assert store.keys("kv::") == [
            "kv::policy::task-9",
            "kv::run::t1::item::batch-001",
            "kv::validation::t1::item::batch-001",
        ]
        assert len(store.keys()) == 5
        assert store.thread_ids() == ["t1"]
        assert store.stats()["total_keys"] == 5

    def test_append_inserts_messages_and_still_merges_the_chat_row(self):
        base = self._base()
        store = KVStore(spark=self._spark, table=base)
        store.set("chat::t1::c1", {"chat_id": "c1", "message_count": 1})
        store.append_many(
            [
                {"key": "msg::t1::0000000000000002-aa", "value": {"ts": 2.0}},
                {"key": "chat::t1::c1", "value": {"chat_id": "c1", "message_count": 2}},
            ]
        )
        assert store.get("chat::t1::c1")["message_count"] == 2
        assert self._spark.table(schema.CHATS.name(base)).count() == 1
        ops = [
            r.operation
            for r in self._spark.sql(
                f"DESCRIBE HISTORY {schema.MESSAGES.name(base)}"
            ).collect()
        ]
        assert "WRITE" in ops and "MERGE" not in ops

    def test_search_is_filtered_in_sql_with_identical_results(self):
        base = self._base()
        store = KVStore(spark=self._spark, table=base)
        store.set("kv::a", "PROC SORT it", tags=["x"])
        store.set("kv::b", "unrelated", tags=["Sorting"])
        store.set("kv::c", "nothing")
        assert store._backend.search_functions is False  # OSS: LIKE fallback
        assert {k for k, _ in store.search("sort")} == {"kv::a", "kv::b"}

    def test_legacy_single_table_store_migrates_once(self):
        from memory.operations import DeltaMemoryMaintenance, VacuumPolicy

        legacy = f"default.kv_legacy_{uuid.uuid4().hex[:10]}"
        self._tables.append(legacy)
        self._spark.sql(
            f"CREATE TABLE {legacy} (kv_key STRING NOT NULL, value STRING NOT NULL,"
            " tags STRING, created_at DOUBLE, updated_at DOUBLE) USING DELTA"
        )
        self._spark.sql(
            f"INSERT INTO {legacy} VALUES "
            "('msg::t::0000000000000001-aa', '{\"ts\": 1}', '[]', 10.0, 11.0),"
            "('kv::run::t::item::b1', '{\"status\": \"ok\"}', '[]', 12.0, 13.0),"
            "('kv::goal', '\"x\"', NULL, 14.0, 15.0)"
        )
        base = legacy  # the usual case: the old table name becomes the base
        self._tables.extend(spec.name(base) for spec in schema.ALL_TABLES)
        ops = DeltaMemoryMaintenance(self._spark, base, policy=VacuumPolicy(7 * 24))
        copied = ops.migrate_legacy(legacy)
        assert copied == {"messages": 1, "run_items": 1, "kv": 1}
        store = KVStore(spark=self._spark, table=base)
        records = dict(store.all_records())
        assert records["kv::goal"]["created_at"] == 14.0
        assert records["kv::run::t::item::b1"]["value"] == {"status": "ok"}
        with pytest.raises(RuntimeError, match="already hold rows"):
            ops.migrate_legacy(legacy)
