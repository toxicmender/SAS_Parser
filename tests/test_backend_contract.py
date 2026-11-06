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
from contextlib import contextmanager

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from memory import schema
from memory.store import KVStore, forget_provisioning



# Known defects of a pyspark/delta-spark pair built against different Spark
# patch releases (see conftest.py and docker/spark/warmup.py): the JVM side
# cannot find a catalyst class or method. Such a pair cannot run the statement
# under test at all, which is an environment limit, not a store defect.
_PAIRING_DEFECTS = ("NoSuchMethodError", "ClassNotFoundException")


@contextmanager
def tolerate_pairing_defects(what: str):
    try:
        yield
    except Exception as exc:
        if any(d in str(exc) for d in _PAIRING_DEFECTS):
            pytest.skip(
                f"this pyspark/delta-spark pair cannot {what} "
                f"({type(exc).__name__}: JVM class/method missing)"
            )
        raise

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
        forget_provisioning()
        yield
        for table in self._tables:
            delta_spark.sql(f"DROP TABLE IF EXISTS {table}")

    def make_store(self) -> KVStore:
        return KVStore(spark=self._spark, table=self._base(), full_text_search=False)

    # ---- Table family setup -----------------------------------------------

    def _base(self) -> str:
        base = f"default.kv_contract_{uuid.uuid4().hex[:10]}"
        self._tables.extend(
            spec.name(base) for spec in (*schema.KV_TABLES, schema.RESTORE_JOURNAL)
        )
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
        store = KVStore(spark=self._spark, table=base, full_text_search=False)
        for spec in schema.KV_TABLES:
            props = self._properties(spec.name(base))
            assert props.get("delta.enableRowTracking") == "true"
            assert (props.get("delta.enableChangeDataFeed") == "true") == spec.cdf
        assert store._backend._cdf_enabled("messages") is True

    def test_cdf_is_re_enabled_when_a_table_lacks_it(self):
        base = self._base()
        KVStore(spark=self._spark, table=base, full_text_search=False)
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
        forget_provisioning(base)  # a new process: re-run the startup checks
        store = KVStore(spark=self._spark, table=base, full_text_search=False)
        assert store._backend._cdf_enabled("messages") is True

    def test_a_second_store_on_the_same_base_issues_no_ddl(self, monkeypatch):
        base = self._base()
        KVStore(spark=self._spark, table=base, full_text_search=False)
        seen: list[str] = []
        real_sql = self._spark.sql
        monkeypatch.setattr(self._spark, "sql", lambda q, *a, **k: (seen.append(q), real_sql(q, *a, **k))[1])
        KVStore(spark=self._spark, table=base, full_text_search=False)
        assert seen == []

    def test_concurrent_first_opens_of_one_base_both_succeed(self):
        import threading

        base = self._base()
        errors: list[BaseException] = []

        def open_store() -> None:
            try:
                KVStore(spark=self._spark, table=base, full_text_search=False)
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        workers = [threading.Thread(target=open_store) for _ in range(2)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        assert errors == []

    def test_keys_are_routed_into_typed_tables(self):
        base = self._base()
        store = KVStore(spark=self._spark, table=base, full_text_search=False)
        store.set_many(
            [
                {"key": "msg::t1::0000000000000001-ab", "value": {
                    "message": {"type": "human", "data": {"content": "hello"}}, "ts": 1.0}},
                {"key": "chat::t1::c1", "value": {"chat_id": "c1", "label": "L", "started_at": 5.0}},
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
        assert (chat.thread_id, chat.chat_id, chat.label, chat.started_at) == ("t1", "c1", "L", 5.0)
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
        store.set("kv::project_goal", "x")
        goal = self._spark.table(schema.KV.name(base)).filter("kv_key = 'kv::project_goal'").first()
        assert (goal.namespace, goal.scope_id) == ("kv", "project_goal")

    def test_append_inserts_messages_and_still_merges_the_chat_row(self):
        base = self._base()
        store = KVStore(spark=self._spark, table=base, full_text_search=False)
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
        store = KVStore(spark=self._spark, table=base, full_text_search=False)
        store.set("kv::a", "PROC SORT it", tags=["x"])
        store.set("kv::b", "unrelated", tags=["Sorting"])
        store.set("kv::c", "nothing")
        assert store._backend.search_functions is False  # OSS: LIKE fallback
        assert {k for k, _ in store.search("sort")} == {"kv::a", "kv::b"}

    def test_cdf_sync_follows_messages_and_chats_with_one_checkpoint_each(self):
        base = self._base()
        audit = f"{base}_audit"
        self._tables.append(audit)
        store = KVStore(spark=self._spark, table=base, audit_table=audit, full_text_search=False)
        first = store.sync_cdf("worker-1")
        assert first.baseline is True
        store.append_many(
            [
                {"key": "msg::t1::0000000000000001-aa", "value": {"ts": 1.0}},
                {"key": "chat::t1::c1", "value": {"chat_id": "c1"}},
            ]
        )
        store.set("kv::run::t1::item::b1", {"status": "ok"})  # no CDF table
        second = store.sync_cdf("worker-1")
        assert second.baseline is False
        changed = {(e["table"].rsplit("_", 1)[-1], e["key"]) for e in second.events}
        assert changed == {
            ("messages", "msg::t1::0000000000000001-aa"),
            ("chats", "chat::t1::c1"),
        }
        # Nothing new: no events, checkpoints unchanged.
        assert store.sync_cdf("worker-1").events == ()
        checkpoints = (
            self._spark.table(audit).filter("change_type = 'checkpoint'").count()
        )
        assert checkpoints == 2

    # ---- chats derived from messages ---------------------------------------

    def test_chat_stats_are_aggregated_from_the_messages(self):
        from langchain_core.messages import HumanMessage

        from memory.store import KVChatMessageHistory

        base = self._base()
        store = KVStore(spark=self._spark, table=base, full_text_search=False)
        a = KVChatMessageHistory("t1", store)
        a.set_chat("chat-a", label="first")
        a.add_messages([HumanMessage("one"), HumanMessage("two")])
        a.set_chat("chat-b")
        a.add_messages([HumanMessage("three")])
        a.set_chat("chat-a")
        a.add_messages([HumanMessage("four")])  # interleaved back into chat-a

        chats = {c["chat_id"]: c for c in a.chats()}
        assert chats["chat-a"]["message_count"] == 3
        assert chats["chat-a"]["label"] == "first"
        assert chats["chat-b"]["message_count"] == 1
        assert [m.content for m in a.messages_in_chat("chat-a")] == ["one", "two", "four"]
        # The static row was written once per chat; appends are message-only.
        ops = [
            r.operation
            for r in self._spark.sql(f"DESCRIBE HISTORY {schema.CHATS.name(base)}").collect()
        ]
        assert ops.count("MERGE") == 2

    # ---- a blind INSERT is never duplicated -------------------------------

    def test_an_insert_whose_commit_landed_is_not_retried(self, monkeypatch):
        base = self._base()
        store = KVStore(spark=self._spark, table=base, full_text_search=False, max_write_retries=3)
        backend = store._backend
        real_sql = self._spark.sql

        def commit_then_fail(query, *args, **kwargs):
            result = real_sql(query, *args, **kwargs)
            if query.startswith("INSERT INTO") and "_messages" in query:
                # The commit landed; the client still sees a conflict.
                raise RuntimeError("[DELTA_CONCURRENT_APPEND] ConcurrentAppendException")
            return result

        monkeypatch.setattr(backend._spark, "sql", commit_then_fail)
        store.append_many([{"key": "msg::t::0000000000000001-aa", "value": {"ts": 1.0}}])
        monkeypatch.setattr(backend._spark, "sql", real_sql)
        assert self._spark.table(schema.MESSAGES.name(base)).count() == 1

    # ---- restore: undo only what the run did, only while safe ---------------

    def _seed(self, store: KVStore) -> None:
        store.set_many(
            [
                {"key": "msg::t::0000000000000001-aa", "value": {"ts": 1.0}},
                {"key": "chat::t::c", "value": {"chat_id": "c"}},
                {"key": "kv::run::t::item::b", "value": {"status": "ok"}},
                {"key": "kv::goal", "value": "keep me"},
            ]
        )

    def _fail_restore_at(self, monkeypatch, store: KVStore, suffix: str) -> None:
        backend = store._backend
        real_stage = backend._stage

        def failing_stage(spec, rows, now):
            if spec.suffix == suffix:
                raise RuntimeError("simulated failure mid-restore")
            return real_stage(spec, rows, now)

        monkeypatch.setattr(backend, "_stage", failing_stage)
        with pytest.raises(RuntimeError, match="mid-restore"):
            store.restore({"kv::other": {"value": 1}})
        monkeypatch.setattr(backend, "_stage", real_stage)

    def _states(self, base: str) -> list[tuple[str, str | None]]:
        return [
            (r.state, r.table_name)
            for r in self._spark.table(schema.RESTORE_JOURNAL.name(base)).collect()
        ]

    def test_a_failed_restore_rolls_back_only_the_tables_it_overwrote(self, monkeypatch):
        with tolerate_pairing_defects("RESTORE TABLE and read it back"):
            base = self._base()
            store = KVStore(spark=self._spark, table=base, full_text_search=False)
            self._seed(store)
            before = store.snapshot()
            versions = {
                s.suffix: store._backend._latest_version(s.suffix) for s in schema.KV_TABLES
            }
            # Fails on run_items: messages and chats were overwritten.
            self._fail_restore_at(monkeypatch, store, "run_items")
            assert store.snapshot() == before
            assert store._backend.unfinished_restores() == {}
            after = {
                s.suffix: store._backend._latest_version(s.suffix) for s in schema.KV_TABLES
            }
            # Never-overwritten tables were not touched (no RESTORE commit).
            assert after["run_items"] == versions["run_items"]
            assert after["kv"] == versions["kv"]
            assert ("rolled_back", None) in self._states(base)

    def test_rollback_spares_a_table_written_to_since_the_overwrite(self, monkeypatch):
        with tolerate_pairing_defects("RESTORE TABLE and read it back"):
            base = self._base()
            store = KVStore(spark=self._spark, table=base, full_text_search=False)
            self._seed(store)
            backend = store._backend
            real_stage = backend._stage

            def stage_then_concurrent_append(spec, rows, now):
                if spec.suffix == "run_items":
                    # Another worker appends to messages after the restore
                    # overwrote it, then the restore fails.
                    other = KVStore(spark=self._spark, table=base, full_text_search=False)
                    other.append_many([{"key": "msg::t::0000000000000009-zz", "value": {"ts": 9.0}}])
                    raise RuntimeError("simulated failure mid-restore")
                return real_stage(spec, rows, now)

            monkeypatch.setattr(backend, "_stage", stage_then_concurrent_append)
            with pytest.raises(RuntimeError, match="mid-restore"):
                store.restore({"kv::other": {"value": 1}})
            monkeypatch.setattr(backend, "_stage", real_stage)

            # The concurrent append survives; messages was left for a human.
            assert store.get("msg::t::0000000000000009-zz") == {"ts": 9.0}
            assert ("rollback_skipped", schema.MESSAGES.name(base)) in self._states(base)
            # chats had no later writes: it was rolled back.
            assert store.get("chat::t::c") == {"chat_id": "c"}

    def test_a_restore_whose_completion_was_not_journaled_is_settled_as_complete(self):
        with tolerate_pairing_defects("RESTORE TABLE and read it back"):
            from memory.operations import DeltaMemoryMaintenance, VacuumPolicy

            base = self._base()
            store = KVStore(spark=self._spark, table=base, full_text_search=False)
            self._seed(store)
            backend = store._backend
            # What a run that finished every overwrite leaves without its
            # 'completed' row.
            pre = {s.suffix: backend._latest_version(s.suffix) for s in schema.KV_TABLES}
            store.restore({"kv::new": {"value": 2}})
            journal = schema.RESTORE_JOURNAL.name(base)
            self._spark.sql(f"DELETE FROM {journal} WHERE state = 'completed'")
            assert len(backend.unfinished_restores()) == 1

            ops = DeltaMemoryMaintenance(self._spark, base, policy=VacuumPolicy(7 * 24))
            (outcome,) = ops.recover_restore().values()
            assert outcome == {"completed": True, "rolled_back": [], "needs_manual": []}
            assert store.get("kv::new") == 2  # the good restore stands
            assert all(backend._latest_version(s) > v for s, v in pre.items())

    def test_an_interrupted_restore_is_reported_and_settled(self, caplog):
        with tolerate_pairing_defects("RESTORE TABLE and read it back"):
            from memory.operations import DeltaMemoryMaintenance, VacuumPolicy

            base = self._base()
            store = KVStore(spark=self._spark, table=base, full_text_search=False)
            store.set("kv::goal", "original")
            backend = store._backend
            pre = {s.suffix: backend._latest_version(s.suffix) for s in schema.KV_TABLES}
            # What a crash leaves: 'started' for every table, then kv
            # overwritten (journaled) and nothing else.
            backend._journal(
                [("crashed", backend._names[k], v, "started", 1.0, None) for k, v in pre.items()]
            )
            self._spark.sql(f"DELETE FROM {schema.KV.name(base)}")
            backend._journal(
                [("crashed", backend._names["kv"], pre["kv"], "overwritten", 2.0,
                  backend._latest_version("kv"))]
            )
            assert store.get("kv::goal") is None

            forget_provisioning(base)  # the next process opens the store
            KVStore(spark=self._spark, table=base, full_text_search=False)
            assert any("interrupted" in r.message for r in caplog.records)
            ops = DeltaMemoryMaintenance(self._spark, base, policy=VacuumPolicy(7 * 24))
            assert ops.recover_restore() == {
                "crashed": {"completed": False, "rolled_back": ["kv"], "needs_manual": []}
            }
            assert store.get("kv::goal") == "original"
            # Tables the crashed run never overwrote were not restored.
            assert {
                s: backend._latest_version(s) for s in pre if s != "kv"
            } == {s: v for s, v in pre.items() if s != "kv"}
            assert backend.unfinished_restores() == {}

    # ---- additive schema migration ----------------------------------------

    def test_a_missing_nullable_column_is_added_and_a_retyped_one_rejected(self):
        with tolerate_pairing_defects("ALTER TABLE ADD COLUMNS"):
            base = self._base()
            KVStore(spark=self._spark, table=base, full_text_search=False)
            kv = schema.KV.name(base)
            # An older table: drop the newest nullable column by recreating it.
            self._spark.sql(f"DROP TABLE {kv}")
            self._spark.sql(
                f"CREATE TABLE {kv} (kv_key STRING NOT NULL, value STRING NOT NULL, "
                "tags STRING, created_at DOUBLE, updated_at DOUBLE, source STRING, "
                "namespace STRING) USING DELTA"
            )
            forget_provisioning(base)
            store = KVStore(spark=self._spark, table=base, full_text_search=False)
            assert "scope_id" in self._spark.table(kv).columns
            store.set("kv::policy::t", 1)
            assert self._spark.table(kv).first().scope_id == "t"

            self._spark.sql(f"DROP TABLE {kv}")
            self._spark.sql(
                f"CREATE TABLE {kv} (kv_key STRING NOT NULL, value STRING NOT NULL, "
                "tags STRING, created_at STRING, updated_at DOUBLE, source STRING, "
                "namespace STRING, scope_id STRING) USING DELTA"
            )
            forget_provisioning(base)
            with pytest.raises(RuntimeError, match="incompatible memory schema"):
                KVStore(spark=self._spark, table=base, full_text_search=False)

    def test_rederive_columns_fixes_rows_written_under_an_old_rule(self):
        from memory.operations import DeltaMemoryMaintenance, VacuumPolicy

        base = self._base()
        store = KVStore(spark=self._spark, table=base, full_text_search=False)
        store.set("kv::project_goal", "x")
        kv = schema.KV.name(base)
        self._spark.sql(f"UPDATE {kv} SET namespace = NULL, scope_id = NULL")
        ops = DeltaMemoryMaintenance(self._spark, base, policy=VacuumPolicy(7 * 24))
        assert ops.rederive_columns()["kv"] == 1
        row = self._spark.table(kv).first()
        assert (row.namespace, row.scope_id, row.value) == ("kv", "project_goal", '"x"')

    # ---- legacy migration -------------------------------------------------

    def test_legacy_single_table_store_migrates_once(self):
        from memory.operations import DeltaMemoryMaintenance, VacuumPolicy
        from memory.store import KVChatMessageHistory

        legacy = f"default.kv_legacy_{uuid.uuid4().hex[:10]}"
        self._tables.append(legacy)
        self._spark.sql(
            f"CREATE TABLE {legacy} (kv_key STRING NOT NULL, value STRING NOT NULL,"
            " tags STRING, created_at DOUBLE, updated_at DOUBLE) USING DELTA"
        )
        msg = '{"message": {"type": "human", "data": {"content": "hi"}}, "ts": 1}'
        chat = (
            '{"chat_id": "c1", "thread_id": "t", "label": "L", "started_at": 1.0,'
            ' "first_key": "msg::t::0000000000000001-aa",'
            ' "last_key": "msg::t::0000000000000001-aa", "message_count": 1}'
        )
        self._spark.sql(
            f"INSERT INTO {legacy} VALUES "
            f"('msg::t::0000000000000001-aa', '{msg}', '[]', 10.0, 11.0),"
            f"('chat::t::c1', '{chat}', '[]', 10.0, 11.0),"
            "('kv::run::t::item::b1', '{\"status\": \"ok\"}', '[]', 12.0, 13.0),"
            "('kv::goal', '\"x\"', NULL, 14.0, 15.0)"
        )
        base = legacy  # the usual case: the old table name becomes the base
        self._tables.extend(spec.name(base) for spec in schema.ALL_TABLES)
        ops = DeltaMemoryMaintenance(self._spark, base, policy=VacuumPolicy(7 * 24))
        copied = ops.migrate_legacy(legacy)
        assert copied == {"messages": 1, "chats": 1, "run_items": 1, "kv": 1}
        store = KVStore(spark=self._spark, table=base, full_text_search=False)
        records = dict(store.all_records())
        assert records["kv::goal"]["created_at"] == 14.0
        assert records["kv::run::t::item::b1"]["value"] == {"status": "ok"}
        # The legacy chat keeps its message: migration stamped it.
        assert records["msg::t::0000000000000001-aa"]["value"]["chat_id"] == "c1"
        thread = KVChatMessageHistory("t", store)
        (c1,) = thread.chats()
        assert (c1["chat_id"], c1["message_count"], c1["label"]) == ("c1", 1, "L")
        assert [m.content for m in thread.messages_in_chat("c1")] == ["hi"]
        with pytest.raises(RuntimeError, match="already hold rows"):
            ops.migrate_legacy(legacy)
