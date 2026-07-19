"""
test_store.py — unit tests for the in-memory (table=None) backend of
memory.store.KVStore and the layers built on top of it.

These deliberately construct the store with ``spark=None, table=None``:
in-memory mode uses the pure-dict _InMemoryBackend, which never touches
Spark — the suite runs without a JVM / SparkSession, and pyspark itself is
not required (the _DeltaBackend is not covered here).
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from memory.store import (
    MemoryHub,
    KVChatMessageHistory,
    KVMemoryStore,
    KVStore,
    ThreadMemoryManager,
    _sql_like_prefix,
)


class TestKVStoreInMemory(unittest.TestCase):
    def _store(self) -> KVStore:
        return KVStore(spark=None, table=None)

    def test_set_get_roundtrip(self):
        s = self._store()
        s.set("kv::a", {"x": 1})
        self.assertEqual(s.get("kv::a"), {"x": 1})
        self.assertIsNone(s.get("kv::missing"))
        self.assertEqual(s.get("kv::missing", "default"), "default")

    def test_keys_prefix_filter(self):
        s = self._store()
        s.set("kv::a", 1)
        s.set("kv::b", 2)
        s.set("msg::t::0", 3)
        self.assertEqual(sorted(s.keys("kv::")), ["kv::a", "kv::b"])
        self.assertEqual(sorted(s.keys()), ["kv::a", "kv::b", "msg::t::0"])

    def test_all_records_deserialises_value_and_tags(self):
        s = self._store()
        s.set("kv::a", {"n": 5}, tags=["t1", "t2"])
        records = dict(s.all_records("kv::"))
        rec = records["kv::a"]
        self.assertEqual(rec["value"], {"n": 5})
        self.assertEqual(rec["tags"], ["t1", "t2"])

    def test_get_by_tag_and_all_tags(self):
        s = self._store()
        s.set("kv::a", 1, tags=["proj"])
        s.set("kv::b", 2, tags=["note"])
        s.set("kv::c", 3, tags=["proj", "note"])
        tagged = dict(s.get_by_tag("proj"))
        self.assertEqual(sorted(tagged), ["kv::a", "kv::c"])
        self.assertEqual(s.all_tags(), ["note", "proj"])

    def test_search_scores_key_value_and_tags(self):
        s = self._store()
        s.set("kv::pipeline", "unrelated", tags=[])
        s.set("kv::b", "a long pipeline pipeline text", tags=[])
        s.set("kv::c", "nothing", tags=["pipeline"])
        hits = dict(s.search("pipeline"))
        # key match (2.0), value has two occurrences (2.0), tag match (1.5)
        self.assertAlmostEqual(hits["kv::pipeline"], 2.0)
        self.assertAlmostEqual(hits["kv::b"], 2.0)
        self.assertAlmostEqual(hits["kv::c"], 1.5)
        # top_k honoured
        self.assertEqual(len(s.search("pipeline", top_k=1)), 1)

    def test_delete_returns_whether_present(self):
        s = self._store()
        s.set("kv::a", 1)
        self.assertTrue(s.delete("kv::a"))
        self.assertFalse(s.delete("kv::a"))

    def test_clear_prefix_counts_removed(self):
        s = self._store()
        s.set("msg::t::0", 1)
        s.set("msg::t::1", 2)
        s.set("kv::a", 3)
        self.assertEqual(s.clear_prefix("msg::"), 2)
        self.assertEqual(sorted(s.keys()), ["kv::a"])

    def test_stats_reports_in_memory_backend(self):
        s = self._store()
        s.set("kv::a", 1, tags=["x"])
        stats = s.stats()
        self.assertEqual(stats["total_keys"], 1)
        self.assertEqual(stats["backend"], "in-memory")
        self.assertEqual(stats["all_tags"], ["x"])

    def test_set_preserves_created_at_and_tags_on_update(self):
        s = self._store()
        s.set("kv::a", 1, tags=["orig"])
        first = dict(s.all_records())["kv::a"]
        s.set("kv::a", 2)  # no tags passed → keep existing
        second = dict(s.all_records())["kv::a"]
        self.assertEqual(second["value"], 2)
        self.assertEqual(second["tags"], ["orig"])
        self.assertEqual(second["created_at"], first["created_at"])


class TestChatHistoryInMemory(unittest.TestCase):
    def test_add_and_read_messages(self):
        store = KVStore(spark=None, table=None)
        h = KVChatMessageHistory("t1", store)
        self.assertFalse(h.has_messages())
        h.add_user_message("q1")
        h.add_ai_message("a1")
        self.assertTrue(h.has_messages())
        self.assertEqual([m.content for m in h.messages], ["q1", "a1"])

    def test_clear_removes_messages(self):
        store = KVStore(spark=None, table=None)
        h = KVChatMessageHistory("t1", store)
        h.add_user_message("q1")
        h.clear()
        self.assertFalse(h.has_messages())
        self.assertEqual(h.messages, [])

    def test_list_threads_only_nonempty(self):
        store = KVStore(spark=None, table=None)
        mgr = ThreadMemoryManager(store)
        mgr.get_thread("t1").add_user_message("hi")
        mgr.get_thread("empty")  # instantiated but no messages
        self.assertEqual(mgr.list_threads(), ["t1"])


class TestFacadeInMemory(unittest.TestCase):
    def test_kv_store_namespacing_and_search(self):
        mem = MemoryHub(spark=None, table=None)
        mem.kv.set("goal", "RAG pipeline", tags=["project"])
        self.assertEqual(mem.kv.get("goal"), "RAG pipeline")
        self.assertEqual(mem.kv.search("pipeline")[0][0], "goal")

    def test_summary_reports_backend(self):
        mem = MemoryHub(spark=None, table=None)
        self.assertEqual(mem.summary()["store"]["backend"], "in-memory")

    def test_kvmemorystore_strips_namespace_in_keys(self):
        store = KVStore(spark=None, table=None)
        kv = KVMemoryStore(store, namespace="kv")
        kv.set("a", 1)
        kv.set("b", 2)
        self.assertEqual(sorted(kv.keys()), ["a", "b"])

    def test_items_with_prefix_filters_at_the_store(self):
        store = KVStore(spark=None, table=None)
        kv = KVMemoryStore(store, namespace="kv")
        kv.set("run::t1::item::c1", {"status": "ok"})
        kv.set("run::t1::item::c2", {"status": "ok"})
        kv.set("run::t2::item::c1", {"status": "ok"})
        kv.set("validation::t1::item::c1", {"passed": True})
        items = kv.items_with_prefix("run::t1::item::")
        self.assertEqual(
            sorted(i["key"] for i in items),
            ["run::t1::item::c1", "run::t1::item::c2"],
        )
        # Same shape as all_items, which stays the unfiltered view.
        self.assertEqual(items[0]["value"], {"status": "ok"})
        self.assertEqual(len(kv.all_items()), 4)


class TestHybridKVSearch(unittest.TestCase):
    """KVMemoryStore.search with an injected HybridRanker (BM25-only here)."""

    def _mem(self):
        from memory.relevance import HybridRanker

        return MemoryHub(ranker=HybridRanker())

    def test_ranks_lexical_match_first(self):
        mem = self._mem()
        mem.kv.set("goal", "build a revenue pipeline", tags=["project"])
        mem.kv.set("note", "unrelated grocery list", tags=[])
        hits = mem.kv.search("revenue pipeline")
        self.assertEqual(hits[0][0], "goal")

    def test_matches_on_tags_and_keys_too(self):
        mem = self._mem()
        mem.kv.set("alpha", "nothing textual", tags=["billing"])
        mem.kv.set("beta", "other content", tags=["misc"])
        hits = mem.kv.search("billing")
        self.assertEqual(hits[0][0], "alpha")

    def test_no_signal_returns_empty(self):
        mem = self._mem()
        mem.kv.set("goal", "build a revenue pipeline")
        self.assertEqual(mem.kv.search("zzz_absent_token"), [])

    def test_empty_store_returns_empty(self):
        self.assertEqual(self._mem().kv.search("anything"), [])

    def test_top_k_honoured_and_scores_descend(self):
        mem = self._mem()
        for i in range(4):
            mem.kv.set(f"k{i}", f"pipeline doc {'pipeline ' * i}")
        hits = mem.kv.search("pipeline", top_k=2)
        self.assertEqual(len(hits), 2)
        self.assertGreater(hits[0][1], hits[1][1])

    def test_without_ranker_substring_scan_still_works(self):
        mem = MemoryHub()
        mem.kv.set("goal", "RAG pipeline")
        self.assertEqual(mem.kv.search("pipeline")[0][0], "goal")


class TestSqlLikePrefix(unittest.TestCase):
    def test_underscore_and_percent_escaped(self):
        self.assertEqual(_sql_like_prefix("msg::thread_1::"), "msg::thread~_1::")
        self.assertEqual(_sql_like_prefix("a%b"), "a~%b")

    def test_escape_char_doubled(self):
        self.assertEqual(_sql_like_prefix("a~b"), "a~~b")

    def test_plain_prefix_unchanged(self):
        self.assertEqual(_sql_like_prefix("kv::plain"), "kv::plain")


class TestSetMany(unittest.TestCase):
    def test_batch_roundtrip(self):
        s = KVStore(spark=None, table=None)
        s.set_many(
            [
                {"key": "kv::a", "value": 1, "tags": ["x"]},
                {"key": "kv::b", "value": {"n": 2}},
            ]
        )
        self.assertEqual(s.get("kv::a"), 1)
        self.assertEqual(s.get("kv::b"), {"n": 2})
        self.assertEqual(dict(s.all_records())["kv::a"]["tags"], ["x"])

    def test_explicit_timestamps_honoured(self):
        s = KVStore(spark=None, table=None)
        s.set_many(
            [{"key": "kv::a", "value": 1, "created_at": 100.0, "updated_at": 200.0}]
        )
        rec = dict(s.all_records())["kv::a"]
        self.assertEqual(rec["created_at"], 100.0)
        self.assertEqual(rec["updated_at"], 200.0)

    def test_empty_entries_noop(self):
        s = KVStore(spark=None, table=None)
        s.set_many([])
        self.assertEqual(s.keys(), [])

    def test_delete_many_counts_existing_only(self):
        s = KVStore(spark=None, table=None)
        s.set("kv::a", 1)
        s.set("kv::b", 2)
        self.assertEqual(s.delete_many(["kv::a", "kv::b", "kv::missing"]), 2)
        self.assertEqual(s.keys(), [])


class TestRestorePreservesTimestamps(unittest.TestCase):
    def test_created_at_survives_snapshot_restore(self):
        s = KVStore(spark=None, table=None)
        s.set("kv::a", 1, tags=["t"])
        before = dict(s.all_records())["kv::a"]

        snap = s.snapshot()
        s2 = KVStore(spark=None, table=None)
        s2.restore(snap)

        after = dict(s2.all_records())["kv::a"]
        self.assertEqual(after["created_at"], before["created_at"])
        self.assertEqual(after["updated_at"], before["updated_at"])
        self.assertEqual(after["value"], 1)
        self.assertEqual(after["tags"], ["t"])


class TestTimestampMessageKeys(unittest.TestCase):
    def test_message_order_preserved_within_batch(self):
        """Messages written in one add_messages call (same wall-clock
        microsecond is possible) must read back in insertion order."""
        store = KVStore(spark=None, table=None)
        h = KVChatMessageHistory("t1", store)
        from langchain_core.messages import AIMessage, HumanMessage

        h.add_messages(
            [HumanMessage(content=f"q{i}") for i in range(5)]
            + [AIMessage(content="a")]
        )
        contents = [m.content for m in h.messages]
        self.assertEqual(contents, ["q0", "q1", "q2", "q3", "q4", "a"])

    def test_legacy_sequence_keys_sort_before_new_keys(self):
        """Pre-existing {seq:08d} keys must stay ahead of timestamp keys."""
        store = KVStore(spark=None, table=None)
        h = KVChatMessageHistory("t1", store)
        # Simulate a legacy message written by the old counter scheme
        store.set(
            "msg::t1::00000000",
            {"role": "human", "content": "legacy", "meta": {}, "ts": 1.0},
        )
        h.add_user_message("new")
        self.assertEqual([m.content for m in h.messages], ["legacy", "new"])

    def test_no_idx_counter_key_written(self):
        store = KVStore(spark=None, table=None)
        h = KVChatMessageHistory("t1", store)
        h.add_user_message("q")
        self.assertEqual(store.keys(prefix="idx::"), [])

    def test_clear_removes_legacy_idx_key(self):
        store = KVStore(spark=None, table=None)
        store.set("idx::t1", 3)  # left over from the old counter scheme
        h = KVChatMessageHistory("t1", store)
        h.add_user_message("q")
        h.clear()
        self.assertEqual(store.keys(), [])


class TestRoleRoundTrip(unittest.TestCase):
    def test_tool_style_message_not_rewritten_as_human(self):
        from langchain_core.messages import ChatMessage

        store = KVStore(spark=None, table=None)
        h = KVChatMessageHistory("t1", store)
        h.add_message(ChatMessage(role="tool", content="result"))
        msgs = h.messages
        self.assertEqual(len(msgs), 1)
        assert isinstance(msgs[0], ChatMessage)  # .role is ChatMessage-only
        self.assertEqual(msgs[0].role, "tool")
        self.assertEqual(msgs[0].content, "result")

    def test_system_message_roundtrip(self):
        from langchain_core.messages import SystemMessage

        store = KVStore(spark=None, table=None)
        h = KVChatMessageHistory("t1", store)
        h.add_message(SystemMessage(content="be brief"))
        self.assertIsInstance(h.messages[0], SystemMessage)


class TestLosslessSerialisation(unittest.TestCase):
    """The stored payload is message_to_dict, so nothing is dropped."""

    def _history(self) -> KVChatMessageHistory:
        return KVChatMessageHistory("t1", KVStore(spark=None, table=None))

    def test_tool_calls_round_trip(self):
        from langchain_core.messages import AIMessage

        h = self._history()
        h.add_message(
            AIMessage(
                content="",
                tool_calls=[{"name": "run_sql", "args": {"q": "select 1"}, "id": "call-1"}],
            )
        )
        msg = h.messages[0]
        assert isinstance(msg, AIMessage)
        self.assertEqual(msg.tool_calls[0]["name"], "run_sql")
        self.assertEqual(msg.tool_calls[0]["args"], {"q": "select 1"})
        self.assertEqual(msg.tool_calls[0]["id"], "call-1")

    def test_tool_message_round_trips_with_tool_call_id(self):
        from langchain_core.messages import ToolMessage

        h = self._history()
        h.add_message(ToolMessage(content="42", tool_call_id="call-1"))
        msg = h.messages[0]
        assert isinstance(msg, ToolMessage)
        self.assertEqual(msg.tool_call_id, "call-1")

    def test_id_and_response_metadata_round_trip(self):
        from langchain_core.messages import AIMessage

        h = self._history()
        h.add_message(
            AIMessage(
                content="a",
                id="msg-abc",
                response_metadata={"model_name": "m", "finish_reason": "stop"},
            )
        )
        msg = h.messages[0]
        self.assertEqual(msg.id, "msg-abc")
        self.assertEqual(msg.response_metadata["finish_reason"], "stop")

    def test_usage_metadata_summed_in_session_metadata(self):
        from langchain_core.messages import AIMessage, HumanMessage

        h = self._history()
        h.add_messages(
            [
                HumanMessage(content="q1"),
                AIMessage(
                    content="a1",
                    usage_metadata={
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                ),
                AIMessage(
                    content="a2",
                    usage_metadata={
                        "input_tokens": 20,
                        "output_tokens": 7,
                        "total_tokens": 27,
                    },
                ),
            ]
        )
        usage = h.get_session_metadata()["total_usage"]
        self.assertEqual(usage["input_tokens"], 30)
        self.assertEqual(usage["output_tokens"], 12)
        self.assertEqual(usage["total_tokens"], 42)
        # And the usage survives the message round-trip itself.
        msg = h.messages[1]
        assert isinstance(msg, AIMessage)
        assert msg.usage_metadata is not None
        self.assertEqual(msg.usage_metadata["total_tokens"], 15)

    def test_empty_thread_reports_zero_usage(self):
        usage = self._history().get_session_metadata()["total_usage"]
        self.assertEqual(
            usage, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        )

    def test_legacy_row_shape_still_readable(self):
        from langchain_core.messages import HumanMessage

        store = KVStore(spark=None, table=None)
        h = KVChatMessageHistory("t1", store)
        # Row written by the pre-lossless schema.
        store.set(
            "msg::t1::00000001",
            {"role": "human", "content": "old", "meta": {"k": "v"}, "ts": 1.0},
        )
        h.add_user_message("new")
        msgs = h.messages
        self.assertIsInstance(msgs[0], HumanMessage)
        self.assertEqual(msgs[0].content, "old")
        self.assertEqual(msgs[0].additional_kwargs, {"k": "v"})
        self.assertEqual(msgs[1].content, "new")


class TestThreadListingFromStore(unittest.TestCase):
    def test_threads_survive_manager_restart(self):
        """A new manager over the same store must see persisted threads."""
        store = KVStore(spark=None, table=None)
        ThreadMemoryManager(store).get_thread("t1").add_user_message("hi")

        fresh_mgr = ThreadMemoryManager(store)  # simulates process restart
        self.assertEqual(fresh_mgr.list_threads(), ["t1"])
        summaries = fresh_mgr.all_summaries()
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["message_count"], 1)

    def test_underscore_session_ids_isolated_on_clear(self):
        """Clearing thread_1 must not delete threadX1 (the old LIKE-based
        Delta delete treated '_' as a wildcard; in-memory mode was always
        literal — this pins the shared contract)."""
        store = KVStore(spark=None, table=None)
        mgr = ThreadMemoryManager(store)
        mgr.get_thread("thread_1").add_user_message("a")
        mgr.get_thread("threadX1").add_user_message("b")
        mgr.delete_thread("thread_1")
        self.assertEqual(mgr.list_threads(), ["threadX1"])


class TestMessageReadCache(unittest.TestCase):
    """After the first load, .messages fetches only the appended tail."""

    def test_second_read_uses_records_after_not_a_full_rescan(self):
        from unittest import mock

        store = KVStore(spark=None, table=None)
        h = KVChatMessageHistory("t1", store)
        h.add_user_message("q1")
        self.assertEqual([m.content for m in h.messages], ["q1"])

        # Any further full-prefix scan through the history is a regression.
        with mock.patch.object(
            store, "all_records", side_effect=AssertionError("full rescan")
        ):
            h.add_ai_message("a1")
            self.assertEqual([m.content for m in h.messages], ["q1", "a1"])
            self.assertEqual([m.content for m in h.messages], ["q1", "a1"])

    def test_cache_invalidated_by_clear(self):
        store = KVStore(spark=None, table=None)
        h = KVChatMessageHistory("t1", store)
        h.add_user_message("q1")
        _ = h.messages
        h.clear()
        self.assertEqual(h.messages, [])

    def test_cache_invalidated_by_prune(self):
        store = KVStore(spark=None, table=None)
        h = KVChatMessageHistory("t1", store)
        h.add_user_message("q1")
        _ = h.messages
        h.prune_before(float("inf"))
        self.assertEqual(h.messages, [])

    def test_retention_prune_after_write_is_reflected(self):
        store = KVStore(spark=None, table=None)
        h = KVChatMessageHistory("t1", store, max_messages=2)
        h.add_user_message("m1")
        _ = h.messages  # warm the cache before retention prunes m1
        h.add_ai_message("m2")
        h.add_user_message("m3")
        self.assertEqual([m.content for m in h.messages], ["m2", "m3"])

    def test_returned_list_is_a_copy(self):
        store = KVStore(spark=None, table=None)
        h = KVChatMessageHistory("t1", store)
        h.add_user_message("q1")
        first = h.messages
        first.append(first[0])  # mutate the returned list; store must be unaffected
        self.assertEqual(len(h.messages), 1)

    def test_appends_from_another_instance_are_picked_up(self):
        store = KVStore(spark=None, table=None)
        h1 = KVChatMessageHistory("t1", store)
        h1.add_user_message("q1")
        _ = h1.messages
        # A second writer over the same store appends; h1's tail read
        # re-reads from the frontier tick, so it sees them even when both
        # writes share a microsecond (see the same-tick test below).
        KVChatMessageHistory("t1", store).add_user_message("q2")
        self.assertEqual([m.content for m in h1.messages], ["q1", "q2"])

    def test_same_tick_append_with_lower_suffix_is_picked_up(self):
        # Two writers can land on the same microsecond tick (the _last_us
        # bump is per-instance); within a tick, key order falls to the
        # random suffix. A frontier of the full last key made a same-tick
        # append with a lower-sorting suffix permanently invisible to the
        # cached reader — this pins the adversarial ordering directly.
        from langchain_core.messages import HumanMessage, message_to_dict

        store = KVStore(spark=None, table=None)
        h1 = KVChatMessageHistory("t1", store)
        h1.add_user_message("q1")
        _ = h1.messages  # warm the cache

        ((q1_key, q1_rec),) = store.all_records(prefix="msg::t1::")
        tick = q1_key[len("msg::t1::") :].split("-", 1)[0]
        # Suffix "-" sorts below "-<any hex>", i.e. below q1's key.
        store.set(
            f"msg::t1::{tick}-",
            {
                "message": message_to_dict(HumanMessage(content="q2")),
                "ts": q1_rec["value"]["ts"],
            },
            tags=["message", "human", "t1"],
        )
        self.assertEqual(
            sorted(m.content for m in h1.messages), ["q1", "q2"]
        )
        # Refreshing must not duplicate the re-read frontier tick.
        self.assertEqual(len(h1.messages), 2)

    def test_facade_restore_invalidates_thread_caches(self):
        mem = MemoryHub()
        thread = mem.get_thread("t1")
        thread.add_user_message("q1")
        _ = thread.messages
        mem.restore(MemoryHub().snapshot())  # restore an empty snapshot
        self.assertEqual(thread.messages, [])


class TestForkThread(unittest.TestCase):
    def test_fork_copies_messages_in_order_and_leaves_source_intact(self):
        mem = MemoryHub()
        src = mem.get_thread("t1")
        src.add_user_message("q1")
        src.add_ai_message("a1")
        src.add_user_message("q2")
        src.add_ai_message("a2")

        copied = mem.fork_thread("t1", "t2")

        self.assertEqual(copied, 4)
        self.assertEqual(
            [m.content for m in mem.get_thread("t2").messages],
            ["q1", "a1", "q2", "a2"],
        )
        self.assertEqual(len(mem.get_thread("t1").messages), 4)
        self.assertEqual(mem.threads.list_threads(), ["t1", "t2"])

    def test_fork_upto_messages_takes_oldest_prefix(self):
        mem = MemoryHub()
        src = mem.get_thread("t1")
        for i in range(3):
            src.add_user_message(f"q{i}")
            src.add_ai_message(f"a{i}")

        copied = mem.fork_thread("t1", "t2", upto_messages=2)

        self.assertEqual(copied, 2)
        self.assertEqual(
            [m.content for m in mem.get_thread("t2").messages], ["q0", "a0"]
        )

    def test_fork_upto_ts_cuts_strictly_before(self):
        from unittest import mock

        mem = MemoryHub()
        src = mem.get_thread("t1")
        with mock.patch("memory.store._now", return_value=100.0):
            src.add_user_message("old")
        with mock.patch("memory.store._now", return_value=200.0):
            src.add_user_message("new")

        copied = mem.fork_thread("t1", "t2", upto_ts=150.0)

        self.assertEqual(copied, 1)
        self.assertEqual(
            [m.content for m in mem.get_thread("t2").messages], ["old"]
        )

    def test_fork_preserves_key_suffixes_and_rewrites_session_tag(self):
        store = KVStore(spark=None, table=None)
        mgr = ThreadMemoryManager(store)
        mgr.get_thread("t1").add_user_message("q1")
        mgr.fork_thread("t1", "t2")

        src_suffixes = [k[len("msg::t1::") :] for k in store.keys("msg::t1::")]
        dst_suffixes = [k[len("msg::t2::") :] for k in store.keys("msg::t2::")]
        self.assertEqual(sorted(src_suffixes), sorted(dst_suffixes))

        (_, rec), = store.all_records(prefix="msg::t2::")
        self.assertIn("t2", rec["tags"])
        self.assertNotIn("t1", rec["tags"])

    def test_fork_rejects_nonempty_destination_and_self(self):
        mem = MemoryHub()
        mem.get_thread("t1").add_user_message("q")
        mem.get_thread("t2").add_user_message("occupied")
        with self.assertRaises(ValueError):
            mem.fork_thread("t1", "t2")
        with self.assertRaises(ValueError):
            mem.fork_thread("t1", "t1")


class TestRetention(unittest.TestCase):
    def test_max_messages_prunes_oldest_after_write(self):
        store = KVStore(spark=None, table=None)
        h = KVChatMessageHistory("t1", store, max_messages=2)
        h.add_user_message("m1")
        h.add_ai_message("m2")
        h.add_user_message("m3")
        self.assertEqual([m.content for m in h.messages], ["m2", "m3"])

    def test_max_age_prunes_expired_messages(self):
        from unittest import mock

        store = KVStore(spark=None, table=None)
        h = KVChatMessageHistory("t1", store, max_age_s=50.0)
        with mock.patch("memory.store._now", return_value=100.0):
            h.add_user_message("old")
        with mock.patch("memory.store._now", return_value=200.0):
            h.add_user_message("new")
        self.assertEqual([m.content for m in h.messages], ["new"])

    def test_facade_forwards_retention_to_threads(self):
        mem = MemoryHub(retention_max_messages=2)
        t = mem.get_thread("t1")
        for i in range(3):
            t.add_user_message(f"m{i}")
        self.assertEqual([m.content for m in t.messages], ["m1", "m2"])

    def test_no_retention_keeps_everything(self):
        mem = MemoryHub()
        t = mem.get_thread("t1")
        for i in range(5):
            t.add_user_message(f"m{i}")
        self.assertEqual(len(t.messages), 5)

    def test_invalid_retention_params_raise(self):
        store = KVStore(spark=None, table=None)
        with self.assertRaises(ValueError):
            KVChatMessageHistory("t1", store, max_age_s=0)
        with self.assertRaises(ValueError):
            KVChatMessageHistory("t1", store, max_messages=0)


class TestPruneBefore(unittest.TestCase):
    def test_prune_removes_only_older_messages(self):
        from unittest import mock

        store = KVStore(spark=None, table=None)
        h = KVChatMessageHistory("t1", store)
        # Pin the clock: consecutive real writes can share a time.time()
        # float, which would make any cutoff between them ambiguous.
        with mock.patch("memory.store._now", return_value=100.0):
            h.add_user_message("old")
        with mock.patch("memory.store._now", return_value=200.0):
            h.add_ai_message("new")
        removed = h.prune_before(150.0)
        self.assertEqual(removed, 1)
        self.assertEqual([m.content for m in h.messages], ["new"])

    def test_prune_via_facade(self):
        mem = MemoryHub(spark=None, table=None)
        mem.get_thread("t1").add_user_message("only")
        import time as _time

        removed = mem.prune_thread("t1", _time.time() + 10)
        self.assertEqual(removed, 1)
        self.assertFalse(mem.get_thread("t1").has_messages())


class TestTruncateTo(unittest.TestCase):
    def test_truncate_keeps_oldest_and_drops_newest(self):
        from langchain_core.messages import HumanMessage

        store = KVStore(spark=None, table=None)
        h = KVChatMessageHistory("t1", store)
        for i in range(6):
            h.add_message(HumanMessage(content=f"m{i}"))

        removed = h.truncate_to(4)
        self.assertEqual(removed, 2)
        # The two NEWEST were dropped (unlike prune_to_count, which drops old).
        self.assertEqual([m.content for m in h.messages], ["m0", "m1", "m2", "m3"])

    def test_truncate_noop_when_within_budget(self):
        store = KVStore(spark=None, table=None)
        h = KVChatMessageHistory("t1", store)
        h.add_user_message("a")
        h.add_ai_message("b")
        self.assertEqual(h.truncate_to(2), 0)
        self.assertEqual(h.truncate_to(5), 0)
        self.assertEqual(len(h.messages), 2)

    def test_truncate_to_zero_clears_and_allows_clean_reappend(self):
        store = KVStore(spark=None, table=None)
        h = KVChatMessageHistory("t1", store)
        h.add_user_message("a")
        h.add_ai_message("b")
        self.assertEqual(h.truncate_to(0), 2)
        self.assertEqual(h.messages, [])
        # Re-appends after a rewind still sort after nothing — order is sane.
        h.add_user_message("c")
        self.assertEqual([m.content for m in h.messages], ["c"])

    def test_truncate_rejects_negative(self):
        store = KVStore(spark=None, table=None)
        h = KVChatMessageHistory("t1", store)
        with self.assertRaises(ValueError):
            h.truncate_to(-1)


if __name__ == "__main__":
    unittest.main()
