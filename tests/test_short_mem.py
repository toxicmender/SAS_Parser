"""
test_short_mem.py — unit tests for the in-memory (table=None) fast-paths of
memory.short_mem.SparkKVStore and the layers built on top of it.

These deliberately construct the store with ``spark=None, table=None``: in
in-memory mode none of the exercised methods touch Spark, so the suite runs
without a JVM / SparkSession (the Delta-mode branches are not covered here).
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from memory.short_mem import (
    DatabricksMemory,
    KVChatMessageHistory,
    KVMemoryStore,
    SparkKVStore,
    ThreadMemoryManager,
    _sql_quote,
)


class TestSparkKVStoreInMemory(unittest.TestCase):
    def _store(self) -> SparkKVStore:
        return SparkKVStore(spark=None, table=None)

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
        self.assertEqual(stats["backend"], "Spark in-memory")
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
        store = SparkKVStore(spark=None, table=None)
        h = KVChatMessageHistory("t1", store)
        self.assertFalse(h.has_messages())
        h.add_user_message("q1")
        h.add_ai_message("a1")
        self.assertTrue(h.has_messages())
        self.assertEqual([m.content for m in h.messages], ["q1", "a1"])

    def test_clear_removes_messages(self):
        store = SparkKVStore(spark=None, table=None)
        h = KVChatMessageHistory("t1", store)
        h.add_user_message("q1")
        h.clear()
        self.assertFalse(h.has_messages())
        self.assertEqual(h.messages, [])

    def test_list_threads_only_nonempty(self):
        store = SparkKVStore(spark=None, table=None)
        mgr = ThreadMemoryManager(store)
        mgr.get_thread("t1").add_user_message("hi")
        mgr.get_thread("empty")  # instantiated but no messages
        self.assertEqual(mgr.list_threads(), ["t1"])


class TestFacadeInMemory(unittest.TestCase):
    def test_kv_store_namespacing_and_search(self):
        mem = DatabricksMemory(spark=None, table=None)
        mem.kv.set("goal", "RAG pipeline", tags=["project"])
        self.assertEqual(mem.kv.get("goal"), "RAG pipeline")
        self.assertEqual(mem.kv.search("pipeline")[0][0], "goal")

    def test_summary_reports_backend(self):
        mem = DatabricksMemory(spark=None, table=None)
        self.assertEqual(mem.summary()["store"]["backend"], "Spark in-memory")

    def test_kvmemorystore_strips_namespace_in_keys(self):
        store = SparkKVStore(spark=None, table=None)
        kv = KVMemoryStore(store, namespace="kv")
        kv.set("a", 1)
        kv.set("b", 2)
        self.assertEqual(sorted(kv.keys()), ["a", "b"])


class TestSqlQuote(unittest.TestCase):
    def test_doubles_single_quotes(self):
        self.assertEqual(_sql_quote("o'brien"), "o''brien")
        self.assertEqual(_sql_quote("plain"), "plain")


if __name__ == "__main__":
    unittest.main()
