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
        table = f"default.kv_contract_{uuid.uuid4().hex[:10]}"
        self._tables.append(table)
        return KVStore(spark=self._spark, table=table)

    # ---- Change Data Feed setup -------------------------------------------

    def test_a_new_table_gets_cdf_without_a_second_write(self):
        """The CREATE enables CDF; the ALTER must not re-issue it.

        Not a micro-optimisation. ``ALTER TABLE ... SET TBLPROPERTIES`` reaches
        catalyst internals that a delta-spark built against an older Spark minor
        cannot resolve, so re-setting a property the CREATE had already set made
        every test in this class fail on a pairing where the backend otherwise
        works perfectly. Issuing it only when the property is genuinely absent
        is what keeps a fresh table off that code path.
        """
        store = self.make_store()
        backend = store._backend  # the Delta backend under test

        properties = {
            row["key"]: row["value"]
            for row in self._spark.sql(
                f"SHOW TBLPROPERTIES {backend._table}"
            ).collect()
        }
        assert properties.get("delta.enableChangeDataFeed") == "true"
        assert backend._cdf_enabled() is True

    def test_cdf_is_re_enabled_when_a_table_lacks_it(self):
        """The upgrade path the ALTER exists for still works.

        A table created without CDF — what a deployment predating it has — must
        still get the property when the backend opens it. Skipped, not failed,
        where the pairing cannot write properties at all: that is the known
        limitation ``docker/spark/warmup.py`` warns about, and it is a
        different defect from the one this test guards.
        """
        table = f"default.kv_precdf_{uuid.uuid4().hex[:10]}"
        self._tables.append(table)
        self._spark.sql(
            f"CREATE TABLE {table} ("
            " kv_key STRING NOT NULL, value STRING NOT NULL, tags STRING,"
            " created_at DOUBLE, updated_at DOUBLE, source STRING"
            ") USING DELTA"
        )
        try:
            self._spark.sql(
                f"ALTER TABLE {table} SET TBLPROPERTIES "
                "('delta.enableChangeDataFeed' = 'true')"
            )
        except Exception as exc:
            pytest.skip(
                f"this pyspark/delta-spark pair cannot write table properties "
                f"({type(exc).__name__}); the upgrade path is unavailable here"
            )
        self._spark.sql(
            f"ALTER TABLE {table} UNSET TBLPROPERTIES "
            "('delta.enableChangeDataFeed')"
        )

        store = KVStore(spark=self._spark, table=table)
        assert store._backend._cdf_enabled() is True
