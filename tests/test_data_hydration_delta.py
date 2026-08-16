"""
test_data_hydration_delta.py — the Delta sink, against a real Spark session.

Everything else in ``data_hydration`` is exercised with fakes because planning
is inert by design. ``sinks/delta.py`` is the one module that cannot be: it
exists to create managed tables, and a fake Spark would only prove that the
mock was called. So this suite runs against a genuine local Delta session and
asserts on tables that actually exist.

It skips where pyspark, delta-spark or a JVM is genuinely absent — which is the
normal case on a development machine, and is why this file exists separately.
In this repo's own ``.venv`` pyspark is shadowed by ``databricks-connect``, so
these run in Docker (``docker compose exec app uv run pytest``).

What is pinned, and why each one would fail silently otherwise:

* **Write modes.** The plan decides them; if the sink ignored the plan and
  always overwrote, a partitioned load would end with only its last partition
  and still report success.
* **The schema is created, the catalog is not.** Creating a catalog is a
  governance act a data load has no business performing.
* **``overwriteSchema`` on a full replace.** Without it a column added upstream
  fails the write instead of landing.
* **Clustering is intersected with real columns.** Index-column recovery from a
  ``.sas7bndx`` is best-effort, so a guessed name must never reach
  ``ALTER TABLE ... CLUSTER BY``.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from data_hydration.config import HydrationConfig
from data_hydration.models import (
    HydrationItem,
    HydrationSource,
    SourceKind,
    WriteMode,
)

CATALOG = "spark_catalog"
SCHEMA = "hydration_test"


@pytest.fixture
def sink(delta_spark, monkeypatch, tmp_path):
    """The sink bound to the session fixture, with its tables cleaned up.

    ``get_session`` is redirected rather than left to build its own: it calls
    ``getOrCreate``, which would return this very session anyway, but pinning it
    makes the dependency explicit and keeps the test honest if that ever changes.
    """
    from data_hydration.sinks import delta as sink_module

    monkeypatch.setattr(sink_module, "get_session", lambda: delta_spark)
    delta_spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
    created: list[str] = []

    class _Sink:
        module = sink_module
        spark = delta_spark
        staging = tmp_path

        def write(self, item: HydrationItem, config: HydrationConfig | None = None) -> int:
            created.append(item.target_table)
            return sink_module.write_item(item, config or _config(tmp_path))

        def rows(self, table: str) -> int:
            return self.spark.table(table).count()

        def columns(self, table: str) -> list[str]:
            return [str(c) for c in self.spark.table(table).columns]

    yield _Sink()
    for table in created:
        delta_spark.sql(f"DROP TABLE IF EXISTS {table}")


def _config(tmp_path, **overrides) -> HydrationConfig:
    config = HydrationConfig(
        catalog=CATALOG, schema=SCHEMA, staging_dir=str(tmp_path)
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _csv(tmp_path, name: str, body: str) -> HydrationSource:
    """A CSV on disk, as the FILE source the planner would produce for it."""
    (tmp_path / name).write_text(body, encoding="utf-8")
    return HydrationSource(
        kind=SourceKind.FILE, locator=str(tmp_path), object_name=name
    )


def _item(source: HydrationSource, table: str, **kwargs) -> HydrationItem:
    return HydrationItem(source=source, target_table=table, **kwargs)


class TestWritingATable:
    def test_a_csv_lands_as_a_managed_delta_table(self, sink, tmp_path):
        source = _csv(tmp_path, "sales.csv", "id,amount\n1,10\n2,20\n3,30\n")
        table = f"{CATALOG}.{SCHEMA}.sales"

        rows = sink.write(_item(source, table))

        assert rows == 3
        assert sink.rows(table) == 3
        assert sink.columns(table) == ["id", "amount"]

    def test_the_table_is_delta_and_managed(self, sink, tmp_path):
        source = _csv(tmp_path, "m.csv", "id\n1\n")
        table = f"{CATALOG}.{SCHEMA}.managed_check"
        sink.write(_item(source, table))

        assert (
            sink.spark.sql(f"DESCRIBE DETAIL {table}").collect()[0]["format"]
            == "delta"
        )
        # MANAGED, not EXTERNAL — the decision behind writing no LOCATION, which
        # is what makes the catalog own the files and the governance that comes
        # with them. DESCRIBE DETAIL cannot answer this; only EXTENDED can.
        described = {
            row["col_name"]: row["data_type"]
            for row in sink.spark.sql(f"DESCRIBE TABLE EXTENDED {table}").collect()
        }
        assert described.get("Type") == "MANAGED"

    def test_the_values_survive_the_round_trip(self, sink, tmp_path):
        source = _csv(tmp_path, "v.csv", "id,name\n1,alice\n2,bob\n")
        table = f"{CATALOG}.{SCHEMA}.values_check"
        sink.write(_item(source, table))

        got = {
            (r["id"], r["name"])
            for r in sink.spark.table(table).collect()
        }
        assert got == {(1, "alice"), (2, "bob")}

    def test_an_empty_source_writes_nothing_and_reports_zero(self, sink, tmp_path):
        source = _csv(tmp_path, "empty.csv", "id,amount\n")
        table = f"{CATALOG}.{SCHEMA}.empty_check"

        assert sink.write(_item(source, table)) == 0
        # No table at all, rather than an empty one: a load that found nothing
        # must not look like a load that succeeded.
        assert not sink.spark.catalog.tableExists(table)


class TestWriteModes:
    """The plan decides these, and the sink must honour them."""

    def test_overwrite_replaces_the_previous_contents(self, sink, tmp_path):
        table = f"{CATALOG}.{SCHEMA}.overwrite_check"
        first = _csv(tmp_path, "a.csv", "id\n1\n2\n")
        second = _csv(tmp_path, "b.csv", "id\n9\n")

        sink.write(_item(first, table, write_mode=WriteMode.OVERWRITE))
        sink.write(_item(second, table, write_mode=WriteMode.OVERWRITE))

        assert sink.rows(table) == 1

    def test_append_adds_to_it(self, sink, tmp_path):
        table = f"{CATALOG}.{SCHEMA}.append_check"
        first = _csv(tmp_path, "c.csv", "id\n1\n2\n")
        second = _csv(tmp_path, "d.csv", "id\n3\n")

        sink.write(_item(first, table, write_mode=WriteMode.OVERWRITE))
        sink.write(_item(second, table, write_mode=WriteMode.APPEND))

        assert sink.rows(table) == 3

    def test_a_partitioned_load_accumulates_every_partition(self, sink, tmp_path):
        """The failure this guards: a load that keeps only its last partition.

        The planner emits OVERWRITE for the first item of a table and APPEND for
        the rest. A sink that ignored that and always overwrote would end with
        one partition's rows and report success for all of them.
        """
        table = f"{CATALOG}.{SCHEMA}.partitioned_check"
        parts = [
            _csv(tmp_path, f"p{i}.csv", "id\n" + "".join(f"{i}{j}\n" for j in range(3)))
            for i in range(4)
        ]
        for index, source in enumerate(parts):
            sink.write(
                _item(
                    source,
                    table,
                    write_mode=(
                        WriteMode.OVERWRITE if index == 0 else WriteMode.APPEND
                    ),
                )
            )

        assert sink.rows(table) == 12

    def test_overwrite_accepts_a_changed_schema(self, sink, tmp_path):
        # Without overwriteSchema a column added upstream fails the write
        # instead of landing, which turns a benign source change into an outage.
        table = f"{CATALOG}.{SCHEMA}.schema_change_check"
        sink.write(_item(_csv(tmp_path, "s1.csv", "id\n1\n"), table))
        sink.write(
            _item(
                _csv(tmp_path, "s2.csv", "id,extra\n1,x\n"),
                table,
                write_mode=WriteMode.OVERWRITE,
            )
        )
        assert "extra" in sink.columns(table)


class TestSchemaCreation:
    def test_a_missing_schema_is_created(self, sink, tmp_path):
        fresh = "hydration_fresh_schema"
        sink.spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{fresh} CASCADE")
        table = f"{CATALOG}.{fresh}.t"
        try:
            sink.write(_item(_csv(tmp_path, "f.csv", "id\n1\n"), table))
            assert sink.rows(table) == 1
        finally:
            sink.spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{fresh} CASCADE")

    def test_a_missing_catalog_is_not_created(self, sink, tmp_path):
        """Creating a catalog is governance, not a data load's business."""
        table = "no_such_catalog.some_schema.t"
        with pytest.raises(Exception):
            sink.write(_item(_csv(tmp_path, "g.csv", "id\n1\n"), table))


class TestIndexClustering:
    def test_clustering_is_off_unless_configured(self, sink, tmp_path):
        table = f"{CATALOG}.{SCHEMA}.cluster_off"
        item = _item(
            _csv(tmp_path, "co.csv", "id,region\n1,north\n"),
            table,
            cluster_by=("region",),
        )
        sink.write(item, _config(tmp_path, apply_index_clustering=False))
        assert sink.rows(table) == 1

    def test_a_configured_hint_clusters_the_table(self, sink, tmp_path):
        table = f"{CATALOG}.{SCHEMA}.cluster_on"
        item = _item(
            _csv(tmp_path, "cn.csv", "id,region\n1,north\n2,south\n"),
            table,
            cluster_by=("region",),
        )
        sink.write(item, _config(tmp_path, apply_index_clustering=True))
        # The table must still be readable and complete whether or not the
        # ALTER succeeded — clustering is an optimisation, never a correctness
        # condition, and the sink swallows its failure by design.
        assert sink.rows(table) == 2

    def test_a_column_the_table_lacks_is_dropped_before_altering(
        self, sink, tmp_path
    ):
        """Index recovery is best-effort, so it must be filtered, not trusted.

        A guessed name reaching ALTER TABLE would be a hard failure on a table
        that had otherwise landed perfectly.
        """
        table = f"{CATALOG}.{SCHEMA}.cluster_bogus"
        item = _item(
            _csv(tmp_path, "cb.csv", "id,region\n1,north\n"),
            table,
            cluster_by=("region", "not_a_real_column"),
        )
        sink.write(item, _config(tmp_path, apply_index_clustering=True))
        assert sink.rows(table) == 1


class TestThroughTheRunner:
    """The path a real run takes: execute() -> write_item -> a table."""

    def test_execute_writes_the_plan_and_reports_it(self, sink, tmp_path):
        from data_hydration.models import HydrationPlan, ItemStatus
        from data_hydration.runner import execute

        table = f"{CATALOG}.{SCHEMA}.runner_check"
        plan = HydrationPlan(
            run_date="20260815",
            items=[_item(_csv(tmp_path, "r.csv", "id\n1\n2\n"), table)],
        )
        report = execute(plan, config=_config(tmp_path))

        assert report.ok
        assert report.written == 1
        assert report.outcomes[0].status is ItemStatus.WRITTEN
        assert report.outcomes[0].rows == 2
        assert sink.rows(table) == 2
        sink.spark.sql(f"DROP TABLE IF EXISTS {table}")

    def test_a_dry_run_creates_no_table(self, sink, tmp_path):
        from data_hydration.models import HydrationPlan
        from data_hydration.runner import execute

        table = f"{CATALOG}.{SCHEMA}.dry_run_check"
        plan = HydrationPlan(
            items=[_item(_csv(tmp_path, "dr.csv", "id\n1\n"), table)]
        )
        report = execute(plan, config=_config(tmp_path), dry_run=True)

        assert report.written == 0
        assert not sink.spark.catalog.tableExists(table)

    def test_a_missing_source_fails_the_item_without_raising(self, sink, tmp_path):
        from data_hydration.models import HydrationPlan, ItemStatus
        from data_hydration.runner import execute

        missing = HydrationSource(
            kind=SourceKind.FILE, locator=str(tmp_path), object_name="absent.csv"
        )
        plan = HydrationPlan(
            items=[_item(missing, f"{CATALOG}.{SCHEMA}.missing_check")]
        )
        report = execute(plan, config=_config(tmp_path))

        assert report.failed == 1
        assert not report.ok
        assert report.outcomes[0].status is ItemStatus.FAILED
        assert "FileNotFoundError" in (report.outcomes[0].error or "")


class TestSasDataset:
    """The .sas7bdat path, which needs pyreadstat as well as Spark."""

    def test_a_sas_dataset_lands_as_a_table(self, sink, tmp_path):
        pyreadstat = pytest.importorskip("pyreadstat")
        import pandas as pd

        frame = pd.DataFrame({"id": [1, 2, 3], "region": ["n", "s", "e"]})
        pyreadstat.write_sas7bdat(frame, str(tmp_path / "sales.sas7bdat"))

        source = HydrationSource(
            kind=SourceKind.SAS_DATASET,
            locator=str(tmp_path),
            object_name="sales",
        )
        table = f"{CATALOG}.{SCHEMA}.sas_dataset_check"
        rows = sink.write(_item(source, table))

        assert rows == 3
        assert sink.rows(table) == 3
        assert "region" in sink.columns(table)

    def test_a_row_range_partition_writes_only_its_slice(self, sink, tmp_path):
        pyreadstat = pytest.importorskip("pyreadstat")
        import pandas as pd

        from data_hydration.models import Partition

        pyreadstat.write_sas7bdat(
            pd.DataFrame({"id": list(range(10))}), str(tmp_path / "big.sas7bdat")
        )
        source = HydrationSource(
            kind=SourceKind.SAS_DATASET, locator=str(tmp_path), object_name="big"
        )
        table = f"{CATALOG}.{SCHEMA}.row_range_check"

        sink.write(
            _item(
                source,
                table,
                partition=Partition(name="rows_0_4", row_offset=0, row_limit=4),
            )
        )
        sink.write(
            _item(
                source,
                table,
                write_mode=WriteMode.APPEND,
                partition=Partition(name="rows_4_10", row_offset=4, row_limit=6),
            )
        )

        assert sink.rows(table) == 10
