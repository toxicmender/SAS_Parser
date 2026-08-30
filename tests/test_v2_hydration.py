"""V2 hydration planning, execution, adapter, and regression contracts."""

from __future__ import annotations

import importlib
import io
import subprocess
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from sas_migrate.adapters.hydration import (
    CONCRETE_DRIVER_KINDS,
    DeltaHydrationSink,
    FilesystemSpdeProbe,
    HydrationDriverUnavailable,
    LazyHydrationDriverRegistry,
    LocalFileDriver,
    RangedRawIO,
    SasDatasetDriver,
    open_buffered,
    require_optional_dependency,
)
from sas_migrate.application.hydration import (
    UNRESOLVED_TARGET,
    HydrationItem,
    HydrationItemOutcome,
    HydrationPartition,
    HydrationPlan,
    HydrationSettings,
    HydrationSource,
    HydrationWorkflow,
    ItemStatus,
    PartitionStrategy,
    SourceKind,
    WriteMode,
    build_corpus_plan,
    build_plan,
)
from sas_migrate.application.hydration.naming import (
    TableNameError,
    render,
    sanitise_part,
)
from sas_migrate.application.hydration.partitioning import plan_partitions
from sas_migrate.core.sas import extract_engine_refs, extract_paths

ROOT = Path(__file__).resolve().parents[1]


def _settings(**updates: Any) -> HydrationSettings:
    values = {
        "catalog": "main",
        "schema_name": "staging",
        "planned_at": datetime(2026, 8, 26, 23, 59, tzinfo=UTC),
    }
    values.update(updates)
    return HydrationSettings(**values)


def _plan(source: str, **updates: Any) -> HydrationPlan:
    return build_plan(
        extract_engine_refs(source),
        extract_paths(source),
        settings=_settings(**updates),
    )


def _item(
    *,
    kind: SourceKind = SourceKind.FILE,
    table: str = "main.staging.sales",
    blockers: tuple[str, ...] = (),
    write_mode: WriteMode = WriteMode.OVERWRITE,
    partition: HydrationPartition | None = None,
) -> HydrationItem:
    return HydrationItem(
        source=HydrationSource(
            kind=kind,
            locator="/data",
            object_name="sales",
            source_name="sales.csv",
        ),
        target_table=table,
        blockers=blockers,
        write_mode=write_mode,
        partition=partition,
    )


def test_importing_hydration_loads_no_legacy_or_optional_driver() -> None:
    code = (
        "import sys; import sas_migrate.application.hydration; "
        "import sas_migrate.adapters.hydration; "
        "blocked=('data_hydration','pyspark','oracledb','paramiko','pyreadstat','saspy','azure'); "
        "assert not [name for name in blocked if name in sys.modules]"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_oracle_libname_and_macro_blocker_match_characterized_behavior() -> None:
    item = _plan(
        'libname edwprod oracle path=EDWPRO schema=fr_dm user="&u." pass="&p.";'
    ).items[0]
    assert item.source.kind is SourceKind.ORACLE
    assert item.source.engine == "oracle"
    assert item.source.locator == "EDWPRO"
    assert item.source.libref == "edwprod"
    assert item.target_table == "main.staging.fr_dm"
    assert "pass" in item.blockers[0] and "user" in item.blockers[0]


@pytest.mark.parametrize(
    ("source", "kind"),
    [
        ("infile '/data/marts/sales.sas7bdat';", SourceKind.SAS_DATASET),
        ("filename raw ftp '/incoming/cust.csv' host='h';", SourceKind.SFTP),
        ("filename raw sftp '/incoming/cust.csv' host='h';", SourceKind.SFTP),
        ("filename raw azure '/incoming/cust.csv';", SourceKind.BLOB),
        ("libname raw spde '/data/spde';", SourceKind.SPDE),
        ("libname raw '/data/plain';", SourceKind.FILE),
    ],
)
def test_path_projection_classifies_supported_sources(source: str, kind: SourceKind) -> None:
    assert _plan(source).items[0].source.kind is kind


@pytest.mark.parametrize(
    "source",
    [
        "%include '/code/common.sas';",
        "filename mail email 'ops@example.com';",
        "filename cmd pipe 'ls -l';",
        "ods html file='/reports/out.html';",
    ],
)
def test_non_data_references_are_not_hydration_items(source: str) -> None:
    assert _plan(source).items == ()


def test_file_source_preserves_physical_name_and_logical_table_name() -> None:
    item = _plan("infile '/data/sales-2026.csv';").items[0]
    assert item.source.object_name == "sales-2026"
    assert item.source.source_name == "sales-2026.csv"
    assert item.target_table.endswith(".sales_2026")


def test_directory_and_missing_schema_block_only_affected_items() -> None:
    source = "libname good oracle path=P schema=accounts;\nlibname flat '/data/library';"
    plan = build_plan(
        extract_engine_refs(source), extract_paths(source), settings=_settings(schema_name=None)
    )
    assert len(plan.items) == 2
    assert plan.items[0].target_table == "main.good.accounts"
    assert plan.items[1].target_table == UNRESOLVED_TARGET
    assert "library directory" in plan.items[1].blockers[0]


def test_spde_requires_session_but_probe_is_not_called_for_static_plan() -> None:
    item = _plan("libname raw spde '/data/spde';").items[0]
    assert item.needs_operator_input
    assert "SAS session" in item.blockers[0]
    assert item.strategy is PartitionStrategy.NONE
    assert "without a live source probe" in item.strategy_reason
    assert _plan(
        "libname raw spde '/data/spde';", sas_session_configured=True
    ).items[0].blockers == ()


def test_sas_index_is_a_note_and_never_a_source_or_cluster_placeholder() -> None:
    plan = _plan(
        "infile '/data/sales.sas7bdat';\nfilename idx '/data/sales.sas7bndx';"
    )
    assert len(plan.items) == 1
    assert "SAS index" in plan.items[0].notes[0]
    assert plan.items[0].cluster_by == ()


def test_target_template_validation_and_sanitisation() -> None:
    assert sanitise_part(" Pre - Prod ") == "pre_prod"
    assert render("<catalog_name>.<schema_name>.<table_name>", catalog_name="Main", schema_name="Raw", table_name="A-B") == "main.raw.a_b"
    with pytest.raises(TableNameError):
        _plan("libname a oracle path=P schema=t;", table_template="<catalog_name>.<unknown>")
    with pytest.raises(TableNameError):
        render("<catalog_name>.<table_name>", catalog_name="main", table_name="t")


def test_duplicate_targets_get_one_overwrite_then_append_and_one_date() -> None:
    source = "libname edw oracle path=P schema=accounts;"
    plan = build_corpus_plan(
        {
            "a.sas": (extract_engine_refs(source), ()),
            "b.sas": (extract_engine_refs(source), ()),
        },
        settings=_settings(table_template="<catalog_name>.<schema_name>.<table_name>_<date>"),
    )
    assert [item.write_mode for item in plan.items] == [WriteMode.OVERWRITE, WriteMode.APPEND]
    assert plan.target_tables == ("main.staging.accounts_20260826",)
    assert len(plan.by_source_id("a.sas")) == 1
    assert plan.counts_by_kind() == {"oracle": 2}


class _Probe:
    def __init__(self, *, native=(), rows=None, ranged=None, explode=False) -> None:
        self.native = native
        self.rows = rows
        self.ranged = ranged
        self.explode = explode

    def native_partitions(self, source: HydrationSource):
        if self.explode:
            raise RuntimeError("unavailable")
        return self.native

    def row_count(self, source: HydrationSource):
        if self.explode:
            raise RuntimeError("unavailable")
        return self.rows

    def range_column(self, source: HydrationSource):
        if self.explode:
            raise RuntimeError("unavailable")
        return self.ranged


def test_oracle_native_partitions_fan_out_with_correct_write_modes() -> None:
    plan = build_plan(
        extract_engine_refs("libname edw oracle path=P schema=sales;"),
        settings=_settings(),
        probe=_Probe(native=("P1", "P2")),
    )
    assert [item.partition.name for item in plan.items if item.partition] == ["P1", "P2"]
    assert [item.write_mode for item in plan.items] == [WriteMode.OVERWRITE, WriteMode.APPEND]


def test_column_ranges_include_maximum_and_probe_errors_downgrade() -> None:
    source = HydrationSource(kind=SourceKind.ORACLE, object_name="s.t")
    ranged = plan_partitions(source, num_partitions=4, probe=_Probe(ranged=("id", 0.0, 100.0)))
    assert ranged.strategy is PartitionStrategy.COLUMN_RANGE
    assert "<= 100.0" in (ranged.partitions[-1].predicate or "")
    assert plan_partitions(source, probe=_Probe(explode=True)).strategy is PartitionStrategy.NONE


def test_sas_row_ranges_are_contiguous_and_bounded() -> None:
    source = HydrationSource(kind=SourceKind.SAS_DATASET, object_name="sales")
    planned = plan_partitions(source, num_partitions=3, probe=_Probe(rows=8))
    assert planned.strategy is PartitionStrategy.ROW_RANGE
    assert [(part.row_offset, part.row_limit) for part in planned.partitions] == [
        (0, 3), (3, 3), (6, 2)
    ]
    assert plan_partitions(source, num_partitions=1, probe=_Probe(rows=8)).partitions == ()


def test_spde_probe_counts_components_without_fanning_out(tmp_path: Path) -> None:
    for name in ("sales.0.dpf", "sales.1.dpf", "other.0.dpf", "sales.mdf"):
        (tmp_path / name).write_bytes(b"")
    source = HydrationSource(kind=SourceKind.SPDE, locator=str(tmp_path), object_name="sales")
    probe = FilesystemSpdeProbe()
    planned = plan_partitions(source, probe=probe)
    assert planned.partitions == ()
    assert "2 component(s)" in planned.reason
    assert probe.row_count(source) is None and probe.range_column(source) is None


def test_contract_validation_and_round_trip() -> None:
    settings = _settings()
    assert HydrationSettings.from_json(settings.to_json()) == settings
    with pytest.raises(ValidationError):
        HydrationSettings(num_partitions=0)
    with pytest.raises(ValidationError):
        HydrationItemOutcome(item=_item(), status=ItemStatus.FAILED)
    with pytest.raises(ValidationError):
        HydrationItemOutcome(item=_item(), status=ItemStatus.WRITTEN, error="bad")


class _Driver:
    def __init__(self, batches: tuple[Any, ...] = ([1, 2],), *, fail=False) -> None:
        self.values = batches
        self.fail = fail
        self.closed = False

    def batches(self, item: HydrationItem):
        if self.fail:
            raise OSError("source down")
        return self.values

    def close(self) -> None:
        self.closed = True


class _Registry:
    def __init__(self, drivers: list[_Driver]) -> None:
        self.drivers = drivers

    def driver_for(self, kind: SourceKind) -> _Driver:
        del kind
        return self.drivers.pop(0)


class _Sink:
    def __init__(self, *, fail=False) -> None:
        self.fail = fail

    def write(self, item: HydrationItem, batches) -> int:
        del item
        if self.fail:
            raise RuntimeError("write rejected")
        return sum(len(batch) for batch in batches)


def test_workflow_dry_run_and_blockers_do_not_resolve_drivers() -> None:
    plan = HydrationPlan(items=(_item(), _item(blockers=("operator input",))))
    report = HydrationWorkflow(drivers=_Registry([]), sink=_Sink()).run(plan, dry_run=True)
    assert report.ok and report.skipped == 2 and report.dry_run
    blocked = HydrationWorkflow(drivers=_Registry([]), sink=_Sink()).run(
        HydrationPlan(items=(_item(blockers=("operator input",)),))
    )
    assert blocked.skipped == 1 and blocked.outcomes[0].error == "operator input"


def test_workflow_writes_closes_and_isolates_failures() -> None:
    failing, healthy = _Driver(fail=True), _Driver()
    report = HydrationWorkflow(
        drivers=_Registry([failing, healthy]), sink=_Sink()
    ).run(HydrationPlan(items=(_item(), _item())))
    assert report.failed == 1 and report.written == 1 and not report.ok
    assert "OSError" in (report.outcomes[0].error or "")
    assert failing.closed and healthy.closed


def test_workflow_stop_policy_and_invalid_policy() -> None:
    workflow = HydrationWorkflow(drivers=_Registry([_Driver(fail=True)]), sink=_Sink())
    report = workflow.run(HydrationPlan(items=(_item(), _item())), on_error="stop")
    assert len(report.outcomes) == 1
    with pytest.raises(ValueError):
        HydrationWorkflow(drivers=_Registry([]), sink=_Sink()).run(HydrationPlan(), on_error="bad")


def test_lazy_registry_constructs_only_requested_factory() -> None:
    calls: list[str] = []
    registry = LazyHydrationDriverRegistry(
        {
            SourceKind.FILE: lambda: (calls.append("file"), _Driver())[1],
            SourceKind.ORACLE: lambda: (calls.append("oracle"), _Driver())[1],
        }
    )
    assert isinstance(registry.driver_for(SourceKind.FILE), _Driver)
    assert calls == ["file"]
    with pytest.raises(HydrationDriverUnavailable, match=r"sas-parser\[hydration\]"):
        registry.driver_for(SourceKind.SFTP)


def test_concrete_driver_inventory_does_not_confuse_imports_with_adapters() -> None:
    assert CONCRETE_DRIVER_KINDS == {SourceKind.FILE, SourceKind.SAS_DATASET}


def test_optional_dependency_error_names_kind_and_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = importlib.import_module

    def fake_import(name: str):
        if name == "oracledb":
            raise ImportError("missing")
        return real_import(name)

    monkeypatch.setattr("sas_migrate.adapters.hydration.drivers.importlib.import_module", fake_import)
    with pytest.raises(HydrationDriverUnavailable, match="oracledb.*hydration"):
        require_optional_dependency(SourceKind.ORACLE)


def test_sas_dataset_driver_passes_partition_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, int]]] = []
    module = types.SimpleNamespace(
        read_sas7bdat=lambda path, **kwargs: (calls.append((path, kwargs)) or [1, 2], object())
    )
    monkeypatch.setattr(
        "sas_migrate.adapters.hydration.drivers.importlib.import_module", lambda name: module
    )
    driver = SasDatasetDriver()
    item = HydrationItem(
        source=HydrationSource(
            kind=SourceKind.SAS_DATASET,
            locator="/data",
            object_name="sales",
            source_name="physical.sas7bdat",
        ),
        target_table="main.s.sales",
        partition=HydrationPartition(name="r", row_offset=4, row_limit=6),
    )
    assert list(driver.batches(item)) == [[1, 2]]
    assert calls[0][0].replace("\\", "/").endswith("/data/physical.sas7bdat")
    assert calls[0][1] == {"row_offset": 4, "row_limit": 6}
    assert driver.close() is None


def test_local_file_driver_uses_physical_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []
    pandas = types.SimpleNamespace(
        read_parquet=lambda path: calls.append(("parquet", path)) or [1],
        read_excel=lambda path: calls.append(("excel", path)) or [1],
        read_csv=lambda path, **kwargs: calls.append(("csv", path, kwargs)) or ([1],),
    )
    monkeypatch.setattr(
        "sas_migrate.adapters.hydration.drivers.importlib.import_module", lambda name: pandas
    )
    driver = LocalFileDriver(batch_rows=25)
    list(driver.batches(_item()))
    parquet = _item().model_copy(
        update={"source": _item().source.model_copy(update={"source_name": "sales.parquet"})}
    )
    excel = _item().model_copy(
        update={"source": _item().source.model_copy(update={"source_name": "sales.xlsx"})}
    )
    list(driver.batches(parquet)); list(driver.batches(excel))
    assert calls[0][0] == "csv" and calls[0][2]["chunksize"] == 25
    assert [call[0] for call in calls] == ["csv", "parquet", "excel"]


class _Object:
    def __init__(self, data: bytes = bytes(range(128)) * 20) -> None:
        self.data = data
        self.calls: list[tuple[int, int]] = []

    def fetch(self, offset: int, length: int) -> bytes:
        self.calls.append((offset, length))
        return self.data[offset : offset + length]


def test_ranged_io_seek_read_ahead_eof_and_buffered_wrapper() -> None:
    obj = _Object()
    raw = RangedRawIO(obj.fetch, size_fn=lambda: len(obj.data), block_size=64)
    assert raw.readable() and raw.seekable() and not raw.writable()
    assert raw.read(4) == obj.data[:4]
    assert raw.read(4) == obj.data[4:8] and len(obj.calls) == 1
    assert raw.seek(-10, io.SEEK_END) == len(obj.data) - 10
    assert raw.read(20) == obj.data[-10:]
    assert raw.read(1) == b""
    with pytest.raises(OSError):
        raw.seek(-1)
    with pytest.raises(ValueError):
        raw.seek(0, 99)
    raw.close()
    assert raw.closed

    handle = open_buffered(RangedRawIO(obj.fetch, size=len(obj.data), block_size=32))
    assert handle.read(20) == obj.data[:20]
    with pytest.raises(ValueError):
        _ = RangedRawIO(obj.fetch).size


class _Writer:
    def __init__(self, calls: list[tuple[Any, ...]]) -> None:
        self.calls = calls

    def format(self, value: str):
        self.calls.append(("format", value)); return self

    def mode(self, value: str):
        self.calls.append(("mode", value)); return self

    def option(self, key: str, value: str):
        self.calls.append(("option", key, value)); return self

    def saveAsTable(self, table: str) -> None:
        self.calls.append(("save", table))


class _Dataset:
    def __init__(self, calls: list[tuple[Any, ...]], columns=("id", "region")) -> None:
        self.calls = calls
        self.columns = columns
        self.write = _Writer(calls)

    def unionByName(self, other: _Dataset) -> _Dataset:
        self.calls.append(("union", other)); return self


class _Spark:
    def __init__(self, *, cluster_fails=False) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.cluster_fails = cluster_fails

    def createDataFrame(self, frame: Any) -> _Dataset:
        self.calls.append(("frame", frame)); return _Dataset(self.calls)

    def sql(self, statement: str) -> None:
        self.calls.append(("sql", statement))
        if self.cluster_fails and statement.startswith("ALTER"):
            raise RuntimeError("unsupported")


def test_delta_sink_writes_once_unions_batches_and_filters_clustering() -> None:
    spark = _Spark()
    item = _item().model_copy(update={"cluster_by": ("region", "missing")})
    rows = DeltaHydrationSink(
        session_factory=lambda: spark, apply_index_clustering=True
    ).write(item, ([{"id": 1}], [{"id": 2}, {"id": 3}]))
    assert rows == 3
    assert ("mode", "overwrite") in spark.calls
    assert ("option", "overwriteSchema", "true") in spark.calls
    assert any(call[0] == "union" for call in spark.calls)
    assert any("CLUSTER BY (`region`)" in call[1] for call in spark.calls if call[0] == "sql")


def test_delta_sink_append_empty_and_clustering_failure_are_safe() -> None:
    spark = _Spark(cluster_fails=True)
    sink = DeltaHydrationSink(
        session_factory=lambda: spark,
        apply_index_clustering=True,
        index_columns=lambda item: ("region",),
    )
    assert sink.write(_item(), ()) == 0
    assert sink.write(_item(write_mode=WriteMode.APPEND), ([1],)) == 1
    assert ("mode", "append") in spark.calls
