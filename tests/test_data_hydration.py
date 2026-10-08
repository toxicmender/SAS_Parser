"""
Tests for data_hydration — planning the data loads a SAS corpus implies.

Three things are being pinned.

First, the **decoupling contract**: importing this package must not import
``chunker`` or any database driver. It is the invariant most easily broken by a
convenience import, and it fails silently — everything still works, the package
is just no longer independent.

Second, that **planning is inert**. ``build_corpus_plan(probe=None)`` opens no
connection, which is what lets ``complexity`` build a plan purely to print one.
A test that needed a database would be evidence the property had been lost.

Third, that **what cannot be decided is recorded rather than guessed**: an
unresolved macro password, an SPD Engine library with no SAS session, a path
with no libref to name a schema. Each becomes a blocker on its own item and
leaves the rest of the plan intact.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from chunker.paths import extract_engine_refs, extract_paths
from data_hydration.config import HydrationConfig
from data_hydration.models import (
    HydrationPlan,
    ItemStatus,
    PartitionStrategy,
    SourceKind,
    WriteMode,
)
from data_hydration.planner import UNRESOLVED_TARGET, build_corpus_plan, build_plan
from data_hydration.runner import execute


def _config(**overrides) -> HydrationConfig:
    """A config with nothing read from the environment or config.json.

    Constructed directly rather than through ``from_env`` so a test never
    depends on the developer's own settings.
    """
    base = HydrationConfig(catalog="main", schema="staging")
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _plan(source: str, **overrides) -> HydrationPlan:
    """The plan for one snippet of SAS, via the real chunker grammar."""
    return build_plan(
        extract_engine_refs(source),
        extract_paths(source),
        config=_config(**overrides),
    )


class TestDecoupling:
    def test_importing_data_hydration_imports_neither_chunker_nor_a_driver(self):
        """The contract in the package README, asserted the only way that works.

        A subprocess, because by the time this test file runs the suite has
        already imported ``chunker`` — checking ``sys.modules`` in-process would
        pass no matter what this package does.
        """
        code = (
            "import sys; import data_hydration; "
            "from data_hydration import build_corpus_plan; "
            "print(','.join(m for m in "
            "('chunker','pipeline','complexity','pyspark','oracledb',"
            "'paramiko','pyreadstat','saspy','azure') if m in sys.modules))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(pathlib.Path(__file__).resolve().parents[1]),
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "", (
            f"data_hydration leaked imports: {result.stdout.strip()}"
        )


class TestEngineSources:
    def test_an_oracle_libname_becomes_an_item(self):
        plan = _plan("libname edwprod oracle path=EDWPRO schema=fr_dm;")
        assert len(plan.items) == 1
        item = plan.items[0]
        assert item.source.kind is SourceKind.ORACLE
        assert item.source.libref == "edwprod"
        assert item.source.locator == "EDWPRO"
        assert item.target_table == "main.staging.fr_dm"

    def test_an_unresolved_password_blocks_the_item_and_names_the_option(self):
        plan = _plan(
            'libname edwprod oracle path=EDWPRO schema=s user="&u." pass="&p.";'
        )
        item = plan.items[0]
        assert item.needs_operator_input
        assert "pass" in item.blockers[0] and "user" in item.blockers[0]

    def test_a_clean_connection_has_no_blockers(self):
        plan = _plan("libname edwprod oracle path=EDWPRO schema=s user=svc;")
        assert plan.items[0].blockers == ()
        assert plan.blocked_count == 0


class TestPathSources:
    def test_a_sas_dataset_is_recognised_by_its_suffix(self):
        plan = _plan("infile '/data/marts/sales.sas7bdat';")
        item = plan.items[0]
        assert item.source.kind is SourceKind.SAS_DATASET
        assert item.source.object_name == "sales"

    def test_an_spde_library_is_told_apart_from_a_directory(self):
        # The reason SasPathRef.engine had to be captured: these two statements
        # differ only by the engine keyword and mean entirely different things.
        spde = _plan("libname raw spde '/data/spde';").items[0]
        plain = _plan("libname raw '/data/plain';").items[0]
        assert spde.source.kind is SourceKind.SPDE
        assert plain.source.kind is SourceKind.FILE

    def test_an_spde_library_without_a_sas_session_is_blocked(self):
        item = _plan("libname raw spde '/data/spde';").items[0]
        assert item.needs_operator_input
        assert "saspy" in item.blockers[0] or "sas_host" in item.blockers[0]

    def test_an_spde_library_with_a_session_configured_is_not_blocked(self):
        item = _plan(
            "libname raw spde '/data/spde';", sas_host="sas.example.com"
        ).items[0]
        assert item.blockers == ()

    def test_a_directory_libname_is_reported_as_a_library_not_a_file(self):
        # `libname flat '/sasdata3/dataetl';` binds a whole library. Modelling it
        # as one file would invent a dataset that does not exist.
        item = _plan("libname flat '/sasdata3/dataetl';").items[0]
        assert item.target_table == UNRESOLVED_TARGET
        assert "library directory" in item.blockers[0]

    def test_a_file_read_through_a_fileref_is_one_item(self):
        # The chunker gives `infile in;` its FILENAME's path; the FILENAME is
        # still the one source of that file, not one of two.
        from chunker import SasSemanticChunker

        src = "filename in '/data/cust.csv';\ndata a; infile in dlm=','; input x; run;\n"
        chunks = SasSemanticChunker(min_words=1, max_words=9_999).chunk_text(src).chunks
        refs = [r for c in chunks for r in c.metadata.external_refs]
        assert [(r.statement, r.path) for r in refs] == [
            ("filename", "/data/cust.csv"),
            ("infile", "/data/cust.csv"),
        ]
        assert len(build_plan((), refs, config=_config()).items) == 1

    def test_an_ftp_filename_becomes_an_sftp_source(self):
        plan = _plan("filename raw ftp '/incoming/cust.csv' host='h';")
        assert plan.items[0].source.kind is SourceKind.SFTP

    @pytest.mark.parametrize(
        "source",
        [
            "%include '/code/macros/common.sas';",  # more SAS, not data
            "filename mail email 'ops@example.com';",  # a mailbox
            "filename cmd pipe 'ls -l';",  # a command line
            "ods html file='/reports/out.html';",  # a report destination
        ],
    )
    def test_references_that_move_no_data_are_not_items(self, source):
        assert _plan(source).items == []


class TestSasIndexHint:
    """A ``.sas7bndx`` is an index, and the plan must say so without inventing.

    The trap: the indexed column names live in an undocumented binary, so the
    planner cannot know them. Recording a placeholder in ``cluster_by`` would
    reach the sink and be emitted as ``CLUSTER BY (`<indexed>`)`` — not valid
    SQL and not a column. The presence is a note; the names come from the
    reader at load time, if at all.
    """

    _SOURCE = (
        "infile '/data/marts/sales.sas7bdat';\n"
        "filename idx '/data/marts/sales.sas7bndx';\n"
    )

    def test_the_index_file_is_never_itself_a_source(self):
        """The bug this guards: an index appended into the table it indexes.

        A ``.sas7bndx`` shares its stem with its dataset, so planned as an
        ordinary file it renders the *same* target table — and, not being first,
        appends. The result is index pages written as rows into real data.
        """
        items = _plan(self._SOURCE).items
        assert len(items) == 1
        assert items[0].source.kind is SourceKind.SAS_DATASET
        assert not any(".sas7bndx" in str(i.source) for i in items)

    def test_an_index_beside_a_dataset_is_noted(self):
        items = _plan(self._SOURCE).items
        dataset = [i for i in items if i.source.kind is SourceKind.SAS_DATASET]
        assert len(dataset) == 1
        assert any("SAS index" in note for note in dataset[0].notes)

    def test_the_note_is_not_a_blocker(self):
        # An index changes nothing about whether the load can run.
        dataset = [
            i for i in _plan(self._SOURCE).items
            if i.source.kind is SourceKind.SAS_DATASET
        ][0]
        assert dataset.blockers == ()

    def test_no_placeholder_reaches_cluster_by(self):
        for item in _plan(self._SOURCE).items:
            assert item.cluster_by == ()

    def test_a_dataset_without_an_index_has_no_note(self):
        items = _plan("infile '/data/marts/sales.sas7bdat';").items
        assert items[0].notes == ()

    def test_recovered_columns_are_intersected_with_the_real_ones(self):
        # Index parsing is best-effort, so a name it invents must never reach
        # ALTER TABLE. Only columns the table actually has survive.
        from data_hydration.models import HydrationItem, HydrationSource
        from data_hydration.sinks.delta import _clustering_columns

        item = HydrationItem(
            source=HydrationSource(kind=SourceKind.SAS_DATASET),
            target_table="main.s.t",
            cluster_by=("region", "not_a_column"),
        )
        assert _clustering_columns(item, {"region", "amount"}) == ("region",)


class TestSasDatasetReader:
    """The .sas7bdat reader, against a fake pyreadstat.

    A fake and not a real file, because **no Python library can write a
    .sas7bdat** — pyreadstat writes dta/por/sav/xport, pandas writes none, and
    the format is proprietary so ReadStat only reads it. A real fixture would
    have to be produced by SAS itself.

    What matters is testable without one: that the planner's row range reaches
    the reader as row_offset/row_limit. If it did not, every partition of a
    partitioned load would read the whole dataset — the load would "succeed"
    with N times the rows.
    """

    def _fake_pyreadstat(self, monkeypatch):
        """Install a stub pyreadstat and return the kwargs recorder."""
        import sys
        import types

        calls: list[dict] = []
        module = types.ModuleType("pyreadstat")

        class _Meta:
            number_rows = 10
            column_names = ["id", "region"]

        def read_sas7bdat(path, **kwargs):
            calls.append({"path": path, **kwargs})
            if kwargs.get("metadataonly"):
                return None, _Meta()
            return [{"id": 1}], _Meta()

        module.read_sas7bdat = read_sas7bdat  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pyreadstat", module)
        return calls

    def _reader(self, partition=None):
        from data_hydration.models import HydrationItem, HydrationSource
        from data_hydration.sources.sas_files import SasDatasetReader

        item = HydrationItem(
            source=HydrationSource(
                kind=SourceKind.SAS_DATASET, locator="/data", object_name="sales"
            ),
            target_table="main.s.sales",
            partition=partition,
        )
        return SasDatasetReader(item, _config())

    def test_an_unpartitioned_read_passes_no_row_bounds(self, monkeypatch):
        calls = self._fake_pyreadstat(monkeypatch)
        list(self._reader().batches())
        assert "row_offset" not in calls[0]
        assert "row_limit" not in calls[0]

    def test_a_row_range_partition_becomes_offset_and_limit(self, monkeypatch):
        from data_hydration.models import Partition

        calls = self._fake_pyreadstat(monkeypatch)
        list(self._reader(Partition(name="r", row_offset=4, row_limit=6)).batches())
        assert calls[0]["row_offset"] == 4
        assert calls[0]["row_limit"] == 6

    def test_info_reads_the_header_only(self, monkeypatch):
        # metadataonly is what makes probing a 40 GB dataset free; without it
        # planning would read every row of every source.
        calls = self._fake_pyreadstat(monkeypatch)
        info = self._reader().info()
        assert calls[0]["metadataonly"] is True
        assert info.rows == 10
        assert info.columns == ("id", "region")

    def test_the_path_is_the_dataset_not_the_directory(self, monkeypatch):
        calls = self._fake_pyreadstat(monkeypatch)
        self._reader().info()
        assert calls[0]["path"].replace("\\", "/").endswith("/data/sales.sas7bdat")

    def test_a_missing_driver_names_the_extra_to_install(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "pyreadstat", None)
        with pytest.raises(ImportError, match="sasdata"):
            self._reader().info()


class TestTargetNaming:
    def test_the_schema_defaults_to_the_libref(self):
        plan = build_plan(
            extract_engine_refs("libname edwprod oracle path=P schema=accounts;"),
            (),
            config=HydrationConfig(catalog="main"),
        )
        assert plan.items[0].target_table == "main.edwprod.accounts"

    def test_a_missing_value_blocks_one_item_without_losing_the_others(self):
        # The whole point of blocking rather than raising: an INFILE with no
        # libref has no schema, and must not cost the operator the other tables.
        source = (
            "libname edwprod oracle path=P schema=accounts;\n"
            "infile '/data/loose.sas7bdat';\n"
        )
        plan = build_plan(
            extract_engine_refs(source),
            extract_paths(source),
            config=HydrationConfig(catalog="main"),
        )
        assert len(plan.items) == 2
        good = [i for i in plan.items if i.target_table != UNRESOLVED_TARGET]
        assert len(good) == 1 and good[0].target_table == "main.edwprod.accounts"

    def test_a_broken_template_raises_instead_of_blocking(self):
        # A bad *template* is a broken configuration, not a per-source problem,
        # and every item would carry the identical blocker.
        from data_hydration.naming import TableNameError

        with pytest.raises(TableNameError):
            _plan(
                "libname a oracle path=p schema=s;",
                table_template="<catalog_name>.<nonsense>",
            )


class TestWriteModes:
    def test_the_first_item_for_a_table_overwrites_and_the_rest_append(self):
        # Two files declaring the same LIBNAME produce items for one table, and
        # exactly one of them may overwrite it.
        source = "libname edwprod oracle path=P schema=accounts;"
        plan = build_corpus_plan(
            {
                "a.sas": (extract_engine_refs(source), ()),
                "b.sas": (extract_engine_refs(source), ()),
            },
            config=_config(),
        )
        assert len(plan.items) == 2
        assert plan.items[0].write_mode is WriteMode.OVERWRITE
        assert plan.items[1].write_mode is WriteMode.APPEND
        assert len(plan.target_tables) == 1


class TestPlanShape:
    def test_by_source_id_groups_items_per_file(self):
        plan = build_corpus_plan(
            {
                "etl/a.sas": (
                    extract_engine_refs("libname x oracle path=P schema=one;"),
                    (),
                ),
                "etl/b.sas": (
                    extract_engine_refs("libname y oracle path=P schema=two;"),
                    (),
                ),
            },
            config=_config(),
        )
        assert len(plan.by_source_id("etl/a.sas")) == 1
        assert plan.by_source_id("etl/a.sas")[0].source.object_name == "one"
        assert plan.by_source_id("nope.sas") == []

    def test_one_run_date_is_used_for_every_item(self):
        # A run crossing midnight must not split one table in two.
        source = (
            "libname a oracle path=P schema=one;\n"
            "libname b oracle path=P schema=two;\n"
        )
        config = _config(table_template="<catalog_name>.<schema_name>.<table_name>_<date>")
        plan = build_plan(extract_engine_refs(source), (), config=config)
        stamps = {i.target_table.rsplit("_", 1)[-1] for i in plan.items}
        assert len(stamps) == 1
        assert stamps == {plan.run_date}

    def test_counts_by_kind(self):
        source = (
            "libname a oracle path=P schema=one;\n"
            "infile '/d/x.sas7bdat';\n"
        )
        plan = _plan(source)
        assert plan.counts_by_kind() == {"oracle": 1, "sas7bdat": 1}


class TestPartitioning:
    def test_without_a_probe_nothing_is_partitioned(self):
        item = _plan("libname a oracle path=P schema=t;").items[0]
        assert item.strategy is PartitionStrategy.NONE
        assert item.partition is None
        assert "without a live connection" in item.strategy_reason

    def test_spde_components_are_counted_but_not_fanned_out(self, tmp_path):
        # Counting them is useful; splitting on them is wrong — a .dpf cannot be
        # read alone, so an item each would read the whole dataset N times.
        library = tmp_path / "spde"
        library.mkdir()
        for index in range(4):
            (library / f"sales.dpf.{index}.dpf").write_bytes(b"")
        (library / "sales.mdf").write_bytes(b"")

        from data_hydration.partition import plan_partitions, spde_partitions

        assert len(spde_partitions(str(library), "sales")) == 4
        chosen = plan_partitions(
            SourceKind.SPDE, locator=str(library), object_name="sales"
        )
        assert chosen.partitions == []
        assert "4 .dpf component(s)" in chosen.reason

    def test_a_probe_that_raises_downgrades_rather_than_failing(self):
        class _ExplodingProbe:
            def native_partitions(self, owner, table):
                raise RuntimeError("ORA-00942: table or view does not exist")

            def row_count(self, owner, table):
                raise RuntimeError("no")

            def range_column(self, owner, table):
                raise RuntimeError("no")

        from data_hydration.partition import plan_partitions

        chosen = plan_partitions(
            SourceKind.ORACLE, object_name="s.t", probe=_ExplodingProbe()
        )
        assert chosen.strategy is PartitionStrategy.NONE

    def test_native_partitions_become_one_item_each(self):
        class _Probe:
            def native_partitions(self, owner, table):
                return ["P2024_01", "P2024_02"]

            def row_count(self, owner, table):
                return None

            def range_column(self, owner, table):
                return None

        plan = build_plan(
            extract_engine_refs("libname edw oracle path=P schema=sales;"),
            (),
            config=_config(),
            probe=_Probe(),
        )
        assert len(plan.items) == 2
        partitions = [i.partition for i in plan.items]
        assert all(p is not None for p in partitions)
        assert [p.name for p in partitions if p] == ["P2024_01", "P2024_02"]
        assert plan.items[0].write_mode is WriteMode.OVERWRITE
        assert plan.items[1].write_mode is WriteMode.APPEND

    def test_a_column_range_covers_the_maximum_value(self):
        # An off-by-one on the last bound silently drops rows, which is the
        # worst failure this module could have.
        from data_hydration.partition import _column_ranges

        parts = _column_ranges("id", 0.0, 100.0, 4)
        assert len(parts) == 4
        assert parts[-1].predicate is not None
        assert "<= 100.0" in parts[-1].predicate


class TestRunner:
    def test_a_dry_run_touches_nothing_and_skips_every_item(self):
        plan = _plan("libname a oracle path=P schema=t;")
        report = execute(plan, config=_config(), dry_run=True)
        assert report.dry_run is True
        assert report.skipped == len(plan.items)
        assert report.written == 0
        assert report.ok is True

    def test_a_blocked_item_is_skipped_without_being_attempted(self):
        # The coordinates are known to be wrong, so trying them is not a
        # fallback — it is a connection attempt with a literal "&p." password.
        plan = _plan('libname a oracle path=P schema=t user="&u." pass="&p.";')
        report = execute(plan, config=_config())
        assert report.skipped == 1
        assert report.failed == 0

    def test_a_failing_item_is_recorded_and_the_run_continues(self, monkeypatch):
        source = (
            "libname a oracle path=P schema=one;\n"
            "libname b oracle path=P schema=two;\n"
        )
        plan = _plan(source)
        assert len(plan.items) == 2

        calls: list[str] = []

        def _boom(item, config):
            calls.append(item.target_table)
            raise RuntimeError("host unreachable")

        monkeypatch.setattr("data_hydration.sinks.delta.write_item", _boom)
        report = execute(plan, config=_config())
        assert len(calls) == 2, "the run stopped at the first failure"
        assert report.failed == 2
        assert report.ok is False
        assert "host unreachable" in (report.outcomes[0].error or "")

    def test_on_error_stop_ends_the_run_at_the_first_failure(self, monkeypatch):
        source = (
            "libname a oracle path=P schema=one;\n"
            "libname b oracle path=P schema=two;\n"
        )
        plan = _plan(source)
        calls: list[str] = []

        def _boom(item, config):
            calls.append(item.target_table)
            raise RuntimeError("nope")

        monkeypatch.setattr("data_hydration.sinks.delta.write_item", _boom)
        report = execute(plan, config=_config(on_error="stop"))
        assert len(calls) == 1
        assert report.failed == 1

    def test_a_written_item_reports_its_row_count(self, monkeypatch):
        plan = _plan("libname a oracle path=P schema=t;")
        monkeypatch.setattr(
            "data_hydration.sinks.delta.write_item", lambda item, config: 42
        )
        report = execute(plan, config=_config())
        assert report.written == 1
        assert report.outcomes[0].status is ItemStatus.WRITTEN
        assert report.outcomes[0].rows == 42


# ---------------------------------------------------------------------------
# Database tables: SQL pass-through and named members of a database LIBNAME
# ---------------------------------------------------------------------------

PASS_THROUGH = """proc sql
connect to oracle (user=&ora_user password=&ora_pass path=&ora_path);
create table nonip as select * from connection to oracle
(select cov_month, count as count from edw_export.current_nonip where table_cd='MED');
disconnect from oracle;
quit;
"""


def _db(source: str, source_id: str = "t.sas"):
    """``(engine_refs, path_refs, db_tables)`` for *source*, via the real chunker."""
    from chunker import SasSemanticChunker

    result = SasSemanticChunker().chunk_text(source, source_id=source_id)
    meta = [c.metadata for c in result.chunks]
    return (
        [r for m in meta for r in m.engine_refs],
        [r for m in meta for r in m.external_refs],
        [t for m in meta for t in m.db_tables],
    )


class TestDatabaseTables:
    def test_a_pass_through_read_is_its_own_item(self):
        engine_refs, path_refs, tables = _db(PASS_THROUGH)
        plan = build_plan(
            engine_refs, path_refs, db_tables=tables, config=_config(schema=None)
        )
        (item,) = plan.items
        assert item.source.kind is SourceKind.ORACLE
        assert item.source.object_name == "edw_export.current_nonip"
        assert (item.source.connection, item.source.libref) == ("oracle", None)
        assert item.source.locator == "&ora_path"
        # No libref to name the schema after, so the Oracle owner does.
        assert item.target_table == "main.edw_export.current_nonip"
        assert "password, path, user" in item.blockers[0]

    def test_writes_are_never_sources(self):
        _, _, tables = _db(
            "proc sql;\nconnect to oracle (path=P);\n"
            "execute (truncate table stage.tmp) by oracle;\nquit;\n"
        )
        assert tables and build_plan(db_tables=tables, config=_config()).items == []

    def test_a_libname_whose_tables_are_named_is_planned_per_table(self):
        source = (
            "libname edw oracle path=EDWPRO schema=fr_dm;\n"
            "libname quiet oracle path=EDWPRO schema=other;\n"
            "data work.a; set edw.accounts; run;\n"
            "data work.b; set edw.orders; run;\n"
        )
        engine_refs, path_refs, tables = _db(source)
        plan = build_plan(
            engine_refs, path_refs, db_tables=tables, config=_config(schema=None)
        )
        objects = sorted(i.source.object_name for i in plan.items)
        # edw's schema-level stand-in gives way to its two named tables; the
        # LIBNAME nothing reads from keeps the stand-in it had.
        assert objects == ["fr_dm.accounts", "fr_dm.orders", "other"]
        targets = sorted(i.target_table for i in plan.items)
        assert targets == ["main.edw.accounts", "main.edw.orders", "main.quiet.other"]

    def test_one_table_read_by_two_files_is_one_item(self):
        # Appending one copy per reader would load the rows twice.
        _, _, first = _db(PASS_THROUGH, "a.sas")
        _, _, second = _db(PASS_THROUGH, "b.sas")
        plan = build_corpus_plan(
            {}, db_tables={"a.sas": first, "b.sas": second}, config=_config()
        )
        (item,) = plan.items
        assert item.write_mode is WriteMode.OVERWRITE
        assert item.source.source_id == "a.sas"

    def test_an_unknown_engine_and_a_dblink_block_the_item(self):
        _, _, tables = _db(
            "proc sql;\ncreate table a as select * from connection to mydb\n"
            "(select * from s.t@prodlink);\nquit;\n"
        )
        (item,) = build_plan(db_tables=tables, config=_config()).items
        joined = " ".join(item.blockers)
        assert "connection 'mydb' is unknown" in joined
        assert "database link 'prodlink'" in joined

    def test_the_reader_selects_owner_and_table(self):
        from data_hydration.sources.oracle import OracleReader

        _, _, qualified = _db(PASS_THROUGH)
        _, _, bare = _db(
            "proc sql;\ncreate table a as select * from connection to oracle\n"
            "(select * from current_nonip);\nquit;\n"
        )
        config = _config()
        reads = {
            OracleReader(item, config)._sql()
            for tables in (qualified, bare)
            for item in build_plan(db_tables=tables, config=config).items
        }
        assert reads == {
            'SELECT * FROM "EDW_EXPORT"."CURRENT_NONIP"',
            'SELECT * FROM "CURRENT_NONIP"',
        }

    def test_the_credential_is_keyed_on_the_connection(self, monkeypatch):
        import types

        import data_hydration.sources.oracle as oracle

        seen: list[str] = []
        monkeypatch.setattr(
            oracle, "resolve_secret", lambda name, **_: seen.append(name) or "pw"
        )
        monkeypatch.setitem(
            sys.modules, "oracledb", types.SimpleNamespace(connect=lambda **_: object())
        )
        _, _, tables = _db(
            "proc sql;\nconnect to oracle as edw (path=EDWPRO user=svc);\n"
            "create table a as select * from connection to edw (select * from s.t);\n"
            "quit;\n"
        )
        (item,) = build_plan(db_tables=tables, config=_config()).items
        oracle.connect(item, _config())
        assert seen == ["oracle_password_edw"]

    def test_the_cli_resolves_a_libname_in_another_file(self, tmp_path):
        import argparse

        from data_hydration.__main__ import _build_plan

        (tmp_path / "a_setup.sas").write_text(
            "libname edw oracle path=EDWPRO schema=fr_dm;\n"
        )
        (tmp_path / "b_job.sas").write_text(
            "data work.a; set edw.accounts; run;\n" + PASS_THROUGH
        )
        args = argparse.Namespace(source_dir=tmp_path, pattern="*.sas", only=None)
        plan = _build_plan(args, _config(schema=None))
        assert sorted(i.source.object_name for i in plan.items) == [
            "edw_export.current_nonip",
            "fr_dm.accounts",
        ]


class TestMacroResolvedTables:
    """Oracle names spelled through macro variables, as the plan sees them."""

    PULL = (
        "%macro pull(schema=edw_export, tbl=, out=);\n"
        "proc sql;\nconnect to oracle (path=EDWPRO);\n"
        "create table &out as select * from connection to oracle\n"
        "  (select * from &schema..&tbl);\n"
        "quit;\n%mend;\n"
    )

    def test_a_calls_table_is_planned_and_the_template_is_not(self):
        _, _, tables = _db(self.PULL + "%pull(tbl=current_nonip, out=nonip);\n")
        plan = build_plan(db_tables=tables, config=_config(schema=None))
        (item,) = plan.items
        assert item.source.object_name == "edw_export.current_nonip"
        assert item.target_table == "main.edw_export.current_nonip"
        assert item.blockers == ()

    def test_a_macro_nobody_calls_hydrates_nothing(self):
        _, _, tables = _db(self.PULL)
        assert tables and build_plan(db_tables=tables, config=_config()).items == []

    def test_a_let_resolved_schema_is_planned_resolved(self):
        _, _, tables = _db(
            "%let sch = EDW_EXPORT;\nproc sql;\nconnect to oracle (path=EDWPRO);\n"
            "create table a as select * from connection to oracle "
            "(select * from &sch..current_nonip);\nquit;\n"
        )
        (item,) = build_plan(db_tables=tables, config=_config(schema=None)).items
        assert item.source.object_name == "edw_export.current_nonip"
        assert item.blockers == ()
