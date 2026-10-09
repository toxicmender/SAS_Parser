"""
test_data_hydration_inventory.py — the reference inventory, and planning from it.

The inventory is every path and dataset a corpus names, resolved or not, as rows
of a Delta table. What is pinned here, with no Spark:

* **Every reference is a row.** Paths, datasets, database tables and database
  LIBNAMEs, with ``resolved`` saying which still hold an unresolved ``&``.
* **The rows are enough to plan from.** ``plan_from_inventory`` gives the plan
  the chunks give, so a job holding the table needs neither the SAS nor the
  chunker.
* **Datasets reach the plan.** A directory LIBNAME whose members are read is
  planned per member, as a database LIBNAME is planned per table.
* **Secrets stay out.** A literal password never reaches a row.

The Delta round trip itself runs against a real session in
``test_data_hydration_delta.py``; here a fake records what is asked of Spark.
"""

from __future__ import annotations

import pathlib
import sys
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from chunker import SasCorpus, SasSemanticChunker, resolve_corpus_references
from data_hydration.config import HydrationConfig
from data_hydration.inventory import (
    _COLUMNS,
    InventoryRow,
    RefKind,
    _quoted_table,
    inventory_rows,
    plan_from_inventory,
    write_inventory,
)
from data_hydration.models import SourceKind
from data_hydration.planner import UNRESOLVED_TARGET, build_corpus_plan

PASS_THROUGH = (
    "proc sql;\nconnect to oracle (user=&ora_user password=&ora_pass path=&ora_path);\n"
    "create table nonip as select * from connection to oracle\n"
    "(select * from edw_export.current_nonip);\ndisconnect from oracle;\nquit;\n"
)


def _results(chunker: SasSemanticChunker | None = None, /, **files: str):
    """The resolved chunk results of ``name=source`` files, in order."""
    chunker = chunker or SasSemanticChunker()
    corpus = SasCorpus(
        file_results=[
            chunker.chunk_text(src, source_id=f"{name}.sas") for name, src in files.items()
        ]
    )
    return resolve_corpus_references(corpus).file_results


def _rows(**files: str) -> list[InventoryRow]:
    return inventory_rows(_results(**files))


def _config() -> HydrationConfig:
    return HydrationConfig(catalog="main")


def _plan(**files: str):
    return plan_from_inventory(_rows(**files), config=_config())


# ── the rows ─────────────────────────────────────────────────────────────────


def test_every_reference_is_a_row_resolved_or_not():
    rows = _rows(
        job=(
            "%let root = /SASData;\nlibname raw \"&root/raw\";\nlibname q \"&nowhere/q\";\n"
            "libname edw oracle path=EDWPRO schema=fr_dm;\n"
            "data out; set raw.customers edw.accounts &lib..x; run;\n" + PASS_THROUGH
        )
    )
    seen = {(r.kind, r.name, r.resolved) for r in rows}
    assert {
        (RefKind.PATH, "/sasdata/raw", True),
        (RefKind.PATH, "&nowhere/q", False),
        (RefKind.ENGINE, "edw", True),
        (RefKind.DB_TABLE, "fr_dm.accounts", True),
        (RefKind.DB_TABLE, "edw_export.current_nonip", True),
        (RefKind.DATASET, "raw.customers", True),
        (RefKind.DATASET, "&lib..x", False),
        (RefKind.DATASET, "work.out", True),
    } <= seen
    # A path keeps both spellings: as written, and the place SAS reads.
    [raw] = [r for r in rows if r.kind is RefKind.PATH and r.libref == "raw"]
    assert (raw.raw, raw.value, raw.role, raw.location) == (
        "&root/raw",
        "/SASData/raw",
        "bind",
        "filesystem",
    )
    [read] = [r for r in rows if r.name == "raw.customers"]
    assert (read.role, read.statement, read.libref, read.member) == ("read", "set", "raw", "customers")


def test_rows_keep_the_corpus_order():
    rows = _rows(a="libname a '/a';\ndata x; set a.t; run;\n", b="data y; set x; run;\n")
    assert [(r.source_id, r.file_order, r.ref_order) for r in rows] == [
        ("a.sas", 0, 0),
        ("a.sas", 0, 1),
        ("a.sas", 0, 2),
        ("b.sas", 1, 0),
        ("b.sas", 1, 1),
    ]
    assert len({r.run_id for r in rows}) == 1


def test_a_split_chunk_is_read_once():
    big = "data out;\n" + "".join(f"  set lib.t{i % 3};\n  x{i} = {i};\n" for i in range(60)) + "run;\n"
    chunker = SasSemanticChunker(min_words=1, max_words=40)
    [result] = _results(chunker, big=big)
    assert any(c.parent_id for c in result.chunks), "the step should have been split"
    rows = inventory_rows([result])
    names = [r.name for r in rows if r.kind is RefKind.DATASET]
    assert sorted(names) == ["lib.t0", "lib.t1", "lib.t2", "work.out"]


def test_passwords_are_redacted_and_macro_references_kept():
    rows = _rows(
        job=(
            'libname edw oracle path=EDWPRO user=svc pass="S3cret!";\n'
            'libname odb odbc noprompt="DSN=dw;UID=u;PWD=hunter2;" schema=s;\n'
            "libname mac oracle path=P user=&u. password=&p.;\n"
            "data a; set edw.t odb.t; run;\n"
        )
    )
    text = " ".join(f"{r.raw} {r.options}" for r in rows)
    assert "S3cret" not in text and "hunter2" not in text
    engines = {r.name: r for r in rows if r.kind is RefKind.ENGINE}
    assert dict(engines["edw"].options)["pass"] == "<redacted>"
    assert 'pass=<redacted>' in engines["edw"].raw
    assert dict(engines["odb"].options)["noprompt"] == "DSN=dw;UID=u;PWD=<redacted>;"
    assert dict(engines["mac"].options)["password"] == "&p."
    # The database tables carry the same, redacted, connection.
    assert {dict(r.options).get("pass") for r in rows if r.kind is RefKind.DB_TABLE} == {
        "<redacted>",
        None,
    }


def test_rows_survive_the_table_round_trip():
    rows = _rows(job="libname edw oracle path=P schema=s;\ndata a; set edw.t; run;\n" + PASS_THROUGH)
    for row in rows:
        stored = row.model_dump()
        # As Spark hands them back: structs as dicts, a naive timestamp.
        stored["options"] = [{"key": k, "value": v} for k, v in row.options]
        stored["sas_targets"] = list(row.sas_targets)
        stored["kind"] = str(row.kind)
        stored["recorded_at"] = row.recorded_at.replace(tzinfo=None)
        back = InventoryRow.model_validate(stored)
        assert back.model_copy(update={"recorded_at": row.recorded_at}) == row


def test_the_columns_are_the_row_fields_in_order():
    assert [name for name, _, _ in _COLUMNS] == list(InventoryRow.model_fields)
    # A comment is a SQL string literal: no quote may end it early.
    assert not any("'" in comment for _, _, comment in _COLUMNS)


# ── planning from the rows ───────────────────────────────────────────────────


def _chunk_plan(results):
    """The plan built from the chunks themselves, as the CLIs once did."""

    def refs(result, field):
        return [r for c in result.chunks if c.parent_id is None for r in getattr(c.metadata, field)]

    return build_corpus_plan(
        {r.source_id: (refs(r, "engine_refs"), refs(r, "external_refs")) for r in results},
        db_tables={r.source_id: refs(r, "db_tables") for r in results},
        datasets={r.source_id: refs(r, "dataset_refs") for r in results},
        config=_config(),
    )


def test_the_plan_from_the_inventory_is_the_plan_from_the_chunks():
    results = _results(
        setup=(
            "%let root = /SASData;\nlibname edw oracle path=EDWPRO schema=fr_dm user=svc;\n"
            "libname raw \"&root/raw\";\nlibname spd spde '/data/spde';\n"
            "filename in '/data/in/cust.csv';\n"
        ),
        job=(
            "data a; set edw.accounts raw.customers; run;\n"
            "data b; infile in dlm=','; input x; run;\n"
            "data c; infile '/data/in/rates.sas7bdat'; run;\n" + PASS_THROUGH
        ),
    )
    from_chunks = _chunk_plan(results)
    from_rows = plan_from_inventory(inventory_rows(results), config=_config())
    assert from_rows.items == from_chunks.items
    assert len(from_rows.items) == 6


def test_only_keeps_one_librefs_references():
    rows = _rows(
        job=(
            "libname raw '/data/raw';\nlibname edw oracle path=P schema=s;\n"
            "data a; set raw.customers edw.accounts; run;\n"
        )
    )
    plan = plan_from_inventory(rows, config=_config(), only=["RAW"])
    assert [(i.source.kind, i.source.object_name) for i in plan.items] == [
        (SourceKind.SAS_DATASET, "customers")
    ]


# ── a directory LIBNAME's members ────────────────────────────────────────────


def test_a_directory_library_is_planned_per_dataset_read():
    plan = _plan(
        setup="libname raw '/data/raw';\n",
        job="data a; set raw.customers raw.orders; run;\nproc print data=raw.customers; run;\n",
    )
    assert [(i.source.kind, i.source.locator, i.source.object_name) for i in plan.items] == [
        (SourceKind.SAS_DATASET, "/data/raw", "customers"),
        (SourceKind.SAS_DATASET, "/data/raw", "orders"),
    ]
    assert [i.target_table for i in plan.items] == ["main.raw.customers", "main.raw.orders"]
    # Owned by the file that reads them, and nothing blocks them.
    assert {i.source.source_id for i in plan.items} == {"job.sas"}
    assert all(not i.blockers for i in plan.items)


def test_a_dataset_the_corpus_creates_is_not_loaded():
    plan = _plan(
        job=(
            "libname stage '/data/stage';\n"
            "data stage.tmp; set stage.ext; run;\ndata b; set stage.tmp; run;\n"
            "proc append base=stage.log data=b; run;\n"
        )
    )
    # stage.tmp is the job's own; stage.log is updated in place, so it must
    # already exist, and nothing in the corpus creates it.
    assert [i.source.object_name for i in plan.items] == ["ext", "log"]


def test_a_list_keeps_the_library_item_beside_its_named_members():
    plan = _plan(job="libname raw '/data/raw';\ndata a; set raw.customers raw.sales_:; run;\n")
    library, member = sorted(plan.items, key=lambda i: i.source.object_name)
    assert (library.source.kind, library.target_table) == (SourceKind.FILE, UNRESOLVED_TARGET)
    assert "library directory" in library.blockers[0]
    assert (member.source.object_name, member.blockers) == ("customers", ())


def test_an_unresolved_member_is_planned_blocked():
    [item] = _plan(job="libname raw '/data/raw';\ndata a; set raw.&tbl; run;\n").items
    assert item.source.object_name == "&tbl"
    assert "the object name '&tbl'" in item.blockers[0]


def test_an_unresolved_directory_names_its_location():
    [item] = _plan(job="libname q \"&nowhere/q\";\ndata a; set q.t; run;\n").items
    assert "the location '&nowhere/q'" in item.blockers[0]


def test_the_libname_in_force_is_the_latest_before_the_read():
    plan = _plan(
        a_first="data a; set raw.early; run;\n",
        b_setup="libname raw '/one';\n",
        c_job="data b; set raw.mid; run;\nlibname raw '/two';\ndata c; set raw.late; run;\n",
    )
    # raw.early is read before any LIBNAME binds raw; within a file the
    # LIBNAMEs are taken before the reads, so c_job reads from /two.
    located = [(i.source.locator, i.source.object_name) for i in plan.items]
    assert ("/one", "early") not in located
    assert ("/two", "mid") in located and ("/two", "late") in located


def test_an_spde_library_is_not_split_into_members():
    [item] = _plan(job="libname spd spde '/data/spde';\ndata a; set spd.big; run;\n").items
    assert item.source.kind is SourceKind.SPDE


# ── the table ────────────────────────────────────────────────────────────────


class _FakeWriter:
    def __init__(self, log: list):
        self.log = log

    def format(self, name):
        self.log.append(("format", name))
        return self

    def mode(self, name):
        self.log.append(("mode", name))
        return self

    def saveAsTable(self, table):
        self.log.append(("saveAsTable", table))


class _FakeFrame:
    def __init__(self, log: list):
        self.write = _FakeWriter(log)


class _FakeSpark:
    """Records the SQL and the frame a write asks of Spark."""

    def __init__(self):
        self.sql_log: list[str] = []
        self.write_log: list = []
        self.created: list = []

    def sql(self, statement: str):
        self.sql_log.append(statement)

    def createDataFrame(self, data, schema):
        self.created.append((data, schema))
        return _FakeFrame(self.write_log)


def test_a_write_appends_one_run_to_a_delta_table():
    rows = _rows(job="libname raw '/data/raw';\ndata a; set raw.customers; run;\n")
    spark = _FakeSpark()
    assert write_inventory(rows, "main.meta.sas_refs", spark=spark) == len(rows)
    create_schema, create_table = spark.sql_log
    assert create_schema == "CREATE SCHEMA IF NOT EXISTS `main`.`meta`"
    assert create_table.startswith("CREATE TABLE IF NOT EXISTS `main`.`meta`.`sas_refs`")
    assert "USING DELTA" in create_table
    assert all(f"`{name}` {sql_type}" in create_table for name, sql_type, _ in _COLUMNS)
    [(data, schema)] = spark.created
    assert schema.split(", ")[0] == "`run_id` STRING"
    # One tuple per row, in column order, in types Spark takes.
    assert data[0][: len(("run_id", "recorded_at"))] == (rows[0].run_id, rows[0].recorded_at)
    assert all(len(values) == len(_COLUMNS) for values in data)
    assert isinstance(data[0][[n for n, _, _ in _COLUMNS].index("kind")], str)
    assert spark.write_log == [("format", "delta"), ("mode", "append"), ("saveAsTable", "main.meta.sas_refs")]


def test_nothing_to_write_touches_no_table():
    spark = _FakeSpark()
    assert write_inventory([], "main.meta.sas_refs", spark=spark) == 0
    assert spark.sql_log == [] and spark.created == []


@pytest.mark.parametrize(
    "table",
    ["", "a.b.c.d", "main.meta.refs; DROP TABLE x", "main.`meta`.refs", "1abc.t", "a..b"],
)
def test_a_table_name_must_be_an_identifier(table):
    with pytest.raises(ValueError):
        _quoted_table(table)


# ── the CLI ──────────────────────────────────────────────────────────────────


@pytest.fixture
def corpus_dir(tmp_path):
    (tmp_path / "a_setup.sas").write_text("libname raw '/data/raw';\n")
    (tmp_path / "b_job.sas").write_text("data a; set raw.customers; run;\n")
    return tmp_path


def _main(*argv: str) -> int:
    from data_hydration.__main__ import main

    return main(list(argv))


def test_the_cli_keeps_the_inventory_and_plans_from_it(corpus_dir, monkeypatch, capsys):
    import data_hydration.inventory as inventory

    stored: dict[str, list[InventoryRow]] = {}
    monkeypatch.setattr(
        inventory, "write_inventory", lambda rows, table, **_: stored.setdefault(table, rows)
    )
    assert _main(str(corpus_dir), "--dry-run", "--inventory-table", "main.meta.refs") == 0
    written = capsys.readouterr().out
    assert {r.name for r in stored["main.meta.refs"]} >= {"/data/raw", "raw.customers"}
    assert "sas7bdat:/data/raw/customers (raw)" in written

    # A later job plans from the table: no source directory, no chunker.
    monkeypatch.setattr(
        inventory, "read_inventory", lambda table, run_id=None, **_: stored[table]
    )
    assert _main("--from-inventory", "--inventory-table", "main.meta.refs", "--dry-run") == 0
    assert capsys.readouterr().out.splitlines()[2:] == written.splitlines()[2:]


def test_a_table_that_cannot_be_written_fails_the_run_not_the_plan(corpus_dir, monkeypatch, capsys):
    import data_hydration.inventory as inventory

    def refuse(*_, **__):
        raise RuntimeError("no Spark here")

    monkeypatch.setattr(inventory, "write_inventory", refuse)
    assert _main(str(corpus_dir), "--dry-run", "--inventory-table", "main.meta.refs") == 1
    assert "sas7bdat:/data/raw/customers (raw)" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--from-inventory", "SRC"], "not a source directory"),
        ([], "a source directory is required"),
        (["SRC", "--run-id", "r1"], "needs --from-inventory"),
        (["SRC", "--inventory-table", "main.meta.refs; drop"], "identifier parts"),
    ],
)
def test_argument_errors(argv, message, corpus_dir, capsys):
    argv = [str(corpus_dir) if a == "SRC" else a for a in argv]
    assert _main(*argv) == 2
    assert message in capsys.readouterr().err


def test_from_inventory_needs_a_table(monkeypatch, capsys):
    monkeypatch.delenv("DATA_HYDRATION_INVENTORY_TABLE", raising=False)
    import data_hydration.__main__ as cli

    monkeypatch.setattr(
        cli, "_config_for", lambda args: HydrationConfig(catalog="main", inventory_table=None)
    )
    assert _main("--from-inventory", "--dry-run") == 2
    assert "needs an inventory table" in capsys.readouterr().err


def test_run_ids_sort_in_time_order():
    from data_hydration.inventory import new_run_id

    early = new_run_id(datetime(2026, 1, 2, 3, 4, 5))
    late = new_run_id(datetime(2026, 1, 2, 3, 4, 6))
    assert early.startswith("20260102T030405Z-") and early < late
