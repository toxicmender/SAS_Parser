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

from chunker import SasCorpus, SasSemanticChunker, resolve_corpus_datasets
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
    return resolve_corpus_datasets(corpus).file_results


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


@pytest.mark.parametrize(
    ("statement", "secret"),
    [
        # USING= is PASSWORD= on the ODBC, DB2, OLE DB and SQL Server engines.
        ("libname dw odbc datasrc=edw user=etl using=S3cr3t;", "S3cr3t"),
        ('libname d db2 database=edw user=etl using="S3cr3t";', "S3cr3t"),
        # A braced ODBC value holds ';' and spaces, all of it the password.
        ('libname o odbc noprompt="DSN=x;UID=u;PWD={s3c r3t};";', "s3c r3t"),
        ('libname o odbc noprompt="DSN=x;Password=s3cr3t";', "s3cr3t"),
        # Credentials written into a URL.
        ('libname h hadoop uri="jdbc:hive2://etl:s3cr3t@hive:10000/edw";', "s3cr3t"),
        # An option nobody listed as safe, whatever it holds.
        ("libname o oracle path=P user=u orapw=s3cr3t;", "s3cr3t"),
    ],
)
def test_no_spelling_of_a_password_reaches_a_row(statement, secret):
    rows = _rows(job=statement + "\n")
    [engine] = [r for r in rows if r.kind is RefKind.ENGINE]
    stored = f"{engine.raw} {engine.options}"
    # No piece of it either: a braced value is the password whole.
    assert not any(piece in stored for piece in secret.split())
    assert engine.raw.startswith("libname ") and engine.raw.endswith(";")
    assert engine.raw.count('"') % 2 == 0, engine.raw  # its quotes still pair


def test_connection_coordinates_and_macro_references_are_kept():
    [engine] = [
        r
        for r in _rows(
            job="libname o odbc noprompt=\"DSN=dw;UID=u;PWD=x\" schema=s readbuff=100 pass=&p.;\n"
        )
        if r.kind is RefKind.ENGINE
    ]
    assert dict(engine.options) == {
        "noprompt": "DSN=dw;UID=u;PWD=<redacted>",
        "schema": "s",
        "readbuff": "<redacted>",
        "pass": "&p.",
    }
    # The pass-through CONNECT options are redacted the same way.
    tables = _rows(
        job="proc sql;\nconnect to db2 (database=edw user=u using=S3cr3t);\n"
        "create table a as select * from connection to db2 (select * from s.t);\nquit;\n"
    )
    [table] = [r for r in tables if r.kind is RefKind.DB_TABLE]
    assert dict(table.options) == {"database": "edw", "user": "u", "using": "<redacted>"}


def test_references_that_differ_only_in_a_secret_are_one_row():
    # Redacted, the two LIBNAMEs of one macro say the same thing.
    rows = _rows(
        job=(
            "%macro conn;\n  libname edw oracle path=P user=etl password=one;\n"
            "  libname edw oracle path=P user=etl password=two;\n%mend;\n"
        )
    )
    [engine] = [r for r in rows if r.kind is RefKind.ENGINE]
    assert engine.raw == "libname edw oracle path=P user=etl password=<redacted>;"


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
    # A row with no options may come back with a null.
    stored = rows[0].model_dump() | {"options": None}
    assert InventoryRow.model_validate(stored).options == ()


def test_the_columns_are_the_row_fields_in_order():
    assert [name for name, _, _ in _COLUMNS] == list(InventoryRow.model_fields)
    # A comment is a SQL string literal: no quote may end it early.
    assert not any("'" in comment for _, _, comment in _COLUMNS)


# ── planning from the rows ───────────────────────────────────────────────────


def _chunk_plan(results):
    """The plan built from the chunks themselves, as the CLIs once did."""

    def refs(result, field):
        return [r for c in result.chunks if c.parent_id is None for r in getattr(c.metadata, field)]

    def per_chunk(result):
        return [list(c.metadata.dataset_refs) for c in result.chunks if c.parent_id is None]

    return build_corpus_plan(
        {r.source_id: (refs(r, "engine_refs"), refs(r, "external_refs")) for r in results},
        db_tables={r.source_id: refs(r, "db_tables") for r in results},
        datasets={r.source_id: per_chunk(r) for r in results},
        config=_config(),
    )


def test_the_plan_from_the_inventory_is_the_plan_from_the_chunks():
    results = _results(
        setup=(
            "%let root = /SASData;\nlibname edw oracle path=EDWPRO schema=fr_dm user=svc;\n"
            "libname raw \"&root/raw\";\nlibname spd spde '/data/spde';\n"
            "filename in '/data/in/cust.csv';\n"
            # A database library nothing reads from: the library is the item.
            "libname arch oracle path=ARCHPRO schema=hist;\n"
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
    assert len(from_rows.items) == 7
    assert "oracle:ARCHPRO/hist (arch)" in [str(i.source) for i in from_rows.items]


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
    # raw.early is read before any LIBNAME binds raw. Within c_job, raw.mid is
    # read while b_setup's LIBNAME is in force, and only raw.late after c_job
    # rebinds raw to /two.
    located = [(i.source.locator, i.source.object_name) for i in plan.items]
    assert ("/one", "early") not in located
    assert ("/one", "mid") in located and ("/two", "late") in located
    assert ("/two", "mid") not in located


def test_a_library_the_corpus_never_assigns_is_one_blocked_item():
    # The manual example's shape: xwrk is assigned by the SAS server (an
    # autoexec, say), never in the corpus. Before, its reads dropped out and
    # the plan said "No external data sources found."
    plan = _plan(
        job=(
            "%let lname = xwrk;\ndata a; set &lname..batch_med; run;\n"
            "data b; set xwrk.other; set sashelp.class; run;\n"
        )
    )
    [item] = plan.items
    assert (item.source.kind, item.source.libref) == (SourceKind.SAS_SESSION, "xwrk")
    assert item.target_table == UNRESOLVED_TARGET
    [blocker] = item.blockers
    assert "xwrk.batch_med, xwrk.other" in blocker and "never assigned" in blocker
    assert "autoexec" in blocker and "XWRK" in blocker


def test_no_item_for_a_library_assigned_written_or_out_of_reach():
    # A LIBNAME anywhere in the corpus assigns it; a table the job makes first
    # is no input; a LIBNAME spelled through a macro variable could assign any
    # libref, so nothing is claimed unassigned; SAS's own librefs need none.
    for job in (
        "data a; set raw.early; run;\nlibname raw '/one';\n",
        "data stg.x; set src; run;\ndata y; set stg.x; run;\n",
        "libname &lib '/x';\ndata a; set q.t; run;\n",
        "data a; set sashelp.class; set sasuser.prefs; run;\n",
    ):
        plan = _plan(job=job)
        assert all(i.source.kind is not SourceKind.SAS_SESSION for i in plan.items), job


def test_an_environment_variable_is_a_libref_or_fileref_on_unix():
    # "An environment variable can be used as a libref": before, educ.class
    # (and ined's file) dropped out of the plan, read through no LIBNAME.
    for setenv in ("x setenv EDUC /users/mydir/educ;", "options set=EDUC '/users/mydir/educ';"):
        plan = _plan(job=f"{setenv}\nproc print data=educ.class; run;\n")
        located = [(i.source.locator, i.source.object_name, i.blockers) for i in plan.items]
        assert located == [("/users/mydir/educ", "class", ())], setenv
    plan = _plan(job="x setenv INED /users/myid/educ.dat;\ndata a; infile ined; input x; run;\n")
    assert [(i.source.locator, i.source.object_name) for i in plan.items] == [
        ("/users/myid", "educ.dat")
    ]


def test_an_environment_variable_nothing_reads_through_is_no_item():
    # A variable is just a variable until a read goes through it; and "if a
    # variable and a libref have the same name ... SAS uses the libref".
    assert _plan(job="x setenv TMPDIR /tmp/work;\ndata a; x=1; run;\n").items == []
    plan = _plan(
        job="x setenv EDUC /env/educ;\nlibname educ '/lib/educ';\nproc print data=educ.class; run;\n"
    )
    assert [(i.source.locator, i.source.object_name) for i in plan.items] == [("/lib/educ", "class")]


def test_a_read_through_a_concatenated_library_is_planned_from_the_first():
    # libname x ('/a' '/b') is one library SAS searches in order. Before, the
    # statement was not recognised: x.m dropped out of the plan, or became a
    # library "never assigned".
    for statement in ("libname x ('/a' '/b');", "libname x v9 ('/a', \"/b\");"):
        plan = _plan(job=f"{statement}\ndata t; set x.m; run;\n")
        [item] = plan.items
        assert (item.source.kind, item.source.locator, item.source.object_name) == (
            SourceKind.SAS_DATASET,
            "/a",
            "m",
        ), statement
        assert item.blockers == ()
        [note] = item.notes
        assert "searches in the order /a, /b" in note and "planned from /a" in note


def test_a_concatenation_takes_a_libref_as_assigned_then():
    # (income, costs, '/u/capgain') searches income's directories, then
    # costs', then /u/capgain; income reassigned afterwards changes nothing.
    plan = _plan(
        job=(
            "libname income ('/u/corpsale', '/u/retail');\n"
            "libname costs ('/u/salaries', '/u/expenses');\n"
            "libname profits (income, costs, '/u/capgain');\n"
            "libname income '/u/elsewhere';\n"
            "data t; set profits.m; run;\n"
        )
    )
    [item] = [i for i in plan.items if i.source.kind is SourceKind.SAS_DATASET]
    assert (item.source.locator, item.source.object_name, item.source.libref) == (
        "/u/corpsale",
        "m",
        "income",
    )
    assert "/u/corpsale, /u/retail, /u/salaries, /u/expenses, /u/capgain" in item.notes[0]
    # The directories searched after the first need no items; the new income,
    # which nothing reads, is a library item as any unread one is.
    assert [i.source.locator for i in plan.items if i is not item] == ["/u/elsewhere"]


def test_a_concatenation_searching_an_unassigned_library_first_is_blocked():
    # SAS looks in prod first, and only a session that assigns prod says where
    # it is; /x stays in the plan, since the member may be there instead.
    plan = _plan(job="libname all (prod '/x');\ndata t; set all.m; run;\n")
    kinds = {i.source.kind: i for i in plan.items}
    assert set(kinds) == {SourceKind.SAS_SESSION, SourceKind.FILE}
    blocker = kinds[SourceKind.SAS_SESSION].blockers[0]
    assert "library 'prod' is never assigned" in blocker
    assert "all.m (the concatenation all searches prod first)" in blocker
    assert kinds[SourceKind.FILE].source.locator == "/x"
    # Searched after a directory, it is only named in the note.
    [item] = _plan(job="libname all ('/x' prod);\ndata t; set all.m; run;\n").items
    assert item.source.locator == "/x"
    assert "/x, library prod (assigned outside the corpus)" in item.notes[0]


def test_a_list_read_through_a_concatenation_keeps_every_directory():
    plan = _plan(job="libname x ('/a' '/b');\ndata t; set x.sales_:; run;\n")
    assert [(i.source.locator, i.source.object_name) for i in plan.items] == [
        ("/a", ""),
        ("/b", ""),
    ]


def test_a_concatenated_fileref_plans_each_file():
    plan = _plan(job="filename f ('/in/q1.csv' '/in/q2.csv');\ndata t; infile f; input x; run;\n")
    assert [(i.source.locator, i.source.object_name) for i in plan.items] == [
        ("/in", "q1.csv"),
        ("/in", "q2.csv"),
    ]


def test_a_concatenation_survives_the_rows():
    # The plan from the rows is the plan from the chunks: the rows keep each
    # place's position, and one concatenation's rows follow each other.
    source = "libname v9 '/v9';\nlibname all (v9 '/x');\ndata t; set all.m; run;\n"
    rows = _rows(job=source)
    assert [(r.name, r.location, r.concat_index) for r in rows if r.statement == "libname"] == [
        ("/v9", "filesystem", None),
        ("v9", "libref", 0),
        ("/x", "filesystem", 1),
    ]
    [result] = _results(job=source)
    chunks = [c for c in result.chunks if c.parent_id is None]
    from_chunks = build_corpus_plan(
        {"job.sas": ([], [r for c in chunks for r in c.metadata.external_refs])},
        datasets={"job.sas": [list(c.metadata.dataset_refs) for c in chunks]},
        config=_config(),
    )
    from_rows = plan_from_inventory(rows, config=_config())
    assert [(i.source.locator, i.source.object_name, i.notes) for i in from_rows.items] == [
        (i.source.locator, i.source.object_name, i.notes) for i in from_chunks.items
    ]
    assert [(i.source.locator, i.source.object_name) for i in from_rows.items] == [("/v9", "m")]


def test_a_view_s_own_libname_serves_its_query_and_nothing_after():
    # CREATE VIEW ... USING LIBNAME: "the scope of the libref is local to the
    # view". Before, it rebound raw for every later read, and a libref only a
    # view assigns counted as assigned.
    plan = _plan(
        job=(
            "libname raw '/data/raw';\nproc sql;\n"
            "create view work.v as select * from raw.t, a.u using libname raw '/other', libname a '/a';\n"
            "quit;\ndata x; set raw.x; set a.k; run;\n"
        )
    )
    located = [(i.source.kind, i.source.locator, i.source.object_name) for i in plan.items]
    assert located == [
        (SourceKind.SAS_DATASET, "/data/raw", "x"),
        (SourceKind.SAS_DATASET, "/other", "t"),
        (SourceKind.SAS_DATASET, "/a", "u"),
        (SourceKind.SAS_SESSION, "", ""),
    ]
    assert "a.k" in plan.items[-1].blockers[0] and plan.items[-1].source.libref == "a"
    rows = _rows(job="proc sql;\ncreate view v as select * from q.t using libname q '/q';\nquit;\n")
    assert [(r.statement, r.role) for r in rows if r.kind is RefKind.PATH] == [("using_libname", "bind")]


def test_a_view_s_own_database_libname_serves_its_query_and_nothing_after():
    plan = _plan(
        job=(
            "libname ora oracle path=P schema=s1;\nproc sql;\n"
            "create view work.v as select * from ora.t using libname ora oracle path=Q schema=s2;\n"
            "quit;\ndata x; set ora.x; run;\n"
        )
    )
    assert [(i.source.locator, i.source.object_name) for i in plan.items] == [
        ("Q", "s2.t"),
        ("P", "s1.x"),
    ]
    rows = _rows(job="proc sql;\ncreate view v as select * from q.t using libname q oracle path=Q;\nquit;\n")
    assert [r.statement for r in rows if r.kind is RefKind.ENGINE] == ["using_libname"]


def test_a_data_set_read_by_its_path_is_planned():
    # `set '/u/2011/budgets/may';` reads may.sas7bdat there: "if you do not
    # specify an extension, SAS uses .sas7bdat" (SAS 9.4 Companion for UNIX
    # Environments). Before, such a read dropped out of the plan, or, holding
    # a dot, became a library "never assigned" named after half the path.
    plan = _plan(
        job=(
            "data a; set '/Data/Prod/Master.sas7bdat' '/u/2011/budgets/may'; run;\n"
            "proc print data='/data/p/y.sas7bdat'; run;\n"
        )
    )
    located = [(i.source.kind, i.source.locator, i.source.object_name) for i in plan.items]
    assert located == [
        (SourceKind.SAS_DATASET, "/Data/Prod", "Master"),
        (SourceKind.SAS_DATASET, "/u/2011/budgets", "may"),
        (SourceKind.SAS_DATASET, "/data/p", "y"),
    ]
    assert not any("never assigned" in b for i in plan.items for b in i.blockers)


def test_a_data_set_read_by_a_relative_path_is_blocked():
    # `merge 'jan' 'feb';` reads from the directory SAS runs in.
    plan = _plan(job="data a; merge 'jan' 'feb'; by id; run;\n")
    assert [(i.source.locator, i.source.object_name) for i in plan.items] == [("", "jan"), ("", "feb")]
    assert all("not a full path" in i.blockers[0] for i in plan.items)


def test_a_data_set_the_job_writes_by_its_path_is_its_own():
    plan = _plan(job="data '/out/res'; set a; run;\ndata b; set '/out/res'; run;\n")
    assert plan.items == []
    # A path a library also reaches is one item, from the first read.
    plan = _plan(job="libname p '/data/prod';\ndata a; set p.master; set '/data/prod/master'; run;\n")
    assert [(i.source.locator, i.source.object_name, i.source.libref) for i in plan.items] == [
        ("/data/prod", "master", "p")
    ]


def test_a_name_literal_is_a_member_not_a_path():
    # 'src tab'n is a WORK member, never a file to load; in a library, the
    # member's file is named without the literal's quotes and n.
    assert _plan(job="data a; set 'src tab'n; run;\n").items == []
    [item] = _plan(job="libname p '/data/p';\ndata a; set p.'My Data'n; run;\n").items
    assert (item.source.locator, item.source.object_name) == ("/data/p", "my data")


def test_a_path_the_sas_server_completes_is_blocked():
    # On UNIX, ~/ is $HOME/, ~name/ a user's home, !sasroot SAS's own
    # directory, and $VARIABLE an environment variable (SAS 9.4 Companion for
    # UNIX Environments, Character Substitutions in Pathnames). Before, each
    # was planned as a literal directory, e.g. sas7bdat:~/data/t, unblocked.
    cases = {
        "libname h '~/data';\ndata a; set h.t; run;\n": "~ is the home directory",
        "data a; infile '~sasdemo/in/x.csv'; input v; run;\n": "~sasdemo is sasdemo's home",
        "libname p '$PROJ/raw';\ndata a; set p.t; run;\n": "$PROJ is an environment variable",
        "data a; infile '/data/$ENV/in.csv'; input v; run;\n": "$ENV is an environment variable",
        "filename s '!sasroot/samples/x.dat';\ndata a; infile s; input v; run;\n": "!SASROOT is",
        "data a; set '~/data/t'; run;\n": "~ is the home directory",
    }
    for job, why in cases.items():
        [item] = _plan(job=job).items
        assert any("completed by the SAS server" in b and why in b for b in item.blockers), job
        # Not "not a full path" as well: the server says where it is.
        assert not any("not a full path" in b for b in item.blockers), job
    [item] = _plan(job="libname q '/data/q';\ndata a; set q.t; run;\n").items
    assert not any("SAS server" in b for b in item.blockers)


def test_a_file_read_through_a_wildcard_is_blocked():
    # UNIX lets FILENAME and INFILE name every file a pattern matches: *, ?
    # and [ ] (SAS 9.4 Companion for UNIX Environments). Before, the plan
    # loaded a file literally named "wild*.dat".
    for job, path in (
        ("filename wild '/data/in/wild*.dat';\ndata; infile wild; input; run;\n", "/data/in/wild*.dat"),
        ("data a; infile '/data/in/m?_[0-9].csv'; input v; run;\n", "/data/in/m?_[0-9].csv"),
        ("filename subfiles '/data/*/*';\ndata; infile subfiles; input; run;\n", "/data/*/*"),
    ):
        [item] = _plan(job=job).items
        # One reason: not also "a library directory", nor "no target name".
        [blocker] = item.blockers
        assert blocker.startswith(f"'{path}' is a wildcard"), job
    plan = _plan(job="filename f ('/a/x*.csv' '/b/y.csv');\ndata; infile f; input; run;\n")
    assert [bool(i.blockers) for i in plan.items] == [True, False]


def test_a_read_through_any_documented_engine_is_planned():
    # SNOW, JDBC, GREENPLM, VERTICA, ... are SAS/ACCESS engines. Unrecognised,
    # their LIBNAME was neither a database nor a directory, and the read gave
    # an empty plan; now it is a database table, blocked like any non-Oracle one.
    for engine in ("snow", "jdbc", "greenplm", "vertica", "ybrick", "sapase", "mongo"):
        plan = _plan(job=f"libname d {engine} server=h user=u;\ndata a; set d.orders; run;\n")
        [item] = plan.items
        assert item.source.object_name == "orders", engine
        assert f"a {engine} database" in item.blockers[0], engine


def test_a_transport_file_is_planned_blocked_once():
    # PROC CIMPORT reads a CPORT transport file, which no reader here opens:
    # planned, blocked, once — by path, or through the FILENAME it names.
    for job in (
        "proc cimport data=work.x infile='/t/x.cpt'; run;\n",
        "filename tf '/t/x.cpt';\nproc cimport ds=x infile=tf; run;\n",
    ):
        [item] = _plan(job=job).items
        assert (item.source.locator, item.source.object_name) == ("/t", "x.cpt")
        assert "transport file" in item.blockers[0]


def test_a_table_appended_to_through_out_is_loaded():
    # OUT= is BASE='s alias in PROC APPEND: the table must already exist, so
    # it is loaded, as with BASE=. Read as a plain write, it dropped out and
    # the whole library was blocked instead.
    for option in ("base", "out"):
        plan = _plan(job=f"libname prod '/data/prod';\nproc append {option}=prod.master data=new; run;\n")
        located = [(i.source.locator, i.source.object_name, i.blockers) for i in plan.items]
        assert located == [("/data/prod", "master", ())], option


def test_a_libname_rebound_in_one_file_binds_only_the_reads_after_it():
    plan = _plan(
        job=(
            "libname raw '/data/2025';\ndata a; set raw.x; run;\n"
            "libname raw '/data/2026';\ndata b; set raw.y; run;\n"
        )
    )
    located = {(i.source.locator, i.source.object_name) for i in plan.items}
    assert located == {("/data/2025", "x"), ("/data/2026", "y")}
    assert not any(i.blockers for i in plan.items)


def test_a_table_a_step_reads_and_rewrites_is_loaded():
    # An in-place sort and `data raw.a; set raw.a;` need the table that was
    # there before them: the step's own write does not make it the job's.
    plan = _plan(
        job=(
            "libname raw '/data/raw';\nproc sort data=raw.customers; by id; run;\n"
            "data raw.accounts; set raw.accounts; where active; run;\n"
            "data work.o; set raw.orders; run;\n"
        )
    )
    assert sorted(i.source.object_name for i in plan.items) == ["accounts", "customers", "orders"]


def test_a_table_renamed_into_place_is_the_jobs_own():
    # CHANGE makes `new` for the statements after it; `old` must exist first.
    plan = _plan(job="libname raw '/data/raw';\nproc datasets lib=raw nolist; change old=new; modify new; quit;\n")
    assert [i.source.object_name for i in plan.items] == ["old"]


def test_a_table_created_before_it_is_read_is_the_jobs_own():
    plan = _plan(
        a_make="libname raw '/data/raw';\ndata raw.made_a; x = 1; run;\n",
        b_read="data work.o; set raw.made_a raw.made_c raw.extern; run;\n",
        c_make="data raw.made_c; x = 1; run;\n",
    )
    # made_a by an earlier file, made_c by another file (taken to run first,
    # as the batcher links it): neither is loaded. extern is nobody's.
    assert [i.source.object_name for i in plan.items] == ["extern"]


def test_a_member_a_macro_call_reads_is_planned():
    src = (
        "libname raw '/data/raw';\n%macro load(ds); data work.copy; set &ds; run; %mend;\n"
        "data work.o; set raw.orders; run;\n%load(raw.customers);\n"
    )
    plan = _plan(job=src)
    assert [i.source.object_name for i in plan.items] == ["orders", "customers"]
    # The call is where the read happens, so the call's file owns it.
    assert {i.source.source_id for i in plan.items} == {"job.sas"}


def test_a_read_through_an_unresolved_library_keeps_every_library_item():
    # &where could be raw: what it reads is in no list the plan can make.
    plan = _plan(job="libname raw '/data/raw';\ndata work.o; set raw.orders &where..t; run;\n")
    kinds = sorted((i.source.kind.value, i.source.object_name) for i in plan.items)
    assert kinds == [("file", ""), ("sas7bdat", "orders")]
    [library] = [i for i in plan.items if i.source.kind is SourceKind.FILE]
    assert "library directory" in library.blockers[0]


def test_a_library_declared_twice_is_planned_once():
    # Assigned again in another file, or again in the same one: one item per
    # member read, where the library is first declared.
    for files in (
        {"a_setup": "libname raw '/data/raw';\n", "b_job": "libname raw '/data/raw';\ndata a; set raw.customers; run;\n"},
        {"job": "libname raw '/data/raw';\nlibname raw '/data/raw';\ndata a; set raw.customers; run;\n"},
    ):
        [item] = _plan(**files).items
        assert str(item.source) == "sas7bdat:/data/raw/customers (raw)", files


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

    def option(self, key, value):
        self.log.append(("option", key, value))
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
    assert data[0][:3] == (rows[0].run_id, rows[0].corpus, rows[0].recorded_at)
    assert all(len(values) == len(_COLUMNS) for values in data)
    assert isinstance(data[0][[n for n, _, _ in _COLUMNS].index("kind")], str)
    # The schema merges, so a table made before a column was added gains it.
    assert spark.write_log == [
        ("format", "delta"),
        ("mode", "append"),
        ("option", "mergeSchema", "true"),
        ("saveAsTable", "main.meta.sas_refs"),
    ]


def test_nothing_to_write_touches_no_table():
    spark = _FakeSpark()
    assert write_inventory([], "main.meta.sas_refs", spark=spark) == 0
    assert spark.sql_log == [] and spark.created == []


def test_without_a_session_the_active_one_is_used(monkeypatch):
    import app_config.spark

    spark = _FakeSpark()
    asked: list[str] = []
    monkeypatch.setattr(
        app_config.spark, "active_or_new_session", lambda name: asked.append(name) or spark
    )
    rows = _rows(job="data a; set b; run;\n")
    assert write_inventory(rows, "sas_refs") == len(rows)
    assert asked == ["sas-parser-hydration"]
    # A one-part name lives in the session's current schema: none to create.
    [create_table] = spark.sql_log
    assert create_table.startswith("CREATE TABLE IF NOT EXISTS `sas_refs`")


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
    argv = (str(corpus_dir), "--dry-run", "--inventory-table", "main.meta.refs", "--corpus", "etl")
    assert _main(*argv) == 0
    written = capsys.readouterr().out
    rows = stored["main.meta.refs"]
    assert {r.name for r in rows} >= {"/data/raw", "raw.customers"}
    assert {r.corpus for r in rows} == {"etl"}
    assert "sas7bdat:/data/raw/customers (raw)" in written

    # A later job plans from the table: no source directory, no chunker.
    asked: list[tuple[str | None, str | None]] = []

    def read(table, run_id=None, corpus=None, **_):
        asked.append((corpus, run_id))
        return stored[table]

    monkeypatch.setattr(inventory, "read_inventory", read)
    argv = ("--from-inventory", "--inventory-table", "main.meta.refs", "--corpus", "etl", "--dry-run")
    assert _main(*argv) == 0
    assert asked == [("etl", None)]
    assert capsys.readouterr().out.splitlines()[2:] == written.splitlines()[2:]


def test_a_dry_run_leaves_a_configured_table_alone(corpus_dir, monkeypatch, capsys):
    # The deployment's table, from config: a trial plan writes nothing to it.
    import data_hydration.__main__ as cli
    import data_hydration.inventory as inventory

    written: list[str] = []
    monkeypatch.setattr(
        cli,
        "_config_for",
        lambda args: HydrationConfig(catalog="main", inventory_table=args.inventory_table or "prod.meta.refs"),
    )
    monkeypatch.setattr(inventory, "write_inventory", lambda rows, table, **_: written.append(table))
    assert _main(str(corpus_dir), "--dry-run") == 0
    assert written == []
    # Named on the command line, the table is written to, dry run or not.
    assert _main(str(corpus_dir), "--dry-run", "--inventory-table", "dev.meta.refs") == 0
    assert written == ["dev.meta.refs"]


def test_a_corpus_is_named_by_its_directory_unless_given_a_name(corpus_dir, monkeypatch):
    import data_hydration.inventory as inventory

    stored: dict[str, list[InventoryRow]] = {}
    monkeypatch.setattr(
        inventory, "write_inventory", lambda rows, table, **_: stored.setdefault(table, rows)
    )
    assert _main(str(corpus_dir), "--dry-run", "--inventory-table", "m.s.t") == 0
    assert {r.corpus for r in stored["m.s.t"]} == {str(corpus_dir.resolve())}


def test_a_table_that_cannot_be_written_fails_the_run_not_the_plan(corpus_dir, monkeypatch, capsys):
    import data_hydration.inventory as inventory

    def refuse(*_, **__):
        raise RuntimeError("no Spark here")

    monkeypatch.setattr(inventory, "write_inventory", refuse)
    assert _main(str(corpus_dir), "--dry-run", "--inventory-table", "main.meta.refs") == 1
    assert "sas7bdat:/data/raw/customers (raw)" in capsys.readouterr().out


def test_a_real_run_loads_the_plan_and_fails_on_a_table_it_could_not_keep(corpus_dir, monkeypatch):
    import data_hydration.inventory as inventory
    import data_hydration.runner as runner
    from data_hydration.models import HydrationReport

    ran: list[int] = []
    monkeypatch.setattr(
        runner, "execute", lambda plan, **_: ran.append(len(plan.items)) or HydrationReport(plan=plan)
    )
    monkeypatch.setattr(inventory, "write_inventory", lambda rows, table, **_: len(rows))
    assert _main(str(corpus_dir), "--inventory-table", "main.meta.refs") == 0

    def refuse(*_, **__):
        raise RuntimeError("no Spark here")

    monkeypatch.setattr(inventory, "write_inventory", refuse)
    assert _main(str(corpus_dir), "--inventory-table", "main.meta.refs") == 1
    # The table costs the run its status, never its loads.
    assert ran == [1, 1]


@pytest.mark.parametrize(
    ("read", "message"),
    [
        (RuntimeError("table not found"), "could not read the reference inventory"),
        ([], "no inventory run to plan from"),
    ],
)
def test_an_inventory_that_cannot_be_read_or_is_empty_plans_nothing(read, message, monkeypatch, capsys):
    import data_hydration.inventory as inventory

    def read_inventory(*_, **__):
        if isinstance(read, Exception):
            raise read
        return read

    monkeypatch.setattr(inventory, "read_inventory", read_inventory)
    argv = ("--from-inventory", "--inventory-table", "main.meta.refs", "--corpus", "etl", "--dry-run")
    assert _main(*argv) == 1
    captured = capsys.readouterr()
    assert message in captured.err
    assert "Hydration plan" not in captured.out


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--from-inventory", "SRC"], "not a source directory"),
        ([], "a source directory is required"),
        (["MISSING"], "source directory not found"),
        (["SRC", "--run-id", "r1"], "needs --from-inventory"),
        # The table may hold many corpora: name the one, or the run.
        (["--from-inventory"], "needs --corpus"),
        (["SRC", "--inventory-table", "main.meta.refs; drop"], "identifier parts"),
    ],
)
def test_argument_errors(argv, message, corpus_dir, capsys):
    places = {"SRC": str(corpus_dir), "MISSING": str(corpus_dir / "nowhere")}
    argv = [places[a] if a in places else a for a in argv]
    assert _main(*argv) == 2
    assert message in capsys.readouterr().err


def test_from_inventory_needs_a_table(monkeypatch, capsys):
    monkeypatch.delenv("DATA_HYDRATION_INVENTORY_TABLE", raising=False)
    import data_hydration.__main__ as cli

    monkeypatch.setattr(
        cli, "_config_for", lambda args: HydrationConfig(catalog="main", inventory_table=None)
    )
    assert _main("--from-inventory", "--corpus", "etl", "--dry-run") == 2
    assert "needs an inventory table" in capsys.readouterr().err


def test_run_ids_sort_in_time_order():
    from data_hydration.inventory import new_run_id

    early = new_run_id(datetime(2026, 1, 2, 3, 4, 5))
    late = new_run_id(datetime(2026, 1, 2, 3, 4, 6))
    assert early.startswith("20260102T030405Z-") and early < late
