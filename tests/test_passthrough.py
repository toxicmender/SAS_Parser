"""
test_passthrough.py — database tables reached through SQL pass-through and
database-engine LIBNAMEs.

SAS reads a database's tables two ways, and both are registered as
:class:`~chunker.models.SasDbTableRef` records — the table in the database's own
terms (engine · schema · table), linked to the SAS dataset the read lands in:

- explicit SQL pass-through (``CONNECT TO`` / ``FROM CONNECTION TO`` /
  ``EXECUTE ... BY``), recognised by ``chunker.passthrough``; the native SQL is
  masked from every SAS-side dataset scan, so ``from connection to oracle`` no
  longer invents ``work.connection`` and an Oracle owner no longer reads as a
  SAS libref;
- SAS names under a libref a database-engine LIBNAME bound
  (``set edw.accounts;``), resolved by ``chunker.metadata.resolve_db_librefs``.

Run:  python -m pytest tests/test_passthrough.py -v
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chunker import (
    DbTableAccess,
    DbTableVia,
    SasChunkBatcher,
    SasCorpus,
    SasSemanticChunker,
    resolve_corpus_references,
)
from chunker.batcher import MultiFileBatcher
from chunker.metadata import resolve_references
from chunker.models import SasChunkKind
from chunker.passthrough import _NativeTables, split_table_name

# ── helpers ────────────────────────────────────────────────────────────────


def _chunk(source: str, source_id: str = "test.sas", **kwargs):
    kwargs.setdefault("min_words", 1)
    kwargs.setdefault("max_words", 9_999)
    return SasSemanticChunker(**kwargs).chunk_text(source, source_id=source_id)


def _of_kind(result, kind: SasChunkKind):
    return [c for c in result.chunks if c.kind == kind]


def _tables(result) -> list[str]:
    """Every database table the result's chunks name, as their one-line form."""
    return [str(t) for c in result.chunks for t in c.metadata.db_tables]


def _native(sql: str) -> tuple[list[str], list[str]]:
    native = _NativeTables(sql)
    return native.reads(), native.writes()


# The example the feature was requested with, verbatim — including the missing
# semicolon after `proc sql`.
EXAMPLE = """proc sql
connect to oracle (user=&ora_user password=&ora_pass path=&ora_path);
create table nonip as select * from connection to oracle
(select cov_month, count as count from edw_export.current_nonip where table_cd='MED');
disconnect from oracle;
quit;
"""

EXAMPLE_LINE = (
    "oracle:edw_export.current_nonip → work.nonip (read via connection_to oracle)"
)


# ── 1. The example ─────────────────────────────────────────────────────────


class TestExample(unittest.TestCase):
    def _proc(self, source: str = EXAMPLE):
        return _of_kind(_chunk(source), SasChunkKind.PROC_STEP)[0]

    def test_the_oracle_table_is_registered_in_oracle_terms(self):
        (table,) = self._proc().metadata.db_tables
        self.assertEqual(table.engine, "oracle")
        self.assertEqual(table.connection, "oracle")
        self.assertEqual(table.db_schema, "edw_export")
        self.assertEqual(table.table, "current_nonip")
        self.assertEqual(table.qualified, "edw_export.current_nonip")
        self.assertIs(table.access, DbTableAccess.READ)
        self.assertIs(table.via, DbTableVia.CONNECTION_TO)
        self.assertEqual(table.raw, "edw_export.current_nonip")
        self.assertFalse(table.has_macro_ref)

    def test_the_sas_copy_is_registered_and_linked(self):
        meta = self._proc().metadata
        self.assertEqual(meta.output_datasets, ["work.nonip"])
        self.assertEqual(meta.db_tables[0].sas_targets, ("work.nonip",))
        self.assertEqual(str(meta.db_tables[0]), EXAMPLE_LINE)

    def test_the_connection_options_ride_on_the_record(self):
        (table,) = self._proc().metadata.db_tables
        self.assertEqual(
            table.option_map,
            {"user": "&ora_user", "password": "&ora_pass", "path": "&ora_path"},
        )
        # The one-line form reaches the LLM prompt; credentials stay out of it.
        self.assertNotIn("&ora_pass", str(table))

    def test_native_sql_invents_no_sas_dataset(self):
        meta = self._proc().metadata
        self.assertEqual(meta.input_datasets, [])
        for bogus in ("work.connection", "work.oracle", "edw_export.current_nonip"):
            self.assertNotIn(bogus, meta.referenced_datasets)

    def test_the_oracle_owner_is_not_a_sas_libref(self):
        self.assertEqual(self._proc().metadata.referenced_librefs, ["work"])

    def test_macro_variables_in_the_connection_are_still_referenced(self):
        refs = self._proc().metadata.referenced_macro_vars
        for name in ("ora_user", "ora_pass", "ora_path"):
            self.assertIn(name, refs)

    def test_with_and_without_the_semicolon_after_proc_sql(self):
        fixed = EXAMPLE.replace("proc sql\n", "proc sql;\n", 1)
        self.assertEqual(
            self._proc(fixed).metadata.db_tables, self._proc().metadata.db_tables
        )

    def test_the_batch_needs_no_libname_for_the_oracle_owner(self):
        result = _chunk(EXAMPLE + "proc means data=nonip; run;\n")
        batch = SasChunkBatcher().batch(result).batches[0]
        self.assertEqual(batch.required_librefs, [])
        self.assertEqual(batch.input_datasets, [])
        self.assertEqual([str(t) for t in batch.db_tables], [EXAMPLE_LINE])


# ── 2. Connections ─────────────────────────────────────────────────────────


class TestConnections(unittest.TestCase):
    def test_an_alias_maps_to_its_engine_and_options(self):
        src = (
            "proc sql;\nconnect to oracle as edw (path=EDWPRO);\n"
            "create table a as select * from connection to edw (select * from s.t);\n"
            "quit;\n"
        )
        (table,) = _chunk(src).chunks[0].metadata.db_tables
        self.assertEqual((table.engine, table.connection), ("oracle", "edw"))
        self.assertEqual(table.option_map, {"path": "EDWPRO"})

    def test_an_engine_named_alias_without_a_connect_is_that_engine(self):
        # The CONNECT was made by a macro call this chunk cannot see.
        src = (
            "proc sql;\n%connect_edw;\n"
            "create table a as select * from connection to oracle (select * from s.t);\n"
            "quit;\n"
        )
        (table,) = _chunk(src).chunks[0].metadata.db_tables
        self.assertEqual(table.engine, "oracle")
        self.assertEqual(table.options, ())

    def test_an_unknown_alias_without_a_connect_has_no_engine(self):
        src = (
            "proc sql;\n"
            "create table a as select * from connection to mydb (select * from s.t);\n"
            "quit;\n"
        )
        (table,) = _chunk(src).chunks[0].metadata.db_tables
        self.assertIsNone(table.engine)
        self.assertTrue(str(table).startswith("mydb:s.t"))

    def test_a_quoted_connection_string_with_parentheses(self):
        src = (
            "proc sql;\n"
            'connect to oracle (path="(DESCRIPTION=(ADDRESS=(HOST=db1)))" user=svc);\n'
            "create table a as select * from connection to oracle (select * from s.t);\n"
            "quit;\n"
        )
        meta = _chunk(src).chunks[0].metadata
        self.assertEqual(
            meta.db_tables[0].option_map["path"], "(DESCRIPTION=(ADDRESS=(HOST=db1)))"
        )
        self.assertEqual(meta.output_datasets, ["work.a"])

    def test_a_reused_alias_takes_the_latest_connects_options(self):
        src = (
            "proc sql;\n"
            "connect to oracle (path=ONE);\n"
            "create table a as select * from connection to oracle (select * from s.t);\n"
            "disconnect from oracle;\n"
            "connect to oracle (path=TWO);\n"
            "create table b as select * from connection to oracle (select * from s.t);\n"
            "quit;\n"
        )
        paths = sorted(
            t.option_map["path"] for t in _chunk(src).chunks[0].metadata.db_tables
        )
        self.assertEqual(paths, ["ONE", "TWO"])

    def test_connect_using_takes_its_engine_from_the_libname(self):
        src = (
            "libname edw oracle path=EDWPRO schema=fr_dm;\n"
            "proc sql;\nconnect using edw;\n"
            "create table a as select * from connection to edw (select * from s.t);\n"
            "quit;\n"
        )
        proc = _of_kind(_chunk(src), SasChunkKind.PROC_STEP)[0]
        (table,) = proc.metadata.db_tables
        self.assertEqual((table.engine, table.connection), ("oracle", "edw"))
        self.assertEqual(table.option_map["path"], "EDWPRO")

    def test_connect_using_without_a_visible_libname_has_no_engine(self):
        src = (
            "proc sql;\nconnect using edw as e;\n"
            "create table a as select * from connection to e (select * from s.t);\n"
            "quit;\n"
        )
        (table,) = _chunk(src).chunks[0].metadata.db_tables
        self.assertIsNone(table.engine)
        self.assertEqual(table.connection, "edw")


# ── 3. EXECUTE ... BY ──────────────────────────────────────────────────────


class TestExecute(unittest.TestCase):
    def _tables(self, statement: str):
        src = f"proc sql;\nconnect to oracle (path=P);\n{statement}\nquit;\n"
        return _chunk(src).chunks[0].metadata

    def test_create_table_as_writes_one_table_and_reads_another(self):
        meta = self._tables(
            "execute (create table stage.t as select * from base.s) by oracle;"
        )
        got = {(t.qualified, t.access, t.via) for t in meta.db_tables}
        self.assertEqual(
            got,
            {
                ("stage.t", DbTableAccess.WRITE, DbTableVia.EXECUTE),
                ("base.s", DbTableAccess.READ, DbTableVia.EXECUTE),
            },
        )
        self.assertTrue(all(t.sas_targets == () for t in meta.db_tables))
        # Nothing on the SAS side: the data stays in Oracle.
        self.assertEqual(meta.input_datasets, [])
        self.assertEqual(meta.output_datasets, [])

    def test_the_newer_execute_by_order(self):
        meta = self._tables("execute by oracle (truncate table stage.tmp);")
        self.assertEqual(
            [(t.qualified, t.access) for t in meta.db_tables],
            [("stage.tmp", DbTableAccess.WRITE)],
        )

    def test_delete_from_is_a_write_not_a_read(self):
        meta = self._tables(
            "execute (delete from s.d where k in (select k from s.keys)) by oracle;"
        )
        got = {(t.qualified, t.access) for t in meta.db_tables}
        self.assertEqual(
            got, {("s.d", DbTableAccess.WRITE), ("s.keys", DbTableAccess.READ)}
        )

    def test_call_execute_in_a_data_step_is_not_pass_through(self):
        src = "data _null_;\n  call execute('%build(x)');\nrun;\n"
        meta = _chunk(src).chunks[0].metadata
        self.assertEqual(meta.db_tables, [])
        self.assertIn("build", meta.invokes_macros)


# ── 4. The native SQL scan ─────────────────────────────────────────────────


class TestNativeSql(unittest.TestCase):
    def test_comma_lists_joins_and_subqueries(self):
        reads, _ = _native(
            "select a from s.t1 x, s.t2 y join s.t3 z on 1=1 "
            "left outer join (select * from s.t4) w on 1=1 "
            "where c in (select d from s.t5)"
        )
        self.assertEqual(reads, ["s.t1", "s.t2", "s.t3", "s.t4", "s.t5"])

    def test_cte_names_and_dual_are_not_tables(self):
        reads, _ = _native(
            "with q as (select * from real.base) "
            "select (select 1 from dual) from q, sys.dual"
        )
        self.assertEqual(reads, ["real.base"])

    def test_from_inside_extract_and_trim_is_an_argument(self):
        reads, _ = _native(
            "select extract(year from dt), trim(leading 'x' from c) from s.t"
        )
        self.assertEqual(reads, ["s.t"])

    def test_literals_and_line_comments_name_nothing(self):
        reads, _ = _native(
            "select * from s.t -- from fake.comment\nwhere x = 'from fake.literal'"
        )
        self.assertEqual(reads, ["s.t"])

    def test_table_functions_lateral_partition_and_sample(self):
        reads, _ = _native(
            "select * from table(f(x)) t, s.big partition (p1) b, "
            "lateral (select 1 from s.lat) l, s.smp sample (10) z"
        )
        self.assertEqual(sorted(reads), ["s.big", "s.lat", "s.smp"])

    def test_write_forms(self):
        cases = {
            "insert all into a.one values (1) into a.two values (2) select * from s.src":
                (["s.src"], ["a.one", "a.two"]),
            "merge into tgt.m t using src.s s on (t.id=s.id) when matched then update set c=s.c":
                (["src.s"], ["tgt.m"]),
            "update s.u set c=(select max(x) from s.mx)": (["s.mx"], ["s.u"]),
            "create global temporary table s.gtt as select * from s.src2":
                (["s.src2"], ["s.gtt"]),
            "create or replace view s.v as select * from s.vsrc": (["s.vsrc"], ["s.v"]),
            "drop table stage.old purge": ([], ["stage.old"]),
            "insert into s.t (a, b) values (1, 2)": ([], ["s.t"]),
            "select * from s.t for update of c": (["s.t"], []),
        }
        for sql, expected in cases.items():
            with self.subTest(sql=sql):
                self.assertEqual(_native(sql), expected)

    def test_split_table_name(self):
        self.assertEqual(split_table_name('"EDW"."Current_NonIP"'), ("edw", "current_nonip", None))
        self.assertEqual(split_table_name("s.t@ProdLink"), ("s", "t", "prodlink"))
        self.assertEqual(split_table_name("&sch..t"), ("&sch.", "t", None))
        self.assertEqual(split_table_name("db.dbo.t"), ("db.dbo", "t", None))
        # Unresolved: a schema keeps the delimiters SAS needs to read it back.
        self.assertEqual(split_table_name('"&sch"."&tbl"'), ("&sch.", "&tbl", None))
        self.assertEqual(split_table_name("&&sch_&env...t"), ("&&sch_&env..", "t", None))
        self.assertEqual(split_table_name("db.&sch..t"), ("db.&sch.", "t", None))
        self.assertEqual(split_table_name("edw.&tbl."), ("edw", "&tbl", None))
        self.assertEqual(split_table_name("db..t"), ("db", "t", None))
        self.assertEqual(split_table_name("t"), (None, "t", None))

    def test_quoted_identifiers_keep_their_spelling_in_raw(self):
        src = (
            "proc sql;\ncreate table a as select * from connection to oracle\n"
            '(select * from "EDW"."T" x);\nquit;\n'
        )
        (table,) = _chunk(src).chunks[0].metadata.db_tables
        self.assertEqual((table.db_schema, table.table), ("edw", "t"))
        self.assertEqual(table.raw, '"EDW"."T"')

    def test_a_dblink_and_an_unqualified_table(self):
        src = (
            "proc sql;\ncreate table a as select * from connection to oracle\n"
            "(select * from s.u@prodlink, current_nonip);\nquit;\n"
        )
        tables = {t.raw: t for t in _chunk(src).chunks[0].metadata.db_tables}
        self.assertEqual(tables["s.u@prodlink"].dblink, "prodlink")
        self.assertIsNone(tables["current_nonip"].db_schema)


# ── 5. The SAS copy ────────────────────────────────────────────────────────


class TestSasTargets(unittest.TestCase):
    def _targets(self, statement: str) -> list[tuple[str, ...]]:
        src = f"proc sql;\nconnect to oracle (path=P);\n{statement}\nquit;\n"
        return [t.sas_targets for t in _chunk(src).chunks[0].metadata.db_tables]

    def test_insert_into_a_sas_table(self):
        self.assertEqual(
            self._targets(
                "insert into mart.x select * from connection to oracle (select * from s.t);"
            ),
            [("mart.x",)],
        )

    def test_create_view(self):
        self.assertEqual(
            self._targets(
                "create view v as select * from connection to oracle (select * from s.t);"
            ),
            [("work.v",)],
        )

    def test_a_bare_select_has_no_copy(self):
        self.assertEqual(
            self._targets("select * from connection to oracle (select * from s.t);"),
            [()],
        )

    def test_select_into_a_macro_variable_has_no_copy(self):
        src = (
            "proc sql;\nconnect to oracle (path=P);\n"
            "select count(*) into :n from connection to oracle (select * from s.t);\n"
            "quit;\n"
        )
        meta = _chunk(src).chunks[0].metadata
        self.assertEqual(meta.db_tables[0].sas_targets, ())
        self.assertIn("n", meta.produces_macrovars)

    def test_two_queries_feeding_one_create(self):
        targets = self._targets(
            "create table u as select * from connection to oracle (select * from s.a)\n"
            "  union select * from connection to oracle (select * from s.b);"
        )
        self.assertEqual(targets, [("work.u",), ("work.u",)])


# ── 6. Masking ─────────────────────────────────────────────────────────────


class TestMasking(unittest.TestCase):
    def test_a_macro_call_inside_native_sql_is_still_invoked(self):
        # SAS resolves macros before the text is sent, so this one runs.
        src = (
            "proc sql;\ncreate table a as select * from connection to oracle\n"
            "(select * from s.t where region in (%region_list));\nquit;\n"
        )
        self.assertIn("region_list", _chunk(src).chunks[0].metadata.invokes_macros)

    def test_inside_a_macro_body(self):
        src = (
            "%macro pull;\nproc sql;\nconnect to oracle as edw (path=&p);\n"
            "create table work.raw as select * from connection to edw "
            "(select * from fin.ledger);\n"
            "execute (truncate table stage.tmp) by edw;\n"
            "disconnect from edw;\nquit;\n%mend;\n"
        )
        macro = _of_kind(_chunk(src), SasChunkKind.MACRO_DEFINITION)[0]
        self.assertEqual(macro.metadata.body_literal_inputs, [])
        self.assertEqual(macro.metadata.body_literal_outputs, ["work.raw"])
        self.assertEqual(
            sorted(str(t) for t in macro.metadata.db_tables),
            [
                "oracle:fin.ledger → work.raw (read via connection_to edw)",
                "oracle:stage.tmp (write via execute edw)",
            ],
        )

    def test_sas_side_tables_beside_a_connection_are_still_read(self):
        src = (
            "proc sql;\ncreate table a as select x.*, y.v from work.local y\n"
            "  join connection to oracle (select * from s.t) x on x.id = y.id;\nquit;\n"
        )
        meta = _chunk(src).chunks[0].metadata
        self.assertEqual(meta.input_datasets, ["work.local"])
        self.assertEqual([t.qualified for t in meta.db_tables], ["s.t"])


# ── 7. Macro variables in names ────────────────────────────────────────────


class TestMacroResolution(unittest.TestCase):
    def _query(self, prefix: str = "", table: str = "&sch..accounts") -> str:
        return (
            f"{prefix}proc sql;\ncreate table a as select * from connection to oracle\n"
            f"(select * from {table});\nquit;\n"
        )

    def test_a_resolved_schema(self):
        result = _chunk(self._query("%let sch = edw;\n"))
        (table,) = _of_kind(result, SasChunkKind.PROC_STEP)[0].metadata.db_tables
        self.assertEqual(table.qualified, "edw.accounts")
        self.assertFalse(table.has_macro_ref)
        self.assertEqual(table.raw, "&sch..accounts")

    def test_an_unresolved_schema_is_kept_as_written(self):
        (table,) = _chunk(self._query()).chunks[0].metadata.db_tables
        self.assertEqual((table.db_schema, table.qualified), ("&sch.", "&sch..accounts"))
        self.assertTrue(table.has_macro_ref)

    def test_a_sas_copy_named_by_a_macro_variable(self):
        src = (
            "%let out = mart.result;\nproc sql;\n"
            "create table &out as select * from connection to oracle (select * from s.t);\n"
            "quit;\n"
        )
        proc = _of_kind(_chunk(src), SasChunkKind.PROC_STEP)[0]
        self.assertEqual(proc.metadata.db_tables[0].sas_targets, ("mart.result",))

    def test_a_let_in_another_file(self):
        setup = _chunk("%let sch = edw;\n", source_id="setup.sas")
        job = _chunk(self._query(), source_id="job.sas")
        corpus = SasCorpus(file_results=[setup, job])
        resolved = resolve_corpus_references(corpus)
        (table,) = resolved.file_results[1].chunks[0].metadata.db_tables
        self.assertEqual(table.qualified, "edw.accounts")
        # The input corpus is left as it was.
        self.assertTrue(job.chunks[0].metadata.db_tables[0].has_macro_ref)
        batched = MultiFileBatcher().batch(corpus)
        self.assertIn(
            "edw.accounts",
            [t.qualified for item in batched.all_ordered_items
             for c in getattr(item, "chunks", [item]) for t in c.metadata.db_tables],
        )


# ── 7b. Every source of a macro variable's value ───────────────────────────


def _read(source: str, **kwargs):
    """Every database table *source* names, as ``(record, chunk kind)`` pairs."""
    return [
        (t, c.kind)
        for c in _chunk(source, **kwargs).chunks
        for t in c.metadata.db_tables
    ]


def _query(sql: str, prefix: str = "", alias: str = "oracle") -> str:
    return (
        f"{prefix}proc sql;\nconnect to oracle (path=P);\n"
        f"create table out as select * from connection to {alias}\n({sql});\nquit;\n"
    )


PULL = (
    "%macro pull(schema=edw_export, tbl=, out=);\n"
    "proc sql;\nconnect to oracle (path=P);\n"
    "create table &out as select * from connection to oracle\n"
    "  (select * from &schema..&tbl);\n"
    "quit;\n%mend;\n"
)


class TestNameShapes(unittest.TestCase):
    """Where a macro reference sits in an Oracle name, resolved or not."""

    def _one(self, source: str):
        (table,) = [t for t, _ in _read(source)]
        return table

    def test_schema_table_or_both(self):
        for sql, prefix in (
            ("&sch..current_nonip", "%let sch = EDW_EXPORT;\n"),
            ("edw_export.&tbl", "%let tbl = CURRENT_NONIP;\n"),
            ("&sch..&tbl", "%let sch = edw_export;\n%let tbl = current_nonip;\n"),
            ("&full", "%let full = edw_export.current_nonip;\n"),
            ('"&sch"."&tbl"', "%let sch = EDW_EXPORT;\n%let tbl = CURRENT_NONIP;\n"),
        ):
            with self.subTest(sql=sql):
                table = self._one(_query(f"select * from {sql}", prefix))
                self.assertEqual(table.qualified, "edw_export.current_nonip")
                self.assertFalse(table.has_macro_ref)

    def test_a_reference_inside_a_name(self):
        table = self._one(_query("select * from edw_export.t_&sfx._v", "%let sfx = med;\n"))
        self.assertEqual(table.qualified, "edw_export.t_med_v")

    def test_an_unresolved_name_splits_where_sas_would(self):
        # The dot after &sfx is its delimiter, not a separator: one table.
        table = self._one(_query("select * from edw_export.t_&sfx._v"))
        self.assertEqual((table.db_schema, table.table), ("edw_export", "t_&sfx._v"))
        self.assertTrue(table.has_macro_ref)
        table = self._one(_query("select * from &sch..current_nonip"))
        self.assertEqual((table.db_schema, table.table), ("&sch.", "current_nonip"))

    def test_an_indirect_reference_needs_its_extra_dot(self):
        prefix = "%let env = prod;\n%let sch_prod = edw_prod;\n"
        table = self._one(_query("select * from &&sch_&env...current_nonip", prefix))
        self.assertEqual(table.qualified, "edw_prod.current_nonip")
        # With two dots each rescan eats one, and SAS itself sends one name.
        table = self._one(_query("select * from &&sch_&env..current_nonip", prefix))
        self.assertEqual(table.qualified, "edw_prodcurrent_nonip")

    def test_one_dot_after_a_reference_concatenates(self):
        table = self._one(_query("select * from &sch.current_nonip", "%let sch = edw_;\n"))
        self.assertEqual(table.qualified, "edw_current_nonip")

    def test_an_unresolved_name_reads_back_as_sas_would(self):
        # &sch.current_nonip would be one name to SAS: the schema keeps its delimiter.
        for sql, qualified in (
            ("&sch..current_nonip", "&sch..current_nonip"),
            ('"&sch"."&tbl"', "&sch..&tbl"),
            ("&&sch_&env...current_nonip", "&&sch_&env...current_nonip"),
            ("edw_export.&tbl", "edw_export.&tbl"),
        ):
            with self.subTest(sql=sql):
                table = self._one(_query(f"select * from {sql}"))
                self.assertEqual(table.qualified, qualified)
                self.assertTrue(table.has_macro_ref)
        table = self._one(_query("select * from &sch..current_nonip"))
        self.assertEqual(
            str(table), "oracle:&sch..current_nonip → work.out (read via connection_to oracle)"
        )


class TestValueSources(unittest.TestCase):
    """Where the value comes from — and when it stops being known."""

    def _qualified(self, source: str) -> list[str]:
        return [t.qualified for t, kind in _read(source) if kind == SasChunkKind.PROC_STEP]

    def test_call_symputx_with_literals(self):
        prefix = "data _null_;\n  call symputx('sch', 'EDW_EXPORT');\nrun;\n"
        self.assertEqual(
            self._qualified(_query("select * from &sch..t", prefix)), ["edw_export.t"]
        )

    def test_a_value_from_a_data_column_is_unknown(self):
        prefix = "%let sch = old;\ndata _null_; set cfg; call symputx('sch', s); run;\n"
        self.assertEqual(self._qualified(_query("select * from &sch..t", prefix)), ["&sch..t"])

    def test_two_conflicting_literal_calls_are_unknown(self):
        prefix = (
            "data _null_; if x then call symputx('sch','A'); "
            "else call symputx('sch','B'); run;\n"
        )
        self.assertEqual(self._qualified(_query("select * from &sch..t", prefix)), ["&sch..t"])

    def test_sql_into_replaces_an_earlier_let(self):
        prefix = "%let sch = old;\nproc sql; select s into :sch from cfg; quit;\n"
        self.assertEqual(self._qualified(_query("select * from &sch..t", prefix))[-1], "&sch..t")

    def test_a_let_to_a_non_name_replaces_an_earlier_one(self):
        prefix = "%let sch = old;\n%let sch = %scan(&list, 2);\n"
        self.assertEqual(self._qualified(_query("select * from &sch..t", prefix)), ["&sch..t"])

    def test_a_libname_schema_option_in_quotes(self):
        src = (
            '%let sch = FR_DM;\nlibname edw oracle path=P schema="&sch";\n'
            "data work.a; set edw.accounts; run;\n"
        )
        self.assertEqual(
            [t.qualified for t, _ in _read(src) if t.via is DbTableVia.LIBNAME],
            ["fr_dm.accounts"],
        )

    def test_an_unresolved_libname_schema(self):
        src = "libname edw oracle path=P schema=&sch;\ndata work.a; set edw.accounts; run;\n"
        (table,) = [t for t, _ in _read(src) if t.via is DbTableVia.LIBNAME]
        self.assertEqual(table.qualified, "&sch..accounts")
        self.assertTrue(table.has_macro_ref)


class TestMacroCalls(unittest.TestCase):
    """Utility macros: tables named by parameters, globals set by a call."""

    def test_a_call_resolves_its_parameters(self):
        rows = _read(PULL + "%pull(tbl=current_nonip, out=nonip);\n")
        template = [t for t, k in rows if k == SasChunkKind.MACRO_DEFINITION]
        called = [t for t, k in rows if k == SasChunkKind.MACRO_CALL]
        self.assertEqual(len(template), 1)
        self.assertTrue(template[0].parameterised)
        (table,) = called
        self.assertEqual(
            str(table),
            "oracle:edw_export.current_nonip → work.nonip "
            "(read via connection_to oracle in %pull)",
        )
        self.assertFalse(table.parameterised)

    def test_positional_arguments_and_a_global_passed_in(self):
        src = (
            "%macro p2(s, t);\nproc sql;\ncreate table x as select * from "
            "connection to oracle (select * from &s..&t);\nquit;\n%mend;\n"
            "%let mytab = accounts;\n%p2(fr_dm, &mytab);\n"
        )
        called = [t.qualified for t, k in _read(src) if k == SasChunkKind.MACRO_CALL]
        self.assertEqual(called, ["fr_dm.accounts"])

    def test_a_parameter_shadows_a_global_of_the_same_name(self):
        rows = _read("%let tbl = wrong;\n" + PULL + "%pull(out=nonip);\n")
        (table,) = [t for t, k in rows if k == SasChunkKind.MACRO_CALL]
        self.assertEqual(table.qualified, "edw_export.&tbl")
        self.assertTrue(table.has_macro_ref)

    def test_each_call_reads_its_own_table(self):
        rows = _read(PULL + "%pull(tbl=a, out=x);\n%pull(tbl=b, out=y);\n")
        called = sorted(t.qualified for t, k in rows if k == SasChunkKind.MACRO_CALL)
        self.assertEqual(called, ["edw_export.a", "edw_export.b"])

    def test_calls_without_semicolons_are_each_a_call(self):
        src = PULL + "%pull(tbl=a, out=x)\n%pull(tbl=b, schema=edw_hist, out=y)\n"
        self.assertEqual(len(_of_kind(_chunk(src), SasChunkKind.MACRO_CALL)), 2)
        rows = _read(src)
        called = sorted(str(t) for t, k in rows if k == SasChunkKind.MACRO_CALL)
        self.assertEqual(
            called,
            [
                "oracle:edw_export.a → work.x (read via connection_to oracle in %pull)",
                "oracle:edw_hist.b → work.y (read via connection_to oracle in %pull)",
            ],
        )

    def test_a_call_binds_against_the_globals_the_call_before_it_left(self):
        src = (
            "%macro init(e);\n%global sch;\n%let sch = edw_&e;\n%mend;\n"
            "%macro pullg(tbl=);\nproc sql;\ncreate table x as select * from "
            "connection to oracle (select * from &sch..&tbl);\nquit;\n%mend;\n"
            "%init(prod)\n%pullg(tbl=t)\n"
        )
        called = [t.qualified for t, k in _read(src) if k == SasChunkKind.MACRO_CALL]
        self.assertEqual(called, ["edw_prod.t"])

    def test_a_call_before_the_definition_reads_nothing_yet(self):
        rows = _read("%pull(tbl=a, out=x);\n" + PULL)
        self.assertEqual([t for t, k in rows if k == SasChunkKind.MACRO_CALL], [])

    def test_a_macro_defined_in_another_file(self):
        macros = _chunk(PULL, source_id="macros.sas")
        job = _chunk("%pull(tbl=current_nonip, out=nonip);\n", source_id="job.sas")
        resolved = resolve_corpus_references(SasCorpus(file_results=[macros, job]))
        (table,) = resolved.file_results[1].chunks[0].metadata.db_tables
        self.assertEqual((table.qualified, table.macro), ("edw_export.current_nonip", "pull"))
        batched = MultiFileBatcher().batch(SasCorpus(file_results=[macros, job]))
        self.assertIn(
            "edw_export.current_nonip",
            [t.qualified for b in batched.batches for t in b.db_tables]
            + [t.qualified for c in batched.singletons for t in c.metadata.db_tables],
        )

    def test_global_set_by_a_called_macro(self):
        src = "%macro init;\n%global sch;\n%let sch = edw_export;\n%mend;\n%init;\n"
        self.assertEqual(
            [t.qualified for t, _ in _read(src + _query("select * from &sch..t"))],
            ["edw_export.t"],
        )

    def test_an_existing_global_updated_inside_a_macro(self):
        src = "%let sch = dev;\n%macro setp;\n%let sch = prod_s;\n%mend;\n%setp;\n"
        self.assertEqual(
            [t.qualified for t, _ in _read(src + _query("select * from &sch..t"))],
            ["prod_s.t"],
        )

    def test_a_conditional_assignment_makes_the_global_unknown(self):
        src = (
            "%let sch = dev;\n%macro sete(e);\n"
            "%if &e = P %then %let sch = prod_s;\n%mend;\n%sete(P);\n"
        )
        self.assertEqual(
            [t.qualified for t, _ in _read(src + _query("select * from &sch..t"))],
            ["&sch..t"],
        )

    def test_local_and_undeclared_assignments_stay_in_the_macro(self):
        local = "%let sch = dev;\n%macro loc;\n%local sch;\n%let sch = inner;\n%mend;\n%loc;\n"
        self.assertEqual(
            [t.qualified for t, _ in _read(local + _query("select * from &sch..t"))],
            ["dev.t"],
        )
        fresh = "%macro newv;\n%let fresh = inner;\n%mend;\n%newv;\n"
        self.assertEqual(
            [t.qualified for t, _ in _read(fresh + _query("select * from &fresh..t"))],
            ["&fresh..t"],
        )

    def test_resolution_with_calls_is_idempotent(self):
        result = _chunk(PULL + "%pull(tbl=current_nonip, out=nonip);\n")
        chunks = list(result.chunks)
        resolve_references(chunks)
        self.assertEqual([c.metadata for c in chunks], [c.metadata for c in result.chunks])


class TestConnectionNames(unittest.TestCase):
    def test_an_alias_spelled_through_a_macro_variable(self):
        src = _query("select * from s.t", "%let db = oracle;\n", alias="&db")
        (table,) = [t for t, _ in _read(src)]
        self.assertEqual((table.engine, table.connection), ("oracle", "oracle"))

    def test_an_unresolved_alias_is_still_pass_through(self):
        # The statement must be recognised even when its alias is unknown, or
        # its SQL is read as SAS and invents work.connection again.
        result = _chunk(_query("select * from s.t", alias="&db"))
        proc = _of_kind(result, SasChunkKind.PROC_STEP)[0]
        self.assertEqual(proc.metadata.input_datasets, [])
        (table,) = proc.metadata.db_tables
        self.assertIsNone(table.engine)
        self.assertEqual(table.connection, "&db")

    def test_an_engine_spelled_through_a_macro_variable(self):
        src = (
            "%let eng = oracle;\nproc sql;\nconnect to &eng as x (path=P);\n"
            "create table a as select * from connection to x (select * from s.t);\nquit;\n"
        )
        (table,) = [t for t, _ in _read(src)]
        self.assertEqual((table.engine, table.connection), ("oracle", "x"))


# ── 8. Database-engine LIBNAMEs ────────────────────────────────────────────


LIBNAME = "libname edw oracle path=EDWPRO schema=fr_dm user=&u password=&p;\n"


class TestLibnameAccess(unittest.TestCase):
    def _step(self, source: str, kind: SasChunkKind = SasChunkKind.DATA_STEP):
        return _of_kind(_chunk(source), kind)[-1]

    def test_a_read_is_registered_with_its_sas_copy(self):
        step = self._step(LIBNAME + "data work.accts; set edw.accounts; run;\n")
        (table,) = step.metadata.db_tables
        self.assertEqual(
            str(table), "oracle:fr_dm.accounts → work.accts (read via libname edw)"
        )
        self.assertEqual(table.option_map["path"], "EDWPRO")
        # SAS code does name edw.accounts, so the SAS name stays an input.
        self.assertEqual(step.metadata.input_datasets, ["edw.accounts"])

    def test_a_write(self):
        step = self._step(LIBNAME + "data edw.newtab; set work.x; run;\n")
        (table,) = step.metadata.db_tables
        self.assertEqual(
            (table.qualified, table.access), ("fr_dm.newtab", DbTableAccess.WRITE)
        )

    def test_without_a_schema_option_the_table_is_unqualified(self):
        step = self._step(
            "libname edw oracle path=EDWPRO;\ndata work.a; set edw.accounts; run;\n"
        )
        self.assertIsNone(step.metadata.db_tables[0].db_schema)

    def test_a_schema_option_spelled_through_a_macro_variable(self):
        step = self._step(
            "%let sch = fr_dm;\nlibname edw oracle path=EDWPRO schema=&sch;\n"
            "data work.a; set edw.accounts; run;\n"
        )
        self.assertEqual(step.metadata.db_tables[0].qualified, "fr_dm.accounts")

    def test_a_libref_spelled_through_a_macro_variable(self):
        step = self._step(
            "%let lib = edw;\nlibname &lib oracle path=EDWPRO schema=fr_dm;\n"
            "data work.a; set edw.accounts; run;\n"
        )
        self.assertEqual(step.metadata.db_tables[0].qualified, "fr_dm.accounts")

    def test_clear_and_a_path_rebind_end_the_binding(self):
        for rebind in ("libname edw clear;\n", "libname edw '/data/edw';\n"):
            with self.subTest(rebind=rebind):
                step = self._step(
                    LIBNAME + rebind + "data work.a; set edw.accounts; run;\n"
                )
                self.assertEqual(step.metadata.db_tables, [])

    def test_a_libname_inside_a_connection_macro_binds(self):
        step = self._step(
            "%macro connect_edw;\n" + LIBNAME + "%mend;\n%connect_edw;\n"
            "data work.a; set edw.accounts; run;\n"
        )
        self.assertEqual(step.metadata.db_tables[0].qualified, "fr_dm.accounts")

    def test_implicit_pass_through_in_proc_sql(self):
        proc = self._step(
            LIBNAME
            + "proc sql;\ncreate table work.x as select * from edw.accounts;\nquit;\n",
            SasChunkKind.PROC_STEP,
        )
        (table,) = proc.metadata.db_tables
        self.assertEqual((table.qualified, table.sas_targets), ("fr_dm.accounts", ("work.x",)))

    def test_a_libname_in_another_file(self):
        setup = _chunk(LIBNAME, source_id="setup.sas")
        job = _chunk("data work.a; set edw.accounts; run;\n", source_id="job.sas")
        # Alone, job.sas cannot know edw is an Oracle libref.
        self.assertEqual(job.chunks[0].metadata.db_tables, [])
        resolved = resolve_corpus_references(SasCorpus(file_results=[setup, job]))
        (table,) = resolved.file_results[1].chunks[0].metadata.db_tables
        self.assertEqual(table.qualified, "fr_dm.accounts")

    def test_resolution_is_idempotent(self):
        result = _chunk(LIBNAME + EXAMPLE + "data work.a; set edw.accounts; run;\n")
        chunks = list(result.chunks)
        resolve_references(chunks)
        self.assertEqual(
            [c.metadata for c in chunks], [c.metadata for c in result.chunks]
        )


# ── 9. Plumbing ────────────────────────────────────────────────────────────


class TestPlumbing(unittest.TestCase):
    def test_merge_keeps_the_parents_tables(self):
        from chunker.metadata import _merge_meta
        from chunker.models import SasChunkMetadata

        whole = _chunk(EXAMPLE).chunks[0].metadata
        slice_without_connect = SasChunkMetadata()
        merged = _merge_meta(whole, slice_without_connect)
        self.assertEqual(merged.db_tables, whole.db_tables)

    def test_split_children_inherit_the_regions_tables(self):
        filler = "".join(f"  select {i} from dual;\n" for i in range(300))
        src = (
            "proc sql;\nconnect to oracle as edw (path=P);\n" + filler
            + "create table a as select * from connection to edw (select * from s.t);\n"
            "quit;\n"
        )
        result = SasSemanticChunker(min_words=1, max_words=200).chunk_text(src)
        children = [c for c in result.chunks if c.parent_id]
        self.assertTrue(children)
        for child in children:
            self.assertEqual(
                [(t.engine, t.qualified) for t in child.metadata.db_tables],
                [("oracle", "s.t")],
            )

    def test_the_databricks_mapping_renames_the_sas_copy(self):
        # The SAS copy is a SAS dataset name like any other: renamed with the
        # rest, or the prompt would list work.nonip beside dev.staging.nonip.
        mapped = SasChunkBatcher(databricks_mapping={"work": "dev.staging"}).batch(
            _chunk(EXAMPLE)
        )
        tables = [
            t
            for item in mapped.all_ordered_items
            for c in getattr(item, "chunks", [item])
            for t in c.metadata.db_tables
        ]
        self.assertEqual([t.sas_targets for t in tables], [("dev.staging.nonip",)])

    def test_batch_result_serialises(self):
        result = _chunk(LIBNAME + EXAMPLE + "data work.a; set edw.accounts; run;\n")
        dumped = json.loads(json.dumps(SasChunkBatcher().batch(result).model_dump()))
        self.assertTrue(dumped)

    def test_hostile_input_stays_linear(self):
        src = (
            "proc sql;\ncreate table a as select * from connection to oracle ("
            + "(" * 20_000
            + ";\n"
            + "execute(" * 5_000
            + ";\nquit;\n"
        )
        start = time.perf_counter()
        _chunk(src)
        self.assertLess(time.perf_counter() - start, 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
