"""chunker.sql: the one SQL grammar, in its SAS dialect.

PROC SQL and PROC FEDSQL statements are read by the same token walk that reads
a database's own SQL in pass-through (its NATIVE dialect, pinned by
tests/test_passthrough.py). These tests pin what the SAS dialect does
differently, and the role each clause gives a table.
"""

from __future__ import annotations

import pytest

from chunker import DatasetRole, SasSemanticChunker
from chunker.sql import Dialect, SqlStatement

R, W, U, D = DatasetRole.READ, DatasetRole.WRITE, DatasetRole.UPDATE, DatasetRole.DROP


def _refs(sql: str) -> list[tuple[str, DatasetRole, str]]:
    return SqlStatement(sql, Dialect.SAS).refs()


@pytest.mark.parametrize(
    ("sql", "refs"),
    [
        # FROM lists, comma joins and JOIN chains read.
        (
            "select * from a as x, lib.b y, c where x.k = y.k",
            [("a", R, "from"), ("lib.b", R, "from"), ("c", R, "from")],
        ),
        (
            "select * from a natural join b left join lib.d on b.k = d.k",
            [("a", R, "from"), ("b", R, "join"), ("lib.d", R, "join")],
        ),
        # A table's own options, then its alias.
        (
            "select * from lib.a(where=(x > 1)) as s, b(keep=k)",
            [("lib.a", R, "from"), ("b", R, "from")],
        ),
        # Subqueries and inline views are walked for their own tables.
        (
            "select * from (select id from lib.b) as s where id in (select id from c)",
            [("lib.b", R, "from"), ("c", R, "from")],
        ),
        # CREATE writes; LIKE reads.
        (
            "create table c as select * from a",
            [("c", W, "create"), ("a", R, "from")],
        ),
        ("create table c(label='copy') like lib.t", [("c", W, "create"), ("lib.t", R, "like")]),
        ("create view v as select * from a", [("v", W, "create"), ("a", R, "from")]),
        # INSERT, UPDATE, DELETE and ALTER rewrite a table in place.
        (
            "insert into lib.t(a, b) select a, b from s",
            [("lib.t", U, "insert"), ("s", R, "from")],
        ),
        ("insert into lib.t values (1, 'a')", [("lib.t", U, "insert")]),
        ("insert into lib.t set a = 1", [("lib.t", U, "insert")]),
        (
            "update lib.t set x = (select max(y) from s) where k = 1",
            [("lib.t", U, "update"), ("s", R, "from")],
        ),
        (
            "delete from lib.t where k in (select k from gone)",
            [("lib.t", U, "delete"), ("gone", R, "from")],
        ),
        ("alter table lib.t add z num", [("lib.t", U, "alter")]),
        # DROP deletes, a list at a time.
        ("drop table a, lib.b", [("a", D, "drop"), ("lib.b", D, "drop")]),
        ("drop view v", [("v", D, "drop")]),
        # Name none.
        ("create index id on lib.t(id)", []),
        ("describe table lib.t", []),
        ("select count(*) into :n trimmed from lib.t", [("lib.t", R, "from")]),
    ],
)
def test_each_clause_gives_its_table_a_role(sql, refs):
    assert _refs(sql) == refs


def test_what_a_sas_table_position_may_hold():
    # A quoted path, a name literal, a macro reference; not a macro call, not
    # SAS's own metadata.
    sql = (
        "select * from '/data/in.sas7bdat' a, 'my data'n b, &lib..&tbl c, "
        "%src(x) d, dictionary.tables t"
    )
    assert [name for name, _, _ in _refs(sql)] == [
        "'/data/in.sas7bdat'",
        "'my data'n",
        "&lib..&tbl",
    ]


def test_strings_and_functions_hide_no_tables():
    sql = "select 'from x' as s, extract(year from d) as y from a where c = 'join b'"
    assert _refs(sql) == [("a", R, "from")]


def test_a_cte_is_no_table():
    sql = "with recent as (select * from lib.log) select * from recent"
    assert _refs(sql) == [("lib.log", R, "from")]


def test_the_native_dialect_reads_a_function_where_sas_reads_options():
    sql = "select * from t(where=(x > 1))"
    assert SqlStatement(sql).reads() == []  # TABLE-function shaped
    assert _refs(sql) == [("t", R, "from")]


def test_proc_sql_and_fedsql_read_through_the_grammar():
    src = (
        "proc sql;\n  create table c as select * from a, lib.b;\n"
        "  insert into lib.log select * from c;\n  drop table c;\nquit;\n"
        "proc fedsql;\n  create table f as select * from c2;\nquit;\n"
    )
    chunks = SasSemanticChunker(min_words=1, max_words=100_000).chunk_text(src).chunks
    sql, fedsql = (c.metadata for c in chunks)
    assert sql.input_datasets == ["work.a", "lib.b", "lib.log", "work.c"]
    assert sql.output_datasets == ["work.c", "lib.log"]
    assert sql.dropped_datasets == ["work.c"]
    assert (fedsql.input_datasets, fedsql.output_datasets) == (["work.c2"], ["work.f"])
