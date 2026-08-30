"""
Tests for the xref/ package — sourcing rows from the SharePoint list,
classifying them, and applying the substitution before or after conversion.

The post-conversion rewriter's one hard rule is checked explicitly: input that
does not parse comes back byte-identical. A rewriter that corrupts generated
code is worse than one that no-ops.
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
import sqlglot
from sqlglot.expressions import Expression

import app_config
from app_config.sharepoint import SharePointConfig, SharePointError
from xref import apply as xref_apply
from xref import pre, rewrite, sourcing


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(app_config.ENV_VAR, str(cfg))
    monkeypatch.delenv("XREF_APPLY", raising=False)
    app_config.clear_cache()
    yield cfg
    app_config.clear_cache()


def _set(cfg_path, mapping) -> None:
    cfg_path.write_text(json.dumps(mapping), encoding="utf-8")
    app_config.clear_cache()


class _FakeTransport:
    def __init__(self, rows):
        self._rows = list(rows)

    def list_items(self, list_id, **_options):
        return list(self._rows)


def _row(title, application, source, target, item_id="1"):
    return {
        "id": item_id,
        "fields": {
            "Title": title,
            "Application": application,
            "OriginalValue": source,
            "NewValue": target,
        },
    }


_CONFIG = SharePointConfig(site_id="SITE", list_id_xref="L-xref")


# ---------------------------------------------------------------------------
# sourcing
# ---------------------------------------------------------------------------


def test_rows_are_filtered_by_application():
    transport = _FakeTransport(
        [
            _row(None, "MyApp", "sales.orders", "cat.sales.orders"),
            _row(None, "OtherApp", "hr.staff", "cat.hr.staff"),
        ]
    )
    result = sourcing.mappings("MyApp", client=transport, config=_CONFIG)

    assert result.exact == {"sales.orders": "cat.sales.orders"}


def test_application_match_is_case_insensitive():
    transport = _FakeTransport([_row(None, "MYAPP", "a.b", "c.a.b")])
    assert sourcing.mappings("myapp", client=transport, config=_CONFIG).exact


def test_dotted_keys_are_exact_and_bare_keys_are_librefs():
    # The two shapes chunker.batcher._split_databricks_mapping already
    # distinguishes, so nothing has to be translated for it.
    transport = _FakeTransport(
        [
            _row(None, "MyApp", "sales.orders", "cat.sales.orders"),
            _row(None, "MyApp", "sales", "cat.sales"),
        ]
    )
    result = sourcing.mappings("MyApp", client=transport, config=_CONFIG)

    assert result.exact == {"sales.orders": "cat.sales.orders"}
    assert result.by_libref == {"sales": "cat.sales"}
    assert result.by_path == {}


def test_half_filled_rows_are_skipped():
    transport = _FakeTransport(
        [
            _row(None, "MyApp", "sales.orders", ""),
            _row(None, "MyApp", "", "cat.x.y"),
            _row(None, "MyApp", "a.b", "cat.a.b"),
        ]
    )
    result = sourcing.mappings("MyApp", client=transport, config=_CONFIG)

    assert result.exact == {"a.b": "cat.a.b"}


@pytest.mark.parametrize("title", [None, "", "   "])
def test_an_unmarked_row_is_a_table_mapping(title):
    # Every existing row keeps working: no backfill is needed for the marker.
    transport = _FakeTransport([_row(title, "MyApp", "a.b", "cat.a.b")])
    result = sourcing.mappings("MyApp", client=transport, config=_CONFIG)

    assert result.exact == {"a.b": "cat.a.b"}
    assert result.by_path == {}


@pytest.mark.parametrize("marker", ["path", "PATH", " Path ", "physical_path", "file"])
def test_a_path_marker_routes_the_row_to_by_path(marker):
    transport = _FakeTransport(
        [_row(marker, "MyApp", "/data/in.csv", "/mnt/lake/in.csv")]
    )
    result = sourcing.mappings("MyApp", client=transport, config=_CONFIG)

    assert result.by_path == {"/data/in.csv": "/mnt/lake/in.csv"}
    assert result.exact == {}


def test_an_unrecognised_marker_warns_and_is_treated_as_a_table(caplog):
    # A mistyped row must be visible now, not silently wrong later.
    import logging

    transport = _FakeTransport([_row("pathh", "MyApp", "a.b", "cat.a.b")])
    with caplog.at_level(logging.WARNING, logger="xref.sourcing"):
        result = sourcing.mappings("MyApp", client=transport, config=_CONFIG)

    assert result.exact == {"a.b": "cat.a.b"}
    assert "unrecognised Title marker" in caplog.text


def test_dataset_mapping_flattens_both_table_slots():
    result = sourcing.XrefMappings(
        exact={"a.b": "c.a.b"}, by_libref={"a": "c.a"}, by_path={"/x": "/y"}
    )
    # Only the two shapes replace_dataset_names understands; by_path is not
    # yet consumed by anything.
    assert result.dataset_mapping == {"a": "c.a", "a.b": "c.a.b"}


def test_missing_xref_list_names_the_config_key():
    with pytest.raises(SharePointError, match="sharepoint.list_id_xref"):
        sourcing.mappings(
            "MyApp", client=_FakeTransport([]), config=SharePointConfig()
        )


# ---------------------------------------------------------------------------
# apply mode resolution
# ---------------------------------------------------------------------------


def test_mode_defaults_to_pre(_isolated):
    assert xref_apply.configured_mode() == "pre"


def test_mode_from_config(_isolated):
    _set(_isolated, {"xref": {"apply": "post"}})
    assert xref_apply.configured_mode() == "post"


def test_mode_env_beats_config(monkeypatch, _isolated):
    _set(_isolated, {"xref": {"apply": "post"}})
    monkeypatch.setenv("XREF_APPLY", "both")
    assert xref_apply.configured_mode() == "both"


def test_unknown_mode_degrades_to_the_default(_isolated, caplog):
    import logging

    _set(_isolated, {"xref": {"apply": "sideways"}})
    with caplog.at_level(logging.WARNING, logger="xref.apply"):
        assert xref_apply.configured_mode() == "pre"
    assert "sideways" in caplog.text


# ---------------------------------------------------------------------------
# apply_pre — delegated to chunker.batcher, which is NOT modified
# ---------------------------------------------------------------------------


def _batch_result(dataset: str):
    from chunker.models import (
        SasBatch,
        SasBatchResult,
        SasChunk,
        SasChunkKind,
        SasChunkMetadata,
    )

    chunk = SasChunk(
        chunk_id="f1-chunk-0001",
        source_id="etl.sas",
        text=f"data work.out; set {dataset}; run;",
        kind=SasChunkKind.DATA_STEP,
        title="Step",
        start_line=1,
        end_line=1,
        start_char=0,
        end_char=10,
        metadata=SasChunkMetadata(input_datasets=[dataset]),
    )
    batch = SasBatch(batch_id="b-001", chunks=[chunk], source_files=["etl.sas"])
    return SasBatchResult(batches=[batch], singletons=[])


def test_apply_pre_rewrites_dataset_metadata():
    result = _batch_result("sales.orders")
    rewritten = xref_apply.apply_pre(result, {"sales.orders": "cat.sales.orders"})

    assert rewritten.batches[0].chunks[0].metadata.input_datasets == [
        "cat.sales.orders"
    ]


def test_apply_pre_with_no_mapping_is_a_no_op():
    result = _batch_result("sales.orders")
    assert xref_apply.apply_pre(result, {}) is result


# ---------------------------------------------------------------------------
# apply_post / the rewriters
# ---------------------------------------------------------------------------


_MAPPING = {"sales.orders": "cat.sales.orders"}
# apply_post and friends take the whole XrefMappings, not a flat dict: the two
# halves address different namespaces (dataset names, physical paths).
_MAPPINGS = sourcing.XrefMappings(exact=dict(_MAPPING))


def test_sql_table_reference_is_rewritten():
    out = rewrite.rewrite_sql("SELECT * FROM sales.orders", _MAPPING)
    assert "cat.sales.orders" in out
    assert "FROM sales.orders" not in out


def test_sql_rewriter_parses_and_renders_with_registered_databricks_dialect(
    monkeypatch,
):
    parse_dialects: list[str | None] = []
    render_dialects: list[str | None] = []
    real_parse = sqlglot.parse
    real_sql = Expression.sql

    def parse(source: str, *, read: str | None = None):
        parse_dialects.append(read)
        return real_parse(source, read=read)

    def render(self, *args, dialect: str | None = None, **kwargs):
        render_dialects.append(dialect)
        return real_sql(self, *args, dialect=dialect, **kwargs)

    monkeypatch.setattr(sqlglot, "parse", parse)
    monkeypatch.setattr(Expression, "sql", render)

    assert rewrite.default_dialect() == "databricks"
    out = rewrite.rewrite_sql("SELECT * FROM sales.orders", _MAPPING)
    assert "cat.sales.orders" in out
    assert parse_dialects == ["databricks"]
    assert render_dialects == ["databricks"]


def test_sql_leaves_unmapped_tables_alone():
    out = rewrite.rewrite_sql("SELECT * FROM hr.staff", _MAPPING)
    assert out == "SELECT * FROM hr.staff"


def test_unparseable_sql_is_returned_untouched(caplog):
    import logging

    broken = "SELECT * FROM ((( sales.orders WHERE"
    with caplog.at_level(logging.WARNING, logger="xref.rewrite"):
        assert rewrite.rewrite_sql(broken, _MAPPING) == broken


def test_unparseable_sql_can_be_made_fatal():
    broken = "SELECT * FROM ((( sales.orders WHERE"
    with pytest.raises(rewrite.XrefRewriteError):
        rewrite.rewrite_sql(broken, _MAPPING, on_failure="error")


def test_pyspark_table_calls_are_rewritten():
    source = (
        "# read the orders\n"
        'df = spark.table("sales.orders")\n'
        'df.write.saveAsTable("sales.orders")\n'
    )
    out = rewrite.rewrite_python(source, _MAPPING)

    assert out.count("cat.sales.orders") == 2
    # Source-span substitution, so the comment and the layout survive.
    assert out.startswith("# read the orders\n")


def test_pyspark_table_rewrite_keeps_the_original_quoting():
    """A table rewrite must not churn quoting the way repr() did.

    The path half preserves style, and a file whose only real change is a table
    name should not also show every double-quoted literal turned single.
    """
    source = 'df = spark.table("sales.orders")\n'
    assert rewrite.rewrite_python(source, _MAPPING) == (
        'df = spark.table("cat.sales.orders")\n'
    )


def test_pyspark_table_rewrite_falls_back_when_style_cannot_be_reproduced():
    # A target embedding the literal's own quote cannot be re-emitted in that
    # style, so repr() takes over rather than producing broken source.
    mapping = {"sales.orders": 'cat."sales".orders'}
    out = rewrite.rewrite_python('df = spark.table("sales.orders")\n', mapping)
    assert 'cat."sales".orders' in out
    assert ast.parse(out)  # still valid Python, which is the point


def test_pyspark_leaves_unrelated_strings_alone():
    source = 'label = "sales.orders"\nprint("sales.orders")\n'
    assert rewrite.rewrite_python(source, _MAPPING) == source


def test_pyspark_recurses_into_spark_sql():
    source = 'df = spark.sql("SELECT * FROM sales.orders")\n'
    out = rewrite.rewrite_python(source, _MAPPING)

    assert "cat.sales.orders" in out
    assert out.startswith("df = spark.sql(")


def test_unparseable_python_is_returned_untouched(caplog):
    import logging

    broken = 'df = spark.table("sales.orders"\n'  # unbalanced paren
    with caplog.at_level(logging.WARNING, logger="xref.rewrite"):
        assert rewrite.rewrite_python(broken, _MAPPING) == broken
    assert "leaving it exactly as the model wrote it" in caplog.text


def test_unparseable_python_can_be_made_fatal():
    broken = 'df = spark.table("sales.orders"\n'
    with pytest.raises(rewrite.XrefRewriteError):
        rewrite.rewrite_python(broken, _MAPPING, on_failure="error")


# ---------------------------------------------------------------------------
# post — the path half
#
# The counterpart of pre.py's statement rewriting, on the other side of the
# conversion. Everything here is source-span substitution: what is NOT rewritten
# matters as much as what is.
# ---------------------------------------------------------------------------


_PATHS = {"/sasdata3": "/Volumes/main/sas", "/data/in": "/Volumes/main/inbound"}


@pytest.mark.parametrize(
    "call, expected",
    [
        ('spark.read.csv("/sasdata3/a.csv")', '"/Volumes/main/sas/a.csv"'),
        ('spark.read.parquet("/sasdata3/p")', '"/Volumes/main/sas/p"'),
        ('spark.read.format("json").load("/data/in/j")', '"/Volumes/main/inbound/j"'),
        ('df.write.save("/sasdata3/out")', '"/Volumes/main/sas/out"'),
        ('dbutils.fs.ls("/sasdata3/dir")', '"/Volumes/main/sas/dir"'),
        ('dbutils.fs.mkdirs("/data/in/new")', '"/Volumes/main/inbound/new"'),
        ('open("/sasdata3/notes.txt")', '"/Volumes/main/sas/notes.txt"'),
        # The one form whose path is the second argument.
        ('df.write.option("path", "/sasdata3/o")', '"/Volumes/main/sas/o"'),
    ],
)
def test_pyspark_path_calls_are_rewritten(call, expected):
    assert expected in rewrite.rewrite_python_paths(call + "\n", _PATHS)


def test_pyspark_paths_change_only_the_mapped_spans():
    """The whole point of span substitution — assert on the full text."""
    source = (
        "# stage from /sasdata3, which this comment names\n"
        'df = spark.read.csv("/sasdata3/a.csv", header=True)\n'
        "label = '/sasdata3/a.csv'\n"  # a bare string, not a path position
        "n = df.head(5)\n"
    )
    assert rewrite.rewrite_python_paths(source, _PATHS) == (
        "# stage from /sasdata3, which this comment names\n"
        'df = spark.read.csv("/Volumes/main/sas/a.csv", header=True)\n'
        "label = '/sasdata3/a.csv'\n"
        "n = df.head(5)\n"
    )


def test_pyspark_paths_preserve_the_original_quoting():
    source = "df = spark.read.csv('/sasdata3/a.csv')\n"
    assert "'/Volumes/main/sas/a.csv'" in rewrite.rewrite_python_paths(source, _PATHS)


def test_pyspark_paths_leave_unmapped_values_alone():
    source = 'df = spark.read.csv("/elsewhere/a.csv")\n'
    assert rewrite.rewrite_python_paths(source, _PATHS) == source


def test_unparseable_pyspark_paths_come_back_byte_identical(caplog):
    import logging

    broken = 'df = spark.read.csv("/sasdata3/a.csv"\n'
    with caplog.at_level(logging.WARNING, logger="xref.rewrite"):
        assert rewrite.rewrite_python_paths(broken, _PATHS) == broken


def test_unparseable_pyspark_paths_can_be_made_fatal():
    broken = 'df = spark.read.csv("/sasdata3/a.csv"\n'
    with pytest.raises(rewrite.XrefRewriteError):
        rewrite.rewrite_python_paths(broken, _PATHS, on_failure="error")


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("CREATE TABLE t (a INT) LOCATION '/sasdata3/t'", "'/Volumes/main/sas/t'"),
        ("COPY INTO t FROM '/data/in/land'", "'/Volumes/main/inbound/land'"),
        ("SELECT * FROM read_files('/sasdata3/s')", "'/Volumes/main/sas/s'"),
        ("CREATE TABLE u USING csv OPTIONS (path '/data/in/u')", "'/Volumes/main/inbound/u'"),
    ],
)
def test_sql_path_positions_are_rewritten(sql, expected):
    assert expected in rewrite.rewrite_sql_paths(sql, _PATHS)


def test_sql_paths_leave_table_names_alone():
    # A table in FROM position is an identifier, not a quoted literal.
    sql = "SELECT * FROM sales.orders"
    assert rewrite.rewrite_sql_paths(sql, _PATHS) == sql


def test_sql_paths_handle_the_quoted_option_key_form():
    # OPTIONS ('path' = '...'). The value is rewritten and the key is not —
    # the failure this guards against is the pattern latching onto the key and
    # rewriting the ` = ` between them as though it were the value.
    sql = "CREATE TABLE u USING csv OPTIONS ('path' = '/data/in/u')"
    assert rewrite.rewrite_sql_paths(sql, _PATHS) == (
        "CREATE TABLE u USING csv OPTIONS ('path' = '/Volumes/main/inbound/u')"
    )


def test_sql_paths_need_no_sqlglot():
    """Unlike rewrite_sql, the path half parses nothing, so it never degrades."""
    assert "/Volumes/main/sas/t" in rewrite.rewrite_sql_paths(
        "CREATE TABLE t LOCATION '/sasdata3/t'", _PATHS
    )


def test_pre_and_post_resolve_a_path_to_the_same_target():
    """One resolver, both ends (Architecture.md invariant 12).

    The longest-prefix case is the one that would drift: '/data' and '/data/in'
    are both keys, and a second copy of the resolution rule picking the shorter
    would rewrite the same location two ways depending on which half saw it.
    """
    mapping = {"/data": "/Volumes/main/other", "/data/in": "/Volumes/main/inbound"}
    sas = "libname raw '/data/in/cust';\n"
    generated = 'df = spark.read.csv("/data/in/cust")\n'

    pre_out, _ = pre.rewrite_source_text(sas, _paths(**mapping))
    post_out = rewrite.rewrite_python_paths(generated, mapping)

    assert "/Volumes/main/inbound/cust" in pre_out
    assert "/Volumes/main/inbound/cust" in post_out


def test_apply_post_rewrites_tables_and_paths_together():
    mappings = sourcing.XrefMappings(exact=dict(_MAPPING), by_path=dict(_PATHS))
    source = (
        'df = spark.table("sales.orders")\n'
        'raw = spark.read.csv("/sasdata3/a.csv")\n'
    )
    out = xref_apply.apply_post(source, "PySpark", mappings)

    assert "cat.sales.orders" in out
    assert "/Volumes/main/sas/a.csv" in out


def test_apply_both_reports_a_path_only_post_reached():
    mappings = sourcing.XrefMappings(by_path=dict(_PATHS))
    generated = 'df = spark.read.csv("/sasdata3/a.csv")\n'
    outcome = xref_apply.apply_both(generated, "PySpark", mappings)

    assert outcome.post_changed is True
    assert outcome.only_post == ["/sasdata3"]


def test_apply_post_dispatches_on_language():
    source = 'df = spark.table("sales.orders")\n'
    assert "cat.sales.orders" in xref_apply.apply_post(source, "PySpark", _MAPPINGS)


def test_apply_post_rejects_an_unsupported_language():
    source = 'val df = spark.table("sales.orders")'
    with pytest.raises(ValueError, match="unknown output language"):
        xref_apply.apply_post(source, "Spark Scala", _MAPPINGS)


# ---------------------------------------------------------------------------
# "both" — the verification mode
# ---------------------------------------------------------------------------


def test_both_reports_what_only_post_reached():
    # The pre pass left this reference alone, so post finding it is the
    # evidence that a dataset name escaped the metadata extraction.
    generated = 'df = spark.table("sales.orders")\n'
    outcome = xref_apply.apply_both(generated, "PySpark", _MAPPINGS)

    assert outcome.post_changed is True
    assert outcome.only_post == ["sales.orders"]


def test_both_reports_nothing_when_pre_already_did_it():
    already = 'df = spark.table("cat.sales.orders")\n'
    outcome = xref_apply.apply_both(
        already, "PySpark", _MAPPINGS, pre_code=already
    )

    assert outcome.only_post == []


def test_apply_dispatches_and_validates_its_inputs():
    result = _batch_result("sales.orders")
    assert xref_apply.apply("pre", result=result, mappings=_MAPPINGS) is not result

    with pytest.raises(ValueError, match="needs a batch result"):
        xref_apply.apply("pre", mappings=_MAPPINGS)
    with pytest.raises(ValueError, match="needs code and language"):
        xref_apply.apply("post", mappings=_MAPPINGS)
    with pytest.raises(ValueError, match="unknown xref apply mode"):
        xref_apply.apply("sideways", code="x", language="PySpark", mappings=_MAPPINGS)


def test_apply_both_through_the_dispatcher_runs_the_pre_pass_too():
    result = _batch_result("sales.orders")
    outcome = xref_apply.apply(
        "both",
        result=result,
        code='df = spark.table("sales.orders")\n',
        language="PySpark",
        mappings=_MAPPINGS,
    )

    assert outcome.pre_applied is True
    assert outcome.result is not None
    assert outcome.result.batches[0].chunks[0].metadata.input_datasets == [
        "cat.sales.orders"
    ]
    assert "cat.sales.orders" in outcome.code


# ---------------------------------------------------------------------------
# The CSV backend — sourcing.load_databricks_mapping_sharepoint
#
# It lives in xref/ rather than chunker/batcher.py because reading it is I/O
# against SharePoint and chunker stays network-free; the parser it delegates
# to is pure and stays in chunker.
# ---------------------------------------------------------------------------


class _FakeFileTransport:
    """Duck-typed stand-in for the SharePoint client: read_file only."""

    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.read_paths: list[str] = []

    def read_file(self, path: str) -> bytes:
        self.read_paths.append(path)
        return self.files[path]


_MAPPING_CSV = (
    b"sas_name,databricks_name\n"
    b"work,dev.staging\n"
    b"mylib,prod.sales\n"
)


def _patch_sharepoint(monkeypatch, files: dict[str, bytes]) -> _FakeFileTransport:
    import app_config.sharepoint as sp_mod

    fake = _FakeFileTransport(files)
    monkeypatch.setattr(sp_mod, "get_sharepoint_client", lambda: fake)
    return fake


def test_databricks_mapping_loaded_from_sharepoint_csv(monkeypatch):
    fake = _patch_sharepoint(monkeypatch, {"maps/sas_to_databricks.csv": _MAPPING_CSV})

    mapping = sourcing.load_databricks_mapping_sharepoint(
        "maps/sas_to_databricks.csv"
    )

    assert fake.read_paths == ["maps/sas_to_databricks.csv"]
    assert mapping == {"work": "dev.staging", "mylib": "prod.sales"}


def test_explicit_databricks_mapping_overrides_the_csv(monkeypatch):
    # The merge is the caller's one-liner: loaded CSV under an explicit dict.
    _patch_sharepoint(monkeypatch, {"m.csv": _MAPPING_CSV})

    mapping = {
        **sourcing.load_databricks_mapping_sharepoint("m.csv"),
        "work": "override.schema",
    }

    assert mapping == {
        "work": "override.schema",  # the explicit entry wins per key
        "mylib": "prod.sales",  # CSV-only entries survive the merge
    }


def test_empty_sharepoint_mapping_csv_raises(monkeypatch):
    # Asking for renaming that cannot happen must stop the run rather than
    # silently produce SAS-named output.
    _patch_sharepoint(monkeypatch, {"m.csv": b"sas_name,databricks_name\n"})

    with pytest.raises(ValueError, match="zero entries"):
        sourcing.load_databricks_mapping_sharepoint("m.csv")


def test_chunker_stays_network_free():
    """chunker must not reach SharePoint; xref owns that (old-spec D3)."""
    import chunker.batcher as batcher

    assert not hasattr(batcher, "load_databricks_mapping_sharepoint")
    source = pathlib.Path(batcher.__file__).read_text(encoding="utf-8")
    assert "app_config.sharepoint" not in source


# ---------------------------------------------------------------------------
# xref.pre — physical paths in LIBNAME / INFILE / %INCLUDE
#
# The half of "pre" that replace_dataset_names cannot reach: _map_ds
# early-returns on quoted paths, so nothing else rewrites them.
# ---------------------------------------------------------------------------


def _paths(**by_path) -> sourcing.XrefMappings:
    return sourcing.XrefMappings(by_path=dict(by_path))


def test_pre_uses_the_chunkers_path_grammar():
    """One owner of "where a path appears in SAS" (Architecture.md invariant 12).

    xref.pre rewrites those statements and the chunker inventories them. A
    second copy of the grammar growing here would mean the two disagreeing
    about what a path is — silently, since each would still pass its own tests.
    """
    from chunker.paths import PATH_STATEMENTS

    source = pathlib.Path(pre.__file__).read_text(encoding="utf-8")
    assert "chunker.paths" in source
    # No local pattern table: the statement regexes must come from the chunker.
    assert "_PATH_STATEMENTS" not in source
    assert any(spec.statement == "libname" for spec in PATH_STATEMENTS)


def test_pre_rewrites_the_wider_statement_set():
    """The shared grammar reaches statements the old local table did not."""
    src = (
        "filename raw '/sas/in/a.txt';\n"
        "proc import datafile='/sas/in/b.xlsx' out=work.b; run;\n"
        "proc export data=work.b outfile='/sas/out/c.csv'; run;\n"
        "ods html file='/sas/rep/d.html';\n"
        "options sasautos='/sas/mac';\n"
    )
    out, stats = pre.rewrite_source_text(src, _paths(**{"/sas": "/Volumes/main/sas"}))

    assert "/Volumes/main/sas/in/a.txt" in out
    assert "/Volumes/main/sas/in/b.xlsx" in out
    assert "/Volumes/main/sas/out/c.csv" in out
    assert "/Volumes/main/sas/rep/d.html" in out
    assert "/Volumes/main/sas/mac" in out
    assert len(stats.rewritten) == 5


def test_pre_leaves_device_filenames_alone():
    """A device form's quoted argument is a command or an address, not a path.

    Rewriting it would corrupt a working statement — ``pipe 'ls -l /sas'`` is a
    shell command that happens to mention a directory.
    """
    src = (
        "filename lister pipe 'ls -l /sas/in';\n"
        "filename notify email 'ops@example.com';\n"
        "filename feed ftp '/sas/in/rates.dat';\n"
        "filename real '/sas/in/real.txt';\n"
    )
    out, stats = pre.rewrite_source_text(src, _paths(**{"/sas": "/Volumes/main/sas"}))

    assert "pipe 'ls -l /sas/in'" in out
    assert "email 'ops@example.com'" in out
    assert "ftp '/sas/in/rates.dat'" in out
    # Only the bare filesystem FILENAME is remapped.
    assert "'/Volumes/main/sas/in/real.txt'" in out
    assert list(stats.rewritten) == ["/sas/in/real.txt"]


def test_pre_is_a_no_op_without_path_mappings():
    src = "libname raw '/data/in';\n"
    out, stats = pre.rewrite_source_text(src, sourcing.XrefMappings())

    assert out == src
    assert not stats
    assert stats.changed is False


def test_pre_rewrites_a_libname_path():
    src = "libname raw '/data/in';\ndata x; set raw.a; run;\n"
    out, stats = pre.rewrite_source_text(src, _paths(**{"/data/in": "/mnt/bronze"}))

    assert "libname raw '/mnt/bronze';" in out
    assert stats.rewritten == {"/data/in": "/mnt/bronze"}
    # Only the path moved; the libref and everything using it are untouched.
    assert "set raw.a;" in out


def test_pre_rewrites_libname_with_an_engine_and_double_quotes():
    src = 'libname raw meta "/data/in";\n'
    out, _ = pre.rewrite_source_text(src, _paths(**{"/data/in": "/mnt/bronze"}))

    assert out == 'libname raw meta "/mnt/bronze";\n'


def test_pre_rewrites_infile_file_and_include():
    src = (
        "%include '/code/common.sas';\n"
        "data x; infile '/data/in/c.csv'; run;\n"
        "data _null_; file '/data/out/r.txt'; run;\n"
    )
    out, stats = pre.rewrite_source_text(
        src,
        _paths(
            **{
                "/code/common.sas": "/repo/common.sas",
                "/data/in/c.csv": "/mnt/bronze/c.csv",
                "/data/out/r.txt": "/mnt/silver/r.txt",
            }
        ),
    )

    assert "'/repo/common.sas'" in out
    assert "'/mnt/bronze/c.csv'" in out
    assert "'/mnt/silver/r.txt'" in out
    assert len(stats.rewritten) == 3


def test_pre_applies_the_longest_matching_key_first():
    # The reference sorts keys longest-first to avoid partial overlaps; the
    # specific mapping must win over the prefix one.
    src = "libname a '/data/in/sub';\nlibname b '/data/in';\n"
    out, _ = pre.rewrite_source_text(
        src, _paths(**{"/data/in": "/mnt/bronze", "/data/in/sub": "/mnt/gold"})
    )

    assert "libname a '/mnt/gold';" in out
    assert "libname b '/mnt/bronze';" in out


def test_pre_treats_a_key_as_a_directory_prefix():
    src = "infile '/data/in/2026/c.csv';\n"
    out, stats = pre.rewrite_source_text(src, _paths(**{"/data/in": "/mnt/bronze"}))

    assert "'/mnt/bronze/2026/c.csv'" in out
    assert stats.rewritten == {"/data/in/2026/c.csv": "/mnt/bronze/2026/c.csv"}


def test_pre_prefix_match_stops_at_a_separator():
    # /data/in must not match /data/inbound.
    src = "infile '/data/inbound/c.csv';\n"
    out, stats = pre.rewrite_source_text(src, _paths(**{"/data/in": "/mnt/bronze"}))

    assert out == "infile '/data/inbound/c.csv';\n"
    assert not stats.rewritten


def test_pre_reports_unresolved_macro_paths_instead_of_guessing(caplog):
    src = "libname raw \"&root/in\";\n"
    with caplog.at_level("WARNING"):
        out, stats = pre.rewrite_source_text(
            src, _paths(**{"/data/in": "/mnt/bronze"}), source_id="etl.sas"
        )

    assert out == src  # left exactly as written
    assert stats.unresolved == ["&root/in"]
    assert not stats.rewritten
    assert "unresolved macro reference" in caplog.text


def test_pre_leaves_an_unmapped_path_alone():
    src = "libname raw '/somewhere/else';\n"
    out, stats = pre.rewrite_source_text(src, _paths(**{"/data/in": "/mnt/bronze"}))

    assert out == src
    assert not stats.rewritten


def test_pre_does_not_touch_a_path_outside_a_path_statement():
    # Matching by statement, not by blind sweep: a bare string that happens to
    # look like a mapped path is not a path reference.
    src = "%let note = '/data/in is the old location';\n"
    out, stats = pre.rewrite_source_text(src, _paths(**{"/data/in": "/mnt/bronze"}))

    assert out == src
    assert not stats.rewritten
