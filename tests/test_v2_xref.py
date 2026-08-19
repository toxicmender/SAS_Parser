"""Phase 7 XREF mapping, rewrite, adapter and architecture gates."""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from sas_migrate.adapters.xref import (
    CsvXrefSource,
    SharePointXrefSource,
    TransportCsvXrefSource,
)
from sas_migrate.application.xref import (
    ParseFailureMode,
    XrefMappings,
    XrefRewriteError,
    XrefRow,
    apply_both,
    apply_post,
    classify_rows,
    resolve_path,
    rewrite_pyspark_paths,
    rewrite_pyspark_tables,
    rewrite_source_text,
    rewrite_sql_paths,
    rewrite_sql_tables,
)
from sas_migrate.core.targets import resolve_local_target


def test_rows_are_classified_into_exact_libref_and_path_namespaces() -> None:
    mappings = classify_rows(
        (
            XrefRow(source="SALES.ORDERS", target="main.sales.orders"),
            XrefRow(source="WORK", target="main.staging", marker="dataset"),
            XrefRow(source="/SAS/Data", target="/Volumes/main/data", marker="path"),
        )
    )
    assert mappings.exact == {"sales.orders": "main.sales.orders"}
    assert mappings.by_libref == {"work": "main.staging"}
    assert mappings.by_path == {"/sas/data": "/Volumes/main/data"}
    assert mappings.dataset_mapping == {
        "work": "main.staging",
        "sales.orders": "main.sales.orders",
    }


def test_path_resolution_uses_exact_then_longest_directory_prefix() -> None:
    mapping = {
        "/data": "/Volumes/main/other",
        "/data/in": "/Volumes/main/inbound",
    }
    assert resolve_path("/DATA/in", mapping) == "/Volumes/main/inbound"
    assert resolve_path("/data/in/2026/a.csv", mapping) == (
        "/Volumes/main/inbound/2026/a.csv"
    )
    assert resolve_path("/data/inbound/a.csv", mapping) == (
        "/Volumes/main/other/inbound/a.csv"
    )
    assert resolve_path("/elsewhere/a.csv", mapping) is None


def test_sas_pre_rewriter_imports_the_v2_core_path_grammar() -> None:
    import sas_migrate.application.xref.sas_rewriter as module

    source = pathlib.Path(module.__file__).read_text("utf-8")
    assert "sas_migrate.core.sas.paths import PATH_STATEMENTS" in source
    assert "_PATH_STATEMENTS" not in source


def test_sas_pre_rewriter_preserves_statement_semantics() -> None:
    mappings = XrefMappings(by_path={"/sas": "/Volumes/main/sas"})
    source = (
        "filename raw '/sas/in/a.txt';\n"
        "filename notify email 'ops@example.com';\n"
        "proc import datafile='/sas/in/b.xlsx' out=work.b; run;\n"
        "options sasautos='/sas/mac';\n"
    )
    output, report = rewrite_source_text(source, mappings, source_id="etl.sas")
    assert "'/Volumes/main/sas/in/a.txt'" in output
    assert "email 'ops@example.com'" in output
    assert "'/Volumes/main/sas/in/b.xlsx'" in output
    assert "'/Volumes/main/sas/mac'" in output
    assert len(report.rewritten) == 3


def test_sas_pre_rewriter_reports_macro_paths_without_guessing(caplog: pytest.LogCaptureFixture) -> None:
    source = 'libname raw "&root/in";\n'
    with caplog.at_level("WARNING"):
        output, report = rewrite_source_text(
            source,
            XrefMappings(by_path={"/data/in": "/mnt/bronze"}),
            source_id="etl.sas",
        )
    assert output == source
    assert report.unresolved == ("&root/in",)
    assert "leaving them exactly as written" in caplog.text


def test_sas_pre_rewriter_is_byte_identical_when_no_path_matches() -> None:
    source = "libname raw '/somewhere/else';\r\n"
    output, report = rewrite_source_text(
        source,
        XrefMappings(by_path={"/data/in": "/mnt/bronze"}),
    )
    assert output == source
    assert not report


TABLES = {"sales.orders": "main.sales.orders"}
PATHS = {"/sasdata3": "/Volumes/main/sas"}


def test_sql_rewriter_uses_databricks_for_read_and_write(monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlglot
    from sqlglot import exp

    parsed_with: list[str | None] = []
    emitted_with: list[str | None] = []
    real_parse = sqlglot.parse
    real_sql = exp.Expression.sql

    def capture_parse(source: str, *, read: Any = None, **kwargs: Any) -> Any:
        parsed_with.append(read)
        return real_parse(source, read=read, **kwargs)

    def capture_sql(
        self: exp.Expression,
        dialect: Any = None,
        **kwargs: Any,
    ) -> str:
        emitted_with.append(dialect)
        return real_sql(self, dialect=dialect, **kwargs)

    monkeypatch.setattr(sqlglot, "parse", capture_parse)
    monkeypatch.setattr(exp.Expression, "sql", capture_sql)

    assert "main.sales.orders" in rewrite_sql_tables(
        "SELECT * FROM sales.orders", TABLES
    )
    assert parsed_with == ["databricks"]
    assert emitted_with and set(emitted_with) == {"databricks"}


def test_sql_table_and_path_rewriters_cover_databricks_positions() -> None:
    sql = "CREATE TABLE sales.orders USING csv LOCATION '/sasdata3/orders'"
    output = rewrite_sql_paths(rewrite_sql_tables(sql, TABLES), PATHS)
    assert "main.sales.orders" in output
    assert "'/Volumes/main/sas/orders'" in output


def test_pyspark_rewriters_preserve_comments_formatting_and_quote_style() -> None:
    source = (
        "# retain this comment\n"
        'df = spark.table("sales.orders")\n'
        "raw = spark.read.csv('/sasdata3/a.csv')\n"
        "label = '/sasdata3/a.csv'\n"
    )
    output = rewrite_pyspark_paths(rewrite_pyspark_tables(source, TABLES), PATHS)
    assert output == (
        "# retain this comment\n"
        'df = spark.table("main.sales.orders")\n'
        "raw = spark.read.csv('/Volumes/main/sas/a.csv')\n"
        "label = '/sasdata3/a.csv'\n"
    )


def test_pyspark_rewriter_recurses_into_spark_sql_under_databricks() -> None:
    source = 'df = spark.sql("SELECT * FROM sales.orders")\n'
    assert "main.sales.orders" in rewrite_pyspark_tables(source, TABLES)


def test_unparseable_pyspark_is_byte_identical_or_fatal() -> None:
    broken = 'df = spark.table("sales.orders"\n'
    assert rewrite_pyspark_tables(broken, TABLES) == broken
    with pytest.raises(XrefRewriteError):
        rewrite_pyspark_tables(
            broken,
            TABLES,
            on_failure=ParseFailureMode.ERROR,
        )


@pytest.mark.parametrize("target_name", ["Spark SQL", "PySpark"])
def test_application_dispatch_covers_only_registered_targets(target_name: str) -> None:
    target = resolve_local_target(target_name)
    code = (
        "SELECT * FROM sales.orders"
        if target_name == "Spark SQL"
        else 'df = spark.table("sales.orders")\n'
    )
    output = apply_post(code, target, XrefMappings(exact=TABLES))
    assert "main.sales.orders" in output


def test_both_mode_reports_names_reached_only_after_conversion() -> None:
    mappings = XrefMappings(exact=TABLES, by_path=PATHS)
    outcome = apply_both(
        'df = spark.table("sales.orders")\n',
        resolve_local_target("PySpark"),
        mappings,
    )
    assert outcome.post_changed
    assert outcome.only_post == ("sales.orders",)


def test_xref_application_has_no_scala_dispatch_or_network_imports() -> None:
    package = pathlib.Path(__file__).resolve().parents[1] / "src" / "sas_migrate" / "application" / "xref"
    source = "\n".join(path.read_text("utf-8") for path in package.rglob("*.py"))
    assert "scala" not in source.casefold()
    assert "sharepoint" not in source.casefold()
    assert "requests" not in source.casefold()


MAPPING_CSV = (
    b"sas_name,databricks_name\n"
    b"work,dev.staging\n"
    b"sales.orders,main.sales.orders\n"
)


class FakeListTransport:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def list_items(self, list_id: str) -> list[dict[str, object]]:
        self.calls.append(list_id)
        return [
            {
                "fields": {
                    "Title": "table",
                    "Application": "Billing",
                    "OriginalValue": "sales.orders",
                    "NewValue": "main.sales.orders",
                }
            },
            {
                "fields": {
                    "Title": "path",
                    "Application": "Billing",
                    "OriginalValue": "/sas/in",
                    "NewValue": "/Volumes/main/in",
                }
            },
            {
                "fields": {
                    "Application": "Other",
                    "OriginalValue": "other.table",
                    "NewValue": "ignored.table",
                }
            },
        ]


class FakeFileTransport:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def read_file(self, path: str) -> bytes:
        self.calls.append(path)
        return MAPPING_CSV


def test_sharepoint_source_does_no_io_until_invoked_and_filters_application() -> None:
    transport = FakeListTransport()
    source = SharePointXrefSource(transport, "xref-list")
    assert transport.calls == []

    mappings = source.load("billing")
    assert transport.calls == ["xref-list"]
    assert mappings.exact == {"sales.orders": "main.sales.orders"}
    assert mappings.by_path == {"/sas/in": "/Volumes/main/in"}


def test_transport_csv_source_does_no_io_until_invoked() -> None:
    transport = FakeFileTransport()
    source = TransportCsvXrefSource(transport, "maps/xref.csv")
    assert transport.calls == []

    mappings = source.load("billing")
    assert transport.calls == ["maps/xref.csv"]
    assert mappings.by_libref == {"work": "dev.staging"}
    assert mappings.exact == {"sales.orders": "main.sales.orders"}


def test_local_csv_source_is_lazy_and_strips_an_excel_bom(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "mapping.csv"
    path.write_bytes(b"\xef\xbb\xbf" + MAPPING_CSV)
    source = CsvXrefSource(path)
    assert source.load("billing").dataset_mapping == {
        "work": "dev.staging",
        "sales.orders": "main.sales.orders",
    }


def test_empty_csv_source_fails_instead_of_silently_skipping_mapping() -> None:
    source = CsvXrefSource(
        "unused.csv",
        read_bytes=lambda _: b"sas_name,databricks_name\n",
    )
    with pytest.raises(ValueError, match="zero entries"):
        source.load("billing")
