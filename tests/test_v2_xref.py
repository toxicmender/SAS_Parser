"""Phase 7 XREF mapping, rewrite, adapter and architecture gates."""

from __future__ import annotations

import pathlib

import pytest

from sas_migrate.application.xref import (
    XrefMappings,
    XrefRow,
    classify_rows,
    resolve_path,
    rewrite_source_text,
)


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
