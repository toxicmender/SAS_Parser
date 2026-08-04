"""
Tests for the SharePoint-driven complexity flow — row projection, source
loading, the staged-then-uploaded delivery, and the argument validation that
guards it.

Everything runs against a fake transport. The local path must not regress:
tests/test_complexity.py is the guard for that and passes unchanged.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

import app_config
from app_config.sharepoint import SharePointConfig, SharePointError
from complexity import __main__ as cli
from complexity import sharepoint as sp

BASE = "Kit/Applications"

_SAS = """\
data work.staging;
    set sales.orders;
    total = qty * price;
run;

proc sql;
    create table work.summary as
    select region, sum(total) as revenue
    from work.staging
    group by region;
quit;
"""


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(app_config.ENV_VAR, str(cfg))
    app_config.clear_cache()
    yield cfg
    app_config.clear_cache()


def _config(**overrides) -> SharePointConfig:
    values: dict[str, Any] = {
        "site_id": "SITE",
        "drive_id": "DRV",
        "file_server_base_path": BASE,
        "list_id_sas_complexity": "L-cx",
    }
    values.update(overrides)
    return SharePointConfig(**values)


class _FakeTransport:
    """The SharePointClient surface the complexity flow uses."""

    def __init__(self, rows=None, files=None, texts=None):
        self.rows = list(rows or [])
        self.files: dict[str, list[str]] = dict(files or {})
        self.texts: dict[str, str] = dict(texts or {})
        self.created: list[str] = []
        self.uploaded: list[tuple[str, str, Any]] = []

    def list_items(self, list_id, **_options):
        return list(self.rows)

    def get_list_item(self, list_id, item_id):
        for row in self.rows:
            if str(row["id"]) == str(item_id):
                return row
        raise SharePointError(f"could not read item {item_id!r}")

    def list_files(self, folder, extensions=None):
        wanted = (
            {e.strip().lstrip(".").lower() for e in extensions}
            if extensions is not None
            else None
        )
        out = []
        for name in self.files.get(folder, []):
            suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if wanted is not None and suffix not in wanted:
                continue
            out.append({"name": name, "is_folder": False, "path": f"{folder}/{name}"})
        return out

    def download_file_as_text(self, path, *, encoding="utf-8"):
        return self.texts[path]

    def create_folder(self, path):
        self.created.append(path)
        return {"name": path.rsplit("/", 1)[-1]}

    def upload_file(self, folder, name, content):
        self.uploaded.append((folder, name, content))
        return {"name": name}


def _row(item_id="42", application="MyApp", language="SparkSQL", llm=None):
    return {
        "id": item_id,
        "fields": {
            "Application": application,
            "Output_Language": language,
            "Preferred_LLM": llm,
        },
    }


def _corpus_transport(*, rows=None, application="MyApp", names=("etl.sas",)):
    folder = f"{BASE}/{application}/scripts_original"
    return _FakeTransport(
        rows=rows if rows is not None else [_row(application=application)],
        files={folder: list(names)},
        texts={f"{folder}/{name}": _SAS for name in names},
    )


# ---------------------------------------------------------------------------
# Row projection
# ---------------------------------------------------------------------------


def test_row_projection():
    request = sp.format_complexity_item_params(
        _row(item_id="42", language="PySpark", llm="anthropic: claude-sonnet-4-5")
    )
    assert request.item_id == "42"
    assert request.application == "MyApp"
    assert request.output_language == "PySpark"
    assert request.preferred_llm == "anthropic: claude-sonnet-4-5"


def test_a_row_without_an_application_is_an_error():
    with pytest.raises(SharePointError, match="Application"):
        sp.format_complexity_item_params({"id": "1", "fields": {}})


def test_requests_are_not_filtered_by_status():
    # Every row is a valid target on every invocation: there is no Status
    # column and no pending concept.
    transport = _FakeTransport(rows=[_row("1"), _row("2", application="Other")])
    assert len(sp.requests(client=transport, config=_config())) == 2


def test_requests_can_be_narrowed_to_one_application():
    transport = _FakeTransport(rows=[_row("1"), _row("2", application="Other")])
    found = sp.requests(application="other", client=transport, config=_config())
    assert [row.item_id for row in found] == ["2"]


def test_requests_skips_a_malformed_row(caplog):
    import logging

    transport = _FakeTransport(rows=[_row("1"), {"id": "2", "fields": {}}])
    with caplog.at_level(logging.WARNING, logger="complexity.sharepoint"):
        assert len(sp.requests(client=transport, config=_config())) == 1
    assert "skipping a malformed row" in caplog.text


def test_request_by_item_id():
    transport = _FakeTransport(rows=[_row("1"), _row("42", application="Other")])
    assert sp.request("42", client=transport, config=_config()).application == "Other"


def test_report_folder_convention():
    assert sp.report_folder("MyApp", "sparksql", "20260804T101500Z", config=_config()) == (
        f"{BASE}/MyApp/complexity/sparksql/20260804T101500Z"
    )


def test_missing_complexity_list_names_the_config_key():
    with pytest.raises(SharePointError, match="sharepoint.list_id_sas_complexity"):
        sp.requests(client=_FakeTransport(), config=SharePointConfig())


# ---------------------------------------------------------------------------
# upload_reports
# ---------------------------------------------------------------------------


def test_upload_reports_preserves_the_staged_tree(tmp_path):
    (tmp_path / "files").mkdir()
    (tmp_path / "complexity-report.md").write_text("# overall", encoding="utf-8")
    (tmp_path / "files" / "etl.md").write_text("# etl", encoding="utf-8")
    (tmp_path / "dependency-graph.png").write_bytes(b"\x89PNG")

    transport = _FakeTransport()
    uploaded = sp.upload_reports(
        "MyApp",
        "sparksql",
        "TS",
        [
            tmp_path / "complexity-report.md",
            tmp_path / "files" / "etl.md",
            tmp_path / "dependency-graph.png",
        ],
        staging_root=tmp_path,
        client=transport,
        config=_config(),
    )
    root = f"{BASE}/MyApp/complexity/sparksql/TS"

    assert uploaded == [
        f"{root}/complexity-report.md",
        f"{root}/files/etl.md",
        f"{root}/dependency-graph.png",
    ]
    assert f"{root}/files" in transport.created
    # Binary artefacts go up as bytes; a PNG read as UTF-8 would not survive.
    assert transport.uploaded[-1][2] == b"\x89PNG"
    assert transport.uploaded[0][2] == "# overall"


def test_run_summary_reports_the_row_and_the_outcome():
    request = sp.ComplexityRequest(
        item_id="42",
        application="MyApp",
        output_language="SparkSQL",
        preferred_llm="claude-sonnet-4-5",
    )
    summary = sp.render_run_summary(
        request,
        target="sparksql",
        model="claude-sonnet-4-5",
        label="claude-sonnet-4-5",
        timestamp="TS",
        files_scored=3,
        failures=["broken.sas"],
        seconds=12.5,
        exit_status=1,
    )
    # It stands in for a Status column, so all four row values and the
    # outcome have to be in it.
    for expected in ("42", "MyApp", "SparkSQL", "claude-sonnet-4-5", "broken.sas"):
        assert expected in summary
    assert "Files scored**: 3" in summary
    assert "Exit status**: 1 (failed)" in summary


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_neither_source_is_rejected():
    assert cli.main([]) == 1


def test_both_sources_are_rejected(tmp_path):
    assert cli.main([str(tmp_path), "--sharepoint"]) == 1


@pytest.mark.parametrize(
    "flag", [["--app", "X"], ["--item-id", "1"], ["--sharepoint-out", "d"],
             ["--no-upload"]]
)
def test_sharepoint_only_flags_need_sharepoint(tmp_path, flag):
    assert cli.main([str(tmp_path), *flag]) == 1


def test_app_and_item_id_are_mutually_exclusive():
    assert cli.main(["--sharepoint", "--app", "X", "--item-id", "1"]) == 1


def test_pdf_still_needs_a_destination_locally(tmp_path):
    assert cli.main([str(tmp_path), "--pdf"]) == 1


def test_pdf_needs_no_out_dir_in_sharepoint_mode(monkeypatch):
    # SharePoint mode always has a destination: it stages into a temporary
    # directory. The guard must not fire before it gets there.
    args = cli.parse_args(["--sharepoint", "--pdf"])
    assert cli._argument_error(args) is None


# ---------------------------------------------------------------------------
# End to end against the fake transport
# ---------------------------------------------------------------------------


@pytest.fixture
def _wired(monkeypatch):
    """Point the CLI's SharePoint client and config at a fake transport."""

    def _wire(transport):
        from app_config import sharepoint as sp_module

        monkeypatch.setattr(sp_module, "get_sharepoint_client", lambda: transport)
        monkeypatch.setattr(
            SharePointConfig, "from_env", classmethod(lambda cls: _config())
        )
        return transport

    return _wire


def test_end_to_end_uploads_the_whole_tree(_wired):
    transport = _wired(_corpus_transport(names=("etl.sas", "load.sas")))
    assert cli.main(["--sharepoint", "--app", "MyApp"]) == 0

    names = [f"{folder}/{name}" for folder, name, _ in transport.uploaded]
    root = f"{BASE}/MyApp/complexity/sparksql"
    assert any(n.endswith("/complexity-report.md") for n in names)
    assert any("/files/" in n for n in names)
    assert any(n.endswith("/run-summary.md") for n in names)
    assert all(n.startswith(root) for n in names)


def test_source_id_is_the_drive_relative_path(_wired):
    # No temporary files: chunk_text takes the text with an explicit
    # source_id, and the library path is exactly the id wanted.
    transport = _wired(_corpus_transport(names=("etl.sas", "load.sas")))
    sources = cli._load_sharepoint_sources("MyApp", client=transport)

    assert sources is not None
    assert [result.source_id for result in sources.file_results] == [
        f"{BASE}/MyApp/scripts_original/etl.sas",
        f"{BASE}/MyApp/scripts_original/load.sas",
    ]


def test_source_ids_drive_the_per_file_report_names(_wired):
    transport = _wired(_corpus_transport(names=("etl.sas", "load.sas")))
    cli.main(["--sharepoint", "--app", "MyApp"])

    # source_stems turns those paths into unique, readable file names.
    per_file = sorted(
        name for folder, name, _ in transport.uploaded if folder.endswith("/files")
    )
    assert per_file == ["etl.md", "load.md"]


def test_no_upload_uploads_nothing(_wired, tmp_path):
    transport = _wired(_corpus_transport())
    status = cli.main(
        ["--sharepoint", "--app", "MyApp", "--no-upload", "--out-dir", str(tmp_path)]
    )

    assert status == 0
    assert transport.uploaded == []
    # ...but the reports are still delivered locally.
    assert (tmp_path / "complexity-report.md").is_file()
    assert (tmp_path / "run-summary.md").is_file()


def test_out_dir_is_kept_as_well_as_uploaded(_wired, tmp_path):
    transport = _wired(_corpus_transport())
    cli.main(["--sharepoint", "--app", "MyApp", "--out-dir", str(tmp_path)])

    assert (tmp_path / "complexity-report.md").is_file()
    assert transport.uploaded


def test_label_is_the_rules_profile_without_llm_eval(_wired):
    transport = _wired(_corpus_transport(rows=[_row(language="PySpark")]))
    cli.main(["--sharepoint", "--app", "MyApp"])

    # Output_Language from the row picks the profile, and the profile names
    # the folder when no model produced the estimate.
    assert all(
        folder.startswith(f"{BASE}/MyApp/complexity/pyspark/")
        for folder, _, _ in transport.uploaded
    )


def test_label_is_the_model_when_llm_eval_ran(_wired, monkeypatch):
    transport = _wired(
        _corpus_transport(rows=[_row(llm="claude-sonnet-4-5")])
    )
    # Preferred_LLM implies --llm-eval, so stub the pass out rather than call.
    monkeypatch.setattr(
        cli, "_run_evaluation", lambda *a, **k: 0
    )
    cli.main(["--sharepoint", "--app", "MyApp"])

    assert all(
        folder.startswith(f"{BASE}/MyApp/complexity/claude-sonnet-4-5/")
        for folder, _, _ in transport.uploaded
    )


def test_explicit_target_beats_the_row(_wired):
    transport = _wired(_corpus_transport(rows=[_row(language="PySpark")]))
    cli.main(["--sharepoint", "--app", "MyApp", "--target", "sparksql"])

    assert all(
        folder.startswith(f"{BASE}/MyApp/complexity/sparksql/")
        for folder, _, _ in transport.uploaded
    )


def test_each_row_gets_its_own_corpus_and_folder(_wired):
    # Applications are never merged: cross-file resolution across unrelated
    # ones would corrupt every verdict.
    transport = _FakeTransport(
        rows=[_row("1", application="AppA"), _row("2", application="AppB")],
        files={
            f"{BASE}/AppA/scripts_original": ["a.sas"],
            f"{BASE}/AppB/scripts_original": ["b.sas"],
        },
        texts={
            f"{BASE}/AppA/scripts_original/a.sas": _SAS,
            f"{BASE}/AppB/scripts_original/b.sas": _SAS,
        },
    )
    _wired(transport)
    assert cli.main(["--sharepoint"]) == 0

    folders = {folder.split("/complexity/")[0] for folder, _, _ in transport.uploaded}
    assert folders == {f"{BASE}/AppA", f"{BASE}/AppB"}


def test_one_failing_row_does_not_stop_the_others(_wired):
    transport = _FakeTransport(
        rows=[_row("1", application="Empty"), _row("2", application="AppB")],
        files={
            f"{BASE}/Empty/scripts_original": [],
            f"{BASE}/AppB/scripts_original": ["b.sas"],
        },
        texts={f"{BASE}/AppB/scripts_original/b.sas": _SAS},
    )
    _wired(transport)
    status = cli.main(["--sharepoint"])

    # Non-zero because one row failed, but the other still delivered.
    assert status == 1
    assert any("/AppB/" in folder for folder, _, _ in transport.uploaded)


def test_sharepoint_out_overrides_the_convention(_wired):
    transport = _wired(_corpus_transport())
    cli.main(["--sharepoint", "--app", "MyApp", "--sharepoint-out", "Scratch/runs"])

    assert all(
        folder.startswith("Scratch/runs") for folder, _, _ in transport.uploaded
    )


def test_no_rows_is_a_failed_run(_wired):
    _wired(_FakeTransport(rows=[]))
    assert cli.main(["--sharepoint"]) == 1
