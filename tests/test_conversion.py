"""
Tests for the conversion/ domain package — paths, request/conversion rows,
source discovery, and uploads.

No live SharePoint: every test drives a fake transport that implements the
handful of SharePointClient methods the package uses. What is checked here is
the *domain* layer — the folder conventions, the SharePoint internal column
names, and what gets uploaded where — not the Graph plumbing beneath it, which
tests/test_sharepoint.py covers.
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
from conversion import paths, requests as conv_requests, sources, upload

BASE = "Kit/Applications"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """An empty config.json, so nothing leaks in from the repo's own."""
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
        "list_id_sas_requests": "L-req",
        "list_id_sas_conversions": "L-conv",
    }
    values.update(overrides)
    return SharePointConfig(**values)


class _FakeTransport:
    """The SharePointClient surface the conversion package actually uses."""

    def __init__(self, *, files=None, texts=None, lists=None):
        # folder -> [file names]
        self.files: dict[str, list[str]] = dict(files or {})
        # path -> text
        self.texts: dict[str, str] = dict(texts or {})
        # list id -> [ {id, fields} ]
        self.lists: dict[str, list[dict]] = dict(lists or {})
        self.created: list[str] = []
        self.uploaded: list[tuple[str, str, Any]] = []
        self.updated: list[tuple[str, str, dict]] = []

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
        try:
            return self.texts[path]
        except KeyError:
            raise SharePointError(f"no file at {path}") from None

    def list_items(self, list_id, **_options):
        return list(self.lists.get(list_id, []))

    def create_folder(self, path):
        self.created.append(path)
        return {"name": path.rsplit("/", 1)[-1]}

    def upload_file(self, folder, name, content):
        self.uploaded.append((folder, name, content))
        return {"name": name}

    def update_list_item(self, list_id, item_id, fields):
        self.updated.append((list_id, str(item_id), dict(fields)))
        return dict(fields)


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def test_folder_conventions():
    cfg = _config()
    assert paths.original_scripts("MyApp", config=cfg) == (
        f"{BASE}/MyApp/scripts_original"
    )
    assert paths.converted_scripts("MyApp", config=cfg) == (
        f"{BASE}/MyApp/scripts_converted"
    )
    assert paths.validation("MyApp", config=cfg) == (
        f"{BASE}/MyApp/scripts_converted/validation"
    )
    assert paths.upload_target(
        "MyApp", "claude-sonnet-4-5", "20260804", config=cfg
    ) == (f"{BASE}/MyApp/scripts_converted/claude-sonnet-4-5/20260804")
    assert paths.prompt_target(
        "MyApp", "claude-sonnet-4-5", "20260804", config=cfg
    ) == (f"{BASE}/MyApp/scripts_converted/claude-sonnet-4-5/20260804/prompts")


def test_paths_strip_the_document_library_prefix(_isolated):
    # The base is normalised on resolution, so a value pasted from a
    # SharePoint URL still yields a drive-relative folder.
    _isolated.write_text(
        json.dumps(
            {
                "sharepoint": {
                    "file_server_base_path": "Shared Documents/Kit/Applications"
                }
            }
        ),
        encoding="utf-8",
    )
    app_config.clear_cache()
    assert paths.original_scripts("MyApp") == f"{BASE}/MyApp/scripts_original"


# ---------------------------------------------------------------------------
# requests / conversions
# ---------------------------------------------------------------------------


_REQUEST_ROW = {
    "id": "7",
    "fields": {
        "Application Name": "MyApp",
        "Source Language": "SAS",
        "Destination Language": "PySpark",
        "Macro File Name x003f ": "macros.sas",
        "Validation x0020 Documents x0020 ": "Yes",
        "Status": "New",
    },
}


def test_request_row_projection_uses_the_internal_column_names():
    # The encoded characters and trailing spaces are literal: SharePoint
    # derives an internal name once and keeps it, so these are transcribed.
    request = conv_requests.format_request_item_params(_REQUEST_ROW)

    assert request.item_id == "7"
    assert request.application_name == "MyApp"
    assert request.input_language == "SAS"
    assert request.output_language == "PySpark"
    assert request.macro_file_name == "macros.sas"
    assert request.is_validation_required is True
    assert request.status == "New"


@pytest.mark.parametrize(
    "raw, expected",
    [("Yes", True), ("no", False), (True, True), (0, False), (1, True), (None, False)],
)
def test_validation_flag_coercion(raw, expected):
    row = {
        "id": "1",
        "fields": {
            "Application Name": "MyApp",
            "Validation x0020 Documents x0020 ": raw,
        },
    }
    assert (
        conv_requests.format_request_item_params(row).is_validation_required is expected
    )


def test_a_request_without_an_application_name_is_an_error():
    with pytest.raises(SharePointError, match="Application Name"):
        conv_requests.format_request_item_params({"id": "1", "fields": {}})


def test_conversion_row_projection():
    item = conv_requests.format_conversion_item_params(
        {
            "id": "3",
            "fields": {
                "Request_ID": "7",
                "Script_Name": "etl.sas",
                "Model": "anthropic: claude-sonnet-4-5",
                "Status": "Pending",
            },
        }
    )
    assert item.app_request_id == 7
    assert item.script_name == "etl.sas"
    # Left in the operator's spelling: llm_client._split_model parses it, and a
    # second parser here would be one to keep in step.
    assert item.preferred_llm == "anthropic: claude-sonnet-4-5"
    assert item.status == "Pending"


def test_a_non_numeric_request_id_is_warned_not_fatal(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="conversion.requests"):
        item = conv_requests.format_conversion_item_params(
            {"id": "3", "fields": {"Request_ID": "seven", "Script_Name": "a.sas"}}
        )
    assert item.app_request_id is None
    assert item.script_name == "a.sas"
    assert "non-numeric" in caplog.text


def test_requests_skips_a_malformed_row(caplog):
    import logging

    transport = _FakeTransport(
        lists={"L-req": [_REQUEST_ROW, {"id": "8", "fields": {"Status": "New"}}]}
    )
    with caplog.at_level(logging.WARNING, logger="conversion.requests"):
        rows = conv_requests.requests(client=transport, config=_config())

    # One bad row must not hide every good one.
    assert [row.application_name for row in rows] == ["MyApp"]
    assert "skipping a malformed row" in caplog.text


def test_pending_requests_filters_completed_rows():
    done = {
        "id": "9",
        "fields": {"Application Name": "OldApp", "Status": "Completed"},
    }
    transport = _FakeTransport(lists={"L-req": [_REQUEST_ROW, done]})
    pending = conv_requests.pending_requests(client=transport, config=_config())

    assert [row.application_name for row in pending] == ["MyApp"]


def test_a_blank_status_is_pending():
    row = {"id": "1", "fields": {"Application Name": "MyApp"}}
    assert conv_requests.format_request_item_params(row).is_pending is True


def test_update_request_status_writes_the_status_column():
    transport = _FakeTransport()
    conv_requests.update_request_status(
        7, "Completed", client=transport, config=_config()
    )

    assert transport.updated == [("L-req", "7", {"Status": "Completed"})]


def test_update_request_status_needs_the_list_configured():
    with pytest.raises(SharePointError, match="sharepoint.list_id_sas_requests"):
        conv_requests.update_request_status(
            7,
            "Completed",
            client=_FakeTransport(),
            config=_config(list_id_sas_requests=None),
        )


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


def test_source_files_accepts_sas_and_txt():
    folder = f"{BASE}/MyApp/scripts_original"
    transport = _FakeTransport(files={folder: ["b.sas", "a.txt", "notes.pdf", "c.sas"]})
    found = sources.source_files("MyApp", client=transport, config=_config())

    # .txt counts: scripts arrive from systems that will not hand over a .sas.
    # Sorted, so cross-file batching is reproducible across runs.
    assert found == [f"{folder}/a.txt", f"{folder}/b.sas", f"{folder}/c.sas"]


def test_unknown_input_type_raises():
    with pytest.raises(SharePointError, match="no source extensions known"):
        sources.source_files(
            "MyApp", "cobol", client=_FakeTransport(), config=_config()
        )


def test_load_returns_the_text():
    transport = _FakeTransport(texts={"a/b.sas": "data a; run;"})
    assert sources.load("a/b.sas", client=transport) == "data a; run;"


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, kind, expected",
    [
        ("etl.sas", "pyspark", "etl.ipynb"),
        ("etl.sas", "sparksql", "etl.ipynb"),
        ("etl", "python", "etl.py"),
        ("etl.sas", "sql", "etl.sql"),
        ("etl.sas", "scala", "etl.txt"),
        ("etl.sas", "unheard-of", "etl.txt"),
    ],
)
def test_set_file_extension_replaces_rather_than_appends(name, kind, expected):
    # The name arrives as the SAS source's, and etl.sas.ipynb would be wrong in
    # a way that only shows up in the library listing.
    assert upload.set_file_extension(name, kind) == expected


def test_upload_converted_script_renders_a_notebook():
    transport = _FakeTransport()
    path = upload.upload_converted_script(
        "MyApp",
        "etl.sas",
        "pyspark",
        "## Translation\n\n```python\nprint(1)\n```\n",
        "claude-sonnet-4-5",
        "20260804",
        client=transport,
        config=_config(),
    )
    folder = f"{BASE}/MyApp/scripts_converted/claude-sonnet-4-5/20260804"

    assert path == f"{folder}/etl.ipynb"
    assert transport.created == [folder]  # created before the upload
    uploaded_folder, name, content = transport.uploaded[0]
    assert (uploaded_folder, name) == (folder, "etl.ipynb")
    notebook = json.loads(content)
    assert notebook["nbformat"] == 4
    assert any(cell["cell_type"] == "code" for cell in notebook["cells"])


def test_upload_converted_script_leaves_a_flat_type_alone():
    transport = _FakeTransport()
    upload.upload_converted_script(
        "MyApp",
        "etl.sas",
        "sql",
        "SELECT 1;",
        "claude-sonnet-4-5",
        "20260804",
        client=transport,
        config=_config(),
    )
    _, name, content = transport.uploaded[0]

    assert name == "etl.sql"
    assert content == "SELECT 1;"


def test_upload_prompt_file_lands_in_the_runs_prompt_subdirectory():
    transport = _FakeTransport()
    path = upload.upload_prompt_file(
        "MyApp",
        "item-1.md",
        "# Effective prompt",
        "claude-sonnet-4-5",
        "20260804",
        client=transport,
        config=_config(),
    )
    folder = f"{BASE}/MyApp/scripts_converted/claude-sonnet-4-5/20260804/prompts"

    assert path == f"{folder}/item-1.md"
    assert transport.created == [folder]
    assert transport.uploaded == [(folder, "item-1.md", "# Effective prompt")]


def test_upload_validation_file_lands_beside_the_converted_scripts():
    transport = _FakeTransport()
    path = upload.upload_validation_file(
        "MyApp", "report.md", "# Report", client=transport, config=_config()
    )
    folder = f"{BASE}/MyApp/scripts_converted/validation"

    assert path == f"{folder}/report.md"
    assert transport.created == [folder]
    assert transport.uploaded == [(folder, "report.md", "# Report")]


def test_upload_validation_file_takes_bytes():
    # The validation report is also written as a PDF.
    transport = _FakeTransport()
    upload.upload_validation_file(
        "MyApp", "report.pdf", b"%PDF-1.4", client=transport, config=_config()
    )
    assert transport.uploaded[0][2] == b"%PDF-1.4"
