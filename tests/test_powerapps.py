"""
Tests for app_config.powerapps (config + list-item parsing) and the
SharePoint-mode helpers in demo_run.

Everything here is offline: no msgraph-sdk, no network. PowerAppsConfig is
resolved from a controlled environment + tmp config.json; the parsing helpers
are pure; and the demo_run upload/download helpers are exercised against a tiny
duck-typed fake client (mirroring the style of tests/test_sharepoint.py) so the
SharePoint output layout can be asserted without the SDK.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

import app_config
import demo_run
from app_config import powerapps

_POWERAPPS_ENV = (
    "POWERAPPS_LIST_NAME",
    "POWERAPPS_REQUEST_ID_FIELD",
    "POWERAPPS_APPLICATION_NAME_FIELD",
    "POWERAPPS_LIVE_VALIDATION_FIELD",
    "POWERAPPS_MODEL_FIELD",
    "POWERAPPS_TIMESTAMP_FIELD",
)

# A model that app_config.is_accessible_model accepts.
_MODEL = app_config.ACCESSIBLE_MODELS[0]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Empty config file, no POWERAPPS_* env vars, file cache cleared."""
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(app_config.ENV_VAR, str(cfg))
    for var in _POWERAPPS_ENV:
        monkeypatch.delenv(var, raising=False)
    app_config.clear_cache()
    yield cfg
    app_config.clear_cache()


def _set(cfg_path, mapping) -> None:
    cfg_path.write_text(json.dumps(mapping), encoding="utf-8")
    app_config.clear_cache()


# ---------------------------------------------------------------------------
# PowerAppsConfig.from_env
# ---------------------------------------------------------------------------


def test_config_defaults(_isolated):
    cfg = powerapps.PowerAppsConfig.from_env()
    assert cfg.list_name is None
    assert cfg.request_id_field == powerapps.DEFAULT_REQUEST_ID_FIELD
    assert cfg.application_name_field == powerapps.DEFAULT_APPLICATION_NAME_FIELD
    assert cfg.live_validation_field == powerapps.DEFAULT_LIVE_VALIDATION_FIELD
    assert cfg.model_field == powerapps.DEFAULT_MODEL_FIELD
    assert cfg.timestamp_field == powerapps.DEFAULT_TIMESTAMP_FIELD


def test_config_from_config_json(_isolated):
    _set(
        _isolated,
        {
            "powerapps": {
                "list_name": "Requests",
                "request_id_field": "ReqID",
                "model_field": "SelectedModel",
            }
        },
    )
    cfg = powerapps.PowerAppsConfig.from_env()
    assert cfg.list_name == "Requests"
    assert cfg.request_id_field == "ReqID"
    assert cfg.model_field == "SelectedModel"
    # Unspecified keys still fall back to the defaults.
    assert cfg.application_name_field == powerapps.DEFAULT_APPLICATION_NAME_FIELD


def test_config_env_beats_config(monkeypatch, _isolated):
    _set(_isolated, {"powerapps": {"list_name": "FromFile"}})
    monkeypatch.setenv("POWERAPPS_LIST_NAME", "FromEnv")
    monkeypatch.setenv("POWERAPPS_REQUEST_ID_FIELD", "EnvReqField")
    cfg = powerapps.PowerAppsConfig.from_env()
    assert cfg.list_name == "FromEnv"
    assert cfg.request_id_field == "EnvReqField"


# ---------------------------------------------------------------------------
# _coerce_bool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("true", True),
        ("True", True),
        ("YES", True),
        ("1", True),
        ("false", False),
        ("no", False),
        ("0", False),
        ("", False),
    ],
)
def test_coerce_bool(value, expected):
    assert powerapps._coerce_bool(value) is expected


def test_coerce_bool_none_uses_default():
    assert powerapps._coerce_bool(None) is False
    assert powerapps._coerce_bool(None, default=True) is True


def test_coerce_bool_unrecognised_warns_and_defaults(caplog):
    with caplog.at_level("WARNING"):
        assert powerapps._coerce_bool("maybe") is False
    assert "unrecognised boolean" in caplog.text


# ---------------------------------------------------------------------------
# select_request
# ---------------------------------------------------------------------------


def _rows():
    return [
        {"id": "1", "fields": {"RequestId": "REQ-1", "ApplicationName": "app_a"}},
        {"id": "2", "fields": {"RequestId": "REQ-2", "ApplicationName": "app_b"}},
    ]


def test_select_request_finds_the_row():
    cfg = powerapps.PowerAppsConfig()
    row = powerapps.select_request(_rows(), "REQ-2", cfg)
    assert row["id"] == "2"


def test_select_request_trims_whitespace():
    cfg = powerapps.PowerAppsConfig()
    row = powerapps.select_request(_rows(), "  REQ-1  ", cfg)
    assert row["id"] == "1"


def test_select_request_no_match_raises():
    cfg = powerapps.PowerAppsConfig()
    with pytest.raises(powerapps.PowerAppsError, match="no Power Apps request"):
        powerapps.select_request(_rows(), "REQ-9", cfg)


def test_select_request_multiple_matches_raises():
    cfg = powerapps.PowerAppsConfig()
    rows = _rows() + [{"id": "3", "fields": {"RequestId": "REQ-1"}}]
    with pytest.raises(powerapps.PowerAppsError, match="expected exactly one"):
        powerapps.select_request(rows, "REQ-1", cfg)


# ---------------------------------------------------------------------------
# parse_run_request
# ---------------------------------------------------------------------------


def _row(**fields):
    base = {
        "RequestId": "REQ-1",
        "ApplicationName": "app_a",
        "Model": _MODEL,
        "LiveValidation": True,
    }
    base.update(fields)
    return {"id": "42", "fields": base}


def test_parse_run_request_happy_path():
    cfg = powerapps.PowerAppsConfig()
    req = powerapps.parse_run_request(_row(Timestamp="2026-07-19T08:30:00Z"), cfg)
    assert req.request_id == "REQ-1"
    assert req.application_name == "app_a"
    assert req.model == _MODEL
    assert req.live_validation is True
    assert req.item_id == "42"
    # Timestamp sanitised into a single safe path segment.
    assert req.timestamp == "2026-07-19T08-30-00Z"
    assert "/" not in req.timestamp and ":" not in req.timestamp


def test_parse_run_request_generates_timestamp_when_absent():
    cfg = powerapps.PowerAppsConfig()
    req = powerapps.parse_run_request(_row(), cfg)
    # Generated stamp: YYYYMMDDTHHMMSSZ.
    assert req.timestamp.endswith("Z") and "T" in req.timestamp
    assert req.timestamp[:8].isdigit()


def test_parse_run_request_validation_flag_off():
    cfg = powerapps.PowerAppsConfig()
    req = powerapps.parse_run_request(_row(LiveValidation="No"), cfg)
    assert req.live_validation is False


def test_parse_run_request_rejects_inaccessible_model():
    cfg = powerapps.PowerAppsConfig()
    with pytest.raises(powerapps.PowerAppsError, match="not an accessible model"):
        powerapps.parse_run_request(_row(Model="totally-made-up"), cfg)


def test_parse_run_request_requires_application_name():
    cfg = powerapps.PowerAppsConfig()
    with pytest.raises(powerapps.PowerAppsError, match="ApplicationName"):
        powerapps.parse_run_request(_row(ApplicationName="  "), cfg)


def test_parse_run_request_requires_request_id():
    cfg = powerapps.PowerAppsConfig()
    with pytest.raises(powerapps.PowerAppsError, match="RequestId"):
        powerapps.parse_run_request(_row(RequestId=""), cfg)


def test_parse_run_request_honours_custom_field_names():
    cfg = powerapps.PowerAppsConfig(
        request_id_field="ReqID",
        application_name_field="App",
        model_field="Mdl",
        live_validation_field="Validate",
        timestamp_field="Stamp",
    )
    row = {
        "id": "7",
        "fields": {
            "ReqID": "R7",
            "App": "myapp",
            "Mdl": _MODEL,
            "Validate": "1",
            "Stamp": "run1",
        },
    }
    req = powerapps.parse_run_request(row, cfg)
    assert (req.request_id, req.application_name, req.timestamp) == (
        "R7",
        "myapp",
        "run1",
    )
    assert req.live_validation is True


# ---------------------------------------------------------------------------
# demo_run SharePoint helpers against a fake client
# ---------------------------------------------------------------------------


class _FakeClient:
    """A duck-typed SharePointClient: an in-memory tree + list rows.

    ``tree`` maps a folder path to a list of child entries (each a dict with
    ``name`` / ``is_folder``); ``files`` maps a file path to bytes. Records the
    directories created and files written so tests can assert the output layout.
    """

    def __init__(self, tree=None, files=None):
        self.tree = tree or {}
        self.files = files or {}
        self.created_dirs: list[str] = []
        self.written: dict[str, str] = {}

    def list_directory(self, path=""):
        return self.tree.get(path, [])

    def read_file(self, path):
        return self.files[path]

    def create_directory(self, path, *, conflict_behavior="fail"):
        self.created_dirs.append(path)
        return {"name": path.rsplit("/", 1)[-1]}

    def write_file(self, path, content):
        self.written[path] = content
        return {"name": path.rsplit("/", 1)[-1]}


def test_discover_sharepoint_sas_recurses_and_sorts():
    client = _FakeClient(
        tree={
            "app_a": [
                {"name": "b.sas", "is_folder": False},
                {"name": "sub", "is_folder": True},
                {"name": "notes.txt", "is_folder": False},
                {"name": "a.sas", "is_folder": False},
            ],
            "app_a/sub": [{"name": "c.sas", "is_folder": False}],
        }
    )
    entries = demo_run._discover_sharepoint_sas(client, "app_a")
    # Only .sas files, sorted by relative path, with library paths preserved.
    assert entries == [
        ("app_a/a.sas", "a.sas"),
        ("app_a/b.sas", "b.sas"),
        ("app_a/sub/c.sas", "sub/c.sas"),
    ]


def test_download_inputs_preserves_structure(tmp_path):
    client = _FakeClient(
        files={"app_a/a.sas": b"data a", "app_a/sub/c.sas": b"data c"}
    )
    entries = [("app_a/a.sas", "a.sas"), ("app_a/sub/c.sas", "sub/c.sas")]
    paths = demo_run._download_inputs(client, entries, tmp_path)
    assert (tmp_path / "a.sas").read_bytes() == b"data a"
    assert (tmp_path / "sub" / "c.sas").read_bytes() == b"data c"
    assert paths == [str(tmp_path / "a.sas"), str(tmp_path / "sub" / "c.sas")]


def test_ensure_directory_swallows_already_exists():
    from app_config.sharepoint import SharePointError

    class _Client(_FakeClient):
        def create_directory(self, path, *, conflict_behavior="fail"):
            self.created_dirs.append(path)
            if path == "app_a/output":
                raise SharePointError("nameAlreadyExists: it is there")
            return {"name": path}

    client = _Client()
    demo_run._ensure_directory(client, "app_a/output/ts1")
    assert client.created_dirs == ["app_a", "app_a/output", "app_a/output/ts1"]


def test_ensure_directory_reraises_other_errors():
    from app_config.sharepoint import SharePointError

    class _Client(_FakeClient):
        def create_directory(self, path, *, conflict_behavior="fail"):
            raise SharePointError("access denied")

    with pytest.raises(SharePointError, match="access denied"):
        demo_run._ensure_directory(_Client(), "app_a/output")


def _req(live_validation):
    return powerapps.RunRequest(
        request_id="REQ-1",
        application_name="app_a",
        model=_MODEL,
        live_validation=live_validation,
        timestamp="ts1",
        item_id="42",
    )


def _outputs():
    return [
        {
            "item_id": "item-1",
            "is_batch": False,
            "kind": "chunk",
            "source_files": ["a.sas"],
            "response": "print('a')",
            "validation": {"passed": True, "score": 0.9, "metrics": []},
        },
        {
            "item_id": "item-2",
            "is_batch": True,
            "kind": "batch",
            "source_files": ["a.sas", "b.sas"],
            "response": "print('b')",
            "validation": {"passed": False, "score": 0.4, "metrics": []},
        },
    ]


def test_upload_outputs_with_validation_writes_expected_layout():
    client = _FakeClient()
    out_dir = demo_run._upload_outputs(
        client, _req(True), _outputs(), validating=True
    )
    assert out_dir == "app_a/output/ts1"
    # Responses.
    assert "app_a/output/ts1/item-1.txt" in client.written
    assert "print('a')" in client.written["app_a/output/ts1/item-1.txt"]
    # Per-item verdicts + aggregate summary.
    assert "app_a/output/ts1/validation/item-1.json" in client.written
    assert "app_a/output/ts1/validation/item-2.json" in client.written
    summary = json.loads(client.written["app_a/output/ts1/validation/summary.json"])
    assert summary["items"] == 2 and summary["passed"] == 1 and summary["failed"] == 1
    # The validation subfolder was created.
    assert "app_a/output/ts1/validation" in client.created_dirs


def test_upload_outputs_without_validation_skips_validation_dir():
    client = _FakeClient()
    demo_run._upload_outputs(client, _req(False), _outputs(), validating=False)
    assert "app_a/output/ts1/item-1.txt" in client.written
    assert not any(p.startswith("app_a/output/ts1/validation") for p in client.written)
    assert "app_a/output/ts1/validation" not in client.created_dirs
