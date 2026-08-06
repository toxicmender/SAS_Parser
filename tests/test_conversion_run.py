"""
Tests for conversion/run.py — one request row, end to end.

The whole flow runs against a fake transport and a fake pipeline, which is why
``run_request`` takes a *pipeline factory* rather than building one: no network,
no LLM, no reference corpus, and no temporary directory. That last one is the
point of ``run_texts`` — a SharePoint-hosted corpus is chunked from text, and
the drive-relative path is the source id, so nothing is ever staged to disk.

What is checked: sources come from ``scripts_original``; output lands under
``scripts_converted/{model}/{timestamp}``; ``Status`` is written on the success
*and* the failure path; one bad row does not take the others down; and
``--no-upload`` writes nothing at all.
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
from llm_client import TokenUsage
from conversion import requests as conv_requests, run as conv_run

BASE = "Kit/Applications"
APP = "MyApp"
MODEL = "claude-sonnet-4-5"
STAMP = "20260806T120000Z"


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
        "list_id_sas_requests": "L-req",
        "list_id_sas_conversions": "L-conv",
    }
    values.update(overrides)
    return SharePointConfig(**values)


class _FakeTransport:
    """The SharePointClient surface conversion.run actually uses."""

    def __init__(self, *, files=None, texts=None, fail_upload=False):
        self.files: dict[str, list[str]] = dict(files or {})
        self.texts: dict[str, str] = dict(texts or {})
        self.created: list[str] = []
        self.uploaded: list[tuple[str, str, Any]] = []
        self.updated: list[tuple[str, str, dict]] = []
        self.fail_upload = fail_upload

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

    def create_folder(self, path):
        self.created.append(path)
        return {"name": path.rsplit("/", 1)[-1]}

    def upload_file(self, folder, name, content):
        if self.fail_upload:
            raise SharePointError("upload refused")
        self.uploaded.append((folder, name, content))
        return {"name": name}

    def update_list_item(self, list_id, item_id, fields):
        self.updated.append((list_id, str(item_id), dict(fields)))
        return dict(fields)

    @property
    def statuses(self) -> list[str]:
        return [fields.get("Status") for _, _, fields in self.updated]


def _no_usage() -> TokenUsage:
    """The real TokenUsage, all zeros — a gateway that reported nothing.

    Not a stub: ValidationReport validates this field, and a duck-typed
    stand-in would only prove the fake matches itself.
    """
    return TokenUsage()


class _FakePipeline:
    """The SasLLMPipeline surface conversion.run touches."""

    output_language = "SparkSQL"
    instructions_fingerprint = None
    policy_fingerprint = None

    def __init__(self, *, outputs=None, raises=None):
        self._outputs = outputs
        self._raises = raises
        self.seen: list[tuple[str, str]] = []
        self.thread_id: str | None = None
        self.token_usage = _no_usage()

    def run_texts(self, sources, *, thread_id=None, resume=False):
        if self._raises is not None:
            raise self._raises
        self.seen = list(sources)
        self.thread_id = thread_id
        if self._outputs is not None:
            return self._outputs
        return [
            {
                "item_id": f"item-{i}",
                "source_files": [source_id],
                "response": "```sql\nSELECT 1\n```",
            }
            for i, (source_id, _text) in enumerate(sources)
        ]


def _request(**overrides) -> conv_requests.ConversionRequest:
    values: dict[str, Any] = {
        "item_id": "42",
        "application_name": APP,
        "output_language": "SparkSQL",
        "is_validation_required": False,
    }
    values.update(overrides)
    return conv_requests.ConversionRequest(**values)


def _transport(**overrides) -> _FakeTransport:
    original = f"{BASE}/{APP}/scripts_original"
    return _FakeTransport(
        files={original: ["etl.sas", "load.sas", "notes.md"]},
        texts={
            f"{original}/etl.sas": "data work.a; set raw.b; run;",
            f"{original}/load.sas": "proc means data=work.a; run;",
        },
        **overrides,
    )


def _run(transport, pipeline=None, **overrides):
    built = pipeline if pipeline is not None else _FakePipeline()
    kwargs: dict[str, Any] = {
        "build_pipeline": lambda model, validate: built,
        "model": MODEL,
        "client": transport,
        "config": _config(),
        "timestamp": STAMP,
    }
    kwargs.update(overrides)
    return conv_run.run_request(_request(), **kwargs), built


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_sources_come_from_scripts_original_as_text():
    transport = _transport()
    outcome, pipeline = _run(transport)

    assert outcome.ok
    # Drive-relative paths, sorted, .md filtered out by the extension set.
    assert [source_id for source_id, _ in pipeline.seen] == [
        f"{BASE}/{APP}/scripts_original/etl.sas",
        f"{BASE}/{APP}/scripts_original/load.sas",
    ]
    assert pipeline.seen[0][1] == "data work.a; set raw.b; run;"


def test_the_request_id_is_the_thread_id():
    _, pipeline = _run(_transport())

    assert pipeline.thread_id == "42"


def test_output_lands_under_model_and_timestamp():
    transport = _transport()
    outcome, _ = _run(transport)

    target = f"{BASE}/{APP}/scripts_converted/{MODEL}/{STAMP}"
    # upload_converted_script creates the folder before every write — Graph's
    # simple upload does not create missing parents, and create_folder is
    # idempotent — so the folder recurs, but only ever this one.
    assert set(transport.created) == {target}
    assert all(folder == target for folder, _, _ in transport.uploaded)
    assert outcome.uploaded
    assert all(path.startswith(target) for path in outcome.uploaded)


def test_notebooks_are_named_after_the_source_files():
    transport = _transport()
    _run(transport)

    names = sorted(name for _, name, _ in transport.uploaded)
    assert names == ["etl.ipynb", "load.ipynb"]


def test_status_is_written_running_then_completed():
    transport = _transport()
    _run(transport)

    assert transport.statuses == [conv_run.STATUS_RUNNING, conv_run.STATUS_DONE]
    assert all(list_id == "L-req" for list_id, _, _ in transport.updated)
    assert all(item_id == "42" for _, item_id, _ in transport.updated)


# ---------------------------------------------------------------------------
# Failure paths — a request list is a queue, not a transaction
# ---------------------------------------------------------------------------


def test_a_translation_failure_marks_the_row_failed_and_does_not_raise():
    transport = _transport()
    outcome, _ = _run(
        transport, pipeline=_FakePipeline(raises=RuntimeError("gateway said no"))
    )

    assert not outcome.ok
    assert outcome.error is not None and "gateway said no" in outcome.error
    assert transport.statuses == [conv_run.STATUS_RUNNING, conv_run.STATUS_FAILED]


def test_an_upload_failure_marks_the_row_failed():
    transport = _transport(fail_upload=True)
    outcome, _ = _run(transport)

    assert not outcome.ok
    assert transport.statuses == [conv_run.STATUS_RUNNING, conv_run.STATUS_FAILED]


def test_no_sources_fails_the_row_without_calling_the_model():
    transport = _FakeTransport(files={}, texts={})
    outcome, pipeline = _run(transport)

    assert not outcome.ok
    assert pipeline.seen == []
    # Nothing was started, so nothing is marked in progress either.
    assert transport.updated == []


def test_one_unreadable_file_does_not_lose_the_application():
    transport = _transport()
    del transport.texts[f"{BASE}/{APP}/scripts_original/load.sas"]
    outcome, pipeline = _run(transport)

    assert outcome.ok
    assert [source_id for source_id, _ in pipeline.seen] == [
        f"{BASE}/{APP}/scripts_original/etl.sas"
    ]


def test_a_status_write_failure_does_not_fail_a_finished_run(monkeypatch):
    # The scripts are uploaded, which is the thing that mattered.
    transport = _transport()

    def _refuse(list_id, item_id, fields):
        raise SharePointError("list is read-only")

    monkeypatch.setattr(transport, "update_list_item", _refuse)
    outcome, _ = _run(transport)

    assert outcome.ok
    assert outcome.uploaded


# ---------------------------------------------------------------------------
# --no-upload, XREF, and row selection
# ---------------------------------------------------------------------------


def test_no_upload_writes_nothing_and_leaves_status_alone():
    transport = _transport()
    outcome, pipeline = _run(transport, upload=False)

    assert outcome.ok
    assert pipeline.seen  # it still converted
    assert transport.uploaded == []
    assert transport.created == []
    assert transport.updated == []


def test_xref_path_mappings_are_applied_to_the_source_text():
    from xref.sourcing import XrefMappings

    transport = _transport()
    original = f"{BASE}/{APP}/scripts_original"
    transport.texts[f"{original}/etl.sas"] = "libname raw '/data/in';\n"
    _, pipeline = _run(
        transport, xref_mappings=XrefMappings(by_path={"/data/in": "/mnt/bronze"})
    )

    text = dict(pipeline.seen)[f"{original}/etl.sas"]
    assert "libname raw '/mnt/bronze';" in text


def test_select_requests_narrows_by_id_then_by_application():
    rows = [
        _request(item_id="1", application_name="Alpha"),
        _request(item_id="2", application_name="Beta"),
        _request(item_id="3", application_name="beta "),
    ]

    assert [r.item_id for r in conv_run.select_requests(rows, request_id="2")] == ["2"]
    # Application matching folds case and strips, unlike the reference's exact
    # comparison, which a trailing space silently defeats.
    assert [
        r.item_id for r in conv_run.select_requests(rows, application="Beta")
    ] == ["2", "3"]
    assert len(conv_run.select_requests(rows)) == 3


def test_model_for_takes_the_first_conversion_row_that_names_one():
    request = _request(item_id="7")
    items = [
        conv_requests.ConversionItem(item_id="a", app_request_id=7, preferred_llm=None),
        conv_requests.ConversionItem(
            item_id="b", app_request_id=7, preferred_llm="gpt-5.4"
        ),
        conv_requests.ConversionItem(
            item_id="c", app_request_id=7, preferred_llm="gemini-3.1-pro"
        ),
    ]

    assert conv_run.model_for(request, items, "fallback") == "gpt-5.4"


def test_model_for_falls_back_when_no_row_names_one():
    request = _request(item_id="7")
    other = conv_requests.ConversionItem(
        item_id="a", app_request_id=99, preferred_llm="gpt-5.4"
    )

    assert conv_run.model_for(request, [other], "fallback") == "fallback"


# ---------------------------------------------------------------------------
# Validation artefacts
# ---------------------------------------------------------------------------


def _verdict(passed=True, score=0.9):
    return {
        "passed": passed,
        "score": score,
        "metrics": [
            {
                "metric": "coverage",
                "score": score,
                "threshold": 0.7,
                "passed": passed,
                "skipped": False,
            }
        ],
    }


def test_validation_artefacts_land_beside_the_converted_scripts():
    transport = _transport()
    outputs = [
        {
            "item_id": "item-0",
            "source_files": ["etl.sas"],
            "response": "```sql\nSELECT 1\n```",
            "validation": _verdict(),
        }
    ]
    conv_run.run_request(
        _request(is_validation_required=True),
        build_pipeline=lambda model, validate: _FakePipeline(outputs=outputs),
        model=MODEL,
        client=transport,
        config=_config(),
        timestamp=STAMP,
    )

    validation_folder = f"{BASE}/{APP}/scripts_converted/validation"
    uploaded = {
        name: content
        for folder, name, content in transport.uploaded
        if folder == validation_folder
    }
    assert "item-0.json" in uploaded
    assert conv_run.SUMMARY_NAME in uploaded
    assert conv_run.REPORT_NAME in uploaded
    assert conv_run.REPORT_PDF_NAME in uploaded
    assert json.loads(uploaded[conv_run.SUMMARY_NAME])["passed"] == 1


def test_validation_summary_counts_and_tolerates_no_usage():
    outputs = [
        {"item_id": "a", "validation": _verdict(passed=True, score=1.0)},
        {"item_id": "b", "validation": _verdict(passed=False, score=0.5)},
        {"item_id": "c"},  # unscored
    ]

    summary = conv_run.validation_summary(outputs, token_usage=_no_usage())

    assert summary["items"] == 2
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["mean_score"] == pytest.approx(0.75)
    # None, not zero: the caller can tell "not reported" from "reported as 0".
    assert summary["token_usage"] is None
