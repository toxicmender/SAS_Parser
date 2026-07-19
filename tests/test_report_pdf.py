"""
Tests for validation.pdf: rendering a ValidationReport to PDF and publishing
it to SharePoint.

No live SharePoint (and no msgraph-sdk install) is needed — the upload is
exercised through a tiny stub client that records ``write_file(path, content)``.
The rendering path uses only pymupdf (a core dependency) and markdown-it-py, so
the produced bytes are reopened with pymupdf to assert on the real PDF.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pymupdf
import pytest

import app_config
from validation import report_from_thread, report_from_verdicts
from validation.models import CaseResult, MetricResult, ValidationReport
from validation.pdf import publish_report_pdf, report_to_pdf


def _report(model: str = "claude-sonnet-4-5", cases: int = 1) -> ValidationReport:
    results = [
        CaseResult(
            case_id=f"case_{i}",
            item_count=1,
            metrics=[
                MetricResult(
                    metric="response_coverage", score=1.0, threshold=1.0, passed=True
                ),
                MetricResult(
                    metric="python_syntax",
                    score=0.0,
                    threshold=1.0,
                    passed=False,
                    details="ast.parse failed",
                ),
            ],
        )
        for i in range(cases)
    ]
    return ValidationReport(model=model, results=results)


class _StubClient:
    """Records the last write_file call and returns a drive-item-shaped dict."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, bytes]] = []

    def write_file(self, path: str, content):
        self.writes.append((path, content))
        return {"name": path.rsplit("/", 1)[-1], "id": "ID", "web_url": f"u/{path}"}


# ---------------------------------------------------------------------------
# report_to_pdf
# ---------------------------------------------------------------------------


def test_report_to_pdf_returns_valid_pdf_bytes():
    data = report_to_pdf(_report())
    assert data[:5] == b"%PDF-"
    doc = pymupdf.open("pdf", data)
    assert doc.page_count >= 1
    text = doc[0].get_text()
    # The report heading and the metric grid made it onto the page.
    assert "Validation report" in text
    assert "case_0" in text
    assert "response_coverage" in text


def test_report_to_pdf_accepts_a_markdown_string():
    data = report_to_pdf("# Hello\n\n- one\n- two\n")
    doc = pymupdf.open("pdf", data)
    assert "Hello" in doc[0].get_text()


def test_report_to_pdf_paginates_a_long_report():
    # Many cases spill the metric table beyond one A4 page.
    data = report_to_pdf(_report(cases=80))
    doc = pymupdf.open("pdf", data)
    assert doc.page_count > 1


# ---------------------------------------------------------------------------
# publish_report_pdf
# ---------------------------------------------------------------------------


def test_publish_uploads_pdf_bytes_to_a_folder_dest():
    client = _StubClient()
    item = publish_report_pdf(_report(), "Reports/Validation", client=client)
    (path, content), = client.writes
    # Folder dest -> a timestamped *.pdf filename appended under it.
    assert path.startswith("Reports/Validation/validation-report-")
    assert path.endswith(".pdf")
    assert content[:5] == b"%PDF-"
    assert item["web_url"] == f"u/{path}"


def test_publish_uses_an_exact_pdf_path_verbatim():
    client = _StubClient()
    publish_report_pdf(_report(), "Reports/run-42.pdf", client=client)
    (path, _), = client.writes
    assert path == "Reports/run-42.pdf"


def test_publish_to_root_when_dest_is_empty():
    client = _StubClient()
    publish_report_pdf(_report(), "", client=client)
    (path, _), = client.writes
    # No folder -> just the timestamped filename at the library root.
    assert path.startswith("validation-report-") and path.endswith(".pdf")
    assert "/" not in path


def test_publish_filename_uses_report_timestamp():
    report = _report()
    client = _StubClient()
    publish_report_pdf(report, "Reports", client=client)
    (path, _), = client.writes
    assert report.created_at.strftime("%Y%m%dT%H%M%SZ") in path


def test_publish_resolves_sharepoint_path_from_config(monkeypatch, tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps({"validation": {"report_sharepoint_path": "Cfg/Reports"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv(app_config.ENV_VAR, str(cfg))
    app_config.clear_cache()
    try:
        client = _StubClient()
        # No explicit path -> config.json validation.report_sharepoint_path.
        publish_report_pdf(_report(), None, client=client)
        (path, _), = client.writes
        assert path.startswith("Cfg/Reports/validation-report-")
    finally:
        app_config.clear_cache()


def test_publish_uses_shared_client_when_none_given(monkeypatch):
    client = _StubClient()
    import validation.pdf as pdf_mod

    # Stand in for app_config.sharepoint.get_sharepoint_client (imported lazily).
    import app_config.sharepoint as sp

    monkeypatch.setattr(sp, "get_sharepoint_client", lambda: client)
    publish_report_pdf(_report(), "Reports/x.pdf")
    assert client.writes and client.writes[0][0] == "Reports/x.pdf"
    assert pdf_mod  # module imported cleanly


# ---------------------------------------------------------------------------
# report_from_verdicts / report_from_thread (inline-validation reporting)
# ---------------------------------------------------------------------------


def _verdict(case_id: str, passed: bool, **extra):
    """A CaseResult dump like out["validation"], plus optional stored-fact keys."""
    result = CaseResult(
        case_id=case_id,
        item_count=1,
        metrics=[
            MetricResult(
                metric="python_syntax",
                score=1.0 if passed else 0.0,
                threshold=1.0,
                passed=passed,
            )
        ],
    )
    return {**result.model_dump(), **extra}


def test_report_from_verdicts_rebuilds_cases_and_recomputes():
    verdicts = [_verdict("c1", True), _verdict("c2", False)]
    report = report_from_verdicts(verdicts, model="claude-sonnet-4-5")
    assert report.model == "claude-sonnet-4-5"
    assert [c.case_id for c in report.results] == ["c1", "c2"]
    # score/passed are computed fields — recomputed from the metrics, not trusted
    # from the (ignored) dumped values.
    assert report.results[0].passed and not report.results[1].passed
    assert not report.passed


def test_report_from_verdicts_ignores_stored_fact_keys():
    # validations_for_thread / get_validation_facts add these; they must not trip
    # the rebuild.
    verdicts = [_verdict("c1", True, item_id="c1", index=1, total=1, ts=123.0)]
    report = report_from_verdicts(verdicts, model="m")
    assert [c.case_id for c in report.results] == ["c1"]


def test_report_from_verdicts_tolerates_lean_verdict_dicts():
    # A minimal verdict (no case_id/item_count) still rebuilds, using item_id.
    report = report_from_verdicts([{"item_id": "x", "metrics": []}], model="m")
    assert report.results[0].case_id == "x"
    assert report.results[0].item_count == 1


def test_report_from_verdicts_renders_to_pdf():
    report = report_from_verdicts([_verdict("etl", True)], model="m")
    assert report_to_pdf(report)[:5] == b"%PDF-"


def test_report_from_thread_reads_stored_verdicts():
    from memory.store import MemoryHub

    from chunker.models import SasChunk, SasChunkKind, SasChunkMetadata
    from validation import LiveValidator

    kv = MemoryHub().kv
    text = "data work.x; set work.a; run;"
    chunk = SasChunk(
        chunk_id="c1",
        source_id="j.sas",
        text=text,
        kind=SasChunkKind.DATA_STEP,
        start_line=1,
        end_line=1,
        start_char=0,
        end_char=len(text),
        metadata=SasChunkMetadata(
            input_datasets=["work.a"], output_datasets=["work.x"]
        ),
    )
    LiveValidator().validate_item(
        chunk,
        "```python\ndf = spark.table('work.a')\n```",
        thread_id="t1",
        kv=kv,
        index=1,
        total=1,
    )
    report = report_from_thread(kv, "t1", model="claude-sonnet-4-5")
    assert [c.case_id for c in report.results] == ["c1"]
    assert "c1" in report.to_markdown()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
