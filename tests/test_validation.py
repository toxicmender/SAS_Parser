"""
Tests for the validation package.

Like the pipeline tests, everything here runs offline: SasLLMPipeline is
constructed with a FakeListChatModel, the LLM judge is a fake returning
canned "SCORE: n" replies, and the MLflow test is skipped when mlflow (the
optional `eval` extra) is not installed.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from chunker.models import SasBatch, SasChunk, SasChunkKind, SasChunkMetadata
from chunker.pipeline import SasLLMPipeline
from validation import (
    DatasetFidelityMetric,
    LLMJudgeMetric,
    PythonSyntaxMetric,
    ReferenceSimilarityMetric,
    RequiredTermsMetric,
    ResponseCoverageMetric,
    ValidationCase,
    ValidationRunner,
    load_cases,
)
from validation.models import CaseRun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_chunk(chunk_id: str, text: str = "data a; run;", **meta_kwargs) -> SasChunk:
    return SasChunk(
        chunk_id=chunk_id,
        source_id="case.sas",
        text=text,
        kind=SasChunkKind.DATA_STEP,
        title=f"Step {chunk_id}",
        start_line=1,
        end_line=3,
        start_char=0,
        end_char=len(text),
        metadata=SasChunkMetadata(**meta_kwargs),
    )


def _mk_run(
    items: list, responses: list[str], **case_kwargs
) -> CaseRun:
    case = ValidationCase(
        case_id=case_kwargs.pop("case_id", "case"),
        sas_source=case_kwargs.pop("sas_source", "data a; run;"),
        **case_kwargs,
    )
    outputs = [
        {"item_id": f"item-{i}", "response": r} for i, r in enumerate(responses)
    ]
    return CaseRun(case=case, items=items, outputs=outputs)


GOOD_RESPONSE = (
    "Translation of the step:\n"
    "```python\n"
    'df = spark.table("sales_raw").filter("amount > 0")\n'
    'df.write.saveAsTable("work.sales_clean")\n'
    "```\n"
    "This produces work.sales_clean from work.sales_raw.\n"
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_response_coverage_counts_empty_and_missing_responses():
    items = [_mk_chunk("c1"), _mk_chunk("c2"), _mk_chunk("c3")]
    run = _mk_run(items, ["ok", "   ", "ok"])  # blank counts as unanswered
    result = ResponseCoverageMetric().evaluate(run)
    assert result.score == pytest.approx(2 / 3)
    assert not result.passed

    short = _mk_run(items, ["ok"])  # fewer outputs than items
    assert ResponseCoverageMetric().evaluate(short).score == pytest.approx(1 / 3)


def test_response_coverage_no_items_is_failure():
    run = _mk_run([], [])
    result = ResponseCoverageMetric().evaluate(run)
    assert result.score == 0.0
    assert not result.passed
    assert not result.skipped


def test_dataset_fidelity_full_bare_and_missing_names():
    chunk = _mk_chunk(
        "c1",
        input_datasets=["work.sales_raw"],
        output_datasets=["work.sales_clean", "work.audit_log"],
    )
    # full name, bare identifier, and one dataset not mentioned at all
    response = 'reads work.sales_raw, writes df.saveAsTable("sales_clean")'
    result = DatasetFidelityMetric().evaluate(_mk_run([chunk], [response]))
    assert result.score == pytest.approx(2 / 3)
    assert "work.audit_log" in result.details


def test_dataset_fidelity_uses_batch_level_io():
    batch = SasBatch(
        batch_id="batch-001",
        chunks=[_mk_chunk("c1")],
        input_datasets=["work.in"],
        output_datasets=["work.out"],
    )
    result = DatasetFidelityMetric().evaluate(
        _mk_run([batch], ["handles work.in and work.out"])
    )
    assert result.score == 1.0
    assert result.passed


def test_dataset_fidelity_skipped_without_datasets():
    result = DatasetFidelityMetric().evaluate(_mk_run([_mk_chunk("c1")], ["hi"]))
    assert result.skipped
    assert result.passed


def test_python_syntax_scores_parse_ratio():
    good = "```python\nx = 1\n```"
    bad = "```python\ndef broken(:\n```"
    result = PythonSyntaxMetric().evaluate(
        _mk_run([_mk_chunk("c1"), _mk_chunk("c2")], [good, bad])
    )
    assert result.score == pytest.approx(0.5)
    assert not result.passed


def test_python_syntax_ignores_non_python_fences():
    response = "```sql\nSELECT 1;\n```\n```python\nx = 1\n```"
    result = PythonSyntaxMetric().evaluate(_mk_run([_mk_chunk("c1")], [response]))
    assert result.score == 1.0


def test_python_syntax_no_code_blocks_scores_zero():
    result = PythonSyntaxMetric().evaluate(
        _mk_run([_mk_chunk("c1")], ["prose only, no code"])
    )
    assert result.score == 0.0
    assert not result.skipped


def test_required_terms_partial_and_skipped():
    items = [_mk_chunk("c1")]
    partial = RequiredTermsMetric().evaluate(
        _mk_run(items, ["uses groupBy here"], required_terms=["groupBy", "filter"])
    )
    assert partial.score == pytest.approx(0.5)
    assert "filter" in partial.details

    skipped = RequiredTermsMetric().evaluate(_mk_run(items, ["anything"]))
    assert skipped.skipped and skipped.passed


def test_reference_similarity_identical_and_disjoint():
    items = [_mk_chunk("c1")]
    same = ReferenceSimilarityMetric().evaluate(
        _mk_run(items, ["df = spark.table('a')"],
                reference_translation="df = spark.table('a')")
    )
    assert same.score == pytest.approx(1.0)

    disjoint = ReferenceSimilarityMetric().evaluate(
        _mk_run(items, ["alpha beta"], reference_translation="gamma delta")
    )
    assert disjoint.score == 0.0

    skipped = ReferenceSimilarityMetric().evaluate(_mk_run(items, ["x"]))
    assert skipped.skipped


# ---------------------------------------------------------------------------
# LLM judge (fake model)
# ---------------------------------------------------------------------------


def test_llm_judge_normalises_scores():
    judge = LLMJudgeMetric(
        llm=FakeListChatModel(responses=["SCORE: 5", "SCORE: 3"])
    )
    result = judge.evaluate(
        _mk_run([_mk_chunk("c1"), _mk_chunk("c2")], ["t1", "t2"])
    )
    # (5-1)/4 = 1.0 and (3-1)/4 = 0.5 -> mean 0.75
    assert result.score == pytest.approx(0.75)
    assert result.passed  # default threshold 0.6


def test_llm_judge_unparseable_reply_scores_zero():
    judge = LLMJudgeMetric(llm=FakeListChatModel(responses=["no idea"]))
    result = judge.evaluate(_mk_run([_mk_chunk("c1")], ["t1"]))
    assert result.score == 0.0
    assert "unparseable" in result.details


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------


def test_load_cases_inline_sas_path_and_list_files(tmp_path):
    (tmp_path / "a_inline.json").write_text(
        json.dumps({"case_id": "inline", "sas_source": "data a; run;"}),
        encoding="utf-8",
    )
    (tmp_path / "prog.sas").write_text("data b; run;", encoding="utf-8")
    (tmp_path / "b_path.json").write_text(
        json.dumps({"case_id": "from_path", "sas_path": "prog.sas"}),
        encoding="utf-8",
    )
    (tmp_path / "c_list.json").write_text(
        json.dumps(
            [
                {"case_id": "l1", "sas_source": "data c; run;"},
                {"case_id": "l2", "sas_source": "data d; run;"},
            ]
        ),
        encoding="utf-8",
    )

    cases = load_cases(tmp_path)
    assert [c.case_id for c in cases] == ["inline", "from_path", "l1", "l2"]
    assert cases[1].sas_source == "data b; run;"


def test_load_cases_rejects_duplicates_and_both_sources(tmp_path):
    (tmp_path / "dup.json").write_text(
        json.dumps(
            [
                {"case_id": "x", "sas_source": "data a; run;"},
                {"case_id": "x", "sas_source": "data b; run;"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(tmp_path)

    both = tmp_path / "both"
    both.mkdir()
    (both / "case.json").write_text(
        json.dumps(
            {"case_id": "y", "sas_source": "data a; run;", "sas_path": "p.sas"}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one"):
        load_cases(both)


def test_bundled_sample_cases_load():
    cases = load_cases(pathlib.Path(__file__).resolve().parents[1] / "validation" / "cases")
    assert {c.case_id for c in cases} == {"macro_flow", "simple_etl"}


# ---------------------------------------------------------------------------
# Runner end-to-end (fake LLM, in-memory store — no network, no Spark)
# ---------------------------------------------------------------------------


def _pipeline(responses: list[str]) -> SasLLMPipeline:
    return SasLLMPipeline(llm=FakeListChatModel(responses=responses))


def test_runner_end_to_end_passing_case():
    case = ValidationCase(
        case_id="etl",
        sas_source=(
            "data work.sales_clean;\n  set work.sales_raw;\nrun;\n\n"
            "proc sql;\n  create table work.sales_summary as\n"
            "  select * from work.sales_clean;\nquit;\n"
        ),
        required_terms=["spark"],
    )
    response = (
        "```python\n"
        'clean = spark.table("work.sales_raw")\n'
        'clean.write.saveAsTable("work.sales_clean")\n'
        'summary = spark.table("work.sales_clean")\n'
        'summary.write.saveAsTable("work.sales_summary")\n'
        "```\n"
        "Covers work.sales_raw, work.sales_clean, work.sales_summary.\n"
    )
    report = ValidationRunner(_pipeline([response] * 4)).run([case])

    assert report.passed
    assert report.score > 0.9
    (result,) = report.results
    assert result.item_count >= 1
    by_name = {m.metric: m for m in result.metrics}
    assert by_name["response_coverage"].score == 1.0
    assert by_name["dataset_fidelity"].passed
    assert by_name["python_syntax"].score == 1.0
    assert by_name["required_terms"].score == 1.0
    assert by_name["reference_similarity"].skipped


def test_runner_flags_prose_only_responses():
    case = ValidationCase(case_id="prose", sas_source="data work.a; x=1; run;")
    report = ValidationRunner(
        _pipeline(["I would translate this to a DataFrame, mentioning work.a."])
    ).run([case])

    assert not report.passed
    by_name = {m.metric: m for m in report.results[0].metrics}
    assert by_name["python_syntax"].score == 0.0
    assert by_name["response_coverage"].passed


def test_runner_multiple_independent_items_get_distinct_responses():
    # Two unrelated DATA steps -> two singleton items, answered in order.
    case = ValidationCase(
        case_id="two_items",
        sas_source="data work.a; x=1; run;\n\ndata work.b; y=2; run;\n",
    )
    r1 = "```python\na = 1\n```\nwork.a done."
    r2 = "```python\nb = 2\n```\nwork.b done."
    report = ValidationRunner(_pipeline([r1, r2])).run([case])

    (result,) = report.results
    assert result.item_count == 2
    by_name = {m.metric: m for m in result.metrics}
    assert by_name["dataset_fidelity"].score == 1.0
    assert report.passed


def test_report_markdown_lists_cases_and_metrics():
    case = ValidationCase(case_id="md_case", sas_source="data work.a; run;")
    report = ValidationRunner(_pipeline([GOOD_RESPONSE])).run([case])
    md = report.to_markdown()
    assert "md_case" in md
    assert "python_syntax" in md
    assert ("PASSED" in md) or ("FAILED" in md)


# ---------------------------------------------------------------------------
# Spark-backed tracking
# ---------------------------------------------------------------------------


def test_report_rows_flatten_run_case_and_metric_levels():
    # Row shaping is pure Python — verified without a JVM.
    from datetime import datetime, timezone

    from validation.tracking import _report_rows

    case = ValidationCase(case_id="tracked", sas_source="data work.a; run;")
    report = ValidationRunner(_pipeline([GOOD_RESPONSE])).run([case])

    logged_at = datetime.now(timezone.utc)
    rows = _report_rows(report, "run-1", logged_at)

    assert len(rows) == len(report.results[0].metrics)
    assert {r["metric"] for r in rows} == {
        m.metric for m in report.results[0].metrics
    }
    for row in rows:
        assert row["run_id"] == "run-1"
        assert row["logged_at"] is logged_at
        assert row["case_id"] == "tracked"
        assert row["case_count"] == 1
        assert row["run_score"] == pytest.approx(report.score)


def test_instructions_fingerprint_flows_into_report_and_rows():
    from datetime import datetime, timezone

    from validation.tracking import _report_rows

    case = ValidationCase(case_id="fp", sas_source="data work.a; run;")

    with_rules = SasLLMPipeline(
        llm=FakeListChatModel(responses=[GOOD_RESPONSE]),
        user_instructions="## Rules\nAlways emit a risk table.",
    )
    report = ValidationRunner(with_rules).run([case])
    assert report.instructions_fingerprint == with_rules.instructions_fingerprint
    rows = _report_rows(report, "run-1", datetime.now(timezone.utc))
    assert all(
        r["instructions_fingerprint"] == report.instructions_fingerprint
        for r in rows
    )

    # No instructions active -> None recorded, so runs stay distinguishable.
    bare = ValidationRunner(_pipeline([GOOD_RESPONSE])).run([case])
    assert bare.instructions_fingerprint is None


def test_resolve_target_prefers_table_over_path():
    from validation.tracking import DEFAULT_PATH, _resolve_target

    assert _resolve_target("cat.schema.runs", "ignored") == (
        "table",
        "cat.schema.runs",
    )
    assert _resolve_target(None, "some/dir") == ("path", "some/dir")
    assert _resolve_target(None, None) == ("path", DEFAULT_PATH)


def test_log_report_appends_parquet_rows_via_spark(tmp_path):
    # Needs a JVM for a local Spark session; skipped where none is available
    # (same standing caveat as memory's Delta backend — see Architecture.md).
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    try:
        spark = (
            SparkSession.builder.master("local[1]")
            .appName("test_validation_tracking")
            .getOrCreate()
        )
    except Exception as exc:  # no JVM / winutils on this machine
        pytest.skip(f"local Spark session unavailable: {exc}")

    from validation import load_runs, log_report

    case = ValidationCase(case_id="tracked", sas_source="data work.a; run;")
    report = ValidationRunner(_pipeline([GOOD_RESPONSE])).run([case])

    target = str(tmp_path / "validation_runs")
    run_id = log_report(report, spark=spark, path=target)

    df = load_runs(spark=spark, path=target)
    rows = df.collect()
    assert {r.run_id for r in rows} == {run_id}
    assert {r.metric for r in rows} == {
        m.metric for m in report.results[0].metrics
    }
    assert all(r.case_id == "tracked" for r in rows)
