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
from memory.store import MemoryHub

from chunker.models import SasBatch, SasChunk, SasChunkKind, SasChunkMetadata
from chunker.pipeline import SasLLMPipeline
from validation import (
    DatasetFidelityMetric,
    Evaluator,
    EvaluationRun,
    LiveValidator,
    LLMJudgeMetric,
    PythonSyntaxMetric,
    ReferenceSimilarityMetric,
    RequiredTermsMetric,
    ResponseCoverageMetric,
    ValidationCase,
    ValidationRunner,
    load_cases,
    run_from_thread,
    validate_thread,
    validate_transcript,
    validations_for_thread,
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
    # run_id is derived from `case` by CaseRun's before-validator, which pyright
    # can't see (mirrors validation/runner.py).
    return CaseRun(case=case, items=items, outputs=outputs)  # pyright: ignore[reportCallIssue]


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
# Evaluator on case-free EvaluationRuns
# ---------------------------------------------------------------------------


def test_evaluator_scores_run_with_items():
    chunk = _mk_chunk("c1", input_datasets=["work.in"], output_datasets=["work.out"])
    run = EvaluationRun(
        run_id="manual",
        items=[chunk],
        outputs=[{"item_id": "c1", "response": GOOD_RESPONSE + " work.in work.out"}],
    )
    result = Evaluator().evaluate(run)
    assert result.case_id == "manual"
    assert result.item_count == 1
    by_name = {m.metric: m for m in result.metrics}
    assert by_name["response_coverage"].score == 1.0
    assert by_name["dataset_fidelity"].score == 1.0
    assert by_name["python_syntax"].score == 1.0


def test_evaluator_scores_run_without_items():
    run = EvaluationRun(
        run_id="itemless",
        prompts=["translate this step", "translate that step"],
        outputs=[
            {"item_id": "turn-1", "response": GOOD_RESPONSE},
            {"item_id": "turn-2", "response": ""},
        ],
        required_terms=["saveAsTable"],
    )
    result = Evaluator().evaluate(run)
    assert result.item_count == 2  # falls back to the prompt count
    by_name = {m.metric: m for m in result.metrics}
    assert by_name["response_coverage"].score == pytest.approx(0.5)
    assert by_name["dataset_fidelity"].skipped  # no item metadata
    assert by_name["required_terms"].score == 1.0


def test_case_run_derives_expectations_from_case():
    run = _mk_run(
        [_mk_chunk("c1")],
        ["x"],
        case_id="derived",
        required_terms=["groupBy"],
        reference_translation="ref",
    )
    assert run.run_id == "derived"
    assert run.required_terms == ["groupBy"]
    assert run.reference_translation == "ref"


# ---------------------------------------------------------------------------
# Live conversations: threads and transcripts
# ---------------------------------------------------------------------------


def test_validate_thread_post_hoc_from_pipeline():
    # Run a conversation first, then score it without re-running anything.
    pipeline = _pipeline([GOOD_RESPONSE] * 4)
    thread_id = "live::etl"
    pipeline.run_text(
        "data work.sales_clean;\n  set work.sales_raw;\nrun;\n",
        source_id="etl.sas",
        thread_id=thread_id,
    )

    result = validate_thread(
        pipeline, thread_id, required_terms=["saveAsTable"]
    )
    assert result.case_id == thread_id
    assert result.item_count >= 1
    by_name = {m.metric: m for m in result.metrics}
    assert by_name["response_coverage"].score == 1.0
    assert by_name["python_syntax"].score == 1.0
    assert by_name["required_terms"].score == 1.0
    assert by_name["dataset_fidelity"].skipped  # items are not persisted


def test_run_from_thread_labels_outputs_with_run_fact_item_ids():
    pipeline = _pipeline([GOOD_RESPONSE] * 4)
    thread_id = "live::ids"
    outputs = pipeline.run_text(
        "data work.a; x=1; run;", source_id="ids.sas", thread_id=thread_id
    )

    run = run_from_thread(pipeline, thread_id)
    assert [o["item_id"] for o in run.outputs] == [o["item_id"] for o in outputs]
    assert run.prompts  # human sides reconstructed
    assert run.responses == [o["response"] for o in outputs]


def test_run_from_thread_empty_thread_raises():
    pipeline = _pipeline([GOOD_RESPONSE])
    with pytest.raises(ValueError, match="no messages"):
        run_from_thread(pipeline, "never-ran")


def test_validate_transcript_pairs_and_messages():
    result = validate_transcript(
        [("translate step 1", GOOD_RESPONSE), ("translate step 2", GOOD_RESPONSE)],
        run_id="pairs",
        required_terms=["spark"],
    )
    assert result.case_id == "pairs"
    by_name = {m.metric: m for m in result.metrics}
    assert by_name["response_coverage"].score == 1.0
    assert by_name["required_terms"].score == 1.0
    assert by_name["dataset_fidelity"].skipped

    from langchain_core.messages import AIMessage, HumanMessage

    msg_result = validate_transcript(
        [HumanMessage("translate step 1"), AIMessage(GOOD_RESPONSE)],
        run_id="messages",
    )
    assert msg_result.case_id == "messages"
    by_name = {m.metric: m for m in msg_result.metrics}
    assert by_name["response_coverage"].score == 1.0
    assert by_name["python_syntax"].score == 1.0


def test_validate_transcript_unanswered_turn_lowers_coverage():
    from langchain_core.messages import AIMessage, HumanMessage

    result = validate_transcript(
        [
            HumanMessage("q1"),
            AIMessage(GOOD_RESPONSE),
            HumanMessage("q2, still awaiting a reply"),
        ]
    )
    by_name = {m.metric: m for m in result.metrics}
    assert by_name["response_coverage"].score == pytest.approx(0.5)
    assert not by_name["response_coverage"].passed


def test_validate_transcript_empty_raises():
    with pytest.raises(ValueError, match="empty transcript"):
        validate_transcript([])


def test_llm_judge_falls_back_to_prompts_without_items():
    judge = LLMJudgeMetric(llm=FakeListChatModel(responses=["SCORE: 5"]))
    run = EvaluationRun(
        run_id="t", prompts=["translate: data a; run;"],
        outputs=[{"item_id": "turn-1", "response": "df = spark.table('a')"}],
    )
    result = judge.evaluate(run)
    assert result.score == 1.0
    assert not result.skipped


def test_llm_judge_skips_without_any_source_context():
    judge = LLMJudgeMetric(llm=FakeListChatModel(responses=["SCORE: 5"]))
    run = EvaluationRun(
        run_id="t", outputs=[{"item_id": "turn-1", "response": "code"}]
    )
    result = judge.evaluate(run)
    assert result.skipped
    assert result.passed


# ---------------------------------------------------------------------------
# Inline validation (during the run)
# ---------------------------------------------------------------------------


def test_live_validator_scores_single_item_and_stores_in_kv():
    from memory.store import MemoryHub

    kv = MemoryHub().kv
    chunk = _mk_chunk(
        "c1", input_datasets=["work.sales_raw"], output_datasets=["work.sales_clean"]
    )
    result = LiveValidator().validate_item(
        chunk, GOOD_RESPONSE, thread_id="t1", kv=kv, index=1, total=1
    )
    # One item carries its own metadata, so dataset_fidelity scores it
    # precisely rather than skipping the way a metadata-less thread does.
    by_name = {m.metric: m for m in result.metrics}
    assert by_name["dataset_fidelity"].score == 1.0
    assert not by_name["dataset_fidelity"].skipped
    assert by_name["python_syntax"].score == 1.0
    assert result.passed

    # The verdict is stored in the conversation KV, keyed per item.
    stored = validations_for_thread(kv, "t1")
    assert [f["item_id"] for f in stored] == ["c1"]
    assert stored[0]["passed"] is True
    assert stored[0]["index"] == 1
    assert stored[0]["score"] == pytest.approx(result.score)


def _validated_pipeline(responses, validator=None):
    return SasLLMPipeline(
        llm=FakeListChatModel(responses=responses),
        validator=validator or LiveValidator(),
    )


def test_inline_validation_runs_per_item_and_persists_to_thread():
    # Two independent DATA steps -> two items, each scored as it is answered.
    source = "data work.a; x=1; run;\n\ndata work.b; y=2; run;\n"
    r1 = "```python\na = 1\n```\nwork.a done."
    r2 = "```python\nb = 2\n```\nwork.b done."
    pipeline = _validated_pipeline([r1, r2])
    thread_id = "run::inline"
    outputs = pipeline.run_text(source, source_id="inline.sas", thread_id=thread_id)

    # A verdict per item, both on the output dicts and in conversation memory.
    assert all(o["validation"] is not None for o in outputs)
    assert all(o["validation"]["passed"] for o in outputs)

    facts = pipeline.get_validation_facts(thread_id)
    assert len(facts) == 2
    assert [f["index"] for f in facts] == [1, 2]
    assert all(f["passed"] for f in facts)
    # Filed beside the run facts, on the same thread, one-to-one.
    run_ids = {f["item_id"] for f in pipeline.get_run_facts(thread_id)}
    assert {f["item_id"] for f in facts} == run_ids


def test_inline_validation_records_failing_item_without_aborting():
    # Prose-only response fails python_syntax but the run still completes and
    # the (failing) verdict is stored -- observe-only, no retry/abort.
    pipeline = _validated_pipeline(["I would translate work.a to a DataFrame."])
    thread_id = "run::failing"
    outputs = pipeline.run_text(
        "data work.a; x=1; run;", source_id="f.sas", thread_id=thread_id
    )

    assert len(outputs) == 1  # run completed
    (fact,) = pipeline.get_validation_facts(thread_id)
    assert fact["passed"] is False
    by_name = {m["metric"]: m for m in fact["metrics"]}
    assert by_name["python_syntax"]["score"] == 0.0


def test_inline_validation_swallows_validator_errors():
    class _BrokenValidator:
        def validate_item(self, *args, **kwargs):
            raise RuntimeError("boom")

    pipeline = _validated_pipeline(
        [GOOD_RESPONSE], validator=_BrokenValidator()
    )
    thread_id = "run::broken"
    # A scoring bug must never break a translation run.
    outputs = pipeline.run_text(
        "data work.a; run;", source_id="b.sas", thread_id=thread_id
    )
    assert outputs[0]["response"] == GOOD_RESPONSE
    assert outputs[0]["validation"] is None
    assert pipeline.get_validation_facts(thread_id) == []
    # The run itself still succeeded and recorded its run fact.
    assert pipeline.get_run_facts(thread_id)[0]["status"] == "ok"


def test_no_validator_leaves_validation_absent_and_facts_empty():
    pipeline = _pipeline([GOOD_RESPONSE])
    thread_id = "run::novalidator"
    outputs = pipeline.run_text(
        "data work.a; run;", source_id="n.sas", thread_id=thread_id
    )
    assert outputs[0]["validation"] is None
    assert pipeline.get_validation_facts(thread_id) == []


def test_resume_recovers_stored_validation_verdicts():
    # A verdict written before an interruption is stored per item; resuming
    # the same thread recovers it onto the skipped item's output, shaped
    # identically to a freshly-scored one.
    pipeline = _validated_pipeline([GOOD_RESPONSE, GOOD_RESPONSE])
    c1 = _mk_chunk(
        "c1", input_datasets=["work.sales_raw"], output_datasets=["work.sales_clean"]
    )
    c2 = _mk_chunk("c2")
    thread_id = "run::resume-val"

    # First attempt "crashes" after item 1: only c1 processed and scored.
    pipeline._process(items=[c1], diagnostics=[], thread_id=thread_id)
    assert len(pipeline.get_validation_facts(thread_id)) == 1

    outputs = pipeline._process(
        items=[c1, c2], diagnostics=[], thread_id=thread_id, resume=True
    )

    assert outputs[0]["skipped"] is True
    assert outputs[0]["validation"] is not None
    assert outputs[0]["validation"]["passed"] is True
    # Recovered verdict is the bare CaseResult dump: bookkeeping keys stripped.
    assert "ts" not in outputs[0]["validation"]
    assert "index" not in outputs[0]["validation"]
    # Freshly scored item is shaped identically to the recovered one.
    assert outputs[1]["skipped"] is False
    assert set(outputs[0]["validation"]) == set(outputs[1]["validation"])


def test_resume_without_stored_verdict_leaves_validation_none():
    # A run that had no validator on the first attempt has no stored verdict;
    # resuming must not invent one for the skipped item.
    first = _pipeline([GOOD_RESPONSE])  # no validator
    thread_id = "run::resume-noverdict"
    first._process(items=[_mk_chunk("c1")], diagnostics=[], thread_id=thread_id)

    # Same store, now with a validator attached, resumes the thread.
    resumed = SasLLMPipeline(
        llm=FakeListChatModel(responses=[GOOD_RESPONSE]),
        memory=first._memory,
        validator=LiveValidator(),
    )
    outputs = resumed._process(
        items=[_mk_chunk("c1"), _mk_chunk("c2")],
        diagnostics=[],
        thread_id=thread_id,
        resume=True,
    )
    assert outputs[0]["skipped"] is True
    assert outputs[0]["validation"] is None  # nothing stored to recover
    assert outputs[1]["validation"] is not None  # newly scored


PROSE_ONLY = "I would translate work.a into a DataFrame, mentioning work.a."


def _retry_pipeline(responses, retries=1, validator=None):
    return SasLLMPipeline(
        llm=FakeListChatModel(responses=responses),
        validator=validator or LiveValidator(),
        validation_retries=retries,
    )


def test_validation_retries_regenerates_failing_item_until_it_passes():
    # First answer is prose-only (fails python_syntax); the retry produces
    # code covering the step's datasets and passes. Only the final, passing
    # turn should persist.
    pipeline = _retry_pipeline([PROSE_ONLY, GOOD_RESPONSE], retries=1)
    thread_id = "run::retry-pass"
    outputs = pipeline.run_text(
        "data work.sales_clean;\n  set work.sales_raw;\nrun;\n",
        source_id="r.sas",
        thread_id=thread_id,
    )

    assert outputs[0]["validation"]["passed"] is True  # final verdict kept
    (fact,) = pipeline.get_validation_facts(thread_id)
    assert fact["passed"] is True
    # Exactly one (human, AI) pair remains — the rolled-back attempt is gone.
    assert len(pipeline.get_thread_messages(thread_id)) == 2
    assert pipeline.get_thread_messages(thread_id)[-1].content == GOOD_RESPONSE
    # The run fact records how many attempts it took.
    assert pipeline.get_run_facts(thread_id)[0]["attempts"] == 2


def test_validation_retries_accepts_last_attempt_when_budget_exhausted():
    # Every attempt fails; the run still completes and keeps the last answer.
    pipeline = _retry_pipeline([PROSE_ONLY, PROSE_ONLY], retries=1)
    thread_id = "run::retry-exhausted"
    outputs = pipeline.run_text(
        "data work.a; x=1; run;", source_id="r.sas", thread_id=thread_id
    )

    assert outputs[0]["validation"]["passed"] is False
    assert len(pipeline.get_thread_messages(thread_id)) == 2  # no duplicate turns
    assert pipeline.get_run_facts(thread_id)[0]["attempts"] == 2


def test_retries_zero_stays_observe_only():
    # The default policy: a failing item is scored and stored but never redone.
    pipeline = _validated_pipeline([PROSE_ONLY])  # validation_retries defaults to 0
    thread_id = "run::observe-only"
    outputs = pipeline.run_text(
        "data work.a; x=1; run;", source_id="r.sas", thread_id=thread_id
    )
    assert outputs[0]["validation"]["passed"] is False
    assert pipeline.get_run_facts(thread_id)[0]["attempts"] == 1


def test_resume_redoes_stored_failing_item_when_retries_enabled():
    # A prior run left item 1 with a FAILING verdict. Resuming with retries
    # enabled rewinds to it and regenerates, rather than skipping it as done.
    c1 = _mk_chunk("c1")
    c2 = _mk_chunk("c2")

    mem = MemoryHub()
    # First attempt (observe-only): item 1 answered with prose (fails), item 2
    # never reached.
    first = SasLLMPipeline(
        llm=FakeListChatModel(responses=[PROSE_ONLY]),
        memory=mem,
        validator=LiveValidator(),
    )
    first._process(items=[c1], diagnostics=[], thread_id="run::redo")
    assert first.get_validation_facts("run::redo")[0]["passed"] is False

    # Resume on the same store: item 1's stored verdict failed, so it is not
    # "done" — it is regenerated (now with code) and the run continues to c2.
    resumed = SasLLMPipeline(
        llm=FakeListChatModel(responses=[GOOD_RESPONSE, GOOD_RESPONSE]),
        memory=mem,
        validator=LiveValidator(),
        validation_retries=1,
    )
    outputs = resumed._process(
        items=[c1, c2], diagnostics=[], thread_id="run::redo", resume=True
    )

    assert outputs[0]["skipped"] is False  # redone, not skipped
    assert outputs[0]["validation"]["passed"] is True
    assert outputs[1]["validation"]["passed"] is True
    facts = {f["item_id"]: f for f in resumed.get_validation_facts("run::redo")}
    assert facts["c1"]["passed"] is True and facts["c2"]["passed"] is True
    # One clean turn pair per item after the rewind+redo.
    assert len(resumed.get_thread_messages("run::redo")) == 4


def test_resume_keeps_passing_prefix_and_redoes_from_first_failure():
    # Items 1 (pass) and 2 (fail) both completed; resume must keep 1 and
    # regenerate from 2 onward.
    c1 = _mk_chunk(
        "c1", input_datasets=["work.sales_raw"], output_datasets=["work.sales_clean"]
    )
    c2 = _mk_chunk("c2")

    mem = MemoryHub()
    # Observe-only prior run: c1 passes, c2 fails, both recorded as they are.
    first = SasLLMPipeline(
        llm=FakeListChatModel(responses=[GOOD_RESPONSE, PROSE_ONLY]),
        memory=mem,
        validator=LiveValidator(),
    )
    first._process(items=[c1, c2], diagnostics=[], thread_id="run::prefix")
    assert first.get_validation_facts("run::prefix")[0]["passed"] is True
    assert first.get_validation_facts("run::prefix")[1]["passed"] is False
    c1_answer = first.get_thread_messages("run::prefix")[1].content

    resumed = SasLLMPipeline(
        llm=FakeListChatModel(responses=[GOOD_RESPONSE]),
        memory=mem,
        validator=LiveValidator(),
        validation_retries=1,
    )
    outputs = resumed._process(
        items=[c1, c2], diagnostics=[], thread_id="run::prefix", resume=True
    )

    assert outputs[0]["skipped"] is True  # passing prefix item kept
    assert outputs[1]["skipped"] is False  # failing item regenerated
    assert outputs[1]["validation"]["passed"] is True
    # Item 1's original answer is untouched; still one pair per item.
    msgs = resumed.get_thread_messages("run::prefix")
    assert len(msgs) == 4
    assert msgs[1].content == c1_answer


def test_fork_run_copies_validation_verdicts_onto_the_branch():
    pipeline = _validated_pipeline([GOOD_RESPONSE, GOOD_RESPONSE, GOOD_RESPONSE])
    c1 = _mk_chunk(
        "c1", input_datasets=["work.sales_raw"], output_datasets=["work.sales_clean"]
    )
    c2 = _mk_chunk("c2")
    pipeline._process(items=[c1, c2], diagnostics=[], thread_id="run::v1")
    assert len(pipeline.get_validation_facts("run::v1")) == 2

    pipeline.fork_run("run::v1", "run::v2", upto_items=1)
    forked = pipeline.get_validation_facts("run::v2")
    assert [f["item_id"] for f in forked] == ["c1"]  # only up to the fork point
    assert forked[0]["passed"] is True

    # Resuming the branch recovers the copied verdict for the skipped item.
    outputs = pipeline._process(
        items=[c1, c2], diagnostics=[], thread_id="run::v2", resume=True
    )
    assert outputs[0]["skipped"] is True
    assert outputs[0]["validation"]["passed"] is True


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
