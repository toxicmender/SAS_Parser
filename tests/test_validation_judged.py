"""
Tests for the LLM-judged validation metrics (deepeval's definitions,
implemented natively in validation/judged.py + rag_metrics.py +
agentic_metrics.py + summarization.py).

Everything runs offline. The judge is a FakeListChatModel scripted with the
exact JSON replies each metric's calls expect, in order — which also exercises
JudgedMetric._ask's prose-JSON fallback, since a fake chat model has no
with_structured_output. Scripts are written as _script(...) lists so a metric's
call sequence is readable at the test site; getting the order wrong shows up as
a score that is off, not as a silent pass.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from chunker.models import SasChunk, SasChunkKind, SasChunkMetadata
from validation import (
    AnalysisSummarizationMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
    EvaluationRun,
    FaithfulnessMetric,
    HallucinationMetric,
    JUDGED_METRIC_NAMES,
    PlanAdherenceMetric,
    PromptAlignmentMetric,
    SummarizationMetric,
    TaskCompletionMetric,
    default_metrics,
    judged_metrics,
)
from validation.judged import (
    JudgedMetric,
    ScoreVerdict,
    Statements,
    Verdicts,
    markdown_section,
    parse_json_reply,
)
from validation.rag_metrics import _weighted_precision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _statements(*items: str) -> str:
    return json.dumps({"statements": list(items)})


def _verdicts(*calls: str) -> str:
    """A verdicts reply from shorthand: "yes", "no", "idk"."""
    return json.dumps(
        {"verdicts": [{"verdict": v, "reason": f"because {v}"} for v in calls]}
    )


def _score(value: float) -> str:
    return json.dumps({"score": value, "reason": "judged"})


def _judge(*responses: str) -> FakeListChatModel:
    return FakeListChatModel(responses=list(responses))


def _chunk(text: str = "data work.a; set work.b; run;") -> SasChunk:
    return SasChunk(
        chunk_id="c1",
        source_id="case.sas",
        text=text,
        kind=SasChunkKind.DATA_STEP,
        title="Step c1",
        start_line=1,
        end_line=3,
        start_char=0,
        end_char=len(text),
        metadata=SasChunkMetadata(),
    )


RESPONSE = (
    "## Analysis\n\nReads work.b and writes work.a.\n\n"
    "## Mapping\n\n- **SET** → spark.table\n\n"
    "## Translation\n\n```python\ndf = spark.table('work.b')\n```\n\n"
    "## Risks\n\n- **P2** — none.\n"
)


def _run(**kwargs) -> EvaluationRun:
    """An EvaluationRun with one unit, overridable field by field."""
    outputs = kwargs.pop(
        "outputs", [{"item_id": "item-0", "response": RESPONSE}]
    )
    return EvaluationRun(run_id="run", outputs=outputs, **kwargs)


# ---------------------------------------------------------------------------
# JudgedMetric plumbing
# ---------------------------------------------------------------------------


def test_parse_json_reply_accepts_bare_fenced_and_embedded_json():
    assert parse_json_reply(Statements, '{"statements": ["a"]}').statements == ["a"]
    fenced = 'Here you go:\n```json\n{"statements": ["b"]}\n```\n'
    assert parse_json_reply(Statements, fenced).statements == ["b"]
    embedded = 'Sure — {"statements": ["c"]} — hope that helps.'
    assert parse_json_reply(Statements, embedded).statements == ["c"]


def test_parse_json_reply_returns_none_for_unusable_text():
    assert parse_json_reply(Statements, "no idea, sorry") is None


def test_ask_prefers_structured_output_when_the_judge_supports_it():
    """An LLMClient-shaped judge is asked for the schema, not for prose."""

    class _StructuredJudge:
        def __init__(self):
            self.structured_calls = 0
            self.invoke_calls = 0

        def supports_structured_output(self, schema):
            return True

        def invoke_structured(self, schema, prompt):
            self.structured_calls += 1
            return {"raw": None, "parsed": Statements(statements=["x"]), "parsing_error": None}

        def invoke(self, prompt):
            self.invoke_calls += 1
            raise AssertionError("should not fall back to invoke")

    judge = _StructuredJudge()
    metric = AnswerRelevancyMetric(judge)
    assert metric._ask(Statements, "prompt").statements == ["x"]
    assert (judge.structured_calls, judge.invoke_calls) == (1, 0)


def test_ask_falls_back_to_the_raw_prose_without_a_second_call():
    """A gateway that won't honour the schema returns parsed=None with prose in
    `raw`; that prose is parsed rather than re-prompted."""

    class _Message:
        content = '{"score": 0.5, "reason": "ok"}'

    class _DemotingJudge:
        def __init__(self):
            self.calls = 0

        def supports_structured_output(self, schema):
            return True

        def invoke_structured(self, schema, prompt):
            self.calls += 1
            return {"raw": _Message(), "parsed": None, "parsing_error": "nope"}

        def invoke(self, prompt):
            raise AssertionError("should not pay for a second call")

    judge = _DemotingJudge()
    verdict = TaskCompletionMetric(judge)._ask(ScoreVerdict, "prompt")
    assert verdict is not None and verdict.score == 0.5
    assert judge.calls == 1


def test_ask_logs_and_returns_none_on_an_unusable_reply(caplog):
    metric = AnswerRelevancyMetric(_judge("I would rather not"))
    with caplog.at_level("WARNING"):
        assert metric._ask(Statements, "prompt") is None
    assert "unusable judge reply" in caplog.text


def test_classify_pads_a_short_verdict_list_with_idk(caplog):
    metric = AnswerRelevancyMetric(_judge(_verdicts("yes")))
    with caplog.at_level("WARNING"):
        verdicts = metric._classify("prompt", 3)
    assert [v.verdict for v in verdicts] == ["yes", "idk", "idk"]
    assert "returned 1 verdict(s) for 3 item(s)" in caplog.text


def test_classify_truncates_an_overlong_verdict_list():
    metric = AnswerRelevancyMetric(_judge(_verdicts("yes", "no", "yes", "yes")))
    assert len(metric._classify("prompt", 2)) == 2


def test_token_usage_is_none_for_a_judge_that_reports_none():
    assert AnswerRelevancyMetric(_judge("{}")).token_usage is None


def test_markdown_section_stops_at_the_next_level_two_heading():
    assert markdown_section(RESPONSE, "Analysis") == "Reads work.b and writes work.a."
    assert "spark.table" in markdown_section(RESPONSE, "Translation")
    # A ### comment heading inside a section does not end it.
    text = "## Translation\n\n### Load\n\n```python\nx = 1\n```\n\n## Risks\n\nnone\n"
    assert "### Load" in markdown_section(text, "Translation")
    assert "none" not in markdown_section(text, "Translation")
    assert markdown_section(RESPONSE, "Nope") == ""


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


def _run_with_context(contexts: list[str], **kwargs) -> EvaluationRun:
    return _run(
        outputs=[
            {
                "item_id": "item-0",
                "response": RESPONSE,
                "prompt": "Translate this DATA step.",
                "retrieval_context": contexts,
            }
        ],
        **kwargs,
    )


def test_faithfulness_scores_the_truthful_claim_ratio():
    judge = _judge(_statements("a", "b", "c", "d"), _verdicts("yes", "yes", "idk", "no"))
    result = FaithfulnessMetric(judge).evaluate(_run_with_context(["SET reads a table."]))
    # yes + idk count as faithful (deepeval's default, ambiguity not penalised).
    assert result.score == pytest.approx(0.75)
    assert not result.skipped
    assert "because no" in result.details


def test_faithfulness_skips_without_retrieval_context():
    result = FaithfulnessMetric(_judge("{}")).evaluate(_run())
    assert result.skipped and result.passed
    assert "no retrieval context" in result.details


def test_faithfulness_treats_an_answer_with_no_claims_as_faithful():
    result = FaithfulnessMetric(_judge(_statements())).evaluate(
        _run_with_context(["guidance"])
    )
    assert result.score == 1.0


def test_faithfulness_scores_zero_when_the_judge_is_unusable():
    result = FaithfulnessMetric(_judge("what?")).evaluate(_run_with_context(["g"]))
    assert result.score == 0.0 and not result.skipped


# ---------------------------------------------------------------------------
# Answer relevancy
# ---------------------------------------------------------------------------


def test_answer_relevancy_scores_the_relevant_statement_ratio():
    judge = _judge(_statements("s1", "s2", "s3", "s4"), _verdicts("yes", "no", "no", "yes"))
    result = AnswerRelevancyMetric(judge).evaluate(
        _run(prompts=["Translate this DATA step."])
    )
    assert result.score == pytest.approx(0.5)


def test_answer_relevancy_falls_back_to_the_item_source_as_input():
    """No prompts and no recorded prompt: the item's SAS text is the input, so
    the metric scores rather than skipping."""
    judge = _judge(_statements("s1"), _verdicts("yes"))
    result = AnswerRelevancyMetric(judge).evaluate(_run(items=[_chunk()]))
    assert not result.skipped and result.score == 1.0


def test_answer_relevancy_skips_without_any_input():
    result = AnswerRelevancyMetric(_judge("{}")).evaluate(_run())
    assert result.skipped and "no request/response pair" in result.details


# ---------------------------------------------------------------------------
# Hallucination (inverted: reports groundedness)
# ---------------------------------------------------------------------------


def test_hallucination_reports_groundedness_and_the_raw_rate():
    run = _run(items=[_chunk()], reference_translation="df = spark.table('work.b')")
    result = HallucinationMetric(_judge(_verdicts("yes", "no"))).evaluate(run)
    assert result.score == pytest.approx(0.5)
    assert "1/2 context(s) contradicted" in result.details
    assert "hallucination rate 0.500" in result.details


def test_hallucination_scores_one_when_nothing_is_contradicted():
    result = HallucinationMetric(_judge(_verdicts("yes"))).evaluate(
        _run(items=[_chunk()])
    )
    assert result.score == 1.0 and result.passed


def test_hallucination_scores_zero_when_everything_is_contradicted():
    result = HallucinationMetric(_judge(_verdicts("no"))).evaluate(
        _run(items=[_chunk()])
    )
    assert result.score == 0.0 and not result.passed


def test_hallucination_counts_an_unusable_reply_as_contradiction():
    result = HallucinationMetric(_judge("hmm")).evaluate(_run(items=[_chunk()]))
    assert result.score == 0.0


def test_hallucination_skips_without_ground_truth_context():
    result = HallucinationMetric(_judge("{}")).evaluate(_run())
    assert result.skipped and "no ground-truth context" in result.details


# ---------------------------------------------------------------------------
# Contextual precision
# ---------------------------------------------------------------------------


def test_weighted_precision_matches_deepevals_formula():
    # Perfect ranking: every relevant node first.
    assert _weighted_precision([True, True, False]) == pytest.approx(1.0)
    # Worst ranking of the same set: (1/2 + 2/3) / 2.
    assert _weighted_precision([False, True, True]) == pytest.approx(
        (1 / 2 + 2 / 3) / 2
    )
    # Alternating: (1/1 + 2/3) / 2.
    assert _weighted_precision([True, False, True]) == pytest.approx((1 + 2 / 3) / 2)
    assert _weighted_precision([False, False]) == 0.0


def test_contextual_precision_penalises_a_bad_ranking():
    run = _run_with_context(["junk", "gold", "gold"], reference_translation="ideal")
    result = ContextualPrecisionMetric(_judge(_verdicts("no", "yes", "yes"))).evaluate(run)
    assert result.score == pytest.approx((1 / 2 + 2 / 3) / 2)


def test_contextual_precision_skips_without_a_reference_translation():
    result = ContextualPrecisionMetric(_judge("{}")).evaluate(
        _run_with_context(["a", "b"])
    )
    assert result.skipped and "no reference translation" in result.details


def test_contextual_precision_skips_without_retrieval_context():
    result = ContextualPrecisionMetric(_judge("{}")).evaluate(
        _run(reference_translation="ideal")
    )
    assert result.skipped and "no retrieval context to rank" in result.details


# ---------------------------------------------------------------------------
# Contextual relevancy
# ---------------------------------------------------------------------------


def test_contextual_relevancy_scores_the_relevant_statement_ratio():
    judge = _judge(_statements("f1", "f2", "f3", "f4"), _verdicts("yes", "yes", "yes", "no"))
    result = ContextualRelevancyMetric(judge).evaluate(
        _run_with_context(["INTNX shifts dates.", "PROC SORT orders rows."])
    )
    assert result.score == pytest.approx(0.75)


def test_contextual_relevancy_skips_without_retrieval_context():
    result = ContextualRelevancyMetric(_judge("{}")).evaluate(
        _run(prompts=["translate"])
    )
    assert result.skipped


# ---------------------------------------------------------------------------
# Prompt alignment
# ---------------------------------------------------------------------------


def test_prompt_alignment_defaults_to_the_system_prompt_contract():
    from validation import DEFAULT_PROMPT_INSTRUCTIONS

    n = len(DEFAULT_PROMPT_INSTRUCTIONS)
    judge = _judge(_verdicts(*(["yes"] * (n - 1) + ["no"])))
    result = PromptAlignmentMetric(judge).evaluate(_run(prompts=["translate"]))
    assert result.score == pytest.approx((n - 1) / n)
    assert f"{n} instruction(s)" in result.details


def test_prompt_alignment_prefers_the_explicit_constructor_list():
    metric = PromptAlignmentMetric(
        _judge(_verdicts("yes", "no")),
        prompt_instructions=["Use groupBy", "Never use collect()"],
    )
    result = metric.evaluate(_run(prompt_instructions=["ignored"]))
    assert result.score == pytest.approx(0.5)
    assert "2 instruction(s)" in result.details


def test_prompt_alignment_uses_the_runs_instructions_over_the_default():
    metric = PromptAlignmentMetric(_judge(_verdicts("yes")))
    result = metric.evaluate(_run(prompt_instructions=["Emit a notebook cell"]))
    assert result.score == 1.0
    assert "1 instruction(s)" in result.details


def test_prompt_alignment_skips_without_responses():
    metric = PromptAlignmentMetric(_judge("{}"))
    result = metric.evaluate(
        EvaluationRun(run_id="r", outputs=[{"item_id": "i", "response": "  "}])
    )
    assert result.skipped


# ---------------------------------------------------------------------------
# Plan adherence
# ---------------------------------------------------------------------------


def test_plan_adherence_scores_analysis_against_the_code_that_followed():
    result = PlanAdherenceMetric(_judge(_score(0.9))).evaluate(
        _run(prompts=["translate"])
    )
    assert result.score == pytest.approx(0.9)
    assert "1 unit(s) with a stated plan" in result.details


def test_plan_adherence_reads_the_structured_document_when_present():
    document = {
        "analysis": "Read work.b, filter, write work.a.",
        "mapping": [{"sas_construct": "SET", "equivalent": "spark.table", "difference": ""}],
        "cells": [{"kind": "code", "source": "df = spark.table('work.b')", "comment": "Load"}],
    }
    run = _run(
        outputs=[
            {
                "item_id": "item-0",
                "response": "(rendered elsewhere)",
                "document": document,
                "prompt": "translate",
            }
        ]
    )
    result = PlanAdherenceMetric(_judge(_score(1.0))).evaluate(run)
    assert result.score == 1.0 and not result.skipped


def test_plan_adherence_skips_when_no_plan_was_stated():
    run = _run(
        outputs=[{"item_id": "i", "response": "```python\nx = 1\n```", "prompt": "go"}]
    )
    result = PlanAdherenceMetric(_judge("{}")).evaluate(run)
    assert result.skipped and result.passed
    assert "no stated plan" in result.details


def test_plan_adherence_clamps_an_out_of_range_judge_score():
    result = PlanAdherenceMetric(_judge(_score(4.2))).evaluate(_run(prompts=["t"]))
    assert result.score == 1.0


# ---------------------------------------------------------------------------
# Task completion
# ---------------------------------------------------------------------------


def test_task_completion_scores_the_whole_run_once():
    judge = _judge(_score(0.8))
    run = _run(
        items=[_chunk()],
        outputs=[
            {"item_id": "a", "response": RESPONSE},
            {"item_id": "b", "response": RESPONSE},
        ],
    )
    result = TaskCompletionMetric(judge).evaluate(run)
    assert result.score == pytest.approx(0.8)
    # One call for two responses — a per-item average would have used two.
    assert judge.i == 0
    assert "whole-run outcome" in result.details


def test_task_completion_fails_an_empty_run():
    run = EvaluationRun(run_id="r", outputs=[{"item_id": "i", "response": ""}])
    result = TaskCompletionMetric(_judge("{}")).evaluate(run)
    assert result.score == 0.0 and not result.skipped


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


def test_summarization_takes_the_minimum_of_alignment_and_coverage():
    judge = _judge(
        _statements("c1", "c2", "c3", "c4"),      # claims in the summary
        _verdicts("yes", "yes", "yes", "yes"),    # alignment: 1.00
        _statements("q1", "q2", "q3", "q4"),      # coverage questions
        _verdicts("yes", "yes", "no", "idk"),     # coverage: 0.50
    )
    run = _run(summary="work.a comes from work.b.", summary_source="long transcript")
    result = SummarizationMetric(judge).evaluate(run)
    assert result.score == pytest.approx(0.5)
    assert "alignment 1.000, coverage 0.500" in result.details
    assert "unanswered: q3" in result.details


def test_summarization_coverage_counts_idk_against_the_summary():
    """A question the summary cannot answer is a dropped key point, unlike an
    ambiguous claim in the alignment half."""
    judge = _judge(
        _statements("c1"),
        _verdicts("idk"),                     # alignment: idk still counts as ok
        _statements("q1", "q2"),
        _verdicts("yes", "idk"),              # coverage: 0.50
    )
    run = _run(summary="s", summary_source="src")
    assert SummarizationMetric(judge).evaluate(run).score == pytest.approx(0.5)


def test_summarization_uses_supplied_assessment_questions_and_skips_generation():
    judge = _judge(
        _statements("c1"),
        _verdicts("yes"),
        _verdicts("yes", "no"),               # answered directly — no question call
    )
    metric = SummarizationMetric(
        judge, assessment_questions=["Is work.a mentioned?", "Is the filter kept?"]
    )
    run = _run(summary="s", summary_source="src")
    assert metric.evaluate(run).score == pytest.approx(0.5)


def test_summarization_skips_without_a_rolling_summary():
    result = SummarizationMetric(_judge("{}")).evaluate(_run())
    assert result.skipped and "no rolling summary" in result.details


def test_analysis_summarization_scores_the_analysis_against_the_sas_source():
    judge = _judge(
        _statements("c1", "c2"),
        _verdicts("yes", "no"),               # alignment: 0.50
        _statements("q1", "q2"),
        _verdicts("yes", "yes"),              # coverage: 1.00
    )
    result = AnalysisSummarizationMetric(judge).evaluate(_run(items=[_chunk()]))
    assert result.score == pytest.approx(0.5)
    assert "mean over 1 item(s)" in result.details


def test_analysis_summarization_skips_without_item_metadata():
    result = AnalysisSummarizationMetric(_judge("{}")).evaluate(_run())
    assert result.skipped and "no Analysis section" in result.details


# ---------------------------------------------------------------------------
# The suite factory
# ---------------------------------------------------------------------------


def test_default_metrics_stays_deterministic_and_judge_free():
    names = [m.name for m in default_metrics()]
    assert names == [
        "response_coverage",
        "dataset_fidelity",
        "python_syntax",
        "required_terms",
        "reference_similarity",
    ]
    assert not any(isinstance(m, JudgedMetric) for m in default_metrics())


def test_judged_metrics_builds_every_metric_by_default():
    metrics = judged_metrics(_judge("{}"))
    assert [m.name for m in metrics] == list(JUDGED_METRIC_NAMES)


def test_judged_metrics_include_filters_and_keeps_canonical_order():
    metrics = judged_metrics(
        _judge("{}"), include=["task_completion", "faithfulness"]
    )
    assert [m.name for m in metrics] == ["faithfulness", "task_completion"]


def test_judged_metrics_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown judged metric"):
        judged_metrics(_judge("{}"), include=["faithfullness"])


def test_judge_spend_reaches_the_report_through_judged_metrics():
    """ValidationRunner sums token_usage across any metric exposing it, so a
    judged metric's cost lands in the report with no extra wiring."""
    from llm_client import TokenUsage
    from validation import Evaluator
    from validation.runner import ValidationRunner

    class _UsageJudge:
        usage = TokenUsage(calls=3, input_tokens=100, output_tokens=20)

        def invoke(self, prompt):
            raise AssertionError("not called in this test")

    metrics = judged_metrics(_UsageJudge(), include=["faithfulness", "hallucination"])
    # The real property, without standing up a pipeline to construct a runner.
    runner = ValidationRunner.__new__(ValidationRunner)
    runner._evaluator = Evaluator(metrics=metrics)
    # Both metrics expose the same shared judge, so each contributes its usage.
    assert runner.judge_token_usage.calls == 6


# ---------------------------------------------------------------------------
# Verdict schema defaults
# ---------------------------------------------------------------------------


def test_verdicts_schema_defaults_to_idk_for_a_malformed_entry():
    parsed = parse_json_reply(Verdicts, json.dumps({"verdicts": [{}]}))
    assert parsed.verdicts[0].verdict == "idk"
