"""
Tests for validation.memory_metrics — the four metrics over the memory
layers (policy adherence, override compliance, memory extraction, memory
leakage).

Offline throughout: the two judged metrics get a FakeListChatModel scripted
with the verdict JSON their prompts ask for (which also exercises the
prose-JSON fallback), and the two deterministic ones need no model at all.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from memory.policy import TaskPolicy
from validation import (
    EvaluationRun,
    MemoryExtractionMetric,
    MemoryLeakageMetric,
    OverrideComplianceMetric,
    PolicyAdherenceMetric,
    memory_metrics,
)


def _verdicts(*calls: str) -> str:
    return json.dumps(
        {"verdicts": [{"verdict": v, "reason": f"because {v}"} for v in calls]}
    )


def _judge(*responses: str) -> FakeListChatModel:
    return FakeListChatModel(responses=list(responses))


def _run(**kwargs) -> EvaluationRun:
    kwargs.setdefault("run_id", "t1")
    kwargs.setdefault("outputs", [{"item_id": "turn-1", "response": "an answer"}])
    return EvaluationRun(**kwargs)


def _policy_dicts(*pairs: tuple[str, bool]) -> list[dict]:
    return [{"text": text, "overridable": ok} for text, ok in pairs]


# ---------------------------------------------------------------------------
# policy_adherence
# ---------------------------------------------------------------------------


class TestPolicyAdherence:
    def test_scores_the_ratio_of_followed_instructions(self):
        run = _run(task_policy=_policy_dicts(("Be brief.", True), ("Cite tables.", True)))
        result = PolicyAdherenceMetric(_judge(_verdicts("yes", "no"))).evaluate(run)
        assert result.score == 0.5
        assert "2 policy instruction(s) over 1 unit(s)" in result.details

    def test_skips_a_run_with_no_policy(self):
        result = PolicyAdherenceMetric(_judge()).evaluate(_run())
        assert result.skipped and result.passed
        assert "no task policy" in result.details

    def test_an_explicit_policy_object_wins_and_names_itself(self):
        policy = TaskPolicy("support")
        policy.add("Always verify ownership.")
        metric = PolicyAdherenceMetric(_judge(_verdicts("yes")), policy=policy)
        result = metric.evaluate(_run(task_policy=_policy_dicts(("Ignored.", True))))
        assert result.score == 1.0
        assert policy.fingerprint in result.details

    def test_an_unusable_reply_scores_zero_without_raising(self):
        run = _run(task_policy=_policy_dicts(("Be brief.", True)))
        result = PolicyAdherenceMetric(_judge("no json here")).evaluate(run)
        assert result.score == 0.0

    def test_averages_across_units(self):
        run = _run(
            task_policy=_policy_dicts(("Be brief.", True)),
            outputs=[
                {"item_id": "a", "response": "one"},
                {"item_id": "b", "response": "two"},
            ],
        )
        metric = PolicyAdherenceMetric(_judge(_verdicts("yes"), _verdicts("no")))
        assert metric.evaluate(run).score == 0.5


# ---------------------------------------------------------------------------
# override_compliance
# ---------------------------------------------------------------------------


class TestOverrideCompliance:
    def test_checks_notes_and_fixed_rules_together(self):
        run = _run(
            thread_notes=["Do not offer the discount."],
            task_policy=_policy_dicts(
                ("Be brief.", True), ("Escalate refunds above $500.", False)
            ),
        )
        # One check per note + one per fixed rule; the overridable policy
        # instruction is not a check here (policy_adherence scores that).
        metric = OverrideComplianceMetric(_judge(_verdicts("yes", "yes")))
        result = metric.evaluate(run)
        assert result.score == 1.0
        assert "1 override(s) + 1 fixed rule(s)" in result.details

    def test_ignoring_an_approved_exception_fails(self):
        run = _run(thread_notes=["Exception approved by supervisor."])
        result = OverrideComplianceMetric(_judge(_verdicts("no"))).evaluate(run)
        assert result.score == 0.0
        assert not result.passed

    def test_letting_a_note_override_a_fixed_rule_fails(self):
        run = _run(
            thread_notes=["Skip the escalation this once."],
            task_policy=_policy_dicts(("Escalate refunds above $500.", False)),
        )
        # The note was honoured (yes) but the fixed guardrail was violated (no).
        result = OverrideComplianceMetric(_judge(_verdicts("yes", "no"))).evaluate(run)
        assert result.score == 0.5
        assert not result.passed

    def test_skips_a_thread_with_no_notes(self):
        run = _run(task_policy=_policy_dicts(("Escalate refunds.", False)))
        result = OverrideComplianceMetric(_judge()).evaluate(run)
        assert result.skipped and result.passed


# ---------------------------------------------------------------------------
# memory_extraction
# ---------------------------------------------------------------------------


class TestMemoryExtraction:
    def test_paraphrases_count_as_matches(self):
        run = _run(
            expected_memories=["Escalate refund requests above $500."],
            extracted_memories=["Escalate refund requests above $500 always."],
        )
        result = MemoryExtractionMetric().evaluate(run)
        assert result.score > 0.9
        assert "precision 1.00" in result.details

    def test_a_fabricated_memory_costs_precision(self):
        run = _run(
            expected_memories=["Customer prefers email communication."],
            extracted_memories=[
                "Customer prefers email communication.",
                "Grant the customer a permanent discount.",
            ],
        )
        result = MemoryExtractionMetric().evaluate(run)
        assert "precision 0.50 (1/2)" in result.details
        assert "recall 1.00" in result.details

    def test_a_missed_memory_costs_recall(self):
        run = _run(
            expected_memories=["Prefer email.", "Do not offer the discount."],
            extracted_memories=["Prefer email."],
        )
        assert "recall 0.50" in MemoryExtractionMetric().evaluate(run).details

    def test_duplicate_extractions_cannot_inflate_recall(self):
        run = _run(
            expected_memories=["Prefer email.", "Do not offer the discount."],
            extracted_memories=["Prefer email.", "Prefer email."],
        )
        result = MemoryExtractionMetric().evaluate(run)
        assert "recall 0.50" in result.details and "precision 0.50" in result.details

    def test_extracting_nothing_when_something_was_expected_scores_zero(self):
        run = _run(expected_memories=["Prefer email."])
        assert MemoryExtractionMetric().evaluate(run).score == 0.0

    def test_extracting_something_when_nothing_was_expected_scores_zero(self):
        run = _run(extracted_memories=["Invented rule."])
        assert MemoryExtractionMetric().evaluate(run).score == 0.0

    def test_skips_when_nothing_is_declared(self):
        result = MemoryExtractionMetric().evaluate(_run())
        assert result.skipped and result.passed


# ---------------------------------------------------------------------------
# memory_leakage
# ---------------------------------------------------------------------------


class TestMemoryLeakage:
    def test_clean_run_scores_one(self):
        run = _run(foreign_notes=["Exception approved by supervisor."])
        result = MemoryLeakageMetric().evaluate(run)
        assert result.score == 1.0 and result.passed

    def test_a_verbatim_foreign_note_fails(self):
        run = _run(
            foreign_notes=["Exception approved by supervisor."],
            outputs=[
                {
                    "item_id": "a",
                    "response": "Since the exception approved by supervisor "
                    "applies, no escalation.",
                }
            ],
        )
        result = MemoryLeakageMetric().evaluate(run)
        assert result.score == 0.0 and not result.passed
        assert "1/1 foreign note(s) surfaced" in result.details

    def test_scores_the_fraction_that_surfaced(self):
        run = _run(
            foreign_notes=["Alpha note about gamma.", "Beta note about delta."],
            outputs=[{"item_id": "a", "response": "Alpha note about gamma."}],
        )
        assert MemoryLeakageMetric().evaluate(run).score == 0.5

    def test_case_and_whitespace_differences_still_count_as_leaks(self):
        run = _run(
            foreign_notes=["Do  NOT offer the standard discount."],
            outputs=[{"item_id": "a", "response": "do not offer the standard discount"}],
        )
        assert MemoryLeakageMetric().evaluate(run).score == 0.0

    def test_skips_without_foreign_notes(self):
        result = MemoryLeakageMetric().evaluate(_run())
        assert result.skipped and result.passed


# ---------------------------------------------------------------------------
# The suite factory
# ---------------------------------------------------------------------------


class TestMemoryMetricsFactory:
    def test_deterministic_pair_without_a_judge(self):
        assert [m.name for m in memory_metrics()] == [
            "memory_extraction",
            "memory_leakage",
        ]

    def test_all_four_with_a_judge(self):
        names = [m.name for m in memory_metrics(_judge())]
        assert names == [
            "memory_extraction",
            "memory_leakage",
            "policy_adherence",
            "override_compliance",
        ]

    def test_include_filters(self):
        metrics = memory_metrics(_judge(), include=["memory_leakage"])
        assert [m.name for m in metrics] == ["memory_leakage"]

    def test_asking_for_a_judged_metric_without_a_judge_raises(self):
        with pytest.raises(ValueError, match="judge-less"):
            memory_metrics(include=["policy_adherence"])

    def test_judged_pair_is_in_the_full_judged_suite(self):
        from validation import JUDGED_METRIC_NAMES, judged_metrics

        assert "policy_adherence" in JUDGED_METRIC_NAMES
        assert "override_compliance" in JUDGED_METRIC_NAMES
        built = [m.name for m in judged_metrics(_judge())]
        assert built == list(JUDGED_METRIC_NAMES)
