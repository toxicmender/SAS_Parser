"""Instruction-following and trace metrics, judged by a model.
See validation/README.md.

Three metrics over what the pipeline was *told to do* and what it then did:

    prompt_alignment   each declared instruction was followed
    plan_adherence     the code matches the reasoning that preceded it
    task_completion    the run as a whole achieved what it was asked

deepeval's Plan Adherence and Task Completion are trace-only metrics, read off
an agent's execution trace. This pipeline has no tool-calling loop, but it does
have the equivalent artefacts, and they are already on the run:

* the **plan** is the item's ``analysis`` — the system prompt requires the model
  to reason through execution order, PDV vs DAG semantics, macro expansion
  timing and the item's hazard flags *before writing any code*
  (:mod:`pipeline.constants`), so ``analysis`` is exactly deepeval's
  "plan extracted from the agent's thinking or reasoning";
* the **execution steps** are the ``mapping`` entries and the ordered ``cells``
  that follow it;
* the **outcome** is the run's responses.

Structured runs read these off the ``TranslationDocument`` dump the pipeline
records; unstructured runs fall back to the rendered ``## Analysis`` /
``## Mapping`` / ``## Translation`` sections, which are the same contract.

Logger name: ``validation.agentic_metrics``.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import app_config

from .judged import JudgedMetric, ScoreVerdict, markdown_section, numbered
from .models import EvaluationRun, MetricResult

logger = logging.getLogger(__name__)


# The standing instructions every run carries, transcribed from the system
# prompt in pipeline/constants.py. Used when neither the run nor
# config.json declares its own, so `prompt_alignment` measures the contract the
# pipeline actually imposes rather than skipping.
DEFAULT_PROMPT_INSTRUCTIONS = [
    "Structure the response as Analysis, Mapping, Translation and Risks, in "
    "that order.",
    "Reason through what bears on correctness — execution order, PDV vs DAG "
    "semantics, macro expansion timing, and any hazard flagged in the item's "
    "context — before writing any code.",
    "Give each SAS construct its target-language equivalent and name any "
    "semantic difference.",
    "Emit the translated code in fenced blocks of runnable target-language "
    "code, preserving execution order across the item's member chunks and "
    "files.",
    "Flag every P0 silent-error risk explicitly, and say so rather than "
    "guessing when a translation is ambiguous or unsafe.",
]


_ALIGNMENT_PROMPT = """\
You are checking whether an answer followed the instructions it was given.

The request it answered:
{input}

The answer:
{answer}

Instructions it was required to follow:
{instructions}

For each instruction, in order, say whether the answer followed it:
- "yes" — followed
- "no"  — not followed
- "idk" — the instruction does not apply to this request

Reply with JSON: {{"verdicts": [{{"verdict": "yes|no|idk", "reason": "..."}}, ...]}}
with exactly {count} verdict(s), in the same order as the instructions.
"""

_PLAN_ADHERENCE_PROMPT = """\
You are checking whether work followed the plan that was stated for it.

The task:
{task}

The plan that was stated before the work began:
{plan}

The steps that were then carried out:
{execution}

Score from 0.0 to 1.0 how well the executed steps follow the stated plan: 1.0
when every planned step was carried out as described, 0.0 when the execution
ignores or contradicts the plan. Judge the *process*, not whether the outcome
is good — a well-executed bad plan still scores high.

Reply with JSON: {{"score": <0.0-1.0>, "reason": "..."}}
"""

_TASK_COMPLETION_PROMPT = """\
You are checking whether a task was completed.

The task:
{task}

The outcome that was produced:
{outcome}

Score from 0.0 to 1.0 how well the outcome achieves the task: 1.0 when the task
is fully achieved, 0.0 when it is not addressed at all. Judge the outcome
against the task, not against any particular way of getting there.

Reply with JSON: {{"score": <0.0-1.0>, "reason": "..."}}
"""

_INFERRED_TASK = (
    "Translate the given Base SAS source into {language}, preserving what the "
    "SAS program does: the same inputs read, the same transformations applied, "
    "the same outputs produced."
)


def _clamped(verdict: ScoreVerdict) -> float:
    return max(0.0, min(1.0, float(verdict.score)))


class PromptAlignmentMetric(JudgedMetric):
    """
    ``instructions followed / total instructions`` — deepeval's
    *Prompt Alignment*.

    deepeval makes ``prompt_instructions`` a mandatory constructor argument.
    Here they resolve through the repo-wide precedence chain plus one extra
    step, so the metric is useful with no configuration at all:

        explicit constructor list
        > the run's ``prompt_instructions`` (a case can declare its own)
        > config.json ``validation.prompt_instructions``
        > :data:`DEFAULT_PROMPT_INSTRUCTIONS`, the system prompt's contract

    One judge call per unit.
    """

    name = "prompt_alignment"
    default_threshold = 0.8

    def __init__(
        self,
        llm: Any,
        *,
        threshold: float | None = None,
        prompt_instructions: Sequence[str] | None = None,
    ) -> None:
        super().__init__(llm, threshold=threshold)
        self._instructions = list(prompt_instructions) if prompt_instructions else []

    def _resolve(self, run: EvaluationRun) -> list[str]:
        if self._instructions:
            return self._instructions
        if run.prompt_instructions:
            return list(run.prompt_instructions)
        configured = app_config.get_value("validation", "prompt_instructions", None)
        if isinstance(configured, list) and configured:
            return [str(entry) for entry in configured]
        return list(DEFAULT_PROMPT_INSTRUCTIONS)

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        instructions = self._resolve(run)
        if not instructions:
            return self._result(1.0, "no instructions declared", skipped=True)
        inputs = run.inputs
        scores: list[float] = []
        judged = 0
        first_reason = ""
        for index, response in enumerate(run.responses):
            if not response.strip():
                continue
            judged += 1
            verdicts = self._classify(
                _ALIGNMENT_PROMPT.format(
                    input=inputs[index] if index < len(inputs) else "",
                    answer=response,
                    instructions=numbered(instructions),
                    count=len(instructions),
                ),
                len(instructions),
            )
            if verdicts is None:
                scores.append(0.0)
                continue
            scores.append(self._ratio(verdicts))
            first_reason = first_reason or self._first_reason(verdicts)
        if not judged:
            return self._result(1.0, "no responses to check", skipped=True)
        details = f"{len(instructions)} instruction(s) over {judged} unit(s)"
        if first_reason:
            details += f"; e.g. {first_reason}"
        return self._result(sum(scores) / len(scores), details)


class PlanAdherenceMetric(JudgedMetric):
    """
    Alignment of the executed steps with the stated plan — deepeval's
    *Plan Adherence*.

    Plan = the item's ``analysis``; execution = its ``mapping`` entries and
    ``cells``; task = the item's prompt. This is the metric that catches an
    answer whose Analysis correctly identifies a hazard (a SYMPUT scope
    problem, a MERGE-vs-join difference) and whose code then ignores it.

    deepeval auto-passes with 1.0 when the trace contains no plan; a unit with
    no analysis is skipped here instead — which also passes, and additionally
    keeps the empty result out of the case mean. One judge call per unit.
    """

    name = "plan_adherence"
    default_threshold = 0.7

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        inputs = run.inputs
        documents = run.documents
        scores: list[float] = []
        judged = 0
        first_reason = ""
        for index, response in enumerate(run.responses):
            document = documents[index] if index < len(documents) else None
            plan = _plan_of(document, response)
            execution = _execution_of(document, response)
            if not plan or not execution:
                continue
            judged += 1
            verdict = self._ask(
                ScoreVerdict,
                _PLAN_ADHERENCE_PROMPT.format(
                    task=inputs[index] if index < len(inputs) else "",
                    plan=plan,
                    execution=execution,
                ),
            )
            if verdict is None:
                scores.append(0.0)
                continue
            scores.append(_clamped(verdict))
            first_reason = first_reason or verdict.reason.strip()
        if not judged:
            return self._result(
                1.0, "no stated plan to adhere to (no Analysis section)", skipped=True
            )
        details = f"mean over {judged} unit(s) with a stated plan"
        if first_reason:
            details += f"; e.g. {first_reason}"
        return self._result(sum(scores) / len(scores), details)


def _plan_of(document: dict[str, Any] | None, response: str) -> str:
    """The item's stated plan: the structured ``analysis`` field, else the
    rendered ``## Analysis`` section."""
    if document:
        analysis = str(document.get("analysis") or "").strip()
        if analysis:
            return analysis
    return markdown_section(response, "Analysis")


def _execution_of(document: dict[str, Any] | None, response: str) -> str:
    """The steps actually carried out: the structured ``mapping`` entries and
    ``cells``, else the rendered ``## Mapping`` + ``## Translation`` sections."""
    if document:
        parts: list[str] = []
        for entry in document.get("mapping") or []:
            if isinstance(entry, dict):
                parts.append(
                    f"- {entry.get('sas_construct', '')} -> "
                    f"{entry.get('equivalent', '')} {entry.get('difference', '')}".strip()
                )
        for cell in document.get("cells") or []:
            if isinstance(cell, dict):
                comment = str(cell.get("comment") or "").strip()
                source = str(cell.get("source") or "").strip()
                parts.append(f"{comment}\n{source}".strip())
        joined = "\n".join(part for part in parts if part)
        if joined.strip():
            return joined
    sections = [
        markdown_section(response, "Mapping"),
        markdown_section(response, "Translation"),
    ]
    return "\n\n".join(section for section in sections if section)


class TaskCompletionMetric(JudgedMetric):
    """
    Alignment of the run's outcome with the task — deepeval's
    *Task Completion*.

    Scored **once per run**, not per unit: the task is the run (translate this
    program), and a per-item average would report a run that translated half a
    program as half-successful rather than incomplete. Referenceless — no
    expected output needed.

    *task* defaults to the pipeline's standing goal, phrased for
    *output_language*; pass one explicitly to score against a narrower brief.
    One judge call per run.
    """

    name = "task_completion"
    default_threshold = 0.7

    def __init__(
        self,
        llm: Any,
        *,
        threshold: float | None = None,
        task: str | None = None,
        output_language: str = "PySpark",
    ) -> None:
        super().__init__(llm, threshold=threshold)
        self._task = task
        self._output_language = output_language

    def evaluate(self, run: EvaluationRun) -> MetricResult:
        outcome = run.joined_responses.strip()
        if not outcome:
            return self._result(0.0, "no outcome to judge")
        task = self._task or _INFERRED_TASK.format(language=self._output_language)
        # The sources the run was given are part of the task statement: "did it
        # translate *this*" is not answerable from the responses alone.
        sources = [source for source in run.item_sources if source]
        if sources:
            task = f"{task}\n\nSAS source:\n" + "\n\n".join(sources)
        elif run.inputs:
            task = f"{task}\n\nRequests:\n" + "\n\n".join(
                request for request in run.inputs if request.strip()
            )
        verdict = self._ask(
            ScoreVerdict,
            _TASK_COMPLETION_PROMPT.format(task=task, outcome=outcome),
        )
        if verdict is None:
            return self._result(0.0, "unusable judge reply")
        details = f"whole-run outcome over {run.expected_units} unit(s)"
        if verdict.reason.strip():
            details += f"; {verdict.reason.strip()}"
        return self._result(_clamped(verdict), details)
