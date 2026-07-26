"""Pydantic models for the validation package. See validation/README.md.

Pure data module — no logging.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, computed_field, model_validator

from chunker.models import SasBatch, SasChunk
from llm_client import TokenUsage


class ValidationCase(BaseModel):
    """
    One evaluation case: a SAS source plus optional expectations.

    Fields
    ------
    case_id
        Stable identifier, e.g. ``"simple_etl"``. Used as the source id
        (``<case_id>.sas``) and in report / MLflow metric keys.
    description
        Free-text note on what the case exercises.
    sas_source
        The SAS program text fed to the pipeline.
    reference_translation
        Optional golden translation. When present,
        :class:`~validation.metrics.ReferenceSimilarityMetric` scores the
        pipeline output against it; when absent that metric is skipped.
    required_terms
        Optional substrings that must appear (case-insensitively) somewhere
        in the concatenated responses, e.g. ``["createDataFrame", "groupBy"]``.
        Empty list skips :class:`~validation.metrics.RequiredTermsMetric`.
    prompt_instructions
        Optional instructions the response must follow, checked one by one by
        :class:`~validation.agentic_metrics.PromptAlignmentMetric`. Empty list
        falls back to that metric's own resolution chain (config.json, then
        the system prompt's standing contract) rather than skipping.
    """

    case_id: str
    description: str = ""
    sas_source: str
    reference_translation: str | None = None
    required_terms: list[str] = Field(default_factory=list)
    prompt_instructions: list[str] = Field(default_factory=list)


class EvaluationRun(BaseModel):
    """
    Case-free unit of scoring: everything a metric needs to evaluate one
    conversation-sized body of LLM output, wherever it came from.

    Three provenances share this shape:

    - **offline case** (:class:`CaseRun` subclass): the runner drove the
      pipeline and re-derived *items*, expectations come from the case;
    - **existing thread**: *prompts*/*outputs* reconstructed from the memory
      store's (human, AI) turn pairs, no *items* (chunker metadata is not
      persisted) — see ``validation.conversation``;
    - **arbitrary transcript**: caller-supplied (prompt, response) pairs.

    Fields
    ------
    run_id
        Label carried into :class:`CaseResult.case_id` — a case id, a
        thread id, or a caller-chosen transcript name.
    items
        Batches/singleton chunks aligned positionally with *outputs*.
        Empty when no chunker metadata exists (thread/transcript modes);
        metrics that need it then skip or fall back to *prompts*.
    prompts
        The human side of each turn, aligned with *outputs*. Empty in the
        offline case mode (the runner scores outputs against *items*).
    outputs
        Raw output dicts (``item_id`` / ``response`` / ``prompt`` /
        ``retrieval_context`` / ``document`` / ... — see
        ``SasLLMPipeline._process``); only ``response`` is required. The
        judged metrics read the optional keys through the properties below,
        so an output dict that lacks them degrades to a skip, never an error.
    required_terms, reference_translation, prompt_instructions
        Optional expectations (see :class:`ValidationCase` for semantics).
    retrieval_contexts
        Per-unit retrieved reference chunks, in rank order, when the caller
        supplies them explicitly. ``None`` (default) reads each unit's from
        its output dict instead — which is where the pipeline puts them.
    summary, summary_source
        A generated summary and the text it summarises, for
        :class:`~validation.summarization.SummarizationMetric`. Populated for
        thread runs from the rolling summary the conversation carries (see
        ``validation.conversation.run_from_thread``); ``None`` skips.
    """

    run_id: str
    items: list[SasBatch | SasChunk] = Field(default_factory=list)
    prompts: list[str] = Field(default_factory=list)
    outputs: list[dict[str, Any]]
    required_terms: list[str] = Field(default_factory=list)
    reference_translation: str | None = None
    prompt_instructions: list[str] = Field(default_factory=list)
    retrieval_contexts: list[list[str]] | None = None
    summary: str | None = None
    summary_source: str | None = None

    @property
    def responses(self) -> list[str]:
        return [str(o.get("response", "")) for o in self.outputs]

    @property
    def joined_responses(self) -> str:
        return "\n\n".join(self.responses)

    @property
    def expected_units(self) -> int:
        """How many answers this run *should* contain: one per item when
        chunker metadata exists, else one per prompt, else whatever
        outputs arrived (an all-outputs transcript is taken at face value)."""
        return len(self.items) or len(self.prompts) or len(self.outputs)

    # ------------------------------------------------------------------
    # Judged-metric views. Each degrades the way `expected_units` does:
    # take the richest source present, fall back, never raise — the same
    # EvaluationRun has to serve an offline case, a reconstructed thread,
    # and a bare transcript.
    # ------------------------------------------------------------------

    @property
    def inputs(self) -> list[str]:
        """The ``input`` each response answered, aligned with :attr:`responses`.

        Prefers the explicit *prompts* (thread/transcript modes), else the
        ``prompt`` the pipeline recorded on the output dict, else the item's
        own SAS source — an offline case scored before the pipeline carried
        prompts still has *something* to judge relevance against. Entries are
        ``""`` only when a unit truly carries none.
        """
        if self.prompts:
            return list(self.prompts)
        return [
            str(output.get("prompt") or self._item_source(index) or "")
            for index, output in enumerate(self.outputs)
        ]

    @property
    def documents(self) -> list[dict[str, Any] | None]:
        """Each unit's structured ``TranslationDocument`` dump, or ``None``
        when the run was unstructured (the metrics then parse the rendered
        Markdown sections instead)."""
        return [
            output.get("document") if isinstance(output.get("document"), dict) else None
            for output in self.outputs
        ]

    def retrieval_context_at(self, index: int) -> list[str]:
        """Reference chunks retrieved for unit *index*, in rank order.

        The explicit :attr:`retrieval_contexts` wins; otherwise the output
        dict's ``retrieval_context``. Empty when guidance was off, or when
        the run was reconstructed from a thread (retrieval is ephemeral and
        never persisted — the same caveat ``dataset_fidelity`` carries).
        """
        if self.retrieval_contexts is not None:
            if index < len(self.retrieval_contexts):
                return list(self.retrieval_contexts[index])
            return []
        if index < len(self.outputs):
            raw = self.outputs[index].get("retrieval_context") or []
            return [str(chunk) for chunk in raw]
        return []

    def contexts_at(self, index: int) -> list[str]:
        """Ground-truth context for unit *index* — what a response must not
        contradict.

        deepeval's ``context`` (reference material the caller supplies), as
        opposed to ``retrieval_context`` (what a retriever fetched): here the
        item's SAS source, plus the golden translation when the run has one.
        Empty when neither exists.
        """
        contexts = []
        source = self._item_source(index)
        if source:
            contexts.append(source)
        if self.reference_translation:
            contexts.append(self.reference_translation)
        return contexts

    @property
    def item_sources(self) -> list[str]:
        """The SAS text behind each unit, in order — empty without chunker
        metadata (thread and transcript modes)."""
        return [self._item_source(i) for i in range(len(self.items))]

    def _item_source(self, index: int) -> str:
        """The SAS text of unit *index*, or ``""`` without chunker metadata."""
        if index >= len(self.items):
            return ""
        item = self.items[index]
        if isinstance(item, SasBatch):
            return "\n".join(chunk.text for chunk in item.chunks)
        return item.text


class CaseRun(EvaluationRun):
    """
    Offline-case specialisation of :class:`EvaluationRun`: the case itself,
    the batches/singleton chunks the batcher produced (re-derived by the
    runner, aligned positionally with *outputs*), and the pipeline's raw
    output dicts. ``run_id`` and the expectation fields derive from the case.
    """

    case: ValidationCase

    @model_validator(mode="before")
    @classmethod
    def _derive_from_case(cls, data: Any) -> Any:
        if isinstance(data, dict) and "case" in data:
            case = data["case"]
            if isinstance(case, dict):
                case = ValidationCase(**case)
            data.setdefault("run_id", case.case_id)
            data.setdefault("required_terms", list(case.required_terms))
            data.setdefault("prompt_instructions", list(case.prompt_instructions))
            data.setdefault("reference_translation", case.reference_translation)
        return data


class MetricResult(BaseModel):
    """
    One metric's verdict on one case. ``skipped`` means the case carried no
    signal for this metric (e.g. no ``reference_translation``); a skipped
    result always passes and is excluded from the case score.
    """

    metric: str
    score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    passed: bool
    skipped: bool = False
    details: str = ""


class CaseResult(BaseModel):
    """All metric results for one case, with derived score / pass views."""

    case_id: str
    item_count: int
    metrics: list[MetricResult]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def score(self) -> float:
        """Mean of non-skipped metric scores; 1.0 when everything skipped."""
        scored = [m.score for m in self.metrics if not m.skipped]
        return sum(scored) / len(scored) if scored else 1.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return all(m.passed for m in self.metrics)


class ValidationReport(BaseModel):
    """Aggregate result of one validation run over a set of cases.

    The two token fields are kept apart on purpose: *token_usage* is what the
    translation under test cost, *judge_token_usage* is what grading it cost.
    Folding them together would make a run look more expensive the more
    thoroughly it was checked, and judging is a choice about the eval, not a
    property of the model being evaluated. Either is ``None`` when nothing
    reported usage — a gateway that returns no usage block, or a judge that is
    not an :class:`llm_client.LLMClient`.
    """

    model: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    # Fingerprint of the user-instruction set active during the run (None
    # when none) — runs under different instructions are not comparable.
    instructions_fingerprint: str | None = None
    results: list[CaseResult] = Field(default_factory=list)
    token_usage: TokenUsage | None = None
    judge_token_usage: TokenUsage | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def score(self) -> float:
        return (
            sum(r.score for r in self.results) / len(self.results)
            if self.results
            else 0.0
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)

    def to_markdown(self) -> str:
        """Human-readable report table (also logged as an MLflow artifact)."""
        lines = [
            f"# Validation report — `{self.model}`",
            "",
            f"- run at: {self.created_at.isoformat()}",
            f"- cases: {len(self.results)}",
            f"- aggregate score: **{self.score:.3f}**",
            f"- overall: **{'PASSED' if self.passed else 'FAILED'}**",
        ]
        if self.token_usage is not None:
            lines.append(f"- tokens (pipeline): {self.token_usage.summary()}")
        if self.judge_token_usage is not None:
            lines.append(f"- tokens (judge): {self.judge_token_usage.summary()}")
        lines += [
            "",
            "| case | metric | score | threshold | status | details |",
            "|---|---|---|---|---|---|",
        ]
        for result in self.results:
            for m in result.metrics:
                status = "skipped" if m.skipped else ("pass" if m.passed else "FAIL")
                details = m.details.replace("|", "\\|").replace("\n", " ")
                lines.append(
                    f"| {result.case_id} | {m.metric} | {m.score:.3f} "
                    f"| {m.threshold:.2f} | {status} | {details} |"
                )
        return "\n".join(lines) + "\n"
