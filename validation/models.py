"""Pydantic models for the validation package. See validation/README.md.

Pure data module — no logging.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, computed_field

from chunker.models import SasBatch, SasChunk


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
    """

    case_id: str
    description: str = ""
    sas_source: str
    reference_translation: str | None = None
    required_terms: list[str] = Field(default_factory=list)


class CaseRun(BaseModel):
    """
    Everything a metric needs to score one case: the case itself, the
    batches/singleton chunks the batcher produced (re-derived by the runner,
    aligned positionally with *outputs*), and the pipeline's raw output dicts
    (``item_id`` / ``response`` / ... — see ``SasLLMPipeline._process``).
    """

    case: ValidationCase
    items: list[SasBatch | SasChunk]
    outputs: list[dict[str, Any]]

    @property
    def responses(self) -> list[str]:
        return [str(o.get("response", "")) for o in self.outputs]

    @property
    def joined_responses(self) -> str:
        return "\n\n".join(self.responses)


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
    """Aggregate result of one validation run over a set of cases."""

    model: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    results: list[CaseResult] = Field(default_factory=list)

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
