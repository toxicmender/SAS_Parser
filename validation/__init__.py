"""Validation harness for the SAS -> LLM pipeline: offline cases and
post-hoc live conversations.

Public API:

- :class:`ValidationCase`, :class:`EvaluationRun`, :class:`CaseRun`,
  :class:`MetricResult`, :class:`CaseResult`, :class:`ValidationReport`
  — data models.
- :class:`ValidationRunner` — offline: cases -> pipeline -> metrics -> report.
- :class:`Evaluator` — the scoring core: one EvaluationRun -> one CaseResult.
- :func:`validate_thread` / :func:`run_from_thread` — post-hoc scoring of an
  existing memory-store thread, without re-running the pipeline.
- :func:`validate_transcript` / :func:`run_from_transcript` — scoring of
  arbitrary (prompt, response) transcripts.
- :class:`LiveValidator` / :func:`validations_for_thread` — inline scoring:
  the pipeline scores each item as its response returns and stores the
  verdict in that conversation's memory (opt in via ``SasLLMPipeline(...,
  validator=LiveValidator())``).
- :func:`default_metrics` and the deterministic metric classes.
- :class:`LLMJudgeMetric` — optional LLM-as-judge metric (needs a model).
- :func:`load_cases` — JSON case files -> cases.
- :func:`log_report` / :func:`load_runs` — Spark-backed run history
  (local parquet directory by default, Delta table on Databricks).

See validation/README.md and the CLI (``python -m validation --help``).
"""

from .conversation import (
    run_from_thread,
    run_from_transcript,
    validate_thread,
    validate_transcript,
)
from .dataset import load_cases
from .evaluator import Evaluator
from .judge import LLMJudgeMetric
from .live import LiveValidator, validations_for_thread
from .metrics import (
    DatasetFidelityMetric,
    PythonSyntaxMetric,
    ReferenceSimilarityMetric,
    RequiredTermsMetric,
    ResponseCoverageMetric,
    ValidationMetric,
    default_metrics,
)
from .models import (
    CaseResult,
    CaseRun,
    EvaluationRun,
    MetricResult,
    ValidationCase,
    ValidationReport,
)
from .runner import ValidationRunner
from .tracking import load_runs, log_report

__all__ = [
    "CaseResult",
    "CaseRun",
    "DatasetFidelityMetric",
    "EvaluationRun",
    "Evaluator",
    "LLMJudgeMetric",
    "LiveValidator",
    "MetricResult",
    "PythonSyntaxMetric",
    "ReferenceSimilarityMetric",
    "RequiredTermsMetric",
    "ResponseCoverageMetric",
    "ValidationCase",
    "ValidationMetric",
    "ValidationReport",
    "ValidationRunner",
    "default_metrics",
    "load_cases",
    "load_runs",
    "log_report",
    "run_from_thread",
    "run_from_transcript",
    "validate_thread",
    "validate_transcript",
    "validations_for_thread",
]
