"""Offline validation harness for the SAS -> LLM pipeline.

Public API:

- :class:`ValidationCase`, :class:`CaseRun`, :class:`MetricResult`,
  :class:`CaseResult`, :class:`ValidationReport` — data models.
- :class:`ValidationRunner` — cases -> pipeline -> metrics -> report.
- :func:`default_metrics` and the deterministic metric classes.
- :class:`LLMJudgeMetric` — optional LLM-as-judge metric (needs a model).
- :func:`load_cases` — JSON case files -> cases.
- :func:`log_report` / :func:`load_runs` — Spark-backed run history
  (local parquet directory by default, Delta table on Databricks).

See validation/README.md and the CLI (``python -m validation --help``).
"""

from .dataset import load_cases
from .judge import LLMJudgeMetric
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
    "LLMJudgeMetric",
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
]
