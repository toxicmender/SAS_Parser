"""V2 validation contracts, metrics, reporting, and use cases."""

from .budgeting import build_token_budget_report, token_budget_compliance
from .evaluator import Evaluator
from .metrics import (
    DatasetFidelityMetric,
    LanguageComplianceMetric,
    ReferenceSimilarityMetric,
    RequiredTermsMetric,
    ResponseCoverageMetric,
    TargetSyntaxMetric,
    ValidationMetric,
    default_metrics,
    metric_names,
)
from .models import (
    CaseResult,
    EvaluationRun,
    MetricResult,
    TokenBudgetPolicy,
    TokenBudgetReport,
    ValidationReport,
    ValidationUnit,
)
from .reporting import render_json, render_markdown
from .service import ValidationService

__all__ = [
    "CaseResult",
    "DatasetFidelityMetric",
    "EvaluationRun",
    "Evaluator",
    "LanguageComplianceMetric",
    "MetricResult",
    "ReferenceSimilarityMetric",
    "RequiredTermsMetric",
    "ResponseCoverageMetric",
    "TargetSyntaxMetric",
    "TokenBudgetPolicy",
    "TokenBudgetReport",
    "ValidationMetric",
    "ValidationReport",
    "ValidationService",
    "ValidationUnit",
    "build_token_budget_report",
    "default_metrics",
    "metric_names",
    "render_json",
    "render_markdown",
    "token_budget_compliance",
]
