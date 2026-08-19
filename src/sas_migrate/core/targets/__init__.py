"""Translation target contracts and pure resolution."""

from .compatibility import CompatibilityAssessment, choose_item_target
from .models import ResolvedTarget, TargetDefinition, TargetId, TargetSource
from .registry import (
    KNOWN_TARGETS,
    PYSPARK,
    SPARK_SQL,
    resolve_local_target,
    resolve_sharepoint_target,
)
from .validation import (
    ResponseValidationResult,
    TargetIssueCode,
    TargetValidationIssue,
)

__all__ = [
    "KNOWN_TARGETS",
    "PYSPARK",
    "SPARK_SQL",
    "CompatibilityAssessment",
    "ResolvedTarget",
    "ResponseValidationResult",
    "TargetDefinition",
    "TargetId",
    "TargetIssueCode",
    "TargetSource",
    "TargetValidationIssue",
    "choose_item_target",
    "resolve_local_target",
    "resolve_sharepoint_target",
]
