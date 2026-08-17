"""Translation target contracts and pure resolution."""

from .compatibility import CompatibilityAssessment, choose_item_target
from .models import ResolvedTarget, TargetDefinition, TargetId, TargetSource
from .registry import KNOWN_TARGETS, resolve_local_target, resolve_sharepoint_target

__all__ = [
    "KNOWN_TARGETS",
    "CompatibilityAssessment",
    "ResolvedTarget",
    "TargetDefinition",
    "TargetId",
    "TargetSource",
    "choose_item_target",
    "resolve_local_target",
    "resolve_sharepoint_target",
]
