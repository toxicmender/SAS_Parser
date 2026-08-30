"""One-way item fallback policy from Spark SQL to PySpark."""

from __future__ import annotations

from pydantic import Field

from ..models import ContractModel
from .models import ResolvedTarget, TargetId, TargetSource
from .registry import PYSPARK


class CompatibilityAssessment(ContractModel):
    """Target compatibility facts produced by a later assessment service."""

    spark_sql_implementable: bool
    pyspark_strictly_better: bool = False
    reasons: tuple[str, ...] = Field(default_factory=tuple)


def choose_item_target(
    run_target: ResolvedTarget,
    assessment: CompatibilityAssessment,
) -> ResolvedTarget:
    """Apply the only permitted target fallback, otherwise retain the run target."""

    should_fallback = (
        run_target.target is TargetId.SPARK_SQL
        and not assessment.spark_sql_implementable
        and assessment.pyspark_strictly_better
    )
    if not should_fallback:
        return run_target
    return ResolvedTarget(
        target=PYSPARK.target,
        display_name=PYSPARK.display_name,
        canonical_language=PYSPARK.canonical_language,
        fence=PYSPARK.fence,
        source=TargetSource.COMPATIBILITY_FALLBACK,
        requested_value=run_target.requested_value,
        fallback_from=TargetId.SPARK_SQL,
        reasons=assessment.reasons,
    )
