"""Network-free orchestration of XREF pre, post and both modes."""

from __future__ import annotations

from sas_migrate.core.sas import SasBatchResult
from sas_migrate.core.targets import ResolvedTarget, TargetId

from .models import (
    BothRewriteResult,
    ParseFailureMode,
    XrefApplyMode,
    XrefMappings,
)
from .sas_rewriter import rewrite_datasets
from .target_rewriters import (
    rewrite_pyspark_paths,
    rewrite_pyspark_tables,
    rewrite_sql_target,
)


def apply_pre(result: SasBatchResult, mappings: XrefMappings) -> SasBatchResult:
    return rewrite_datasets(result, mappings)


def apply_post(
    code: str,
    target: ResolvedTarget,
    mappings: XrefMappings,
    *,
    on_failure: ParseFailureMode = ParseFailureMode.WARN,
) -> str:
    """Apply target-specific table and path rewriting."""

    if not mappings or not code.strip():
        return code
    if target.target is TargetId.SPARK_SQL:
        return rewrite_sql_target(
            code,
            mappings.dataset_mapping,
            mappings.by_path,
            on_failure=on_failure,
        )
    if target.target is TargetId.PYSPARK:
        output = rewrite_pyspark_tables(
            code,
            mappings.dataset_mapping,
            on_failure=on_failure,
        )
        return rewrite_pyspark_paths(
            output,
            mappings.by_path,
            on_failure=on_failure,
        )
    raise ValueError(f"unsupported XREF target {target.target!r}")


def apply_both(
    code: str,
    target: ResolvedTarget,
    mappings: XrefMappings,
    *,
    pre_code: str | None = None,
    result: SasBatchResult | None = None,
    on_failure: ParseFailureMode = ParseFailureMode.WARN,
) -> BothRewriteResult:
    """Apply the post pass and report mappings reached only after conversion."""

    baseline = pre_code if pre_code is not None else code
    rewritten = apply_post(code, target, mappings, on_failure=on_failure)
    checked = {**mappings.dataset_mapping, **mappings.by_path}
    only_post = tuple(
        sorted(
            source
            for source, mapped in checked.items()
            if mapped in rewritten and mapped not in baseline
        )
    )
    return BothRewriteResult(
        code=rewritten,
        result=apply_pre(result, mappings) if result is not None else None,
        pre_applied=pre_code is not None or result is not None,
        post_changed=rewritten != code,
        only_post=only_post,
    )


def apply(
    mode: XrefApplyMode,
    *,
    mappings: XrefMappings,
    result: SasBatchResult | None = None,
    code: str | None = None,
    target: ResolvedTarget | None = None,
    pre_code: str | None = None,
    on_failure: ParseFailureMode = ParseFailureMode.WARN,
) -> SasBatchResult | str | BothRewriteResult:
    """Dispatch the explicit v2 XREF mode without reading config or doing I/O."""

    if mode is XrefApplyMode.PRE:
        if result is None:
            raise ValueError("XREF pre mode requires a batch result")
        return apply_pre(result, mappings)
    if code is None or target is None:
        raise ValueError(f"XREF {mode.value} mode requires code and a resolved target")
    if mode is XrefApplyMode.POST:
        return apply_post(code, target, mappings, on_failure=on_failure)
    if mode is XrefApplyMode.BOTH:
        return apply_both(
            code,
            target,
            mappings,
            pre_code=pre_code,
            result=result,
            on_failure=on_failure,
        )
    raise ValueError(f"unsupported XREF mode {mode!r}")


__all__ = ["apply", "apply_both", "apply_post", "apply_pre"]
