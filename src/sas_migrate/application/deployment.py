"""Offline deployment smoke for the installed v2 application.

The smoke deliberately crosses core and application boundaries without
requiring credentials, Spark, or a model call.  It is safe to run as an image
health check and strict enough to catch source-only packaging successes.
"""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import json
import os
from collections.abc import Callable
from typing import Literal

from pydantic import Field, model_validator

from sas_migrate.core.models import ContractModel, VersionedContract
from sas_migrate.core.responses import ResponseTargetValidator, normalize_raw_response
from sas_migrate.core.sas import SasSemanticChunker
from sas_migrate.core.targets import SPARK_SQL, TargetId, resolve_local_target
from sas_migrate.core.tokens import CallTokenRecord, TokenCallLedger, TokenCategory

from .validation import (
    EvaluationRun,
    TokenBudgetPolicy,
    ValidationService,
    ValidationUnit,
)

_DISTRIBUTION = "sas-parser"
_SMOKE_SOURCE = "data work.result; set work.source; total = amount * 2; run;"
_SMOKE_SQL = (
    "CREATE OR REPLACE TABLE work.result AS "
    "SELECT amount * 2 AS total FROM work.source"
)


class DeploymentSmokeCheck(ContractModel):
    """One independently reported deployment invariant."""

    name: str = Field(min_length=1)
    passed: bool
    details: str = ""


class DeploymentSmokeReport(VersionedContract):
    """Machine-readable result emitted by ``sas-migrate smoke``."""

    distribution: str = _DISTRIBUTION
    distribution_version: str
    installation_mode: Literal["wheel", "editable", "unknown"]
    target: TargetId = TargetId.SPARK_SQL
    sqlglot_dialect: str = "databricks"
    checks: tuple[DeploymentSmokeCheck, ...]
    passed: bool

    @model_validator(mode="after")
    def validate_status(self) -> DeploymentSmokeReport:
        expected = bool(self.checks) and all(check.passed for check in self.checks)
        if self.passed != expected:
            raise ValueError("deployment smoke status must agree with its checks")
        return self


def _installation() -> tuple[str, Literal["wheel", "editable", "unknown"]]:
    distribution = importlib.metadata.distribution(_DISTRIBUTION)
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        return distribution.version, "wheel"
    try:
        metadata = json.loads(direct_url)
    except json.JSONDecodeError:
        return distribution.version, "unknown"
    editable = metadata.get("dir_info", {}).get("editable") is True
    return distribution.version, "editable" if editable else "wheel"


def _runtime_identity() -> tuple[bool | None, str]:
    getuid = getattr(os, "geteuid", None)
    if getuid is None:
        return None, "effective user id is unavailable on this platform"
    uid = getuid()
    return uid != 0, f"effective user id is {uid}"


def _packaged_schema() -> str:
    resource = importlib.resources.files("sas_migrate.resources").joinpath(
        "contracts/schema-v2.json"
    )
    schema = json.loads(resource.read_text("utf-8"))
    if schema.get("properties", {}).get("schema_version", {}).get("const") != 2:
        raise RuntimeError("packaged contract schema does not require schema_version=2")
    return "packaged schema requires schema_version=2"


def _application_flow() -> str:
    target = resolve_local_target("sql")
    if target.target is not TargetId.SPARK_SQL:
        raise RuntimeError(f"resolved unexpected target {target.target.value!r}")
    if SPARK_SQL.sqlglot_dialect != "databricks":
        raise RuntimeError("Spark SQL is not configured for the Databricks dialect")

    parsed = SasSemanticChunker(timeout=None).chunk_text(
        _SMOKE_SOURCE,
        source_id="deployment-smoke.sas",
    )
    if len(parsed.chunks) != 1:
        raise RuntimeError(f"expected one SAS chunk, received {len(parsed.chunks)}")
    chunk_id = parsed.chunks[0].chunk_id
    raw_response = (
        "## Analysis\nPreserve the input and output datasets.\n\n"
        "## Translation\n"
        f"### Chunk `{chunk_id}`\n"
        f"```sql\n{_SMOKE_SQL}\n```\n"
    )
    normalized = normalize_raw_response(raw_response, target)
    target_result = ResponseTargetValidator().validate(
        normalized.document,
        target,
        known_chunk_ids={chunk_id},
        normalization_issues=normalized.issues,
    )
    if not target_result.valid:
        issues = "; ".join(issue.message for issue in target_result.issues)
        raise RuntimeError(f"normalized response failed target validation: {issues}")

    input_tokens = {
        TokenCategory.SAS_SOURCE: 16,
        TokenCategory.PROJECT_INSTRUCTIONS: 8,
    }
    ledger = TokenCallLedger(
        records=(
            CallTokenRecord(
                run_id="deployment-smoke",
                thread_id="deployment-smoke",
                item_id=chunk_id,
                attempt=1,
                target=target.target,
                estimator="deployment-smoke",
                encoding="deterministic",
                estimated_input_by_category=input_tokens,
                estimated_input_total=sum(input_tokens.values()),
                estimated_output_by_category={TokenCategory.CODE_OUTPUT: 12},
                accepted_attempt=True,
            ),
        )
    )
    validation = ValidationService().validate(
        EvaluationRun(
            run_id="deployment-smoke",
            target=target.target,
            units=(
                ValidationUnit(
                    unit_id=chunk_id,
                    source=_SMOKE_SOURCE,
                    response=normalized.document.to_markdown(default_fence=target.fence),
                    input_datasets=("work.source",),
                    output_datasets=("work.result",),
                    target_validation=target_result,
                ),
            ),
        ),
        model="offline/deployment-smoke",
        translation_ledger=ledger,
        translation_policy=TokenBudgetPolicy(max_run_tokens=100),
    )
    budget = validation.translation_tokens
    if not validation.passed or budget is None or not budget.compliant:
        raise RuntimeError("v2 validation or token-budget smoke failed")
    if budget.input_by_category != {
        "project_instructions": 8,
        "sas_source": 16,
    }:
        raise RuntimeError(f"unexpected token attribution: {budget.input_by_category}")
    return (
        f"1 SAS chunk; {target.target.value}; dialect={SPARK_SQL.sqlglot_dialect}; "
        f"validation score={validation.score:.3f}; tokens={budget.current_run_tokens}"
    )


def _run_check(name: str, operation: Callable[[], str]) -> DeploymentSmokeCheck:
    try:
        details = operation()
    except Exception as exc:  # noqa: BLE001 - smoke must report the failed boundary
        return DeploymentSmokeCheck(
            name=name,
            passed=False,
            details=f"{type(exc).__name__}: {exc}",
        )
    return DeploymentSmokeCheck(name=name, passed=True, details=details)


def run_deployment_smoke(
    *,
    require_wheel: bool = False,
    require_non_root: bool = False,
) -> DeploymentSmokeReport:
    """Exercise a packaged, credential-free v2 path and report each invariant."""

    try:
        version, installation_mode = _installation()
    except importlib.metadata.PackageNotFoundError:
        version, installation_mode = "unknown", "unknown"

    installation_ok = installation_mode != "unknown" and (
        not require_wheel or installation_mode == "wheel"
    )
    installation_details = f"{_DISTRIBUTION} {version} installed as {installation_mode}"
    if require_wheel and installation_mode != "wheel":
        installation_details += "; a wheel installation is required"

    non_root, identity_details = _runtime_identity()
    identity_ok = not require_non_root or non_root is True
    if require_non_root and non_root is not True:
        identity_details += "; a non-root runtime is required"

    checks = (
        DeploymentSmokeCheck(
            name="installed_distribution",
            passed=installation_ok,
            details=installation_details,
        ),
        DeploymentSmokeCheck(
            name="runtime_identity",
            passed=identity_ok,
            details=identity_details,
        ),
        _run_check("packaged_schema", _packaged_schema),
        _run_check("v2_application_flow", _application_flow),
    )
    return DeploymentSmokeReport(
        distribution_version=version,
        installation_mode=installation_mode,
        checks=checks,
        passed=all(check.passed for check in checks),
    )


__all__ = [
    "DeploymentSmokeCheck",
    "DeploymentSmokeReport",
    "run_deployment_smoke",
]
