"""Phase 1 acceptance tests for versioned v2 contracts."""

from __future__ import annotations

import importlib.resources
import json
import pathlib
import subprocess
import sys
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from sas_migrate.core.errors import TargetResolutionError
from sas_migrate.core.responses import (
    ResponseEnvelope,
    ResponseMode,
    TranslationCell,
    TranslationCellKind,
    TranslationDocument,
)
from sas_migrate.core.runs import RunEvent, RunEventType
from sas_migrate.core.targets import (
    PYSPARK,
    SPARK_SQL,
    CompatibilityAssessment,
    TargetId,
    TargetSource,
    choose_item_target,
    resolve_local_target,
    resolve_sharepoint_target,
)
from sas_migrate.core.targets.validation import (
    ResponseValidationResult,
    TargetIssueCode,
    TargetValidationIssue,
)
from sas_migrate.core.tokens import (
    CallTokenRecord,
    MessageRole,
    PromptAssembly,
    PromptComponent,
    TokenBudgetPolicy,
    TokenCategory,
)


@pytest.mark.parametrize("value", ["Spark SQL", "spark_sql", "SQL", "sparksql"])
def test_target_resolution_canonicalizes_spark_sql(value: str) -> None:
    resolved = resolve_local_target(value)
    assert resolved.target is TargetId.SPARK_SQL
    assert resolved.canonical_language == "sql"
    assert resolved.source is TargetSource.EXPLICIT


def test_v2_registry_pins_sqlglot_to_databricks() -> None:
    assert SPARK_SQL.sqlglot_dialect == "databricks"
    assert PYSPARK.sqlglot_dialect is None


@pytest.mark.parametrize("value", ["scala", "Spark Scala", "spark-scala", "java"])
def test_target_resolution_rejects_removed_and_unknown_targets(value: str) -> None:
    with pytest.raises(TargetResolutionError):
        resolve_local_target(value)


def test_target_resolution_precedence_is_boundary_specific() -> None:
    assert resolve_local_target(configured="pyspark").source is TargetSource.CONFIG
    assert resolve_local_target().target is TargetId.SPARK_SQL
    sharepoint = resolve_sharepoint_target(
        "pyspark", explicit_fallback="sql", configured="sql"
    )
    assert sharepoint.target is TargetId.PYSPARK
    assert sharepoint.source is TargetSource.REQUEST


def test_compatibility_fallback_is_one_way_and_auditable() -> None:
    run_target = resolve_local_target("sql")
    item_target = choose_item_target(
        run_target,
        CompatibilityAssessment(
            spark_sql_implementable=False,
            pyspark_strictly_better=True,
            reasons=("requires a Python-only API",),
        ),
    )
    assert item_target.target is TargetId.PYSPARK
    assert item_target.fallback_from is TargetId.SPARK_SQL
    assert item_target.source is TargetSource.COMPATIBILITY_FALLBACK

    pyspark = resolve_local_target("pyspark")
    assert choose_item_target(
        pyspark,
        CompatibilityAssessment(
            spark_sql_implementable=True, pyspark_strictly_better=False
        ),
    ) is pyspark


def _document(target: TargetId = TargetId.SPARK_SQL) -> TranslationDocument:
    return TranslationDocument(
        target=target,
        analysis="Preserve row semantics.",
        cells=(
            TranslationCell(
                kind=TranslationCellKind.CODE,
                source="SELECT * FROM source",
                language="sql",
                chunk_id="chunk-1",
            ),
        ),
    )


def test_response_envelope_round_trips_with_schema_version_two() -> None:
    target = resolve_local_target("sql")
    envelope = ResponseEnvelope(
        mode=ResponseMode.STRUCTURED,
        raw_message='{"target":"spark_sql"}',
        document=_document(),
        resolved_target=target,
        validation=ResponseValidationResult.accepted(target.target),
    )
    restored = ResponseEnvelope.from_json(envelope.to_json())
    assert restored == envelope
    assert json.loads(envelope.to_json())["schema_version"] == 2
    assert restored.document is not None
    assert "```sql" in restored.document.to_markdown()


def test_raw_fallback_retains_parse_error_and_validation() -> None:
    target = resolve_local_target("sql")
    envelope = ResponseEnvelope(
        mode=ResponseMode.RAW_FALLBACK,
        raw_message="## Translation\n```sql\nSELECT 1\n```",
        structured_error="provider ignored response schema",
        document=_document(),
        resolved_target=target,
        validation=ResponseValidationResult.accepted(target.target),
    )
    assert ResponseEnvelope.from_json(envelope.to_json()) == envelope


def test_response_envelope_rejects_target_validation_disagreement() -> None:
    target = resolve_local_target("sql")
    with pytest.raises(ValidationError):
        ResponseEnvelope(
            mode=ResponseMode.STRUCTURED,
            raw_message="raw",
            document=_document(),
            resolved_target=target,
            validation=ResponseValidationResult.accepted(TargetId.PYSPARK),
        )


def test_valid_target_result_cannot_accept_a_reported_target_mismatch() -> None:
    with pytest.raises(ValidationError):
        ResponseValidationResult(
            valid=True,
            resolved_target=TargetId.SPARK_SQL,
            reported_target=TargetId.PYSPARK,
        )

    rejected = ResponseValidationResult(
        valid=False,
        resolved_target=TargetId.SPARK_SQL,
        reported_target=TargetId.PYSPARK,
        issues=(
            TargetValidationIssue(
                code=TargetIssueCode.TARGET_MISMATCH,
                message="document reported PySpark for a Spark SQL item",
            ),
        ),
    )
    assert not rejected.valid


def test_prompt_assembly_and_call_record_reconcile_component_counts() -> None:
    assembly = PromptAssembly(
        estimator="tiktoken",
        encoding="o200k_base",
        components=(
            PromptComponent(
                category=TokenCategory.SAS_SOURCE,
                text="data out; set in; run;",
                message_role=MessageRole.USER,
                token_count=8,
            ),
            PromptComponent(
                category=TokenCategory.PROJECT_INSTRUCTIONS,
                text="Preserve missing-value semantics.",
                message_role=MessageRole.SYSTEM,
                token_count=5,
            ),
        ),
    )
    assert assembly.estimated_input_total == 13
    assert assembly.input_by_category()[TokenCategory.SAS_SOURCE] == 8

    record = CallTokenRecord(
        run_id="run-1",
        thread_id="thread-1",
        item_id="item-1",
        attempt=1,
        target=TargetId.SPARK_SQL,
        estimator=assembly.estimator,
        encoding=assembly.encoding,
        estimated_input_by_category=assembly.input_by_category(),
        estimated_input_total=13,
        provider_input_tokens=12,
        provider_output_tokens=7,
        provider_total_tokens=19,
        provider_input_delta=-1,
        accepted_attempt=True,
    )
    assert CallTokenRecord.from_json(record.to_json()) == record


def test_token_policy_round_trip_and_available_input() -> None:
    policy = TokenBudgetPolicy(
        max_input_tokens=10_000,
        reserved_output_tokens=2_000,
        safety_margin_tokens=500,
        max_sas_source_tokens=4_000,
    )
    assert policy.available_input_tokens == 7_500
    assert TokenBudgetPolicy.from_json(policy.to_json()) == policy


def test_run_event_round_trip() -> None:
    event = RunEvent(
        event_id="event-1",
        event_type=RunEventType.ITEM_STARTED,
        occurred_at=datetime(2026, 8, 18, tzinfo=UTC),
        run_id="run-1",
        thread_id="thread-1",
        item_id="item-1",
        attempt=1,
    )
    assert RunEvent.from_json(event.to_json()) == event


def test_core_import_does_not_load_application_or_adapters() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys; import sas_migrate.core; "
            "assert not any(n.startswith(('sas_migrate.application', "
            "'sas_migrate.adapters')) for n in sys.modules)"
        ),
    ]
    result = subprocess.run(
        command,
        cwd=SRC,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_schema_resource_is_bundled_and_versioned() -> None:
    resource = importlib.resources.files("sas_migrate.resources").joinpath(
        "contracts/schema-v2.json"
    )
    schema = json.loads(resource.read_text("utf-8"))
    assert schema["properties"]["schema_version"]["const"] == 2
