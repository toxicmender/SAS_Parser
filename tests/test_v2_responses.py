"""Phase 3 response normalization, validation, retry, and publication gates."""

from __future__ import annotations

import asyncio
import pathlib
import sys
from typing import Literal

import pytest
from pydantic import ValidationError

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from sas_migrate.application import (
    ResponseAcceptanceOutcome,
    ResponseAcceptanceService,
    ResponseAttempt,
)
from sas_migrate.application.ports import ProviderResponse
from sas_migrate.core.errors import ResponseContractError
from sas_migrate.core.responses import (
    MappingEntry,
    ResponseEnvelope,
    ResponseMode,
    RiskNote,
    RiskSeverity,
    TranslationCell,
    TranslationCellKind,
    TranslationDocument,
    normalize_raw_response,
)
from sas_migrate.core.runs import ItemStatus
from sas_migrate.core.targets import TargetId, resolve_local_target
from sas_migrate.core.targets.validation import (
    ResponseValidationResult,
    TargetIssueCode,
    TargetValidationIssue,
)

FENCE = chr(96) * 3
Case = tuple[str, TargetId, Literal["python", "sql"], str, str, str]
CASES: tuple[Case, ...] = (
    (
        "pyspark",
        TargetId.PYSPARK,
        "python",
        "result = source.select('*')",
        "for = 1",
        "sql",
    ),
    (
        "spark sql",
        TargetId.SPARK_SQL,
        "sql",
        "SELECT * FROM source",
        "SELECT FROM",
        "python",
    ),
)


def _document(
    target: TargetId,
    language: Literal["python", "sql"] | None,
    source: str,
    *,
    chunk_id: str | None = "chunk-1",
) -> TranslationDocument:
    return TranslationDocument(
        target=target,
        analysis="Preserve source semantics.",
        mapping=(
            MappingEntry(
                sas_construct="DATA step",
                equivalent="target transformation",
            ),
        ),
        cells=(
            TranslationCell(
                kind=TranslationCellKind.CODE,
                source=source,
                language=language,
                chunk_id=chunk_id,
            ),
        ),
        risks=(RiskNote(severity=RiskSeverity.P2, note="Review performance."),),
    )


def _raw(info: str, source: str, *, translation: bool = True) -> str:
    heading = "## Translation\n\n" if translation else ""
    return (
        "## Analysis\n\nPreserve source semantics.\n\n"
        "## Mapping\n\n- **DATA step** → target transformation\n\n"
        f"{heading}{FENCE}{info}\n{source}\n{FENCE}\n\n"
        "## Risks\n\n- **P2** — Review performance.\n"
    )


def _codes(result) -> set[TargetIssueCode]:
    return {issue.code for issue in result.issues}


@pytest.mark.parametrize(
    ("target_name", "target_id", "language", "valid_source", "_bad", "_foreign"),
    CASES,
)
def test_valid_structured_response_for_each_target(
    target_name: str,
    target_id: TargetId,
    language: Literal["python", "sql"],
    valid_source: str,
    _bad: str,
    _foreign: str,
) -> None:
    target = resolve_local_target(target_name)
    response = ProviderResponse(
        raw_message="provider raw payload",
        structured_document=_document(target_id, language, valid_source),
    )
    envelope = ResponseAcceptanceService().envelope(
        response,
        target,
        known_chunk_ids={"chunk-1"},
    )
    assert envelope.mode is ResponseMode.STRUCTURED
    assert envelope.validation.valid
    assert envelope.document is response.structured_document


@pytest.mark.parametrize("target_name,target_id,language,source,_bad,_foreign", CASES)
def test_structured_wrong_target_is_rejected(
    target_name: str,
    target_id: TargetId,
    language: Literal["python", "sql"],
    source: str,
    _bad: str,
    _foreign: str,
) -> None:
    other = TargetId.SPARK_SQL if target_id is TargetId.PYSPARK else TargetId.PYSPARK
    envelope = ResponseAcceptanceService().envelope(
        ProviderResponse(
            raw_message="wrong target",
            structured_document=_document(other, language, source),
        ),
        resolve_local_target(target_name),
        known_chunk_ids={"chunk-1"},
    )
    assert TargetIssueCode.TARGET_MISMATCH in _codes(envelope.validation)


@pytest.mark.parametrize("target_name,target_id,_language,source,_bad,foreign", CASES)
def test_structured_foreign_language_is_rejected(
    target_name: str,
    target_id: TargetId,
    _language: Literal["python", "sql"],
    source: str,
    _bad: str,
    foreign: Literal["python", "sql"],
) -> None:
    envelope = ResponseAcceptanceService().envelope(
        ProviderResponse(
            raw_message="foreign language",
            structured_document=_document(target_id, foreign, source),
        ),
        resolve_local_target(target_name),
        known_chunk_ids={"chunk-1"},
    )
    assert TargetIssueCode.FOREIGN_LANGUAGE in _codes(envelope.validation)


@pytest.mark.parametrize("target_name,target_id,language,_source,bad,_foreign", CASES)
def test_structured_syntax_error_is_rejected(
    target_name: str,
    target_id: TargetId,
    language: Literal["python", "sql"],
    _source: str,
    bad: str,
    _foreign: str,
) -> None:
    envelope = ResponseAcceptanceService().envelope(
        ProviderResponse(
            raw_message="bad syntax",
            structured_document=_document(target_id, language, bad),
        ),
        resolve_local_target(target_name),
        known_chunk_ids={"chunk-1"},
    )
    assert TargetIssueCode.SYNTAX_ERROR in _codes(envelope.validation)


@pytest.mark.parametrize(
    ("target_name", "target_id", "language", "source", "_bad", "_foreign"),
    CASES,
)
@pytest.mark.parametrize("explicit_fence", [True, False])
def test_raw_fallback_uses_the_same_validation_path(
    target_name: str,
    target_id: TargetId,
    language: Literal["python", "sql"],
    source: str,
    _bad: str,
    _foreign: str,
    explicit_fence: bool,
) -> None:
    target = resolve_local_target(target_name)
    envelope = ResponseAcceptanceService().envelope(
        ProviderResponse(
            raw_message=_raw(language if explicit_fence else "", source),
            structured_error="provider ignored schema",
        ),
        target,
        known_chunk_ids={"chunk-1"},
    )
    assert envelope.mode is ResponseMode.RAW_FALLBACK
    assert envelope.validation.valid
    assert envelope.document is not None
    assert envelope.document.target is target_id
    assert envelope.raw_message.startswith("## Analysis")


@pytest.mark.parametrize("target_name,target_id,_language,source,_bad,foreign", CASES)
def test_raw_explicit_wrong_fence_is_rejected(
    target_name: str,
    target_id: TargetId,
    _language: str,
    source: str,
    _bad: str,
    foreign: str,
) -> None:
    envelope = ResponseAcceptanceService().envelope(
        ProviderResponse(
            raw_message=_raw(foreign, source),
            structured_error="schema failed",
        ),
        resolve_local_target(target_name),
        known_chunk_ids={"chunk-1"},
    )
    assert envelope.document is not None
    assert envelope.document.target is target_id
    assert TargetIssueCode.FOREIGN_LANGUAGE in _codes(envelope.validation)


def test_raw_mixed_target_fences_fail_instead_of_being_guessed() -> None:
    raw = _raw("sql", "SELECT 1").replace(
        "## Risks",
        f"{FENCE}python\nx = 1\n{FENCE}\n\n## Risks",
    )
    envelope = ResponseAcceptanceService().envelope(
        ProviderResponse(raw_message=raw, structured_error="schema failed"),
        resolve_local_target("sql"),
        known_chunk_ids={"chunk-1"},
    )
    assert TargetIssueCode.MIXED_TARGETS in _codes(envelope.validation)
    assert TargetIssueCode.FOREIGN_LANGUAGE in _codes(envelope.validation)


@pytest.mark.parametrize(
    "raw",
    [
        "## Analysis\n\nNo translation section.",
        "provider returned malformed prose without target code",
        "",
    ],
)
def test_missing_or_malformed_raw_translation_is_retained_but_rejected(raw: str) -> None:
    envelope = ResponseAcceptanceService().envelope(
        ProviderResponse(raw_message=raw, structured_error="schema failed"),
        resolve_local_target("sql"),
        known_chunk_ids=set(),
    )
    assert envelope.raw_message == raw
    assert envelope.document is not None
    assert TargetIssueCode.EMPTY_CODE in _codes(envelope.validation)


def test_raw_scala_fence_is_retained_for_audit_and_rejected() -> None:
    raw = _raw("scala", "val result = source")
    envelope = ResponseAcceptanceService().envelope(
        ProviderResponse(raw_message=raw, structured_error="schema failed"),
        resolve_local_target("pyspark"),
        known_chunk_ids={"chunk-1"},
    )
    assert envelope.raw_message == raw
    assert TargetIssueCode.FOREIGN_LANGUAGE in _codes(envelope.validation)


def test_multi_member_cells_require_known_chunk_attribution() -> None:
    target = resolve_local_target("pyspark")
    envelope = ResponseAcceptanceService().envelope(
        ProviderResponse(
            raw_message="missing attribution",
            structured_document=_document(
                TargetId.PYSPARK,
                "python",
                "x = 1",
                chunk_id=None,
            ),
        ),
        target,
        known_chunk_ids={"chunk-1", "chunk-2"},
    )
    assert TargetIssueCode.UNKNOWN_CHUNK in _codes(envelope.validation)


def test_canonical_markdown_normalizes_mapping_risks_and_target_code() -> None:
    target = resolve_local_target("sql")
    normalized = normalize_raw_response(_raw("sql", "SELECT 1"), target)
    assert normalized.document.mapping[0].sas_construct == "DATA step"
    assert normalized.document.risks[0].severity is RiskSeverity.P2
    envelope = ResponseAcceptanceService().envelope(
        ProviderResponse(
            raw_message="structured",
            structured_document=normalized.document,
        ),
        target,
        known_chunk_ids=set(),
    )
    assert envelope.validation.valid
    assert "## Translation" in normalized.document.to_markdown()


def test_canonical_markdown_renders_mapping_differences_and_markdown_cells() -> None:
    document = TranslationDocument(
        target=TargetId.PYSPARK,
        analysis="Analyze.",
        mapping=(
            MappingEntry(
                sas_construct="PROC SORT",
                equivalent="orderBy",
                difference="Null ordering requires review.",
            ),
        ),
        cells=(
            TranslationCell(
                kind=TranslationCellKind.MARKDOWN,
                source="Keep this explanation.",
            ),
            TranslationCell(
                kind=TranslationCellKind.CODE,
                language="python",
                source="result = source.orderBy('id')",
            ),
        ),
    )
    markdown = document.to_markdown()
    assert "orderBy — Null ordering requires review." in markdown
    assert "Keep this explanation." in markdown


def test_markdown_cell_cannot_declare_a_code_language() -> None:
    with pytest.raises(ValidationError, match="markdown cells"):
        TranslationCell(
            kind=TranslationCellKind.MARKDOWN,
            source="Explanation.",
            language="sql",
        )


def test_response_envelope_rejects_inconsistent_contract_states() -> None:
    sql_target = resolve_local_target("sql")
    pyspark_target = resolve_local_target("pyspark")
    document = _document(TargetId.SPARK_SQL, "sql", "SELECT 1")
    accepted = ResponseValidationResult.accepted(TargetId.SPARK_SQL)
    invalid = ResponseValidationResult(
        valid=False,
        resolved_target=TargetId.SPARK_SQL,
        reported_target=TargetId.PYSPARK,
        issues=(
            TargetValidationIssue(
                code=TargetIssueCode.TARGET_MISMATCH,
                message="wrong target",
            ),
        ),
    )

    invalid_values = (
        {
            "mode": ResponseMode.STRUCTURED,
            "structured_error": "unexpected",
            "resolved_target": sql_target,
            "validation": accepted,
            "document": document,
        },
        {
            "mode": ResponseMode.RAW_FALLBACK,
            "resolved_target": sql_target,
            "validation": accepted,
            "document": document,
        },
        {
            "mode": ResponseMode.STRUCTURED,
            "resolved_target": sql_target,
            "validation": invalid,
            "document": document,
        },
        {
            "mode": ResponseMode.STRUCTURED,
            "resolved_target": pyspark_target,
            "validation": accepted,
            "document": document,
        },
        {
            "mode": ResponseMode.STRUCTURED,
            "resolved_target": sql_target,
            "validation": accepted,
        },
    )
    for values in invalid_values:
        with pytest.raises(ValidationError):
            ResponseEnvelope(raw_message="raw", **values)


def test_raw_prose_and_chunk_heading_preserve_cell_order_and_attribution() -> None:
    tick = chr(96)
    raw = (
        "## Analysis\n\nOne.\n\n"
        "## Analysis\n\nTwo.\n\n"
        "## Translation\n\nA note before code.\n\n"
        f"### Chunk {tick}chunk-1{tick}\n\n"
        f"{FENCE}sql\nSELECT 1\n{FENCE}\n"
    )
    normalized = normalize_raw_response(raw, resolve_local_target("sql"))
    assert normalized.document.analysis == "One.\n\nTwo."
    assert [cell.kind for cell in normalized.document.cells] == [
        TranslationCellKind.MARKDOWN,
        TranslationCellKind.CODE,
    ]
    assert normalized.document.cells[1].chunk_id == "chunk-1"


def test_unknown_chunk_empty_code_and_empty_sql_statement_are_rejected() -> None:
    service = ResponseAcceptanceService()
    target = resolve_local_target("sql")
    unknown = service.envelope(
        ProviderResponse(
            raw_message="unknown chunk",
            structured_document=_document(
                TargetId.SPARK_SQL,
                "sql",
                "SELECT 1",
                chunk_id="missing",
            ),
        ),
        target,
        known_chunk_ids={"chunk-1"},
    )
    assert TargetIssueCode.UNKNOWN_CHUNK in _codes(unknown.validation)

    for source in (" ", ";"):
        invalid = service.envelope(
            ProviderResponse(
                raw_message="invalid SQL",
                structured_document=_document(
                    TargetId.SPARK_SQL,
                    "sql",
                    source,
                    chunk_id="chunk-1",
                ),
            ),
            target,
            known_chunk_ids={"chunk-1"},
        )
        expected = (
            TargetIssueCode.EMPTY_CODE
            if not source.strip()
            else TargetIssueCode.SYNTAX_ERROR
        )
        assert expected in _codes(invalid.validation)


def test_raw_fallback_supplies_a_structured_error_when_provider_omits_one() -> None:
    envelope = ResponseAcceptanceService().envelope(
        ProviderResponse(raw_message=_raw("sql", "SELECT 1")),
        resolve_local_target("sql"),
        known_chunk_ids=set(),
    )
    assert envelope.validation.valid
    assert envelope.structured_error == (
        "provider returned no target-bearing structured document"
    )


def test_response_outcome_contract_rejects_inconsistent_states() -> None:
    service = ResponseAcceptanceService()
    target = resolve_local_target("sql")
    valid_envelope = service.envelope(
        ProviderResponse(
            raw_message="valid",
            structured_document=_document(TargetId.SPARK_SQL, "sql", "SELECT 1"),
        ),
        target,
        known_chunk_ids={"chunk-1"},
    )
    invalid_envelope = service.envelope(
        ProviderResponse(raw_message="invalid", structured_error="schema failed"),
        target,
        known_chunk_ids={"chunk-1"},
    )
    valid_attempt = ResponseAttempt(attempt=1, envelope=valid_envelope)
    invalid_attempt = ResponseAttempt(attempt=1, envelope=invalid_envelope)
    document = valid_envelope.document
    assert document is not None

    invalid_values = (
        {
            "status": ItemStatus.ACCEPTED,
            "attempts": (),
            "accepted_document": document,
        },
        {
            "status": ItemStatus.ACCEPTED,
            "attempts": (valid_attempt,),
        },
        {
            "status": ItemStatus.ACCEPTED,
            "attempts": (valid_attempt,),
            "accepted_document": document,
            "error": "unexpected",
        },
        {
            "status": ItemStatus.FAILED,
            "attempts": (valid_attempt,),
            "accepted_document": document,
            "error": "failed",
        },
        {
            "status": ItemStatus.FAILED,
            "attempts": (invalid_attempt,),
        },
        {
            "status": ItemStatus.RUNNING,
            "attempts": (invalid_attempt,),
        },
    )
    for values in invalid_values:
        with pytest.raises(ValidationError):
            ResponseAcceptanceOutcome(item_id="item", **values)


def test_retry_repairs_invalid_response_and_serializes_for_resume() -> None:
    target = resolve_local_target("sql")
    received_feedback: list[tuple[TargetIssueCode, ...]] = []

    async def request(attempt: int, feedback):
        received_feedback.append(tuple(issue.code for issue in feedback))
        raw = _raw("python", "x = 1") if attempt == 1 else _raw("sql", "SELECT 1")
        return ProviderResponse(raw_message=raw, structured_error="schema failed")

    outcome = asyncio.run(
        ResponseAcceptanceService().accept(
            item_id="item-1",
            target=target,
            known_chunk_ids={"chunk-1"},
            request_attempt=request,
            max_retries=1,
        )
    )
    assert outcome.status is ItemStatus.ACCEPTED
    assert len(outcome.attempts) == 2
    assert received_feedback[0] == ()
    assert TargetIssueCode.FOREIGN_LANGUAGE in received_feedback[1]
    assert len(outcome.runnable_code_cells) == 1
    assert "## Translation" in outcome.canonical_markdown()
    assert ResponseAcceptanceOutcome.from_json(outcome.to_json()) == outcome


def test_exhausted_retry_fails_item_and_publishes_no_runnable_code() -> None:
    raw = _raw("scala", "val result = source")

    async def request(_attempt: int, _feedback):
        return ProviderResponse(raw_message=raw, structured_error="schema failed")

    outcome = asyncio.run(
        ResponseAcceptanceService().accept(
            item_id="item-2",
            target=resolve_local_target("pyspark"),
            known_chunk_ids={"chunk-1"},
            request_attempt=request,
            max_retries=1,
        )
    )
    assert outcome.status is ItemStatus.FAILED
    assert len(outcome.attempts) == 2
    assert all(attempt.envelope.raw_message == raw for attempt in outcome.attempts)
    assert outcome.runnable_code_cells == ()
    with pytest.raises(ResponseContractError):
        outcome.canonical_markdown()
    assert ResponseAcceptanceOutcome.from_json(outcome.to_json()) == outcome


def test_negative_retry_budget_is_rejected_before_provider_invocation() -> None:
    async def request(_attempt: int, _feedback):
        raise AssertionError("provider must not be called")

    with pytest.raises(ValueError, match="max_retries"):
        asyncio.run(
            ResponseAcceptanceService().accept(
                item_id="item-3",
                target=resolve_local_target("sql"),
                known_chunk_ids=set(),
                request_attempt=request,
                max_retries=-1,
            )
        )
