"""Phase 4 prompt assembly, accounting, budgets, and audit gates."""

from __future__ import annotations

import asyncio
import pathlib
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from sas_migrate.application import (
    BudgetedResponseAttemptService,
    PromptAssembler,
    TokenAccountingService,
    TokenAuditPersistenceService,
    TokenBudgetEnforcer,
)
from sas_migrate.application.ports import (
    ArtifactWrite,
    ProviderResponse,
    ProviderTokenUsage,
)
from sas_migrate.core.errors import TokenBudgetError
from sas_migrate.core.responses import (
    MappingEntry,
    RiskNote,
    RiskSeverity,
    TranslationCell,
    TranslationCellKind,
    TranslationDocument,
)
from sas_migrate.core.targets import ResolvedTarget, TargetId, resolve_local_target
from sas_migrate.core.tokens import (
    BudgetExceededAction,
    MessageRole,
    PromptAssembly,
    PromptComponentDraft,
    TokenBudgetIssueCode,
    TokenBudgetPolicy,
    TokenCallLedger,
    TokenCategory,
    TokenEstimator,
)


def _counter() -> TokenEstimator:
    return TokenEstimator(
        encoding="test_characters",
        text_counter=len,
        estimator="test",
    )


def _prompt(*drafts: PromptComponentDraft):
    return PromptAssembler(_counter()).assemble(drafts)


def _document() -> TranslationDocument:
    return TranslationDocument(
        target=TargetId.SPARK_SQL,
        analysis="analysis",
        mapping=(
            MappingEntry(sas_construct="DATA", equivalent="SELECT", difference="review"),
        ),
        cells=(
            TranslationCell(
                kind=TranslationCellKind.MARKDOWN,
                source="explanation",
            ),
            TranslationCell(
                kind=TranslationCellKind.CODE,
                source="SELECT 1",
                language="sql",
            ),
        ),
        risks=(RiskNote(severity=RiskSeverity.P2, note="check"),),
    )


def test_prompt_assembly_keeps_categories_while_rendering_shared_messages() -> None:
    prompt = _prompt(
        PromptComponentDraft(
            category=TokenCategory.SYSTEM_STATIC,
            text="system",
            message_role=MessageRole.SYSTEM,
        ),
        PromptComponentDraft(
            category=TokenCategory.REFERENCE_GUIDANCE,
            text="reference",
            message_role=MessageRole.SYSTEM,
            source_id="guide-1",
        ),
        PromptComponentDraft(
            category=TokenCategory.PROJECT_INSTRUCTIONS,
            text="project",
            message_role=MessageRole.SYSTEM,
            source_id="project-1",
        ),
        PromptComponentDraft(
            category=TokenCategory.SAS_SOURCE,
            text="data x; run;",
            message_role=MessageRole.USER,
        ),
    )

    messages = prompt.render_messages()
    assert [message.role for message in messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
    ]
    assert messages[0].content == "system\n\nreference\n\nproject"
    counts = prompt.input_by_category()
    assert counts[TokenCategory.REFERENCE_GUIDANCE] == len("reference")
    assert counts[TokenCategory.PROJECT_INSTRUCTIONS] == len("project")
    assert counts[TokenCategory.SAS_SOURCE] == len("data x; run;")
    assert counts[TokenCategory.CHAT_FRAMING] == 9
    assert sum(counts.values()) == prompt.estimated_input_total


def test_retry_feedback_is_counted_only_when_the_retry_contains_it() -> None:
    base = PromptComponentDraft(
        category=TokenCategory.SAS_SOURCE,
        text="source",
        message_role=MessageRole.USER,
    )
    first = _prompt(base)
    retry = _prompt(
        base,
        PromptComponentDraft(
            category=TokenCategory.RETRY_FEEDBACK,
            text="fix the target mismatch",
            message_role=MessageRole.USER,
        ),
    )
    assert TokenCategory.RETRY_FEEDBACK not in first.input_by_category()
    assert retry.input_by_category()[TokenCategory.RETRY_FEEDBACK] == len(
        "fix the target mismatch"
    )


def test_call_records_reconcile_provider_and_normalized_output_categories() -> None:
    prompt = _prompt(
        PromptComponentDraft(
            category=TokenCategory.SAS_SOURCE,
            text="source",
            message_role=MessageRole.USER,
        )
    )
    accounting = TokenAccountingService(_counter())
    record = accounting.record(
        run_id="run-1",
        thread_id="thread-1",
        item_id="item-1",
        attempt=1,
        target=TargetId.SPARK_SQL,
        prompt=prompt,
        response=ProviderResponse(
            raw_message="raw provider envelope with extra content",
            structured_document=_document(),
            usage=ProviderTokenUsage(
                input_tokens=20,
                output_tokens=30,
                cache_read_tokens=5,
                cache_write_tokens=2,
            ),
        ),
        document=_document(),
        accepted_attempt=True,
    )
    assert record.provider_total_tokens == 50
    assert record.provider_input_delta == 20 - prompt.estimated_input_total
    assert record.provider_cache_read_tokens == 5
    assert record.estimated_output_by_category[TokenCategory.CODE_OUTPUT] == len(
        "SELECT 1"
    )
    assert record.estimated_output_by_category[TokenCategory.MARKDOWN_OUTPUT] == len(
        "explanation"
    )


def test_ledger_retains_discarded_retry_cost_and_excludes_recovered_usage() -> None:
    prompt = _prompt(
        PromptComponentDraft(
            category=TokenCategory.SAS_SOURCE,
            text="source",
            message_role=MessageRole.USER,
        )
    )
    accounting = TokenAccountingService(_counter())

    def record(attempt: int, *, accepted: bool, recovered: bool = False):
        return accounting.record(
            run_id="run-1",
            thread_id="thread-1",
            item_id="item-1",
            attempt=attempt,
            target=TargetId.SPARK_SQL,
            prompt=prompt,
            response=ProviderResponse(
                raw_message="SELECT 1",
                usage=ProviderTokenUsage(input_tokens=10, output_tokens=5),
            ),
            document=_document(),
            accepted_attempt=accepted,
            recovered=recovered,
        )

    discarded = record(1, accepted=False)
    accepted = record(2, accepted=True)
    historical = record(3, accepted=True, recovered=True)
    ledger = accounting.ledger((discarded, accepted, historical))
    assert ledger.current_run_total_tokens == 30
    assert ledger.recovered_total_tokens == 15
    assert ledger.retry_overhead_tokens == 15


def test_missing_provider_usage_keeps_estimates_and_fallback_label() -> None:
    counter = TokenEstimator.approximate_for_model("unknown-model")
    prompt = PromptAssembler(counter).assemble(
        (
            PromptComponentDraft(
                category=TokenCategory.SAS_SOURCE,
                text="12345678",
                message_role=MessageRole.USER,
            ),
        )
    )
    record = TokenAccountingService(counter).record(
        run_id="run-1",
        thread_id="thread-1",
        item_id="item-1",
        attempt=1,
        target=TargetId.SPARK_SQL,
        prompt=prompt,
        response=ProviderResponse(raw_message="SELECT 1"),
        document=_document(),
        accepted_attempt=True,
    )
    assert prompt.approximate
    assert prompt.estimator == "character_approximation"
    assert record.provider_total_tokens is None
    assert record.provider_input_delta is None
    assert record.accounted_total_tokens > 0


def test_budget_rejects_required_source_and_reports_instruction_share() -> None:
    prompt = _prompt(
        PromptComponentDraft(
            category=TokenCategory.PROJECT_INSTRUCTIONS,
            text="instructions",
            message_role=MessageRole.SYSTEM,
        ),
        PromptComponentDraft(
            category=TokenCategory.SAS_SOURCE,
            text="required-source",
            message_role=MessageRole.USER,
        ),
    )
    policy = TokenBudgetPolicy(
        max_input_tokens=100,
        reserved_output_tokens=10,
        safety_margin_tokens=5,
        max_sas_source_tokens=5,
        instruction_warning_share=0.2,
    )
    enforcer = TokenBudgetEnforcer(PromptAssembler(_counter()))
    decision = enforcer.preflight(prompt, policy)
    assert not decision.allowed
    assert {issue.code for issue in decision.violations} == {
        TokenBudgetIssueCode.SAS_SOURCE_LIMIT
    }
    assert {issue.code for issue in decision.warnings} == {
        TokenBudgetIssueCode.INSTRUCTION_SHARE
    }
    with pytest.raises(TokenBudgetError, match="sas_source_limit"):
        enforcer.require(prompt, policy)


def test_optional_context_trims_old_history_then_low_ranked_guidance() -> None:
    prompt = _prompt(
        PromptComponentDraft(
            category=TokenCategory.SYSTEM_STATIC,
            text="system",
            message_role=MessageRole.SYSTEM,
        ),
        PromptComponentDraft(
            category=TokenCategory.REFERENCE_GUIDANCE,
            text="high-guide",
            message_role=MessageRole.SYSTEM,
            source_id="guide-high",
        ),
        PromptComponentDraft(
            category=TokenCategory.REFERENCE_GUIDANCE,
            text="low-guide",
            message_role=MessageRole.SYSTEM,
            source_id="guide-low",
        ),
        PromptComponentDraft(
            category=TokenCategory.SELECTED_HISTORY,
            text="old-history",
            message_role=MessageRole.USER,
            source_id="history-old",
        ),
        PromptComponentDraft(
            category=TokenCategory.SELECTED_HISTORY,
            text="new-history",
            message_role=MessageRole.ASSISTANT,
            source_id="history-new",
        ),
        PromptComponentDraft(
            category=TokenCategory.SAS_SOURCE,
            text="source",
            message_role=MessageRole.USER,
        ),
    )
    policy = TokenBudgetPolicy(
        max_input_tokens=48,
        reserved_output_tokens=5,
        safety_margin_tokens=1,
        on_exceeded=BudgetExceededAction.SHRINK_OPTIONAL_CONTEXT,
    )
    decision = TokenBudgetEnforcer(PromptAssembler(_counter())).preflight(
        prompt,
        policy,
    )
    assert decision.allowed
    assert decision.removed_source_ids[:2] == ("history-old", "history-new")
    if "guide-low" in decision.removed_source_ids:
        assert decision.removed_source_ids[-1] == "guide-low"
    assert "guide-high" not in decision.removed_source_ids
    assert decision.final_input_tokens <= policy.available_input_tokens


def test_run_cap_uses_current_calls_and_does_not_charge_recovered_records() -> None:
    prompt = _prompt(
        PromptComponentDraft(
            category=TokenCategory.SAS_SOURCE,
            text="source",
            message_role=MessageRole.USER,
        )
    )
    accounting = TokenAccountingService(_counter())

    def historical(*, recovered: bool):
        return accounting.record(
            run_id="run-1",
            thread_id="thread-1",
            item_id="history" if recovered else "current",
            attempt=1,
            target=TargetId.SPARK_SQL,
            prompt=prompt,
            response=ProviderResponse(
                raw_message="SELECT 1",
                usage=ProviderTokenUsage(input_tokens=10, output_tokens=5),
            ),
            document=_document(),
            accepted_attempt=True,
            recovered=recovered,
        )

    policy = TokenBudgetPolicy(
        max_input_tokens=100,
        reserved_output_tokens=5,
        safety_margin_tokens=1,
        max_run_tokens=30,
    )
    enforcer = TokenBudgetEnforcer(PromptAssembler(_counter()))
    recovered_only = TokenCallLedger(records=(historical(recovered=True),))
    assert enforcer.preflight(prompt, policy, ledger=recovered_only).allowed
    with_current = TokenCallLedger(
        records=(historical(recovered=True), historical(recovered=False))
    )
    rejected = enforcer.preflight(prompt, policy, ledger=with_current)
    assert TokenBudgetIssueCode.RUN_LIMIT in {
        issue.code for issue in rejected.violations
    }


class _RecordRepository:
    def __init__(self) -> None:
        self.records = []

    async def append(self, record) -> None:
        self.records.append(record)


class _ArtifactRepository:
    def __init__(self) -> None:
        self.writes: list[tuple[str, ArtifactWrite]] = []

    async def write(self, run_id: str, artifact: ArtifactWrite) -> str:
        self.writes.append((run_id, artifact))
        return f"memory://{artifact.artifact_id}"


class _LLM:
    def __init__(self, responses: list[ProviderResponse]) -> None:
        self.responses = responses
        self.prompts = []

    async def invoke(
        self,
        prompt: PromptAssembly,
        target: ResolvedTarget,
        *,
        attempt: int,
    ) -> ProviderResponse:
        del target
        self.prompts.append((attempt, prompt))
        return self.responses.pop(0)


def _attempt_service(llm: _LLM):
    counter = _counter()
    record_repository = _RecordRepository()
    artifact_repository = _ArtifactRepository()
    service = BudgetedResponseAttemptService(
        llm=llm,
        budgets=TokenBudgetEnforcer(PromptAssembler(counter)),
        accounting=TokenAccountingService(counter),
        audit=TokenAuditPersistenceService(
            record_repository,
            artifact_repository,
        ),
    )
    return service, record_repository, artifact_repository


def test_budgeted_attempt_does_not_send_rejected_prompt_and_audits_failure() -> None:
    secret = "Bearer super-secret-token"
    prompt = _prompt(
        PromptComponentDraft(
            category=TokenCategory.SAS_SOURCE,
            text=secret,
            message_role=MessageRole.USER,
            source_id="https://source.invalid?sig=secret",
        )
    )
    llm = _LLM([])
    service, records, artifacts = _attempt_service(llm)
    result = asyncio.run(
        service.invoke(
            run_id="run-1",
            thread_id="thread-1",
            item_id="item-1",
            attempt=1,
            target=resolve_local_target("sql"),
            known_chunk_ids=set(),
            prompt=prompt,
            policy=TokenBudgetPolicy(
                max_input_tokens=20,
                reserved_output_tokens=5,
                safety_margin_tokens=1,
            ),
        )
    )
    assert not result.sent
    assert not llm.prompts
    assert not records.records
    audit_content = artifacts.writes[0][1].content.decode("utf-8")
    assert secret not in audit_content
    assert "sig=secret" not in audit_content
    assert "text_sha256" in audit_content


def test_budgeted_retry_records_discarded_and_accepted_attempts() -> None:
    invalid = TranslationDocument(
        target=TargetId.PYSPARK,
        analysis="wrong target",
        cells=(
            TranslationCell(
                kind=TranslationCellKind.CODE,
                source="x = 1",
                language="python",
            ),
        ),
    )
    llm = _LLM(
        [
            ProviderResponse(
                raw_message="invalid",
                structured_document=invalid,
                usage=ProviderTokenUsage(input_tokens=20, output_tokens=5),
            ),
            ProviderResponse(
                raw_message="valid",
                structured_document=_document(),
                usage=ProviderTokenUsage(input_tokens=25, output_tokens=6),
            ),
        ]
    )
    service, records, artifacts = _attempt_service(llm)
    base = PromptComponentDraft(
        category=TokenCategory.SAS_SOURCE,
        text="source",
        message_role=MessageRole.USER,
    )
    policy = TokenBudgetPolicy(
        max_input_tokens=200,
        reserved_output_tokens=20,
        safety_margin_tokens=5,
    )
    first = asyncio.run(
        service.invoke(
            run_id="run-1",
            thread_id="thread-1",
            item_id="item-1",
            attempt=1,
            target=resolve_local_target("sql"),
            known_chunk_ids=set(),
            prompt=_prompt(base),
            policy=policy,
        )
    )
    assert first.sent and first.token_record is not None
    assert not first.token_record.accepted_attempt
    retry_prompt = _prompt(
        base,
        PromptComponentDraft(
            category=TokenCategory.RETRY_FEEDBACK,
            text="return Spark SQL",
            message_role=MessageRole.USER,
        ),
    )
    second = asyncio.run(
        service.invoke(
            run_id="run-1",
            thread_id="thread-1",
            item_id="item-1",
            attempt=2,
            target=resolve_local_target("sql"),
            known_chunk_ids=set(),
            prompt=retry_prompt,
            policy=policy,
            ledger=TokenCallLedger(records=(first.token_record,)),
        )
    )
    assert second.sent and second.token_record is not None
    assert second.token_record.accepted_attempt
    assert TokenCategory.RETRY_FEEDBACK in second.token_record.estimated_input_by_category
    ledger = TokenCallLedger(records=tuple(records.records))
    assert ledger.retry_overhead_tokens == 25
    assert len(artifacts.writes) == 2
