"""Phase 4 prompt assembly, accounting, budgets, and audit gates."""

from __future__ import annotations

import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from sas_migrate.application import PromptAssembler, TokenAccountingService
from sas_migrate.application.ports import ProviderResponse, ProviderTokenUsage
from sas_migrate.core.responses import (
    MappingEntry,
    RiskNote,
    RiskSeverity,
    TranslationCell,
    TranslationCellKind,
    TranslationDocument,
)
from sas_migrate.core.targets import TargetId
from sas_migrate.core.tokens import (
    MessageRole,
    PromptComponentDraft,
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
