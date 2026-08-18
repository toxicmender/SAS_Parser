"""Create attempt records from attributed prompts and normalized responses."""

from __future__ import annotations

from collections import defaultdict

from sas_migrate.application.ports import ProviderResponse
from sas_migrate.core.ids import ItemId, RunId, ThreadId
from sas_migrate.core.responses import TranslationCellKind, TranslationDocument
from sas_migrate.core.targets import TargetId
from sas_migrate.core.tokens import (
    CallTokenRecord,
    PromptAssembly,
    TokenCallLedger,
    TokenCategory,
    TokenCounter,
)


class TokenAccountingService:
    def __init__(self, counter: TokenCounter) -> None:
        self._counter = counter

    def estimate_output(
        self,
        document: TranslationDocument | None,
        *,
        raw_message: str,
    ) -> dict[TokenCategory, int]:
        totals: defaultdict[TokenCategory, int] = defaultdict(int)
        if document is not None:
            totals[TokenCategory.ANALYSIS_OUTPUT] = self._counter.count_text(
                document.analysis
            )
            for entry in document.mapping:
                totals[TokenCategory.MAPPING_OUTPUT] += self._counter.count_text(
                    f"{entry.sas_construct}\n{entry.equivalent}\n{entry.difference}"
                )
            for cell in document.cells:
                category = (
                    TokenCategory.CODE_OUTPUT
                    if cell.kind is TranslationCellKind.CODE
                    else TokenCategory.MARKDOWN_OUTPUT
                )
                totals[category] += self._counter.count_text(cell.source)
            for risk in document.risks:
                totals[TokenCategory.RISK_OUTPUT] += self._counter.count_text(
                    f"{risk.severity.value} {risk.note}"
                )

        represented = sum(totals.values())
        raw_total = self._counter.count_text(raw_message)
        totals[TokenCategory.RAW_OUTPUT_OVERHEAD] = max(0, raw_total - represented)
        return dict(totals)

    def record(
        self,
        *,
        run_id: RunId,
        thread_id: ThreadId,
        item_id: ItemId,
        attempt: int,
        target: TargetId,
        prompt: PromptAssembly,
        response: ProviderResponse,
        document: TranslationDocument | None,
        accepted_attempt: bool,
        recovered: bool = False,
    ) -> CallTokenRecord:
        usage = response.usage
        provider_total = (
            usage.input_tokens + usage.output_tokens
            if usage.input_tokens is not None and usage.output_tokens is not None
            else None
        )
        provider_delta = (
            usage.input_tokens - prompt.estimated_input_total
            if usage.input_tokens is not None
            else None
        )
        return CallTokenRecord(
            run_id=run_id,
            thread_id=thread_id,
            item_id=item_id,
            attempt=attempt,
            target=target,
            estimator=prompt.estimator,
            encoding=prompt.encoding,
            approximate=prompt.approximate,
            estimated_input_by_category=prompt.input_by_category(),
            estimated_input_total=prompt.estimated_input_total,
            provider_input_tokens=usage.input_tokens,
            provider_output_tokens=usage.output_tokens,
            provider_cache_read_tokens=usage.cache_read_tokens,
            provider_cache_write_tokens=usage.cache_write_tokens,
            provider_total_tokens=provider_total,
            provider_input_delta=provider_delta,
            estimated_output_by_category=self.estimate_output(
                document,
                raw_message=response.raw_message,
            ),
            accepted_attempt=accepted_attempt,
            recovered=recovered,
        )

    @staticmethod
    def ledger(records: tuple[CallTokenRecord, ...]) -> TokenCallLedger:
        return TokenCallLedger(records=records)


__all__ = ["TokenAccountingService"]
