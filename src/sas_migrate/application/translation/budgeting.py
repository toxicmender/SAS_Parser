"""Shared preflight policy for packing and provider invocation."""

from __future__ import annotations

from collections.abc import Callable

from sas_migrate.core.errors import TokenBudgetError
from sas_migrate.core.tokens import (
    BudgetExceededAction,
    PromptAssembly,
    PromptBudgetDecision,
    PromptComponentDraft,
    TokenBudgetIssue,
    TokenBudgetIssueCode,
    TokenBudgetPolicy,
    TokenCallLedger,
    TokenCategory,
)

from .prompt_assembly import PromptAssembler

_INSTRUCTION_CATEGORIES = frozenset(
    {
        TokenCategory.REFERENCE_GUIDANCE,
        TokenCategory.PROJECT_INSTRUCTIONS,
        TokenCategory.TASK_POLICY,
        TokenCategory.THREAD_NOTES,
    }
)
_HISTORY_CATEGORIES = frozenset(
    {TokenCategory.SELECTED_HISTORY, TokenCategory.ROLLING_SUMMARY}
)


class TokenBudgetEnforcer:
    def __init__(
        self,
        assembler: PromptAssembler,
        *,
        summary_compressor: Callable[[str], str] | None = None,
    ) -> None:
        self._assembler = assembler
        self._summary_compressor = summary_compressor

    @staticmethod
    def _category_total(
        prompt: PromptAssembly,
        categories: frozenset[TokenCategory],
    ) -> int:
        return sum(
            count
            for category, count in prompt.input_by_category().items()
            if category in categories
        )

    def _issues(
        self,
        prompt: PromptAssembly,
        policy: TokenBudgetPolicy,
        *,
        run_tokens_before: int,
    ) -> tuple[tuple[TokenBudgetIssue, ...], tuple[TokenBudgetIssue, ...], int]:
        violations: list[TokenBudgetIssue] = []
        warnings: list[TokenBudgetIssue] = []
        counts = prompt.input_by_category()
        input_total = prompt.estimated_input_total
        projected_run = input_total + policy.reserved_output_tokens + run_tokens_before

        def violation(
            code: TokenBudgetIssueCode,
            actual: int,
            limit: int,
            categories: tuple[TokenCategory, ...],
            message: str,
        ) -> None:
            if actual > limit:
                violations.append(
                    TokenBudgetIssue(
                        code=code,
                        actual_tokens=actual,
                        limit_tokens=limit,
                        categories=categories,
                        message=message,
                    )
                )

        violation(
            TokenBudgetIssueCode.INPUT_LIMIT,
            input_total,
            policy.available_input_tokens,
            tuple(sorted(counts)),
            "estimated prompt exceeds capacity after output reservation and safety margin",
        )
        if policy.max_sas_source_tokens is not None:
            violation(
                TokenBudgetIssueCode.SAS_SOURCE_LIMIT,
                counts.get(TokenCategory.SAS_SOURCE, 0),
                policy.max_sas_source_tokens,
                (TokenCategory.SAS_SOURCE,),
                "SAS source exceeds its required-content cap",
            )
        instruction_total = self._category_total(prompt, _INSTRUCTION_CATEGORIES)
        if policy.max_instruction_tokens is not None:
            violation(
                TokenBudgetIssueCode.INSTRUCTION_LIMIT,
                instruction_total,
                policy.max_instruction_tokens,
                tuple(sorted(_INSTRUCTION_CATEGORIES)),
                "instruction components exceed their category cap",
            )
        history_total = self._category_total(prompt, _HISTORY_CATEGORIES)
        if policy.max_history_tokens is not None:
            violation(
                TokenBudgetIssueCode.HISTORY_LIMIT,
                history_total,
                policy.max_history_tokens,
                tuple(sorted(_HISTORY_CATEGORIES)),
                "history components exceed their category cap",
            )
        if policy.max_run_tokens is not None:
            violation(
                TokenBudgetIssueCode.RUN_LIMIT,
                projected_run,
                policy.max_run_tokens,
                tuple(sorted(counts)),
                "projected run usage exceeds the configured hard cap",
            )
        if (
            policy.instruction_warning_share is not None
            and input_total
            and instruction_total / input_total > policy.instruction_warning_share
        ):
            warnings.append(
                TokenBudgetIssue(
                    code=TokenBudgetIssueCode.INSTRUCTION_SHARE,
                    actual_tokens=instruction_total,
                    limit_tokens=int(input_total * policy.instruction_warning_share),
                    categories=tuple(sorted(_INSTRUCTION_CATEGORIES)),
                    message="instructions exceed their configured share of estimated input",
                )
            )
        return tuple(violations), tuple(warnings), projected_run

    @staticmethod
    def _drafts(prompt: PromptAssembly) -> list[PromptComponentDraft]:
        return [
            PromptComponentDraft(
                category=component.category,
                text=component.text,
                message_role=component.message_role,
                source_id=component.source_id,
                cacheable=component.cacheable,
                ephemeral=component.ephemeral,
            )
            for component in prompt.components
            if component.category is not TokenCategory.CHAT_FRAMING
        ]

    def preflight(
        self,
        prompt: PromptAssembly,
        policy: TokenBudgetPolicy,
        *,
        ledger: TokenCallLedger | None = None,
    ) -> PromptBudgetDecision:
        run_tokens_before = ledger.current_run_total_tokens if ledger is not None else 0
        violations, warnings, projected = self._issues(
            prompt,
            policy,
            run_tokens_before=run_tokens_before,
        )
        if not violations or policy.on_exceeded is BudgetExceededAction.REJECT:
            return PromptBudgetDecision(
                prompt=prompt,
                original_input_tokens=prompt.estimated_input_total,
                run_tokens_before=run_tokens_before,
                projected_run_tokens=projected,
                violations=violations,
                warnings=warnings,
            )

        drafts = self._drafts(prompt)
        removed: list[str] = []
        summary_compressed = False
        while violations:
            index = next(
                (
                    position
                    for position, draft in enumerate(drafts)
                    if draft.category is TokenCategory.SELECTED_HISTORY
                ),
                None,
            )
            if index is None:
                index = next(
                    (
                        position
                        for position in range(len(drafts) - 1, -1, -1)
                        if drafts[position].category
                        is TokenCategory.REFERENCE_GUIDANCE
                    ),
                    None,
                )
            if index is not None:
                removed_draft = drafts.pop(index)
                removed.append(removed_draft.source_id or removed_draft.category.value)
            elif self._summary_compressor is not None and not summary_compressed:
                changed = False
                compressed: list[PromptComponentDraft] = []
                for draft in drafts:
                    if draft.category is TokenCategory.ROLLING_SUMMARY:
                        new_text = self._summary_compressor(draft.text)
                        changed = changed or new_text != draft.text
                        draft = draft.model_copy(update={"text": new_text})
                    compressed.append(draft)
                drafts = compressed
                summary_compressed = changed
                if not changed:
                    break
            else:
                break

            candidate = self._assembler.assemble(drafts)
            violations, warnings, projected = self._issues(
                candidate,
                policy,
                run_tokens_before=run_tokens_before,
            )

        final_prompt = self._assembler.assemble(drafts)
        violations, warnings, projected = self._issues(
            final_prompt,
            policy,
            run_tokens_before=run_tokens_before,
        )
        return PromptBudgetDecision(
            prompt=final_prompt,
            original_input_tokens=prompt.estimated_input_total,
            run_tokens_before=run_tokens_before,
            projected_run_tokens=projected,
            violations=violations,
            warnings=warnings,
            removed_source_ids=tuple(removed),
            summary_compressed=summary_compressed,
        )

    def require(
        self,
        prompt: PromptAssembly,
        policy: TokenBudgetPolicy,
        *,
        ledger: TokenCallLedger | None = None,
    ) -> PromptBudgetDecision:
        decision = self.preflight(prompt, policy, ledger=ledger)
        if not decision.allowed:
            codes = ", ".join(issue.code.value for issue in decision.violations)
            raise TokenBudgetError(f"token budget rejected prompt: {codes}")
        return decision


__all__ = ["TokenBudgetEnforcer"]
