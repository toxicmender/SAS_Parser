"""One budgeted, normalized, and token-accounted provider attempt."""

from __future__ import annotations

from collections.abc import Collection

from pydantic import model_validator

from sas_migrate.application.ports import LLMPort
from sas_migrate.application.response_acceptance import ResponseAcceptanceService
from sas_migrate.core.ids import ItemId, RunId, ThreadId
from sas_migrate.core.models import VersionedContract
from sas_migrate.core.responses import ResponseEnvelope
from sas_migrate.core.targets import ResolvedTarget
from sas_migrate.core.tokens import (
    CallTokenRecord,
    PromptAssembly,
    PromptBudgetDecision,
    TokenBudgetPolicy,
    TokenCallLedger,
)

from .budgeting import TokenBudgetEnforcer
from .token_accounting import TokenAccountingService
from .token_audit import TokenAuditPersistenceService


class BudgetedResponseAttempt(VersionedContract):
    sent: bool
    budget: PromptBudgetDecision
    envelope: ResponseEnvelope | None = None
    token_record: CallTokenRecord | None = None
    audit_location: str
    error: str | None = None

    @model_validator(mode="after")
    def validate_attempt(self) -> BudgetedResponseAttempt:
        if self.sent:
            if not self.budget.allowed or self.envelope is None or self.token_record is None:
                raise ValueError("sent attempt requires an allowed budget and accounted response")
            if self.error is not None:
                raise ValueError("sent attempt cannot carry a preflight error")
        else:
            if self.budget.allowed or self.envelope is not None or self.token_record is not None:
                raise ValueError("rejected preflight cannot contain provider response data")
            if not self.error:
                raise ValueError("rejected preflight must explain the failure")
        return self


class BudgetedResponseAttemptService:
    def __init__(
        self,
        *,
        llm: LLMPort,
        budgets: TokenBudgetEnforcer,
        accounting: TokenAccountingService,
        audit: TokenAuditPersistenceService,
        responses: ResponseAcceptanceService | None = None,
    ) -> None:
        self._llm = llm
        self._budgets = budgets
        self._accounting = accounting
        self._audit = audit
        self._responses = responses or ResponseAcceptanceService()

    async def invoke(
        self,
        *,
        run_id: RunId,
        thread_id: ThreadId,
        item_id: ItemId,
        attempt: int,
        target: ResolvedTarget,
        known_chunk_ids: Collection[str],
        prompt: PromptAssembly,
        policy: TokenBudgetPolicy,
        ledger: TokenCallLedger | None = None,
    ) -> BudgetedResponseAttempt:
        decision = self._budgets.preflight(prompt, policy, ledger=ledger)
        if not decision.allowed:
            location = await self._audit.persist(
                run_id=run_id,
                thread_id=thread_id,
                item_id=item_id,
                attempt=attempt,
                target=target.target,
                decision=decision,
                call_record=None,
            )
            codes = ", ".join(issue.code.value for issue in decision.violations)
            return BudgetedResponseAttempt(
                sent=False,
                budget=decision,
                audit_location=location,
                error=f"token budget rejected prompt: {codes}",
            )

        response = await self._llm.invoke(decision.prompt, target, attempt=attempt)
        envelope = self._responses.envelope(
            response,
            target,
            known_chunk_ids=known_chunk_ids,
        )
        record = self._accounting.record(
            run_id=run_id,
            thread_id=thread_id,
            item_id=item_id,
            attempt=attempt,
            target=target.target,
            prompt=decision.prompt,
            response=response,
            document=envelope.document,
            accepted_attempt=envelope.validation.valid,
        )
        location = await self._audit.persist(
            run_id=run_id,
            thread_id=thread_id,
            item_id=item_id,
            attempt=attempt,
            target=target.target,
            decision=decision,
            call_record=record,
        )
        return BudgetedResponseAttempt(
            sent=True,
            budget=decision,
            envelope=envelope,
            token_record=record,
            audit_location=location,
        )


__all__ = ["BudgetedResponseAttempt", "BudgetedResponseAttemptService"]
