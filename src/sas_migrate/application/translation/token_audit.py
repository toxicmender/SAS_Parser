"""Persist call facts and redacted component manifests."""

from __future__ import annotations

import hashlib

from sas_migrate.application.ports import (
    ArtifactRepository,
    ArtifactWrite,
    TokenRecordRepository,
)
from sas_migrate.core.ids import ItemId, RunId, ThreadId
from sas_migrate.core.targets import TargetId
from sas_migrate.core.tokens import (
    CallTokenRecord,
    PromptBudgetDecision,
    PromptComponentAudit,
    TokenAuditArtifact,
    TokenBudgetAudit,
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class TokenAuditPersistenceService:
    def __init__(
        self,
        records: TokenRecordRepository,
        artifacts: ArtifactRepository,
    ) -> None:
        self._records = records
        self._artifacts = artifacts

    async def persist(
        self,
        *,
        run_id: RunId,
        thread_id: ThreadId,
        item_id: ItemId,
        attempt: int,
        target: TargetId,
        decision: PromptBudgetDecision,
        call_record: CallTokenRecord | None,
    ) -> str:
        components = tuple(
            PromptComponentAudit(
                category=component.category,
                message_role=component.message_role,
                token_count=component.token_count,
                text_sha256=_sha256(component.text),
                source_id_sha256=(
                    _sha256(component.source_id)
                    if component.source_id is not None
                    else None
                ),
                cacheable=component.cacheable,
                ephemeral=component.ephemeral,
            )
            for component in decision.prompt.components
        )
        audit = TokenAuditArtifact(
            run_id=run_id,
            thread_id=thread_id,
            item_id=item_id,
            attempt=attempt,
            target=target,
            budget=TokenBudgetAudit(
                allowed=decision.allowed,
                original_input_tokens=decision.original_input_tokens,
                final_input_tokens=decision.final_input_tokens,
                run_tokens_before=decision.run_tokens_before,
                projected_run_tokens=decision.projected_run_tokens,
                violations=decision.violations,
                warnings=decision.warnings,
                removed_component_count=len(decision.removed_source_ids),
                summary_compressed=decision.summary_compressed,
            ),
            components=components,
            call_record=call_record,
        )
        if call_record is not None:
            await self._records.append(call_record)
        digest = _sha256(f"{run_id}\0{thread_id}\0{item_id}\0{attempt}")[:24]
        return await self._artifacts.write(
            run_id,
            ArtifactWrite(
                artifact_id=f"token-call-{digest}",
                media_type="application/json",
                content=audit.to_json().encode("utf-8"),
                metadata={
                    "kind": "token_call_audit",
                    "allowed": str(decision.allowed).lower(),
                },
            ),
        )


__all__ = ["TokenAuditPersistenceService"]
