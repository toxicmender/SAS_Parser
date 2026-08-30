"""Budgeted, resumable corpus translation orchestration."""

from __future__ import annotations

from pydantic import Field, model_validator

from sas_migrate.application.ports import MemoryPort, TokenRecordRepository
from sas_migrate.core.errors import ContractError
from sas_migrate.core.ids import ItemId, RunId, ThreadId
from sas_migrate.core.models import VersionedContract
from sas_migrate.core.responses import TranslationDocument
from sas_migrate.core.runs import ItemStatus, RunState, RunStatus
from sas_migrate.core.targets import ResolvedTarget
from sas_migrate.core.tokens import TokenBudgetPolicy, TokenCallLedger

from .artifacts import (
    ArtifactLocator,
    NotebookTranslation,
    TranslationArtifactService,
)
from .attempts import BudgetedResponseAttemptService
from .models import TranslationItem
from .prompting import PromptContext, TranslationPromptBuilder
from .run_state import RunStateService


class TranslateCorpusRequest(VersionedContract):
    run_id: RunId
    thread_id: ThreadId
    target: ResolvedTarget
    items: tuple[TranslationItem, ...] = Field(min_length=1)
    policy: TokenBudgetPolicy
    context: PromptContext = Field(default_factory=PromptContext)
    max_attempts: int = Field(default=2, ge=1)
    resume: bool = False

    @model_validator(mode="after")
    def validate_item_ids(self) -> TranslateCorpusRequest:
        item_ids = [item.item_id for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("translation corpus item ids must be unique")
        return self


class TranslationItemOutcome(VersionedContract):
    item_id: ItemId
    status: ItemStatus
    target: ResolvedTarget
    attempts: int = Field(ge=0)
    recovered: bool = False
    document: TranslationDocument | None = None
    error: str | None = None
    artifacts: tuple[ArtifactLocator, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_outcome(self) -> TranslationItemOutcome:
        if self.status is ItemStatus.ACCEPTED:
            if self.document is None or self.error is not None:
                raise ValueError("accepted translation requires a document and no error")
        elif self.status is ItemStatus.FAILED:
            if self.document is not None or not self.error:
                raise ValueError("failed translation requires an error and no document")
        else:
            raise ValueError("translation outcome must be accepted or failed")
        return self


class TranslationRunOutcome(VersionedContract):
    state: RunState
    items: tuple[TranslationItemOutcome, ...]
    tokens: TokenCallLedger
    artifacts: tuple[ArtifactLocator, ...] = Field(default_factory=tuple)


class TranslateCorpus:
    def __init__(
        self,
        *,
        attempts: BudgetedResponseAttemptService,
        prompts: TranslationPromptBuilder,
        artifacts: TranslationArtifactService,
        runs: RunStateService,
        memory: MemoryPort,
        token_records: TokenRecordRepository,
    ) -> None:
        self._attempts = attempts
        self._prompts = prompts
        self._artifacts = artifacts
        self._runs = runs
        self._memory = memory
        self._token_records = token_records

    async def run(self, request: TranslateCorpusRequest) -> TranslationRunOutcome:
        initial = await self._runs.start(
            request.run_id,
            request.thread_id,
            request.target,
            allow_existing=request.resume,
        )
        known_items = {item.item_id for item in request.items}
        unexpected = {item.item_id for item in initial.items} - known_items
        if unexpected:
            values = ", ".join(sorted(unexpected))
            raise ContractError(f"stored run contains items outside the corpus: {values}")

        persisted = await self._token_records.records(
            request.run_id,
            request.thread_id,
        )
        records = [
            record.model_copy(update={"recovered": True})
            if request.resume
            else record
            for record in persisted
        ]
        outcomes: list[TranslationItemOutcome] = []
        notebook_items: list[NotebookTranslation] = []
        run_artifacts: list[ArtifactLocator] = []
        failed = False
        made_attempt = False

        for item in request.items:
            target = self._prompts.target_for(item, request.target)
            state = await self._required_state(request.run_id, request.thread_id)
            item_state = next(
                (candidate for candidate in state.items if candidate.item_id == item.item_id),
                None,
            )
            if item_state is not None and item_state.status is ItemStatus.ACCEPTED:
                recovered = await self._memory.accepted_response(
                    request.run_id,
                    request.thread_id,
                    item.item_id,
                )
                if (
                    recovered is not None
                    and recovered.validation.valid
                    and recovered.resolved_target == target
                    and recovered.document is not None
                ):
                    canonical = await self._artifacts.persist_canonical(
                        request.run_id,
                        item.item_id,
                        recovered.document,
                    )
                    outcome = TranslationItemOutcome(
                        item_id=item.item_id,
                        status=ItemStatus.ACCEPTED,
                        target=target,
                        attempts=item_state.attempt,
                        recovered=True,
                        document=recovered.document,
                        artifacts=(canonical,),
                    )
                    outcomes.append(outcome)
                    notebook_items.append(
                        NotebookTranslation(
                            item=item,
                            target=target,
                            document=recovered.document,
                            recovered=True,
                        )
                    )
                    run_artifacts.append(canonical)
                    continue
                await self._runs.rewind(
                    request.run_id,
                    request.thread_id,
                    tuple(candidate.item_id for candidate in request.items),
                    item.item_id,
                )
                state = await self._required_state(request.run_id, request.thread_id)
                item_state = next(
                    candidate for candidate in state.items if candidate.item_id == item.item_id
                )

            first_attempt = (item_state.attempt if item_state is not None else 0) + 1
            feedback: tuple[str, ...] = ()
            item_artifacts: list[ArtifactLocator] = []
            accepted_document: TranslationDocument | None = None
            final_error = "maximum attempts already exhausted"
            last_attempt = first_attempt - 1

            for attempt_number in range(first_attempt, request.max_attempts + 1):
                made_attempt = True
                last_attempt = attempt_number
                prompt = self._prompts.build(
                    item,
                    target,
                    context=request.context,
                    retry_feedback=feedback,
                )
                await self._runs.item_started(
                    request.run_id,
                    request.thread_id,
                    item.item_id,
                    attempt_number,
                )
                attempt = await self._attempts.invoke(
                    run_id=request.run_id,
                    thread_id=request.thread_id,
                    item_id=item.item_id,
                    attempt=attempt_number,
                    target=target,
                    known_chunk_ids=item.known_chunk_ids,
                    prompt=prompt,
                    policy=request.policy,
                    ledger=TokenCallLedger(records=tuple(records)),
                )
                attempt_artifacts = await self._artifacts.persist_attempt(
                    request.run_id,
                    item,
                    target,
                    attempt_number,
                    attempt.budget.prompt,
                    attempt.envelope,
                )
                item_artifacts.extend(attempt_artifacts)
                run_artifacts.extend(attempt_artifacts)
                await self._runs.attempt_completed(
                    request.run_id,
                    request.thread_id,
                    item.item_id,
                    attempt_number,
                    valid=bool(
                        attempt.envelope is not None
                        and attempt.envelope.validation.valid
                    ),
                    sent=attempt.sent,
                )
                if attempt.token_record is not None:
                    records.append(attempt.token_record)
                if (
                    attempt.envelope is not None
                    and attempt.envelope.validation.valid
                    and attempt.envelope.document is not None
                ):
                    accepted_document = attempt.envelope.document
                    await self._memory.remember_accepted(
                        request.run_id,
                        request.thread_id,
                        item.item_id,
                        attempt.envelope,
                    )
                    await self._runs.item_accepted(
                        request.run_id,
                        request.thread_id,
                        item.item_id,
                        attempt_number,
                    )
                    canonical = await self._artifacts.persist_canonical(
                        request.run_id,
                        item.item_id,
                        accepted_document,
                    )
                    item_artifacts.append(canonical)
                    run_artifacts.append(canonical)
                    break
                if not attempt.sent:
                    final_error = attempt.error or "token budget rejected prompt"
                    break
                if attempt.envelope is not None:
                    feedback = tuple(
                        issue.message for issue in attempt.envelope.validation.issues
                    )
                    final_error = (
                        "response validation failed: " + "; ".join(feedback)
                    )

            if accepted_document is not None:
                outcomes.append(
                    TranslationItemOutcome(
                        item_id=item.item_id,
                        status=ItemStatus.ACCEPTED,
                        target=target,
                        attempts=last_attempt,
                        document=accepted_document,
                        artifacts=tuple(item_artifacts),
                    )
                )
                notebook_items.append(
                    NotebookTranslation(
                        item=item,
                        target=target,
                        document=accepted_document,
                    )
                )
                continue

            failed = True
            event_attempt = max(last_attempt, item_state.attempt if item_state else 1)
            await self._runs.item_failed(
                request.run_id,
                request.thread_id,
                item.item_id,
                event_attempt,
                final_error,
            )
            await self._runs.failed(
                request.run_id,
                request.thread_id,
                final_error,
            )
            outcomes.append(
                TranslationItemOutcome(
                    item_id=item.item_id,
                    status=ItemStatus.FAILED,
                    target=target,
                    attempts=event_attempt,
                    error=final_error,
                    artifacts=tuple(item_artifacts),
                )
            )
            break

        if not failed and (initial.status is not RunStatus.COMPLETED or made_attempt):
            await self._runs.completed(request.run_id, request.thread_id)
        if notebook_items:
            notebooks = await self._artifacts.persist_notebooks(
                request.run_id,
                tuple(notebook_items),
            )
            run_artifacts.extend(notebooks)
        final_state = await self._required_state(request.run_id, request.thread_id)
        return TranslationRunOutcome(
            state=final_state,
            items=tuple(outcomes),
            tokens=TokenCallLedger(records=tuple(records)),
            artifacts=tuple(run_artifacts),
        )

    async def _required_state(self, run_id: RunId, thread_id: ThreadId) -> RunState:
        state = await self._runs.state(run_id, thread_id)
        if state is None:
            raise RuntimeError("run state disappeared during translation")
        return state


__all__ = [
    "TranslateCorpus",
    "TranslateCorpusRequest",
    "TranslationItemOutcome",
    "TranslationRunOutcome",
]
