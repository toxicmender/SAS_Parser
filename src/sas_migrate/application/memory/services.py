"""Conversation history, policy, notes, summaries, context, and extraction."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import timedelta
from uuid import uuid4

from sas_migrate.application.ports import Clock
from sas_migrate.application.ports.conversation_memory import (
    ConversationMemoryRepository,
    MemoryClassifier,
)
from sas_migrate.core.ids import ItemId, ThreadId
from sas_migrate.core.tokens import (
    MessageRole,
    PromptComponentDraft,
    TokenCategory,
    TokenCounter,
)

from .models import (
    ChatMessage,
    ChatRole,
    ExtractionResult,
    MemoryCandidate,
    MemoryContextResult,
    MemoryScope,
    PolicyInstruction,
    PolicyProposal,
    ProposalStatus,
    RollingSummary,
    TaskPolicySnapshot,
    ThreadNote,
)

_WORD = re.compile(r"[a-zA-Z0-9_]+")


def _tokens(value: str) -> frozenset[str]:
    return frozenset(match.group(0).casefold() for match in _WORD.finditer(value))


class ConversationMemoryService:
    def __init__(
        self,
        repository: ConversationMemoryRepository,
        clock: Clock,
        *,
        identifier: Callable[[], str] | None = None,
    ) -> None:
        self._repository = repository
        self._clock = clock
        self._identifier = identifier or (lambda: uuid4().hex)

    async def record_accepted_turn(
        self,
        *,
        thread_id: ThreadId,
        chat_id: str,
        item_id: ItemId,
        user_content: str,
        assistant_content: str,
        ephemeral_components: Sequence[PromptComponentDraft] = (),
    ) -> tuple[ChatMessage, ChatMessage]:
        del ephemeral_components
        history = await self._repository.messages(thread_id)
        existing = tuple(message for message in history if message.item_id == item_id)
        if existing:
            human = next(
                (message for message in existing if message.role is ChatRole.HUMAN),
                None,
            )
            assistant = next(
                (message for message in existing if message.role is ChatRole.ASSISTANT),
                None,
            )
            if human is None or assistant is None:
                raise RuntimeError("accepted item has an incomplete persisted turn")
            return human, assistant
        sequence = history[-1].sequence + 1 if history else 1
        human = ChatMessage(
            message_id=self._identifier(),
            thread_id=thread_id,
            chat_id=chat_id,
            sequence=sequence,
            role=ChatRole.HUMAN,
            content=user_content,
            created_at=self._clock.now(),
            item_id=item_id,
        )
        assistant = ChatMessage(
            message_id=self._identifier(),
            thread_id=thread_id,
            chat_id=chat_id,
            sequence=sequence + 1,
            role=ChatRole.ASSISTANT,
            content=assistant_content,
            created_at=self._clock.now(),
            item_id=item_id,
        )
        await self._repository.append_message(human)
        await self._repository.append_message(assistant)
        return human, assistant

    async def rewind(self, thread_id: ThreadId, *, after_sequence: int) -> int:
        return await self._repository.rewind_messages(
            thread_id,
            after_sequence=after_sequence,
        )


class TaskPolicyService:
    def __init__(
        self,
        repository: ConversationMemoryRepository,
        clock: Clock,
        *,
        identifier: Callable[[], str] | None = None,
    ) -> None:
        self._repository = repository
        self._clock = clock
        self._identifier = identifier or (lambda: uuid4().hex)

    async def get(self, task_id: str) -> TaskPolicySnapshot:
        stored = await self._repository.policy(task_id)
        return stored or TaskPolicySnapshot(
            task_id=task_id,
            version=0,
            updated_at=self._clock.now(),
        )

    async def add(
        self,
        task_id: str,
        text: str,
        *,
        overridable: bool = True,
        source: str = "operator",
    ) -> PolicyInstruction:
        policy = await self.get(task_id)
        instruction = PolicyInstruction(
            instruction_id=self._identifier(),
            text=text,
            overridable=overridable,
            source=source,
        )
        await self._repository.put_policy(
            policy.model_copy(
                update={
                    "version": policy.version + 1,
                    "instructions": (*policy.instructions, instruction),
                    "updated_at": self._clock.now(),
                }
            )
        )
        return instruction

    async def remove(self, task_id: str, instruction_id: str) -> bool:
        policy = await self.get(task_id)
        retained = tuple(
            instruction
            for instruction in policy.instructions
            if instruction.instruction_id != instruction_id
        )
        if len(retained) == len(policy.instructions):
            return False
        await self._repository.put_policy(
            policy.model_copy(
                update={
                    "version": policy.version + 1,
                    "instructions": retained,
                    "updated_at": self._clock.now(),
                }
            )
        )
        return True


class ThreadNoteService:
    def __init__(
        self,
        repository: ConversationMemoryRepository,
        clock: Clock,
        *,
        identifier: Callable[[], str] | None = None,
    ) -> None:
        self._repository = repository
        self._clock = clock
        self._identifier = identifier or (lambda: uuid4().hex)

    async def add(
        self,
        thread_id: ThreadId,
        text: str,
        *,
        kind: str = "note",
        source: str = "operator",
        ttl_seconds: int | None = None,
    ) -> ThreadNote:
        now = self._clock.now()
        note = ThreadNote(
            note_id=self._identifier(),
            thread_id=thread_id,
            text=text,
            kind=kind,
            source=source,
            created_at=now,
            expires_at=(
                now + timedelta(seconds=ttl_seconds)
                if ttl_seconds is not None
                else None
            ),
        )
        await self._repository.put_note(note)
        return note

    async def live(self, thread_id: ThreadId) -> tuple[ThreadNote, ...]:
        return await self._repository.notes(thread_id, now=self._clock.now())

    async def fork(self, source_thread_id: str, destination_thread_id: str) -> int:
        return await self._repository.fork_thread(
            source_thread_id, destination_thread_id
        )


class RelevantHistorySelector:
    def __init__(self, counter: TokenCounter) -> None:
        self._counter = counter

    def select(
        self,
        messages: tuple[ChatMessage, ...],
        query: str,
        *,
        max_tokens: int,
        max_turns: int = 8,
    ) -> tuple[ChatMessage, ...]:
        turns: list[tuple[ChatMessage, ...]] = []
        current: list[ChatMessage] = []
        for message in messages:
            current.append(message)
            if message.role is ChatRole.ASSISTANT:
                turns.append(tuple(current))
                current = []
        if current:
            turns.append(tuple(current))
        query_tokens = _tokens(query)
        ranked = sorted(
            enumerate(turns),
            key=lambda item: (
                -len(
                    query_tokens.intersection(
                        _tokens(" ".join(message.content for message in item[1]))
                    )
                ),
                -item[0],
            ),
        )
        chosen: list[tuple[int, tuple[ChatMessage, ...]]] = []
        token_total = 0
        for index, turn in ranked:
            turn_tokens = sum(
                self._counter.count_text(message.content) for message in turn
            )
            if token_total + turn_tokens > max_tokens:
                continue
            chosen.append((index, turn))
            token_total += turn_tokens
            if len(chosen) >= max_turns:
                break
        return tuple(
            message
            for _index, turn in sorted(chosen, key=lambda item: item[0])
            for message in turn
        )


class RollingSummaryService:
    def __init__(
        self,
        repository: ConversationMemoryRepository,
        counter: TokenCounter,
        clock: Clock,
    ) -> None:
        self._repository = repository
        self._counter = counter
        self._clock = clock

    async def refresh(
        self,
        thread_id: ThreadId,
        *,
        trigger_tokens: int = 2_000,
        keep_recent_messages: int = 4,
        max_summary_tokens: int = 500,
    ) -> RollingSummary | None:
        messages = await self._repository.messages(thread_id)
        if (
            sum(self._counter.count_text(message.content) for message in messages)
            <= trigger_tokens
        ):
            return await self._repository.summary(thread_id)
        candidates = (
            messages[:-keep_recent_messages] if keep_recent_messages else messages
        )
        if not candidates:
            return await self._repository.summary(thread_id)
        lines = [f"{message.role.value}: {message.content}" for message in candidates]
        content = self._truncate("\n".join(lines), max_summary_tokens)
        summary = RollingSummary(
            thread_id=thread_id,
            content=content,
            through_sequence=candidates[-1].sequence,
            token_count=self._counter.count_text(content),
            updated_at=self._clock.now(),
        )
        await self._repository.put_summary(summary)
        return summary

    def _truncate(self, value: str, max_tokens: int) -> str:
        if self._counter.count_text(value) <= max_tokens:
            return value
        words = value.split()
        while words and self._counter.count_text(" ".join(words)) > max_tokens:
            words.pop()
        return " ".join(words)


class MemoryContextAssembler:
    def __init__(
        self,
        repository: ConversationMemoryRepository,
        selector: RelevantHistorySelector,
        counter: TokenCounter,
        clock: Clock,
    ) -> None:
        self._repository = repository
        self._selector = selector
        self._counter = counter
        self._clock = clock

    async def assemble(
        self,
        *,
        thread_id: ThreadId,
        task_id: str | None,
        query: str,
        max_history_tokens: int = 2_000,
        max_turns: int = 8,
    ) -> MemoryContextResult:
        components: list[PromptComponentDraft] = []
        policy = await self._repository.policy(task_id) if task_id else None
        if policy is not None and policy.instructions:
            lines = [
                f"{index}. {instruction.text}"
                + (" (fixed)" if not instruction.overridable else "")
                for index, instruction in enumerate(policy.instructions, start=1)
            ]
            components.append(
                PromptComponentDraft(
                    category=TokenCategory.TASK_POLICY,
                    text="Standing task policy:\n" + "\n".join(lines),
                    message_role=MessageRole.SYSTEM,
                    source_id=f"policy:{policy.task_id}:v{policy.version}",
                    cacheable=True,
                )
            )
        summary = await self._repository.summary(thread_id)
        if summary is not None:
            components.append(
                PromptComponentDraft(
                    category=TokenCategory.ROLLING_SUMMARY,
                    text=summary.content,
                    message_role=MessageRole.SYSTEM,
                    source_id=f"summary:{thread_id}:{summary.through_sequence}",
                )
            )
        messages = await self._repository.messages(
            thread_id,
            after_sequence=summary.through_sequence if summary else 0,
        )
        selected = self._selector.select(
            messages,
            query,
            max_tokens=max_history_tokens,
            max_turns=max_turns,
        )
        for message in selected:
            components.append(
                PromptComponentDraft(
                    category=TokenCategory.SELECTED_HISTORY,
                    text=message.content,
                    message_role=(
                        MessageRole.USER
                        if message.role is ChatRole.HUMAN
                        else MessageRole.ASSISTANT
                    ),
                    source_id=message.message_id,
                )
            )
        notes = await self._repository.notes(thread_id, now=self._clock.now())
        if notes:
            fixed = bool(
                policy
                and any(
                    not instruction.overridable for instruction in policy.instructions
                )
            )
            note_text = "Thread notes:\n" + "\n".join(
                f"- [{note.kind}] {note.text}" for note in notes
            )
            if policy is not None:
                note_text += (
                    "\nNotes cannot override a fixed standing task instruction."
                    if fixed
                    else "\nThread notes are more local than overridable task policy."
                )
            components.append(
                PromptComponentDraft(
                    category=TokenCategory.THREAD_NOTES,
                    text=note_text,
                    message_role=MessageRole.SYSTEM,
                    source_id=f"notes:{thread_id}",
                    ephemeral=True,
                )
            )
        return MemoryContextResult(
            thread_id=thread_id,
            components=tuple(components),
            selected_message_ids=tuple(message.message_id for message in selected),
            selected_history_tokens=sum(
                self._counter.count_text(message.content) for message in selected
            ),
            policy_fingerprint=policy.fingerprint if policy else "",
            note_count=len(notes),
        )


class MemoryExtractionService:
    def __init__(
        self,
        repository: ConversationMemoryRepository,
        classifier: MemoryClassifier,
        policy: TaskPolicyService,
        notes: ThreadNoteService,
        clock: Clock,
        *,
        identifier: Callable[[], str] | None = None,
        max_candidates: int = 5,
    ) -> None:
        self._repository = repository
        self._classifier = classifier
        self._policy = policy
        self._notes = notes
        self._clock = clock
        self._identifier = identifier or (lambda: uuid4().hex)
        self._max_candidates = max_candidates

    async def observe(
        self,
        *,
        thread_id: ThreadId,
        task_id: str,
        user_content: str,
        assistant_content: str,
    ) -> ExtractionResult:
        try:
            raw = await self._classifier.extract(user_content, assistant_content)
        except (RuntimeError, TypeError, ValueError):
            return ExtractionResult()
        candidates = tuple(
            candidate for candidate in raw if isinstance(candidate, MemoryCandidate)
        )[: self._max_candidates]
        existing_notes = {
            note.text.casefold() for note in await self._notes.live(thread_id)
        }
        applied: list[ThreadNote] = []
        pending: list[PolicyProposal] = []
        for candidate in candidates:
            if candidate.scope is MemoryScope.TEMPORARY:
                if candidate.text.casefold() in existing_notes:
                    continue
                note = await self._notes.add(
                    thread_id,
                    candidate.text,
                    kind=candidate.kind,
                    source="memory_extraction",
                    ttl_seconds=candidate.ttl_seconds,
                )
                applied.append(note)
                existing_notes.add(candidate.text.casefold())
                continue
            proposal = PolicyProposal(
                proposal_id=self._identifier(),
                thread_id=thread_id,
                task_id=task_id,
                text=candidate.text,
                reason=candidate.reason,
                created_at=self._clock.now(),
            )
            await self._repository.put_proposal(proposal)
            pending.append(proposal)
        return ExtractionResult(
            applied_notes=tuple(applied),
            pending_proposals=tuple(pending),
        )

    async def approve(
        self,
        proposal_id: str,
        *,
        overridable: bool = True,
    ) -> PolicyInstruction | None:
        proposal = await self._repository.proposal(proposal_id)
        if proposal is None or proposal.status is not ProposalStatus.PENDING:
            return None
        instruction = await self._policy.add(
            proposal.task_id,
            proposal.text,
            overridable=overridable,
            source="approved_memory",
        )
        await self._repository.put_proposal(
            proposal.model_copy(
                update={
                    "status": ProposalStatus.APPROVED,
                    "resolved_at": self._clock.now(),
                }
            )
        )
        return instruction

    async def reject(self, proposal_id: str) -> bool:
        proposal = await self._repository.proposal(proposal_id)
        if proposal is None or proposal.status is not ProposalStatus.PENDING:
            return False
        await self._repository.put_proposal(
            proposal.model_copy(
                update={
                    "status": ProposalStatus.REJECTED,
                    "resolved_at": self._clock.now(),
                }
            )
        )
        return True


__all__ = [
    "ConversationMemoryService",
    "MemoryContextAssembler",
    "MemoryExtractionService",
    "RelevantHistorySelector",
    "RollingSummaryService",
    "TaskPolicyService",
    "ThreadNoteService",
]
