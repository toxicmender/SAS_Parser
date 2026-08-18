"""Phase 6 memory services and Spark-free repository contracts."""

from __future__ import annotations

import asyncio
import pathlib
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from sas_migrate.adapters.memory import InMemoryMemoryRepository
from sas_migrate.application.memory import (
    ChatMessage,
    ChatRole,
    ConversationMemoryService,
    MemoryCandidate,
    MemoryContextAssembler,
    MemoryExtractionService,
    MemoryScope,
    ProposalStatus,
    RelevantHistorySelector,
    RollingSummaryService,
    TaskPolicyService,
    ThreadNoteService,
)
from sas_migrate.core.tokens import (
    MessageRole,
    PromptComponentDraft,
    TokenCategory,
    TokenEstimator,
)


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 19, tzinfo=UTC)

    def now(self) -> datetime:
        current = self.value
        self.value += timedelta(seconds=1)
        return current

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


class _Classifier:
    def __init__(self, candidates: tuple[MemoryCandidate, ...]) -> None:
        self.candidates = candidates
        self.calls = 0

    async def extract(
        self,
        user_content: str,
        assistant_content: str,
    ) -> tuple[MemoryCandidate, ...]:
        del user_content, assistant_content
        self.calls += 1
        return self.candidates


class _BrokenClassifier:
    async def extract(
        self,
        user_content: str,
        assistant_content: str,
    ) -> tuple[MemoryCandidate, ...]:
        del user_content, assistant_content
        raise RuntimeError("classifier unavailable")


def _counter() -> TokenEstimator:
    return TokenEstimator(
        encoding="characters",
        text_counter=len,
        estimator="test",
    )


def _services():
    clock = _Clock()
    identifiers = iter(f"id-{index}" for index in range(1_000))
    repository = InMemoryMemoryRepository(clock, identifier=identifiers.__next__)
    history = ConversationMemoryService(
        repository,
        clock,
        identifier=identifiers.__next__,
    )
    policy = TaskPolicyService(
        repository,
        clock,
        identifier=identifiers.__next__,
    )
    notes = ThreadNoteService(
        repository,
        clock,
        identifier=identifiers.__next__,
    )
    return clock, repository, history, policy, notes, identifiers


def test_chat_contract_rejects_ephemeral_persistence() -> None:
    with pytest.raises(ValidationError, match="ephemeral content"):
        ChatMessage(
            message_id="m1",
            thread_id="t1",
            chat_id="c1",
            sequence=1,
            role=ChatRole.HUMAN,
            content="temporary prompt note",
            created_at=datetime(2026, 8, 19, tzinfo=UTC),
            ephemeral=True,
        )


def test_accepted_turn_is_persisted_once_and_ephemeral_components_never_enter_history() -> (
    None
):
    _, repository, history, _, _, _ = _services()
    note = PromptComponentDraft(
        category=TokenCategory.THREAD_NOTES,
        text="do not persist me",
        message_role=MessageRole.SYSTEM,
        ephemeral=True,
    )

    async def scenario() -> None:
        first = await history.record_accepted_turn(
            thread_id="t1",
            chat_id="c1",
            item_id="item-1",
            user_content="translate item one",
            assistant_content="accepted translation",
            ephemeral_components=(note,),
        )
        second = await history.record_accepted_turn(
            thread_id="t1",
            chat_id="c1",
            item_id="item-1",
            user_content="duplicate",
            assistant_content="duplicate",
            ephemeral_components=(note,),
        )
        assert first == second
        messages = await repository.messages("t1")
        assert [message.role for message in messages] == [
            ChatRole.HUMAN,
            ChatRole.ASSISTANT,
        ]
        assert not any("do not persist" in message.content for message in messages)

    asyncio.run(scenario())


def test_context_keeps_policy_notes_summary_and_history_categories_distinct() -> None:
    clock, repository, history, policy, notes, _ = _services()

    async def scenario():
        await policy.add("sas", "Use catalog-qualified tables.")
        await policy.add("sas", "Never expose credentials.", overridable=False)
        await notes.add("t1", "Use the approved sandbox catalog.", kind="exception")
        await history.record_accepted_turn(
            thread_id="t1",
            chat_id="c1",
            item_id="item-1",
            user_content="translate customer joins",
            assistant_content="joined customer tables",
        )
        await history.record_accepted_turn(
            thread_id="t1",
            chat_id="c1",
            item_id="item-2",
            user_content="translate date arithmetic",
            assistant_content="used date functions",
        )
        summarizer = RollingSummaryService(repository, _counter(), clock)
        await summarizer.refresh(
            "t1",
            trigger_tokens=1,
            keep_recent_messages=2,
            max_summary_tokens=200,
        )
        assembler = MemoryContextAssembler(
            repository,
            RelevantHistorySelector(_counter()),
            _counter(),
            clock,
        )
        return await assembler.assemble(
            thread_id="t1",
            task_id="sas",
            query="date functions",
            max_history_tokens=1_000,
        )

    context = asyncio.run(scenario())
    categories = [component.category for component in context.components]
    assert categories == [
        TokenCategory.TASK_POLICY,
        TokenCategory.ROLLING_SUMMARY,
        TokenCategory.SELECTED_HISTORY,
        TokenCategory.SELECTED_HISTORY,
        TokenCategory.THREAD_NOTES,
    ]
    assert context.components[-1].ephemeral
    assert "cannot override a fixed" in context.components[-1].text
    assert context.policy_fingerprint
    assert context.note_count == 1


def test_relevant_history_prefers_matching_turn_and_respects_whole_turn_budget() -> (
    None
):
    clock, _, history, _, _, _ = _services()

    async def scenario():
        for index, text in enumerate(("customer joins", "date intervals", "unrelated")):
            await history.record_accepted_turn(
                thread_id="t1",
                chat_id="c1",
                item_id=f"item-{index}",
                user_content=text,
                assistant_content=f"answer about {text}",
            )
        repository = history._repository
        return await repository.messages("t1")

    messages = asyncio.run(scenario())
    selected = RelevantHistorySelector(_counter()).select(
        messages,
        "date interval conversion",
        max_tokens=100,
        max_turns=1,
    )
    assert len(selected) == 2
    assert all("date interval" in message.content for message in selected)
    assert selected[0].created_at < selected[1].created_at
    assert clock.value > selected[-1].created_at


def test_snapshot_restore_rewind_fork_retention_and_audit_are_available() -> None:
    clock, repository, history, _, notes, _ = _services()

    async def scenario() -> None:
        await history.record_accepted_turn(
            thread_id="src",
            chat_id="c1",
            item_id="item-1",
            user_content="one",
            assistant_content="answer one",
        )
        await notes.add("src", "temporary", ttl_seconds=60)
        snapshot = await repository.snapshot("src")
        await history.record_accepted_turn(
            thread_id="src",
            chat_id="c1",
            item_id="item-2",
            user_content="two",
            assistant_content="answer two",
        )
        assert await history.rewind("src", after_sequence=2) == 2
        await repository.restore(snapshot)
        assert len(await repository.messages("src")) == 2
        copied = await notes.fork("src", "dst")
        assert copied == 3
        assert len(await repository.messages("dst")) == 2
        inherited = await notes.live("dst")
        assert inherited[0].inherited_from == "src"
        clock.advance(120)
        assert await notes.live("dst") == ()
        removed = await repository.prune(before=clock.now())
        assert removed >= 4
        operations = {event.operation for event in await repository.audit_events()}
        assert {
            "snapshot_created",
            "snapshot_restored",
            "messages_rewound",
            "thread_forked",
            "retention_pruned",
        }.issubset(operations)

    asyncio.run(scenario())


def test_extraction_applies_temporary_notes_and_holds_policy_for_approval() -> None:
    clock, repository, _, policy, notes, identifiers = _services()
    classifier = _Classifier(
        (
            MemoryCandidate(
                text="Use the sandbox catalog.",
                scope=MemoryScope.TEMPORARY,
                kind="exception",
            ),
            MemoryCandidate(
                text="Always qualify table names.",
                scope=MemoryScope.PERMANENT,
                reason="operator preference",
            ),
        )
    )
    extraction = MemoryExtractionService(
        repository,
        classifier,
        policy,
        notes,
        clock,
        identifier=identifiers.__next__,
    )

    async def scenario() -> None:
        result = await extraction.observe(
            thread_id="t1",
            task_id="sas",
            user_content="from now on qualify names",
            assistant_content="understood",
        )
        assert len(result.applied_notes) == 1
        assert len(result.pending_proposals) == 1
        proposal = result.pending_proposals[0]
        assert proposal.status is ProposalStatus.PENDING
        assert (await policy.get("sas")).instructions == ()
        instruction = await extraction.approve(
            proposal.proposal_id,
            overridable=False,
        )
        assert instruction is not None and not instruction.overridable
        approved = await repository.proposal(proposal.proposal_id)
        assert approved is not None
        assert approved.status is ProposalStatus.APPROVED
        second = await extraction.observe(
            thread_id="t1",
            task_id="sas",
            user_content="repeat",
            assistant_content="ok",
        )
        assert second.applied_notes == ()

    asyncio.run(scenario())
    assert classifier.calls == 2


def test_broken_extractor_is_non_fatal() -> None:
    clock, repository, _, policy, notes, identifiers = _services()
    extraction = MemoryExtractionService(
        repository,
        _BrokenClassifier(),
        policy,
        notes,
        clock,
        identifier=identifiers.__next__,
    )
    result = asyncio.run(
        extraction.observe(
            thread_id="t1",
            task_id="sas",
            user_content="remember this",
            assistant_content="ok",
        )
    )
    assert result.applied_notes == () and result.pending_proposals == ()


def test_in_memory_adapter_imports_without_spark() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys; from sas_migrate.adapters.memory import "
            "InMemoryMemoryRepository; "
            "assert 'pyspark' not in sys.modules; assert InMemoryMemoryRepository"
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
