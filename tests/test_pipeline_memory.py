"""
Tests for the memory wiring in SasLLMPipeline: the long-term task policy,
short-term thread notes, the chat layer, and memory extraction.

The load-bearing contracts, all asserted here:

* the policy is folded into the (cacheable) SYSTEM prompt, once;
* thread notes are *prompted* but never persisted to the msg:: history, and
  never reach a thread they do not belong to;
* one pipeline instance is one chat on every thread it writes;
* extraction sees only the turn that was kept, never one a failing validator
  rolled back.

Fully offline: a recording chat-model stub captures every prompted message
list, and the extractor's classifier is a FakeListChatModel.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, SystemMessage

from llm_client import LLMClientConfig
from pipeline import SasLLMPipeline
from pipeline.setup import MemorySetup, PromptingSetup, ValidationSetup
from memory.extractor import MemoryExtractor
from memory.policy import TaskPolicy
from memory.store import MemoryHub
from memory.thread_mem import ThreadMemory

SAS = "data out; set in; run;"
NOTE = "Name every output table with a zz_ prefix for this conversation only."


class _RecordingChatModel:
    """Chat-model stub that records every prompted message list."""

    def __init__(self) -> None:
        self.prompts: list[list] = []

    def invoke(self, messages, config=None):
        self.prompts.append(list(messages))
        return AIMessage(f"## Analysis\nresp {len(self.prompts)}")


# The pipeline takes its knobs only as grouped configs. This helper keeps them
# as individual arguments and does the grouping, because what the ~20 call
# sites below are about is which memory wiring produces which prompt — and
# spelling out three constructors at each of them would bury the point.
_MEMORY_KEYS = (
    "memory", "task_id", "task_policy", "thread_memory", "memory_extractor",
    "chat_id", "spark", "delta_table", "window_k", "history_selector",
    "summarizer",
)
_PROMPTING_KEYS = (
    "system_prompt", "structured_output", "prompt_caching", "prompt_builder",
    "user_instructions",
)
_VALIDATION_KEYS = ("validator", "retries")


def _pipeline(llm=None, **kwargs) -> SasLLMPipeline:
    kwargs.setdefault("structured_output", False)
    # An empty instruction set keeps the standing-instructions file (if the
    # developer has one configured) out of these prompts.
    kwargs.setdefault("user_instructions", "")
    memory = {k: kwargs.pop(k) for k in list(kwargs) if k in _MEMORY_KEYS}
    prompting = {k: kwargs.pop(k) for k in list(kwargs) if k in _PROMPTING_KEYS}
    validation = {k: kwargs.pop(k) for k in list(kwargs) if k in _VALIDATION_KEYS}
    model = kwargs.pop("model", "unused-because-llm-injected")
    return SasLLMPipeline(
        llm_config=LLMClientConfig(model=model),
        memory_setup=MemorySetup(**memory),
        prompting=PromptingSetup(**prompting),
        validation=ValidationSetup(**validation),
        llm=llm or _RecordingChatModel(),
        **kwargs,
    )


def _system_texts(prompt) -> list[str]:
    return [str(m.content) for m in prompt if isinstance(m, SystemMessage)]


# ---------------------------------------------------------------------------
# Long-term memory: the task policy
# ---------------------------------------------------------------------------


class TestTaskPolicyWiring:
    def test_task_id_loads_the_stored_policy_into_the_system_prompt(self):
        hub = MemoryHub()
        TaskPolicy("sas", store=hub.kv).add("Name every output table explicitly.")
        llm = _RecordingChatModel()
        pipeline = _pipeline(llm, memory=hub, task_id="sas")
        pipeline.run_text(SAS, thread_id="t1")
        assert "Name every output table explicitly." in pipeline._system_prompt
        # ...and it is the SYSTEM message that carried it, not a later one.
        assert "Name every output table explicitly." in _system_texts(llm.prompts[0])[0]

    def test_no_task_id_leaves_the_system_prompt_untouched(self):
        pipeline = _pipeline()
        assert "Standing instructions for this task" not in pipeline._system_prompt
        assert pipeline.policy_fingerprint is None

    def test_policy_rides_inside_the_cache_breakpoint(self):
        hub = MemoryHub()
        TaskPolicy("sas", store=hub.kv).add("Be explicit.")
        pipeline = _pipeline(
            memory=hub, task_id="sas", model="claude-sonnet-4-5", prompt_caching=True
        )
        system_entry: Any = pipeline._prompt.messages[0]
        block = system_entry.content[0]
        assert block["cache_control"] == {"type": "ephemeral"}
        assert "Be explicit." in block["text"]

    def test_the_policy_snapshot_is_fixed_for_the_pipeline_s_life(self):
        hub = MemoryHub()
        policy = TaskPolicy("sas", store=hub.kv)
        policy.add("First rule.")
        pipeline = _pipeline(memory=hub, task_policy=policy)
        before = pipeline.policy_fingerprint
        policy.add("Second rule.")
        # The running pipeline keeps prompting what it cached...
        assert "Second rule." not in pipeline._system_prompt
        assert pipeline.policy_fingerprint == before
        # ...and the next pipeline — the next chat — picks the edit up.
        assert "Second rule." in _pipeline(memory=hub, task_id="sas")._system_prompt

    def test_an_injected_store_less_policy_is_bound_and_reloaded(self):
        hub = MemoryHub()
        TaskPolicy("sas", store=hub.kv).add("Stored earlier.")
        pipeline = _pipeline(memory=hub, task_policy=TaskPolicy("sas"))
        assert "Stored earlier." in pipeline._system_prompt
        assert pipeline.task_id == "sas"


# ---------------------------------------------------------------------------
# Short-term memory: thread notes
# ---------------------------------------------------------------------------


class TestThreadNoteWiring:
    def test_notes_are_prompted_but_never_persisted(self):
        llm = _RecordingChatModel()
        pipeline = _pipeline(llm, thread_memory=ThreadMemory())
        pipeline.remember("t1", NOTE, kind="exception")
        pipeline.run_text(SAS, thread_id="t1")
        assert any(NOTE in text for text in _system_texts(llm.prompts[0]))
        stored = pipeline.get_thread_messages("t1")
        assert len(stored) == 2  # the item message and the response, nothing else
        assert not any(NOTE in str(m.content) for m in stored)

    def test_notes_do_not_reach_another_thread(self):
        llm = _RecordingChatModel()
        pipeline = _pipeline(llm, thread_memory=ThreadMemory())
        pipeline.remember("t1", NOTE)
        pipeline.run_text(SAS, thread_id="t2")
        assert not any(NOTE in text for text in _system_texts(llm.prompts[0]))

    def test_notes_are_prompted_after_the_history(self):
        # Order matters: the note must read as the most local instruction the
        # model was given, so it sits behind the history, next to the item.
        llm = _RecordingChatModel()
        pipeline = _pipeline(llm, thread_memory=ThreadMemory())
        pipeline.remember("t1", NOTE)
        pipeline.run_text(SAS, thread_id="t1")
        prompt = llm.prompts[0]
        note_index = next(i for i, m in enumerate(prompt) if NOTE in str(m.content))
        assert note_index == len(prompt) - 2  # last thing before the item message

    def test_remember_without_a_thread_memory_raises(self):
        with pytest.raises(RuntimeError, match="no thread_memory"):
            _pipeline().remember("t1", NOTE)

    def test_fork_run_carries_the_notes_onto_the_branch(self):
        notes = ThreadMemory()
        pipeline = _pipeline(thread_memory=notes)
        pipeline.remember("src", "Exception approved by supervisor.", kind="exception")
        pipeline.run_text(SAS, thread_id="src")
        pipeline.fork_run("src", "dst")
        assert notes.texts("dst") == ["Exception approved by supervisor."]


# ---------------------------------------------------------------------------
# The chat layer
# ---------------------------------------------------------------------------


class TestChatWiring:
    def test_one_pipeline_instance_is_one_chat(self):
        hub = MemoryHub()
        first = _pipeline(memory=hub)
        first.run_text(SAS, thread_id="t1")
        chats = first.get_chats("t1")
        assert [c["chat_id"] for c in chats] == [first.chat_id]
        assert chats[0]["message_count"] == 2

    def test_resuming_a_thread_from_a_new_instance_opens_a_new_chat(self):
        hub = MemoryHub()
        first = _pipeline(memory=hub)
        first.run_text(SAS, thread_id="t1")
        second = _pipeline(memory=hub)
        second.run_text("data other; set in; run;", thread_id="t1")
        chats = second.get_chats("t1")
        assert [c["chat_id"] for c in chats] == [first.chat_id, second.chat_id]
        assert [c["message_count"] for c in chats] == [2, 2]
        # One thread throughout — chats index the history, they do not split it.
        assert len(second.get_thread_messages("t1")) == 4

    def test_an_explicit_chat_id_is_honoured(self):
        pipeline = _pipeline(chat_id="nightly-2026-07-27")
        pipeline.run_text(SAS, thread_id="t1")
        assert pipeline.get_chats("t1")[0]["chat_id"] == "nightly-2026-07-27"


# ---------------------------------------------------------------------------
# Memory extraction
# ---------------------------------------------------------------------------


def _extractor(*replies: str, **kwargs) -> MemoryExtractor:
    return MemoryExtractor(
        FakeListChatModel(responses=list(replies)), cues=[], **kwargs
    )


def _extraction(text: str, scope: str = "temporary") -> str:
    return json.dumps(
        {"memories": [{"text": text, "scope": scope, "kind": "note", "reason": "r"}]}
    )


class TestExtractionWiring:
    def test_an_extractor_implies_a_thread_memory(self):
        extractor = _extractor(_extraction("Use zz_."))
        pipeline = _pipeline(memory_extractor=extractor)
        assert pipeline.thread_memory is not None
        assert extractor.thread_memory is pipeline.thread_memory

    def test_an_accepted_turn_writes_its_temporary_memory(self):
        notes = ThreadMemory()
        pipeline = _pipeline(
            thread_memory=notes,
            memory_extractor=_extractor(_extraction("Use zz_.")),
        )
        pipeline.run_text(SAS, thread_id="t1")
        assert notes.texts("t1") == ["Use zz_."]

    def test_a_permanent_memory_needs_approval_before_it_becomes_policy(self):
        hub = MemoryHub()
        policy = TaskPolicy("sas", store=hub.kv)
        extractor = _extractor(_extraction("Always be explicit.", "permanent"))
        pipeline = _pipeline(
            memory=hub, task_policy=policy, memory_extractor=extractor
        )
        pipeline.run_text(SAS, thread_id="t1")
        assert policy.texts() == []
        (candidate,) = extractor.pending("t1")
        extractor.approve(candidate["id"])
        assert policy.texts() == ["Always be explicit."]

    def test_a_broken_extractor_cannot_fail_a_run(self):
        class _Exploding:
            def invoke(self, *_args, **_kwargs):
                raise RuntimeError("classifier down")

        pipeline = _pipeline(memory_extractor=MemoryExtractor(_Exploding(), cues=[]))
        assert pipeline.run_text(SAS, thread_id="t1")  # the run still completed
        assert len(pipeline.get_thread_messages("t1")) == 2

    def test_extraction_runs_once_per_item_on_the_kept_answer(self):
        # Two scripted replies, but a single-item run: exactly one extraction
        # call, so the second reply is never consumed.
        extractor = _extractor(_extraction("First."), _extraction("Second."))
        pipeline = _pipeline(memory_extractor=extractor)
        pipeline.run_text(SAS, thread_id="t1")
        assert extractor.thread_memory.texts("t1") == ["First."]
