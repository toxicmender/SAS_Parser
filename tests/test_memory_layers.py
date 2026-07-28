"""
Tests for the long-term / short-term memory layers:
memory.policy (TaskPolicy), memory.thread_mem (ThreadMemory),
memory.context (MemoryContext) and memory.extractor (MemoryExtractor).

Everything runs offline. Persistence is exercised against the real
KVMemoryStore over the in-memory backend (no Spark, no JVM), and the
extractor's classifier is a FakeListChatModel scripted with the JSON its
prompt asks for — which also exercises the prose-JSON fallback, since a fake
chat model has no with_structured_output.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from memory.context import MemoryContext
from memory.extractor import DEFAULT_CUES, MemoryExtractor, cue_hit
from memory.policy import PolicyInstruction, TaskPolicy
from memory.store import MemoryHub
from memory.thread_mem import ThreadMemory


def _kv():
    return MemoryHub().kv


def _extraction(*memories: dict) -> str:
    return json.dumps({"memories": list(memories)})


def _memory(text: str, scope: str = "temporary", kind: str = "note") -> dict:
    return {"text": text, "scope": scope, "kind": kind, "reason": "because"}


# ---------------------------------------------------------------------------
# TaskPolicy — long-term memory
# ---------------------------------------------------------------------------


class TestTaskPolicy:
    def test_add_and_render_in_order(self):
        policy = TaskPolicy("support", store=_kv())
        policy.add("Prefer concise responses.")
        policy.add("Escalate refund requests above $500.", overridable=False)
        rendered = policy.render()
        assert rendered.index("Prefer concise") < rendered.index("Escalate refund")
        assert "1. Prefer concise responses." in rendered
        assert "(fixed:" in rendered.split("Escalate refund")[1]
        # Only the non-overridable one carries the marker.
        assert rendered.count("(fixed:") == 1

    def test_empty_policy_renders_nothing(self):
        policy = TaskPolicy("support", store=_kv())
        assert policy.render() == ""
        assert policy.fingerprint == ""

    def test_persists_across_instances(self):
        kv = _kv()
        TaskPolicy("support", store=kv).add("Always verify account ownership.")
        reloaded = TaskPolicy("support", store=kv)
        assert reloaded.texts() == ["Always verify account ownership."]
        assert reloaded.version == 1

    def test_defaults_seed_only_an_unknown_task(self):
        kv = _kv()
        TaskPolicy("support", store=kv, default_instructions=["Be brief."])
        # A later run with different defaults must not overwrite the stored
        # policy an operator may have edited since.
        second = TaskPolicy("support", store=kv, default_instructions=["Be verbose."])
        assert second.texts() == ["Be brief."]

    def test_tasks_are_isolated(self):
        kv = _kv()
        TaskPolicy("support", store=kv).add("Support rule.")
        assert TaskPolicy("billing", store=kv).texts() == []

    def test_remove_and_replace_bump_the_version(self):
        policy = TaskPolicy("support", store=_kv())
        first = policy.add("One.")
        policy.add("Two.")
        assert policy.version == 2
        assert policy.remove(first.id) is True
        assert policy.texts() == ["Two."]
        assert policy.version == 3
        assert policy.remove("nope") is False
        policy.replace(["Three.", PolicyInstruction(text="Four.", overridable=False)])
        assert policy.texts() == ["Three.", "Four."]
        assert policy.texts(overridable=False) == ["Four."]
        assert policy.version == 4

    def test_fingerprint_tracks_prompted_text_not_version(self):
        policy = TaskPolicy("support", store=_kv())
        policy.add("Be brief.")
        first = policy.fingerprint
        instruction = policy.instructions[0]
        policy.replace([PolicyInstruction(text="Be brief.", id=instruction.id)])
        # Same prompted text, later version -> same fingerprint.
        assert policy.fingerprint == first and policy.version == 2
        # Same text, now fixed -> different prompt, different fingerprint.
        policy.replace([PolicyInstruction(text="Be brief.", overridable=False)])
        assert policy.fingerprint != first

    def test_reload_picks_up_an_out_of_band_edit(self):
        kv = _kv()
        one = TaskPolicy("support", store=kv)
        two = TaskPolicy("support", store=kv)
        two.add("Written by the other instance.")
        assert one.texts() == []
        one.reload()
        assert one.texts() == ["Written by the other instance."]

    def test_process_local_without_a_store(self):
        policy = TaskPolicy("support")
        policy.add("Be brief.")
        assert policy.texts() == ["Be brief."]
        assert TaskPolicy("support").texts() == []

    def test_rejects_empty_ids_and_text(self):
        with pytest.raises(ValueError):
            TaskPolicy("")
        with pytest.raises(ValueError):
            TaskPolicy("support").add("   ")

    def test_clear_drops_the_record(self):
        kv = _kv()
        policy = TaskPolicy("support", store=kv)
        policy.add("Be brief.")
        policy.clear()
        assert policy.texts() == []
        assert TaskPolicy("support", store=kv).texts() == []


# ---------------------------------------------------------------------------
# ThreadMemory — short-term memory
# ---------------------------------------------------------------------------


class TestThreadMemory:
    def test_add_and_render(self):
        notes = ThreadMemory(store=_kv())
        notes.add("t1", "Customer prefers email.", kind="preference")
        notes.add("t1", "Discount approved.", kind="exception", source="supervisor")
        rendered = notes.render("t1")
        assert "[preference] Customer prefers email." in rendered
        assert "(source: supervisor)" in rendered
        assert notes.texts("t1") == [
            "Customer prefers email.",
            "Discount approved.",
        ]

    def test_threads_are_isolated(self):
        notes = ThreadMemory(store=_kv())
        notes.add("t1", "Only for t1.")
        assert notes.texts("t2") == []
        assert notes.render("t2") == ""

    def test_expired_notes_are_filtered_and_purged(self):
        kv = _kv()
        notes = ThreadMemory(store=kv)
        notes.add("t1", "Just this call.", ttl_s=100)
        notes.add("t1", "Standing for the thread.")
        # Read past the TTL: the expired note is filtered out...
        assert [n.text for n in notes.notes("t1", now=_far_future())] == [
            "Standing for the thread."
        ]
        # ...and the next write purges it from the record for good.
        notes.remove("t1", notes.notes("t1")[0].id)
        assert notes.texts("t1") == ["Standing for the thread."]

    def test_default_ttl_applies_to_bare_adds(self):
        notes = ThreadMemory(store=_kv(), default_ttl_s=60)
        note = notes.add("t1", "Temporary by default.")
        assert note.expires_at is not None
        assert notes.add("t1", "Explicitly permanent.", ttl_s=None).expires_at

    def test_fork_copies_live_notes_onto_the_branch(self):
        notes = ThreadMemory(store=_kv())
        notes.add("src", "Exception approved by supervisor.", kind="exception")
        assert notes.fork("src", "dst") == 1
        assert notes.texts("dst") == ["Exception approved by supervisor."]
        # The source is untouched, and the copies are independent.
        notes.add("dst", "Branch-only note.")
        assert notes.texts("src") == ["Exception approved by supervisor."]

    def test_remove_and_clear(self):
        notes = ThreadMemory(store=_kv())
        note = notes.add("t1", "Gone soon.")
        assert notes.remove("t1", "unknown") is False
        assert notes.remove("t1", note.id) is True
        notes.add("t1", "Also gone.")
        notes.clear("t1")
        assert notes.texts("t1") == []

    def test_rejects_empty_text(self):
        with pytest.raises(ValueError):
            ThreadMemory().add("t1", "  ")

    def test_survives_an_unreadable_record(self):
        kv = _kv()
        kv.set("tmem::t1", [{"not": "a note"}, {"text": "Fine."}])
        assert ThreadMemory(store=kv).texts("t1") == ["Fine."]


def _far_future() -> float:
    import time

    return time.time() + 10_000


# ---------------------------------------------------------------------------
# MemoryContext — assembly and the precedence rule
# ---------------------------------------------------------------------------


class TestMemoryContext:
    def _both(self):
        kv = _kv()
        policy = TaskPolicy("support", store=kv)
        policy.add("Prefer concise responses.")
        policy.add("Escalate refunds above $500.", overridable=False)
        notes = ThreadMemory(store=kv)
        notes.add("t1", "Do not offer the standard discount.", kind="exception")
        return MemoryContext(policy=policy, thread_memory=notes)

    def test_policy_goes_to_the_system_half_notes_to_the_ephemeral_half(self):
        assembled = self._both().assemble("t1")
        assert "Prefer concise responses." in assembled.system_suffix
        assert "standard discount" not in assembled.system_suffix
        assert len(assembled.ephemeral) == 1
        assert "standard discount" in assembled.ephemeral[0].content
        assert assembled.note_count == 1
        assert assembled.policy_fingerprint

    def test_precedence_rule_is_stated_with_the_notes(self):
        content = self._both().assemble("t1").ephemeral[0].content
        assert "conflicts with a standing task instruction" in content
        assert "(fixed)" in content

    def test_no_precedence_rule_without_a_policy(self):
        notes = ThreadMemory()
        notes.add("t1", "A note.")
        content = MemoryContext(thread_memory=notes).assemble("t1").ephemeral[0].content
        assert "A note." in content
        assert "standing task instruction" not in content

    def test_a_thread_without_notes_gets_no_ephemeral_message(self):
        assembled = self._both().assemble("other-thread")
        assert assembled.ephemeral == []
        assert assembled.note_count == 0
        # ...but the policy still applies to it.
        assert "Prefer concise responses." in assembled.system_suffix

    def test_empty_context_is_inert(self):
        assembled = MemoryContext().assemble("t1")
        assert assembled.system_suffix == ""
        assert assembled.ephemeral == []
        assert assembled.policy_fingerprint == ""


# ---------------------------------------------------------------------------
# MemoryExtractor — the write path
# ---------------------------------------------------------------------------


class TestMemoryExtractor:
    def _extractor(self, *replies: str, **kwargs):
        kv = _kv()
        policy = kwargs.pop("policy", TaskPolicy("support", store=kv))
        notes = kwargs.pop("thread_memory", ThreadMemory(store=kv))
        return MemoryExtractor(
            FakeListChatModel(responses=list(replies)),
            policy=policy,
            thread_memory=notes,
            store=kv,
            **kwargs,
        )

    def test_cue_gate_skips_an_ordinary_turn_without_calling_the_model(self):
        # No scripted reply at all: if the model were called, it would raise.
        extractor = self._extractor()
        result = extractor.observe("t1", "Translate this DATA step.", "done")
        assert result.gated is True
        assert result.applied == [] and result.pending == []

    def test_temporary_memories_are_applied_immediately(self):
        extractor = self._extractor(
            _extraction(_memory("Do not offer the standard discount.", kind="exception"))
        )
        result = extractor.observe(
            "t1", "For this conversation, do not offer the discount.", "understood"
        )
        assert result.applied == ["Do not offer the standard discount."]
        assert extractor.thread_memory.texts("t1") == [
            "Do not offer the standard discount."
        ]

    def test_permanent_memories_wait_for_approval(self):
        extractor = self._extractor(
            _extraction(_memory("Always verify account ownership.", scope="permanent"))
        )
        result = extractor.observe("t1", "From now on, always verify ownership.", "ok")
        assert result.applied == [] and result.promoted == []
        assert len(result.pending) == 1
        # Nothing reached the durable policy yet.
        assert extractor.policy.texts() == []

        candidate_id = result.pending[0]["id"]
        assert extractor.pending("t1")[0]["id"] == candidate_id
        instruction = extractor.approve(candidate_id, overridable=False)
        assert instruction is not None
        assert extractor.policy.texts() == ["Always verify account ownership."]
        assert extractor.policy.texts(overridable=False) == [
            "Always verify account ownership."
        ]
        # Approving consumes the candidate.
        assert extractor.pending() == []

    def test_reject_drops_a_candidate(self):
        extractor = self._extractor(
            _extraction(_memory("Always do X.", scope="permanent"))
        )
        pending = extractor.observe("t1", "From now on, always do X.", "ok").pending
        assert extractor.reject(pending[0]["id"]) is True
        assert extractor.pending() == []
        assert extractor.reject(pending[0]["id"]) is False

    def test_auto_apply_permanent_writes_straight_to_the_policy(self):
        extractor = self._extractor(
            _extraction(_memory("Always do X.", scope="permanent")),
            auto_apply_permanent=True,
        )
        result = extractor.observe("t1", "From now on, always do X.", "ok")
        assert result.promoted == ["Always do X."]
        assert extractor.policy.texts() == ["Always do X."]

    def test_duplicate_temporary_memories_are_not_re_added(self):
        extractor = self._extractor(
            _extraction(_memory("Use email.")),
            _extraction(_memory("use email.")),
        )
        extractor.observe("t1", "Always use email.", "ok")
        second = extractor.observe("t1", "Always use email.", "ok")
        assert second.applied == []
        assert extractor.thread_memory.texts("t1") == ["Use email."]

    def test_an_unusable_reply_is_a_warning_not_an_exception(self, caplog):
        extractor = self._extractor("I could not decide.")
        result = extractor.observe("t1", "From now on, always do X.", "ok")
        assert result.applied == [] and result.pending == []
        assert "unusable classifier reply" in caplog.text

    def test_json_wrapped_in_prose_is_recovered(self):
        extractor = self._extractor(
            "Here is what I found:\n```json\n"
            + _extraction(_memory("Use email."))
            + "\n```\nHope that helps."
        )
        result = extractor.observe("t1", "Always use email.", "ok")
        assert result.applied == ["Use email."]

    def test_max_memories_caps_a_flood(self):
        extractor = self._extractor(
            _extraction(*[_memory(f"Note {i}.") for i in range(10)]),
            max_memories=2,
        )
        result = extractor.observe("t1", "Always remember these.", "ok")
        assert len(result.applied) == 2

    def test_cue_gate_can_be_disabled(self):
        extractor = self._extractor(_extraction(_memory("Use email.")), cues=[])
        assert extractor.observe("t1", "no cues here", "ok").applied == ["Use email."]

    def test_cue_hit_helper_matches_instruction_language(self):
        assert cue_hit("From now on, use snake_case")
        assert cue_hit("Never drop the index")
        assert not cue_hit("Translate this DATA step into PySpark")
        assert cue_hit("anything", cues=[])
        assert len(DEFAULT_CUES) > 10
