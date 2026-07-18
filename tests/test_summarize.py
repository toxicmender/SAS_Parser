"""
Tests for memory.summarize.RollingSummarizer — fully offline: the
summarizing "model" is a recording callable (or a tiny .invoke stub), and
persistence runs on the in-memory KV backend.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from memory.store import MemoryHub
from memory.summarize import RollingSummarizer


class RecordingModel:
    """Callable fake: returns 'summary vN' and records every prompt."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return f"summary v{len(self.prompts)}"


def _mk_history(*pair_texts: str) -> list:
    history: list = []
    for i, text in enumerate(pair_texts):
        history.append(HumanMessage(text))
        history.append(AIMessage(f"reply {i}"))
    return history


def test_no_summary_below_trigger():
    model = RecordingModel()
    s = RollingSummarizer(model, trigger_tokens=10_000, keep_last_turns=0)
    assert s.refresh("t1", _mk_history("a", "b", "c")) is None
    assert model.prompts == []


def test_folds_eligible_turns_and_spares_the_tail():
    model = RecordingModel()
    s = RollingSummarizer(model, trigger_tokens=1, keep_last_turns=1)
    history = _mk_history("first turn text", "second turn text", "newest turn text")

    msg = s.refresh("t1", history)

    assert isinstance(msg, SystemMessage)
    assert "summary v1" in msg.content
    assert len(model.prompts) == 1
    # The two old turns were folded; the keep_last tail was not.
    assert "first turn text" in model.prompts[0]
    assert "second turn text" in model.prompts[0]
    assert "newest turn text" not in model.prompts[0]


def test_incremental_folding_never_resummarizes_covered_turns():
    model = RecordingModel()
    s = RollingSummarizer(model, trigger_tokens=1, keep_last_turns=0)
    s.refresh("t1", _mk_history("alpha turn", "beta turn"))
    s.refresh("t1", _mk_history("alpha turn", "beta turn", "gamma turn"))

    assert len(model.prompts) == 2
    # Second fold sees only the new turn, plus the previous summary text.
    assert "gamma turn" in model.prompts[1]
    assert "alpha turn" not in model.prompts[1]
    assert "summary v1" in model.prompts[1]


def test_state_persists_through_external_store():
    kv = MemoryHub().kv
    model1 = RecordingModel()
    s1 = RollingSummarizer(model1, store=kv, trigger_tokens=1, keep_last_turns=0)
    history = _mk_history("alpha turn", "beta turn")
    s1.refresh("t1", history)

    # Fresh summarizer over the same store: summary is already covered, so
    # the model is never called, yet the message is still produced.
    model2 = RecordingModel()
    s2 = RollingSummarizer(model2, store=kv, trigger_tokens=1, keep_last_turns=0)
    msg = s2.refresh("t1", history)
    assert isinstance(msg, SystemMessage)
    assert "summary v1" in msg.content
    assert model2.prompts == []


def test_shrunken_thread_resets_stale_summary():
    kv = MemoryHub().kv
    model = RecordingModel()
    # Trigger sized so the full 3-turn thread folds but a single leftover
    # turn stays under threshold after the reset.
    s = RollingSummarizer(model, store=kv, trigger_tokens=5, keep_last_turns=0)
    s.refresh("t1", _mk_history("a", "b", "c"))

    # Thread now has fewer turns than the summary covered (cleared/forked).
    assert s.refresh("t1", _mk_history("a")) is None
    # And the reset was persisted — no stale coverage left in the store.
    assert kv.get("summary::t1")["covered_turns"] == 0


def test_reset_discards_summary():
    kv = MemoryHub().kv
    s = RollingSummarizer(RecordingModel(), store=kv, trigger_tokens=1, keep_last_turns=0)
    history = _mk_history("a", "b")
    s.refresh("t1", history)
    s.reset("t1")
    assert kv.get("summary::t1") is None


def test_invoke_style_model_supported():
    class InvokeModel:
        def invoke(self, prompt: str):
            class _Msg:
                content = "invoked summary"

            return _Msg()

    s = RollingSummarizer(InvokeModel(), trigger_tokens=1, keep_last_turns=0)
    msg = s.refresh("t1", _mk_history("a", "b"))
    assert msg is not None
    assert "invoked summary" in msg.content


def test_keep_last_turns_blocks_folding_entirely_on_short_threads():
    model = RecordingModel()
    s = RollingSummarizer(model, trigger_tokens=1, keep_last_turns=2)
    assert s.refresh("t1", _mk_history("a", "b")) is None
    assert model.prompts == []
