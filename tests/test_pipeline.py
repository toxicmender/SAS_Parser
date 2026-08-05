"""
Smoke tests for pipeline.py's integration with memory.store.

These tests deliberately avoid any live LLM call: SasLLMPipeline is
constructed with a FakeListChatModel so we can verify the actual thing
this integration is responsible for — that batches/chunks get formatted
correctly and that conversation state round-trips through
MemoryHub / KVChatMessageHistory — without needing API credentials
or network access.

The memory layer runs on its in-memory backend, so no Spark session (or
JVM, or pyspark install) is needed anywhere in this module.

Requires: langchain-core (FakeListChatModel).
"""

from __future__ import annotations

import pathlib
import pytest
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from memory.store import MemoryHub

from chunker.models import (
    SasBatch,
    SasChunk,
    SasChunkKind,
    SasChunkMetadata,
)
from llm_client import LLMClientConfig
from pipeline import SasLLMPipeline
from pipeline.setup import MemorySetup
from pipeline.prompting import _format_batch_message
from target_language import resolve_target_language

# What an output_language-less pipeline resolves to, so the prompt assertions
# below stay true if the configured default changes.
DEFAULT_OUTPUT_LANGUAGE_DISPLAY = resolve_target_language(None).display_name


def _mk_chunk(chunk_id: str, source_id: str, text: str, **meta_kwargs) -> SasChunk:
    return SasChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        text=text,
        kind=SasChunkKind.DATA_STEP,
        title=f"Step {chunk_id}",
        start_line=1,
        end_line=3,
        start_char=0,
        end_char=len(text),
        metadata=SasChunkMetadata(**meta_kwargs),
    )


def _mk_batch(batch_id: str, chunks: list[SasChunk], **kwargs) -> SasBatch:
    return SasBatch(batch_id=batch_id, chunks=chunks, **kwargs)


def _wrap(chunk: SasChunk) -> SasBatch:
    """A one-member batch carrying the chunk's own id — the shape _process
    takes now that items are SasBatch only (coalesce_into_batches wraps every
    singleton before anything is prompted)."""
    return SasBatch(
        batch_id=chunk.chunk_id,
        chunks=[chunk],
        source_files=[chunk.source_id or "unknown"],
    )


# ---------------------------------------------------------------------------
# Pure formatting functions — no Spark/LLM required
# ---------------------------------------------------------------------------


def test_format_batch_message_covers_the_single_member_case():
    # There is no per-chunk formatter any more: a lone chunk is prompted as a
    # one-member batch, and its key fields must all surface there.
    chunk = _mk_chunk(
        "f1-chunk-0001",
        "etl.sas",
        "data work.out; set work.in; run;",
        input_datasets=["work.in"],
        output_datasets=["work.out"],
        symput_scope_hazard=True,
        symput_hazard_vars=["cutoff"],
    )
    batch = _mk_batch(
        "merged-001",
        [chunk],
        source_files=["etl.sas"],
        input_datasets=["work.in"],
        output_datasets=["work.out"],
    )
    msg = _format_batch_message(batch, index=1, total=1, diagnostics=[])

    assert "work.in" in msg
    assert "work.out" in msg
    assert "yes" in msg  # symput hazard aggregate line
    assert "data work.out; set work.in; run;" in msg


def test_format_batch_message_includes_all_members_and_cross_file_flag():
    c1 = _mk_chunk(
        "f1-chunk-0001",
        "etl.sas",
        "data work.base; run;",
        output_datasets=["work.base"],
    )
    c2 = _mk_chunk(
        "f2-chunk-0001",
        "report.sas",
        "proc print data=work.base; run;",
        input_datasets=["work.base"],
    )
    batch = _mk_batch(
        "batch-001",
        [c1, c2],
        source_files=["etl.sas", "report.sas"],
        input_datasets=[],
        output_datasets=["work.base"],
        reason="dataset_flow(work.base): f1-chunk-0001 -> f2-chunk-0001",
    )

    msg = _format_batch_message(batch, index=1, total=1, diagnostics=[])

    assert "batch-001" in msg
    assert "yes" in msg  # is_cross_file
    assert "f1-chunk-0001" in msg and "f2-chunk-0001" in msg
    assert "proc print data=work.base; run;" in msg
    assert "dataset_flow(work.base)" in msg


# ---------------------------------------------------------------------------
# Persistence wiring — KVChatMessageHistory round-trip via MemoryHub
# ---------------------------------------------------------------------------


def test_memory_thread_round_trips_messages():
    mem = MemoryHub()  # in-memory backend, no Spark
    thread = mem.get_thread("run::etl.sas")

    thread.add_user_message("batch-001 content")
    thread.add_ai_message("translated PySpark for batch-001")

    same_thread = mem.get_thread("run::etl.sas")  # same id -> same history
    assert len(same_thread.messages) == 2
    assert isinstance(same_thread.messages[0], HumanMessage)
    assert isinstance(same_thread.messages[1], AIMessage)
    assert same_thread.messages[1].content == "translated PySpark for batch-001"


def test_different_thread_ids_are_isolated():
    mem = MemoryHub()
    mem.get_thread("run::a.sas").add_user_message("hello a")
    mem.get_thread("run::b.sas").add_user_message("hello b")

    assert [m.content for m in mem.get_thread("run::a.sas").messages] == ["hello a"]
    assert [m.content for m in mem.get_thread("run::b.sas").messages] == ["hello b"]


# ---------------------------------------------------------------------------
# End-to-end pipeline wiring with a fake LLM — no network/API key needed
# ---------------------------------------------------------------------------


def test_pipeline_accumulates_history_across_batches():
    fake_llm = FakeListChatModel(
        responses=["translation for item 1", "translation for item 2"]
    )
    mem = MemoryHub()

    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused-because-llm-injected"),
        memory_setup=MemorySetup(memory=mem),
        llm=fake_llm,
        window_k=None,  # no trimming, so we can assert full history length
    )

    c1 = _mk_chunk(
        "f1-chunk-0001", "etl.sas", "data work.a; run;", output_datasets=["work.a"]
    )
    c2 = _mk_chunk(
        "f1-chunk-0002",
        "etl.sas",
        "proc print data=work.a; run;",
        input_datasets=["work.a"],
    )
    batch = _mk_batch(
        "batch-001",
        [c1, c2],
        source_files=["etl.sas"],
        output_datasets=["work.a"],
    )

    singleton = _mk_batch("merged-001", [c1], source_files=["etl.sas"])
    results = pipeline._process(
        items=[batch, singleton],  # a real batch, then a one-member merge
        diagnostics=[],
        thread_id="run::etl.sas",
    )

    assert len(results) == 2
    assert results[0]["item_id"] == "batch-001"
    # No "is_batch"/"kind" in the output shape: every item is a batch, so a
    # constant field told a reader nothing and invited branching on it.
    assert "is_batch" not in results[0] and "kind" not in results[0]
    assert results[0]["response"] == "translation for item 1"
    assert results[1]["item_id"] == "merged-001"
    assert results[1]["response"] == "translation for item 2"

    # Both turns landed in the SAME thread, in order: human/ai x2
    history = pipeline.get_thread_messages("run::etl.sas")
    assert len(history) == 4
    assert isinstance(history[0], HumanMessage)
    assert isinstance(history[1], AIMessage)
    assert history[1].content == "translation for item 1"
    assert isinstance(history[2], HumanMessage)
    assert isinstance(history[3], AIMessage)
    assert history[3].content == "translation for item 2"


def test_run_text_invokes_llm_only_per_batch():
    # Every unit sent to the LLM is a SasBatch: the run's singletons are
    # coalesced so no standalone SasChunk is ever prompted. Packing is on by
    # default, so the three independent steps share one packed call.
    fake_llm = FakeListChatModel(responses=[f"r{i}" for i in range(10)])
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused"),
        memory_setup=MemorySetup(memory=MemoryHub()),
        llm=fake_llm,
    )

    src = (
        "data work.a; x=1; run;\n"
        "data work.b; y=2; run;\n"
        "data work.c; z=3; run;\n"
    )
    outputs = pipeline.run_text(src, source_id="etl.sas")

    assert outputs
    # Three independent steps pack into a single batch under the default
    # token budget (well over three tiny DATA steps).
    assert len(outputs) == 1
    assert outputs[0]["item_id"] == "packed-001"


def test_max_merged_chunks_caps_calls_per_batch():
    fake_llm = FakeListChatModel(responses=[f"r{i}" for i in range(10)])
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused"), memory_setup=MemorySetup(memory=MemoryHub()), llm=fake_llm, max_merged_chunks=1
    )

    src = (
        "data work.a; x=1; run;\n"
        "data work.b; y=2; run;\n"
        "data work.c; z=3; run;\n"
    )
    outputs = pipeline.run_text(src, source_id="etl.sas")

    # max_merged_chunks=1 wraps each singleton as its own one-member batch.
    assert [o["item_id"] for o in outputs] == ["merged-001", "merged-002", "merged-003"]


def test_max_merged_tokens_packs_adjacent_items_into_one_call():
    # Token-budgeted packing (Phase 4): with a generous budget the whole
    # run — independent singletons AND the dependency batch — shares one
    # LLM call, as a packed-NNN batch.
    fake_llm = FakeListChatModel(responses=[f"r{i}" for i in range(10)])
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused"),
        memory_setup=MemorySetup(memory=MemoryHub()),
        llm=fake_llm,
        max_merged_tokens=100_000,
    )

    src = (
        "data work.x; p=1; run;\n"
        "data work.dep; q=1; run;\n"
        "proc print data=work.dep; run;\n"
        "data work.y; r=1; run;\n"
    )
    outputs = pipeline.run_text(src, source_id="etl.sas")

    assert [o["item_id"] for o in outputs] == ["packed-001"]
    assert len(outputs[0]["chunk_ids"]) == 4
    # Every output maps its member chunks to their source files, so the
    # notebook renderer can split multi-source items per file (Phase 5).
    assert outputs[0]["chunk_sources"] == {
        cid: "etl.sas" for cid in outputs[0]["chunk_ids"]
    }


def test_max_merged_tokens_zero_disables_packing():
    # max_merged_tokens=0 turns packing off: the dependency batch is its
    # own call and flushes the singleton runs around it (3 calls, not 1).
    fake_llm = FakeListChatModel(responses=[f"r{i}" for i in range(10)])
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused"),
        memory_setup=MemorySetup(memory=MemoryHub()),
        llm=fake_llm,
        max_merged_tokens=0,
    )

    src = (
        "data work.x; p=1; run;\n"
        "data work.dep; q=1; run;\n"
        "proc print data=work.dep; run;\n"
        "data work.y; r=1; run;\n"
    )
    outputs = pipeline.run_text(src, source_id="etl.sas")

    assert len(outputs) == 3
    assert [o["item_id"] for o in outputs][0] == "merged-001"


def test_max_merged_tokens_validates():
    with pytest.raises(ValueError, match="max_merged_tokens"):
        SasLLMPipeline(
            llm_config=LLMClientConfig(model="unused"),
            memory_setup=MemorySetup(memory=MemoryHub()),
            llm=FakeListChatModel(responses=["ok"]),
            max_merged_tokens=-1,
        )


def test_packing_budget_defaults_and_derivation():
    from pipeline.engine import _DEFAULT_MAX_MERGED_TOKENS

    def _mk(max_input_tokens=None, **kwargs):
        return SasLLMPipeline(
            llm_config=LLMClientConfig(
                model="unused", max_input_tokens=max_input_tokens
            ),
            memory_setup=MemorySetup(memory=MemoryHub()),
            llm=FakeListChatModel(responses=["ok"]),
            **kwargs,
        )

    # No input budget: the conservative hard default applies.
    assert _mk()._max_merged_tokens == _DEFAULT_MAX_MERGED_TOKENS
    # With an input budget, the packing budget derives from its headroom and
    # scales with it.
    derived = _mk(max_input_tokens=100_000)._max_merged_tokens
    assert derived is not None and derived > _DEFAULT_MAX_MERGED_TOKENS
    assert derived < 100_000
    # A budget too small to pack under disables packing outright.
    assert _mk(max_input_tokens=2_000)._max_merged_tokens is None
    # An explicit budget wins over derivation.
    assert _mk(max_merged_tokens=1_234)._max_merged_tokens == 1_234


def test_pipeline_window_trimming_limits_injected_history():
    fake_llm = FakeListChatModel(responses=[f"resp {i}" for i in range(6)])
    mem = MemoryHub()
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused-because-llm-injected"),
        memory_setup=MemorySetup(memory=mem),
        llm=fake_llm,
        window_k=1,  # keep only last 1 human/ai pair in the prompt
    )

    chunks: list[SasBatch | SasChunk] = [
        _mk_chunk(f"f1-chunk-000{i}", "etl.sas", f"data work.t{i}; run;")
        for i in range(3)
    ]
    pipeline._process(items=[_wrap(c) for c in chunks], diagnostics=[], thread_id="run::etl.sas")

    # Full history is still persisted (trimming only affects the prompt)...
    full_history = pipeline.get_thread_messages("run::etl.sas")
    assert len(full_history) == 6  # 3 human + 3 ai


def test_snapshot_delegates_to_memory():
    fake_llm = FakeListChatModel(responses=["ok"])
    mem = MemoryHub()
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused"),
        memory_setup=MemorySetup(memory=mem),
        llm=fake_llm,
    )

    c1 = _mk_chunk("f1-chunk-0001", "etl.sas", "data work.a; run;")
    pipeline._process(items=[_wrap(c1)], diagnostics=[], thread_id="run::etl.sas")

    snap = pipeline.snapshot()
    assert snap == mem.snapshot()
    assert any("run::etl.sas" in k for k in snap)


# ---------------------------------------------------------------------------
# Run facts — the per-item KV write channel
# ---------------------------------------------------------------------------


def test_run_facts_recorded_per_item():
    fake_llm = FakeListChatModel(responses=["resp 1", "resp 2"])
    mem = MemoryHub()
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused"),
        memory_setup=MemorySetup(memory=mem),
        llm=fake_llm,
    )

    chunks: list[SasBatch | SasChunk] = [
        _mk_chunk("f1-chunk-0001", "etl.sas", "data work.a; run;"),
        _mk_chunk("f1-chunk-0002", "etl.sas", "proc print data=work.a; run;"),
    ]
    pipeline._process(items=[_wrap(c) for c in chunks], diagnostics=[], thread_id="run::etl.sas")

    facts = pipeline.get_run_facts("run::etl.sas")
    assert [f["item_id"] for f in facts] == ["f1-chunk-0001", "f1-chunk-0002"]
    assert all(f["status"] == "ok" for f in facts)
    assert [f["index"] for f in facts] == [1, 2]
    assert all(f["response_chars"] == len("resp 1") for f in facts)
    # Facts live in the KV layer, isolated from the msg:: history.
    assert len(pipeline.get_thread_messages("run::etl.sas")) == 4


def test_run_facts_isolated_per_thread():
    fake_llm = FakeListChatModel(responses=["a", "b"])
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused"),
        memory_setup=MemorySetup(memory=MemoryHub()),
        llm=fake_llm,
    )
    pipeline._process(
        items=[_wrap(_mk_chunk("c1", "a.sas", "data work.a; run;"))],
        diagnostics=[],
        thread_id="run::a.sas",
    )
    pipeline._process(
        items=[_wrap(_mk_chunk("c2", "b.sas", "data work.b; run;"))],
        diagnostics=[],
        thread_id="run::b.sas",
    )
    assert [f["item_id"] for f in pipeline.get_run_facts("run::a.sas")] == ["c1"]
    assert [f["item_id"] for f in pipeline.get_run_facts("run::b.sas")] == ["c2"]


# ---------------------------------------------------------------------------
# Resume + fork_run — crash recovery and KV-native time travel
# ---------------------------------------------------------------------------


def test_resume_skips_completed_items_and_recovers_responses():
    mem = MemoryHub()
    fake_llm = FakeListChatModel(responses=["resp 1", "resp 2"])
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused"),
        memory_setup=MemorySetup(memory=mem),
        llm=fake_llm,
    )

    c1 = _mk_chunk("f1-chunk-0001", "etl.sas", "data work.a; run;")
    c2 = _mk_chunk("f1-chunk-0002", "etl.sas", "proc print data=work.a; run;")

    # First run "crashed" after item 1: only c1 was processed.
    pipeline._process(items=[_wrap(c1)], diagnostics=[], thread_id="run::etl.sas")

    outputs = pipeline._process(
        items=[_wrap(c1), _wrap(c2)], diagnostics=[], thread_id="run::etl.sas", resume=True
    )

    assert outputs[0]["skipped"] is True
    assert outputs[0]["response"] == "resp 1"  # recovered from the thread
    assert outputs[1]["skipped"] is False
    assert outputs[1]["response"] == "resp 2"
    # c1 was not replayed: exactly one turn pair per item.
    assert len(pipeline.get_thread_messages("run::etl.sas")) == 4


def test_resume_reprocesses_items_with_error_facts():
    mem = MemoryHub()
    fake_llm = FakeListChatModel(responses=["resp 1", "resp 2"])
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused"),
        memory_setup=MemorySetup(memory=mem),
        llm=fake_llm,
    )

    c1 = _mk_chunk("f1-chunk-0001", "etl.sas", "data work.a; run;")
    pipeline._process(items=[_wrap(c1)], diagnostics=[], thread_id="run::etl.sas")
    c2 = _mk_chunk("f1-chunk-0002", "etl.sas", "proc print data=work.a; run;")
    # Simulate a crashed second item: an error fact, no persisted turn.
    mem.kv.set(
        "run::run::etl.sas::item::f1-chunk-0002",
        {"status": "error", "index": 2, "error": "boom"},
    )

    outputs = pipeline._process(
        items=[_wrap(c1), _wrap(c2)], diagnostics=[], thread_id="run::etl.sas", resume=True
    )

    assert outputs[1]["skipped"] is False  # error fact does not skip
    assert outputs[1]["response"] == "resp 2"
    facts = pipeline.get_run_facts("run::etl.sas")
    assert [f["status"] for f in facts] == ["ok", "ok"]  # overwritten


def test_fork_run_then_resume_continues_from_the_fork():
    mem = MemoryHub()
    fake_llm = FakeListChatModel(responses=["resp 1", "resp 2", "resp 2 redone"])
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused"),
        memory_setup=MemorySetup(memory=mem),
        llm=fake_llm,
    )

    c1 = _mk_chunk("f1-chunk-0001", "etl.sas", "data work.a; run;")
    c2 = _mk_chunk("f1-chunk-0002", "etl.sas", "proc print data=work.a; run;")
    pipeline._process(items=[_wrap(c1), _wrap(c2)], diagnostics=[], thread_id="run::v1")

    # Rewind to after item 1 and redo item 2 on a fresh branch.
    copied = pipeline.fork_run("run::v1", "run::v2", upto_items=1)
    assert copied == 2  # one (human, AI) pair

    outputs = pipeline._process(
        items=[_wrap(c1), _wrap(c2)], diagnostics=[], thread_id="run::v2", resume=True
    )

    assert outputs[0]["skipped"] is True
    assert outputs[0]["response"] == "resp 1"
    assert outputs[1]["skipped"] is False
    assert outputs[1]["response"] == "resp 2 redone"
    # The branch has its own full history; the original is untouched.
    assert len(pipeline.get_thread_messages("run::v2")) == 4
    assert [m.content for m in pipeline.get_thread_messages("run::v1")][-1] == "resp 2"


# ---------------------------------------------------------------------------
# Rolling summarization wiring
# ---------------------------------------------------------------------------


def test_summarizer_gets_pipeline_store_and_summary_never_persisted():
    from langchain_core.messages import SystemMessage
    from memory.summarize import RollingSummarizer

    fake_llm = FakeListChatModel(responses=["resp 1", "resp 2", "resp 3"])
    mem = MemoryHub()
    summarizer = RollingSummarizer(
        lambda prompt: "condensed history",
        trigger_tokens=1,
        keep_last_turns=0,
    )
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused"),
        memory_setup=MemorySetup(memory=mem),
        llm=fake_llm,
        window_k=None,
        summarizer=summarizer,
    )
    # A store-less summarizer is wired to the pipeline's KV layer.
    assert summarizer.store is mem.kv

    chunks: list[SasBatch | SasChunk] = [
        _mk_chunk(f"f1-chunk-000{i}", "etl.sas", f"data work.t{i}; run;")
        for i in range(3)
    ]
    pipeline._process(items=[_wrap(c) for c in chunks], diagnostics=[], thread_id="run::etl.sas")

    # The summary state lives in the KV layer and covered the folded turns…
    state = mem.kv.get("summary::run::etl.sas")
    assert state is not None
    assert state["summary"] == "condensed history"
    assert state["covered_turns"] == 2  # item 3 saw 2 completed turns
    # …while the persisted history stays pure human/AI — the summary
    # SystemMessage is prompted but never stored.
    history = pipeline.get_thread_messages("run::etl.sas")
    assert len(history) == 6
    assert not any(isinstance(m, SystemMessage) for m in history)


# ---------------------------------------------------------------------------
# LLM endpoint overrides — pipeline arguments reach the ChatOpenAI constructor
# ---------------------------------------------------------------------------


def test_endpoint_overrides_reach_the_chat_model(monkeypatch):
    import llm_client.client as client_mod

    captured: dict = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return FakeListChatModel(responses=["built"])

    monkeypatch.setattr(client_mod, "ChatOpenAI", fake_chat_openai)

    SasLLMPipeline(
        llm_config=LLMClientConfig(model="some-model", temperature=0.2, base_url="https://gateway.example/v1", api_key="sk-secret", url_headers={"X-Team": "sas"}, timeout=42.5, model_kwargs={"top_k": 40}, kwargs={"stop": ["END"]}),
        memory_setup=MemorySetup(memory=MemoryHub()),
    )

    assert captured["model"] == "some-model"
    assert captured["temperature"] == 0.2
    assert captured["base_url"] == "https://gateway.example/v1"
    assert captured["api_key"] == "sk-secret"
    # llm_client also mirrors the key into an `api-key` header for gateways
    # that authenticate on it; see tests/test_llm_client.py.
    assert captured["default_headers"] == {"X-Team": "sas", "api-key": "sk-secret"}
    assert captured["timeout"] == 42.5
    assert captured["model_kwargs"] == {"top_k": 40}
    assert captured["stop"] == ["END"]  # llm_kwargs escape hatch, merged last


# ---------------------------------------------------------------------------
# Prompt caching (Anthropic cache_control on the system prompt)
# ---------------------------------------------------------------------------


def test_prompt_caching_marks_system_block_for_anthropic_models():
    from langchain_core.messages import SystemMessage

    mem = MemoryHub()
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="claude-sonnet-4-5"),
        memory_setup=MemorySetup(memory=mem),
        llm=FakeListChatModel(responses=["ok"]),
        prompt_caching=True,
    )
    system_msg = pipeline._prompt.messages[0]
    assert isinstance(system_msg, SystemMessage)
    (block,) = system_msg.content
    assert block["type"] == "text"
    assert block["cache_control"] == {"type": "ephemeral"}
    # The real system prompt rides in the block — and names the run's target.
    assert DEFAULT_OUTPUT_LANGUAGE_DISPLAY in block["text"]

    # End-to-end: the block-shaped system message flows through the graph.
    c1 = _mk_chunk("f1-chunk-0001", "etl.sas", "data work.a; run;")
    out = pipeline._process(items=[_wrap(c1)], diagnostics=[], thread_id="run::cache")
    assert out[0]["response"] == "ok"


def test_prompt_caching_ignored_for_non_anthropic_models():
    from langchain_core.messages import SystemMessage

    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="gpt-5.4"),
        memory_setup=MemorySetup(memory=MemoryHub()),
        llm=FakeListChatModel(responses=["ok"]),
        prompt_caching=True,
    )
    # Falls back to the plain template tuple (no concrete SystemMessage).
    assert not isinstance(pipeline._prompt.messages[0], SystemMessage)


def test_prompt_caching_off_by_default():
    from langchain_core.messages import SystemMessage

    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="claude-sonnet-4-5"),
        memory_setup=MemorySetup(memory=MemoryHub()),
        llm=FakeListChatModel(responses=["ok"]),
    )
    assert not isinstance(pipeline._prompt.messages[0], SystemMessage)


# ---------------------------------------------------------------------------
# Grouped constructor configs (llm_config / memory_setup)
# ---------------------------------------------------------------------------


def test_llm_config_is_the_canonical_transport_form():
    from llm_client import LLMClientConfig

    config = LLMClientConfig(model="claude-sonnet-4-5", max_retries=7)
    pipeline = SasLLMPipeline(
        llm_config=config,
        memory_setup=MemorySetup(memory=MemoryHub()),
        llm=FakeListChatModel(responses=["ok"]),
    )
    # The config's model becomes the pipeline's model...
    assert pipeline.model == "claude-sonnet-4-5"
    # ...and the client uses the config object as-is.
    assert pipeline._llm_client.config is config


@pytest.mark.parametrize(
    "legacy",
    ["model", "temperature", "base_url", "api_key", "timeout", "max_retries",
     "url_headers", "model_kwargs", "llm_kwargs", "max_input_tokens",
     "requests_per_second", "gateway_version"],
)
def test_transport_kwargs_are_gone(legacy):
    # llm_config is the only spelling now: the individual transport arguments
    # were removed rather than deprecated, so passing one is a TypeError
    # naming it, not a value that quietly goes nowhere.
    with pytest.raises(TypeError, match=legacy):
        SasLLMPipeline(**{legacy: "x"}, llm=FakeListChatModel(responses=["ok"]))


def test_llm_config_defaults_when_omitted():
    # The no-argument constructor still works; it just resolves everything
    # through LLMClientConfig() rather than through a second set of kwargs.
    pipeline = SasLLMPipeline(llm=FakeListChatModel(responses=["ok"]))
    assert pipeline.model == LLMClientConfig().model


def test_memory_setup_is_the_canonical_memory_form():
    from pipeline import MemorySetup

    hub = MemoryHub()
    pipeline = SasLLMPipeline(
        memory_setup=MemorySetup(memory=hub, chat_id="chat-42"),
        llm=FakeListChatModel(responses=["ok"]),
    )
    assert pipeline._memory is hub
    assert pipeline.chat_id == "chat-42"


@pytest.mark.parametrize(
    "legacy",
    ["memory", "task_id", "task_policy", "thread_memory", "memory_extractor",
     "chat_id", "spark", "delta_table"],
)
def test_memory_kwargs_are_gone(legacy):
    # Same rule as the transport half: memory_setup is the only way in.
    with pytest.raises(TypeError, match=legacy):
        SasLLMPipeline(**{legacy: "x"}, llm=FakeListChatModel(responses=["ok"]))


def test_memory_setup_defaults_to_an_in_memory_store():
    pipeline = SasLLMPipeline(llm=FakeListChatModel(responses=["ok"]))
    assert isinstance(pipeline._memory, MemoryHub)


def test_memory_setup_extractor_implies_thread_memory():
    from memory.extractor import MemoryExtractor
    from pipeline import MemorySetup

    hub = MemoryHub()
    built = MemorySetup(
        memory=hub, memory_extractor=MemoryExtractor(model=None)
    ).build()
    # The extractor needs somewhere to put temporary memories, so a thread
    # memory is implied, bound to the hub, and shared with the extractor.
    assert built.thread_memory is not None
    assert built.thread_memory.store is hub.kv
    assert built.extractor.thread_memory is built.thread_memory


# ---------------------------------------------------------------------------
# SAS→Databricks dataset-name mapping (SharePoint CSV step)
# ---------------------------------------------------------------------------


class _FakeSharePointClient:
    """Duck-typed stand-in for app_config.sharepoint's client: read_file only."""

    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.read_paths: list[str] = []

    def read_file(self, path: str) -> bytes:
        self.read_paths.append(path)
        return self.files[path]


_MAPPING_CSV = (
    b"sas_name,databricks_name\n"
    b"work,dev.staging\n"
    b"mylib,prod.sales\n"
)


def _patch_sharepoint(monkeypatch, files: dict[str, bytes]) -> _FakeSharePointClient:
    import app_config.sharepoint as sp_mod

    fake = _FakeSharePointClient(files)
    monkeypatch.setattr(sp_mod, "get_sharepoint_client", lambda: fake)
    return fake


def test_databricks_mapping_loaded_from_sharepoint_csv(monkeypatch):
    from chunker.batcher import load_databricks_mapping_sharepoint

    fake = _patch_sharepoint(monkeypatch, {"maps/sas_to_databricks.csv": _MAPPING_CSV})
    mapping = load_databricks_mapping_sharepoint("maps/sas_to_databricks.csv")
    assert fake.read_paths == ["maps/sas_to_databricks.csv"]
    assert mapping == {
        "work": "dev.staging",
        "mylib": "prod.sales",
    }
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused"),
        memory_setup=MemorySetup(memory=MemoryHub()),
        llm=FakeListChatModel(responses=["ok"]),
        databricks_mapping=mapping,
    )
    assert pipeline.databricks_mapping == mapping
    # The mapping reaches both batchers and rewrites batched dataset names.
    src = (
        "data work.clean;\n set mylib.raw;\n run;\n"
        "proc means data=work.clean; run;\n"
    )
    chunk_result = pipeline.chunker.chunk_text(src, source_id="etl.sas")
    batch_result = pipeline.batcher.batch(chunk_result)
    all_outputs = [
        ds
        for b in batch_result.batches
        for ds in b.output_datasets
    ] + [
        ds for c in batch_result.singletons for ds in c.metadata.output_datasets
    ]
    assert "dev.staging.clean" in all_outputs
    assert pipeline.multi_batcher.databricks_mapping == pipeline.databricks_mapping


def test_explicit_databricks_mapping_overrides_sharepoint_csv(monkeypatch):
    # The merge is the caller's one-liner now: loaded CSV under explicit dict.
    from chunker.batcher import load_databricks_mapping_sharepoint

    _patch_sharepoint(monkeypatch, {"m.csv": _MAPPING_CSV})
    mapping = {
        **load_databricks_mapping_sharepoint("m.csv"),
        **{"work": "override.schema"},
    }
    assert mapping == {
        "work": "override.schema",  # explicit dict wins per key
        "mylib": "prod.sales",  # CSV-only entries survive the merge
    }


def test_empty_sharepoint_mapping_csv_raises(monkeypatch):
    import pytest

    from chunker.batcher import load_databricks_mapping_sharepoint

    _patch_sharepoint(monkeypatch, {"m.csv": b"sas_name,databricks_name\n"})
    with pytest.raises(ValueError, match="zero entries"):
        load_databricks_mapping_sharepoint("m.csv")


def test_no_mapping_keeps_batchers_unmapped():
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused"),
        memory_setup=MemorySetup(memory=MemoryHub()),
        llm=FakeListChatModel(responses=["ok"]),
    )
    assert pipeline.databricks_mapping is None
    assert pipeline.multi_batcher.databricks_mapping is None


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


class _StructuredFakeChatModel(FakeListChatModel):
    """A fake that *can* answer a schema, unlike plain FakeListChatModel.

    ``with_structured_output(..., include_raw=True)`` returns the LangChain
    envelope, so the pipeline exercises the same unpacking a real gateway hits.
    Set ``parsed`` to None (and ``parsing_error``) to simulate a model that
    accepted the schema but did not honour it.
    """

    documents: list = []
    parsing_error: object = None

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        from langchain_core.runnables import RunnableLambda

        documents = list(self.documents)
        parsing_error = self.parsing_error
        raw_responses = list(self.responses)

        def _call(_input, config=None):
            parsed = documents.pop(0) if documents else None
            raw = AIMessage(raw_responses[0] if raw_responses else "")
            if not include_raw:
                return parsed
            return {"raw": raw, "parsed": parsed, "parsing_error": parsing_error}

        return RunnableLambda(_call)


def _translation_document(code: str = 'df = spark.table("a")'):
    from pipeline.response_models import (
        MappingEntry,
        RiskNote,
        TranslationCell,
        TranslationDocument,
    )

    return TranslationDocument(
        analysis="Reads and filters.",
        mapping=[MappingEntry(sas_construct="DATA step", equivalent="filter")],
        cells=[TranslationCell(kind="code", language="python", source=code)],
        risks=[RiskNote(severity="P0", note="Row count may differ.")],
    )


def test_structured_output_attaches_the_document_and_renders_markdown():
    doc = _translation_document()
    fake = _StructuredFakeChatModel(responses=["ignored raw"], documents=[doc])
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused"),
        memory_setup=MemorySetup(memory=MemoryHub()),
        llm=fake,
    )

    outputs = pipeline.run_text("data work.a; x=1; run;", source_id="etl.sas")

    assert len(outputs) == 1
    # The structured document rides alongside...
    assert outputs[0]["document"]["cells"][0]["source"] == 'df = spark.table("a")'
    # ...and the response is the rendered Markdown, not the raw content, so
    # memory and the validation metrics see what they always saw.
    response = outputs[0]["response"]
    assert "## Analysis" in response and "## Translation" in response
    assert "```python" in response
    assert "ignored raw" not in response


def test_structured_turn_persists_rendered_markdown_to_memory():
    mem = MemoryHub()
    doc = _translation_document()
    fake = _StructuredFakeChatModel(responses=["ignored raw"], documents=[doc])
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused"),
        memory_setup=MemorySetup(memory=mem),
        llm=fake,
    )

    pipeline.run_text("data work.a; x=1; run;", source_id="etl.sas", thread_id="t1")

    stored = mem.get_thread("t1").messages
    ai = [m for m in stored if isinstance(m, AIMessage)]
    assert len(ai) == 1
    # An empty stored turn would break resume, history selection, and scoring.
    assert "## Translation" in ai[0].content
    assert ai[0].additional_kwargs["translation_document"]["risks"]


def test_resume_recovers_the_stored_document():
    mem = MemoryHub()
    doc = _translation_document()
    fake = _StructuredFakeChatModel(responses=["raw"], documents=[doc, doc])
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused"),
        memory_setup=MemorySetup(memory=mem),
        llm=fake,
    )

    src = "data work.a; x=1; run;"
    pipeline.run_text(src, source_id="etl.sas", thread_id="t1")
    resumed = pipeline.run_text(
        src, source_id="etl.sas", thread_id="t1", resume=True
    )

    assert resumed[0]["skipped"] is True
    # A recovered item must still be able to produce a notebook.
    assert resumed[0]["document"]["cells"][0]["source"] == 'df = spark.table("a")'


def test_unparsable_structured_response_falls_back_to_the_raw_prose():
    fake = _StructuredFakeChatModel(
        responses=["## Translation\n\n```python\nx = 1\n```\n"],
        documents=[],
        parsing_error=ValueError("schema not honoured"),
    )
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused"),
        memory_setup=MemorySetup(memory=MemoryHub()),
        llm=fake,
    )

    outputs = pipeline.run_text("data work.a; x=1; run;", source_id="etl.sas")

    assert outputs[0]["document"] is None
    assert "x = 1" in outputs[0]["response"]


def test_model_without_structured_support_prompts_for_markdown():
    # FakeListChatModel has no with_structured_output; the pipeline must not
    # fail at construction, and must send the Markdown system prompt.
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused"),
        memory_setup=MemorySetup(memory=MemoryHub()),
        llm=FakeListChatModel(responses=["ok"]),
        structured_output=True,
    )
    assert pipeline._structured_output is False
    assert "Structure every response with these Markdown sections" in (
        pipeline._system_prompt
    )


def test_structured_output_can_be_turned_off():
    doc = _translation_document()
    fake = _StructuredFakeChatModel(responses=["plain answer"], documents=[doc])
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused"), memory_setup=MemorySetup(memory=MemoryHub()), llm=fake, structured_output=False
    )

    outputs = pipeline.run_text("data work.a; x=1; run;", source_id="etl.sas")

    assert outputs[0]["document"] is None
    assert outputs[0]["response"] == "plain answer"


# ---------------------------------------------------------------------------
# Target output language
# ---------------------------------------------------------------------------


def _language_pipeline(**kwargs) -> SasLLMPipeline:
    return SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused"),
        memory_setup=MemorySetup(memory=MemoryHub()),
        llm=FakeListChatModel(responses=["ok"]),
        **kwargs,
    )


def test_output_language_is_resolved_and_canonicalised_once():
    pipeline = _language_pipeline(output_language="spark sql")
    assert pipeline.output_language == "Spark SQL"
    assert pipeline.target_language.default_fence == "sql"
    # The canonical name, not the caller's spelling, is what gets prompted.
    assert "SAS-to-Spark SQL" in pipeline._system_prompt


def test_unknown_output_language_is_rejected_at_construction():
    from target_language import UnknownTargetLanguage

    with pytest.raises(UnknownTargetLanguage):
        _language_pipeline(output_language="Cobol")


def test_system_prompt_names_the_fence_the_translation_must_carry():
    pipeline = _language_pipeline(output_language="SparkSQL")
    prompt = pipeline._system_prompt
    assert "```sql" in prompt
    assert "Translate into Spark SQL and nothing else" in prompt


def test_structured_prompt_names_the_cell_language_instead_of_a_fence():
    doc = _translation_document()
    fake = _StructuredFakeChatModel(responses=["ok"], documents=[doc])
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused"),
        memory_setup=MemorySetup(memory=MemoryHub()),
        llm=fake,
        structured_output=True,
        output_language="SparkSQL",
    )
    prompt = pipeline._system_prompt
    assert pipeline._structured_output is True
    assert "'sql'" in prompt
    # The structured path forbids fences, so it must not ask for one.
    assert "```sql" not in prompt


def test_output_language_falls_back_to_config(monkeypatch):
    # Stands in for config.json pipeline.output_language, which is what
    # _configured_default reads.
    import target_language

    monkeypatch.setattr(target_language, "_configured_default", lambda: "PySpark")
    assert _language_pipeline().output_language == "PySpark"


def test_structured_document_renders_with_the_target_fence():
    from pipeline.response_models import TranslationCell, TranslationDocument

    doc = TranslationDocument(
        analysis="a",
        cells=[TranslationCell(kind="code", source="SELECT 1")],
    )
    fake = _StructuredFakeChatModel(responses=["ok"], documents=[doc])
    pipeline = SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused"),
        memory_setup=MemorySetup(memory=MemoryHub()),
        llm=fake,
        structured_output=True,
        output_language="SparkSQL",
    )

    outputs = pipeline.run_text("data work.a; x=1; run;", source_id="etl.sas")

    # Untagged cell + Spark SQL run must not render as ```python: the
    # validation suite reads exactly this text.
    assert "```sql" in outputs[0]["response"]
    assert "```python" not in outputs[0]["response"]
