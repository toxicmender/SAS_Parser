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
from chunker.pipeline import (
    SasLLMPipeline,
    _format_batch_message,
    _format_chunk_message,
)


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


# ---------------------------------------------------------------------------
# Pure formatting functions — no Spark/LLM required
# ---------------------------------------------------------------------------


def test_format_chunk_message_includes_key_fields():
    chunk = _mk_chunk(
        "f1-chunk-0001",
        "etl.sas",
        "data work.out; set work.in; run;",
        input_datasets=["work.in"],
        output_datasets=["work.out"],
        symput_scope_hazard=True,
        symput_hazard_vars=["cutoff"],
    )
    msg = _format_chunk_message(chunk, index=1, total=1, diagnostics=[])

    assert "work.in" in msg
    assert "work.out" in msg
    assert "yes (cutoff)" in msg  # symput hazard line
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
        model="unused-because-llm-injected",
        memory=mem,
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

    results = pipeline._process(
        items=[batch, c1],  # a batch, then an unrelated singleton
        diagnostics=[],
        thread_id="run::etl.sas",
    )

    assert len(results) == 2
    assert results[0]["is_batch"] is True
    assert results[0]["item_id"] == "batch-001"
    assert results[0]["response"] == "translation for item 1"
    assert results[1]["is_batch"] is False
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


def test_pipeline_window_trimming_limits_injected_history():
    fake_llm = FakeListChatModel(responses=[f"resp {i}" for i in range(6)])
    mem = MemoryHub()
    pipeline = SasLLMPipeline(
        model="unused-because-llm-injected",
        memory=mem,
        llm=fake_llm,
        window_k=1,  # keep only last 1 human/ai pair in the prompt
    )

    chunks: list[SasBatch | SasChunk] = [
        _mk_chunk(f"f1-chunk-000{i}", "etl.sas", f"data work.t{i}; run;")
        for i in range(3)
    ]
    pipeline._process(items=chunks, diagnostics=[], thread_id="run::etl.sas")

    # Full history is still persisted (trimming only affects the prompt)...
    full_history = pipeline.get_thread_messages("run::etl.sas")
    assert len(full_history) == 6  # 3 human + 3 ai


def test_snapshot_delegates_to_memory():
    fake_llm = FakeListChatModel(responses=["ok"])
    mem = MemoryHub()
    pipeline = SasLLMPipeline(model="unused", memory=mem, llm=fake_llm)

    c1 = _mk_chunk("f1-chunk-0001", "etl.sas", "data work.a; run;")
    pipeline._process(items=[c1], diagnostics=[], thread_id="run::etl.sas")

    snap = pipeline.snapshot()
    assert snap == mem.snapshot()
    assert any("run::etl.sas" in k for k in snap)


# ---------------------------------------------------------------------------
# Run facts — the per-item KV write channel
# ---------------------------------------------------------------------------


def test_run_facts_recorded_per_item():
    fake_llm = FakeListChatModel(responses=["resp 1", "resp 2"])
    mem = MemoryHub()
    pipeline = SasLLMPipeline(model="unused", memory=mem, llm=fake_llm)

    chunks: list[SasBatch | SasChunk] = [
        _mk_chunk("f1-chunk-0001", "etl.sas", "data work.a; run;"),
        _mk_chunk("f1-chunk-0002", "etl.sas", "proc print data=work.a; run;"),
    ]
    pipeline._process(items=chunks, diagnostics=[], thread_id="run::etl.sas")

    facts = pipeline.get_run_facts("run::etl.sas")
    assert [f["item_id"] for f in facts] == ["f1-chunk-0001", "f1-chunk-0002"]
    assert all(f["status"] == "ok" for f in facts)
    assert [f["index"] for f in facts] == [1, 2]
    assert all(f["response_chars"] == len("resp 1") for f in facts)
    # Facts live in the KV layer, isolated from the msg:: history.
    assert len(pipeline.get_thread_messages("run::etl.sas")) == 4


def test_run_facts_isolated_per_thread():
    fake_llm = FakeListChatModel(responses=["a", "b"])
    pipeline = SasLLMPipeline(model="unused", memory=MemoryHub(), llm=fake_llm)
    pipeline._process(
        items=[_mk_chunk("c1", "a.sas", "data work.a; run;")],
        diagnostics=[],
        thread_id="run::a.sas",
    )
    pipeline._process(
        items=[_mk_chunk("c2", "b.sas", "data work.b; run;")],
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
    pipeline = SasLLMPipeline(model="unused", memory=mem, llm=fake_llm)

    c1 = _mk_chunk("f1-chunk-0001", "etl.sas", "data work.a; run;")
    c2 = _mk_chunk("f1-chunk-0002", "etl.sas", "proc print data=work.a; run;")

    # First run "crashed" after item 1: only c1 was processed.
    pipeline._process(items=[c1], diagnostics=[], thread_id="run::etl.sas")

    outputs = pipeline._process(
        items=[c1, c2], diagnostics=[], thread_id="run::etl.sas", resume=True
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
    pipeline = SasLLMPipeline(model="unused", memory=mem, llm=fake_llm)

    c1 = _mk_chunk("f1-chunk-0001", "etl.sas", "data work.a; run;")
    pipeline._process(items=[c1], diagnostics=[], thread_id="run::etl.sas")
    c2 = _mk_chunk("f1-chunk-0002", "etl.sas", "proc print data=work.a; run;")
    # Simulate a crashed second item: an error fact, no persisted turn.
    mem.kv.set(
        "run::run::etl.sas::item::f1-chunk-0002",
        {"status": "error", "index": 2, "error": "boom"},
    )

    outputs = pipeline._process(
        items=[c1, c2], diagnostics=[], thread_id="run::etl.sas", resume=True
    )

    assert outputs[1]["skipped"] is False  # error fact does not skip
    assert outputs[1]["response"] == "resp 2"
    facts = pipeline.get_run_facts("run::etl.sas")
    assert [f["status"] for f in facts] == ["ok", "ok"]  # overwritten


def test_fork_run_then_resume_continues_from_the_fork():
    mem = MemoryHub()
    fake_llm = FakeListChatModel(responses=["resp 1", "resp 2", "resp 2 redone"])
    pipeline = SasLLMPipeline(model="unused", memory=mem, llm=fake_llm)

    c1 = _mk_chunk("f1-chunk-0001", "etl.sas", "data work.a; run;")
    c2 = _mk_chunk("f1-chunk-0002", "etl.sas", "proc print data=work.a; run;")
    pipeline._process(items=[c1, c2], diagnostics=[], thread_id="run::v1")

    # Rewind to after item 1 and redo item 2 on a fresh branch.
    copied = pipeline.fork_run("run::v1", "run::v2", upto_items=1)
    assert copied == 2  # one (human, AI) pair

    outputs = pipeline._process(
        items=[c1, c2], diagnostics=[], thread_id="run::v2", resume=True
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
        model="unused",
        memory=mem,
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
    pipeline._process(items=chunks, diagnostics=[], thread_id="run::etl.sas")

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
# LLM endpoint overrides — pipeline arguments reach init_chat_model
# ---------------------------------------------------------------------------


def test_endpoint_overrides_reach_init_chat_model(monkeypatch):
    import llm_client.client as client_mod

    captured: dict = {}

    def fake_init(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return FakeListChatModel(responses=["built"])

    monkeypatch.setattr(client_mod, "init_chat_model", fake_init)

    SasLLMPipeline(
        model="some-model",
        temperature=0.2,
        base_url="https://gateway.example/v1",
        api_key="sk-secret",
        url_headers={"X-Team": "sas"},
        timeout=42.5,
        model_kwargs={"top_k": 40},
        llm_kwargs={"stop": ["END"]},
        memory=MemoryHub(),
    )

    assert captured["model"] == "some-model"
    assert captured["temperature"] == 0.2
    assert captured["base_url"] == "https://gateway.example/v1"
    assert captured["api_key"] == "sk-secret"
    assert captured["default_headers"] == {"X-Team": "sas"}
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
        model="claude-sonnet-4-5",
        memory=mem,
        llm=FakeListChatModel(responses=["ok"]),
        prompt_caching=True,
    )
    system_msg = pipeline._prompt.messages[0]
    assert isinstance(system_msg, SystemMessage)
    (block,) = system_msg.content
    assert block["type"] == "text"
    assert block["cache_control"] == {"type": "ephemeral"}
    assert "PySpark" in block["text"]  # the real system prompt rides in the block

    # End-to-end: the block-shaped system message flows through the graph.
    c1 = _mk_chunk("f1-chunk-0001", "etl.sas", "data work.a; run;")
    out = pipeline._process(items=[c1], diagnostics=[], thread_id="run::cache")
    assert out[0]["response"] == "ok"


def test_prompt_caching_ignored_for_non_anthropic_models():
    from langchain_core.messages import SystemMessage

    pipeline = SasLLMPipeline(
        model="gpt-5.4",
        memory=MemoryHub(),
        llm=FakeListChatModel(responses=["ok"]),
        prompt_caching=True,
    )
    # Falls back to the plain template tuple (no concrete SystemMessage).
    assert not isinstance(pipeline._prompt.messages[0], SystemMessage)


def test_prompt_caching_off_by_default():
    from langchain_core.messages import SystemMessage

    pipeline = SasLLMPipeline(
        model="claude-sonnet-4-5",
        memory=MemoryHub(),
        llm=FakeListChatModel(responses=["ok"]),
    )
    assert not isinstance(pipeline._prompt.messages[0], SystemMessage)


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
    fake = _patch_sharepoint(monkeypatch, {"maps/sas_to_databricks.csv": _MAPPING_CSV})
    pipeline = SasLLMPipeline(
        model="unused",
        memory=MemoryHub(),
        llm=FakeListChatModel(responses=["ok"]),
        databricks_mapping_sharepoint="maps/sas_to_databricks.csv",
    )
    assert fake.read_paths == ["maps/sas_to_databricks.csv"]
    assert pipeline.databricks_mapping == {
        "work": "dev.staging",
        "mylib": "prod.sales",
    }
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
    _patch_sharepoint(monkeypatch, {"m.csv": _MAPPING_CSV})
    pipeline = SasLLMPipeline(
        model="unused",
        memory=MemoryHub(),
        llm=FakeListChatModel(responses=["ok"]),
        databricks_mapping={"work": "override.schema"},
        databricks_mapping_sharepoint="m.csv",
    )
    assert pipeline.databricks_mapping == {
        "work": "override.schema",  # explicit dict wins per key
        "mylib": "prod.sales",  # CSV-only entries survive the merge
    }


def test_empty_sharepoint_mapping_csv_raises(monkeypatch):
    import pytest

    _patch_sharepoint(monkeypatch, {"m.csv": b"sas_name,databricks_name\n"})
    with pytest.raises(ValueError, match="zero entries"):
        SasLLMPipeline(
            model="unused",
            memory=MemoryHub(),
            llm=FakeListChatModel(responses=["ok"]),
            databricks_mapping_sharepoint="m.csv",
        )


def test_no_mapping_keeps_batchers_unmapped():
    pipeline = SasLLMPipeline(
        model="unused",
        memory=MemoryHub(),
        llm=FakeListChatModel(responses=["ok"]),
    )
    assert pipeline.databricks_mapping is None
    assert pipeline.multi_batcher.databricks_mapping is None
