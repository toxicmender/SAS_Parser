"""
Smoke tests for pipeline.py's integration with memory.short_mem.py.

These tests deliberately avoid any live LLM call: SasLLMPipeline is
constructed with a FakeListChatModel so we can verify the actual thing
this integration is responsible for — that batches/chunks get formatted
correctly and that conversation state round-trips through
DatabricksMemory / KVChatMessageHistory — without needing API credentials
or network access.

Requires: pyspark, langchain-core (FakeListChatModel).
"""

from __future__ import annotations

import pytest

pyspark = pytest.importorskip("pyspark")

from chunker.persistent_memory import DatabricksMemory
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pyspark.sql import SparkSession

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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spark():
    s = (
        SparkSession.builder.master("local[1]")
        .appName("chunker_pipeline_tests")
        .getOrCreate()
    )
    yield s
    s.stop()


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
# Persistence wiring — KVChatMessageHistory round-trip via DatabricksMemory
# ---------------------------------------------------------------------------


def test_memory_thread_round_trips_messages(spark):
    mem = DatabricksMemory(spark=spark)  # in-memory, no Delta table
    thread = mem.get_thread("run::etl.sas")

    thread.add_user_message("batch-001 content")
    thread.add_ai_message("translated PySpark for batch-001")

    same_thread = mem.get_thread("run::etl.sas")  # same id -> same history
    assert len(same_thread.messages) == 2
    assert isinstance(same_thread.messages[0], HumanMessage)
    assert isinstance(same_thread.messages[1], AIMessage)
    assert same_thread.messages[1].content == "translated PySpark for batch-001"


def test_different_thread_ids_are_isolated(spark):
    mem = DatabricksMemory(spark=spark)
    mem.get_thread("run::a.sas").add_user_message("hello a")
    mem.get_thread("run::b.sas").add_user_message("hello b")

    assert [m.content for m in mem.get_thread("run::a.sas").messages] == ["hello a"]
    assert [m.content for m in mem.get_thread("run::b.sas").messages] == ["hello b"]


# ---------------------------------------------------------------------------
# End-to-end pipeline wiring with a fake LLM — no network/API key needed
# ---------------------------------------------------------------------------


def test_pipeline_accumulates_history_across_batches(spark):
    fake_llm = FakeListChatModel(
        responses=["translation for item 1", "translation for item 2"]
    )
    mem = DatabricksMemory(spark=spark)

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


def test_pipeline_window_trimming_limits_injected_history(spark):
    fake_llm = FakeListChatModel(responses=[f"resp {i}" for i in range(6)])
    mem = DatabricksMemory(spark=spark)
    pipeline = SasLLMPipeline(
        model="unused-because-llm-injected",
        memory=mem,
        llm=fake_llm,
        window_k=1,  # keep only last 1 human/ai pair in the prompt
    )

    chunks = [
        _mk_chunk(f"f1-chunk-000{i}", "etl.sas", f"data work.t{i}; run;")
        for i in range(3)
    ]
    pipeline._process(items=chunks, diagnostics=[], thread_id="run::etl.sas")

    # Full history is still persisted (trimming only affects the prompt)...
    full_history = pipeline.get_thread_messages("run::etl.sas")
    assert len(full_history) == 6  # 3 human + 3 ai


def test_snapshot_delegates_to_memory(spark):
    fake_llm = FakeListChatModel(responses=["ok"])
    mem = DatabricksMemory(spark=spark)
    pipeline = SasLLMPipeline(model="unused", memory=mem, llm=fake_llm)

    c1 = _mk_chunk("f1-chunk-0001", "etl.sas", "data work.a; run;")
    pipeline._process(items=[c1], diagnostics=[], thread_id="run::etl.sas")

    snap = pipeline.snapshot()
    assert snap == mem.snapshot()
    assert any("run::etl.sas" in k for k in snap)
