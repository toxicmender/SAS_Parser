"""
Tests for reference-guidance injection in SasLLMPipeline (Phase 6 wiring).

The load-bearing contract: guidance is *prompted* to the LLM but *never*
persisted to the thread's message history. Fully offline — a recording
chat-model stub captures every prompted message list, and the PromptBuilder
runs over an in-memory chunk corpus (no PDFs).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage

from chunker.models import SasChunk, SasChunkKind, SasChunkMetadata
from chunker.pipeline import (
    SasLLMPipeline,
    _constructs_for_item,
    _query_for_item,
)
from memory.short_mem import DatabricksMemory
from prompt_builder.builder import PromptBuilder
from prompt_builder.models import ConstructKey, DocRole, InstructionChunk

# A phrase that appears ONLY in the reference corpus, never in a SAS chunk, so
# its presence unambiguously means guidance was injected.
GUIDANCE_MARKER = "ZZGUIDEMARKER advances a sas date by intervals"


class _RecordingChatModel:
    """Chat-model stub that records every prompted message list."""

    def __init__(self) -> None:
        self.prompts: list[list] = []

    def invoke(self, messages, config=None):
        self.prompts.append(list(messages))
        return AIMessage(f"resp {len(self.prompts)}")


def _guidance_corpus() -> list[InstructionChunk]:
    return [
        InstructionChunk(
            chunk_id="functions::c0",
            doc_id="functions",
            section_path="Funcs > INTNX Function",
            text=f"Funcs > INTNX Function\n\n{GUIDANCE_MARKER}",
            page_start=41,
            page_end=43,
            role=DocRole.SAS_REFERENCE,
            construct_keys=[ConstructKey(kind="function", name="intnx")],
        )
    ]


def _intnx_chunk() -> SasChunk:
    text = "data out; set in; x = intnx('month', d, 1); run;"
    return SasChunk(
        chunk_id="c1",
        source_id="etl.sas",
        text=text,
        kind=SasChunkKind.DATA_STEP,
        title="compute month offset",
        start_line=1,
        end_line=1,
        start_char=0,
        end_char=len(text),
        metadata=SasChunkMetadata(recognized_functions=["intnx"]),
    )


def _pipeline(llm, prompt_builder):
    return SasLLMPipeline(
        model="unused-because-llm-injected",
        memory=DatabricksMemory(),
        llm=llm,
        prompt_builder=prompt_builder,
    )


# ---------------------------------------------------------------------------
# Metadata -> query / constructs mapping
# ---------------------------------------------------------------------------


def test_constructs_for_item_maps_functions_and_hazards():
    chunk = _intnx_chunk()
    chunk.metadata.symput_scope_hazard = True
    keys = _constructs_for_item(chunk)
    assert ConstructKey(kind="function", name="intnx") in keys
    assert ConstructKey(kind="call_routine", name="symput") in keys  # hazard added


def test_query_for_item_uses_constructs_not_dataset_names():
    query = _query_for_item(_intnx_chunk())
    assert "intnx" in query
    assert "out" not in query.split()  # dataset name is not a query token


# ---------------------------------------------------------------------------
# Injection contract: prompted, never persisted
# ---------------------------------------------------------------------------


def test_guidance_is_prompted():
    llm = _RecordingChatModel()
    pipeline = _pipeline(llm, PromptBuilder(_guidance_corpus()))
    pipeline._process(items=[_intnx_chunk()], diagnostics=[], thread_id="run::etl.sas")

    prompted = "\n".join(str(m.content) for m in llm.prompts[0])
    assert GUIDANCE_MARKER in prompted  # guidance reached the LLM


def test_guidance_is_not_persisted_to_history():
    llm = _RecordingChatModel()
    pipeline = _pipeline(llm, PromptBuilder(_guidance_corpus()))
    pipeline._process(items=[_intnx_chunk()], diagnostics=[], thread_id="run::etl.sas")

    stored = pipeline.get_thread_messages("run::etl.sas")
    stored_text = "\n".join(str(m.content) for m in stored)
    assert len(stored) == 2  # exactly the human item message + AI response
    assert GUIDANCE_MARKER not in stored_text  # guidance never entered the store


def test_no_prompt_builder_means_no_guidance_message():
    llm = _RecordingChatModel()
    pipeline = _pipeline(llm, None)
    pipeline._process(items=[_intnx_chunk()], diagnostics=[], thread_id="run::etl.sas")

    # system + (empty history) + (empty instructions) + human == 2 messages.
    assert len(llm.prompts[0]) == 2


def test_irrelevant_item_injects_no_guidance():
    llm = _RecordingChatModel()
    pipeline = _pipeline(llm, PromptBuilder(_guidance_corpus()))
    # A chunk whose constructs/tokens don't match the corpus at all.
    text = "proc print data=work.x; run;"
    chunk = SasChunk(
        chunk_id="c9",
        source_id="etl.sas",
        text=text,
        kind=SasChunkKind.PROC_STEP,
        title="print",
        start_line=1,
        end_line=1,
        start_char=0,
        end_char=len(text),
        metadata=SasChunkMetadata(proc_name="print"),
    )
    pipeline._process(items=[chunk], diagnostics=[], thread_id="run::etl.sas")
    assert len(llm.prompts[0]) == 2  # no guidance system message added
