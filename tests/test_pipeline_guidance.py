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

from chunker.models import SasBatch, SasChunk, SasChunkKind, SasChunkMetadata
from chunker.pipeline import (
    SasLLMPipeline,
    _constructs_for_item,
    _kinds_for_item,
    _meta_flags_for_item,
    _query_for_item,
)
from memory.store import MemoryHub
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
        memory=MemoryHub(),
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


def test_component_objects_map_to_constructs_and_query():
    text = "data a; declare hash h1(dataset:'work.notes'); h1.definedone(); run;"
    chunk = SasChunk(
        chunk_id="c2",
        source_id="etl.sas",
        text=text,
        kind=SasChunkKind.DATA_STEP,
        title="hash lookup",
        start_line=1,
        end_line=1,
        start_char=0,
        end_char=len(text),
        metadata=SasChunkMetadata(component_objects=["hash"]),
    )
    assert ConstructKey(kind="component_object", name="hash") in _constructs_for_item(
        chunk
    )
    assert "hash object" in _query_for_item(chunk)


# ---------------------------------------------------------------------------
# SasBatch construct-metadata aggregation (sets) and batch-level gating
# ---------------------------------------------------------------------------


def _meta_chunk(chunk_id: str, **meta) -> SasChunk:
    kind = meta.pop("kind", SasChunkKind.DATA_STEP)
    return SasChunk(
        chunk_id=chunk_id,
        source_id="etl.sas",
        text="x;",
        kind=kind,
        start_line=1,
        end_line=1,
        start_char=0,
        end_char=2,
        metadata=SasChunkMetadata(**meta),
    )


def _batch(*chunks: SasChunk) -> SasBatch:
    return SasBatch(batch_id="batch-001", chunks=list(chunks))


def test_sasbatch_aggregates_member_metadata_as_sets():
    batch = _batch(
        _meta_chunk("c1", recognized_functions=["intck", "intnx"]),
        _meta_chunk("c2", recognized_functions=["intnx"], component_objects=["hash"]),
        _meta_chunk("c3", kind=SasChunkKind.PROC_STEP, proc_name="sql"),
        _meta_chunk("c4", recognized_call_routines=["symput"], symput_scope_hazard=True),
    )
    assert batch.recognized_functions == {"intck", "intnx"}  # deduped union
    assert batch.recognized_call_routines == {"symput"}
    assert batch.component_objects == {"hash"}
    assert batch.proc_names == {"sql"}
    assert batch.has_symput_scope_hazard is True
    assert batch.has_abort is False
    assert isinstance(batch.recognized_functions, set)


def test_batch_model_dump_stays_json_serializable():
    # The set-valued aggregates are plain properties, not fields, so they do
    # not leak into model_dump() (json.dumps would choke on a set).
    import json

    batch = _batch(_meta_chunk("c1", recognized_functions=["intck"]))
    json.dumps(batch.model_dump())  # must not raise


def test_constructs_for_item_unions_across_batch_members():
    batch = _batch(
        _meta_chunk("c1", recognized_functions=["intck"]),
        _meta_chunk("c2", recognized_functions=["intnx"], component_objects=["hash"]),
    )
    keys = _constructs_for_item(batch)
    assert ConstructKey(kind="function", name="intck") in keys
    assert ConstructKey(kind="function", name="intnx") in keys
    assert ConstructKey(kind="component_object", name="hash") in keys


def test_instruction_injected_only_when_construct_present_in_batch():
    # Two construct-scoped rules; a batch pulls only the ones whose construct
    # it actually uses — the load-bearing gating the request asks us to verify.
    rules = (
        "## [when: function:intck] INTCK rule\nCount interval boundaries.\n"
        "## [when: function:intnx] INTNX rule\nAdvance by intervals."
    )
    builder = PromptBuilder([], user_instructions=rules)

    intck_batch = _batch(_meta_chunk("c1", recognized_functions=["intck"]))
    out = builder.build(
        _query_for_item(intck_batch), _constructs_for_item(intck_batch)
    )
    assert "INTCK rule" in out
    assert "INTNX rule" not in out  # not used by this batch -> not injected

    other_batch = _batch(_meta_chunk("c2", recognized_functions=["put"]))
    assert builder.build(
        _query_for_item(other_batch), _constructs_for_item(other_batch)
    ) is None  # neither rule's construct present -> no guidance at all


# ---------------------------------------------------------------------------
# [kind:] / [meta:] item mapping and end-to-end gating
# ---------------------------------------------------------------------------


def test_kinds_for_item_unions_member_kinds():
    batch = _batch(
        _meta_chunk("c1", kind=SasChunkKind.DATA_STEP),
        _meta_chunk("c2", kind=SasChunkKind.PROC_STEP, proc_name="sql"),
    )
    assert _kinds_for_item(batch) == {"DATA_STEP", "PROC_STEP"}
    assert _kinds_for_item(_intnx_chunk()) == {"DATA_STEP"}


def test_meta_flags_for_item_maps_metadata_predicates():
    batch = _batch(
        _meta_chunk("c1", symput_scope_hazard=True),
        _meta_chunk("c2", component_objects=["hash"], has_unclosed_block=True),
    )
    flags = _meta_flags_for_item(batch)
    assert {"symput_hazard", "component_object", "unclosed_block"} <= flags
    assert "abort" not in flags


def test_kind_and_meta_gate_instruction_injection_end_to_end():
    rules = (
        "## [kind: PROC_STEP] PROC rule\nOnly for PROC steps.\n"
        "## [meta: symput_hazard] SYMPUT rule\nMind the write/read ordering."
    )
    builder = PromptBuilder([], user_instructions=rules)

    proc_batch = _batch(_meta_chunk("c1", kind=SasChunkKind.PROC_STEP, proc_name="sql"))
    out = builder.build(
        _query_for_item(proc_batch),
        _constructs_for_item(proc_batch),
        kinds=_kinds_for_item(proc_batch),
        meta_flags=_meta_flags_for_item(proc_batch),
    )
    assert "PROC rule" in out
    assert "SYMPUT rule" not in out  # no hazard flag on this batch

    hazard_batch = _batch(
        _meta_chunk("c2", kind=SasChunkKind.DATA_STEP, symput_scope_hazard=True)
    )
    out2 = builder.build(
        _query_for_item(hazard_batch),
        _constructs_for_item(hazard_batch),
        kinds=_kinds_for_item(hazard_batch),
        meta_flags=_meta_flags_for_item(hazard_batch),
    )
    assert "SYMPUT rule" in out2
    assert "PROC rule" not in out2  # not a PROC step


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


# ---------------------------------------------------------------------------
# User instructions through the pipeline (step 3 wiring)
# ---------------------------------------------------------------------------

# Like GUIDANCE_MARKER: phrases that exist only in operator rules, so their
# presence in a prompt or the store is unambiguous.
USER_MARKER = "ZZUSERRULE emit one risk table per step"
OLD_MARKER = "ZZOLDRULE previous project law"


def test_user_instructions_without_builder_prompted_not_persisted():
    llm = _RecordingChatModel()
    pipeline = SasLLMPipeline(
        model="unused-because-llm-injected",
        memory=MemoryHub(),
        llm=llm,
        user_instructions=f"## Output rules\n{USER_MARKER}.",
    )
    pipeline._process(items=[_intnx_chunk()], diagnostics=[], thread_id="run::etl.sas")

    # Prompted: system + instructions + human.
    assert len(llm.prompts[0]) == 3
    prompted = "\n".join(str(m.content) for m in llm.prompts[0])
    assert USER_MARKER in prompted
    assert "## Project instructions" in prompted

    stored = pipeline.get_thread_messages("run::etl.sas")
    assert len(stored) == 2  # item + response only
    assert USER_MARKER not in "\n".join(str(m.content) for m in stored)


def test_user_instructions_fold_into_given_builder():
    llm = _RecordingChatModel()
    pipeline = SasLLMPipeline(
        model="unused-because-llm-injected",
        memory=MemoryHub(),
        llm=llm,
        prompt_builder=PromptBuilder(_guidance_corpus()),
        user_instructions=f"## Output rules\n{USER_MARKER}.",
    )
    pipeline._process(items=[_intnx_chunk()], diagnostics=[], thread_id="run::etl.sas")

    instructions_msg = str(llm.prompts[0][1].content)
    assert USER_MARKER in instructions_msg  # operator rules present...
    assert GUIDANCE_MARKER in instructions_msg  # ...alongside reference guidance
    # Project block renders above the reference-guidance block.
    assert instructions_msg.index("## Project instructions") < instructions_msg.index(
        "## Relevant migration guidance"
    )


def test_pipeline_level_instructions_replace_builders_own():
    llm = _RecordingChatModel()
    builder = PromptBuilder(
        _guidance_corpus(), user_instructions=f"## Old\n{OLD_MARKER}."
    )
    pipeline = SasLLMPipeline(
        model="unused-because-llm-injected",
        memory=MemoryHub(),
        llm=llm,
        prompt_builder=builder,
        user_instructions=f"## New\n{USER_MARKER}.",
    )
    pipeline._process(items=[_intnx_chunk()], diagnostics=[], thread_id="run::etl.sas")

    prompted = "\n".join(str(m.content) for m in llm.prompts[0])
    assert USER_MARKER in prompted
    assert OLD_MARKER not in prompted
    # The original builder object is untouched.
    assert OLD_MARKER in builder.build("zzz", [])


def test_conditional_rule_scoped_end_to_end():
    llm = _RecordingChatModel()
    pipeline = SasLLMPipeline(
        model="unused-because-llm-injected",
        memory=MemoryHub(),
        llm=llm,
        user_instructions=f"## [when: function:intnx] Date rules\n{USER_MARKER}.",
    )
    intnx = _intnx_chunk()
    print_text = "proc print data=work.x; run;"
    print_chunk = SasChunk(
        chunk_id="c9",
        source_id="etl.sas",
        text=print_text,
        kind=SasChunkKind.PROC_STEP,
        title="print",
        start_line=1,
        end_line=1,
        start_char=0,
        end_char=len(print_text),
        metadata=SasChunkMetadata(proc_name="print"),
    )
    pipeline._process(
        items=[intnx, print_chunk], diagnostics=[], thread_id="run::etl.sas"
    )

    assert USER_MARKER in "\n".join(str(m.content) for m in llm.prompts[0])
    # The PROC PRINT item has no intnx construct, and turn 1's guidance was
    # ephemeral — so the marker is absent from turn 2's ENTIRE prompt,
    # history included.
    assert USER_MARKER not in "\n".join(str(m.content) for m in llm.prompts[1])


def test_standing_instructions_file_from_config(monkeypatch, tmp_path):
    import json

    import app_config

    rules_path = tmp_path / "instructions.md"
    rules_path.write_text(f"## Output rules\n{USER_MARKER}.", encoding="utf-8")
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps({"user_instructions": {"path": str(rules_path)}}),
        encoding="utf-8",
    )
    monkeypatch.setenv(app_config.ENV_VAR, str(cfg))
    app_config.clear_cache()
    try:
        llm = _RecordingChatModel()
        # No user_instructions argument: the configured standing file applies.
        pipeline = SasLLMPipeline(
            model="unused-because-llm-injected",
            memory=MemoryHub(),
            llm=llm,
        )
        pipeline._process(
            items=[_intnx_chunk()], diagnostics=[], thread_id="run::etl.sas"
        )
        prompted = "\n".join(str(m.content) for m in llm.prompts[0])
        assert USER_MARKER in prompted
        assert pipeline.instructions_fingerprint is not None
        assert len(pipeline.instructions_fingerprint) == 16
    finally:
        app_config.clear_cache()


def test_instructions_fingerprint_property():
    llm = _RecordingChatModel()
    with_rules = SasLLMPipeline(
        model="unused-because-llm-injected",
        memory=MemoryHub(),
        llm=llm,
        user_instructions="## A\nrule body.",
    )
    without = _pipeline(_RecordingChatModel(), None)
    assert with_rules.instructions_fingerprint is not None
    assert without.instructions_fingerprint is None


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
