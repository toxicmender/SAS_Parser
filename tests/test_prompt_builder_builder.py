"""
Tests for prompt_builder.builder — PromptBuilder guidance-block formatting,
the None-when-empty contract, pinned sections, and from_specs loading.

Fully offline: chunks are built directly; from_specs uses a pymupdf fixture PDF.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pymupdf

from prompt_builder.builder import PromptBuilder
from prompt_builder.catalog import DocumentSpec
from prompt_builder.models import ConstructKey, DocRole, InstructionChunk


def _chunk(chunk_id, section_path, body, *, keys=None, page_start=1, page_end=1):
    return InstructionChunk(
        chunk_id=chunk_id,
        doc_id="functions",
        section_path=section_path,
        text=f"{section_path}\n\n{body}",
        page_start=page_start,
        page_end=page_end,
        role=DocRole.SAS_REFERENCE,
        construct_keys=keys or [],
    )


INTNX = ConstructKey(kind="function", name="intnx")


def _corpus():
    return [
        _chunk(
            "c0",
            "Funcs > INTNX Function",
            "advances a sas date by a number of intervals",
            keys=[INTNX],
            page_start=41,
            page_end=43,
        ),
        _chunk("c1", "Guidelines > Output Format", "always return structured markdown"),
    ]


def test_build_formats_markdown_block():
    pb = PromptBuilder(_corpus())
    block = pb.build("advance a date interval", [INTNX])
    assert block is not None
    # Focus hints (directional stimulus) render above the reference guidance.
    assert block.startswith("## Focus hints")
    assert "- Constructs to map: INTNX function" in block
    assert block.index("## Focus hints") < block.index(
        "## Relevant migration guidance"
    )
    assert (
        "### [functions · Funcs > INTNX Function · pp. 41-43 · construct: intnx]"
        in block
    )
    assert "advances a sas date" in block
    # The retrieval-only breadcrumb prefix is stripped from the body.
    assert "Funcs > INTNX Function\n\nadvances" not in block


def test_build_returns_none_when_nothing_relevant():
    pb = PromptBuilder(_corpus())
    assert pb.build("zzz totally unrelated gibberish", []) is None


def test_single_page_citation_format():
    key = ConstructKey(kind="function", name="a")
    pb = PromptBuilder(
        [_chunk("c0", "Funcs > A", "body text here", keys=[key], page_start=7, page_end=7)]
    )
    block = pb.build("anything", [key])  # deterministic construct hit
    assert "· p. 7 ·" in block


def test_pinned_section_appears_in_block():
    pb = PromptBuilder(_corpus(), pinned_sections=["Output Format"])
    block = pb.build("zzz nothing", [])
    assert block is not None
    assert "Output Format" in block


def test_max_instruction_words_caps_block():
    big = [
        _chunk("c0", "Funcs > A", " ".join(f"w{i}" for i in range(400)), keys=[INTNX]),
        _chunk(
            "c1",
            "Funcs > B",
            " ".join(f"x{i}" for i in range(400)),
            keys=[ConstructKey(kind="function", name="b")],
        ),
    ]
    pb = PromptBuilder(big, max_instruction_words=420)
    block = pb.build("w1 w2", [INTNX, ConstructKey(kind="function", name="b")])
    # Only the first ~400-word chunk fits the 420 budget.
    assert "Funcs > A" in block
    assert "Funcs > B" not in block


# ---------------------------------------------------------------------------
# User instructions — two-block output
# ---------------------------------------------------------------------------


def test_user_instructions_render_in_project_block_above_guidance():
    pb = PromptBuilder(
        _corpus(),
        user_instructions="## Output rules\nAlways emit a risk table.",
    )
    block = pb.build("zzz", [INTNX])
    assert block is not None
    assert block.startswith("## Project instructions")
    assert "### Output rules" in block
    assert "Always emit a risk table." in block
    guidance_at = block.index("## Relevant migration guidance")
    assert block.index("### Output rules") < guidance_at  # project block first
    assert "INTNX Function" in block[guidance_at:]
    # Operator rules cite no doc/pages; reference chunks still do.
    project_block = block[:guidance_at]
    assert "· p" not in project_block


def test_user_instructions_only_no_reference_corpus():
    pb = PromptBuilder([], user_instructions="Always target Delta Lake tables.")
    block = pb.build("anything at all", [])
    assert block is not None
    assert block.startswith("## Project instructions")
    assert "Always target Delta Lake tables." in block
    assert "## Relevant migration guidance" not in block


def test_user_instructions_str_is_parsed_and_fingerprinted():
    pb = PromptBuilder([], user_instructions="## A\nrule body")
    assert pb.user_instructions is not None
    assert len(pb.user_instructions) == 1
    assert len(pb.user_instructions.fingerprint) == 16


def test_with_user_instructions_rebuilds_over_same_corpus():
    original = PromptBuilder(_corpus(), user_instructions="## Old\nOLDMARK body.")
    rebuilt = original.with_user_instructions("## New\nNEWMARK body.")

    block = rebuilt.build("zzz", [INTNX])
    assert "NEWMARK" in block
    assert "OLDMARK" not in block
    assert "INTNX Function" in block  # reference corpus carried over
    # The original builder is untouched.
    assert "OLDMARK" in original.build("zzz", [])


def test_no_user_instructions_output_unchanged():
    pb = PromptBuilder(_corpus())
    block = pb.build("advance a date interval", [INTNX])
    assert block.startswith("## Focus hints")
    assert "## Relevant migration guidance" in block
    assert "## Project instructions" not in block


# ---------------------------------------------------------------------------
# Focus hints (directional stimulus block)
# ---------------------------------------------------------------------------

SYMPUT = ConstructKey(kind="call_routine", name="symput")


def _hazard_corpus():
    return _corpus() + [
        _chunk(
            "c2",
            "CALL Routines > SYMPUT Routine",
            "assigns a data step value to a macro variable",
            keys=[SYMPUT],
        ),
        _chunk(
            "c3",
            "Spark > DataFrames and SQL",
            "advance a date interval with dataframe expressions",
        ),
    ]


def test_hints_hazard_line_carries_caution():
    pb = PromptBuilder(_hazard_corpus())
    block = pb.build("zzz", [SYMPUT])
    assert "## Focus hints" in block
    assert (
        "- ⚠️ Hazards to address explicitly: CALL SYMPUT routine — "
        "run-time macro-variable write; scope/timing differs from %LET" in block
    )
    # A hazard construct is not repeated on the ordinary constructs line.
    assert "- Constructs to map:" not in block


def test_hints_hazard_listed_even_without_reference_match():
    # No SYMPUT section in the corpus: the construct lookup finds nothing,
    # but the item still deserves the hazard caution.
    pb = PromptBuilder(_corpus())
    block = pb.build("advance a date interval", [INTNX, SYMPUT])
    assert "⚠️ Hazards to address explicitly: CALL SYMPUT routine" in block


def test_hints_topics_line_from_topical_picks():
    pb = PromptBuilder(_hazard_corpus())
    block = pb.build("advance a date interval with dataframe", [])
    assert "- Related reference topics:" in block
    assert "DataFrames and SQL" in block.split("## Relevant migration guidance")[0]


def test_hints_render_between_project_and_reference_blocks():
    pb = PromptBuilder(
        _corpus(), user_instructions="## Output rules\nAlways emit a risk table."
    )
    block = pb.build("zzz", [INTNX])
    assert (
        block.index("## Project instructions")
        < block.index("## Focus hints")
        < block.index("## Relevant migration guidance")
    )


def test_hints_absent_when_nothing_to_hint():
    # Pinned-only selection: no hazards, no construct hits, no topical picks.
    pb = PromptBuilder(_corpus(), pinned_sections=["Output Format"])
    block = pb.build("zzz nothing", [])
    assert block is not None
    assert "## Focus hints" not in block


def test_focus_hints_flag_disables_block():
    pb = PromptBuilder(_corpus(), focus_hints=False)
    block = pb.build("advance a date interval", [INTNX])
    assert block.startswith("## Relevant migration guidance")
    assert "## Focus hints" not in block


def test_with_user_instructions_preserves_focus_hints_flag():
    pb = PromptBuilder(_corpus(), focus_hints=False)
    rebuilt = pb.with_user_instructions("## A\nrule body")
    block = rebuilt.build("zzz", [INTNX])
    assert "## Focus hints" not in block


def test_hints_construct_labels_by_kind():
    keys = [
        ConstructKey(kind="proc", name="sql"),
        ConstructKey(kind="component_object", name="hash"),
        ConstructKey(kind="macro_statement", name="let"),
    ]
    corpus = [
        _chunk("p0", "Procs > SQL Procedure", "ansi sql queries", keys=[keys[0]]),
        _chunk("p1", "Objects > Hash Object", "in-memory lookup", keys=[keys[1]]),
        _chunk("p2", "Macro > %LET Statement", "assigns macro vars", keys=[keys[2]]),
    ]
    block = PromptBuilder(corpus).build("zzz", keys)
    assert (
        "- Constructs to map: PROC SQL, hash object, %LET macro statement" in block
    )


# ---------------------------------------------------------------------------
# Selection-reason annotations on reference chunk headers
# ---------------------------------------------------------------------------


def test_reference_header_annotates_hazard_reason():
    pb = PromptBuilder(_hazard_corpus())
    block = pb.build("zzz", [SYMPUT])
    assert "· hazard: symput]" in block


def test_reference_header_annotates_topical_reason():
    pb = PromptBuilder(_hazard_corpus())
    block = pb.build("advance a date interval with dataframe", [])
    assert "· topical]" in block


def test_reference_header_annotates_pinned_reason():
    pb = PromptBuilder(_corpus(), pinned_sections=["Output Format"])
    block = pb.build("zzz nothing", [])
    assert "· pinned]" in block


def test_reason_annotations_distinguish_construct_from_topical():
    # INTNX arrives via construct lookup; the Spark chunk via ranking — the
    # same block must label the two provenances differently.
    pb = PromptBuilder(_hazard_corpus())
    block = pb.build("advance a date interval with dataframe", [INTNX])
    assert "INTNX Function · pp. 41-43 · construct: intnx]" in block
    assert "· topical]" in block


# ---------------------------------------------------------------------------
# from_specs (load + cache path)
# ---------------------------------------------------------------------------


def _write_funcs_pdf(path: pathlib.Path) -> str:
    doc = pymupdf.open()
    for lines in [
        [
            ("Dictionary of Functions", 16),
            ("INTNX Function", 16),
            ("Advances a SAS date by a number of intervals. " * 6, 11),
        ],
    ]:
        page = doc.new_page(width=612, height=792)
        y = 72.0
        for text, size in lines:
            page.insert_text((72, y), text, fontsize=size, fontname="helv")
            y += size * 1.6 + 6
    doc.set_toc([[1, "Dictionary of Functions", 1], [2, "INTNX Function", 1]])
    doc.save(str(path))
    doc.close()
    return str(path)


def test_from_specs_loads_and_builds(tmp_path):
    pdf = _write_funcs_pdf(tmp_path / "funcs.pdf")
    spec = DocumentSpec(path=pdf, doc_id="functions", section_level=2)
    pb = PromptBuilder.from_specs([spec], cache_dir=str(tmp_path / "cache"))
    block = pb.build("advance a date", [INTNX])
    assert block is not None
    assert "INTNX Function" in block
