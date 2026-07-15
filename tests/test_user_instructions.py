"""
Tests for prompt_builder.user_instructions — parsing operator-supplied text
into scoped instruction chunks: heading splitting, scope directives, graceful
degradation toward over-inclusion, fingerprinting, and file loading.

Fully offline: plain strings in, chunks out.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from prompt_builder.models import ConstructKey, DocRole, InstructionChunk
from prompt_builder.user_instructions import (
    SCOPE_ALWAYS,
    SCOPE_EXAMPLE,
    SCOPE_TOPIC,
    SCOPE_WHEN,
    UserInstructionSet,
    langs_of,
    normalize_language,
    scope_of,
)

RULES = """\
Always target Delta Lake tables, never pandas.

## Output format
Respond with one fenced PySpark block per SAS step, then a risk table.

## [when: proc:sql, component_object:hash] Lookup rules
Translate PROC SQL joins and hash-object lookups to broadcast joins
when the lookup side is small.

## [topic] Partitioning guidance
Wide fact tables are partitioned by load_date; repartition before window
functions over customer_id.
"""


# ---------------------------------------------------------------------------
# Splitting and scopes
# ---------------------------------------------------------------------------


def test_headingless_text_is_one_always_instruction():
    ins = UserInstructionSet.from_text("Never use pandas. Always use Delta writes.")
    assert len(ins) == 1
    chunk = ins.chunks[0]
    assert chunk.role is DocRole.USER_INSTRUCTION
    assert scope_of(chunk) == SCOPE_ALWAYS
    assert chunk.section_path == "General"
    assert "Never use pandas" in chunk.text
    assert (chunk.page_start, chunk.page_end) == (0, 0)


def test_sections_split_in_order_with_scopes():
    ins = UserInstructionSet.from_text(RULES)
    assert [c.section_path for c in ins.chunks] == [
        "General",  # preamble before the first heading
        "Output format",
        "Lookup rules",
        "Partitioning guidance",
    ]
    assert [scope_of(c) for c in ins.chunks] == [
        SCOPE_ALWAYS,
        SCOPE_ALWAYS,
        SCOPE_WHEN,
        SCOPE_TOPIC,
    ]
    assert ins.diagnostics == []


def test_scope_views_partition_the_chunks():
    ins = UserInstructionSet.from_text(RULES)
    assert [c.section_path for c in ins.always_chunks] == ["General", "Output format"]
    assert [c.section_path for c in ins.conditional_chunks] == ["Lookup rules"]
    assert [c.section_path for c in ins.topical_chunks] == ["Partitioning guidance"]


# ---------------------------------------------------------------------------
# [example] directive (few-shot worked pairs)
# ---------------------------------------------------------------------------


def test_example_directive_with_keys():
    ins = UserInstructionSet.from_text(
        "## [example: proc:sql, function:intnx] SQL join\n"
        "```sas\nproc sql; ...\n```\n```python\nspark.sql(...)\n```"
    )
    chunk = ins.chunks[0]
    assert scope_of(chunk) == SCOPE_EXAMPLE
    assert chunk.section_path == "SQL join"
    assert chunk.construct_keys == [
        ConstructKey(kind="proc", name="sql"),
        ConstructKey(kind="function", name="intnx"),
    ]
    assert ins.example_chunks == [chunk]
    assert ins.diagnostics == []


def test_bare_example_directive_is_unconditional():
    ins = UserInstructionSet.from_text("## [example] Canonical shape\nbody")
    chunk = ins.chunks[0]
    assert scope_of(chunk) == SCOPE_EXAMPLE
    assert chunk.construct_keys == []
    assert ins.diagnostics == []


def test_example_directive_with_no_valid_keys_degrades_to_unconditional():
    ins = UserInstructionSet.from_text("## [example: not a key] E\nbody")
    chunk = ins.chunks[0]
    # Stays an example (not an always-on rule) but loses the condition.
    assert scope_of(chunk) == SCOPE_EXAMPLE
    assert chunk.construct_keys == []
    assert [d.code for d in ins.diagnostics] == [
        "INVALID_CONSTRUCT_KEY",
        "INVALID_CONSTRUCT_KEY",
    ]  # one for the bad token, one for the empty result


def test_when_keys_parse_to_construct_keys():
    ins = UserInstructionSet.from_text(RULES)
    lookup = ins.conditional_chunks[0]
    assert lookup.construct_keys == [
        ConstructKey(kind="proc", name="sql"),
        ConstructKey(kind="component_object", name="hash"),
    ]


def test_when_keys_lowercased_and_deduped():
    ins = UserInstructionSet.from_text(
        "## [when: PROC:SQL, proc:sql, Function:INTNX] Rules\nbody text"
    )
    assert ins.conditional_chunks[0].construct_keys == [
        ConstructKey(kind="proc", name="sql"),
        ConstructKey(kind="function", name="intnx"),
    ]


def test_chunk_text_leads_with_title_and_ids_are_unique():
    ins = UserInstructionSet.from_text(RULES, doc_id="proj")
    assert all(c.doc_id == "proj" for c in ins.chunks)
    assert all(c.text.startswith(f"{c.section_path}\n\n") for c in ins.chunks)
    ids = [c.chunk_id for c in ins.chunks]
    assert len(ids) == len(set(ids))
    assert ids[0] == "proj::c0000"


# ---------------------------------------------------------------------------
# [lang: ...] directive and stacked bracket groups
# ---------------------------------------------------------------------------


def test_lang_directive_scopes_a_section():
    ins = UserInstructionSet.from_text(
        "## [lang: sparksql] Emit SQL\nTarget Spark SQL."
    )
    chunk = ins.chunks[0]
    assert scope_of(chunk) == SCOPE_ALWAYS  # lang is a modifier, not a scope
    assert chunk.section_path == "Emit SQL"
    assert langs_of(chunk) == ["sparksql"]
    assert ins.diagnostics == []


def test_lang_directive_normalizes_and_dedupes():
    ins = UserInstructionSet.from_text(
        "## [lang: Spark SQL, spark_sql, PySpark] Rule\nbody"
    )
    assert langs_of(ins.chunks[0]) == ["sparksql", "pyspark"]


def test_agnostic_section_has_no_lang():
    ins = UserInstructionSet.from_text("## Output format\nbody")
    assert langs_of(ins.chunks[0]) == []


def test_stacked_when_and_lang_groups_combine():
    ins = UserInstructionSet.from_text(
        "## [when: proc:sql] [lang: sparksql] SQL rules\nbody"
    )
    chunk = ins.chunks[0]
    assert scope_of(chunk) == SCOPE_WHEN
    assert chunk.construct_keys == [ConstructKey(kind="proc", name="sql")]
    assert langs_of(chunk) == ["sparksql"]
    assert chunk.section_path == "SQL rules"
    assert ins.diagnostics == []


def test_stacked_lang_before_topic():
    ins = UserInstructionSet.from_text(
        "## [lang: sparksql] [topic] Partitioning\nbody"
    )
    chunk = ins.chunks[0]
    assert scope_of(chunk) == SCOPE_TOPIC
    assert langs_of(chunk) == ["sparksql"]
    assert chunk.section_path == "Partitioning"


def test_normalize_language_helper():
    assert (
        normalize_language("SparkSQL")
        == normalize_language("Spark SQL")
        == normalize_language("spark_sql")
        == "sparksql"
    )


# ---------------------------------------------------------------------------
# Graceful degradation — diagnostics, never exceptions, over-inclusion
# ---------------------------------------------------------------------------


def test_unknown_directive_degrades_to_always_with_diagnostic():
    ins = UserInstructionSet.from_text("## [whenever: x] Odd rule\nbody")
    assert scope_of(ins.chunks[0]) == SCOPE_ALWAYS
    assert ins.chunks[0].section_path == "Odd rule"
    assert [d.code for d in ins.diagnostics] == ["UNKNOWN_DIRECTIVE"]


def test_malformed_when_key_is_dropped_with_diagnostic():
    ins = UserInstructionSet.from_text(
        "## [when: proc:sql, notakey] Rules\nbody text"
    )
    chunk = ins.chunks[0]
    assert scope_of(chunk) == SCOPE_WHEN  # the valid key survives
    assert chunk.construct_keys == [ConstructKey(kind="proc", name="sql")]
    assert [d.code for d in ins.diagnostics] == ["INVALID_CONSTRUCT_KEY"]


def test_when_with_no_valid_keys_becomes_always():
    ins = UserInstructionSet.from_text("## [when: nocolon] Rules\nbody text")
    assert scope_of(ins.chunks[0]) == SCOPE_ALWAYS  # over-include, don't drop
    codes = [d.code for d in ins.diagnostics]
    assert codes.count("INVALID_CONSTRUCT_KEY") == 2  # bad key + empty fallback


def test_empty_body_section_is_skipped_with_diagnostic():
    ins = UserInstructionSet.from_text("## Rule one\ncontent\n## Rule two\n\n")
    assert [c.section_path for c in ins.chunks] == ["Rule one"]
    assert [d.code for d in ins.diagnostics] == ["EMPTY_INSTRUCTION"]


def test_blank_input_yields_empty_set():
    ins = UserInstructionSet.from_text("   \n\n  ")
    assert len(ins) == 0
    assert ins.diagnostics == []


# ---------------------------------------------------------------------------
# Fingerprint and file loading
# ---------------------------------------------------------------------------


def test_fingerprint_stable_and_content_sensitive():
    a1 = UserInstructionSet.from_text(RULES)
    a2 = UserInstructionSet.from_text(RULES)
    b = UserInstructionSet.from_text(RULES + "\n## Extra\nrule")
    assert a1.fingerprint == a2.fingerprint
    assert a1.fingerprint != b.fingerprint
    assert len(a1.fingerprint) == 16


def test_from_file_reads_and_records_source(tmp_path):
    path = tmp_path / "instructions.md"
    path.write_text(RULES, encoding="utf-8")
    ins = UserInstructionSet.from_file(str(path))
    assert ins.source == str(path)
    assert len(ins) == 4
    assert ins.fingerprint == UserInstructionSet.from_text(RULES).fingerprint


# ---------------------------------------------------------------------------
# from_dir — merged markdown files, subdirectory names the target language
# ---------------------------------------------------------------------------


def _write_instruction_tree(root):
    (root / "sparksql").mkdir(parents=True)
    (root / "pyspark").mkdir(parents=True)
    (root / "_common").mkdir(parents=True)
    (root / "sparksql" / "sql.md").write_text(
        "## SQL rule\nEmit Spark SQL.", encoding="utf-8"
    )
    (root / "pyspark" / "df.md").write_text(
        "## DF rule\nEmit DataFrames.", encoding="utf-8"
    )
    (root / "_common" / "format.md").write_text(
        "## Format\nStructured markdown.", encoding="utf-8"
    )
    (root / "top.md").write_text("## Top rule\nAlways.", encoding="utf-8")


def test_from_dir_scopes_by_subdirectory_language(tmp_path):
    _write_instruction_tree(tmp_path)
    ins = UserInstructionSet.from_dir(str(tmp_path))
    by_title = {c.section_path: c for c in ins.chunks}
    assert langs_of(by_title["SQL rule"]) == ["sparksql"]
    assert langs_of(by_title["DF rule"]) == ["pyspark"]
    assert langs_of(by_title["Format"]) == []  # _common is agnostic
    assert langs_of(by_title["Top rule"]) == []  # root file is agnostic


def test_from_dir_merges_all_files_with_unique_ids(tmp_path):
    _write_instruction_tree(tmp_path)
    ins = UserInstructionSet.from_dir(str(tmp_path))
    assert len(ins) == 4
    ids = [c.chunk_id for c in ins.chunks]
    assert len(ids) == len(set(ids))
    assert ins.source == str(tmp_path)


def test_from_dir_explicit_lang_overrides_subdirectory(tmp_path):
    (tmp_path / "sparksql").mkdir()
    (tmp_path / "sparksql" / "x.md").write_text(
        "## Odd\n## [lang: pyspark] Override\nbody", encoding="utf-8"
    )
    ins = UserInstructionSet.from_dir(str(tmp_path))
    override = next(c for c in ins.chunks if c.section_path == "Override")
    assert langs_of(override) == ["pyspark"]


def test_from_dir_fingerprint_stable_and_content_sensitive(tmp_path):
    _write_instruction_tree(tmp_path)
    a = UserInstructionSet.from_dir(str(tmp_path))
    b = UserInstructionSet.from_dir(str(tmp_path))
    assert a.fingerprint == b.fingerprint and len(a.fingerprint) == 16
    (tmp_path / "sparksql" / "sql.md").write_text(
        "## SQL rule\nEmit Spark SQL, always.", encoding="utf-8"
    )
    c = UserInstructionSet.from_dir(str(tmp_path))
    assert c.fingerprint != a.fingerprint


def test_from_dir_missing_directory_returns_empty(tmp_path, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="prompt_builder.user_instructions"):
        ins = UserInstructionSet.from_dir(str(tmp_path / "nope"))
    assert len(ins) == 0
    assert "not a directory" in caplog.text


# ---------------------------------------------------------------------------
# from_config — the standing instructions file
# ---------------------------------------------------------------------------


def _isolated_config(monkeypatch, tmp_path, mapping) -> None:
    """Point app_config at a tmp file; each caller clear_cache()s in finally."""
    import json

    import app_config

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(mapping), encoding="utf-8")
    monkeypatch.setenv(app_config.ENV_VAR, str(cfg))
    app_config.clear_cache()


def test_from_config_unset_returns_none(monkeypatch, tmp_path):
    import app_config

    _isolated_config(monkeypatch, tmp_path, {})
    try:
        assert UserInstructionSet.from_config() is None
    finally:
        app_config.clear_cache()


def test_from_config_loads_configured_file(monkeypatch, tmp_path):
    import app_config

    rules_path = tmp_path / "instructions.md"
    rules_path.write_text(RULES, encoding="utf-8")
    _isolated_config(
        monkeypatch, tmp_path, {"user_instructions": {"path": str(rules_path)}}
    )
    try:
        ins = UserInstructionSet.from_config()
        assert ins is not None
        assert ins.source == str(rules_path)
        assert ins.fingerprint == UserInstructionSet.from_text(RULES).fingerprint
    finally:
        app_config.clear_cache()


def test_from_config_loads_configured_dir(monkeypatch, tmp_path):
    import app_config

    instr = tmp_path / "instructions"
    (instr / "sparksql").mkdir(parents=True)
    (instr / "sparksql" / "sql.md").write_text(
        "## SQL rule\nEmit Spark SQL.", encoding="utf-8"
    )
    _isolated_config(
        monkeypatch, tmp_path, {"user_instructions": {"dir": str(instr)}}
    )
    try:
        ins = UserInstructionSet.from_config()
        assert ins is not None
        assert [langs_of(c) for c in ins.chunks] == [["sparksql"]]
    finally:
        app_config.clear_cache()


def test_from_config_dir_takes_precedence_over_path(monkeypatch, tmp_path):
    import app_config

    instr = tmp_path / "instructions"
    (instr / "sparksql").mkdir(parents=True)
    (instr / "sparksql" / "sql.md").write_text(
        "## SQL rule\nEmit Spark SQL.", encoding="utf-8"
    )
    path_file = tmp_path / "single.md"
    path_file.write_text("## Single\nbody", encoding="utf-8")
    _isolated_config(
        monkeypatch,
        tmp_path,
        {"user_instructions": {"dir": str(instr), "path": str(path_file)}},
    )
    try:
        ins = UserInstructionSet.from_config()
        assert ins is not None
        assert [c.section_path for c in ins.chunks] == ["SQL rule"]
    finally:
        app_config.clear_cache()


def test_from_config_missing_dir_warns_and_returns_none(
    monkeypatch, tmp_path, caplog
):
    import logging

    import app_config

    _isolated_config(
        monkeypatch,
        tmp_path,
        {"user_instructions": {"dir": str(tmp_path / "gone")}},
    )
    try:
        with caplog.at_level(
            logging.WARNING, logger="prompt_builder.user_instructions"
        ):
            assert UserInstructionSet.from_config() is None
        assert "not found" in caplog.text
    finally:
        app_config.clear_cache()


def test_from_config_missing_file_warns_and_returns_none(
    monkeypatch, tmp_path, caplog
):
    import logging

    import app_config

    _isolated_config(
        monkeypatch,
        tmp_path,
        {"user_instructions": {"path": str(tmp_path / "gone.md")}},
    )
    try:
        with caplog.at_level(
            logging.WARNING, logger="prompt_builder.user_instructions"
        ):
            assert UserInstructionSet.from_config() is None
        assert "not found" in caplog.text
    finally:
        app_config.clear_cache()


# ---------------------------------------------------------------------------
# Interop — scope survives the LangChain Document round-trip
# ---------------------------------------------------------------------------


def test_scope_survives_document_round_trip():
    ins = UserInstructionSet.from_text(RULES)
    for original in ins.chunks:
        rebuilt = InstructionChunk.from_document(original.to_document())
        assert rebuilt.model_dump() == original.model_dump()
        assert scope_of(rebuilt) == scope_of(original)
