"""
Tests for the app_config loader and its wiring into the word/token-limit
consumers (SasSemanticChunker, InstructionChunker, PromptBuilder,
LLMClientConfig).

Each test that changes the environment points SAS_PARSER_CONFIG at a tmp file
and clears the process cache around itself, so the repo's own config.json
never leaks in or out.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

import app_config


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch, tmp_path):
    """Every test starts with no config found (empty file via env override)."""
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(app_config.ENV_VAR, str(cfg))
    app_config.clear_cache()
    yield cfg
    app_config.clear_cache()


def _set(cfg_path, mapping) -> None:
    cfg_path.write_text(json.dumps(mapping), encoding="utf-8")
    app_config.clear_cache()


# ---------------------------------------------------------------------------
# Loader semantics
# ---------------------------------------------------------------------------


def test_missing_file_yields_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv(app_config.ENV_VAR, str(tmp_path / "nope.json"))
    monkeypatch.chdir(tmp_path)  # no cwd config.json either
    app_config.clear_cache()
    assert app_config.get_value("sas_chunker", "min_words", 300) == 300


def test_value_read_from_file(_isolated_config):
    _set(_isolated_config, {"sas_chunker": {"min_words": 42}})
    assert app_config.get_value("sas_chunker", "min_words", 300) == 42


def test_null_means_unset(_isolated_config):
    _set(_isolated_config, {"sas_chunker": {"min_words": None}})
    assert app_config.get_value("sas_chunker", "min_words", 300) == 300


def test_resolve_precedence_explicit_beats_config(_isolated_config):
    _set(_isolated_config, {"sas_chunker": {"min_words": 42}})
    assert app_config.resolve(7, "sas_chunker", "min_words", 300) == 7
    assert app_config.resolve(None, "sas_chunker", "min_words", 300) == 42


def test_unreadable_file_is_skipped(monkeypatch, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv(app_config.ENV_VAR, str(bad))
    monkeypatch.chdir(tmp_path)
    app_config.clear_cache()
    assert app_config.get_value("x", "y", "fallback") == "fallback"


def test_repo_config_json_matches_code_defaults():
    """The shipped template must be a no-op: every value == the hard default."""
    repo_cfg = json.loads(
        (pathlib.Path(__file__).resolve().parents[1] / "config.json").read_text(
            encoding="utf-8"
        )
    )
    assert repo_cfg["sas_chunker"] == {"min_words": 300, "max_words": 700}
    assert repo_cfg["instruction_chunker"] == {
        "min_words": 120,
        "max_words": 900,
        "overlap_words": 60,
    }
    assert repo_cfg["prompt_builder"] == {
        "top_k": 6,
        "max_instruction_words": 1500,
        "focus_hints": None,  # null = unset -> code default (True)
    }
    assert repo_cfg["llm_client"] == {
        "max_input_tokens": None,
        "max_output_tokens": None,
    }
    assert repo_cfg["user_instructions"] == {"path": None, "max_words": None}


# ---------------------------------------------------------------------------
# Consumer wiring — config value applies, explicit argument wins
# ---------------------------------------------------------------------------


def test_sas_chunker_reads_config(_isolated_config):
    from chunker.chunker import SasSemanticChunker

    _set(_isolated_config, {"sas_chunker": {"min_words": 111, "max_words": 222}})
    chunker = SasSemanticChunker()
    assert (chunker.min_words, chunker.max_words) == (111, 222)
    explicit = SasSemanticChunker(min_words=5, max_words=10)
    assert (explicit.min_words, explicit.max_words) == (5, 10)


def test_instruction_chunker_reads_config(_isolated_config):
    from prompt_builder.doc_chunker import InstructionChunker

    _set(
        _isolated_config,
        {"instruction_chunker": {"min_words": 11, "max_words": 33, "overlap_words": 2}},
    )
    chunker = InstructionChunker()
    assert (chunker.min_words, chunker.max_words, chunker.overlap_words) == (11, 33, 2)
    assert InstructionChunker(max_words=99).max_words == 99


def test_prompt_builder_reads_config(_isolated_config):
    from prompt_builder.builder import PromptBuilder

    _set(
        _isolated_config,
        {"prompt_builder": {"top_k": 2, "max_instruction_words": 77}},
    )
    builder = PromptBuilder([])
    assert (builder.top_k, builder.max_instruction_words) == (2, 77)
    assert PromptBuilder([], top_k=9).top_k == 9


def test_llm_client_config_reads_config(_isolated_config):
    from llm_client import LLMClientConfig

    _set(
        _isolated_config,
        {"llm_client": {"max_input_tokens": 123456, "max_output_tokens": 4096}},
    )
    cfg = LLMClientConfig()
    assert cfg.max_input_tokens == 123456
    assert cfg.max_output_tokens == 4096
    # Explicit None still means "disabled", overriding the config value.
    assert LLMClientConfig(max_input_tokens=None).max_input_tokens is None


def test_defaults_without_config(_isolated_config):
    from chunker.chunker import SasSemanticChunker
    from llm_client import LLMClientConfig
    from prompt_builder.builder import PromptBuilder
    from prompt_builder.doc_chunker import InstructionChunker

    assert SasSemanticChunker().min_words == 300
    assert InstructionChunker().max_words == 900
    assert PromptBuilder([]).max_instruction_words == 1500
    assert LLMClientConfig().max_input_tokens is None
