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
import logging
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


def test_bom_prefixed_file_loads(_isolated_config):
    # Windows editors and PowerShell 5.1 commonly prepend a UTF-8 BOM, which
    # the loader must tolerate (utf-8-sig) instead of skipping the file.
    _isolated_config.write_bytes(
        b"\xef\xbb\xbf" + json.dumps({"sas_chunker": {"min_words": 42}}).encode("utf-8")
    )
    app_config.clear_cache()
    assert app_config.get_value("sas_chunker", "min_words", 300) == 42


def test_resolve_path_anchors_relative_values_to_the_repo_root(tmp_path, monkeypatch):
    """A relative path *inside* config.json must survive the cwd.

    config.json itself is found relative to the repo root even when the
    process starts elsewhere, so a value like 'prompt_builder/instructions'
    resolving only against cwd would silently miss for a Docker entrypoint or
    a scheduled job.
    """
    repo_root = pathlib.Path(app_config.__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    assert app_config.resolve_path("prompt_builder/instructions").is_dir()
    assert (
        app_config.resolve_path("prompt_builder/instructions")
        == repo_root / "prompt_builder/instructions"
    )


def test_resolve_path_prefers_cwd_and_honours_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "local").mkdir()
    # Exists relative to cwd -> taken as given, not rewritten to the repo root.
    assert app_config.resolve_path("local") == pathlib.Path("local")
    absolute = tmp_path / "elsewhere"
    assert app_config.resolve_path(str(absolute)) == absolute


def test_repo_config_json_matches_code_defaults():
    """The shipped file is a no-op except where it deliberately is not.

    Every value must equal the hard default, so a fresh checkout behaves as
    though no config existed — with one documented exception, the bundled
    instruction set and the budget sized for it (asserted explicitly below).
    """
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
    # Sized for the bundled instruction set enabled below, not the code
    # default: at 1500 a dependency batch received 2 of the 7 rules its
    # constructs matched, silently dropping silent-error guidance.
    assert repo_cfg["prompt_builder"] == {
        "top_k": 6,
        "max_instruction_words": 4000,
        "focus_hints": None,  # null = unset -> code default (True)
        "reasoning_directives": None,  # null = unset -> code default (True)
    }
    assert repo_cfg["llm_client"] == {
        "model": None,
        "model_provider": None,
        "gateway_version": None,  # null = unset -> no ai-gateway-version header
        "provider_client": None,  # null = unset -> code default (ChatOpenAI)
        "base_url": None,
        "url_headers": None,
        "timeout": None,
        "cert_file": None,
        "temperature": None,
        "max_retries": None,
        "model_kwargs": None,
        "max_input_tokens": None,
        "max_output_tokens": None,
        "prompt_caching": None,  # null = unset -> code default (False)
        "requests_per_second": None,  # null = unset -> code default (2.0)
        "max_bucket_size": None,  # null = unset -> code default (1)
        # Sparse overlays: an all-null role changes nothing, so the template
        # stays a no-op while showing the shape.
        "roles": {
            "validator": {"timeout": None, "model": None},
            "complexity": {"timeout": None, "model": None},
        },
    }
    # The one section that ships non-null: the bundled SparkSQL instruction
    # set is on by default. It is scoped [lang: sparksql], so it is inert
    # under any other output_language, and `max_words` is set with it so
    # operator rules cannot consume the whole retrieval budget and starve the
    # reference corpus.
    assert repo_cfg["user_instructions"] == {
        "path": None,
        "dir": "prompt_builder/instructions",
        "max_words": 2800,
    }
    assert (
        pathlib.Path(__file__).resolve().parents[1]
        / repo_cfg["user_instructions"]["dir"]
    ).is_dir(), "config.json points user_instructions.dir at a missing directory"
    # Operator rules must leave room for the reference corpus they sit above.
    assert (
        repo_cfg["user_instructions"]["max_words"]
        < repo_cfg["prompt_builder"]["max_instruction_words"]
    ), "user_instructions.max_words must leave budget for reference chunks"
    # Every section the refactor added ships all-null too, so a fresh checkout
    # behaves exactly as it did before any of them are filled in.
    for section in ("vault", "azure", "databricks", "sharepoint", "xref",
                    "adls", "sftp", "sas"):
        assert all(value is None for value in repo_cfg[section].values()), section
    assert "powerapps" not in repo_cfg


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


def test_get_typed_value_wrong_type_falls_back_with_warning(
    _isolated_config, caplog
):
    _set(_isolated_config, {"llm_client": {"timeout": "sixty"}})
    with caplog.at_level(logging.WARNING, logger="app_config"):
        assert app_config.get_typed_value("llm_client", "timeout", (int, float)) is None
    assert "timeout" in caplog.text
    assert "sixty" in caplog.text


def test_get_typed_value_bool_is_not_a_number(_isolated_config):
    # JSON true/false must not satisfy an int/float expectation.
    _set(_isolated_config, {"llm_client": {"max_retries": True}})
    assert app_config.get_typed_value("llm_client", "max_retries", int, 3) == 3


def test_llm_client_value_checks_url_header_values(_isolated_config):
    _set(_isolated_config, {"llm_client": {"url_headers": {"X-Team": 1}}})
    assert app_config.llm_client_value("url_headers") is None
    _set(_isolated_config, {"llm_client": {"url_headers": {"X-Team": "sas"}}})
    assert app_config.llm_client_value("url_headers") == {"X-Team": "sas"}


def test_llm_client_model_accepts_accessible_variants(_isolated_config):
    # Bare IDs, provider prefixes, and dated snapshots of accessible models
    # all resolve; the allowlist spans every provider we can reach.
    for value in (
        "claude-sonnet-4-5",
        "anthropic:claude-opus-4-6",
        "claude-sonnet-4-5-20250929",
        "openai:gpt-5.4",
        "gemini-3.1-pro",
    ):
        _set(_isolated_config, {"llm_client": {"model": value}})
        assert app_config.llm_client_value("model") == value


def test_llm_client_model_rejects_inaccessible_with_warning(
    _isolated_config, caplog
):
    _set(_isolated_config, {"llm_client": {"model": "claude-2.1"}})
    with caplog.at_level(logging.WARNING, logger="app_config"):
        assert (
            app_config.llm_client_value("model", "fallback-model")
            == "fallback-model"
        )
    assert "claude-2.1" in caplog.text
    assert "not an accessible model" in caplog.text


def test_llm_client_provider_client_rejects_unknown_strategy(
    _isolated_config, caplog
):
    _set(_isolated_config, {"llm_client": {"provider_client": "carrier-pigeon"}})
    with caplog.at_level(logging.WARNING, logger="app_config"):
        assert app_config.llm_client_value("provider_client") is None
    assert "carrier-pigeon" in caplog.text


def test_llm_client_new_gateway_keys_read_config(_isolated_config):
    _set(
        _isolated_config,
        {
            "llm_client": {
                "gateway_version": "v2",
                "model_provider": "anthropic",
                "provider_client": "native",
            }
        },
    )
    assert app_config.llm_client_value("gateway_version") == "v2"
    assert app_config.llm_client_value("model_provider") == "anthropic"
    assert app_config.llm_client_value("provider_client") == "native"


def test_llm_client_value_rejects_unknown_key():
    # api_key is deliberately outside the schema: secrets never come from
    # config.json, and any other unknown key is a programming error.
    with pytest.raises(KeyError):
        app_config.llm_client_value("api_key")


def test_malformed_llm_client_section_degrades_gracefully(_isolated_config):
    from llm_client import LLMClientConfig

    _set(
        _isolated_config,
        {
            "llm_client": {
                "model": 123,
                "timeout": "sixty",
                "temperature": "warm",
                "url_headers": ["not", "a", "mapping"],
                "max_input_tokens": "lots",
            }
        },
    )
    cfg = LLMClientConfig()  # must not raise: bad entries -> hard defaults
    assert cfg.model == "gpt-5.4"
    assert cfg.timeout is None
    assert cfg.temperature is None
    assert cfg.url_headers is None
    assert cfg.max_input_tokens is None


def test_llm_client_endpoint_knobs_read_config(_isolated_config):
    from llm_client import LLMClientConfig

    _set(
        _isolated_config,
        {
            "llm_client": {
                "model": "claude-opus-4-6",
                "base_url": "https://gateway.example/v1",
                "url_headers": {"X-Team": "sas"},
                "timeout": 30,
                "cert_file": "certs/gateway.crt",
                "temperature": 0.4,
                "max_retries": 7,
                "model_kwargs": {"top_k": 40},
            }
        },
    )
    cfg = LLMClientConfig()
    assert cfg.model == "claude-opus-4-6"
    assert cfg.base_url == "https://gateway.example/v1"
    assert cfg.url_headers == {"X-Team": "sas"}
    assert cfg.timeout == 30
    assert cfg.cert_file == "certs/gateway.crt"
    assert cfg.temperature == 0.4
    assert cfg.max_retries == 7
    assert cfg.model_kwargs == {"top_k": 40}
    # Explicit argument still beats the config value.
    assert LLMClientConfig(model="explicit").model == "explicit"
    assert LLMClientConfig(timeout=5.0).timeout == 5.0


# ---------------------------------------------------------------------------
# Per-role llm_client overlays (role_value)
# ---------------------------------------------------------------------------


_ROLES_CONFIG = {
    "llm_client": {
        "timeout": 6000,
        "model": "claude-sonnet-4-5",
        "max_retries": 4,
        "roles": {
            "validator": {"timeout": 12000},
            "complexity": {"model": "claude-opus-4-6"},
        },
    }
}


def test_role_value_overlay_beats_base(_isolated_config):
    _set(_isolated_config, _ROLES_CONFIG)
    assert app_config.role_value("validator", "timeout") == 12000
    assert app_config.role_value("complexity", "model") == "claude-opus-4-6"


def test_role_value_falls_through_to_base_then_default(_isolated_config):
    _set(_isolated_config, _ROLES_CONFIG)
    # The role does not mention these, so the base section stands...
    assert app_config.role_value("validator", "model") == "claude-sonnet-4-5"
    assert app_config.role_value("complexity", "timeout") == 6000
    # ...and a key neither one sets falls through to the hard default.
    assert app_config.role_value("validator", "max_bucket_size", 1) == 1


def test_role_value_unknown_role_uses_base(_isolated_config):
    _set(_isolated_config, _ROLES_CONFIG)
    assert app_config.role_value("no-such-role", "timeout") == 6000
    assert app_config.role_value(None, "timeout") == 6000


def test_role_value_without_roles_section(_isolated_config):
    _set(_isolated_config, {"llm_client": {"timeout": 30}})
    assert app_config.role_value("validator", "timeout") == 30


def test_role_value_wrong_typed_overlay_degrades_to_base(_isolated_config, caplog):
    _set(
        _isolated_config,
        {
            "llm_client": {
                "timeout": 6000,
                "roles": {"validator": {"timeout": "twelve thousand"}},
            }
        },
    )
    with caplog.at_level(logging.WARNING, logger="app_config"):
        assert app_config.role_value("validator", "timeout") == 6000
    assert "llm_client.roles.validator.timeout" in caplog.text


def test_role_value_non_object_role_degrades_to_base(_isolated_config, caplog):
    _set(
        _isolated_config,
        {"llm_client": {"timeout": 6000, "roles": {"validator": "12000"}}},
    )
    with caplog.at_level(logging.WARNING, logger="app_config"):
        assert app_config.role_value("validator", "timeout") == 6000
    assert "llm_client.roles.validator" in caplog.text


def test_role_value_null_overlay_entry_is_unset(_isolated_config):
    # A template listing every role key as null must change nothing.
    _set(
        _isolated_config,
        {"llm_client": {"timeout": 6000, "roles": {"validator": {"timeout": None}}}},
    )
    assert app_config.role_value("validator", "timeout") == 6000


def test_role_value_overlay_model_must_be_accessible(_isolated_config, caplog):
    _set(
        _isolated_config,
        {
            "llm_client": {
                "model": "claude-sonnet-4-5",
                "roles": {"validator": {"model": "claude-2.1"}},
            }
        },
    )
    with caplog.at_level(logging.WARNING, logger="app_config"):
        assert app_config.role_value("validator", "model") == "claude-sonnet-4-5"
    assert "not an accessible model" in caplog.text


def test_role_value_cannot_overlay_roles_itself(_isolated_config):
    _set(
        _isolated_config,
        {"llm_client": {"roles": {"validator": {"roles": {"nested": {}}}}}},
    )
    assert app_config.role_value("validator", "roles") == {
        "validator": {"roles": {"nested": {}}}
    }


def test_defaults_without_config(_isolated_config):
    from chunker.chunker import SasSemanticChunker
    from llm_client import LLMClientConfig
    from prompt_builder.builder import PromptBuilder
    from prompt_builder.doc_chunker import InstructionChunker

    assert SasSemanticChunker().min_words == 300
    assert InstructionChunker().max_words == 900
    assert PromptBuilder([]).max_instruction_words == 1500
    assert LLMClientConfig().max_input_tokens is None
    assert LLMClientConfig().model == "gpt-5.4"
    assert LLMClientConfig().max_retries == 3


# ---------------------------------------------------------------------------
# Spark master resolution (app_config.spark)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_spark_master_env(monkeypatch):
    """A real SPARK_MASTER_URL (the Docker stack sets one) must not leak in."""
    monkeypatch.delenv("SPARK_MASTER_URL", raising=False)


def test_spark_master_defaults_to_local(_isolated_config):
    from app_config.spark import DEFAULT_MASTER, master_url

    assert master_url() == DEFAULT_MASTER == "local[*]"


def test_spark_master_read_from_config(_isolated_config):
    from app_config.spark import master_url

    _set(_isolated_config, {"spark": {"master": "spark://from-file:7077"}})
    assert master_url() == "spark://from-file:7077"


def test_spark_master_env_beats_config(_isolated_config, monkeypatch):
    from app_config.spark import master_url

    _set(_isolated_config, {"spark": {"master": "spark://from-file:7077"}})
    monkeypatch.setenv("SPARK_MASTER_URL", "spark://from-env:7077")
    assert master_url() == "spark://from-env:7077"


def test_spark_master_explicit_beats_everything(_isolated_config, monkeypatch):
    from app_config.spark import master_url

    _set(_isolated_config, {"spark": {"master": "spark://from-file:7077"}})
    monkeypatch.setenv("SPARK_MASTER_URL", "spark://from-env:7077")
    assert master_url("local[2]") == "local[2]"


def test_spark_master_blank_env_is_unset(_isolated_config, monkeypatch):
    """An empty value in a .env file means "not configured", not "no master"."""
    from app_config.spark import master_url

    monkeypatch.setenv("SPARK_MASTER_URL", "   ")
    assert master_url() == "local[*]"


def test_spark_master_wrong_type_degrades(_isolated_config, caplog):
    """A non-string entry degrades to the default with a WARNING, as elsewhere."""
    from app_config.spark import master_url

    _set(_isolated_config, {"spark": {"master": 7077}})
    with caplog.at_level(logging.WARNING):
        assert master_url() == "local[*]"
    assert "spark.master" in caplog.text


def test_spark_module_does_not_import_pyspark():
    """app_config stays the dependency-free leaf (Architecture.md invariant 8)."""
    import subprocess

    code = (
        "import sys, app_config.spark; "
        "sys.exit(1 if 'pyspark' in sys.modules else 0)"
    )
    completed = subprocess.run([sys.executable, "-c", code], check=False)
    assert completed.returncode == 0, "importing app_config.spark pulled in pyspark"


def test_utc_stamp_is_path_safe_and_sortable():
    import re

    stamp = app_config.utc_stamp()

    # No ':' — not a filename character on Windows, percent-encoded in a
    # SharePoint URL — and lexically sortable, which is what makes a listing
    # of run folders read chronologically.
    assert re.fullmatch(r"\d{8}T\d{6}Z", stamp)
    assert ":" not in stamp


def test_utc_stamp_uses_the_shared_format():
    from datetime import datetime, timezone

    moment = datetime(2026, 8, 6, 12, 0, 0, tzinfo=timezone.utc)

    assert f"{moment:{app_config.UTC_STAMP_FORMAT}}" == "20260806T120000Z"


def test_the_two_clis_stamp_run_folders_identically():
    """A folder named differently by the tool that wrote it and the tool that
    lists it is a folder nobody finds."""
    import complexity.__main__ as complexity_main
    from conversion import run as conv_run

    assert conv_run.utc_stamp is app_config.utc_stamp
    assert len(complexity_main._timestamp()) == len(app_config.utc_stamp())
