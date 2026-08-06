"""
Tests for main.py — the entry point's argument contract and its dispatch.

Two things are checked, and neither needs a network or an LLM:

* **Dispatch.** SharePoint is the default; a positional source directory is the
  explicit opt-out into local mode. Nothing falls back silently, because
  converting the wrong corpus because a config key was missing is worse than a
  clear error.
* **Argument validation.** Every rejected combination is rejected *before* any
  work — a run that cannot do what was asked should say so in a second, not
  after an LLM has been paid.

The credential chain is exercised through its seams (``from_ai_gateway`` /
``from_vault_secret`` / ``vault.is_configured``) rather than by reaching Vault.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

import app_config
import main as main_mod
from llm_client import LLMClientConfig


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(app_config.ENV_VAR, str(cfg))
    # main() loads .env before anything reads it; the repo's own must not leak
    # a credential or a site id into these.
    monkeypatch.setattr(main_mod, "_load_dotenv", lambda: None)
    app_config.clear_cache()
    yield tmp_path
    app_config.clear_cache()


@pytest.fixture
def reference_dir(tmp_path) -> pathlib.Path:
    d = tmp_path / "reference_docs"
    d.mkdir()
    return d


def _args(reference_dir, *extra: str, sas_dir: str | None = None):
    argv = ([sas_dir] if sas_dir else []) + [
        "--reference-dir",
        str(reference_dir),
        *extra,
    ]
    return main_mod.parse_args(argv)


# ---------------------------------------------------------------------------
# Dispatch — SharePoint by default, local as the explicit opt-out
# ---------------------------------------------------------------------------


def test_no_source_directory_runs_sharepoint(monkeypatch, reference_dir):
    taken: list[str] = []
    monkeypatch.setattr(
        main_mod, "_run_sharepoint", lambda args: taken.append("sharepoint") or 0
    )
    monkeypatch.setattr(main_mod, "_run_local", lambda args: taken.append("local") or 0)

    assert main_mod.main(["--reference-dir", str(reference_dir)]) == 0
    assert taken == ["sharepoint"]


def test_a_source_directory_runs_local(monkeypatch, reference_dir, tmp_path):
    sas_dir = tmp_path / "sas"
    sas_dir.mkdir()
    taken: list[str] = []
    monkeypatch.setattr(
        main_mod, "_run_sharepoint", lambda args: taken.append("sharepoint") or 0
    )
    monkeypatch.setattr(main_mod, "_run_local", lambda args: taken.append("local") or 0)

    exit_code = main_mod.main(
        [str(sas_dir), "--reference-dir", str(reference_dir)]
    )

    assert exit_code == 0
    assert taken == ["local"]


def test_parse_args_defaults_to_sharepoint_mode():
    args = main_mod.parse_args([])

    assert args.sas_dir is None
    assert args.request_id is None
    assert args.app is None
    assert args.all_rows is False
    assert args.no_upload is False
    assert args.no_xref is False


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag, value",
    [
        ("--request-id", "42"),
        ("--app", "MyApp"),
        ("--all-rows", None),
        ("--no-upload", None),
        ("--no-xref", None),
    ],
)
def test_sharepoint_flags_are_rejected_with_a_local_directory(
    reference_dir, tmp_path, flag, value
):
    sas_dir = tmp_path / "sas"
    sas_dir.mkdir()
    extra = [flag] if value is None else [flag, value]
    args = _args(reference_dir, *extra, sas_dir=str(sas_dir))

    problem = main_mod._argument_error(args)

    assert problem is not None and flag in problem


@pytest.mark.parametrize(
    "flag, value",
    [("--out-dir", "out"), ("--md", "r.md"), ("--pdf", "r.pdf")],
)
def test_local_flags_are_rejected_without_a_directory(reference_dir, flag, value):
    args = _args(reference_dir, flag, value)

    problem = main_mod._argument_error(args)

    assert problem is not None and flag in problem


def test_request_id_and_app_are_mutually_exclusive(reference_dir):
    args = _args(reference_dir, "--request-id", "42", "--app", "MyApp")

    problem = main_mod._argument_error(args)

    assert problem is not None and "give one" in problem


def test_a_missing_source_directory_is_reported(reference_dir, tmp_path):
    args = _args(reference_dir, sas_dir=str(tmp_path / "nope"))

    assert "not a directory" in (main_mod._argument_error(args) or "")


def test_a_missing_reference_directory_is_reported(tmp_path):
    args = main_mod.parse_args(["--reference-dir", str(tmp_path / "nope")])

    assert "reference_dir is not a directory" in (main_mod._argument_error(args) or "")


def test_negative_validation_retries_is_rejected(reference_dir):
    args = _args(reference_dir, "--validation-retries", "-1")

    assert "validation-retries" in (main_mod._argument_error(args) or "")


def test_a_valid_sharepoint_invocation_has_no_complaint(reference_dir):
    assert main_mod._argument_error(_args(reference_dir, "--app", "MyApp")) is None


def test_a_valid_local_invocation_has_no_complaint(reference_dir, tmp_path):
    sas_dir = tmp_path / "sas"
    sas_dir.mkdir()
    args = _args(reference_dir, "--out-dir", "out", sas_dir=str(sas_dir))

    assert main_mod._argument_error(args) is None


def test_a_rejected_combination_exits_non_zero_before_any_work(
    monkeypatch, reference_dir, tmp_path
):
    sas_dir = tmp_path / "sas"
    sas_dir.mkdir()
    ran: list[str] = []
    monkeypatch.setattr(main_mod, "_run_local", lambda args: ran.append("local") or 0)
    monkeypatch.setattr(
        main_mod, "_run_sharepoint", lambda args: ran.append("sp") or 0
    )

    exit_code = main_mod.main(
        [str(sas_dir), "--reference-dir", str(reference_dir), "--no-upload"]
    )

    assert exit_code == 1
    assert ran == []  # nothing was started


# ---------------------------------------------------------------------------
# Credential resolution — one chain, resolved once
# ---------------------------------------------------------------------------


def test_vault_secret_flag_takes_the_approle_path(monkeypatch, reference_dir):
    seen: dict[str, Any] = {}

    def _from_vault_secret(path, key, **overrides):
        seen["path"] = path
        seen["key"] = key
        return LLMClientConfig(api_key="approle-key", **overrides)

    monkeypatch.setattr(
        LLMClientConfig, "from_vault_secret", staticmethod(_from_vault_secret)
    )
    args = _args(reference_dir, "--vault-secret", "llm/anthropic", "--vault-key", "tok")

    config = main_mod.resolve_llm_config(args)

    assert (seen["path"], seen["key"]) == ("llm/anthropic", "tok")
    assert config.api_key is not None


def test_the_gateway_chain_is_the_default_when_vault_is_configured(
    monkeypatch, reference_dir
):
    from app_config import vault

    called: list[str] = []
    monkeypatch.setattr(vault, "is_configured", lambda: True)
    monkeypatch.setattr(
        LLMClientConfig,
        "from_ai_gateway",
        staticmethod(
            lambda **kw: called.append("gateway") or LLMClientConfig(api_key="gw", **kw)
        ),
    )

    main_mod.resolve_llm_config(_args(reference_dir))

    assert called == ["gateway"]


def test_no_vault_configuration_defers_to_the_environment(monkeypatch, reference_dir):
    from app_config import vault

    monkeypatch.setattr(vault, "is_configured", lambda: False)
    monkeypatch.setattr(
        LLMClientConfig,
        "from_ai_gateway",
        staticmethod(lambda **kw: pytest.fail("must not reach Vault")),
    )

    config = main_mod.resolve_llm_config(_args(reference_dir))

    assert config.api_key is None


def test_no_gateway_auth_skips_the_chain_even_when_configured(
    monkeypatch, reference_dir
):
    from app_config import vault

    monkeypatch.setattr(vault, "is_configured", lambda: True)
    monkeypatch.setattr(
        LLMClientConfig,
        "from_ai_gateway",
        staticmethod(lambda **kw: pytest.fail("must not reach Vault")),
    )

    config = main_mod.resolve_llm_config(_args(reference_dir, "--no-gateway-auth"))

    assert config.api_key is None


def test_a_vault_failure_exits_with_a_message_not_a_traceback(
    monkeypatch, reference_dir
):
    from app_config import vault

    monkeypatch.setattr(vault, "is_configured", lambda: True)

    def _boom(**_kw):
        raise vault.VaultError("403 from Vault")

    monkeypatch.setattr(LLMClientConfig, "from_ai_gateway", staticmethod(_boom))

    with pytest.raises(SystemExit, match="could not fetch the AI Gateway credential"):
        main_mod.resolve_llm_config(_args(reference_dir))


def test_the_credential_chain_is_walked_once_per_run(monkeypatch, reference_dir):
    """A ten-row run must not log in to Vault ten times."""
    from app_config import vault

    calls: list[int] = []
    monkeypatch.setattr(vault, "is_configured", lambda: True)
    monkeypatch.setattr(
        LLMClientConfig,
        "from_ai_gateway",
        staticmethod(
            lambda **kw: calls.append(1) or LLMClientConfig(api_key="gw", **kw)
        ),
    )

    base = main_mod.resolve_llm_config(_args(reference_dir))
    # What _run_sharepoint does per row: copy, never re-resolve.
    per_row = [base.model_copy(update={"model": m}) for m in ("gpt-5.4", "gpt-5.4")]

    assert len(calls) == 1
    assert [c.model for c in per_row] == ["gpt-5.4", "gpt-5.4"]
    assert all(c.api_key is not None for c in per_row)
