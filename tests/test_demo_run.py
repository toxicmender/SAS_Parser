"""
Tests for demo_run's LLM credential resolution.

The demo picks the pipeline's API key from one of three places, and which one
it picks is security-relevant: silently falling back to a different credential
than the operator intended is worse than failing the run. These tests pin that
precedence and, in particular, the difference between "Vault is not configured"
(fall back to the provider env var) and "Vault is configured but broken" (exit
non-zero).

No Vault, Entra ID, or Databricks is needed: app_config.vault is stubbed at the
seams demo_run actually calls.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

import app_config
import demo_run
from app_config import vault

_VAULT_ENV = (
    "VAULT_ADDR",
    "VAULT_TOKEN",
    "VAULT_ROLE_ID",
    "VAULT_SECRET_ID",
    "VAULT_OIDC_ROLE",
    "VAULT_NAMESPACE",
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Empty config file, no Vault env vars, caches cleared."""
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(app_config.ENV_VAR, str(cfg))
    for var in _VAULT_ENV:
        monkeypatch.delenv(var, raising=False)
    app_config.clear_cache()
    vault.clear_cache()
    yield cfg
    app_config.clear_cache()
    vault.clear_cache()


def _args(**overrides):
    """An argparse.Namespace with the credential flags at their defaults."""
    import argparse

    defaults = {"vault_secret": None, "vault_key": "api_key", "no_gateway_auth": False}
    return argparse.Namespace(**{**defaults, **overrides})


def _vault_configured(monkeypatch) -> None:
    """The env of a workspace whose Vault does the azuread (JWT) login."""
    monkeypatch.setenv("VAULT_ADDR", "https://vault.example:8200")
    monkeypatch.setenv("VAULT_OIDC_ROLE", "sas-parser")


def _gateway_secret(monkeypatch, data) -> None:
    monkeypatch.setattr(vault, "get_ai_gateway_secret", lambda path=None: data)


# ---------------------------------------------------------------------------
# The default: the AI Gateway chain
# ---------------------------------------------------------------------------


def test_gateway_chain_is_the_default(monkeypatch, _isolated):
    _vault_configured(monkeypatch)
    _gateway_secret(monkeypatch, {"token": "gw-token"})
    assert demo_run._resolve_llm_credentials(_args()) == ("gw-token", None)


def test_gateway_base_url_forwarded_when_the_secret_has_one(monkeypatch, _isolated):
    _vault_configured(monkeypatch)
    _gateway_secret(monkeypatch, {"token": "gw-token", "base_url": "https://gw/v1"})
    assert demo_run._resolve_llm_credentials(_args()) == ("gw-token", "https://gw/v1")


def test_no_base_url_leaves_config_json_in_charge(monkeypatch, _isolated):
    _vault_configured(monkeypatch)
    _gateway_secret(monkeypatch, {"token": "gw-token"})
    _, base_url = demo_run._resolve_llm_credentials(_args())
    # None, not "", so the pipeline omits it rather than overriding the config.
    assert base_url is None


def test_token_auth_also_counts_as_configured(monkeypatch, _isolated):
    # The chain is not exclusive to azuread login; any usable Vault auth reads
    # the same secret.
    monkeypatch.setenv("VAULT_ADDR", "https://vault.example:8200")
    monkeypatch.setenv("VAULT_TOKEN", "s.token")
    _gateway_secret(monkeypatch, {"token": "gw-token"})
    assert demo_run._resolve_llm_credentials(_args())[0] == "gw-token"


# ---------------------------------------------------------------------------
# Falling back to the provider env var
# ---------------------------------------------------------------------------


def test_unconfigured_vault_defers_to_the_provider_env_var(_isolated):
    # The local-development path: no Vault settings at all.
    assert demo_run._resolve_llm_credentials(_args()) == (None, None)


def test_no_gateway_auth_flag_opts_out(monkeypatch, _isolated):
    _vault_configured(monkeypatch)

    def _boom(path=None):
        raise AssertionError("--no-gateway-auth must not touch Vault")

    monkeypatch.setattr(vault, "get_ai_gateway_secret", _boom)
    assert demo_run._resolve_llm_credentials(_args(no_gateway_auth=True)) == (None, None)


def test_address_without_a_login_method_is_not_configured(monkeypatch, _isolated):
    monkeypatch.setenv("VAULT_ADDR", "https://vault.example:8200")
    # No token, no AppRole, no oidc role — nothing to log in with.
    assert demo_run._resolve_llm_credentials(_args()) == (None, None)


# ---------------------------------------------------------------------------
# A configured-but-broken Vault must not silently fall back
# ---------------------------------------------------------------------------


def test_broken_vault_exits_rather_than_falling_back(monkeypatch, _isolated):
    _vault_configured(monkeypatch)

    def _fail(path=None):
        raise vault.VaultError("permission denied")

    monkeypatch.setattr(vault, "get_ai_gateway_secret", _fail)
    with pytest.raises(SystemExit) as excinfo:
        demo_run._resolve_llm_credentials(_args())
    message = str(excinfo.value)
    assert "permission denied" in message
    # The message must say how to proceed deliberately.
    assert "--no-gateway-auth" in message


def test_missing_token_field_exits(monkeypatch, _isolated):
    _vault_configured(monkeypatch)
    _gateway_secret(monkeypatch, {"username": "u"})
    with pytest.raises(SystemExit):
        demo_run._resolve_llm_credentials(_args())


# ---------------------------------------------------------------------------
# --vault-secret still overrides everything
# ---------------------------------------------------------------------------


def test_vault_secret_flag_beats_the_gateway_chain(monkeypatch, _isolated):
    _vault_configured(monkeypatch)

    def _boom(path=None):
        raise AssertionError("--vault-secret must not use the gateway chain")

    monkeypatch.setattr(vault, "get_ai_gateway_secret", _boom)
    monkeypatch.setattr(
        demo_run, "_fetch_api_key_from_vault", lambda path, key: f"approle:{path}/{key}"
    )
    assert demo_run._resolve_llm_credentials(_args(vault_secret="llm/anthropic")) == (
        "approle:llm/anthropic/api_key",
        None,
    )


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_no_gateway_auth_defaults_to_off():
    args = demo_run.parse_args(["local", "some_dir"])
    assert args.no_gateway_auth is False
    assert args.vault_secret is None


def test_no_gateway_auth_parses():
    assert demo_run.parse_args(["local", "some_dir", "--no-gateway-auth"]).no_gateway_auth
