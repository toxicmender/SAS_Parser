"""
Tests for the app_config.vault credential client.

No live Vault (and no hvac install) is needed: connection settings are
resolved from a controlled environment + tmp config.json, and the read/auth
paths are exercised through an injected fake hvac client. Each test isolates
SAS_PARSER_CONFIG and the Vault env vars, and clears both the app_config file
cache and the vault client cache around itself.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

import app_config
from app_config import vault

_VAULT_ENV = (
    "VAULT_ADDR",
    "VAULT_NAMESPACE",
    "VAULT_TOKEN",
    "VAULT_ROLE_ID",
    "VAULT_SECRET_ID",
    "VAULT_CACERT",
    "VAULT_SKIP_VERIFY",
    "VAULT_AUTH_PATH",
    "VAULT_OIDC_ROLE",
    "VAULT_AZURE_SCOPES",
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Empty config file, no Vault env vars, both caches cleared."""
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


def _set(cfg_path, mapping) -> None:
    cfg_path.write_text(json.dumps(mapping), encoding="utf-8")
    app_config.clear_cache()


# ---------------------------------------------------------------------------
# Fake hvac client
# ---------------------------------------------------------------------------


class _FakeKvV2:
    def __init__(self, store):
        self._store = store

    def read_secret_version(self, path, mount_point, raise_on_deleted_version):
        try:
            data = self._store[(mount_point, path)]
        except KeyError:
            raise RuntimeError(f"no secret at {mount_point}/{path}")
        return {"data": {"data": data}}


class _FakeKvV1:
    def __init__(self, store):
        self._store = store

    def read_secret(self, path, mount_point):
        return {"data": self._store[(mount_point, path)]}


class _FakeJwtAuth:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls: list[dict] = []

    def jwt_login(self, role, jwt, path):
        if self.fail:
            raise RuntimeError("role not found")
        self.calls.append({"role": role, "jwt": jwt, "path": path})
        return {"auth": {"client_token": "vault-token-from-jwt"}}


class _FakeClient:
    def __init__(self, store, authenticated=True, jwt_fail=False):
        # Stub the hvac client.secrets.kv.v1/v2 and auth.jwt namespaces with
        # dynamically-built objects. Bound through Any-typed locals so the
        # attribute assignments aren't flagged against the empty synthesized
        # classes.
        kv: Any = type("KV", (), {})()
        kv.v2 = _FakeKvV2(store)
        kv.v1 = _FakeKvV1(store)
        secrets: Any = type("S", (), {})()
        secrets.kv = kv
        self.secrets = secrets
        auth: Any = type("A", (), {})()
        auth.jwt = _FakeJwtAuth(fail=jwt_fail)
        self.auth = auth
        self._authenticated = authenticated

    def is_authenticated(self):
        return self._authenticated


class _FakeAzureClient:
    """Duck-typed stand-in for app_config.azure.AzureAuthClient."""

    def __init__(self, scopes=(), client_id=None, fail=False):
        from app_config.azure import AzureAuthConfig

        self.config = AzureAuthConfig(client_id=client_id, scopes=tuple(scopes))
        self.fail = fail
        self.requested_scopes: tuple[str, ...] | None = None

    def get_token(self, scopes=None):
        from app_config.azure import AzureAuthError

        if self.fail:
            raise AzureAuthError("entra said no")
        self.requested_scopes = tuple(scopes) if scopes else self.config.scopes
        if not self.requested_scopes:
            raise AzureAuthError("no Azure scopes requested")
        return "entra-jwt"


# ---------------------------------------------------------------------------
# VaultConfig resolution
# ---------------------------------------------------------------------------


def test_from_env_reads_env_first(monkeypatch, _isolated):
    monkeypatch.setenv("VAULT_ADDR", "https://vault.example:8200")
    monkeypatch.setenv("VAULT_NAMESPACE", "team-sas")
    monkeypatch.setenv("VAULT_TOKEN", "s.sometoken")
    cfg = vault.VaultConfig.from_env()
    assert cfg.address == "https://vault.example:8200"
    assert cfg.namespace == "team-sas"
    assert cfg.token == "s.sometoken"
    assert cfg.auth_method == "token"


def test_from_env_falls_back_to_config_json(_isolated):
    _set(
        _isolated,
        {"vault": {"address": "https://cfg:8200", "mount_point": "kv", "kv_version": 1}},
    )
    cfg = vault.VaultConfig.from_env()
    assert cfg.address == "https://cfg:8200"
    assert cfg.mount_point == "kv"
    assert cfg.kv_version == 1


def test_env_addr_beats_config(monkeypatch, _isolated):
    _set(_isolated, {"vault": {"address": "https://cfg:8200"}})
    monkeypatch.setenv("VAULT_ADDR", "https://env:8200")
    assert vault.VaultConfig.from_env().address == "https://env:8200"


def test_defaults_without_env_or_config(_isolated):
    cfg = vault.VaultConfig.from_env()
    assert cfg.address is None
    assert cfg.mount_point == vault.DEFAULT_MOUNT_POINT
    assert cfg.kv_version == vault.DEFAULT_KV_VERSION
    assert cfg.timeout == vault.DEFAULT_TIMEOUT
    assert cfg.verify is True
    assert cfg.auth_method is None


def test_wrong_typed_kv_version_degrades(_isolated):
    _set(_isolated, {"vault": {"kv_version": "two"}})
    assert vault.VaultConfig.from_env().kv_version == vault.DEFAULT_KV_VERSION


def test_verify_resolution(monkeypatch, _isolated):
    _set(_isolated, {"vault": {"verify": "/etc/ca.pem"}})
    assert vault.VaultConfig.from_env().verify == "/etc/ca.pem"
    monkeypatch.setenv("VAULT_CACERT", "/env/ca.pem")
    assert vault.VaultConfig.from_env().verify == "/env/ca.pem"
    monkeypatch.setenv("VAULT_SKIP_VERIFY", "true")
    assert vault.VaultConfig.from_env().verify is False


def test_approle_auth_method(monkeypatch, _isolated):
    monkeypatch.setenv("VAULT_ROLE_ID", "role")
    monkeypatch.setenv("VAULT_SECRET_ID", "secret")
    assert vault.VaultConfig.from_env().auth_method == "approle"


def test_token_wins_over_approle(monkeypatch, _isolated):
    monkeypatch.setenv("VAULT_TOKEN", "tok")
    monkeypatch.setenv("VAULT_ROLE_ID", "role")
    monkeypatch.setenv("VAULT_SECRET_ID", "secret")
    assert vault.VaultConfig.from_env().auth_method == "token"


def test_azuread_auth_method(monkeypatch, _isolated):
    monkeypatch.setenv("VAULT_OIDC_ROLE", "sas-parser")
    cfg = vault.VaultConfig.from_env()
    assert cfg.auth_method == "azuread"
    assert cfg.oidc_role == "sas-parser"
    assert cfg.auth_path == vault.DEFAULT_AUTH_PATH


def test_approle_wins_over_azuread(monkeypatch, _isolated):
    monkeypatch.setenv("VAULT_ROLE_ID", "role")
    monkeypatch.setenv("VAULT_SECRET_ID", "secret")
    monkeypatch.setenv("VAULT_OIDC_ROLE", "sas-parser")
    assert vault.VaultConfig.from_env().auth_method == "approle"


def test_azuread_config_json_fallback(_isolated):
    _set(
        _isolated,
        {
            "vault": {
                "oidc_role": "cfg-role",
                "auth_path": "oidc",
                "azure_scopes": ["api://vault/.default"],
            }
        },
    )
    cfg = vault.VaultConfig.from_env()
    assert cfg.oidc_role == "cfg-role"
    assert cfg.auth_path == "oidc"
    assert cfg.azure_scopes == ("api://vault/.default",)


def test_azure_scopes_env_parsing(monkeypatch, _isolated):
    monkeypatch.setenv("VAULT_AZURE_SCOPES", "a/.default, b/.default")
    assert vault.VaultConfig.from_env().azure_scopes == ("a/.default", "b/.default")


def test_wrong_typed_azure_scopes_degrades(_isolated):
    _set(_isolated, {"vault": {"azure_scopes": [1, 2]}})
    assert vault.VaultConfig.from_env().azure_scopes == ()


def test_secrets_never_in_repr():
    cfg = vault.VaultConfig(
        address="https://v", token="tok", role_id="r", secret_id="s"
    )
    text = repr(cfg)
    assert "tok" not in text and "https://v" in text


# ---------------------------------------------------------------------------
# VaultClient reads (injected fake client)
# ---------------------------------------------------------------------------


def _client(store, *, kv_version=2, mount="secret"):
    cfg = vault.VaultConfig(mount_point=mount, kv_version=kv_version, token="tok")
    return vault.VaultClient(cfg, client=_FakeClient(store))


def test_get_secret_whole_dict():
    client = _client({("secret", "llm/anthropic"): {"api_key": "sk", "org": "o"}})
    assert client.get_secret("llm/anthropic") == {"api_key": "sk", "org": "o"}


def test_get_secret_single_key():
    client = _client({("secret", "llm/anthropic"): {"api_key": "sk"}})
    assert client.get_secret("llm/anthropic", "api_key") == "sk"


def test_get_secret_missing_key_raises():
    client = _client({("secret", "p"): {"a": "1"}})
    with pytest.raises(vault.VaultError, match="key 'b' not found"):
        client.get_secret("p", "b")


def test_get_secret_missing_path_raises():
    client = _client({("secret", "p"): {"a": "1"}})
    with pytest.raises(vault.VaultError, match="could not read Vault secret 'q'"):
        client.get_secret("q")


def test_kv_v1_read():
    client = _client({("secret", "p"): {"a": "1"}}, kv_version=1)
    assert client.get_secret("p", "a") == "1"


def test_mount_point_override():
    store = {("other", "p"): {"a": "1"}}
    client = _client(store)
    assert client.get_secret("p", "a", mount_point="other") == "1"


# ---------------------------------------------------------------------------
# Build-time config validation (runs before hvac import)
# ---------------------------------------------------------------------------


def test_missing_address_raises():
    client = vault.VaultClient(vault.VaultConfig(token="tok"))
    with pytest.raises(vault.VaultError, match="no Vault address"):
        _ = client.client


def test_missing_credentials_raises():
    client = vault.VaultClient(vault.VaultConfig(address="https://v"))
    with pytest.raises(vault.VaultError, match="no Vault credentials"):
        _ = client.client


def test_missing_hvac_raises_helpful_error():
    # hvac is an optional extra; when it is not installed a fully-configured
    # client still fails at import with an install hint (skip if it is present).
    try:
        import hvac  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("hvac is installed; the import-error path is unreachable")
    client = vault.VaultClient(vault.VaultConfig(address="https://v", token="tok"))
    with pytest.raises(vault.VaultError, match="hvac is required"):
        _ = client.client


def test_authentication_failure_raises():
    cfg = vault.VaultConfig(token="tok")
    fake = _FakeClient({}, authenticated=False)
    with pytest.raises(vault.VaultError, match="authentication failed"):
        vault._authenticate(fake, cfg)


# ---------------------------------------------------------------------------
# azuread (Entra ID OIDC) login
# ---------------------------------------------------------------------------


def _patch_azure(monkeypatch, fake):
    from app_config import azure

    monkeypatch.setattr(azure, "get_azure_client", lambda: fake)


def test_azuread_login_flow(monkeypatch):
    fake_azure = _FakeAzureClient(scopes=("api://vault/.default",))
    _patch_azure(monkeypatch, fake_azure)
    cfg = vault.VaultConfig(address="https://v", oidc_role="sas", auth_path="oidc")
    fake = _FakeClient({})
    vault._authenticate(fake, cfg)
    assert fake.auth.jwt.calls == [
        {"role": "sas", "jwt": "entra-jwt", "path": "oidc"}
    ]


def test_azuread_vault_scopes_win(monkeypatch):
    fake_azure = _FakeAzureClient(scopes=("azure-configured/.default",))
    _patch_azure(monkeypatch, fake_azure)
    cfg = vault.VaultConfig(
        address="https://v", oidc_role="sas", azure_scopes=("vault-pinned/.default",)
    )
    vault._authenticate(_FakeClient({}), cfg)
    assert fake_azure.requested_scopes == ("vault-pinned/.default",)


def test_azuread_scopes_fall_back_to_client_id(monkeypatch):
    fake_azure = _FakeAzureClient(client_id="abc-123")
    _patch_azure(monkeypatch, fake_azure)
    cfg = vault.VaultConfig(address="https://v", oidc_role="sas")
    vault._authenticate(_FakeClient({}), cfg)
    assert fake_azure.requested_scopes == ("abc-123/.default",)


def test_azuread_azure_error_wrapped(monkeypatch):
    _patch_azure(monkeypatch, _FakeAzureClient(fail=True))
    cfg = vault.VaultConfig(address="https://v", oidc_role="sas")
    with pytest.raises(vault.VaultError, match="could not acquire an Entra ID token"):
        vault._authenticate(_FakeClient({}), cfg)


def test_azuread_login_failure_wrapped(monkeypatch):
    _patch_azure(monkeypatch, _FakeAzureClient(scopes=("s/.default",)))
    cfg = vault.VaultConfig(address="https://v", oidc_role="sas")
    fake = _FakeClient({}, jwt_fail=True)
    with pytest.raises(vault.VaultError, match="azuread login failed for role 'sas'"):
        vault._authenticate(fake, cfg)


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------


def test_get_vault_client_is_cached():
    first = vault.get_vault_client()
    assert vault.get_vault_client() is first
    vault.clear_cache()
    assert vault.get_vault_client() is not first


# ---------------------------------------------------------------------------
# AI Gateway secret
# ---------------------------------------------------------------------------


def _gateway_client(monkeypatch, data, *, mount="secret", path=None) -> None:
    """Seed the shared VaultClient with an AI Gateway secret."""
    store = {(mount, path or vault.AI_GATEWAY_PATH): data}
    cfg = vault.VaultConfig(address="https://v", token="s.t", mount_point=mount)
    monkeypatch.setattr(
        vault, "_client_cache", vault.VaultClient(cfg, client=_FakeClient(store))
    )


def test_ai_gateway_path_is_the_documented_one():
    # <vault_addr>/v1/secret/data/appsvc/ai_gateway with the default mount and
    # KV v2, where "secret" is the mount and "data" the KV v2 infix.
    assert vault.AI_GATEWAY_PATH == "appsvc/ai_gateway"
    assert vault.DEFAULT_MOUNT_POINT == "secret"
    assert vault.DEFAULT_KV_VERSION == 2


def test_get_ai_gateway_secret_reads_the_default_path(monkeypatch, _isolated):
    _gateway_client(monkeypatch, {"token": "gw-token", "base_url": "https://gw"})
    assert vault.get_ai_gateway_secret() == {
        "token": "gw-token",
        "base_url": "https://gw",
    }


def test_ai_gateway_token_finds_the_common_keys(monkeypatch, _isolated):
    _gateway_client(monkeypatch, {"api_key": "gw-token"})
    assert vault.ai_gateway_token() == "gw-token"


def test_ai_gateway_token_prefers_an_explicit_key(monkeypatch, _isolated):
    _gateway_client(monkeypatch, {"token": "wrong", "gateway_pat": "right"})
    assert vault.ai_gateway_token(key="gateway_pat") == "right"


def test_ai_gateway_key_configurable_in_config_json(monkeypatch, _isolated):
    _set(_isolated, {"vault": {"ai_gateway_key": "gateway_pat"}})
    _gateway_client(monkeypatch, {"token": "wrong", "gateway_pat": "right"})
    assert vault.ai_gateway_token() == "right"


def test_ai_gateway_token_error_lists_the_available_keys(monkeypatch, _isolated):
    _gateway_client(monkeypatch, {"username": "u", "password": "p"})
    with pytest.raises(vault.VaultError, match=r"\['password', 'username'\]"):
        vault.ai_gateway_token()


def test_ai_gateway_missing_explicit_key_raises(monkeypatch, _isolated):
    _gateway_client(monkeypatch, {"token": "gw-token"})
    with pytest.raises(vault.VaultError, match="key 'nope' not found"):
        vault.ai_gateway_token(key="nope")


def test_ai_gateway_base_url_is_optional(monkeypatch, _isolated):
    _gateway_client(monkeypatch, {"token": "gw-token"})
    # No endpoint in the secret leaves the configured llm_client.base_url alone.
    assert vault.ai_gateway_base_url() is None


def test_ai_gateway_base_url_from_the_secret(monkeypatch, _isolated):
    _gateway_client(monkeypatch, {"token": "t", "endpoint": "https://gw.example"})
    assert vault.ai_gateway_base_url() == "https://gw.example"


def test_ai_gateway_reads_are_not_cached(monkeypatch, _isolated):
    # Rotated secrets must be picked up without a restart, so every call hits
    # Vault — the client is cached, the read is not.
    store = {("secret", vault.AI_GATEWAY_PATH): {"token": "first"}}
    cfg = vault.VaultConfig(address="https://v", token="s.t")
    monkeypatch.setattr(
        vault, "_client_cache", vault.VaultClient(cfg, client=_FakeClient(store))
    )
    assert vault.ai_gateway_token() == "first"
    store[("secret", vault.AI_GATEWAY_PATH)] = {"token": "rotated"}
    assert vault.ai_gateway_token() == "rotated"
