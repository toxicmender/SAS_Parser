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
    "VAULT_APP_NAME",
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


class _Forbidden(RuntimeError):
    """Stands in for hvac.exceptions.Forbidden, which carries the status code."""

    status_code = 403


class _FakeKvV2:
    def __init__(self, store, forbid_first=0):
        self._store = store
        # Reject this many reads with a 403 before answering normally, so the
        # re-authenticate-and-retry path can be exercised.
        self.forbid_first = forbid_first
        self.reads = 0

    def read_secret_version(self, path, mount_point, raise_on_deleted_version):
        self.reads += 1
        if self.forbid_first > 0:
            self.forbid_first -= 1
            raise _Forbidden("permission denied")
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
    def __init__(self, fail=False, lease=None):
        self.fail = fail
        self.lease = lease
        self.calls: list[dict] = []

    def jwt_login(self, role, jwt, path):
        if self.fail:
            raise RuntimeError("role not found")
        self.calls.append({"role": role, "jwt": jwt, "path": path})
        auth = {"client_token": "vault-token-from-jwt"}
        if self.lease is not None:
            auth["lease_duration"] = self.lease
        return {"auth": auth}


class _FakeApproleAuth:
    def __init__(self, fail=False, lease=None):
        self.fail = fail
        self.lease = lease
        self.calls: list[dict] = []

    def login(self, role_id, secret_id):
        if self.fail:
            raise RuntimeError("invalid role or secret id")
        self.calls.append({"role_id": role_id, "secret_id": secret_id})
        auth = {"client_token": "vault-token-from-approle"}
        if self.lease is not None:
            auth["lease_duration"] = self.lease
        return {"auth": auth}


class _FakeClient:
    def __init__(
        self,
        store,
        authenticated=True,
        jwt_fail=False,
        approle_fail=False,
        lease=None,
        forbid_first=0,
    ):
        # Stub the hvac client.secrets.kv.v1/v2 and auth.jwt/auth.approle
        # namespaces with dynamically-built objects. Bound through Any-typed
        # locals so the attribute assignments aren't flagged against the empty
        # synthesized classes.
        kv: Any = type("KV", (), {})()
        kv.v2 = _FakeKvV2(store, forbid_first=forbid_first)
        kv.v1 = _FakeKvV1(store)
        secrets: Any = type("S", (), {})()
        secrets.kv = kv
        self.secrets = secrets
        auth: Any = type("A", (), {})()
        auth.jwt = _FakeJwtAuth(fail=jwt_fail, lease=lease)
        auth.approle = _FakeApproleAuth(fail=approle_fail, lease=lease)
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
    monkeypatch.setenv("VAULT_APP_NAME", "sas-parser")
    cfg = vault.VaultConfig.from_env()
    assert cfg.auth_method == "azuread"
    assert cfg.app_name == "sas-parser"
    assert cfg.auth_path == vault.DEFAULT_AUTH_PATH


def test_approle_wins_over_azuread(monkeypatch, _isolated):
    monkeypatch.setenv("VAULT_ROLE_ID", "role")
    monkeypatch.setenv("VAULT_SECRET_ID", "secret")
    monkeypatch.setenv("VAULT_APP_NAME", "sas-parser")
    assert vault.VaultConfig.from_env().auth_method == "approle"


def test_azuread_config_json_fallback(_isolated):
    _set(
        _isolated,
        {
            "vault": {
                "app_name": "cfg-role",
                "auth_path": "oidc",
                "azure_scopes": ["api://vault/.default"],
            }
        },
    )
    cfg = vault.VaultConfig.from_env()
    assert cfg.app_name == "cfg-role"
    assert cfg.auth_path == "oidc"
    assert cfg.azure_scopes == ("api://vault/.default",)


def test_app_name_env_beats_config(monkeypatch, _isolated):
    _set(_isolated, {"vault": {"app_name": "cfg-role"}})
    monkeypatch.setenv("VAULT_APP_NAME", "env-role")
    assert vault.VaultConfig.from_env().app_name == "env-role"


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
    cfg = vault.VaultConfig(address="https://v", app_name="sas", auth_path="oidc")
    fake = _FakeClient({})
    vault._authenticate(fake, cfg)
    # app_name is the Vault role for the login, not just a path prefix.
    assert fake.auth.jwt.calls == [
        {"role": "sas", "jwt": "entra-jwt", "path": "oidc"}
    ]


def test_azuread_vault_scopes_win(monkeypatch):
    fake_azure = _FakeAzureClient(scopes=("azure-configured/.default",))
    _patch_azure(monkeypatch, fake_azure)
    cfg = vault.VaultConfig(
        address="https://v", app_name="sas", azure_scopes=("vault-pinned/.default",)
    )
    vault._authenticate(_FakeClient({}), cfg)
    assert fake_azure.requested_scopes == ("vault-pinned/.default",)


def test_azuread_azure_module_scopes_beat_the_arm_default(monkeypatch):
    fake_azure = _FakeAzureClient(scopes=("azure-configured/.default",))
    _patch_azure(monkeypatch, fake_azure)
    cfg = vault.VaultConfig(address="https://v", app_name="sas")
    vault._authenticate(_FakeClient({}), cfg)
    assert fake_azure.requested_scopes == ("azure-configured/.default",)


def test_azuread_scopes_default_to_arm(monkeypatch):
    # Nothing configured anywhere: the JWT is minted for Azure Resource
    # Manager, which is the audience the deployment's Vault role is bound to.
    # A client id being present must NOT divert it to <client_id>/.default.
    fake_azure = _FakeAzureClient(client_id="abc-123")
    _patch_azure(monkeypatch, fake_azure)
    cfg = vault.VaultConfig(address="https://v", app_name="sas")
    vault._authenticate(_FakeClient({}), cfg)
    assert fake_azure.requested_scopes == (vault._ARM_DEFAULT_SCOPE,)
    assert vault._ARM_DEFAULT_SCOPE == "https://management.azure.com//.default"


def test_azuread_azure_error_wrapped(monkeypatch):
    _patch_azure(monkeypatch, _FakeAzureClient(fail=True))
    cfg = vault.VaultConfig(address="https://v", app_name="sas")
    with pytest.raises(vault.VaultError, match="could not acquire an Entra ID token"):
        vault._authenticate(_FakeClient({}), cfg)


def test_azuread_login_failure_wrapped(monkeypatch):
    _patch_azure(monkeypatch, _FakeAzureClient(scopes=("s/.default",)))
    cfg = vault.VaultConfig(address="https://v", app_name="sas")
    fake = _FakeClient({}, jwt_fail=True)
    with pytest.raises(vault.VaultError, match="azuread login failed for role 'sas'"):
        vault._authenticate(fake, cfg)


# ---------------------------------------------------------------------------
# approle login
# ---------------------------------------------------------------------------


def test_approle_login_flow():
    cfg = vault.VaultConfig(address="https://v", role_id="r", secret_id="s")
    fake = _FakeClient({})
    vault._authenticate(fake, cfg)
    assert fake.auth.approle.calls == [{"role_id": "r", "secret_id": "s"}]


def test_approle_login_failure_is_a_vault_error():
    # Callers except VaultError around a lookup regardless of auth method, so
    # the approle branch must wrap hvac's exception like the azuread one does.
    cfg = vault.VaultConfig(address="https://v", role_id="r", secret_id="s")
    fake = _FakeClient({}, approle_fail=True)
    with pytest.raises(vault.VaultError, match="approle login failed"):
        vault._authenticate(fake, cfg)


# ---------------------------------------------------------------------------
# Token lifetime: lease-driven refresh and the 403 retry
# ---------------------------------------------------------------------------


def _leased_client(store, *, lease=None, forbid_first=0):
    """A VaultClient on approle auth whose fake login reports *lease*."""
    cfg = vault.VaultConfig(
        address="https://v", role_id="r", secret_id="s", mount_point="secret"
    )
    fake = _FakeClient(store, lease=lease, forbid_first=forbid_first)
    client = vault.VaultClient(cfg, client=fake)
    return client, fake


def test_login_lease_is_recorded(monkeypatch):
    cfg = vault.VaultConfig(address="https://v", role_id="r", secret_id="s")
    fake = _FakeClient({}, lease=3600)
    client = vault.VaultClient(cfg)
    monkeypatch.setattr(
        vault.VaultClient,
        "_build_client",
        staticmethod(lambda config: (fake, vault._authenticate(fake, config))),
    )
    _ = client.client
    assert client._expires_at is not None


def test_expired_lease_reauthenticates_before_a_read():
    store = {("secret", "p"): {"a": "1"}}
    client, fake = _leased_client(store, lease=3600)
    client._expires_at = 0.0  # as if the lease had already run out
    assert client.get_secret("p", "a") == "1"
    assert len(fake.auth.approle.calls) == 1
    # The fresh lease is tracked, so the next read does not log in again.
    assert client.get_secret("p", "a") == "1"
    assert len(fake.auth.approle.calls) == 1


def test_live_lease_does_not_reauthenticate():
    store = {("secret", "p"): {"a": "1"}}
    client, fake = _leased_client(store, lease=3600)
    client._set_expiry(3600)
    client.get_secret("p", "a")
    assert fake.auth.approle.calls == []


def test_token_auth_is_exempt_from_expiry():
    # An operator-supplied VAULT_TOKEN has a lifetime that is their business.
    cfg = vault.VaultConfig(address="https://v", token="s.tok")
    client = vault.VaultClient(cfg, client=_FakeClient({("secret", "p"): {"a": "1"}}))
    client.get_secret("p", "a")
    assert client._expires_at is None


def test_forbidden_read_retries_once_behind_a_fresh_login():
    store = {("secret", "p"): {"a": "1"}}
    client, fake = _leased_client(store, forbid_first=1)
    assert client.get_secret("p", "a") == "1"
    assert len(fake.auth.approle.calls) == 1
    assert fake.secrets.kv.v2.reads == 2


def test_forbidden_read_raises_after_one_retry():
    store = {("secret", "p"): {"a": "1"}}
    client, fake = _leased_client(store, forbid_first=2)
    with pytest.raises(vault.VaultError, match="even after re-authenticating"):
        client.get_secret("p", "a")
    assert len(fake.auth.approle.calls) == 1  # retried once, not in a loop


def test_forbidden_read_is_not_retried_under_token_auth():
    cfg = vault.VaultConfig(address="https://v", token="s.tok")
    fake = _FakeClient({("secret", "p"): {"a": "1"}}, forbid_first=1)
    client = vault.VaultClient(cfg, client=fake)
    with pytest.raises(vault.VaultError, match="could not read Vault secret"):
        client.get_secret("p", "a")
    assert fake.secrets.kv.v2.reads == 1


def test_a_missing_secret_is_not_retried():
    client, fake = _leased_client({("secret", "p"): {"a": "1"}})
    with pytest.raises(vault.VaultError, match="could not read Vault secret 'q'"):
        client.get_secret("q")
    assert fake.auth.approle.calls == []


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


_APP_NAME = "sas-parser"
_DEFAULT_GATEWAY_PATH = f"{_APP_NAME}/ai_gateway"


def _gateway_client(
    monkeypatch, data, *, mount="secret", path=None, app_name=_APP_NAME, **config
) -> None:
    """Seed the shared VaultClient with an AI Gateway secret."""
    store = {(mount, path or _DEFAULT_GATEWAY_PATH): data}
    cfg = vault.VaultConfig(
        address="https://v",
        token="s.t",
        mount_point=mount,
        app_name=app_name,
        **config,
    )
    monkeypatch.setattr(
        vault, "_client_cache", vault.VaultClient(cfg, client=_FakeClient(store))
    )


def test_gateway_path_derives_from_app_name(monkeypatch, _isolated):
    # The reference reads /v1/secret/data/<app_name>/ai_gateway, where "secret"
    # is the mount and "data" the KV v2 infix.
    cfg = vault.VaultConfig(app_name="appsvc102630")
    assert cfg.resolved_ai_gateway_path == "appsvc102630/ai_gateway"
    assert vault.DEFAULT_MOUNT_POINT == "secret"
    assert vault.DEFAULT_KV_VERSION == 2


def test_gateway_path_prefers_the_configured_one(monkeypatch, _isolated):
    # vault.ai_gateway_path wins over the app_name derivation.
    _gateway_client(
        monkeypatch,
        {"token": "configured"},
        path="team/gateway",
        ai_gateway_path="team/gateway",
    )
    assert vault.get_ai_gateway_secret() == {"token": "configured"}


def test_gateway_path_falls_back_to_the_app_name(monkeypatch, _isolated):
    _gateway_client(monkeypatch, {"token": "by-app-name"})
    assert vault.get_ai_gateway_secret() == {"token": "by-app-name"}


def test_gateway_path_unresolvable_raises(monkeypatch, _isolated):
    # Neither configured nor derivable: no universal default is guessed.
    _gateway_client(monkeypatch, {"token": "unreachable"}, app_name=None)
    with pytest.raises(vault.VaultError, match="no AI Gateway secret path"):
        vault.get_ai_gateway_secret()


def test_explicit_path_argument_beats_every_fallback(monkeypatch, _isolated):
    _gateway_client(monkeypatch, {"token": "explicit"}, path="given/path")
    assert vault.get_ai_gateway_secret("given/path") == {"token": "explicit"}


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
    _gateway_client(
        monkeypatch,
        {"token": "wrong", "gateway_pat": "right"},
        ai_gateway_key="gateway_pat",
    )
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


def test_ai_gateway_version_is_optional(monkeypatch, _isolated):
    _gateway_client(monkeypatch, {"token": "t"})
    assert vault.ai_gateway_version() is None


def test_ai_gateway_version_from_the_secret(monkeypatch, _isolated):
    _gateway_client(monkeypatch, {"token": "t", "gateway_version": "v2"})
    assert vault.ai_gateway_version() == "v2"


def test_ai_gateway_reads_are_not_cached(monkeypatch, _isolated):
    # Rotated secrets must be picked up without a restart, so every call hits
    # Vault — the client is cached, the read is not.
    store = {("secret", _DEFAULT_GATEWAY_PATH): {"token": "first"}}
    cfg = vault.VaultConfig(address="https://v", token="s.t", app_name=_APP_NAME)
    monkeypatch.setattr(
        vault, "_client_cache", vault.VaultClient(cfg, client=_FakeClient(store))
    )
    assert vault.ai_gateway_token() == "first"
    store[("secret", _DEFAULT_GATEWAY_PATH)] = {"token": "rotated"}
    assert vault.ai_gateway_token() == "rotated"
