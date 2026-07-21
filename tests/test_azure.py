"""
Tests for the app_config.azure Entra ID auth client.

No live Entra ID (and no msal install) is needed: settings are resolved from a
controlled environment + tmp config.json, and the token-acquisition paths are
exercised through an injected fake MSAL app. Each test isolates
SAS_PARSER_CONFIG and the Azure env vars, and clears both the app_config file
cache and the azure client cache around itself.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

import app_config
from app_config import azure, databricks

_AZURE_ENV = (
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_AUTHORITY_HOST",
    "AZURE_SCOPES",
    "AZURE_CLIENT_CERTIFICATE_PATH",
    "AZURE_CLIENT_CERTIFICATE_THUMBPRINT",
)

# get_azure_client falls back to the service principal in the Databricks
# secret scope, so the Databricks env has to be isolated here too.
_DATABRICKS_ENV = (
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_SECRET_SCOPE",
    "DATABRICKS_RUNTIME_VERSION",
    "ARM_TENANT_ID",
    "ARM_CLIENT_ID",
    "ARM_CLIENT_SECRET",
)

_SCOPES = ("api://x/.default",)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Empty config file, no Azure/Databricks env vars, all caches cleared."""
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(app_config.ENV_VAR, str(cfg))
    for var in _AZURE_ENV + _DATABRICKS_ENV:
        monkeypatch.delenv(var, raising=False)
    app_config.clear_cache()
    azure.clear_cache()
    databricks.clear_cache()
    yield cfg
    app_config.clear_cache()
    azure.clear_cache()
    databricks.clear_cache()


def _set(cfg_path, mapping) -> None:
    cfg_path.write_text(json.dumps(mapping), encoding="utf-8")
    app_config.clear_cache()


# ---------------------------------------------------------------------------
# Fake MSAL apps
# ---------------------------------------------------------------------------


class _FakeConfidentialApp:
    """Records the scopes of each call so cache hits are observable."""

    def __init__(self, expires_in=3600, result=None):
        self.calls: list[tuple[str, ...]] = []
        self._expires_in = expires_in
        self._result = result

    def acquire_token_for_client(self, scopes):
        self.calls.append(tuple(scopes))
        if self._result is not None:
            return self._result
        return {
            "access_token": f"tok-{len(self.calls)}",
            "expires_in": self._expires_in,
        }


class _FakePublicApp:
    def __init__(self, *, accounts=(), silent=None, flow=None, result=None):
        self._accounts = list(accounts)
        self._silent = silent
        self._flow = flow if flow is not None else {
            "user_code": "ABC-123",
            "message": "go to https://microsoft.com/devicelogin and enter ABC-123",
        }
        self._result = result or {"access_token": "device-tok", "expires_in": 3600}
        self.initiated = False

    def get_accounts(self):
        return self._accounts

    def acquire_token_silent(self, scopes, account):
        return self._silent

    def initiate_device_flow(self, scopes):
        self.initiated = True
        return self._flow

    def acquire_token_by_device_flow(self, flow):
        return self._result


def _sp_config(**kwargs):
    """A service-principal config: client_credentials flow, secret credential."""
    base: dict[str, Any] = dict(
        tenant_id="t-1", client_id="c-1", client_secret="s-1", scopes=_SCOPES
    )
    base.update(kwargs)
    return azure.AzureAuthConfig(**base)


# ---------------------------------------------------------------------------
# AzureAuthConfig resolution
# ---------------------------------------------------------------------------


def test_from_env_reads_env_first(monkeypatch, _isolated):
    monkeypatch.setenv("AZURE_TENANT_ID", "tenant-guid")
    monkeypatch.setenv("AZURE_CLIENT_ID", "client-guid")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "shh")
    cfg = azure.AzureAuthConfig.from_env()
    assert cfg.tenant_id == "tenant-guid"
    assert cfg.client_id == "client-guid"
    assert cfg.client_secret == "shh"
    assert cfg.auth_flow == azure.FLOW_CLIENT_CREDENTIALS


def test_from_env_falls_back_to_config_json(_isolated):
    _set(
        _isolated,
        {"azure": {"tenant_id": "cfg-tenant", "client_id": "cfg-client", "timeout": 5}},
    )
    cfg = azure.AzureAuthConfig.from_env()
    assert cfg.tenant_id == "cfg-tenant"
    assert cfg.client_id == "cfg-client"
    assert cfg.timeout == 5


def test_env_tenant_beats_config(monkeypatch, _isolated):
    _set(_isolated, {"azure": {"tenant_id": "cfg-tenant"}})
    monkeypatch.setenv("AZURE_TENANT_ID", "env-tenant")
    assert azure.AzureAuthConfig.from_env().tenant_id == "env-tenant"


def test_defaults_without_env_or_config(_isolated):
    cfg = azure.AzureAuthConfig.from_env()
    assert cfg.tenant_id is None
    assert cfg.client_id is None
    assert cfg.authority_host == azure.DEFAULT_AUTHORITY_HOST
    assert cfg.scopes == ()
    assert cfg.flow is None
    assert cfg.timeout == azure.DEFAULT_TIMEOUT
    assert cfg.auth_flow is None


def test_wrong_typed_timeout_degrades(_isolated):
    _set(_isolated, {"azure": {"timeout": "quick"}})
    assert azure.AzureAuthConfig.from_env().timeout == azure.DEFAULT_TIMEOUT


def test_authority_host_trailing_slash_stripped(monkeypatch, _isolated):
    monkeypatch.setenv("AZURE_AUTHORITY_HOST", "https://login.microsoftonline.us/")
    monkeypatch.setenv("AZURE_TENANT_ID", "t")
    cfg = azure.AzureAuthConfig.from_env()
    assert cfg.authority == "https://login.microsoftonline.us/t"


def test_scopes_from_env_space_or_comma_separated(monkeypatch, _isolated):
    monkeypatch.setenv("AZURE_SCOPES", "api://x/.default, api://y/.default")
    assert azure.AzureAuthConfig.from_env().scopes == (
        "api://x/.default",
        "api://y/.default",
    )
    monkeypatch.setenv("AZURE_SCOPES", "api://x/.default api://y/.default")
    assert len(azure.AzureAuthConfig.from_env().scopes) == 2


def test_scopes_from_config_list(_isolated):
    _set(_isolated, {"azure": {"scopes": ["api://x/.default"]}})
    assert azure.AzureAuthConfig.from_env().scopes == ("api://x/.default",)


def test_env_scopes_beat_config(monkeypatch, _isolated):
    _set(_isolated, {"azure": {"scopes": ["api://cfg/.default"]}})
    monkeypatch.setenv("AZURE_SCOPES", "api://env/.default")
    assert azure.AzureAuthConfig.from_env().scopes == ("api://env/.default",)


def test_wrong_typed_scopes_degrade(_isolated):
    _set(_isolated, {"azure": {"scopes": "api://x/.default"}})
    assert azure.AzureAuthConfig.from_env().scopes == ()


def test_non_string_scope_entries_degrade(_isolated):
    _set(_isolated, {"azure": {"scopes": ["api://x/.default", 7]}})
    assert azure.AzureAuthConfig.from_env().scopes == ()


def test_flow_can_be_pinned(_isolated):
    _set(_isolated, {"azure": {"client_id": "c", "flow": "device_code"}})
    cfg = azure.AzureAuthConfig.from_env()
    assert cfg.flow == azure.FLOW_DEVICE_CODE
    assert cfg.auth_flow == azure.FLOW_DEVICE_CODE


def test_pinned_flow_beats_inference(monkeypatch, _isolated):
    # A secret would otherwise infer client_credentials.
    _set(_isolated, {"azure": {"flow": "device_code"}})
    monkeypatch.setenv("AZURE_CLIENT_ID", "c")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "s")
    assert azure.AzureAuthConfig.from_env().auth_flow == azure.FLOW_DEVICE_CODE


def test_unknown_flow_degrades_to_inference(_isolated):
    _set(_isolated, {"azure": {"client_id": "c", "flow": "saml"}})
    cfg = azure.AzureAuthConfig.from_env()
    assert cfg.flow is None
    assert cfg.auth_flow == azure.FLOW_DEVICE_CODE


def test_certificate_infers_client_credentials(monkeypatch, _isolated):
    monkeypatch.setenv("AZURE_CLIENT_ID", "c")
    monkeypatch.setenv("AZURE_CLIENT_CERTIFICATE_PATH", "/certs/sp.pem")
    monkeypatch.setenv("AZURE_CLIENT_CERTIFICATE_THUMBPRINT", "AABB")
    cfg = azure.AzureAuthConfig.from_env()
    assert cfg.has_certificate
    assert cfg.auth_flow == azure.FLOW_CLIENT_CREDENTIALS


def test_half_a_certificate_is_not_a_credential(monkeypatch, _isolated):
    monkeypatch.setenv("AZURE_CLIENT_ID", "c")
    monkeypatch.setenv("AZURE_CLIENT_CERTIFICATE_PATH", "/certs/sp.pem")
    cfg = azure.AzureAuthConfig.from_env()
    assert not cfg.has_certificate
    assert cfg.auth_flow == azure.FLOW_DEVICE_CODE


def test_client_id_alone_infers_device_code(monkeypatch, _isolated):
    monkeypatch.setenv("AZURE_CLIENT_ID", "c")
    assert azure.AzureAuthConfig.from_env().auth_flow == azure.FLOW_DEVICE_CODE


def test_secret_never_in_repr():
    text = repr(_sp_config(client_secret="hunter2"))
    assert "hunter2" not in text
    assert "c-1" in text


# ---------------------------------------------------------------------------
# Token acquisition (injected fake app)
# ---------------------------------------------------------------------------


def test_get_token_client_credentials():
    app = _FakeConfidentialApp()
    client = azure.AzureAuthClient(_sp_config(), app=app)
    assert client.get_token() == "tok-1"
    assert app.calls == [_SCOPES]


def test_configured_scopes_used_by_default():
    app = _FakeConfidentialApp()
    cfg = _sp_config(scopes=("api://configured/.default",))
    azure.AzureAuthClient(cfg, app=app).get_token()
    assert app.calls == [("api://configured/.default",)]


def test_explicit_scopes_beat_configured():
    app = _FakeConfidentialApp()
    client = azure.AzureAuthClient(_sp_config(), app=app)
    client.get_token(("api://other/.default",))
    assert app.calls == [("api://other/.default",)]


def test_token_is_cached_per_scope_set():
    app = _FakeConfidentialApp()
    client = azure.AzureAuthClient(_sp_config(), app=app)
    assert client.get_token() == "tok-1"
    assert client.get_token() == "tok-1"  # served from cache, no second call
    assert len(app.calls) == 1
    assert client.get_token(("api://other/.default",)) == "tok-2"
    assert len(app.calls) == 2


def test_near_expiry_token_is_not_cached():
    # A token with 30s left is inside the skew window, so it must be re-acquired
    # rather than handed out for a call that could outlive it. The literal is
    # deliberate: deriving it from _EXPIRY_SKEW would make this test move with
    # the constant and assert nothing about the size of the window.
    app = _FakeConfidentialApp(expires_in=30)
    client = azure.AzureAuthClient(_sp_config(), app=app)
    assert client.get_token() == "tok-1"
    assert client.get_token() == "tok-2"
    assert len(app.calls) == 2


def test_long_lived_token_is_cached():
    # The other side of the skew window: comfortably valid, so no re-acquire.
    app = _FakeConfidentialApp(expires_in=3600)
    client = azure.AzureAuthClient(_sp_config(), app=app)
    assert client.get_token() == "tok-1"
    assert client.get_token() == "tok-1"
    assert len(app.calls) == 1


def test_no_scopes_raises():
    client = azure.AzureAuthClient(_sp_config(scopes=()), app=_FakeConfidentialApp())
    with pytest.raises(azure.AzureAuthError, match="no Azure scopes requested"):
        client.get_token()


def test_msal_in_band_error_raises():
    app = _FakeConfidentialApp(
        result={"error": "invalid_client", "error_description": "bad secret"}
    )
    client = azure.AzureAuthClient(_sp_config(), app=app)
    with pytest.raises(azure.AzureAuthError, match="invalid_client: bad secret"):
        client.get_token()


def test_unreachable_authority_raises():
    class _Boom:
        def acquire_token_for_client(self, scopes):
            raise OSError("dns failure")

    client = azure.AzureAuthClient(_sp_config(), app=_Boom())
    with pytest.raises(azure.AzureAuthError, match="could not reach Entra ID"):
        client.get_token()


def test_missing_expires_in_still_caches_conservatively():
    app = _FakeConfidentialApp(result={"access_token": "tok"})
    client = azure.AzureAuthClient(_sp_config(), app=app)
    assert client.get_token() == "tok"
    assert client.get_token() == "tok"  # re-acquired, not trusted for longer
    assert len(app.calls) == 2


# ---------------------------------------------------------------------------
# Device-code flow
# ---------------------------------------------------------------------------


def _device_config():
    return azure.AzureAuthConfig(tenant_id="t-1", client_id="c-1", scopes=_SCOPES)


def test_device_code_flow_acquires_token(caplog):
    app = _FakePublicApp()
    client = azure.AzureAuthClient(_device_config(), app=app)
    with caplog.at_level("WARNING", logger="app_config.azure"):
        assert client.get_token() == "device-tok"
    assert app.initiated
    # The user cannot complete the login without seeing the code.
    assert "ABC-123" in caplog.text


def test_device_code_prefers_a_silent_token():
    app = _FakePublicApp(
        accounts=[{"username": "u"}],
        silent={"access_token": "silent-tok", "expires_in": 3600},
    )
    client = azure.AzureAuthClient(_device_config(), app=app)
    assert client.get_token() == "silent-tok"
    assert not app.initiated  # no interactive prompt when a cached account works


def test_device_code_falls_through_when_silent_fails():
    app = _FakePublicApp(accounts=[{"username": "u"}], silent=None)
    client = azure.AzureAuthClient(_device_config(), app=app)
    assert client.get_token() == "device-tok"
    assert app.initiated


def test_device_flow_initiation_failure_raises():
    app = _FakePublicApp(
        flow={"error": "invalid_client", "error_description": "unknown app"}
    )
    client = azure.AzureAuthClient(_device_config(), app=app)
    with pytest.raises(azure.AzureAuthError, match="device-code flow"):
        client.get_token()


# ---------------------------------------------------------------------------
# Build-time config validation (runs before the msal import)
# ---------------------------------------------------------------------------


def test_missing_tenant_raises():
    client = azure.AzureAuthClient(azure.AzureAuthConfig(client_id="c"))
    with pytest.raises(azure.AzureAuthError, match="no Azure tenant"):
        _ = client.app


def test_missing_client_id_raises():
    client = azure.AzureAuthClient(azure.AzureAuthConfig(tenant_id="t"))
    with pytest.raises(azure.AzureAuthError, match="no Azure client id"):
        _ = client.app


def test_pinned_client_credentials_without_credential_raises():
    cfg = azure.AzureAuthConfig(
        tenant_id="t", client_id="c", flow=azure.FLOW_CLIENT_CREDENTIALS
    )
    with pytest.raises(azure.AzureAuthError, match="needs a credential"):
        _ = azure.AzureAuthClient(cfg).app


def test_missing_msal_raises_helpful_error():
    # msal is an optional extra; when it is not installed a fully-configured
    # client still fails at import with an install hint (skip if it is present).
    try:
        import msal  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("msal is installed; the import-error path is unreachable")
    with pytest.raises(azure.AzureAuthError, match="msal is required"):
        _ = azure.AzureAuthClient(_sp_config()).app


# ---------------------------------------------------------------------------
# Certificate credential
# ---------------------------------------------------------------------------


def test_certificate_credential_reads_the_key(tmp_path):
    pem = tmp_path / "sp.pem"
    pem.write_text("-----BEGIN PRIVATE KEY-----\nkey\n", encoding="utf-8")
    cfg = azure.AzureAuthConfig(
        tenant_id="t",
        client_id="c",
        certificate_path=str(pem),
        certificate_thumbprint="AABB",
    )
    credential = azure._client_credential(cfg)
    assert credential["thumbprint"] == "AABB"
    assert "BEGIN PRIVATE KEY" in credential["private_key"]


def test_secret_beats_certificate(tmp_path):
    cfg = _sp_config(certificate_path=str(tmp_path / "nope.pem"), certificate_thumbprint="A")
    assert azure._client_credential(cfg) == "s-1"


def test_unreadable_certificate_raises(tmp_path):
    cfg = azure.AzureAuthConfig(
        tenant_id="t",
        client_id="c",
        certificate_path=str(tmp_path / "absent.pem"),
        certificate_thumbprint="AABB",
    )
    with pytest.raises(azure.AzureAuthError, match="could not read Azure certificate"):
        azure._client_credential(cfg)


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------


def test_get_azure_client_is_cached():
    first = azure.get_azure_client()
    assert azure.get_azure_client() is first
    azure.clear_cache()
    assert azure.get_azure_client() is not first


# ---------------------------------------------------------------------------
# Service principal pinned to an explicit identity
# ---------------------------------------------------------------------------


def test_for_principal_pins_the_identity(monkeypatch, _isolated):
    # The environment supplies an unrelated identity and a sovereign-cloud
    # authority; only the identity should be overridden.
    monkeypatch.setenv("AZURE_TENANT_ID", "env-tenant")
    monkeypatch.setenv("AZURE_CLIENT_ID", "env-client")
    monkeypatch.setenv("AZURE_AUTHORITY_HOST", "https://login.microsoftonline.us")
    cfg = azure.AzureAuthConfig.for_principal("t-pin", "c-pin", "s-pin")
    assert (cfg.tenant_id, cfg.client_id, cfg.client_secret) == (
        "t-pin",
        "c-pin",
        "s-pin",
    )
    assert cfg.authority_host == "https://login.microsoftonline.us"
    assert cfg.auth_flow == azure.FLOW_CLIENT_CREDENTIALS


def test_for_principal_clears_certificate_fields(monkeypatch, _isolated):
    monkeypatch.setenv("AZURE_CLIENT_CERTIFICATE_PATH", "/tmp/x.pem")
    monkeypatch.setenv("AZURE_CLIENT_CERTIFICATE_THUMBPRINT", "AABB")
    cfg = azure.AzureAuthConfig.for_principal("t", "c", "s")
    # The pinned secret must be unambiguously the credential.
    assert not cfg.has_certificate
    assert azure._client_credential(cfg) == "s"


def test_client_for_principal_is_cached_per_identity(_isolated):
    first = azure.get_client_for_principal("t-1", "c-1", "s-1")
    assert azure.get_client_for_principal("t-1", "c-1", "s-1") is first
    assert azure.get_client_for_principal("t-1", "c-2", "s-1") is not first
    azure.clear_cache()
    assert azure.get_client_for_principal("t-1", "c-1", "s-1") is not first


# ---------------------------------------------------------------------------
# Service principal sourced from Databricks secrets
# ---------------------------------------------------------------------------


def _databricks_secret_scope(monkeypatch, principal=("kv-t", "kv-c", "kv-s")) -> None:
    """A workspace whose secret scope holds the service principal."""
    monkeypatch.setenv("DATABRICKS_HOST", "https://adb-1.azuredatabricks.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-boot")
    monkeypatch.setenv("DATABRICKS_SECRET_SCOPE", "kv")
    tenant, client, secret = principal
    values = {
        databricks.SECRET_KEY_TENANT_ID: tenant,
        databricks.SECRET_KEY_CLIENT_ID: client,
        databricks.SECRET_KEY_CLIENT_SECRET: secret,
    }
    monkeypatch.setattr(
        databricks,
        "read_workspace_secrets",
        lambda scope, keys, *, config=None: {key: values[key] for key in keys},
    )


def test_service_principal_read_from_databricks(monkeypatch, _isolated):
    _databricks_secret_scope(monkeypatch)
    principal = azure.databricks_service_principal()
    assert (principal.tenant_id, principal.client_id) == ("kv-t", "kv-c")
    assert principal.client_secret == "kv-s"


def test_databricks_client_is_pinned_to_that_principal(monkeypatch, _isolated):
    _databricks_secret_scope(monkeypatch)
    client = azure.get_databricks_client()
    assert (client.config.tenant_id, client.config.client_id) == ("kv-t", "kv-c")
    assert client.config.auth_flow == azure.FLOW_CLIENT_CREDENTIALS


def test_shared_client_falls_back_to_databricks_secrets(monkeypatch, _isolated):
    # No AZURE_* identity at all, so the Databricks-stored principal is the
    # only one available — this is the Vault-login path.
    _databricks_secret_scope(monkeypatch)
    client = azure.get_azure_client()
    assert client.config.client_id == "kv-c"
    # And it is the same instance the pinned accessor hands out.
    assert client is azure.get_databricks_client()


def test_azure_env_identity_beats_the_databricks_scope(monkeypatch, _isolated):
    _databricks_secret_scope(monkeypatch)
    monkeypatch.setenv("AZURE_TENANT_ID", "env-tenant")
    monkeypatch.setenv("AZURE_CLIENT_ID", "env-client")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "env-secret")
    assert azure.get_azure_client().config.client_id == "env-client"


def test_no_scope_leaves_the_environment_config_alone(_isolated):
    # Nothing configured anywhere: the plain env config stands, and its own
    # "no identity" error is what a caller should see.
    assert azure.get_azure_client().config.client_id is None


def test_unreadable_scope_surfaces_as_an_azure_error(monkeypatch, _isolated):
    monkeypatch.setenv("DATABRICKS_SECRET_SCOPE", "kv")
    # No workspace credential to read the scope with.
    with pytest.raises(
        azure.AzureAuthError, match="could not read the Entra ID service principal"
    ):
        azure.get_azure_client()
