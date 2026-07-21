"""
End-to-end test of the credential chain, with every network edge faked.

The links, in order:

1. The Entra ID service principal is read from a Databricks secret scope
   (``app_config.databricks``), bootstrapped by a PAT or the cluster runtime.
2. ``msal`` mints a JWT for that principal (``app_config.azure``).
3. That JWT logs in to Vault's ``jwt`` auth method (``app_config.vault``).
4. The Vault session reads the AI Gateway secret at ``appsvc/ai_gateway``.
5. The gateway token lands on ``llm_client.LLMClientConfig.api_key``.

The unit tests for each module cover the branches within it; what is checked
here is that the *seams* line up — that the principal read in step 1 is the
one msal signs in step 2, that its JWT is the string Vault receives in step 3,
and that the secret from step 4 is what the model is configured with in step 5.

Three fakes stand in for the three external systems: the Databricks SDK's
``WorkspaceClient``, msal's confidential-client app, and ``hvac.Client``.
Nothing here touches a network.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

import app_config
from app_config import azure, databricks, vault
from llm_client import LLMClientConfig

SPN_TENANT = "chain-tenant"
SPN_CLIENT = "chain-client"
SPN_SECRET = "chain-secret"
GATEWAY_TOKEN = "gw-token-from-vault"
MINTED_JWT = "jwt-signed-for-the-spn"

_ENV = (
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_SECRET_SCOPE",
    "DATABRICKS_RUNTIME_VERSION",
    "ARM_TENANT_ID",
    "ARM_CLIENT_ID",
    "ARM_CLIENT_SECRET",
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_SCOPES",
    "VAULT_ADDR",
    "VAULT_TOKEN",
    "VAULT_ROLE_ID",
    "VAULT_SECRET_ID",
    "VAULT_OIDC_ROLE",
    "VAULT_AUTH_PATH",
    "VAULT_AZURE_SCOPES",
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Empty config file, none of the chain's env vars, every cache cleared."""
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(app_config.ENV_VAR, str(cfg))
    for var in _ENV:
        monkeypatch.delenv(var, raising=False)
    for module in (app_config, azure, databricks, vault):
        module.clear_cache()
    yield cfg
    for module in (app_config, azure, databricks, vault):
        module.clear_cache()


# ---------------------------------------------------------------------------
# The three external systems
# ---------------------------------------------------------------------------


class _FakeDatabricks:
    """WorkspaceClient whose dbutils.secrets serves the service principal."""

    instances: list["_FakeDatabricks"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeDatabricks.instances.append(self)
        secrets: Any = type("Secrets", (), {})()
        secrets.get = self._get
        dbutils: Any = type("DbUtils", (), {})()
        dbutils.secrets = secrets
        self.dbutils = dbutils

    @staticmethod
    def _get(scope, key):
        values = {
            databricks.SECRET_KEY_TENANT_ID: SPN_TENANT,
            databricks.SECRET_KEY_CLIENT_ID: SPN_CLIENT,
            databricks.SECRET_KEY_CLIENT_SECRET: SPN_SECRET,
        }
        if scope != "kv" or key not in values:
            raise RuntimeError(f"no secret {key} in {scope}")
        return values[key]


class _FakeMsalApp:
    """Confidential-client app that signs a JWT for whatever it was built with."""

    def __init__(self, config):
        self.config = config
        self.scopes: list[list[str]] = []

    def acquire_token_for_client(self, scopes):
        self.scopes.append(list(scopes))
        return {"access_token": MINTED_JWT, "expires_in": 3600}


class _FakeVault:
    """hvac.Client recording the JWT login and serving the gateway secret."""

    instances: list["_FakeVault"] = []
    # Mutable so a test can rotate the stored secret under a live client.
    payload: dict[str, str] = {
        "token": GATEWAY_TOKEN,
        "base_url": "https://gw.example/v1",
    }

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.logins: list[dict] = []
        _FakeVault.instances.append(self)

        jwt_auth: Any = type("Jwt", (), {})()
        jwt_auth.jwt_login = self._jwt_login
        auth: Any = type("Auth", (), {})()
        auth.jwt = jwt_auth
        self.auth = auth

        kv_v2: Any = type("KvV2", (), {})()
        kv_v2.read_secret_version = self._read
        kv: Any = type("Kv", (), {})()
        kv.v2 = kv_v2
        secrets: Any = type("Secrets", (), {})()
        secrets.kv = kv
        self.secrets = secrets

    def _jwt_login(self, role, jwt, path):
        self.logins.append({"role": role, "jwt": jwt, "path": path})
        return {"auth": {"client_token": "s.vault-session"}}

    @classmethod
    def _read(cls, path, mount_point, raise_on_deleted_version):
        if (mount_point, path) != ("secret", vault.AI_GATEWAY_PATH):
            raise RuntimeError(f"no secret at {mount_point}/{path}")
        return {"data": {"data": dict(cls.payload)}}

    def is_authenticated(self):
        return bool(self.logins)


@pytest.fixture
def chain(monkeypatch):
    """Wire the whole chain up against the three fakes."""
    _FakeDatabricks.instances.clear()
    _FakeVault.instances.clear()
    monkeypatch.setattr(
        _FakeVault,
        "payload",
        {"token": GATEWAY_TOKEN, "base_url": "https://gw.example/v1"},
    )

    # 1. A workspace whose secret scope holds the principal, reachable with a PAT.
    monkeypatch.setenv("DATABRICKS_HOST", "https://adb-1.azuredatabricks.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-bootstrap")
    monkeypatch.setenv("DATABRICKS_SECRET_SCOPE", "kv")
    sdk = pytest.importorskip("databricks.sdk", reason="databricks-sdk is not installed")
    monkeypatch.setattr(sdk, "WorkspaceClient", _FakeDatabricks)

    # 2. msal, stubbed at the app-construction seam so config resolution,
    #    caching and the token flow above it all stay real.
    monkeypatch.setattr(
        azure.AzureAuthClient, "_build_app", staticmethod(_FakeMsalApp)
    )

    # 3. Vault, reached with the jwt auth method.
    monkeypatch.setenv("VAULT_ADDR", "https://vault.example:8200")
    monkeypatch.setenv("VAULT_OIDC_ROLE", "sas-parser")
    hvac = pytest.importorskip("hvac", reason="hvac is not installed")
    monkeypatch.setattr(hvac, "Client", _FakeVault)


# ---------------------------------------------------------------------------
# The chain, link by link
# ---------------------------------------------------------------------------


def test_gateway_token_reaches_the_llm_config(chain):
    config = LLMClientConfig.from_ai_gateway(model="some-model")
    assert config.api_key is not None
    assert config.api_key.get_secret_value() == GATEWAY_TOKEN
    assert config.base_url == "https://gw.example/v1"


def test_the_principal_comes_from_the_databricks_scope(chain):
    LLMClientConfig.from_ai_gateway()
    # The msal app was built for the principal in the scope, not from AZURE_*.
    app = azure.get_azure_client().app
    assert (app.config.tenant_id, app.config.client_id) == (SPN_TENANT, SPN_CLIENT)
    assert app.config.client_secret == SPN_SECRET


def test_the_scope_read_bootstraps_off_the_pat(chain):
    LLMClientConfig.from_ai_gateway()
    # Exactly one workspace client, authenticated by the PAT — the principal it
    # fetches is never what authenticates the fetch.
    assert len(_FakeDatabricks.instances) == 1
    assert _FakeDatabricks.instances[0].kwargs == {
        "host": "https://adb-1.azuredatabricks.net",
        "token": "dapi-bootstrap",
    }


def test_vault_receives_the_minted_jwt(chain):
    LLMClientConfig.from_ai_gateway()
    logins = _FakeVault.instances[0].logins
    assert len(logins) == 1
    assert logins[0] == {
        "role": "sas-parser",
        "jwt": MINTED_JWT,
        "path": vault.DEFAULT_AUTH_PATH,
    }


def test_jwt_audience_is_the_principals_own_app_id(chain):
    LLMClientConfig.from_ai_gateway()
    # With no scopes configured anywhere, the login JWT is requested for
    # "<client_id>/.default" — the audience a Vault role with
    # bound_audiences=<app id> expects.
    assert azure.get_azure_client().app.scopes == [[f"{SPN_CLIENT}/.default"]]


def test_the_chain_runs_once_across_repeated_use(chain):
    for _ in range(3):
        LLMClientConfig.from_ai_gateway()
    # One scope read, one msal app, one Vault login — the caches hold.
    assert len(_FakeDatabricks.instances) == 1
    assert len(_FakeVault.instances) == 1
    assert len(_FakeVault.instances[0].logins) == 1
    assert azure.get_azure_client().app.scopes == [[f"{SPN_CLIENT}/.default"]]


def test_rotated_gateway_token_is_picked_up_without_a_restart(chain, monkeypatch):
    first = LLMClientConfig.from_ai_gateway()
    assert first.api_key is not None
    assert first.api_key.get_secret_value() == GATEWAY_TOKEN

    monkeypatch.setattr(_FakeVault, "payload", {"token": "rotated"})
    # The Vault session is cached; the secret read is not.
    config = LLMClientConfig.from_ai_gateway()
    assert config.api_key is not None
    assert config.api_key.get_secret_value() == "rotated"


def test_arm_env_vars_take_over_from_the_scope(chain, monkeypatch):
    monkeypatch.setenv("ARM_TENANT_ID", "env-tenant")
    monkeypatch.setenv("ARM_CLIENT_ID", "env-client")
    monkeypatch.setenv("ARM_CLIENT_SECRET", "env-secret")
    databricks.clear_cache()
    azure.clear_cache()

    config = LLMClientConfig.from_ai_gateway()
    assert config.api_key is not None
    assert config.api_key.get_secret_value() == GATEWAY_TOKEN
    # The chain still completes, but nothing read the secret scope.
    assert _FakeDatabricks.instances == []
    assert azure.get_azure_client().app.config.client_id == "env-client"
