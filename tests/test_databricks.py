"""
Tests for the app_config.databricks workspace config.

No live workspace (and no databricks-sdk install) is needed: settings are
resolved from a controlled environment + tmp config.json, and the azure-ad
credential path is exercised by monkeypatching app_config.azure.get_token —
which databricks imports lazily, inside the call, precisely so it can be
swapped like this. Each test isolates SAS_PARSER_CONFIG, the DATABRICKS_* and
AZURE_* env vars (auth_method consults both), and clears the app_config file
cache plus the databricks config cache around itself.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

import app_config
from app_config import azure, databricks

_DATABRICKS_ENV = (
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_HTTP_PATH",
    "DATABRICKS_WAREHOUSE_ID",
    "DATABRICKS_CLUSTER_ID",
    "DATABRICKS_CATALOG",
    "DATABRICKS_SCHEMA",
    "DATABRICKS_RUNTIME_VERSION",
    "DATABRICKS_AZURE_RESOURCE_ID",
    "DATABRICKS_SECRET_SCOPE",
    # Databricks' own names for the Entra service principal; distinct from the
    # AZURE_* identity that app_config.azure resolves.
    "ARM_TENANT_ID",
    "ARM_CLIENT_ID",
    "ARM_CLIENT_SECRET",
)

_AZURE_ENV = (
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_SCOPES",
    "AZURE_CLIENT_CERTIFICATE_PATH",
    "AZURE_CLIENT_CERTIFICATE_THUMBPRINT",
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Empty config file, no Databricks/Azure env vars, all caches cleared."""
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(app_config.ENV_VAR, str(cfg))
    for var in _DATABRICKS_ENV + _AZURE_ENV:
        monkeypatch.delenv(var, raising=False)
    app_config.clear_cache()
    databricks.clear_cache()
    azure.clear_cache()
    yield cfg
    app_config.clear_cache()
    databricks.clear_cache()
    azure.clear_cache()


def _set(cfg_path, mapping) -> None:
    cfg_path.write_text(json.dumps(mapping), encoding="utf-8")
    app_config.clear_cache()


def _service_principal(monkeypatch) -> None:
    """The env of a workspace reached with app_config.azure's shared identity."""
    monkeypatch.setenv("AZURE_TENANT_ID", "t-1")
    monkeypatch.setenv("AZURE_CLIENT_ID", "c-1")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "s-1")


def _arm_service_principal(monkeypatch) -> None:
    """The env of a workspace reached with a Databricks-style Entra SPN."""
    monkeypatch.setenv("ARM_TENANT_ID", "arm-tenant")
    monkeypatch.setenv("ARM_CLIENT_ID", "arm-client")
    monkeypatch.setenv("ARM_CLIENT_SECRET", "arm-secret")


class _FakeAzureClient:
    """Stands in for AzureAuthClient, recording the config it was pinned to."""

    last: "_FakeAzureClient | None" = None

    def __init__(self, config=None, *, app=None):
        self.config = config
        self.asked: list = []
        _FakeAzureClient.last = self

    def get_token(self, scopes=None):
        self.asked.append(scopes)
        return f"token-for:{scopes[0]}"


def _patched_azure_client(monkeypatch) -> type[_FakeAzureClient]:
    """Swap AzureAuthClient out; databricks imports it lazily, inside the call."""
    _FakeAzureClient.last = None
    monkeypatch.setattr(azure, "AzureAuthClient", _FakeAzureClient)
    return _FakeAzureClient


def _patched_secrets(monkeypatch, values: dict) -> dict:
    """A WorkspaceClient whose dbutils.secrets serves *values* keyed by (scope, key)."""
    sdk = pytest.importorskip("databricks.sdk", reason="databricks-sdk is not installed")
    built: dict = {}

    class _Secrets:
        @staticmethod
        def get(scope, key):
            try:
                return values[(scope, key)]
            except KeyError:
                raise RuntimeError(f"no secret {key} in {scope}") from None

    class _DbUtils:
        secrets = _Secrets()

    class _FakeWorkspaceClient:
        def __init__(self, **kwargs):
            built.update(kwargs)
            self.dbutils = _DbUtils()

    monkeypatch.setattr(sdk, "WorkspaceClient", _FakeWorkspaceClient)
    return built


# ---------------------------------------------------------------------------
# DatabricksConfig resolution
# ---------------------------------------------------------------------------


def test_from_env_reads_env_first(monkeypatch, _isolated):
    monkeypatch.setenv("DATABRICKS_HOST", "https://adb-1.azuredatabricks.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-1")
    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "w-1")
    monkeypatch.setenv("DATABRICKS_CATALOG", "main")
    monkeypatch.setenv("DATABRICKS_SCHEMA", "sas")
    cfg = databricks.DatabricksConfig.from_env()
    assert cfg.host == "https://adb-1.azuredatabricks.net"
    assert cfg.token == "dapi-1"
    assert cfg.warehouse_id == "w-1"
    assert (cfg.catalog, cfg.schema) == ("main", "sas")
    assert cfg.auth_method == databricks.AUTH_PAT


def test_from_env_falls_back_to_config_json(_isolated):
    _set(
        _isolated,
        {
            "databricks": {
                "host": "https://cfg.azuredatabricks.net",
                "warehouse_id": "w-cfg",
                "catalog": "cfg_cat",
                "schema": "cfg_sch",
                "timeout": 5,
            }
        },
    )
    cfg = databricks.DatabricksConfig.from_env()
    assert cfg.host == "https://cfg.azuredatabricks.net"
    assert cfg.warehouse_id == "w-cfg"
    assert (cfg.catalog, cfg.schema) == ("cfg_cat", "cfg_sch")
    assert cfg.timeout == 5


def test_env_host_beats_config(monkeypatch, _isolated):
    _set(_isolated, {"databricks": {"host": "https://cfg.azuredatabricks.net"}})
    monkeypatch.setenv("DATABRICKS_HOST", "https://env.azuredatabricks.net")
    assert (
        databricks.DatabricksConfig.from_env().host
        == "https://env.azuredatabricks.net"
    )


def test_defaults_without_env_or_config(_isolated):
    cfg = databricks.DatabricksConfig.from_env()
    assert cfg.host is None
    assert cfg.sql_http_path is None
    assert cfg.catalog is None and cfg.schema is None
    assert cfg.timeout == databricks.DEFAULT_TIMEOUT
    assert cfg.auth_method is None


def test_wrong_typed_timeout_degrades(_isolated):
    _set(_isolated, {"databricks": {"timeout": "slow"}})
    assert databricks.DatabricksConfig.from_env().timeout == databricks.DEFAULT_TIMEOUT


def test_token_never_in_repr():
    cfg = databricks.DatabricksConfig(host="https://adb-1.net", token="dapi-SECRET")
    text = repr(cfg)
    assert "dapi-SECRET" not in text
    assert "https://adb-1.net" in text


# ---------------------------------------------------------------------------
# Host normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        # The workspace UI shows a bare hostname; the SDK needs a URL.
        ("adb-1.azuredatabricks.net", "https://adb-1.azuredatabricks.net"),
        ("https://adb-1.azuredatabricks.net", "https://adb-1.azuredatabricks.net"),
        ("https://adb-1.azuredatabricks.net/", "https://adb-1.azuredatabricks.net"),
        ("  adb-1.azuredatabricks.net  ", "https://adb-1.azuredatabricks.net"),
        ("http://localhost:8080", "http://localhost:8080"),
        (None, None),
        ("", None),
        ("   ", None),
    ],
)
def test_host_normalisation(raw, expected):
    assert databricks._normalise_host(raw) == expected


# ---------------------------------------------------------------------------
# auth_method
# ---------------------------------------------------------------------------


def test_notebook_auth_on_a_cluster(monkeypatch, _isolated):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
    cfg = databricks.DatabricksConfig.from_env()
    assert databricks.in_databricks_runtime()
    assert cfg.auth_method == databricks.AUTH_NOTEBOOK
    # The runtime authenticates itself; there is no token to hand out.
    assert cfg.get_token() is None


def test_notebook_auth_beats_a_set_token(monkeypatch, _isolated):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-1")
    assert databricks.DatabricksConfig.from_env().auth_method == databricks.AUTH_NOTEBOOK


def test_pat_auth(monkeypatch, _isolated):
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-1")
    cfg = databricks.DatabricksConfig.from_env()
    assert cfg.auth_method == databricks.AUTH_PAT
    assert cfg.get_token() == "dapi-1"


def test_azure_ad_auth_when_a_service_principal_is_configured(monkeypatch, _isolated):
    _service_principal(monkeypatch)
    assert (
        databricks.DatabricksConfig.from_env().auth_method == databricks.AUTH_AZURE_AD
    )


def test_pat_beats_azure_ad(monkeypatch, _isolated):
    _service_principal(monkeypatch)
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-1")
    assert databricks.DatabricksConfig.from_env().auth_method == databricks.AUTH_PAT


def test_device_code_identity_is_not_azure_ad_auth(monkeypatch, _isolated):
    # A public client (no secret) would block on an interactive login, so it
    # does not count as a usable workspace credential.
    monkeypatch.setenv("AZURE_TENANT_ID", "t-1")
    monkeypatch.setenv("AZURE_CLIENT_ID", "c-1")
    assert databricks.DatabricksConfig.from_env().auth_method is None


def test_no_credentials_raises(_isolated):
    with pytest.raises(databricks.DatabricksError, match="no Databricks credentials"):
        databricks.DatabricksConfig.from_env().get_token()


def test_azure_ad_token_requests_the_databricks_scope(monkeypatch, _isolated):
    _service_principal(monkeypatch)
    asked: list = []

    def _fake_get_token(scopes=None):
        asked.append(scopes)
        return "aad-token"

    monkeypatch.setattr(azure, "get_token", _fake_get_token)
    cfg = databricks.DatabricksConfig.from_env()
    assert cfg.get_token() == "aad-token"
    assert asked == [(databricks.AZURE_DATABRICKS_SCOPE,)]


def test_azure_login_failure_surfaces_as_databricks_error(monkeypatch, _isolated):
    _service_principal(monkeypatch)

    def _boom(scopes=None):
        raise azure.AzureAuthError("bad secret")

    monkeypatch.setattr(azure, "get_token", _boom)
    cfg = databricks.DatabricksConfig.from_env()
    with pytest.raises(databricks.DatabricksError, match="could not mint an Entra ID"):
        cfg.get_token()


# ---------------------------------------------------------------------------
# Azure service principal: resolution
# ---------------------------------------------------------------------------


def test_arm_env_vars_resolve_the_service_principal(monkeypatch, _isolated):
    _arm_service_principal(monkeypatch)
    monkeypatch.setenv("DATABRICKS_AZURE_RESOURCE_ID", "/subscriptions/s/rg/ws")
    monkeypatch.setenv("DATABRICKS_SECRET_SCOPE", "kv")
    cfg = databricks.DatabricksConfig.from_env()
    assert (cfg.azure_tenant_id, cfg.azure_client_id) == ("arm-tenant", "arm-client")
    assert cfg.azure_client_secret == "arm-secret"
    assert cfg.azure_resource_id == "/subscriptions/s/rg/ws"
    assert cfg.secret_scope == "kv"
    assert cfg.has_service_principal


def test_service_principal_falls_back_to_config_json(_isolated):
    _set(
        _isolated,
        {
            "databricks": {
                "azure_tenant_id": "cfg-tenant",
                "azure_client_id": "cfg-client",
                "azure_workspace_resource_id": "/subscriptions/cfg",
                "secret_scope": "cfg-scope",
            }
        },
    )
    cfg = databricks.DatabricksConfig.from_env()
    assert (cfg.azure_tenant_id, cfg.azure_client_id) == ("cfg-tenant", "cfg-client")
    assert cfg.azure_resource_id == "/subscriptions/cfg"
    assert cfg.secret_scope == "cfg-scope"
    # The secret is never read from config.json, so the principal is incomplete.
    assert cfg.azure_client_secret is None
    assert not cfg.has_service_principal


def test_client_secret_never_in_repr():
    cfg = databricks.DatabricksConfig(
        host="https://adb-1.net",
        azure_tenant_id="t",
        azure_client_id="c",
        azure_client_secret="arm-SECRET",
    )
    assert "arm-SECRET" not in repr(cfg)


def test_service_principal_from_local_config_reads_no_secrets(_isolated):
    cfg = databricks.DatabricksConfig(
        azure_tenant_id="t-arm",
        azure_client_id="c-arm",
        azure_client_secret="s-arm",
        # A scope is set too, but the complete local principal wins outright —
        # so this must not attempt a workspace call (there is no client here).
        secret_scope="kv",
    )
    principal = cfg.service_principal()
    assert principal == databricks.AzureServicePrincipal("t-arm", "c-arm", "s-arm")


def test_service_principal_read_from_the_secret_scope(monkeypatch, _isolated):
    built = _patched_secrets(
        monkeypatch,
        {
            ("kv", databricks.SECRET_KEY_TENANT_ID): "kv-tenant",
            ("kv", databricks.SECRET_KEY_CLIENT_ID): "kv-client",
            ("kv", databricks.SECRET_KEY_CLIENT_SECRET): "kv-secret",
        },
    )
    cfg = databricks.DatabricksConfig(
        host="https://adb-1.net", token="dapi-boot", secret_scope="kv"
    )
    assert cfg.service_principal() == databricks.AzureServicePrincipal(
        "kv-tenant", "kv-client", "kv-secret"
    )
    # The scope read bootstraps off the PAT, never off the principal it fetches.
    assert built == {"host": "https://adb-1.net", "token": "dapi-boot"}


def test_all_three_keys_share_one_client(monkeypatch, _isolated):
    clients: list = []
    sdk = pytest.importorskip("databricks.sdk", reason="databricks-sdk is not installed")

    class _Counting:
        def __init__(self, **kwargs):
            clients.append(kwargs)
            secrets = type("S", (), {"get": staticmethod(lambda scope, key: f"v-{key}")})
            self.dbutils = type("D", (), {"secrets": secrets()})()

    monkeypatch.setattr(sdk, "WorkspaceClient", _Counting)
    cfg = databricks.DatabricksConfig(
        host="https://adb-1.net", token="dapi-boot", secret_scope="kv"
    )
    cfg.service_principal()
    # Building a client authenticates; three keys must not mean three logins.
    assert len(clients) == 1


def test_service_principal_is_cached(monkeypatch, _isolated):
    calls: list = []

    def _counting_read(scope, keys, *, config=None):
        calls.append(tuple(keys))
        return {key: f"{key}-value" for key in keys}

    monkeypatch.setattr(databricks, "read_workspace_secrets", _counting_read)
    cfg = databricks.DatabricksConfig(
        host="https://adb-1.net", token="dapi-boot", secret_scope="kv"
    )
    first = cfg.service_principal()
    assert cfg.service_principal() is first
    assert len(calls) == 1  # resolved once, not once per access


def test_secret_scope_read_on_a_cluster_needs_no_pat(monkeypatch, _isolated):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
    built = _patched_secrets(monkeypatch, {("kv", "sp-hsv-appid"): "kv-client"})
    cfg = databricks.DatabricksConfig(secret_scope="kv")
    got = databricks.read_workspace_secret("kv", "sp-hsv-appid", config=cfg)
    assert got == "kv-client"
    # The runtime authenticates itself; passing host or token would override it.
    assert built == {}


def test_secret_scope_without_a_bootstrap_credential_raises(_isolated):
    cfg = databricks.DatabricksConfig(host="https://adb-1.net", secret_scope="kv")
    with pytest.raises(
        databricks.DatabricksError, match="does not come from the scope"
    ):
        cfg.service_principal()


def test_missing_secret_surfaces_as_databricks_error(monkeypatch, _isolated):
    _patched_secrets(monkeypatch, {})
    cfg = databricks.DatabricksConfig(
        host="https://adb-1.net", token="dapi-boot", secret_scope="kv"
    )
    with pytest.raises(databricks.DatabricksError, match="could not read secret"):
        cfg.service_principal()


def test_no_principal_anywhere_raises(_isolated):
    with pytest.raises(
        databricks.DatabricksError, match="no Azure service principal configured"
    ):
        databricks.DatabricksConfig(host="https://adb-1.net").service_principal()


# ---------------------------------------------------------------------------
# Azure service principal: auth_method and tokens
# ---------------------------------------------------------------------------


def test_azure_sp_auth_method(monkeypatch, _isolated):
    _arm_service_principal(monkeypatch)
    assert (
        databricks.DatabricksConfig.from_env().auth_method == databricks.AUTH_AZURE_SP
    )


def test_azure_sp_beats_the_shared_azure_ad_identity(monkeypatch, _isolated):
    _service_principal(monkeypatch)
    _arm_service_principal(monkeypatch)
    assert (
        databricks.DatabricksConfig.from_env().auth_method == databricks.AUTH_AZURE_SP
    )


def test_pat_beats_azure_sp(monkeypatch, _isolated):
    _arm_service_principal(monkeypatch)
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-1")
    assert databricks.DatabricksConfig.from_env().auth_method == databricks.AUTH_PAT


def test_incomplete_arm_principal_is_not_azure_sp(monkeypatch, _isolated):
    # A tenant and client id with no secret cannot log in.
    monkeypatch.setenv("ARM_TENANT_ID", "arm-tenant")
    monkeypatch.setenv("ARM_CLIENT_ID", "arm-client")
    assert databricks.DatabricksConfig.from_env().auth_method is None


def test_a_scope_alone_is_not_azure_sp(monkeypatch, _isolated):
    # Reading the scope needs a credential of its own, so a principal that
    # exists only there cannot also be what authenticates the read.
    monkeypatch.setenv("DATABRICKS_SECRET_SCOPE", "kv")
    assert databricks.DatabricksConfig.from_env().auth_method is None


def test_unreachable_scope_falls_back_to_the_shared_identity(monkeypatch, _isolated):
    # A scope with nothing to authenticate the read of it is not usable, so
    # the shared AZURE_* identity gets its turn rather than a hard failure.
    _service_principal(monkeypatch)
    monkeypatch.setenv("DATABRICKS_SECRET_SCOPE", "kv")
    monkeypatch.setattr(azure, "get_token", lambda scopes=None: "shared-token")
    cfg = databricks.DatabricksConfig.from_env()
    assert cfg.auth_method == databricks.AUTH_AZURE_AD
    assert cfg.get_token() == "shared-token"


def test_azure_sp_token_uses_the_pinned_principal(monkeypatch, _isolated):
    _arm_service_principal(monkeypatch)
    _patched_azure_client(monkeypatch)
    cfg = databricks.DatabricksConfig.from_env()
    assert cfg.get_token() == f"token-for:{databricks.AZURE_DATABRICKS_SCOPE}"
    pinned = _FakeAzureClient.last.config
    # Pinned to this config's principal, not to app_config.azure's AZURE_*.
    assert (pinned.tenant_id, pinned.client_id) == ("arm-tenant", "arm-client")
    assert pinned.client_secret == "arm-secret"
    assert pinned.flow == azure.FLOW_CLIENT_CREDENTIALS


def test_azure_sp_login_failure_surfaces_as_databricks_error(monkeypatch, _isolated):
    _arm_service_principal(monkeypatch)

    class _Failing(_FakeAzureClient):
        def get_token(self, scopes=None):
            raise azure.AzureAuthError("bad secret")

    monkeypatch.setattr(azure, "AzureAuthClient", _Failing)
    cfg = databricks.DatabricksConfig.from_env()
    with pytest.raises(databricks.DatabricksError, match="could not mint an Entra ID"):
        cfg.get_token()


def test_no_credentials_error_names_the_arm_vars(_isolated):
    with pytest.raises(databricks.DatabricksError, match="ARM_TENANT_ID"):
        databricks.DatabricksConfig.from_env().get_token()


# ---------------------------------------------------------------------------
# workspace_headers (the azure_resource_id path)
# ---------------------------------------------------------------------------


def test_no_headers_without_a_resource_id(monkeypatch, _isolated):
    _arm_service_principal(monkeypatch)
    assert databricks.DatabricksConfig.from_env().workspace_headers() == {}


def test_resource_id_headers_carry_a_management_token(monkeypatch, _isolated):
    _arm_service_principal(monkeypatch)
    _patched_azure_client(monkeypatch)
    cfg = databricks.DatabricksConfig.from_env()
    cfg.azure_resource_id = "/subscriptions/s/rg/ws"
    assert cfg.workspace_headers() == {
        databricks.WORKSPACE_RESOURCE_ID_HEADER: "/subscriptions/s/rg/ws",
        databricks.SP_MANAGEMENT_TOKEN_HEADER: (
            f"token-for:{databricks.AZURE_MANAGEMENT_SCOPE}"
        ),
    }


def test_management_scope_targets_the_arm_resource():
    # MSAL scopes are "<resource>/.default" and the resource ends in a slash,
    # so the doubled slash is deliberate — it is the audience Databricks wants.
    assert databricks.AZURE_MANAGEMENT_SCOPE == (
        "https://management.core.windows.net//.default"
    )


def test_resource_id_ignored_on_the_pat_path(monkeypatch, _isolated):
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-1")
    monkeypatch.setenv("DATABRICKS_AZURE_RESOURCE_ID", "/subscriptions/s/rg/ws")
    # Neither header means anything alongside a PAT.
    assert databricks.DatabricksConfig.from_env().workspace_headers() == {}


# ---------------------------------------------------------------------------
# sql_http_path
# ---------------------------------------------------------------------------


def test_http_path_derived_from_warehouse_id():
    cfg = databricks.DatabricksConfig(warehouse_id="w-1")
    assert cfg.sql_http_path == "/sql/1.0/warehouses/w-1"


def test_explicit_http_path_beats_warehouse_id():
    cfg = databricks.DatabricksConfig(http_path="/sql/custom", warehouse_id="w-1")
    assert cfg.sql_http_path == "/sql/custom"


def test_cluster_id_alone_yields_no_http_path():
    # A cluster path needs the workspace org id, which this config never sees.
    assert databricks.DatabricksConfig(cluster_id="c-1").sql_http_path is None


# ---------------------------------------------------------------------------
# full_table_name
# ---------------------------------------------------------------------------


def test_full_table_name_qualifies_with_catalog_and_schema():
    cfg = databricks.DatabricksConfig(catalog="main", schema="sas")
    assert cfg.full_table_name("mem") == "main.sas.mem"


def test_full_table_name_schema_only():
    cfg = databricks.DatabricksConfig(schema="sas")
    assert cfg.full_table_name("mem") == "sas.mem"


def test_full_table_name_catalog_only():
    cfg = databricks.DatabricksConfig(catalog="main")
    assert cfg.full_table_name("mem") == "main.mem"


def test_full_table_name_unqualified_when_nothing_configured():
    assert databricks.DatabricksConfig().full_table_name("mem") == "mem"


def test_full_table_name_leaves_a_qualified_name_alone():
    # Callers may pipe an already-qualified configured name through this.
    cfg = databricks.DatabricksConfig(catalog="main", schema="sas")
    assert cfg.full_table_name("other.cat.tbl") == "other.cat.tbl"


# ---------------------------------------------------------------------------
# sql_connect_params
# ---------------------------------------------------------------------------


def test_sql_connect_params():
    cfg = databricks.DatabricksConfig(
        host="https://adb-1.azuredatabricks.net",
        warehouse_id="w-1",
        catalog="main",
        schema="sas",
        token="dapi-1",
    )
    assert cfg.sql_connect_params() == {
        # The connector wants the bare hostname, not the URL.
        "server_hostname": "adb-1.azuredatabricks.net",
        "http_path": "/sql/1.0/warehouses/w-1",
        "access_token": "dapi-1",
        "catalog": "main",
        "schema": "sas",
    }


def test_sql_connect_params_omits_unset_namespace():
    cfg = databricks.DatabricksConfig(
        host="https://adb-1.azuredatabricks.net", warehouse_id="w-1", token="dapi-1"
    )
    params = cfg.sql_connect_params()
    assert "catalog" not in params and "schema" not in params


def test_sql_connect_params_without_host_raises():
    cfg = databricks.DatabricksConfig(warehouse_id="w-1", token="dapi-1")
    with pytest.raises(databricks.DatabricksError, match="no Databricks host"):
        cfg.sql_connect_params()


def test_sql_connect_params_without_a_path_raises():
    cfg = databricks.DatabricksConfig(host="https://adb-1.net", token="dapi-1")
    with pytest.raises(databricks.DatabricksError, match="no Databricks SQL path"):
        cfg.sql_connect_params()


def test_sql_connect_params_on_a_cluster_raises(monkeypatch, _isolated):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
    cfg = databricks.DatabricksConfig(
        host="https://adb-1.net", warehouse_id="w-1", token="dapi-1"
    )
    with pytest.raises(databricks.DatabricksError, match="outside Databricks"):
        cfg.sql_connect_params()


# ---------------------------------------------------------------------------
# get_workspace_client (validation runs before the SDK import)
# ---------------------------------------------------------------------------


def test_workspace_client_without_host_raises(_isolated):
    cfg = databricks.DatabricksConfig(token="dapi-1")
    with pytest.raises(databricks.DatabricksError, match="no Databricks host"):
        databricks.get_workspace_client(cfg)


def test_workspace_client_without_credentials_raises(_isolated):
    cfg = databricks.DatabricksConfig(host="https://adb-1.net")
    with pytest.raises(databricks.DatabricksError, match="no Databricks credentials"):
        databricks.get_workspace_client(cfg)


def _patched_sdk(monkeypatch):
    """Capture WorkspaceClient(**kwargs) instead of building a real client."""
    sdk = pytest.importorskip(
        "databricks.sdk", reason="databricks-sdk is not installed"
    )
    built: dict = {}

    class _FakeWorkspaceClient:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setattr(sdk, "WorkspaceClient", _FakeWorkspaceClient)
    return built


def test_workspace_client_forwards_host_and_token(monkeypatch, _isolated):
    built = _patched_sdk(monkeypatch)
    cfg = databricks.DatabricksConfig(host="https://adb-1.net", token="dapi-1")
    databricks.get_workspace_client(cfg)
    assert built == {"host": "https://adb-1.net", "token": "dapi-1"}


def test_workspace_client_on_a_cluster_passes_no_token(monkeypatch, _isolated):
    # In notebook mode the SDK picks the runtime's own credentials up; passing
    # a token would override them.
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
    built = _patched_sdk(monkeypatch)
    databricks.get_workspace_client(databricks.DatabricksConfig(host="https://adb-1.net"))
    assert built == {"host": "https://adb-1.net", "token": None}


def test_workspace_client_defaults_to_the_shared_config(monkeypatch, _isolated):
    monkeypatch.setenv("DATABRICKS_HOST", "adb-shared.azuredatabricks.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-shared")
    built = _patched_sdk(monkeypatch)
    databricks.get_workspace_client()
    assert built == {
        "host": "https://adb-shared.azuredatabricks.net",
        "token": "dapi-shared",
    }


def test_workspace_client_hands_the_sdk_the_service_principal(monkeypatch, _isolated):
    _arm_service_principal(monkeypatch)
    monkeypatch.setenv("DATABRICKS_HOST", "adb-1.azuredatabricks.net")
    monkeypatch.setenv("DATABRICKS_AZURE_RESOURCE_ID", "/subscriptions/s/rg/ws")
    built = _patched_sdk(monkeypatch)
    databricks.get_workspace_client()
    # No token: the SDK runs the Entra flow itself, so it can refresh the
    # bearer token and add the management-token header as it goes.
    assert built == {
        "host": "https://adb-1.azuredatabricks.net",
        "azure_tenant_id": "arm-tenant",
        "azure_client_id": "arm-client",
        "azure_client_secret": "arm-secret",
        "azure_workspace_resource_id": "/subscriptions/s/rg/ws",
    }


def test_workspace_client_omits_an_unset_resource_id(monkeypatch, _isolated):
    _arm_service_principal(monkeypatch)
    built = _patched_sdk(monkeypatch)
    databricks.get_workspace_client(
        databricks.DatabricksConfig(
            host="https://adb-1.net",
            azure_tenant_id="arm-tenant",
            azure_client_id="arm-client",
            azure_client_secret="arm-secret",
        )
    )
    assert "azure_workspace_resource_id" not in built
    assert "token" not in built


def test_sql_connect_params_carry_the_resource_id_headers(monkeypatch, _isolated):
    _arm_service_principal(monkeypatch)
    _patched_azure_client(monkeypatch)
    cfg = databricks.DatabricksConfig(
        host="https://adb-1.azuredatabricks.net",
        warehouse_id="w-1",
        azure_tenant_id="arm-tenant",
        azure_client_id="arm-client",
        azure_client_secret="arm-secret",
        azure_resource_id="/subscriptions/s/rg/ws",
    )
    params = cfg.sql_connect_params()
    assert params["access_token"] == f"token-for:{databricks.AZURE_DATABRICKS_SCOPE}"
    # The connector wants (key, value) pairs, not a mapping.
    assert params["http_headers"] == [
        (databricks.WORKSPACE_RESOURCE_ID_HEADER, "/subscriptions/s/rg/ws"),
        (
            databricks.SP_MANAGEMENT_TOKEN_HEADER,
            f"token-for:{databricks.AZURE_MANAGEMENT_SCOPE}",
        ),
    ]


def test_sql_connect_params_omit_headers_without_a_resource_id(monkeypatch, _isolated):
    _arm_service_principal(monkeypatch)
    _patched_azure_client(monkeypatch)
    cfg = databricks.DatabricksConfig(
        host="https://adb-1.net",
        warehouse_id="w-1",
        azure_tenant_id="arm-tenant",
        azure_client_id="arm-client",
        azure_client_secret="arm-secret",
    )
    assert "http_headers" not in cfg.sql_connect_params()


def test_missing_sdk_raises_helpful_error(_isolated):
    # databricks-sdk is an optional extra; when it is not installed a
    # fully-configured call still fails at import with an install hint.
    try:
        from databricks.sdk import WorkspaceClient  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("databricks-sdk is installed; the import path is unreachable")
    cfg = databricks.DatabricksConfig(host="https://adb-1.net", token="dapi-1")
    with pytest.raises(databricks.DatabricksError, match="databricks-sdk is required"):
        databricks.get_workspace_client(cfg)


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------


def test_get_databricks_config_is_cached():
    first = databricks.get_databricks_config()
    assert databricks.get_databricks_config() is first
    databricks.clear_cache()
    assert databricks.get_databricks_config() is not first
