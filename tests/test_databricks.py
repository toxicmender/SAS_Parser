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
    """The env of a workspace reached with an Entra ID service principal."""
    monkeypatch.setenv("AZURE_TENANT_ID", "t-1")
    monkeypatch.setenv("AZURE_CLIENT_ID", "c-1")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "s-1")


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
