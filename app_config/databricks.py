"""Databricks workspace connection settings.

Submodule of :mod:`app_config`. Answers "which workspace, which compute, which
catalog, and how do we authenticate to it?" — the settings that
``memory.store`` (Delta-backed KV) and ``validation.tracking`` (run history)
need in order to reach a real workspace rather than their local fallbacks.

This module holds *configuration*, not a Spark session: on Databricks the
runtime already provides ``spark``, and the existing consumers take a session
plus a table name. What they lack is a single place that knows the workspace
URL, the SQL warehouse, the Unity Catalog namespace, and the credential — so
that is what lives here, alongside two thin accessors
(:meth:`DatabricksConfig.sql_connect_params`, :func:`get_workspace_client`)
for code that talks to the workspace from *outside* a notebook.

Split of concerns
-----------------
* **Non-secret settings** — host, HTTP path / warehouse or cluster id, Unity
  Catalog catalog and schema, and request timeout — resolve through
  :meth:`DatabricksConfig.from_env`, which reads the standard Databricks
  environment variables first (``DATABRICKS_HOST``, ``DATABRICKS_HTTP_PATH``,
  ``DATABRICKS_WAREHOUSE_ID``, ``DATABRICKS_CLUSTER_ID``,
  ``DATABRICKS_CATALOG``, ``DATABRICKS_SCHEMA``) and falls back to the optional
  ``databricks`` section of ``config.json`` (via :func:`app_config.get_value` /
  :func:`app_config.get_typed_value`, so a wrong-typed entry degrades to the
  hard default with a WARNING rather than crashing).
* **Secrets** — the personal access token — come *only* from the environment
  (``DATABRICKS_TOKEN``), in a field marked ``repr=False`` so the token never
  appears in a ``repr`` or a log line.

Authentication
--------------
:attr:`DatabricksConfig.auth_method` picks, in order:

``notebook``
    Running on a Databricks cluster (``DATABRICKS_RUNTIME_VERSION`` is set) —
    the runtime authenticates itself and :meth:`~DatabricksConfig.get_token`
    returns ``None``. Nothing to configure; this is the production path.
``pat``
    A personal access token in ``DATABRICKS_TOKEN``.
``azure-ad``
    No PAT, but an Entra ID service principal is configured — a token is
    minted through :mod:`app_config.azure` against the Azure Databricks
    resource (:data:`AZURE_DATABRICKS_SCOPE`). The recommended credential for
    an Azure workspace: nothing long-lived is stored anywhere.

``None`` means no usable credentials, and any call needing one raises
:class:`DatabricksError`.

Dependencies
------------
Both clients are *optional* dependencies (extra ``databricks``):
``pip install "sas-parser[databricks]"``. ``databricks-sdk`` is imported lazily
inside :func:`get_workspace_client` and ``msal`` only if the ``azure-ad`` path
is taken, so ``import app_config.databricks`` costs nothing and keeps
``app_config`` the dependency-free leaf the rest of the package relies on.

Typical use
-----------
    from app_config.databricks import get_databricks_config

    cfg = get_databricks_config()
    table = cfg.full_table_name("sas_parser_memory")   # catalog.schema.table
    mem = DatabricksMemory(spark=spark, table=table)

Call :func:`clear_cache` to force re-resolution after the environment changes
(tests do).

Logger name: ``app_config.databricks``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from . import get_typed_value, get_value

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60.0

AUTH_NOTEBOOK = "notebook"
AUTH_PAT = "pat"
AUTH_AZURE_AD = "azure-ad"

# Fixed application id of the Azure Databricks resource in every Entra ID
# tenant; "<resource>/.default" requests the app-level token for it.
AZURE_DATABRICKS_SCOPE = "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default"

# Set by the Databricks runtime on every cluster and job, and by nothing else.
_RUNTIME_ENV_VAR = "DATABRICKS_RUNTIME_VERSION"


class DatabricksError(RuntimeError):
    """Databricks is misconfigured, unreachable, or has no usable credentials.

    A single error type so callers can ``except DatabricksError`` around a
    workspace call regardless of which stage failed; the message says which.
    """


def in_databricks_runtime() -> bool:
    """True when this process is running on a Databricks cluster or job."""
    return bool(os.environ.get(_RUNTIME_ENV_VAR))


def _normalise_host(host: str | None) -> str | None:
    """
    A workspace URL as ``https://<host>``, no trailing slash — accepting the
    bare hostname (``adb-123.4.azuredatabricks.net``) that the Databricks UI
    shows, which the SQL connector wants and the SDK does not.
    """
    if not host:
        return None
    host = host.strip().rstrip("/")
    if not host:
        return None
    if "://" not in host:
        return f"https://{host}"
    return host


@dataclass
class DatabricksConfig:
    """
    Workspace coordinates, compute target, Unity Catalog namespace, and
    credential.

    Construct it directly to pin values explicitly, or call :meth:`from_env`
    for the standard environment-then-``config.json`` resolution.
    :attr:`token` is ``repr=False`` and is never logged.

    Attributes
    ----------
    host : str | None
        Workspace URL (``https://adb-123.4.azuredatabricks.net``); a bare
        hostname is accepted and normalised. ``DATABRICKS_HOST`` /
        ``config.json`` ``databricks.host``. Required for any call from
        outside a notebook.
    http_path : str | None
        SQL connection path. ``DATABRICKS_HTTP_PATH`` /
        ``databricks.http_path``; usually left unset in favour of
        :attr:`warehouse_id` — see :attr:`sql_http_path`.
    warehouse_id : str | None
        SQL warehouse id, the compute for :meth:`sql_connect_params`.
        ``DATABRICKS_WAREHOUSE_ID`` / ``databricks.warehouse_id``.
    cluster_id : str | None
        All-purpose cluster id, for SDK calls (Databricks Connect, jobs) that
        target a cluster rather than a warehouse. ``DATABRICKS_CLUSTER_ID`` /
        ``databricks.cluster_id``.
    catalog, schema : str | None
        Unity Catalog namespace that :meth:`full_table_name` qualifies
        unqualified table names with. ``DATABRICKS_CATALOG`` /
        ``DATABRICKS_SCHEMA``, or ``databricks.catalog`` /
        ``databricks.schema``. Both ``None`` leaves table names untouched, so
        the metastore's own defaults apply.
    timeout : float
        Per-request timeout in seconds. ``databricks.timeout``, default ``60``
        (higher than Vault's: a cold warehouse is slow to answer).
    token : str | None
        Personal access token. ``DATABRICKS_TOKEN`` only — never from
        ``config.json``.
    """

    host: str | None = None
    http_path: str | None = None
    warehouse_id: str | None = None
    cluster_id: str | None = None
    catalog: str | None = None
    schema: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    token: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> "DatabricksConfig":
        """
        Resolve settings from the standard Databricks environment variables,
        falling back to the ``databricks`` section of ``config.json`` for the
        non-secret knobs. The token is read from the environment only.
        """
        return cls(
            host=_normalise_host(
                os.environ.get("DATABRICKS_HOST") or get_value("databricks", "host")
            ),
            http_path=(
                os.environ.get("DATABRICKS_HTTP_PATH")
                or get_value("databricks", "http_path")
            ),
            warehouse_id=(
                os.environ.get("DATABRICKS_WAREHOUSE_ID")
                or get_value("databricks", "warehouse_id")
            ),
            cluster_id=(
                os.environ.get("DATABRICKS_CLUSTER_ID")
                or get_value("databricks", "cluster_id")
            ),
            catalog=(
                os.environ.get("DATABRICKS_CATALOG")
                or get_value("databricks", "catalog")
            ),
            schema=(
                os.environ.get("DATABRICKS_SCHEMA")
                or get_value("databricks", "schema")
            ),
            timeout=get_typed_value(
                "databricks", "timeout", (int, float), DEFAULT_TIMEOUT
            ),
            token=os.environ.get("DATABRICKS_TOKEN"),
        )

    @property
    def auth_method(self) -> str | None:
        """
        ``"notebook"`` on a Databricks cluster (the runtime authenticates
        itself), else ``"pat"`` when a token is set, else ``"azure-ad"`` when
        an Entra ID service principal is configured, else ``None`` (no usable
        credentials).
        """
        if in_databricks_runtime():
            return AUTH_NOTEBOOK
        if self.token:
            return AUTH_PAT
        # Imported here, not at module scope: the azure path is optional, and
        # this keeps `import app_config.databricks` free of it.
        from .azure import FLOW_CLIENT_CREDENTIALS, AzureAuthConfig

        azure = AzureAuthConfig.from_env()
        if azure.tenant_id and azure.auth_flow == FLOW_CLIENT_CREDENTIALS:
            return AUTH_AZURE_AD
        return None

    @property
    def sql_http_path(self) -> str | None:
        """
        The HTTP path for a SQL connection: explicit :attr:`http_path`, else
        one derived from :attr:`warehouse_id`, else ``None``. (A cluster needs
        its workspace org id in the path, so :attr:`cluster_id` alone cannot
        produce one — set :attr:`http_path` for that case.)
        """
        if self.http_path:
            return self.http_path
        if self.warehouse_id:
            return f"/sql/1.0/warehouses/{self.warehouse_id}"
        return None

    def get_token(self) -> str | None:
        """
        The bearer token for workspace calls, per :attr:`auth_method`:
        the PAT, a freshly-minted Entra ID token, or ``None`` in ``notebook``
        mode (where the runtime authenticates and no token is needed).

        Raises
        ------
        DatabricksError
            No usable credentials, or the Entra ID login failed.
        """
        method = self.auth_method
        if method == AUTH_NOTEBOOK:
            return None
        if method == AUTH_PAT:
            return self.token
        if method == AUTH_AZURE_AD:
            from .azure import AzureAuthError, get_token

            try:
                return get_token(scopes=(AZURE_DATABRICKS_SCOPE,))
            except AzureAuthError as exc:
                raise DatabricksError(
                    f"could not mint an Entra ID token for Databricks: {exc}"
                ) from exc
        raise DatabricksError(
            "no Databricks credentials: set DATABRICKS_TOKEN, or configure an "
            "Entra ID service principal (AZURE_TENANT_ID, AZURE_CLIENT_ID, "
            "AZURE_CLIENT_SECRET)"
        )

    def full_table_name(self, table: str) -> str:
        """
        *table* qualified with the configured :attr:`catalog` and
        :attr:`schema` (``catalog.schema.table``).

        An already-qualified name (one containing a dot) is returned unchanged,
        so a caller may always pass its configured table name through here. A
        schema with no catalog yields ``schema.table``; with neither
        configured, *table* is returned as-is and the metastore's defaults
        apply.
        """
        if "." in table:
            return table
        parts = [p for p in (self.catalog, self.schema) if p]
        parts.append(table)
        return ".".join(parts)

    def sql_connect_params(self) -> dict[str, object]:
        """
        Keyword arguments for ``databricks.sql.connect(**params)`` — the
        connector used to reach a SQL warehouse from outside a notebook.

        Raises
        ------
        DatabricksError
            The host, the HTTP path, or the credentials are missing.
        """
        if not self.host:
            raise DatabricksError(
                "no Databricks host configured: set DATABRICKS_HOST or "
                "databricks.host in config.json"
            )
        http_path = self.sql_http_path
        if not http_path:
            raise DatabricksError(
                "no Databricks SQL path configured: set DATABRICKS_WAREHOUSE_ID "
                "(or databricks.warehouse_id), or an explicit "
                "DATABRICKS_HTTP_PATH"
            )
        token = self.get_token()
        if not token:
            raise DatabricksError(
                "sql_connect_params is for connecting from outside Databricks; "
                "on a cluster, use the runtime's own spark session instead"
            )
        params: dict[str, object] = {
            # The connector wants the bare hostname, not the URL.
            "server_hostname": self.host.split("://", 1)[-1],
            "http_path": http_path,
            "access_token": token,
        }
        if self.catalog:
            params["catalog"] = self.catalog
        if self.schema:
            params["schema"] = self.schema
        return params


def get_workspace_client(config: DatabricksConfig | None = None):
    """
    A ``databricks.sdk.WorkspaceClient`` for *config* (default: the shared
    :func:`get_databricks_config`).

    Raises
    ------
    DatabricksError
        The host or credentials are missing, or ``databricks-sdk`` is not
        installed.
    """
    config = config if config is not None else get_databricks_config()
    # Validate before importing the SDK so a misconfiguration reports the real
    # problem instead of a missing-dependency error.
    if not config.host:
        raise DatabricksError(
            "no Databricks host configured: set DATABRICKS_HOST or "
            "databricks.host in config.json"
        )
    token = config.get_token()
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError as exc:
        raise DatabricksError(
            "databricks-sdk is required for workspace access; install it with "
            "'pip install \"sas-parser[databricks]\"'"
        ) from exc
    logger.info(
        f"get_workspace_client: {config.host} via {config.auth_method} "
        f"(catalog={config.catalog}, schema={config.schema})"
    )
    # No token in notebook mode: the SDK picks the runtime's own credentials up.
    return WorkspaceClient(host=config.host, token=token)


# One resolved config per process, mirroring app_config's config cache.
_config_cache: DatabricksConfig | None = None


def get_databricks_config() -> DatabricksConfig:
    """The process-wide :class:`DatabricksConfig` (resolved from the environment)."""
    global _config_cache
    if _config_cache is None:
        _config_cache = DatabricksConfig.from_env()
    return _config_cache


def clear_cache() -> None:
    """Drop the cached config so the next access re-resolves (for tests)."""
    global _config_cache
    _config_cache = None
