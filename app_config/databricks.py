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
* **Secrets** — the personal access token and the service-principal client
  secret — come *only* from the environment (``DATABRICKS_TOKEN``,
  ``ARM_CLIENT_SECRET``) or from the workspace secret scope, in fields marked
  ``repr=False`` so neither ever appears in a ``repr`` or a log line.

Authentication
--------------
:attr:`DatabricksConfig.auth_method` picks, in order:

``notebook``
    Running on a Databricks cluster (``DATABRICKS_RUNTIME_VERSION`` is set) —
    the runtime authenticates itself and :meth:`~DatabricksConfig.get_token`
    returns ``None``. Nothing to configure; this is the production path.
``pat``
    A personal access token in ``DATABRICKS_TOKEN``.
``azure-sp``
    A Microsoft Entra service principal configured the way Databricks
    documents it — ``ARM_TENANT_ID`` / ``ARM_CLIENT_ID`` /
    ``ARM_CLIENT_SECRET``, or the matching ``databricks.azure_tenant_id`` /
    ``azure_client_id`` config entries. The recommended credential for an
    Azure workspace reached from outside: nothing long-lived is stored
    anywhere, and the identity is pinned to *this* config rather than shared
    with whatever the rest of the process authenticates as.

    The principal may equally be stored in a Databricks secret scope — see
    below.

    Databricks expects the service principal to be assigned to the workspace.
    When it is not, set ``DATABRICKS_AZURE_RESOURCE_ID`` to the workspace's
    ARM resource id and :meth:`~DatabricksConfig.workspace_headers` supplies
    the resource-id and management-token headers the control plane then
    requires (this needs Contributor or Owner on the Azure resource).
``azure-ad``
    No workspace-scoped service principal either, but the process-wide Entra
    ID identity of :mod:`app_config.azure` (``AZURE_TENANT_ID`` /
    ``AZURE_CLIENT_ID`` / ``AZURE_CLIENT_SECRET``) is usable. Same token
    exchange, different place to have configured the credential.

Both Entra paths mint a short-lived token against the Azure Databricks
resource (:data:`AZURE_DATABRICKS_SCOPE`) through :mod:`app_config.azure`.
``None`` means no usable credentials, and any call needing one raises
:class:`DatabricksError`.

See https://learn.microsoft.com/en-us/azure/databricks/dev-tools/auth/azure-sp.

The service principal in a secret scope
---------------------------------------
Rather than in ``ARM_*``, the principal may live in a Databricks secret scope
under the keys :data:`SECRET_KEY_TENANT_ID`, :data:`SECRET_KEY_CLIENT_ID`, and
:data:`SECRET_KEY_CLIENT_SECRET`. Point ``DATABRICKS_SECRET_SCOPE`` (or
``databricks.secret_scope``) at the scope and
:meth:`DatabricksConfig.service_principal` fetches it on first use, via
:func:`read_workspace_secret`.

The scope is a *fallback*, not an override: an ``ARM_*`` principal that is
fully configured wins, matching this module's environment-first rule
everywhere else and keeping a local override possible.

Reading the scope needs a workspace credential of its own, which is why it
never appears in :attr:`~DatabricksConfig.auth_method` — a principal that only
exists in the scope cannot also be what authenticates the read. In practice
that bootstrap is the Databricks runtime's own credentials (on a cluster) or a
PAT; without either, the read raises :class:`DatabricksError`.

Dependencies
------------
Both clients are *optional* dependencies (extra ``databricks``):
``pip install "sas-parser[databricks]"``. ``databricks-sdk`` is imported lazily
inside :func:`get_workspace_client` / :func:`read_workspace_secret`, and
``msal`` only if an Entra path is taken, so ``import app_config.databricks``
costs nothing and keeps ``app_config`` the dependency-free leaf the rest of the
package relies on.

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
from typing import TYPE_CHECKING, Any

from . import get_typed_value, get_value

if TYPE_CHECKING:  # real types without importing the optional azure path
    from .azure import AzureAuthClient

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60.0

AUTH_NOTEBOOK = "notebook"
AUTH_PAT = "pat"
AUTH_AZURE_SP = "azure-sp"
AUTH_AZURE_AD = "azure-ad"

# Fixed application id of the Azure Databricks resource in every Entra ID
# tenant; "<resource>/.default" requests the app-level token for it.
AZURE_DATABRICKS_SCOPE = "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default"

# The ARM resource whose token the control plane wants in the management-token
# header. The resource itself ends in a slash and MSAL scopes are
# "<resource>/.default", so the doubled slash below is deliberate — it yields
# the same audience the Databricks SDK requests.
AZURE_MANAGEMENT_RESOURCE = "https://management.core.windows.net/"
AZURE_MANAGEMENT_SCOPE = f"{AZURE_MANAGEMENT_RESOURCE}/.default"

# Sent together, and only when the service principal is not itself assigned to
# the workspace — see DatabricksConfig.workspace_headers.
WORKSPACE_RESOURCE_ID_HEADER = "X-Databricks-Azure-Workspace-Resource-Id"
SP_MANAGEMENT_TOKEN_HEADER = "X-Databricks-Azure-SP-Management-Token"

# Keys the service principal is filed under in the workspace secret scope.
SECRET_KEY_CLIENT_ID = "sp-hsv-appid"
SECRET_KEY_CLIENT_SECRET = "sp-hsv-secret"
SECRET_KEY_TENANT_ID = "sp-hsv-tenantid"

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


@dataclass
class AzureServicePrincipal:
    """
    The three values an Entra ID service-principal login needs, however they
    were sourced (``ARM_*`` environment variables, ``config.json``, or the
    workspace secret scope). :attr:`client_secret` is ``repr=False``.
    """

    tenant_id: str
    client_id: str
    client_secret: str = field(repr=False)


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
    azure_resource_id : str | None
        ARM resource id of the workspace
        (``/subscriptions/.../providers/Microsoft.Databricks/workspaces/...``),
        for the case where the service principal is *not* assigned to the
        workspace. ``DATABRICKS_AZURE_RESOURCE_ID`` /
        ``databricks.azure_workspace_resource_id``. Unset (the recommended
        setup) means the service principal is assigned to the workspace and no
        extra headers are needed — see :meth:`workspace_headers`.
    azure_tenant_id, azure_client_id : str | None
        Entra ID service principal for the ``azure-sp`` path. ``ARM_TENANT_ID``
        / ``ARM_CLIENT_ID`` (the names Databricks documents) or
        ``databricks.azure_tenant_id`` / ``databricks.azure_client_id``. Left
        unset, they are fetched from :attr:`secret_scope`.
    azure_client_secret : str | None
        That service principal's secret. ``ARM_CLIENT_SECRET`` only — never
        from ``config.json``; the alternative is :attr:`secret_scope`.
    secret_scope : str | None
        Databricks secret scope holding the service principal under the keys
        :data:`SECRET_KEY_CLIENT_ID`, :data:`SECRET_KEY_CLIENT_SECRET`, and
        :data:`SECRET_KEY_TENANT_ID`. ``DATABRICKS_SECRET_SCOPE`` /
        ``databricks.secret_scope``. Read on demand by
        :meth:`service_principal`, and only when the ``ARM_*`` values are
        absent.
    """

    host: str | None = None
    http_path: str | None = None
    warehouse_id: str | None = None
    cluster_id: str | None = None
    catalog: str | None = None
    schema: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    token: str | None = field(default=None, repr=False)
    azure_resource_id: str | None = None
    azure_tenant_id: str | None = None
    azure_client_id: str | None = None
    azure_client_secret: str | None = field(default=None, repr=False)
    secret_scope: str | None = None
    # Resolved service principal and Entra ID client, populated on first use.
    # Excluded from init/repr/eq: they are caches, not configuration.
    _principal: AzureServicePrincipal | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _azure: "AzureAuthClient | None" = field(
        default=None, init=False, repr=False, compare=False
    )

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
            azure_resource_id=(
                os.environ.get("DATABRICKS_AZURE_RESOURCE_ID")
                or get_value("databricks", "azure_workspace_resource_id")
            ),
            # ARM_*, not AZURE_*: these are the names Databricks documents for
            # its own Entra service-principal flow, and they deliberately do
            # not collide with app_config.azure's process-wide identity.
            azure_tenant_id=(
                os.environ.get("ARM_TENANT_ID")
                or get_value("databricks", "azure_tenant_id")
            ),
            azure_client_id=(
                os.environ.get("ARM_CLIENT_ID")
                or get_value("databricks", "azure_client_id")
            ),
            azure_client_secret=os.environ.get("ARM_CLIENT_SECRET"),
            secret_scope=(
                os.environ.get("DATABRICKS_SECRET_SCOPE")
                or get_value("databricks", "secret_scope")
            ),
        )

    @property
    def has_service_principal(self) -> bool:
        """
        True when a complete service principal is configured *locally* (the
        ``ARM_*`` environment variables or ``config.json``). Cheap: unlike
        :meth:`service_principal` it never reads the secret scope.
        """
        return bool(
            self.azure_tenant_id and self.azure_client_id and self.azure_client_secret
        )

    @property
    def _can_read_secret_scope(self) -> bool:
        """
        True when :attr:`secret_scope` is set *and* something can authenticate
        the read of it — the Databricks runtime on a cluster, or a PAT. Keeps
        :meth:`entra_token` from insisting on a scope it could never reach,
        so a shared :mod:`app_config.azure` identity still gets its turn.
        """
        return bool(self.secret_scope) and bool(
            in_databricks_runtime() or self.token
        )

    @property
    def auth_method(self) -> str | None:
        """
        ``"notebook"`` on a Databricks cluster (the runtime authenticates
        itself), else ``"pat"`` when a token is set, else ``"azure-sp"`` when a
        service principal is configured locally, else ``"azure-ad"`` when
        :mod:`app_config.azure`'s process-wide identity is usable, else
        ``None`` (no usable credentials).

        Note that :attr:`secret_scope` never appears here: reading the scope
        needs a workspace credential of its own, so a service principal that
        lives only in the scope cannot also be what authenticates the read.
        On a cluster or with a PAT that bootstrap is already in place, and
        :meth:`service_principal` fetches it on demand.
        """
        if in_databricks_runtime():
            return AUTH_NOTEBOOK
        if self.token:
            return AUTH_PAT
        if self.has_service_principal:
            return AUTH_AZURE_SP
        # Imported here, not at module scope: the azure path is optional, and
        # this keeps `import app_config.databricks` free of it.
        from .azure import FLOW_CLIENT_CREDENTIALS, AzureAuthConfig

        azure = AzureAuthConfig.from_env()
        if azure.tenant_id and azure.auth_flow == FLOW_CLIENT_CREDENTIALS:
            return AUTH_AZURE_AD
        return None

    def service_principal(self) -> AzureServicePrincipal:
        """
        The Entra ID service principal for this workspace: the locally
        configured ``ARM_*`` values when complete, else the three keys read
        out of :attr:`secret_scope` (:data:`SECRET_KEY_TENANT_ID`,
        :data:`SECRET_KEY_CLIENT_ID`, :data:`SECRET_KEY_CLIENT_SECRET`).

        The scope read goes through :func:`read_workspace_secret`, which needs
        a workspace credential that is *not* this service principal — the
        Databricks runtime's own on a cluster, or a PAT. Resolved once and
        cached on the config; :func:`clear_cache` drops it with everything
        else.

        Raises
        ------
        DatabricksError
            Neither source yields a complete service principal, or the secret
            scope could not be read.
        """
        if self._principal is not None:
            return self._principal
        if self.has_service_principal:
            # Narrowed by has_service_principal; asserted for the type checker.
            assert self.azure_tenant_id and self.azure_client_id
            assert self.azure_client_secret
            principal = AzureServicePrincipal(
                tenant_id=self.azure_tenant_id,
                client_id=self.azure_client_id,
                client_secret=self.azure_client_secret,
            )
            logger.info(
                f"service_principal: using the locally configured principal "
                f"{principal.client_id} in tenant {principal.tenant_id}"
            )
        elif self.secret_scope:
            # One bootstrap client for all three keys, not one per key.
            values = read_workspace_secrets(
                self.secret_scope,
                (SECRET_KEY_TENANT_ID, SECRET_KEY_CLIENT_ID, SECRET_KEY_CLIENT_SECRET),
                config=self,
            )
            principal = AzureServicePrincipal(
                tenant_id=values[SECRET_KEY_TENANT_ID],
                client_id=values[SECRET_KEY_CLIENT_ID],
                client_secret=values[SECRET_KEY_CLIENT_SECRET],
            )
            logger.info(
                f"service_principal: read principal {principal.client_id} in "
                f"tenant {principal.tenant_id} from secret scope "
                f"{self.secret_scope}"
            )
        else:
            raise DatabricksError(
                "no Azure service principal configured: set ARM_TENANT_ID, "
                "ARM_CLIENT_ID and ARM_CLIENT_SECRET, or point "
                "DATABRICKS_SECRET_SCOPE at the scope holding "
                f"{SECRET_KEY_TENANT_ID}/{SECRET_KEY_CLIENT_ID}/"
                f"{SECRET_KEY_CLIENT_SECRET}"
            )
        self._principal = principal
        return principal

    def entra_token(self, scope: str = AZURE_DATABRICKS_SCOPE) -> str:
        """
        An Entra ID access token for *scope*, minted through
        :mod:`app_config.azure` from :meth:`service_principal` when one is
        available, else from that module's own process-wide identity.

        Defaults to the Azure Databricks resource, so ``cfg.entra_token()`` is
        a workspace bearer token; pass :data:`AZURE_MANAGEMENT_SCOPE` for the
        ARM management token instead. Tokens are cached until shortly before
        they expire.

        Raises
        ------
        DatabricksError
            No usable Entra ID identity, or the login failed.
        """
        from .azure import AzureAuthError

        try:
            if self.has_service_principal or self._can_read_secret_scope:
                return self._azure_client().get_token((scope,))
            # Late attribute lookup, so app_config.azure's shared client (and
            # its token cache) is reused rather than rebuilt per config.
            from .azure import get_token

            return get_token(scopes=(scope,))
        except AzureAuthError as exc:
            raise DatabricksError(
                f"could not mint an Entra ID token for Databricks: {exc}"
            ) from exc

    def _azure_client(self) -> "AzureAuthClient":
        """
        An :class:`~app_config.azure.AzureAuthClient` pinned to
        :meth:`service_principal`, built once and cached on this config.
        """
        if self._azure is None:
            from .azure import get_client_for_principal

            principal = self.service_principal()
            # Via app_config.azure's per-principal cache, not a private client:
            # the same identity minting a Databricks token here and a Vault
            # login JWT there should share one MSAL token cache.
            self._azure = get_client_for_principal(
                principal.tenant_id, principal.client_id, principal.client_secret
            )
        return self._azure

    def workspace_headers(self) -> dict[str, str]:
        """
        The extra headers the control plane requires when the workspace is
        addressed by :attr:`azure_resource_id` rather than by a service
        principal assigned to it: the resource id, and an ARM management token
        for the same principal.

        Empty when no :attr:`azure_resource_id` is set (the recommended setup,
        where the principal is assigned to the workspace) or when the
        credential is not an Entra ID one, since neither header means anything
        alongside a PAT.

        Raises
        ------
        DatabricksError
            The management token could not be minted.
        """
        if not self.azure_resource_id:
            return {}
        if self.auth_method not in (AUTH_AZURE_SP, AUTH_AZURE_AD):
            logger.warning(
                f"workspace_headers: ignoring azure_resource_id — it only "
                f"applies to an Entra ID credential, not {self.auth_method}"
            )
            return {}
        return {
            WORKSPACE_RESOURCE_ID_HEADER: self.azure_resource_id,
            SP_MANAGEMENT_TOKEN_HEADER: self.entra_token(AZURE_MANAGEMENT_SCOPE),
        }

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
        if method in (AUTH_AZURE_SP, AUTH_AZURE_AD):
            return self.entra_token()
        raise DatabricksError(
            "no Databricks credentials: set DATABRICKS_TOKEN, or configure an "
            "Entra ID service principal (ARM_TENANT_ID, ARM_CLIENT_ID, "
            "ARM_CLIENT_SECRET)"
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
        headers = self.workspace_headers()
        if headers:
            # The connector wants (key, value) pairs, not a mapping.
            params["http_headers"] = list(headers.items())
        return params


def _import_workspace_client():
    """``databricks.sdk.WorkspaceClient``, with an install hint when missing."""
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError as exc:
        raise DatabricksError(
            "databricks-sdk is required for workspace access; install it with "
            "'pip install \"sas-parser[databricks]\"'"
        ) from exc
    return WorkspaceClient


def _bootstrap_client(config: DatabricksConfig, scope: str):
    """
    A ``WorkspaceClient`` for reading secret *scope*, authenticated by a
    credential that cannot itself have come out of that scope — the Databricks
    runtime's own on a cluster, or a PAT.

    Deliberately not :func:`get_workspace_client`: that one may authenticate
    with the very service principal the scope holds, which would recurse.
    """
    in_runtime = in_databricks_runtime()
    if not in_runtime and not config.token:
        raise DatabricksError(
            f"reading secret scope '{scope}' needs a credential that does not "
            f"come from the scope itself: run on a Databricks cluster, or set "
            f"DATABRICKS_TOKEN"
        )
    if not in_runtime and not config.host:
        raise DatabricksError(
            "no Databricks host configured: set DATABRICKS_HOST or "
            "databricks.host in config.json"
        )
    WorkspaceClient = _import_workspace_client()
    # On a cluster the SDK picks the runtime's own credentials up, and host
    # and token are both absent by design.
    # dict[str, Any], not dict[str, object]: this is splatted into the SDK's
    # typed __init__, and `object` is assignable to none of its parameters.
    params: dict[str, Any] = {}
    if config.host:
        params["host"] = config.host
    if config.token:
        params["token"] = config.token
    return WorkspaceClient(**params)


def read_workspace_secrets(
    scope: str,
    keys: tuple[str, ...] | list[str],
    *,
    config: DatabricksConfig | None = None,
) -> dict[str, str]:
    """
    The values of *keys* in Databricks secret *scope*, read through a single
    bootstrap client (see :func:`_bootstrap_client`) — one client for the whole
    set, not one per key.

    Reads via the SDK's ``dbutils`` shim rather than ``secrets.get_secret``:
    the underlying REST call rejects a normal user outside a notebook, and the
    shim is the surface Databricks documents (it also base64-decodes for us).

    Raises
    ------
    DatabricksError
        No bootstrap credential, ``databricks-sdk`` is not installed, or a
        scope or key does not exist (or is not readable).
    """
    config = config if config is not None else get_databricks_config()
    client = _bootstrap_client(config, scope)
    values: dict[str, str] = {}
    for key in keys:
        try:
            value = client.dbutils.secrets.get(scope, key)
        except Exception as exc:
            raise DatabricksError(
                f"could not read secret '{key}' from Databricks scope "
                f"'{scope}': {exc}"
            ) from exc
        if not value:
            raise DatabricksError(
                f"secret '{key}' in Databricks scope '{scope}' is empty"
            )
        values[key] = value
    logger.info(
        f"read_workspace_secrets: read {sorted(values)} from scope '{scope}'"
    )
    return values


def read_workspace_secret(
    scope: str, key: str, *, config: DatabricksConfig | None = None
) -> str:
    """
    The value of a single *key* in Databricks secret *scope*, as a string.
    Thin wrapper over :func:`read_workspace_secrets`.
    """
    return read_workspace_secrets(scope, (key,), config=config)[key]


def get_workspace_client(config: DatabricksConfig | None = None):
    """
    A ``databricks.sdk.WorkspaceClient`` for *config* (default: the shared
    :func:`get_databricks_config`).

    On the ``azure-sp`` path the service principal is handed to the SDK itself
    rather than a token minted here, so the SDK runs the documented Entra flow:
    it refreshes the bearer token as it expires and, when
    :attr:`~DatabricksConfig.azure_resource_id` is set, adds the resource-id
    and management-token headers — neither of which a static token can do. The
    other paths pass a token (or, in ``notebook`` mode, none at all, leaving
    the SDK to pick the runtime's own credentials up).

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
    method = config.auth_method
    if method == AUTH_AZURE_SP:
        principal = config.service_principal()
        # dict[str, Any] for the same reason as _bootstrap_client's params.
        credentials: dict[str, Any] = {
            "azure_tenant_id": principal.tenant_id,
            "azure_client_id": principal.client_id,
            "azure_client_secret": principal.client_secret,
        }
        if config.azure_resource_id:
            credentials["azure_workspace_resource_id"] = config.azure_resource_id
    else:
        credentials = {"token": config.get_token()}
    WorkspaceClient = _import_workspace_client()
    logger.info(
        f"get_workspace_client: {config.host} via {method} "
        f"(catalog={config.catalog}, schema={config.schema})"
    )
    return WorkspaceClient(host=config.host, **credentials)


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
