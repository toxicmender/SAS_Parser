"""Microsoft SharePoint access through the Microsoft Graph API.

Submodule of :mod:`app_config`, and the first *consumer* of
:mod:`app_config.azure` after :mod:`app_config.databricks`: where ``azure``
turns an Entra ID identity into an access token, this module points that token
at a SharePoint site's document library (a Graph *drive*) and its lists, and
exposes the operations the conversion, xref and complexity flows need —
browsing folders, reading and writing files, creating folders, and reading and
updating list items.

It stays free of domain knowledge: which folder holds a given application's
scripts, and what the columns of the request list mean, belong to those flows,
not here. What this module owns is the transport and the addressing —
including :meth:`SharePointConfig.drive_path`, so nobody concatenates path
segments by hand.

Split of concerns
-----------------
* **Non-secret settings** — which site and which document library — resolve
  through :meth:`SharePointConfig.from_env`, which reads ``SHAREPOINT_*``
  environment variables first and falls back to the optional ``sharepoint``
  section of ``config.json`` (via :func:`app_config.get_value` /
  :func:`app_config.get_typed_value`, so a wrong-typed entry degrades to the
  hard default with a WARNING rather than crashing).
* **Secrets** — there are none here. Authentication is delegated to
  :mod:`app_config.azure`, which mints a Graph token
  (:data:`GRAPH_DEFAULT_SCOPE`) for one of two identities. With a secret scope
  configured, it is SharePoint's **own** service principal, read from that
  scope under ``saact-hsv-tenantid`` / ``-appid`` / ``-secret`` — a different
  principal from the one that reaches Vault, which is how the deployment is
  set up. With no scope, it is the shared ``AZURE_TENANT_ID`` /
  ``AZURE_CLIENT_ID`` / ``AZURE_CLIENT_SECRET`` identity. Nothing long-lived is
  stored anywhere; either app registration needs the application permission
  ``Sites.ReadWrite.All`` (files and lists) granted with admin consent.

Addressing
----------
Files and folders are addressed by their path *relative to the document
library root* — ``""`` (or ``"/"``) is the root, ``"Reports/2024"`` a
sub-folder. Internally a path becomes the Graph drive-item alias ``root:/<path>:``
(:func:`_drive_item_id`). The target library is either an explicit
:attr:`~SharePointConfig.drive_id`, or the site's default document library
resolved once from :attr:`~SharePointConfig.site_id` / the
``site_hostname`` + ``site_path`` pair.

Synchronous facade
------------------
The Graph SDK is asynchronous; every method here is an ordinary blocking call,
to match the rest of the package. Each :class:`SharePointClient` owns one
private event loop and drives its coroutines to completion on it
(:meth:`~SharePointClient._run`), so the underlying ``httpx`` connection pool
stays bound to a single loop. Calling these methods from *inside* a running
event loop is therefore rejected with a clear error — use them from synchronous
code, or drive the Graph SDK directly.

Dependency
----------
The ``msgraph-sdk`` client is an *optional* dependency (extra ``sharepoint``):
``pip install "sas-parser[sharepoint]"``. It — and every helper it pulls in —
is imported lazily inside :meth:`SharePointClient._build_client`, so
``import app_config.sharepoint`` costs nothing and keeps ``app_config`` the
dependency-free leaf the rest of the package relies on.

Typical use
-----------
    from app_config.sharepoint import get_sharepoint_client

    sp = get_sharepoint_client()
    for entry in sp.list_directory("Reports"):
        print(entry["name"], entry["is_folder"])
    body = sp.read_file("Reports/summary.csv")
    sp.write_file("Reports/notes.txt", "hello")
    sp.create_directory("Reports/2024")
    rows = sp.list_items("Tasks")               # every row, fields expanded

Call :func:`clear_cache` to drop the shared client after the environment
changes (tests do).

Logger name: ``app_config.sharepoint``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from . import get_typed_value, get_value

if TYPE_CHECKING:  # real types without importing the optional azure path
    from .databricks import SecretKeySet

logger = logging.getLogger(__name__)

# The Microsoft Graph resource, as a client-credentials ".default" scope: it
# requests the application permissions consented on the app registration.
GRAPH_DEFAULT_SCOPE = "https://graph.microsoft.com/.default"

DEFAULT_TIMEOUT = 60.0

# Document-library prefix a SharePoint URL shows but Graph does not use: items
# are addressed *inside* the library, so a base path pasted from the browser
# has to have this stripped or every lookup goes hunting for a folder of that
# literal name. Matched case-insensitively — the reference strips it exactly.
DOC_LIBRARY_PREFIX = "shared documents/"

# Where the SharePoint service principal is filed in the Databricks secret
# scope. A different principal from the workspace/Vault one under `sp-hsv-*`,
# in the same scope.
DEFAULT_TENANT_ID_KEY = "saact-hsv-tenantid"
DEFAULT_CLIENT_ID_KEY = "saact-hsv-appid"
DEFAULT_CLIENT_SECRET_KEY = "saact-hsv-secret"

# The lists a run is driven by, as `list_id(kind)` names them, mapped to the
# config key each id is read from.
LIST_KINDS: dict[str, str] = {
    "requests": "list_id_sas_requests",
    "conversions": "list_id_sas_conversions",
    "xref": "list_id_xref",
    "complexity": "list_id_sas_complexity",
}

# Lifetime stamped on the AccessToken handed to the Graph SDK. app_config.azure
# holds the authoritative token cache (it re-acquires shortly before real
# expiry); this short window just makes the SDK re-consult that cache rather
# than trust one token for an unknown lifetime. See _GraphTokenCredential.
_CREDENTIAL_TTL = 300


class SharePointError(RuntimeError):
    """SharePoint is misconfigured, unreachable, unauthorised, or the item is absent.

    A single error type so callers can ``except SharePointError`` around an
    operation regardless of which stage failed; the message says which.
    """


def _drive_item_id(path: str) -> str:
    """
    The Graph drive-item id for a path *relative to the library root*: the
    ``"root"`` alias for the root itself, else the ``"root:/<path>:"`` path
    alias the SDK percent-encodes into the request URL. Leading/trailing
    slashes and surrounding whitespace are ignored.
    """
    clean = path.strip().strip("/")
    return "root" if not clean else f"root:/{clean}:"


def _normalise_base_path(raw: str | None) -> str:
    """
    A configured applications root as a **drive-relative** path: leading and
    trailing slashes dropped, and a leading ``"Shared Documents/"`` stripped
    (case-insensitively).

    That prefix is what a SharePoint URL shows, so it is what gets pasted into
    config, but Graph addresses items inside the library — leaving it on would
    look for a folder literally named "Shared Documents". ``""`` when unset,
    meaning the library root.
    """
    if not raw:
        return ""
    trimmed = raw.strip().strip("/")
    lowered = trimmed.lower()
    if lowered == DOC_LIBRARY_PREFIX.rstrip("/"):
        return ""  # the library itself, i.e. the drive root
    if lowered.startswith(DOC_LIBRARY_PREFIX):
        trimmed = trimmed[len(DOC_LIBRARY_PREFIX):].strip("/")
    return trimmed


def _normalise_site_path(site_path: str | None) -> str | None:
    """A server-relative site path as ``/sites/Name`` — leading slash, no
    trailing slash — or ``None`` when unset."""
    if not site_path:
        return None
    trimmed = site_path.strip().strip("/")
    return f"/{trimmed}" if trimmed else None


@dataclass
class SharePointConfig:
    """
    Which SharePoint site and document library to operate on.

    Construct it directly to pin values explicitly, or call :meth:`from_env`
    for the standard environment-then-``config.json`` resolution. There are no
    secret fields: authentication is delegated to :mod:`app_config.azure`.

    Attributes
    ----------
    site_hostname : str | None
        SharePoint host, e.g. ``contoso.sharepoint.com``.
        ``SHAREPOINT_SITE_HOSTNAME`` / ``config.json`` ``sharepoint.site_hostname``.
        Combined with :attr:`site_path` to identify the site when
        :attr:`site_id` is unset.
    site_path : str | None
        Server-relative site path, e.g. ``/sites/Engineering``.
        ``SHAREPOINT_SITE_PATH`` / ``sharepoint.site_path``. A bare or
        slash-wrapped value is normalised.
    site_id : str | None
        An explicit Graph site id (a GUID triple, or the
        ``hostname:/sites/Name`` composite). ``SHAREPOINT_SITE_ID`` /
        ``sharepoint.site_id``. Wins over :attr:`site_hostname` +
        :attr:`site_path` — see :attr:`resolved_site_id`.
    drive_id : str | None
        An explicit document-library (drive) id, targeting a specific library.
        ``SHAREPOINT_DRIVE_ID`` / ``sharepoint.drive_id``. When unset the site's
        default library is resolved once from :attr:`resolved_site_id`.
    scopes : tuple[str, ...]
        Graph scopes to request. ``SHAREPOINT_SCOPES`` (space- or
        comma-separated) / ``sharepoint.scopes``, default
        ``("https://graph.microsoft.com/.default",)``.
    timeout : float
        Per-request timeout in seconds. ``sharepoint.timeout``, default ``60``.
    file_server_base_path : str
        The applications root inside the library, drive-relative — a leading
        ``"Shared Documents/"`` is stripped on resolution (see
        :func:`_normalise_base_path`). ``SHAREPOINT_FILE_SERVER_BASE_PATH`` /
        ``sharepoint.file_server_base_path``. ``""`` (default) is the library
        root. Join onto it with :meth:`drive_path`, never by hand.
    secret_scope : str | None
        Databricks secret scope holding SharePoint's *own* service principal.
        ``SHAREPOINT_SECRET_SCOPE`` / ``sharepoint.secret_scope``, defaulting
        to the workspace's ``DATABRICKS_SECRET_SCOPE`` /
        ``databricks.secret_scope`` — the reference keeps both principals in
        one scope. ``None`` (nothing configured anywhere) falls back to the
        shared :mod:`app_config.azure` identity, which is the local path.
    tenant_id_key, client_id_key, client_secret_key : str
        Which keys in that scope hold the principal.
        ``SHAREPOINT_TENANT_ID_KEY`` / ``SHAREPOINT_CLIENT_ID_KEY`` /
        ``SHAREPOINT_CLIENT_SECRET_KEY``, or the ``sharepoint.*`` equivalents;
        defaulting to ``saact-hsv-tenantid`` / ``saact-hsv-appid`` /
        ``saact-hsv-secret``.
    list_id_sas_requests, list_id_sas_conversions, list_id_xref,
    list_id_sas_complexity : str | None
        The four lists a run is driven by. ``SHAREPOINT_LIST_ID_*`` /
        ``sharepoint.list_id_*``. Read them through :meth:`list_id`, which
        names the missing config key rather than passing ``None`` to Graph.
    """

    site_hostname: str | None = None
    site_path: str | None = None
    site_id: str | None = None
    drive_id: str | None = None
    scopes: tuple[str, ...] = (GRAPH_DEFAULT_SCOPE,)
    timeout: float = DEFAULT_TIMEOUT
    file_server_base_path: str = ""
    secret_scope: str | None = None
    tenant_id_key: str = DEFAULT_TENANT_ID_KEY
    client_id_key: str = DEFAULT_CLIENT_ID_KEY
    client_secret_key: str = DEFAULT_CLIENT_SECRET_KEY
    list_id_sas_requests: str | None = None
    list_id_sas_conversions: str | None = None
    list_id_xref: str | None = None
    list_id_sas_complexity: str | None = None

    @classmethod
    def from_env(cls) -> "SharePointConfig":
        """
        Resolve settings from the ``SHAREPOINT_*`` environment variables,
        falling back to the ``sharepoint`` section of ``config.json``.
        Authentication settings are read separately by :mod:`app_config.azure`.
        """
        return cls(
            site_hostname=(
                os.environ.get("SHAREPOINT_SITE_HOSTNAME")
                or get_value("sharepoint", "site_hostname")
            ),
            site_path=_normalise_site_path(
                os.environ.get("SHAREPOINT_SITE_PATH")
                or get_value("sharepoint", "site_path")
            ),
            site_id=(
                os.environ.get("SHAREPOINT_SITE_ID")
                or get_value("sharepoint", "site_id")
            ),
            drive_id=(
                os.environ.get("SHAREPOINT_DRIVE_ID")
                or get_value("sharepoint", "drive_id")
            ),
            scopes=_resolve_scopes(),
            timeout=get_typed_value(
                "sharepoint", "timeout", (int, float), DEFAULT_TIMEOUT
            ),
            file_server_base_path=_normalise_base_path(
                os.environ.get("SHAREPOINT_FILE_SERVER_BASE_PATH")
                or get_value("sharepoint", "file_server_base_path")
            ),
            secret_scope=(
                os.environ.get("SHAREPOINT_SECRET_SCOPE")
                or get_value("sharepoint", "secret_scope")
                # SharePoint's principal lives in the same scope as the
                # workspace's, under different keys — so the scope defaults
                # across rather than having to be configured twice.
                or os.environ.get("DATABRICKS_SECRET_SCOPE")
                or get_value("databricks", "secret_scope")
            ),
            tenant_id_key=(
                os.environ.get("SHAREPOINT_TENANT_ID_KEY")
                or get_value("sharepoint", "tenant_id_key", DEFAULT_TENANT_ID_KEY)
            ),
            client_id_key=(
                os.environ.get("SHAREPOINT_CLIENT_ID_KEY")
                or get_value("sharepoint", "client_id_key", DEFAULT_CLIENT_ID_KEY)
            ),
            client_secret_key=(
                os.environ.get("SHAREPOINT_CLIENT_SECRET_KEY")
                or get_value(
                    "sharepoint", "client_secret_key", DEFAULT_CLIENT_SECRET_KEY
                )
            ),
            **{
                field_name: (
                    os.environ.get(f"SHAREPOINT_{field_name.upper()}")
                    or get_value("sharepoint", field_name)
                )
                for field_name in LIST_KINDS.values()
            },
        )

    @property
    def resolved_site_id(self) -> str | None:
        """
        The Graph site id: explicit :attr:`site_id`, else the
        ``hostname:/path`` composite built from :attr:`site_hostname` and
        :attr:`site_path`, else ``None`` (unknown — only a
        :attr:`drive_id` can then identify a library).
        """
        if self.site_id:
            return self.site_id
        if self.site_hostname and self.site_path:
            return f"{self.site_hostname}:{self.site_path}"
        return None

    @property
    def secret_keys(self) -> "SecretKeySet":
        """This config's key triple as a
        :class:`~app_config.databricks.SecretKeySet`."""
        from .databricks import SecretKeySet

        return SecretKeySet(
            tenant_id_key=self.tenant_id_key,
            client_id_key=self.client_id_key,
            client_secret_key=self.client_secret_key,
        )

    def drive_path(self, *parts: str) -> str:
        """
        :attr:`file_server_base_path` joined with *parts*, as one
        drive-relative path.

        The single place path segments are concatenated, so no caller has to
        remember whether the base carries a trailing slash or the document
        library prefix. Empty and ``None``-ish parts are skipped, and each
        part's own leading/trailing slashes are absorbed, so
        ``drive_path("MyApp", "/scripts_original/")`` and
        ``drive_path("MyApp/scripts_original")`` agree.
        """
        segments = [self.file_server_base_path, *parts]
        cleaned = [segment.strip().strip("/") for segment in segments if segment]
        return "/".join(part for part in cleaned if part)

    def list_id(self, kind: str) -> str:
        """
        The configured id of one of the four driving lists — *kind* being
        ``"requests"``, ``"conversions"``, ``"xref"`` or ``"complexity"``
        (:data:`LIST_KINDS`).

        Raises
        ------
        SharePointError
            The id is not configured. The message names the config key to
            set, which beats Graph reporting that ``None`` is not a list.
        """
        try:
            key = LIST_KINDS[kind]
        except KeyError:
            raise SharePointError(
                f"unknown SharePoint list kind {kind!r}; expected one of "
                f"{', '.join(sorted(LIST_KINDS))}"
            ) from None
        value = getattr(self, key)
        if not value:
            raise SharePointError(
                f"no SharePoint {kind} list configured: set "
                f"SHAREPOINT_{key.upper()} or sharepoint.{key} in config.json"
            )
        return value


def _resolve_scopes() -> tuple[str, ...]:
    """
    Graph scopes from ``SHAREPOINT_SCOPES`` (space- or comma-separated) or the
    ``sharepoint.scopes`` config list, defaulting to
    :data:`GRAPH_DEFAULT_SCOPE`. A wrong-typed or non-string-list config entry
    degrades to the default with a WARNING.
    """
    env = os.environ.get("SHAREPOINT_SCOPES")
    if env:
        return tuple(env.replace(",", " ").split())
    configured = get_typed_value("sharepoint", "scopes", list)
    if configured is None:
        return (GRAPH_DEFAULT_SCOPE,)
    if not configured or not all(isinstance(s, str) for s in configured):
        logger.warning(
            "sharepoint: config.json sharepoint.scopes must be a non-empty list "
            "of strings; ignoring it (default Graph scope applies)"
        )
        return (GRAPH_DEFAULT_SCOPE,)
    return tuple(configured)


class _GraphTokenCredential:
    """
    A synchronous ``azure.core.credentials.TokenCredential`` that mints Graph
    tokens through :mod:`app_config.azure`.

    The Graph SDK's auth provider calls :meth:`get_token`; this forwards to the
    injected *token_provider* (by default :func:`app_config.azure.get_token`)
    and wraps the result in an ``AccessToken``. The stamped expiry is
    deliberately short (:data:`_CREDENTIAL_TTL`): the ``azure`` client already
    caches and refreshes tokens against their real lifetime, so this only needs
    the SDK to re-consult that cache rather than hold a token of unknown
    lifetime for longer than it is valid.
    """

    def __init__(
        self,
        token_provider: Callable[[tuple[str, ...]], str],
        default_scopes: tuple[str, ...],
    ) -> None:
        self._token_provider = token_provider
        self._default_scopes = default_scopes

    def get_token(self, *scopes: str, **_kwargs: Any) -> Any:
        from azure.core.credentials import AccessToken

        wanted = tuple(scopes) or self._default_scopes
        token = self._token_provider(wanted)
        return AccessToken(token, int(time.time()) + _CREDENTIAL_TTL)


def _token_provider(config: SharePointConfig) -> Callable[[tuple[str, ...]], str]:
    """
    The function that mints Graph tokens for *config*.

    With a secret scope configured, SharePoint authenticates as its **own**
    service principal, read from that scope under
    :attr:`~SharePointConfig.tenant_id_key` and friends — a different identity
    from the one that reaches Vault, which is how the reference deployment is
    set up. With no scope anywhere, the shared :mod:`app_config.azure`
    identity is used, which is the local/dev path.
    """
    if config.secret_scope:
        keys = config.secret_keys
        scope = config.secret_scope

        def _as_sharepoint_principal(scopes: tuple[str, ...]) -> str:
            from .azure import get_databricks_client

            return get_databricks_client(scope, keys=keys).get_token(scopes)

        return _as_sharepoint_principal

    def _as_shared_identity(scopes: tuple[str, ...]) -> str:
        from .azure import get_token

        return get_token(scopes=scopes)

    return _as_shared_identity


def _drive_item_to_dict(item: Any) -> dict[str, Any]:
    """A Graph ``DriveItem`` flattened to the fields a directory listing needs."""
    folder = getattr(item, "folder", None)
    modified = getattr(item, "last_modified_date_time", None)
    return {
        "name": getattr(item, "name", None),
        "id": getattr(item, "id", None),
        "is_folder": folder is not None,
        "size": getattr(item, "size", None),
        "web_url": getattr(item, "web_url", None),
        "last_modified": modified.isoformat() if modified is not None else None,
        "child_count": getattr(folder, "child_count", None),
    }


def _list_item_to_dict(item: Any) -> dict[str, Any]:
    """
    A Graph ``ListItem`` flattened to ``{id, web_url, fields}`` — ``fields``
    holding the column values a PowerApps grid binds to.
    """
    fields = getattr(item, "fields", None)
    data = getattr(fields, "additional_data", None) if fields is not None else None
    return {
        "id": getattr(item, "id", None),
        "web_url": getattr(item, "web_url", None),
        "fields": dict(data) if data else {},
    }


class SharePointClient:
    """
    Reads and writes a SharePoint document library (and reads its lists) over
    Microsoft Graph.

    Parameters
    ----------
    config : SharePointConfig | None
        Site/library settings. ``None`` (default) uses
        :meth:`SharePointConfig.from_env`.
    client : Any | None
        A pre-built ``msgraph.GraphServiceClient`` (or a duck-typed stand-in) to
        use as-is. When given, no client is constructed and ``msgraph-sdk`` need
        not be installed — the escape hatch for custom credentials and tests.
    credential : Any | None
        A ``TokenCredential`` to authenticate with, overriding the default
        :class:`_GraphTokenCredential` (which delegates to
        :mod:`app_config.azure`). Ignored when *client* is supplied.

    The Graph client is built lazily on first :attr:`client` access, so
    constructing a :class:`SharePointClient` never touches the network or
    requires ``msgraph-sdk`` to be importable.
    """

    def __init__(
        self,
        config: SharePointConfig | None = None,
        *,
        client: Any | None = None,
        credential: Any | None = None,
    ) -> None:
        self.config = config if config is not None else SharePointConfig.from_env()
        self._client = client
        self._credential = credential
        self._resolved_drive_id: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- client construction ------------------------------------------------

    @property
    def client(self) -> Any:
        """The underlying ``GraphServiceClient``, built on demand."""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        # Validate the identity before importing the SDK so a misconfiguration
        # reports the real problem instead of a missing-dependency error.
        from .azure import AzureAuthConfig

        azure_config = AzureAuthConfig.from_env()
        if not self.config.secret_scope and not (
            azure_config.tenant_id and azure_config.client_id
        ):
            raise SharePointError(
                "no Entra ID identity for SharePoint: point "
                "SHAREPOINT_SECRET_SCOPE (or DATABRICKS_SECRET_SCOPE) at the "
                "scope holding SharePoint's service principal, or set "
                "AZURE_TENANT_ID and AZURE_CLIENT_ID (plus AZURE_CLIENT_SECRET "
                "or a certificate) to mint a Microsoft Graph token with the "
                "shared identity"
            )
        credential = self._credential or _GraphTokenCredential(
            _token_provider(self.config), self.config.scopes
        )
        try:
            import httpx
            from kiota_authentication_azure.azure_identity_authentication_provider import (
                AzureIdentityAuthenticationProvider,
            )
            from msgraph.graph_request_adapter import GraphRequestAdapter
            from msgraph.graph_service_client import GraphServiceClient
            from msgraph_core import GraphClientFactory
        except ImportError as exc:
            raise SharePointError(
                "msgraph-sdk is required for SharePoint access; install it with "
                "'pip install \"sas-parser[sharepoint]\"'"
            ) from exc
        auth_provider = AzureIdentityAuthenticationProvider(
            credentials=credential, scopes=list(self.config.scopes)
        )
        http_client = GraphClientFactory.create_with_default_middleware(
            client=httpx.AsyncClient(timeout=self.config.timeout)
        )
        adapter = GraphRequestAdapter(auth_provider, http_client)
        logger.info(
            f"SharePointClient: built Graph client for site "
            f"{self.config.resolved_site_id!r} drive "
            f"{self.config.drive_id or '(site default)'!r} "
            f"(timeout={self.config.timeout}s)"
        )
        return GraphServiceClient(request_adapter=adapter)

    # -- async plumbing -----------------------------------------------------

    def _run(self, coro: Any) -> Any:
        """
        Drive *coro* to completion on this client's private event loop.

        Raising rather than nesting when a loop is already running keeps the
        ``httpx`` connection pool bound to one loop; an async caller should
        drive the Graph SDK directly instead of through this blocking facade.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            coro.close()
            raise SharePointError(
                "SharePointClient's synchronous methods cannot be called from a "
                "running event loop; call them from synchronous code, or use the "
                "msgraph-sdk client directly for async access"
            )
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def close(self) -> None:
        """Close the private event loop (if any). The client is single-use after."""
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
        self._loop = None

    # -- addressing ---------------------------------------------------------

    def _site_id(self) -> str:
        site_id = self.config.resolved_site_id
        if not site_id:
            raise SharePointError(
                "no SharePoint site configured: set SHAREPOINT_SITE_ID, or "
                "SHAREPOINT_SITE_HOSTNAME and SHAREPOINT_SITE_PATH (or the "
                "sharepoint.* equivalents in config.json)"
            )
        return site_id

    def _drive_id(self) -> str:
        """
        The target document library id: the configured
        :attr:`~SharePointConfig.drive_id`, else the site's default library,
        resolved once and cached on this client.
        """
        if self.config.drive_id:
            return self.config.drive_id
        resolved = self._resolved_drive_id
        if resolved is None:
            site_id = self._site_id()
            drive = self._run(self.client.sites.by_site_id(site_id).drive.get())
            resolved = getattr(drive, "id", None)
            if not resolved:
                raise SharePointError(
                    f"site {site_id!r} has no accessible default document library; "
                    f"set SHAREPOINT_DRIVE_ID to target one explicitly"
                )
            self._resolved_drive_id = resolved
        return resolved

    def _drive(self) -> Any:
        return self.client.drives.by_drive_id(self._drive_id())

    def _item(self, path: str) -> Any:
        return self._drive().items.by_drive_item_id(_drive_item_id(path))

    # -- operations ---------------------------------------------------------

    def get_token(self, scopes: tuple[str, ...] | list[str] | None = None) -> str:
        """
        A Microsoft Graph access token for the configured identity — the
        authentication step, exposed for callers that need the bearer token
        directly. Uses the same provider the Graph client authenticates with
        (:func:`_token_provider`), so this is never a *different* identity
        from the one the operations use.

        Raises
        ------
        SharePointError
            The Entra ID login failed.
        """
        from .azure import AzureAuthError

        wanted = tuple(scopes) if scopes else self.config.scopes
        try:
            return _token_provider(self.config)(wanted)
        except AzureAuthError as exc:
            raise SharePointError(
                f"could not mint a Microsoft Graph token for SharePoint: {exc}"
            ) from exc

    def list_directory(self, path: str = "") -> list[dict[str, Any]]:
        """
        The immediate children of the folder at *path* (default: the library
        root), each as a dict from :func:`_drive_item_to_dict`. Paged results
        are followed and concatenated.

        Raises
        ------
        SharePointError
            The folder is absent, or the listing otherwise fails.
        """
        try:
            return self._run(self._collect_children(path))
        except SharePointError:
            raise
        except Exception as exc:
            raise SharePointError(
                f"could not list SharePoint directory {path or '/'!r}: {exc}"
            ) from exc

    def list_files(
        self, path: str = "", extensions: tuple[str, ...] | list[str] | None = None
    ) -> list[dict[str, Any]]:
        """
        The *files* directly inside the folder at *path*, optionally filtered
        by extension.

        Folders are dropped, so a caller looking for source files does not have
        to. *extensions* are matched case-insensitively with or without a
        leading dot, so ``("sas", "txt")`` and ``(".SAS", ".TXT")`` agree;
        ``None`` (default) keeps every file. Each entry additionally carries a
        ``"path"`` key — the item's drive-relative path — since that is what
        :meth:`download_file_as_text` and the domain layers pass around.

        Raises
        ------
        SharePointError
            The folder is absent, or the listing otherwise fails.
        """
        wanted = (
            {ext.strip().lstrip(".").lower() for ext in extensions}
            if extensions is not None
            else None
        )
        folder = path.strip().strip("/")
        files: list[dict[str, Any]] = []
        for entry in self.list_directory(path):
            if entry["is_folder"]:
                continue
            name = entry.get("name") or ""
            if wanted is not None:
                _, dot, suffix = name.rpartition(".")
                if not dot or suffix.lower() not in wanted:
                    continue
            files.append({**entry, "path": f"{folder}/{name}" if folder else name})
        return files

    async def _collect_children(self, path: str) -> list[dict[str, Any]]:
        builder = self._item(path).children
        response = await builder.get()
        items: list[dict[str, Any]] = []
        while response is not None:
            for item in response.value or []:
                items.append(_drive_item_to_dict(item))
            next_link = getattr(response, "odata_next_link", None)
            if not next_link:
                break
            response = await builder.with_url(next_link).get()
        return items

    def read_file(self, path: str) -> bytes:
        """
        The raw bytes of the file at *path*.

        Raises
        ------
        SharePointError
            The file is absent, is a folder, or the download otherwise fails.
        """
        try:
            content = self._run(self._item(path).content.get())
        except SharePointError:
            raise
        except Exception as exc:
            raise SharePointError(
                f"could not read SharePoint file {path!r}: {exc}"
            ) from exc
        if content is None:
            raise SharePointError(f"SharePoint file {path!r} returned no content")
        return content

    def download_file_as_text(self, path: str, *, encoding: str = "utf-8") -> str:
        """
        The file at *path* decoded as text.

        The encoding is explicit and undecodable bytes are replaced rather than
        raising: these are SAS sources typed on assorted machines over assorted
        decades, and one stray byte must not lose the whole file. ``utf-8-sig``
        by effect — a BOM is stripped — since Windows editors add one.

        Raises
        ------
        SharePointError
            The file is absent, is a folder, or the download otherwise fails.
        """
        raw = self.read_file(path)
        if encoding.lower() in ("utf-8", "utf8"):
            encoding = "utf-8-sig"  # also accepts BOM-less files
        return raw.decode(encoding, errors="replace")

    def read_json_text(self, path: str, *, encoding: str = "utf-8") -> Any:
        """
        The JSON document at *path*, parsed.

        Raises
        ------
        SharePointError
            The file is absent, the download fails, or the content is not
            valid JSON — the last of which names the path, because "expecting
            value: line 1" on its own says nothing about which file.
        """
        text = self.download_file_as_text(path, encoding=encoding)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise SharePointError(
                f"SharePoint file {path!r} is not valid JSON: {exc}"
            ) from exc

    def write_file(self, path: str, content: bytes | str) -> dict[str, Any]:
        """
        Create or replace the file at *path* with *content* (``str`` is encoded
        as UTF-8), returning the resulting item as a dict.

        A simple upload — suitable for the small files a PowerApps flow handles;
        Graph caps a single PUT at a few hundred MB, above which an upload
        session (not implemented here) is required.

        Raises
        ------
        SharePointError
            The path names no file, or the upload fails.
        """
        clean = path.strip().strip("/")
        if not clean:
            raise SharePointError("write_file needs a file path, not the library root")
        body = content.encode("utf-8") if isinstance(content, str) else content
        try:
            item = self._run(self._item(clean).content.put(body))
        except SharePointError:
            raise
        except Exception as exc:
            raise SharePointError(
                f"could not write SharePoint file {path!r}: {exc}"
            ) from exc
        return _drive_item_to_dict(item)

    def upload_file(
        self, folder: str, name: str, content: bytes | str
    ) -> dict[str, Any]:
        """
        :meth:`write_file` addressed as *folder* + *name* rather than as one
        joined path.

        The shape the domain layers want: they compose a folder from a
        configured base (:meth:`SharePointConfig.drive_path`) and then upload
        files into it by name, so joining the two here means no caller has to
        get slashes right. ``upload_file("a/b", "c.sas", ...)`` and
        ``write_file("a/b/c.sas", ...)`` are the same call.

        Raises
        ------
        SharePointError
            *name* is empty, or the upload fails.
        """
        leaf = name.strip().strip("/")
        if not leaf:
            raise SharePointError("upload_file needs a file name")
        parent = folder.strip().strip("/")
        return self.write_file(f"{parent}/{leaf}" if parent else leaf, content)

    def create_folder(self, path: str) -> dict[str, Any]:
        """
        Ensure the folder at *path* exists, creating it and any missing parents,
        and return it.

        **Idempotent**, unlike :meth:`create_directory`: the upload flows call
        this unconditionally before writing into a folder, so an existing one
        is a success rather than a conflict. Each segment is created with
        ``conflict_behavior="replace"``, which for a *folder* body means "use
        the existing one" — Graph does not empty it.

        Raises
        ------
        SharePointError
            *path* names no folder, or a segment could not be created.
        """
        clean = path.strip().strip("/")
        if not clean:
            raise SharePointError(
                "create_folder needs a folder path, not the library root"
            )
        item: dict[str, Any] | None = None
        walked = ""
        for segment in clean.split("/"):
            if not segment:
                continue
            walked = f"{walked}/{segment}" if walked else segment
            item = self.create_directory(walked, conflict_behavior="replace")
        assert item is not None  # clean is non-empty, so the loop ran
        return item

    def create_directory(
        self, path: str, *, conflict_behavior: str = "fail"
    ) -> dict[str, Any]:
        """
        Create the folder at *path* (its parent must exist), returning the new
        folder as a dict.

        Parameters
        ----------
        conflict_behavior : str
            What Graph does when a child of that name already exists: ``"fail"``
            (default), ``"replace"``, or ``"rename"``.

        Raises
        ------
        SharePointError
            The path names no folder, the parent is missing, or (with
            ``conflict_behavior="fail"``) the folder already exists.
        """
        clean = path.strip().strip("/")
        if not clean:
            raise SharePointError(
                "create_directory needs a folder path, not the library root"
            )
        parent, _, name = clean.rpartition("/")
        try:
            item = self._run(self._create_folder(parent, name, conflict_behavior))
        except SharePointError:
            raise
        except Exception as exc:
            raise SharePointError(
                f"could not create SharePoint directory {path!r}: {exc}"
            ) from exc
        return _drive_item_to_dict(item)

    async def _create_folder(
        self, parent: str, name: str, conflict_behavior: str
    ) -> Any:
        from msgraph.generated.models.drive_item import DriveItem
        from msgraph.generated.models.folder import Folder

        body = DriveItem(
            name=name,
            folder=Folder(),
            additional_data={"@microsoft.graph.conflictBehavior": conflict_behavior},
        )
        return await self._item(parent).children.post(body)

    def list_items(
        self,
        list_name: str,
        *,
        select: list[str] | None = None,
        expand: str = "fields",
        top: int | None = None,
        filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Read items from the SharePoint list *list_name* (its id or display
        name), each flattened to ``{id, web_url, fields}`` by
        :func:`_list_item_to_dict`. Paged results are followed and
        concatenated.

        This is the name the domain layers and the reference system both use.

        Parameters
        ----------
        select, top, filter : optional
            Standard Graph query options ``$select`` / ``$top`` / ``$filter``.
        expand : str
            What to expand, default ``"fields"`` so column values come back in
            one round trip.

        Raises
        ------
        SharePointError
            No site is configured, the list is absent, or the read fails.
        """
        site_id = self._site_id()
        try:
            return self._run(
                self._collect_list_items(
                    site_id, list_name, select, expand, top, filter
                )
            )
        except SharePointError:
            raise
        except Exception as exc:
            raise SharePointError(
                f"could not read SharePoint list {list_name!r}: {exc}"
            ) from exc

    async def _collect_list_items(
        self,
        site_id: str,
        list_name: str,
        select: list[str] | None,
        expand: str,
        top: int | None,
        filter: str | None,
    ) -> list[dict[str, Any]]:
        from kiota_abstractions.base_request_configuration import RequestConfiguration
        from msgraph.generated.sites.item.lists.item.items.items_request_builder import (
            ItemsRequestBuilder,
        )

        builder = (
            self.client.sites.by_site_id(site_id).lists.by_list_id(list_name).items
        )
        query = ItemsRequestBuilder.ItemsRequestBuilderGetQueryParameters(
            expand=[expand] if expand else None,
            select=select,
            top=top,
            filter=filter,
        )
        config = RequestConfiguration(query_parameters=query)
        response = await builder.get(config)
        items: list[dict[str, Any]] = []
        while response is not None:
            for item in response.value or []:
                items.append(_list_item_to_dict(item))
            next_link = getattr(response, "odata_next_link", None)
            if not next_link:
                break
            response = await builder.with_url(next_link).get()
        return items

    # -- list items ---------------------------------------------------------

    def get_list_item(self, list_id: str, item_id: str | int) -> dict[str, Any]:
        """
        One item from *list_id* by its ``ID``, flattened to
        ``{id, web_url, fields}`` like :meth:`list_items`.

        Raises
        ------
        SharePointError
            No site is configured, the list or item is absent, or the read
            fails.
        """
        site_id = self._site_id()
        try:
            item = self._run(
                self.client.sites.by_site_id(site_id)
                .lists.by_list_id(list_id)
                .items.by_list_item_id(str(item_id))
                .get()
            )
        except SharePointError:
            raise
        except Exception as exc:
            raise SharePointError(
                f"could not read item {item_id!r} of SharePoint list "
                f"{list_id!r}: {exc}"
            ) from exc
        if item is None:
            raise SharePointError(
                f"SharePoint list {list_id!r} has no item {item_id!r}"
            )
        return _list_item_to_dict(item)

    def update_list_item(
        self, list_id: str, item_id: str | int, fields: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Write *fields* onto one item of *list_id*, returning the updated column
        values.

        *fields* is keyed by **internal** column name, which is what the list's
        columns are addressed by and often differs from the display name (a
        space becomes ``_x0020_``, a renamed column keeps its original name).
        A partial mapping is a partial update — columns not named are left
        alone.

        Raises
        ------
        SharePointError
            *fields* is empty, no site is configured, the list or item is
            absent, or the write fails.
        """
        if not fields:
            raise SharePointError(
                f"update_list_item was given no fields to write to item "
                f"{item_id!r} of list {list_id!r}"
            )
        site_id = self._site_id()
        try:
            updated = self._run(self._update_fields(site_id, list_id, item_id, fields))
        except SharePointError:
            raise
        except Exception as exc:
            raise SharePointError(
                f"could not update item {item_id!r} of SharePoint list "
                f"{list_id!r}: {exc}"
            ) from exc
        data = getattr(updated, "additional_data", None)
        logger.info(
            f"update_list_item: wrote {sorted(fields)} to item {item_id} of "
            f"list {list_id!r}"
        )
        return dict(data) if data else {}

    async def _update_fields(
        self, site_id: str, list_id: str, item_id: str | int, fields: dict[str, Any]
    ) -> Any:
        from msgraph.generated.models.field_value_set import FieldValueSet

        return await (
            self.client.sites.by_site_id(site_id)
            .lists.by_list_id(list_id)
            .items.by_list_item_id(str(item_id))
            .fields.patch(FieldValueSet(additional_data=dict(fields)))
        )


# One client per process, mirroring app_config's config cache.
_client_cache: SharePointClient | None = None


def get_sharepoint_client() -> SharePointClient:
    """The process-wide :class:`SharePointClient` (built from the environment)."""
    global _client_cache
    if _client_cache is None:
        _client_cache = SharePointClient()
    return _client_cache


def clear_cache() -> None:
    """Drop the cached client so the next access rebuilds it (for tests)."""
    global _client_cache
    if _client_cache is not None:
        _client_cache.close()
    _client_cache = None


# ---------------------------------------------------------------------------
# Resolution helpers for the domain layers
#
# Every domain function takes an optional client and config so a run can be
# driven against a fake transport, which means every one of them needs the
# same two lines to turn "or None" into "the shared one". Those two lines were
# copied into five modules across three packages; they live here instead —
# transport-level and domain-free, so sharing them adds no coupling.
# ---------------------------------------------------------------------------


def resolve_client(client: Any | None = None) -> Any:
    """*client*, or the process-wide one when it is ``None``.

    The import stays inside the call in the domain layers that use it, so this
    keeps the same property: :func:`get_sharepoint_client` is only reached when
    no client was passed, and a test that injects a fake never builds a real
    Graph client (or needs ``msgraph-sdk`` installed).
    """
    return client if client is not None else get_sharepoint_client()


def resolve_config(config: "SharePointConfig | None" = None) -> "SharePointConfig":
    """*config*, or one resolved from the environment when it is ``None``."""
    return config if config is not None else SharePointConfig.from_env()


def field_text(fields: dict[str, Any], column: str) -> str | None:
    """
    A list column's value as stripped text, or ``None`` when blank or absent.

    Blank-is-absent is the rule every projection here wants: SharePoint returns
    an empty string for a column that was never filled in, and a caller
    checking ``if value:`` and one checking ``if value is not None:`` would
    otherwise disagree about the same row.
    """
    value = fields.get(column)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def upload_into(
    transport: Any, folder: str, name: str, content: str | bytes
) -> str:
    """
    Create *folder*, upload *name* into it, and return the path it landed at.

    Graph's simple upload does not create missing parents, so every upload site
    had the same create-then-upload-then-join sequence. :meth:`create_folder`
    is idempotent, so calling it before each write is safe rather than merely
    tolerable.
    """
    transport.create_folder(folder)
    transport.upload_file(folder, name, content)
    return f"{folder}/{name}"


def project_rows(
    rows: list[dict[str, Any]],
    formatter: Callable[[dict[str, Any]], Any],
    *,
    label: str,
) -> list[Any]:
    """
    Project *rows* through *formatter*, skipping malformed ones with a WARNING.

    One bad row must not hide every good one — a list is edited by hand, so a
    row missing the field its projection requires is a normal occurrence rather
    than an outage. *label* names the caller in the log line.
    """
    out: list[Any] = []
    for row in rows:
        try:
            out.append(formatter(row))
        except SharePointError as exc:
            logger.warning(f"{label}: skipping a malformed row: {exc}")
    if len(out) != len(rows):
        logger.info(f"{label}: projected {len(out)} of {len(rows)} row(s)")
    return out
