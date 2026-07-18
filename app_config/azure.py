"""Microsoft Entra ID (Azure AD) authentication for acquiring access tokens.

Submodule of :mod:`app_config`, and a sibling of :mod:`app_config.vault`: where
``vault`` fetches secrets *stored* in Vault, this module turns an Entra ID
identity into a short-lived OAuth 2.0 access token for calling an Azure-fronted
API. :mod:`app_config.databricks` is the first consumer — it exchanges an Entra
ID token for workspace access when no personal access token is set.

Split of concerns
-----------------
* **Non-secret settings** — tenant, client (application) id, authority host,
  scopes, flow, and request timeout — resolve through
  :meth:`AzureAuthConfig.from_env`, which reads the standard Azure environment
  variables first (``AZURE_TENANT_ID``, ``AZURE_CLIENT_ID``,
  ``AZURE_AUTHORITY_HOST``, ``AZURE_CLIENT_CERTIFICATE_PATH``) and falls back
  to the optional ``azure`` section of ``config.json`` (via
  :func:`app_config.get_value` / :func:`app_config.get_typed_value`, so a
  wrong-typed entry degrades to the hard default with a WARNING rather than
  crashing).
* **Secrets** — the client secret and the certificate private key — come *only*
  from the environment (``AZURE_CLIENT_SECRET``) or from a key file on disk.
  The secret is held in a field marked ``repr=False`` so it never appears in a
  ``repr`` or a log line, and the certificate's private key is read at login
  time and never stored on the config.

Supported flows
---------------
``client_credentials``
    Service-principal login, authenticated by either a client secret or a
    certificate. The unattended default, and the only flow suitable for jobs.
``device_code``
    Interactive user login for a local workstation: the verification URL and
    code are emitted at WARNING so they show up without debug logging. Blocks
    until the user completes the login in a browser, so never use it in CI.

The flow is inferred from which credentials are present
(:attr:`AzureAuthConfig.auth_flow`); set ``azure.flow`` in ``config.json`` to
pin it explicitly.

Dependency
----------
The ``msal`` library is an *optional* dependency (extra ``azure``):
``pip install "sas-parser[azure]"``. It is imported lazily inside
:meth:`AzureAuthClient._build_app`, so ``import app_config.azure`` costs
nothing and keeps ``app_config`` the dependency-free leaf the rest of the
package relies on. Only actually acquiring a token requires ``msal``.

Typical use
-----------
    from app_config.azure import get_token

    token = get_token()                                  # configured scopes
    token = get_token(scopes=("https://graph.microsoft.com/.default",))

Tokens are cached per scope set until shortly before they expire
(:data:`_EXPIRY_SKEW`), and the module-level helpers reuse one
:class:`AzureAuthClient` per process (:func:`get_azure_client`); call
:func:`clear_cache` to force a fresh login after the environment changes
(tests do).

Logger name: ``app_config.azure``.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import get_typed_value, get_value

logger = logging.getLogger(__name__)

DEFAULT_AUTHORITY_HOST = "https://login.microsoftonline.com"
DEFAULT_TIMEOUT = 30.0

FLOW_CLIENT_CREDENTIALS = "client_credentials"
FLOW_DEVICE_CODE = "device_code"
_FLOWS = frozenset({FLOW_CLIENT_CREDENTIALS, FLOW_DEVICE_CODE})

# Re-acquire this many seconds before the token actually expires, so a token
# handed out here stays valid for the length of the call it is used on.
_EXPIRY_SKEW = 60.0


class AzureAuthError(RuntimeError):
    """Entra ID is misconfigured, unreachable, or rejected the credentials.

    A single error type so callers can ``except AzureAuthError`` around a token
    acquisition regardless of which stage failed; the message says which.
    """


def _resolve_scopes() -> tuple[str, ...]:
    """
    Scopes from ``AZURE_SCOPES`` (space- or comma-separated) or the
    ``azure.scopes`` config list. Empty when unset — a caller that passes
    scopes to :meth:`AzureAuthClient.get_token` needs neither.
    """
    env = os.environ.get("AZURE_SCOPES")
    if env:
        return tuple(env.replace(",", " ").split())
    configured = get_typed_value("azure", "scopes", list)
    if configured is None:
        return ()
    if not all(isinstance(s, str) for s in configured):
        logger.warning(
            "azure: config.json azure.scopes must be a list of strings; "
            "ignoring it (no default scopes apply)"
        )
        return ()
    return tuple(configured)


def _resolve_flow() -> str | None:
    """
    An explicitly pinned flow from ``azure.flow``, or ``None`` to infer it from
    the credentials present. An unrecognised name degrades to inference with a
    WARNING, the same rule as a wrong-typed config entry.
    """
    configured = get_typed_value("azure", "flow", str)
    if configured is None:
        return None
    if configured not in _FLOWS:
        logger.warning(
            f"azure: config.json azure.flow {configured!r} is not one of "
            f"{'/'.join(sorted(_FLOWS))}; ignoring it (the flow is inferred "
            f"from the credentials present)"
        )
        return None
    return configured


@dataclass
class AzureAuthConfig:
    """
    Everything :class:`AzureAuthClient` needs to log in to Entra ID.

    Construct it directly to pin values explicitly, or call :meth:`from_env`
    for the standard environment-then-``config.json`` resolution.
    :attr:`client_secret` is ``repr=False`` and is never logged.

    Attributes
    ----------
    tenant_id : str | None
        Directory (tenant) GUID, or ``organizations`` / ``common``.
        ``AZURE_TENANT_ID`` / ``config.json`` ``azure.tenant_id``. Required;
        a missing tenant raises :class:`AzureAuthError`.
    client_id : str | None
        Application (client) GUID of the app registration.
        ``AZURE_CLIENT_ID`` / ``azure.client_id``. Required.
    authority_host : str
        Login endpoint, ``azure.authority_host`` / ``AZURE_AUTHORITY_HOST``,
        default ``https://login.microsoftonline.com``. Override for sovereign
        clouds (e.g. ``https://login.microsoftonline.us``).
    scopes : tuple[str, ...]
        Default scopes for :meth:`AzureAuthClient.get_token`. For the
        client-credentials flow these are resource-scoped ``.default`` values
        (``<resource>/.default``), not delegated permissions.
        ``AZURE_SCOPES`` / ``azure.scopes``.
    flow : str | None
        Pin the flow to ``"client_credentials"`` or ``"device_code"``.
        ``azure.flow``; ``None`` (default) infers it — see :attr:`auth_flow`.
    timeout : float
        Per-request timeout in seconds. ``azure.timeout``, default ``30``.
        Not applied to the user-wait leg of the device-code flow, which is
        bounded by the code's own expiry.
    client_secret : str | None
        App-registration client secret. ``AZURE_CLIENT_SECRET`` only — never
        from ``config.json``.
    certificate_path : str | None
        PEM file holding the private key for certificate-based
        client-credentials login, used when no :attr:`client_secret` is set.
        ``AZURE_CLIENT_CERTIFICATE_PATH`` / ``azure.certificate_path``. The
        key is read at login time and never held on this config.
    certificate_thumbprint : str | None
        SHA-1 thumbprint of that certificate (hex, no separators), as shown in
        the app registration. ``AZURE_CLIENT_CERTIFICATE_THUMBPRINT`` /
        ``azure.certificate_thumbprint``. Required alongside
        :attr:`certificate_path`.
    """

    tenant_id: str | None = None
    client_id: str | None = None
    authority_host: str = DEFAULT_AUTHORITY_HOST
    scopes: tuple[str, ...] = ()
    flow: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    client_secret: str | None = field(default=None, repr=False)
    certificate_path: str | None = None
    certificate_thumbprint: str | None = None

    @classmethod
    def from_env(cls) -> "AzureAuthConfig":
        """
        Resolve settings from the standard Azure environment variables,
        falling back to the ``azure`` section of ``config.json`` for the
        non-secret knobs. The client secret is read from the environment only.
        """
        return cls(
            tenant_id=(
                os.environ.get("AZURE_TENANT_ID") or get_value("azure", "tenant_id")
            ),
            client_id=(
                os.environ.get("AZURE_CLIENT_ID") or get_value("azure", "client_id")
            ),
            authority_host=(
                os.environ.get("AZURE_AUTHORITY_HOST")
                or get_value("azure", "authority_host", DEFAULT_AUTHORITY_HOST)
            ).rstrip("/"),
            scopes=_resolve_scopes(),
            flow=_resolve_flow(),
            timeout=get_typed_value("azure", "timeout", (int, float), DEFAULT_TIMEOUT),
            client_secret=os.environ.get("AZURE_CLIENT_SECRET"),
            certificate_path=(
                os.environ.get("AZURE_CLIENT_CERTIFICATE_PATH")
                or get_value("azure", "certificate_path")
            ),
            certificate_thumbprint=(
                os.environ.get("AZURE_CLIENT_CERTIFICATE_THUMBPRINT")
                or get_value("azure", "certificate_thumbprint")
            ),
        )

    @property
    def authority(self) -> str:
        """The MSAL authority URL, ``<authority_host>/<tenant_id>``."""
        return f"{self.authority_host}/{self.tenant_id}"

    @property
    def has_certificate(self) -> bool:
        """True when both halves of a certificate credential are configured."""
        return bool(self.certificate_path and self.certificate_thumbprint)

    @property
    def auth_flow(self) -> str | None:
        """
        The flow to use: :attr:`flow` when pinned, else
        ``"client_credentials"`` when a client secret or a certificate is
        configured, else ``"device_code"`` (public client, user login), else
        ``None`` (no usable identity — :attr:`client_id` is missing).
        """
        if self.flow:
            return self.flow
        if self.client_secret or self.has_certificate:
            return FLOW_CLIENT_CREDENTIALS
        if self.client_id:
            return FLOW_DEVICE_CODE
        return None


def _client_credential(config: AzureAuthConfig) -> Any:
    """
    The MSAL ``client_credential`` for the client-credentials flow: the secret
    string, or a ``{"private_key", "thumbprint"}`` mapping read from
    :attr:`~AzureAuthConfig.certificate_path`.
    """
    if config.client_secret:
        return config.client_secret
    certificate_path = config.certificate_path
    if certificate_path is None:
        raise AzureAuthError(
            "certificate-based login needs azure.certificate_path plus "
            "azure.certificate_thumbprint"
        )
    try:
        private_key = Path(certificate_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise AzureAuthError(
            f"could not read Azure certificate '{certificate_path}': {exc}"
        ) from exc
    return {"private_key": private_key, "thumbprint": config.certificate_thumbprint}


class AzureAuthClient:
    """
    Acquires Entra ID access tokens for the configured identity.

    Parameters
    ----------
    config : AzureAuthConfig | None
        Identity and flow settings. ``None`` (default) uses
        :meth:`AzureAuthConfig.from_env`.
    app : Any | None
        A pre-built MSAL application (``ConfidentialClientApplication`` /
        ``PublicClientApplication``, or a duck-typed stand-in) to use as-is.
        When given, no application is constructed and ``msal`` need not be
        installed — the escape hatch for custom credential types and tests.

    The MSAL application is built lazily on first :attr:`app` access, so
    constructing an :class:`AzureAuthClient` never touches the network or
    requires ``msal`` to be importable.
    """

    def __init__(
        self, config: AzureAuthConfig | None = None, *, app: Any | None = None
    ) -> None:
        self.config = config if config is not None else AzureAuthConfig.from_env()
        self._app = app
        # (scopes) -> (access_token, expires_at_epoch_seconds)
        self._tokens: dict[tuple[str, ...], tuple[str, float]] = {}

    @property
    def app(self) -> Any:
        """The underlying MSAL application, built on demand."""
        if self._app is None:
            self._app = self._build_app(self.config)
        return self._app

    @staticmethod
    def _build_app(config: AzureAuthConfig) -> Any:
        # Validate config before importing msal so a misconfiguration reports
        # the real problem instead of a missing-dependency error.
        if not config.tenant_id:
            raise AzureAuthError(
                "no Azure tenant configured: set AZURE_TENANT_ID or "
                "azure.tenant_id in config.json"
            )
        if not config.client_id:
            raise AzureAuthError(
                "no Azure client id configured: set AZURE_CLIENT_ID or "
                "azure.client_id in config.json"
            )
        flow = config.auth_flow
        if flow == FLOW_CLIENT_CREDENTIALS and not (
            config.client_secret or config.has_certificate
        ):
            raise AzureAuthError(
                "the client_credentials flow needs a credential: set "
                "AZURE_CLIENT_SECRET, or azure.certificate_path plus "
                "azure.certificate_thumbprint"
            )
        try:
            import msal
        except ImportError as exc:
            raise AzureAuthError(
                "msal is required for Entra ID authentication; install it "
                "with 'pip install \"sas-parser[azure]\"'"
            ) from exc
        if flow == FLOW_CLIENT_CREDENTIALS:
            app = msal.ConfidentialClientApplication(
                client_id=config.client_id,
                authority=config.authority,
                client_credential=_client_credential(config),
                timeout=config.timeout,
            )
        else:
            app = msal.PublicClientApplication(
                client_id=config.client_id,
                authority=config.authority,
                timeout=config.timeout,
            )
        credential = (
            "certificate"
            if flow == FLOW_CLIENT_CREDENTIALS and not config.client_secret
            else "secret"
            if flow == FLOW_CLIENT_CREDENTIALS
            else "user"
        )
        logger.info(
            f"AzureAuthClient: built {flow} app for client "
            f"{config.client_id} on {config.authority} (credential={credential})"
        )
        return app

    def get_token(self, scopes: tuple[str, ...] | list[str] | None = None) -> str:
        """
        An access token for *scopes*, from the per-process cache when one is
        still comfortably valid.

        Parameters
        ----------
        scopes : tuple[str, ...] | list[str] | None
            Scopes to request. ``None`` (default) uses the configured
            :attr:`~AzureAuthConfig.scopes`; with neither, this raises.

        Raises
        ------
        AzureAuthError
            No scopes, no usable identity, ``msal`` missing, or Entra ID
            rejected the request.
        """
        wanted = tuple(scopes) if scopes else self.config.scopes
        if not wanted:
            raise AzureAuthError(
                "no Azure scopes requested: pass scopes=, or set AZURE_SCOPES "
                "or azure.scopes in config.json"
            )
        cached = self._tokens.get(wanted)
        if cached is not None and cached[1] > time.time():
            return cached[0]
        result = self._acquire(wanted)
        token = result.get("access_token")
        if not token:
            # MSAL reports failures in-band rather than by raising.
            raise AzureAuthError(
                f"Azure token request for {' '.join(wanted)} failed: "
                f"{result.get('error', 'unknown_error')}: "
                f"{result.get('error_description', 'no description')}"
            )
        # expires_in is seconds-from-now; default low rather than trusting a
        # token of unknown lifetime for longer than one call.
        expires_in = float(result.get("expires_in", _EXPIRY_SKEW))
        self._tokens[wanted] = (token, time.time() + max(expires_in - _EXPIRY_SKEW, 0))
        logger.info(
            f"AzureAuthClient: acquired token for {' '.join(wanted)} "
            f"via {self.config.auth_flow} (expires_in={expires_in:.0f}s)"
        )
        return token

    def _acquire(self, scopes: tuple[str, ...]) -> dict[str, Any]:
        app = self.app
        flow = self.config.auth_flow
        try:
            if flow == FLOW_CLIENT_CREDENTIALS:
                # MSAL keeps its own token cache; this only round-trips when
                # the cached app-token has actually expired.
                return app.acquire_token_for_client(scopes=list(scopes))
            return self._acquire_by_device_code(app, scopes)
        except AzureAuthError:
            raise
        except Exception as exc:  # network / TLS / bad authority surface here
            raise AzureAuthError(
                f"could not reach Entra ID at {self.config.authority}: {exc}"
            ) from exc

    def _acquire_by_device_code(self, app: Any, scopes: tuple[str, ...]) -> dict:
        """Silent-first device-code login; blocks until the user completes it."""
        accounts = app.get_accounts()
        if accounts:
            silent = app.acquire_token_silent(list(scopes), account=accounts[0])
            if silent and silent.get("access_token"):
                return silent
        flow = app.initiate_device_flow(scopes=list(scopes))
        if "user_code" not in flow:
            raise AzureAuthError(
                f"could not start the Azure device-code flow: "
                f"{flow.get('error', 'unknown_error')}: "
                f"{flow.get('error_description', 'no description')}"
            )
        # WARNING, not INFO: the user has to read and act on this to proceed.
        logger.warning(f"AzureAuthClient: {flow['message']}")
        return app.acquire_token_by_device_flow(flow)


# One client (and its token cache) per process, mirroring app_config's config
# cache and app_config.vault's client cache.
_client_cache: AzureAuthClient | None = None


def get_azure_client() -> AzureAuthClient:
    """The process-wide :class:`AzureAuthClient` (built from the environment)."""
    global _client_cache
    if _client_cache is None:
        _client_cache = AzureAuthClient()
    return _client_cache


def get_token(scopes: tuple[str, ...] | list[str] | None = None) -> str:
    """Convenience token acquisition via the shared :func:`get_azure_client`."""
    return get_azure_client().get_token(scopes)


def clear_cache() -> None:
    """Drop the cached client and its tokens so the next access re-logs in."""
    global _client_cache
    _client_cache = None
