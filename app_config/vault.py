"""HashiCorp Vault client for retrieving credentials at runtime.

Submodule of :mod:`app_config`. Where the rest of the package answers "what
are the tunable limits?" from a committed ``config.json``, this module answers
"what are the secrets?" from a running Vault server — the two never mix, so no
credential is ever read from (or written to) the committed file.

Split of concerns
-----------------
* **Non-secret connection settings** — the Vault address, namespace, KV mount
  point, KV engine version, request timeout, and TLS verification — resolve
  through :meth:`VaultConfig.from_env`, which reads the standard Vault
  environment variables first (``VAULT_ADDR``, ``VAULT_NAMESPACE``,
  ``VAULT_CACERT``, ``VAULT_SKIP_VERIFY``) and falls back to the optional
  ``vault`` section of ``config.json`` (via :func:`app_config.get_value` /
  :func:`app_config.get_typed_value`, so a wrong-typed entry degrades to the
  hard default with a WARNING rather than crashing).
* **Secrets** — the auth token, or the AppRole ``role_id`` / ``secret_id`` —
  come *only* from environment variables (``VAULT_TOKEN``, ``VAULT_ROLE_ID``,
  ``VAULT_SECRET_ID``). They are held in fields marked ``repr=False`` so they
  never appear in a ``repr`` or a log line.

Auth methods
------------
Three ways in, tried in this order (:attr:`VaultConfig.auth_method`):

``token``
    ``VAULT_TOKEN`` is set — use it as-is.
``approle``
    ``VAULT_ROLE_ID`` + ``VAULT_SECRET_ID`` are set — AppRole login.
``azuread``
    ``VAULT_OIDC_ROLE`` (or ``vault.oidc_role``) is set — OIDC login backed
    by Microsoft Entra ID (Azure AD). An Entra access token is acquired
    through the sibling :mod:`app_config.azure` module (service principal or
    device-code, per its own configuration) and presented as the JWT to
    Vault's jwt/oidc auth method at ``auth/<vault.auth_path>/login`` (default
    ``jwt``), per
    https://developer.hashicorp.com/vault/docs/auth/jwt/oidc-providers/azuread.
    The Vault role's ``bound_audiences`` must match the token's ``aud`` —
    normally the app registration's client id, which is what the default
    scope ``<client_id>/.default`` requests when neither
    ``vault.azure_scopes`` nor the azure module's scopes are configured.

Callers that want to bypass the environment entirely can construct
:class:`VaultConfig` directly (an explicit argument always wins) or inject a
pre-built ``hvac.Client`` into :class:`VaultClient` (custom auth backends,
tests).

Dependency
----------
The ``hvac`` client library is an *optional* dependency (extra ``vault``):
``pip install "sas-parser[vault]"``. It is imported lazily inside
:meth:`VaultClient._build_client`, so ``import app_config.vault`` costs nothing
and keeps ``app_config`` the dependency-free leaf the rest of the package
relies on. Only actually talking to Vault requires ``hvac`` to be installed.
``azuread`` login additionally needs ``msal`` (extra ``azure``), imported just
as lazily by :mod:`app_config.azure` when the JWT is acquired.

Typical use
-----------
    from app_config.vault import get_secret

    creds = get_secret("llm/anthropic")       # -> {"api_key": "sk-...", ...}
    key = get_secret("llm/anthropic", "api_key")

The module-level helpers reuse one authenticated :class:`VaultClient` per
process (:func:`get_vault_client`); call :func:`clear_cache` to force
re-authentication after the environment changes (tests do). Individual secret
*reads* are never cached — every :meth:`~VaultClient.get_secret` hits Vault, so
rotated secrets are picked up without a restart.

Logger name: ``app_config.vault``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from . import get_typed_value, get_value

logger = logging.getLogger(__name__)

DEFAULT_MOUNT_POINT = "secret"
DEFAULT_KV_VERSION = 2
DEFAULT_TIMEOUT = 30.0
DEFAULT_AUTH_PATH = "jwt"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Path of the AI Gateway credential *relative to the KV mount*: with the
# default mount ("secret") and KV v2, this is the API's
# <vault_addr>/v1/secret/data/appsvc/ai_gateway.
AI_GATEWAY_PATH = "appsvc/ai_gateway"

# Field names the gateway token is commonly filed under, tried in order when
# no explicit key is given. Override with vault.ai_gateway_key when the secret
# uses something else.
_AI_GATEWAY_TOKEN_KEYS = ("token", "api_key", "apikey", "ai_gateway_token", "value")

# Field names carrying the gateway's own endpoint, if the secret ships one.
_AI_GATEWAY_URL_KEYS = ("base_url", "endpoint", "url")


class VaultError(RuntimeError):
    """Vault is misconfigured, unreachable, unauthenticated, or the secret is absent.

    A single error type so callers can ``except VaultError`` around a lookup
    regardless of which stage failed; the message says which.
    """


def _resolve_verify() -> bool | str:
    """
    TLS verification for the Vault connection, resolved as
    ``VAULT_SKIP_VERIFY`` (disable) > ``VAULT_CACERT`` (path to a CA bundle) >
    ``config.json`` ``vault.verify`` > ``True`` (verify against system CAs).
    """
    if os.environ.get("VAULT_SKIP_VERIFY", "").strip().lower() in _TRUTHY:
        return False
    cacert = os.environ.get("VAULT_CACERT")
    if cacert:
        return cacert
    configured = get_value("vault", "verify")
    if isinstance(configured, (bool, str)):
        return configured
    return True


def _resolve_azure_scopes() -> tuple[str, ...]:
    """
    Entra ID scopes to request for the ``azuread`` login JWT, from
    ``VAULT_AZURE_SCOPES`` (space- or comma-separated) or the
    ``vault.azure_scopes`` config list. Empty when unset — :func:`_azure_jwt`
    then falls back to the azure module's own scopes, and finally to
    ``<client_id>/.default``.
    """
    env = os.environ.get("VAULT_AZURE_SCOPES")
    if env:
        return tuple(env.replace(",", " ").split())
    configured = get_typed_value("vault", "azure_scopes", list)
    if configured is None:
        return ()
    if not all(isinstance(s, str) for s in configured):
        logger.warning(
            "vault: config.json vault.azure_scopes must be a list of strings; "
            "ignoring it (scopes fall back to the azure section's own)"
        )
        return ()
    return tuple(configured)


@dataclass
class VaultConfig:
    """
    Everything :class:`VaultClient` needs to connect and authenticate.

    Construct it directly to pin values explicitly, or call
    :meth:`from_env` for the standard environment-then-``config.json``
    resolution. Secret fields (:attr:`token`, :attr:`role_id`,
    :attr:`secret_id`) are ``repr=False`` and are never logged.

    Attributes
    ----------
    address : str | None
        Vault server URL (``https://vault.example:8200``).
        ``VAULT_ADDR`` / ``config.json`` ``vault.address``. Required to
        connect; a missing address raises :class:`VaultError`.
    namespace : str | None
        Vault Enterprise namespace. ``VAULT_NAMESPACE`` /
        ``vault.namespace``. ``None`` for open-source Vault / the root
        namespace.
    mount_point : str
        Mount path of the KV secrets engine. ``vault.mount_point``,
        default ``"secret"``.
    kv_version : int
        KV engine version, ``2`` (versioned) or ``1``. ``vault.kv_version``,
        default ``2``. Selects the read API used by
        :meth:`VaultClient.get_secret`.
    timeout : float
        Per-request timeout in seconds. ``vault.timeout``, default ``30``.
    verify : bool | str
        TLS verification: ``True`` (system CAs), ``False`` (disable — dev
        only), or a path to a CA bundle. See :func:`_resolve_verify`.
    auth_path : str
        Mount path of the jwt/oidc auth method used by ``azuread`` login
        (``auth/<auth_path>/login``). ``VAULT_AUTH_PATH`` /
        ``vault.auth_path``, default ``"jwt"``; set to ``"oidc"`` (or
        wherever the method is mounted) to match the server.
    oidc_role : str | None
        Vault role name for ``azuread`` login. ``VAULT_OIDC_ROLE`` /
        ``vault.oidc_role``. Setting it is what enables the method — it is
        the role's name in Vault, not a credential.
    azure_scopes : tuple[str, ...]
        Entra ID scopes requested for the login JWT. ``VAULT_AZURE_SCOPES``
        / ``vault.azure_scopes``. Empty (default) falls back to the azure
        module's configured scopes, then to ``<client_id>/.default`` so the
        token's audience matches a Vault role bound to the app registration.
    token : str | None
        Vault token for token auth. ``VAULT_TOKEN`` only — never from
        ``config.json``.
    role_id, secret_id : str | None
        AppRole credentials, used when no :attr:`token` is set.
        ``VAULT_ROLE_ID`` / ``VAULT_SECRET_ID`` only.
    """

    address: str | None = None
    namespace: str | None = None
    mount_point: str = DEFAULT_MOUNT_POINT
    kv_version: int = DEFAULT_KV_VERSION
    timeout: float = DEFAULT_TIMEOUT
    verify: bool | str = True
    auth_path: str = DEFAULT_AUTH_PATH
    oidc_role: str | None = None
    azure_scopes: tuple[str, ...] = ()
    token: str | None = field(default=None, repr=False)
    role_id: str | None = field(default=None, repr=False)
    secret_id: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> "VaultConfig":
        """
        Resolve connection settings from the standard Vault environment
        variables, falling back to the ``vault`` section of ``config.json``
        for the non-secret knobs. Secrets are read from the environment only.
        """
        return cls(
            address=os.environ.get("VAULT_ADDR") or get_value("vault", "address"),
            namespace=(
                os.environ.get("VAULT_NAMESPACE") or get_value("vault", "namespace")
            ),
            mount_point=get_value("vault", "mount_point", DEFAULT_MOUNT_POINT),
            kv_version=get_typed_value(
                "vault", "kv_version", int, DEFAULT_KV_VERSION
            ),
            timeout=get_typed_value(
                "vault", "timeout", (int, float), DEFAULT_TIMEOUT
            ),
            verify=_resolve_verify(),
            auth_path=(
                os.environ.get("VAULT_AUTH_PATH")
                or get_value("vault", "auth_path", DEFAULT_AUTH_PATH)
            ),
            oidc_role=(
                os.environ.get("VAULT_OIDC_ROLE") or get_value("vault", "oidc_role")
            ),
            azure_scopes=_resolve_azure_scopes(),
            token=os.environ.get("VAULT_TOKEN"),
            role_id=os.environ.get("VAULT_ROLE_ID"),
            secret_id=os.environ.get("VAULT_SECRET_ID"),
        )

    @property
    def auth_method(self) -> str | None:
        """
        ``"token"`` when a token is set, else ``"approle"`` when both AppRole
        credentials are set, else ``"azuread"`` (Entra ID OIDC) when an
        :attr:`oidc_role` is set, else ``None`` (no usable credentials).
        """
        if self.token:
            return "token"
        if self.role_id and self.secret_id:
            return "approle"
        if self.oidc_role:
            return "azuread"
        return None


def _azure_jwt(config: VaultConfig) -> str:
    """
    The Entra ID access token presented as the login JWT for the ``azuread``
    auth method, acquired through the shared :mod:`app_config.azure` client.
    Scopes resolve as :attr:`VaultConfig.azure_scopes` > the azure module's
    configured scopes > ``<client_id>/.default`` (the app registration's own
    audience — what a Vault role with ``bound_audiences=<client_id>`` expects).
    """
    from . import azure  # sibling module; msal stays a lazy import inside it

    # Client construction is inside the try too: resolving the identity can
    # itself fail (e.g. reading the service principal out of a Databricks
    # secret scope), and a caller should still only have to except VaultError.
    try:
        azure_client = azure.get_azure_client()
        scopes = config.azure_scopes or azure_client.config.scopes
        if not scopes and azure_client.config.client_id:
            scopes = (f"{azure_client.config.client_id}/.default",)
        return azure_client.get_token(scopes)
    except azure.AzureAuthError as exc:
        raise VaultError(
            f"could not acquire an Entra ID token for Vault azuread login: {exc}"
        ) from exc


def _authenticate(client: Any, config: VaultConfig) -> None:
    """
    Log *client* in per :attr:`VaultConfig.auth_method`, then confirm the
    session is live. Raises :class:`VaultError` on unreachable server or a
    rejected credential.
    """
    method = config.auth_method
    if method == "token":
        client.token = config.token
    elif method == "approle":
        client.auth.approle.login(
            role_id=config.role_id, secret_id=config.secret_id
        )  # hvac stores the returned token on the client
    elif method == "azuread":
        jwt = _azure_jwt(config)
        try:
            client.auth.jwt.jwt_login(
                role=config.oidc_role, jwt=jwt, path=config.auth_path
            )  # hvac stores the returned Vault token on the client
        except VaultError:
            raise
        except Exception as exc:  # rejected JWT / unknown role / bad mount
            raise VaultError(
                f"Vault azuread login failed for role '{config.oidc_role}' "
                f"at auth path '{config.auth_path}': {exc}"
            ) from exc
    else:  # unreachable via _build_client, which checks first — defensive
        raise VaultError(
            "no Vault credentials: set VAULT_TOKEN; VAULT_ROLE_ID and "
            "VAULT_SECRET_ID; or VAULT_OIDC_ROLE for Entra ID OIDC login"
        )
    try:
        authenticated = client.is_authenticated()
    except Exception as exc:  # network / TLS / bad URL surface here
        raise VaultError(
            f"could not reach Vault at {config.address}: {exc}"
        ) from exc
    if not authenticated:
        raise VaultError(
            f"Vault authentication failed for auth method '{method}'"
        )
    logger.info(
        f"VaultClient: authenticated to {config.address} via {method} "
        f"(namespace={config.namespace}, mount={config.mount_point}, "
        f"kv_version={config.kv_version})"
    )


class VaultClient:
    """
    Thin wrapper over ``hvac.Client`` for reading KV secrets.

    Parameters
    ----------
    config : VaultConfig | None
        Connection/auth settings. ``None`` (default) uses
        :meth:`VaultConfig.from_env`.
    client : Any | None
        A pre-built, already-authenticated ``hvac.Client`` (or a duck-typed
        stand-in) to use as-is. When given, :attr:`config` is used only for
        its :attr:`~VaultConfig.mount_point` / :attr:`~VaultConfig.kv_version`
        read defaults and no connection or auth is performed — the escape
        hatch for custom auth backends and tests.

    The underlying client is built lazily on first :attr:`client` access, so
    constructing a :class:`VaultClient` never touches the network or requires
    ``hvac`` to be importable.
    """

    def __init__(
        self, config: VaultConfig | None = None, *, client: Any | None = None
    ) -> None:
        self.config = config if config is not None else VaultConfig.from_env()
        self._client = client

    @property
    def client(self) -> Any:
        """The underlying ``hvac.Client``, built and authenticated on demand."""
        if self._client is None:
            self._client = self._build_client(self.config)
        return self._client

    @staticmethod
    def _build_client(config: VaultConfig) -> Any:
        # Validate config before importing hvac so a misconfiguration reports
        # the real problem instead of a missing-dependency error.
        if not config.address:
            raise VaultError(
                "no Vault address configured: set VAULT_ADDR or "
                "vault.address in config.json"
            )
        if config.auth_method is None:
            raise VaultError(
                "no Vault credentials: set VAULT_TOKEN; VAULT_ROLE_ID and "
                "VAULT_SECRET_ID; or VAULT_OIDC_ROLE for Entra ID OIDC login"
            )
        try:
            import hvac
        except ImportError as exc:
            raise VaultError(
                "hvac is required for Vault access; install it with "
                "'pip install \"sas-parser[vault]\"'"
            ) from exc
        client = hvac.Client(
            url=config.address,
            namespace=config.namespace,
            verify=config.verify,
            # hvac's stub types timeout as int, but it reaches requests, which
            # takes float seconds; casting would floor sub-second timeouts to 0.
            timeout=config.timeout,  # pyright: ignore[reportArgumentType]
        )
        _authenticate(client, config)
        return client

    def get_secret(
        self, path: str, key: str | None = None, *, mount_point: str | None = None
    ) -> Any:
        """
        Read the secret at *path* from the KV engine.

        Parameters
        ----------
        path : str
            Secret path *relative to the mount* (e.g. ``"llm/anthropic"``,
            not ``"secret/data/llm/anthropic"`` — the mount and the KV v2
            ``data/`` infix are added for you).
        key : str | None
            When given, return just that field's value; a missing field
            raises :class:`VaultError`. ``None`` (default) returns the whole
            secret as a ``dict``.
        mount_point : str | None
            Override the configured :attr:`~VaultConfig.mount_point` for this
            read.

        Raises
        ------
        VaultError
            The secret or field is absent, or the read otherwise fails.
        """
        mount = mount_point or self.config.mount_point
        data = self._read(path, mount)
        if key is None:
            return data
        try:
            return data[key]
        except KeyError:
            raise VaultError(
                f"key '{key}' not found in Vault secret '{path}' "
                f"(mount '{mount}')"
            ) from None

    def _read(self, path: str, mount: str) -> dict[str, Any]:
        client = self.client
        try:
            if self.config.kv_version == 2:
                resp = client.secrets.kv.v2.read_secret_version(
                    path=path,
                    mount_point=mount,
                    raise_on_deleted_version=True,
                )
                return resp["data"]["data"]
            resp = client.secrets.kv.v1.read_secret(path=path, mount_point=mount)
            return resp["data"]
        except Exception as exc:
            raise VaultError(
                f"could not read Vault secret '{path}' (mount '{mount}'): {exc}"
            ) from exc


# One authenticated client per process, mirroring app_config's config cache.
_client_cache: VaultClient | None = None


def get_vault_client() -> VaultClient:
    """The process-wide :class:`VaultClient` (built from the environment)."""
    global _client_cache
    if _client_cache is None:
        _client_cache = VaultClient()
    return _client_cache


def get_secret(
    path: str, key: str | None = None, *, mount_point: str | None = None
) -> Any:
    """Convenience read via the shared :func:`get_vault_client`."""
    return get_vault_client().get_secret(path, key, mount_point=mount_point)


def get_ai_gateway_secret(path: str | None = None) -> dict[str, Any]:
    """
    The whole AI Gateway secret — by default the one at
    :data:`AI_GATEWAY_PATH` (``<vault_addr>/v1/secret/data/appsvc/ai_gateway``
    with the default mount and KV v2).

    Read through the shared :func:`get_vault_client`, so the Vault login
    happens once per process. With ``azuread`` auth that login presents an
    Entra ID JWT — which, when no ``AZURE_*`` identity is configured, comes
    from the service principal in the Databricks secret scope. See
    :func:`app_config.azure.get_azure_client`.

    Raises
    ------
    VaultError
        Vault is unreachable or unauthenticated, or the secret is absent.
    """
    return get_secret(path or AI_GATEWAY_PATH)


def ai_gateway_token(
    secret: dict[str, Any] | None = None, *, key: str | None = None
) -> str:
    """
    The bearer token out of the AI Gateway secret, for
    :class:`llm_client.LLMClientConfig`.

    Parameters
    ----------
    secret : dict[str, Any] | None
        An already-read secret. ``None`` (default) reads it via
        :func:`get_ai_gateway_secret`.
    key : str | None
        The field holding the token. ``None`` (default) uses
        ``vault.ai_gateway_key`` from ``config.json``, else the first of
        :data:`_AI_GATEWAY_TOKEN_KEYS` that is present — so a secret filed
        under ``token`` or ``api_key`` needs no configuration at all.

    Raises
    ------
    VaultError
        The secret has no recognisable token field, or an explicitly named
        *key* is absent. The message lists the field names that *are* there,
        which is what you need to pick the right one.
    """
    data = get_ai_gateway_secret() if secret is None else secret
    wanted = key or get_value("vault", "ai_gateway_key")
    if wanted:
        try:
            return data[wanted]
        except KeyError:
            raise VaultError(
                f"key '{wanted}' not found in the Vault AI Gateway secret; "
                f"it has {sorted(data)}"
            ) from None
    for candidate in _AI_GATEWAY_TOKEN_KEYS:
        if data.get(candidate):
            return data[candidate]
    raise VaultError(
        f"no AI Gateway token found in the Vault secret: none of "
        f"{list(_AI_GATEWAY_TOKEN_KEYS)} is set, and it has {sorted(data)}. "
        f"Set vault.ai_gateway_key in config.json to name the right field"
    )


def ai_gateway_base_url(secret: dict[str, Any] | None = None) -> str | None:
    """
    The gateway endpoint carried alongside the token, if the secret ships one
    (``base_url`` / ``endpoint`` / ``url``), else ``None`` so the configured
    ``llm_client.base_url`` stands.
    """
    data = get_ai_gateway_secret() if secret is None else secret
    for candidate in _AI_GATEWAY_URL_KEYS:
        if data.get(candidate):
            return data[candidate]
    return None


def clear_cache() -> None:
    """Drop the cached client so the next access re-authenticates (for tests)."""
    global _client_cache
    _client_cache = None
