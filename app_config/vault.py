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
    ``VAULT_APP_NAME`` (or ``vault.app_name``) is set — OIDC login backed
    by Microsoft Entra ID (Azure AD). An Entra access token is acquired
    through the sibling :mod:`app_config.azure` module (service principal or
    device-code, per its own configuration) and presented as the JWT to
    Vault's jwt/oidc auth method at ``auth/<vault.auth_path>/login``, per
    https://developer.hashicorp.com/vault/docs/auth/jwt/oidc-providers/azuread.
    The Vault role's ``bound_audiences`` must match the token's ``aud``. The
    default audience is Azure Resource Manager
    (:data:`_ARM_DEFAULT_SCOPE`) — the reference deployment binds its role to
    ARM — and ``vault.azure_scopes`` overrides it for a role bound to
    something else.

``app_name`` is one value playing three parts, exactly as the reference
deployment uses it: the Vault *role* name for that login, the KV secret-path
*prefix* (so the AI Gateway credential lives at ``<app_name>/ai_gateway``), and
the application's own name.

Token lifetime
--------------
The tokens Vault mints for ``approle`` and ``azuread`` logins are short-lived,
so :class:`VaultClient` records the lease it was given and re-authenticates
before a read once it has lapsed (:data:`_TOKEN_SKEW` seconds early). A read
rejected with 403 is retried once behind a fresh login as well, since a
revoked-early token is indistinguishable from an expired one. A ``VAULT_TOKEN``
supplied by the operator is exempt from both — its lifetime is their business.

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
import time
from dataclasses import dataclass, field
from typing import Any

from . import _TRUTHY, _verify_setting, get_typed_value, get_value

logger = logging.getLogger(__name__)

DEFAULT_MOUNT_POINT = "secret"
DEFAULT_KV_VERSION = 2
DEFAULT_TIMEOUT = 30.0
DEFAULT_AUTH_PATH = "jwt/azuread/inspirewellness"
# Default audience for the Entra ID token presented to Vault, used whenever
# neither VaultConfig.azure_scopes nor the azure module's own scopes are
# configured. The deployment's Vault role is bound to the Azure Resource
# Manager audience; the doubled slash is ARM's documented `.default` form, not
# a typo.
_ARM_DEFAULT_SCOPE = "https://management.azure.com//.default"

# Re-authenticate this many seconds before the Vault token's lease actually
# runs out, so a token used for a read stays valid for the length of it.
# Mirrors app_config.azure._EXPIRY_SKEW.
_TOKEN_SKEW = 60.0

# Leaf name of the AI Gateway credential under the application's own prefix:
# the reference reads /v1/secret/data/<app_name>/ai_gateway.
AI_GATEWAY_LEAF = "ai_gateway"

# Field names the gateway token is commonly filed under, tried in order when
# no explicit key is given. Override with vault.ai_gateway_key when the secret
# uses something else.
_AI_GATEWAY_TOKEN_KEYS = ("token", "api_key", "apikey", "ai_gateway_token", "value")

# Field names carrying the gateway's own endpoint, if the secret ships one.
_AI_GATEWAY_URL_KEYS = ("base_url", "endpoint", "url")

# Field names carrying the gateway's protocol version ("v2"), if it ships one.
_AI_GATEWAY_VERSION_KEYS = ("gateway_version", "ai_gateway_version", "version")


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
    configured = _verify_setting(get_value("vault", "verify"))
    if configured is not None:
        return configured
    return True


def _resolve_azure_scopes() -> tuple[str, ...]:
    """
    Entra ID scopes to request for the ``azuread`` login JWT, from
    ``VAULT_AZURE_SCOPES`` (space- or comma-separated) or the
    ``vault.azure_scopes`` config list. Empty when unset — :func:`_azure_jwt`
    then falls back to the azure module's own scopes, and finally to
    :data:`_ARM_DEFAULT_SCOPE`.
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
        ``vault.auth_path``, default
        :data:`DEFAULT_AUTH_PATH`; set it to wherever the method is mounted
        on your server.
    app_name : str | None
        The application's name in Vault: the *role* for ``azuread`` login,
        and the KV secret-path *prefix* the application's own secrets live
        under. ``VAULT_APP_NAME`` / ``vault.app_name``. Setting it is what
        enables the ``azuread`` method — it is a name, not a credential.
    azure_scopes : tuple[str, ...]
        Entra ID scopes requested for the login JWT. ``VAULT_AZURE_SCOPES``
        / ``vault.azure_scopes``. Empty (default) falls back to the azure
        module's configured scopes, then to :data:`_ARM_DEFAULT_SCOPE`, whose
        audience is what the deployment's Vault role is bound to.
    ai_gateway_path : str | None
        Where the AI Gateway credential is filed, *relative to the mount*.
        ``vault.ai_gateway_path``; ``None`` (default) derives
        ``<app_name>/ai_gateway`` — see :attr:`resolved_ai_gateway_path`.
    ai_gateway_key : str | None
        The field inside that secret holding the token.
        ``vault.ai_gateway_key``; ``None`` tries
        :data:`_AI_GATEWAY_TOKEN_KEYS` in order.
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
    app_name: str | None = None
    azure_scopes: tuple[str, ...] = ()
    ai_gateway_path: str | None = None
    ai_gateway_key: str | None = None
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
            app_name=(
                os.environ.get("VAULT_APP_NAME") or get_value("vault", "app_name")
            ),
            azure_scopes=_resolve_azure_scopes(),
            ai_gateway_path=get_value("vault", "ai_gateway_path"),
            ai_gateway_key=get_value("vault", "ai_gateway_key"),
            token=os.environ.get("VAULT_TOKEN"),
            role_id=os.environ.get("VAULT_ROLE_ID"),
            secret_id=os.environ.get("VAULT_SECRET_ID"),
        )

    @property
    def auth_method(self) -> str | None:
        """
        ``"token"`` when a token is set, else ``"approle"`` when both AppRole
        credentials are set, else ``"azuread"`` (Entra ID OIDC) when an
        :attr:`app_name` is set, else ``None`` (no usable credentials).
        """
        if self.token:
            return "token"
        if self.role_id and self.secret_id:
            return "approle"
        if self.app_name:
            return "azuread"
        return None

    @property
    def resolved_ai_gateway_path(self) -> str | None:
        """
        Where the AI Gateway credential lives, relative to the mount: an
        explicit :attr:`ai_gateway_path`, else ``<app_name>/ai_gateway``, else
        ``None`` — there is no universal default, so a deployment that
        configures neither is asked which it means rather than guessing.
        """
        if self.ai_gateway_path:
            return self.ai_gateway_path
        if self.app_name:
            return f"{self.app_name}/{AI_GATEWAY_LEAF}"
        return None


_NO_CREDENTIALS = (
    "no Vault credentials: set VAULT_TOKEN; VAULT_ROLE_ID and "
    "VAULT_SECRET_ID; or VAULT_APP_NAME for Entra ID OIDC login"
)


def _azure_jwt(config: VaultConfig) -> str:
    """
    The Entra ID access token presented as the login JWT for the ``azuread``
    auth method, acquired through the shared :mod:`app_config.azure` client.
    Scopes resolve as :attr:`VaultConfig.azure_scopes` > the azure module's
    configured scopes > :data:`_ARM_DEFAULT_SCOPE`.

    The app registration's own audience (``<client_id>/.default``) is
    deliberately *not* in that chain: a Vault role bound to it is the unusual
    case, and an operator who wants it sets ``vault.azure_scopes`` explicitly.
    """
    from . import azure  # sibling module; msal stays a lazy import inside it

    # Client construction is inside the try too: resolving the identity can
    # itself fail (e.g. reading the service principal out of a Databricks
    # secret scope), and a caller should still only have to except VaultError.
    try:
        azure_client = azure.get_azure_client()
        scopes = config.azure_scopes or azure_client.config.scopes or (
            _ARM_DEFAULT_SCOPE,
        )
        return azure_client.get_token(scopes)
    except azure.AzureAuthError as exc:
        raise VaultError(
            f"could not acquire an Entra ID token for Vault azuread login: {exc}"
        ) from exc


def _lease_duration(response: Any) -> float | None:
    """
    Seconds the Vault token in a login *response* is good for, or ``None`` when
    the response carries no (or an unusable) ``auth.lease_duration``. A
    root/periodic token reports ``0``, which also reads as "no expiry to track".
    """
    if not isinstance(response, dict):
        return None
    auth = response.get("auth")
    if not isinstance(auth, dict):
        return None
    lease = auth.get("lease_duration")
    if isinstance(lease, bool) or not isinstance(lease, (int, float)):
        return None
    return float(lease) if lease > 0 else None


def _authenticate(client: Any, config: VaultConfig) -> float | None:
    """
    Log *client* in per :attr:`VaultConfig.auth_method`, then confirm the
    session is live. Raises :class:`VaultError` on unreachable server or a
    rejected credential.

    Returns the lease duration of the token Vault issued, in seconds, or
    ``None`` when there is none to track — an operator-supplied
    ``VAULT_TOKEN``, or a login response without a usable
    ``auth.lease_duration``.
    """
    method = config.auth_method
    lease: float | None = None
    if method == "token":
        client.token = config.token
    elif method == "approle":
        try:
            # hvac stores the returned token on the client.
            lease = _lease_duration(
                client.auth.approle.login(
                    role_id=config.role_id, secret_id=config.secret_id
                )
            )
        except VaultError:
            raise
        except Exception as exc:  # rejected credentials / bad mount
            raise VaultError(f"Vault approle login failed: {exc}") from exc
    elif method == "azuread":
        jwt = _azure_jwt(config)
        try:
            # hvac stores the returned Vault token on the client.
            lease = _lease_duration(
                client.auth.jwt.jwt_login(
                    role=config.app_name, jwt=jwt, path=config.auth_path
                )
            )
        except VaultError:
            raise
        except Exception as exc:  # rejected JWT / unknown role / bad mount
            raise VaultError(
                f"Vault azuread login failed for role '{config.app_name}' "
                f"at auth path '{config.auth_path}': {exc}"
            ) from exc
    else:  # unreachable via _build_client, which checks first — defensive
        raise VaultError(_NO_CREDENTIALS)
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
        f"kv_version={config.kv_version}, "
        f"lease={f'{lease:.0f}s' if lease else 'untracked'})"
    )
    return lease


def _is_forbidden(exc: BaseException) -> bool:
    """
    True when *exc* is Vault's 403 — an expired, revoked, or under-privileged
    token. ``hvac`` raises ``hvac.exceptions.Forbidden``, which carries the
    status code; the class-name check covers a stand-in that does not.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 403:
        return True
    return type(exc).__name__ == "Forbidden"


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

    Once built, the session is kept alive: the lease Vault reported at login is
    recorded and a read that arrives after it has lapsed re-authenticates
    first, and a read rejected with 403 is retried once behind a fresh login.
    Token auth is exempt (there is no lease to renew against). An injected
    *client* starts with no known lease, so only the 403 rule applies to it.
    """

    def __init__(
        self, config: VaultConfig | None = None, *, client: Any | None = None
    ) -> None:
        self.config = config if config is not None else VaultConfig.from_env()
        self._client = client
        # Epoch seconds after which the Vault token is assumed dead; None means
        # "no lease to track" (token auth, or a login that reported none).
        self._expires_at: float | None = None

    @property
    def client(self) -> Any:
        """The underlying ``hvac.Client``, built and authenticated on demand."""
        if self._client is None:
            self._client, lease = self._build_client(self.config)
            self._set_expiry(lease)
        return self._client

    def _set_expiry(self, lease: float | None) -> None:
        self._expires_at = (
            time.time() + max(lease - _TOKEN_SKEW, 0.0) if lease else None
        )

    @staticmethod
    def _build_client(config: VaultConfig) -> tuple[Any, float | None]:
        """The authenticated ``hvac.Client`` and its token's lease duration."""
        # Validate config before importing hvac so a misconfiguration reports
        # the real problem instead of a missing-dependency error.
        if not config.address:
            raise VaultError(
                "no Vault address configured: set VAULT_ADDR or "
                "vault.address in config.json"
            )
        if config.auth_method is None:
            raise VaultError(_NO_CREDENTIALS)
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
        return client, _authenticate(client, config)

    def _reauthenticate(self, why: str) -> None:
        """Log the existing client in again, replacing its Vault token."""
        logger.info(f"VaultClient: re-authenticating ({why})")
        self._set_expiry(_authenticate(self.client, self.config))

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
        client = self.client  # builds and authenticates on first use
        if self._expires_at is not None and time.time() >= self._expires_at:
            self._reauthenticate("the Vault token's lease has run out")
        try:
            return self._read_once(client, path, mount)
        except Exception as exc:
            # A 403 is the one failure worth a second attempt: a token revoked
            # early is indistinguishable from one whose lease we mis-tracked,
            # and re-authenticating is exactly the fix for both. Anything else
            # (absent secret, network, bad mount) would fail identically twice.
            if not (_is_forbidden(exc) and self.config.auth_method != "token"):
                raise VaultError(
                    f"could not read Vault secret '{path}' (mount '{mount}'): "
                    f"{exc}"
                ) from exc
            self._reauthenticate(f"Vault refused the read of '{path}' with 403")
            try:
                return self._read_once(self.client, path, mount)
            except Exception as retry_exc:
                raise VaultError(
                    f"could not read Vault secret '{path}' (mount '{mount}') "
                    f"even after re-authenticating: {retry_exc}"
                ) from retry_exc

    def _read_once(self, client: Any, path: str, mount: str) -> dict[str, Any]:
        if self.config.kv_version == 2:
            resp = client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=mount,
                raise_on_deleted_version=True,
            )
            return resp["data"]["data"]
        resp = client.secrets.kv.v1.read_secret(path=path, mount_point=mount)
        return resp["data"]


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


def is_configured() -> bool:
    """
    True when Vault is set up well enough to *try* the AI Gateway chain — an
    address plus some way to log in.

    Deliberately about **configuration, not reachability**. A workstation with
    no Vault settings at all is a normal local-development case and should fall
    back to the provider environment variable without ceremony; a
    half-configured or broken Vault is not, and must fail loudly rather than
    quietly use a different credential than the operator intended. So this
    answers only "was Vault asked for?", and any failure after it says yes is
    raised, never swallowed.

    Performs no I/O — it reads :meth:`VaultConfig.from_env` and nothing else,
    so a caller can branch on it before paying for a login.
    """
    config = VaultConfig.from_env()
    return bool(config.address and config.auth_method)


def get_ai_gateway_secret(path: str | None = None) -> dict[str, Any]:
    """
    The whole AI Gateway secret — by default the one under the application's
    own prefix, ``<app_name>/ai_gateway`` (the API's
    ``<vault_addr>/v1/secret/data/<app_name>/ai_gateway`` with the default
    mount and KV v2). ``vault.ai_gateway_path`` pins a different location.

    Read through the shared :func:`get_vault_client`, so the Vault login
    happens once per process. With ``azuread`` auth that login presents an
    Entra ID JWT — which, when no ``AZURE_*`` identity is configured, comes
    from the service principal in the Databricks secret scope. See
    :func:`app_config.azure.get_azure_client`.

    Raises
    ------
    VaultError
        No path can be resolved, or Vault is unreachable or unauthenticated,
        or the secret is absent.
    """
    client = get_vault_client()
    if path is None:
        path = client.config.resolved_ai_gateway_path
    if not path:
        raise VaultError(
            "no AI Gateway secret path: set VAULT_APP_NAME (the path is then "
            "'<app_name>/ai_gateway'), or vault.ai_gateway_path in "
            "config.json to name it outright"
        )
    return client.get_secret(path)


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
    wanted = key or get_vault_client().config.ai_gateway_key
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


def ai_gateway_version(secret: dict[str, Any] | None = None) -> str | None:
    """
    The gateway's protocol version carried alongside the token
    (``gateway_version`` / ``ai_gateway_version`` / ``version``), else ``None``
    so the configured ``llm_client.gateway_version`` stands. It is sent as the
    ``ai-gateway-version`` request header — see
    :meth:`llm_client.LLMClientConfig.from_ai_gateway`.
    """
    data = get_ai_gateway_secret() if secret is None else secret
    for candidate in _AI_GATEWAY_VERSION_KEYS:
        value = data.get(candidate)
        if value:
            return str(value)
    return None


def clear_cache() -> None:
    """Drop the cached client so the next access re-authenticates (for tests)."""
    global _client_cache
    _client_cache = None
