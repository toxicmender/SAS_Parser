"""Where a hydration credential comes from — one chain, four layers.

A migration run reaches Oracle, sFTP and two Azure storage services, and the
deployments this runs in keep those credentials in different places. Rather than
each source growing its own lookup, every credential resolves here, in one order:

1. **Databricks secret scope** — the scope this deployment already files its
   service principals in (:mod:`app_config.databricks`).
2. **HashiCorp Vault** — under the configured ``app_name`` KV prefix
   (:mod:`app_config.vault`).
3. **Entra ID** — for Azure storage there *is* no password; the credential is a
   short-lived token minted by :mod:`app_config.azure`. :func:`entra_credential`
   is the adapter that hands one to an Azure SDK client.
4. **The environment** — ``DATA_HYDRATION_<NAME>``.

Never ``config.json``. It is checked into deployments; the repo's rule is that
non-secret settings live there and secrets do not, and
:class:`~data_hydration.config.HydrationConfig` has no secret field for exactly
this reason.

Each layer is skipped when it is not configured and its failure is logged at
DEBUG before falling through — an unreachable Vault must not stop a credential
that was in the environment all along. Only exhausting all four raises.

This is invariant 12's "one credential chain" applied to hydration: assembling a
lookup by hand out of ``vault.read`` looks equivalent and silently loses the
fallbacks below it.

Logger name: ``data_hydration.secrets``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class HydrationCredentialError(RuntimeError):
    """No layer of the chain could supply a credential that was needed.

    The message names what was looked for and which layers were tried, because
    the fix differs per layer: a missing scope key is filed differently from a
    missing environment variable.
    """


def _env_name(name: str) -> str:
    """The environment variable a credential named *name* is read from."""
    return f"DATA_HYDRATION_{name.upper().replace('-', '_').replace('.', '_')}"


def _from_databricks(name: str, scope: str | None) -> str | None:
    """The secret from the Databricks scope, or ``None`` if unavailable."""
    if not scope:
        return None
    try:
        from app_config.databricks import read_workspace_secret

        return read_workspace_secret(scope, name)
    except Exception as exc:
        # Not configured, not reachable, key absent — all reasons to try the
        # next layer, none a reason to end the run.
        logger.debug(f"secrets: Databricks scope '{scope}' did not supply {name}: {exc}")
        return None


#: The KV leaf hydration credentials live under, mirroring how the AI Gateway
#: credential lives at ``<app_name>/ai_gateway``.
VAULT_LEAF = "data_hydration"


def _vault_path() -> str:
    """``<app_name>/data_hydration``, or the bare leaf when no app name is set.

    The same ``<app_name>/<leaf>`` convention
    :attr:`app_config.vault.VaultConfig.resolved_ai_gateway_path` follows — one
    prefix per application, one leaf per consumer.
    """
    from app_config.vault import get_vault_client

    app_name = get_vault_client().config.app_name
    return f"{app_name}/{VAULT_LEAF}" if app_name else VAULT_LEAF


def _from_vault(name: str) -> str | None:
    """The secret from Vault's KV store, or ``None`` if unavailable."""
    try:
        from app_config.vault import get_secret

        value = get_secret(_vault_path(), name)
        return str(value) if value else None
    except Exception as exc:
        logger.debug(f"secrets: Vault did not supply {name}: {exc}")
        return None


def _from_env(name: str) -> str | None:
    """The secret from ``DATA_HYDRATION_<NAME>``."""
    return os.environ.get(_env_name(name)) or None


def resolve_secret(
    name: str,
    *,
    scope: str | None = None,
    required: bool = True,
) -> str | None:
    """The credential called *name*, from the first layer that has it.

    Parameters
    ----------
    name
        A logical credential name — ``oracle_password``, ``sftp_passphrase``.
        Used verbatim as the Databricks scope key and the Vault field, and
        upper-cased into ``DATA_HYDRATION_<NAME>`` for the environment.
    scope
        Databricks secret scope to read. Normally
        :attr:`~data_hydration.config.HydrationConfig.secret_scope`.
    required
        When ``True`` (default), exhausting every layer raises. Pass ``False``
        for a genuinely optional credential — an sFTP key passphrase, say —
        where ``None`` means "there isn't one", not "we could not find it".

    Raises
    ------
    HydrationCredentialError
        *required* and no layer supplied a value.
    """
    for layer, fetch in (
        ("Databricks secret scope", lambda: _from_databricks(name, scope)),
        ("Vault", lambda: _from_vault(name)),
        ("environment", lambda: _from_env(name)),
    ):
        value = fetch()
        if value:
            logger.info(f"resolve_secret: '{name}' resolved from the {layer}")
            return value
    if not required:
        logger.debug(f"resolve_secret: optional credential '{name}' not set")
        return None
    raise HydrationCredentialError(
        f"no credential '{name}' found: tried the Databricks secret scope "
        f"({scope or 'not configured'}), Vault, and the environment variable "
        f"{_env_name(name)}"
    )


class EntraCredential:
    """An Azure-SDK ``TokenCredential`` backed by this repo's Entra ID login.

    The Azure storage SDKs take a credential object with a ``get_token`` method
    rather than a password, and :mod:`app_config.azure` already owns the MSAL
    login this deployment authenticates with — service principal or device code,
    with the TLS and proxy settings a corporate network needs. Duck-typing that
    surface here is what keeps hydration on the one credential chain instead of
    building a second Azure login beside it.

    ``azure.core`` is not imported: the SDK only needs an object with the right
    method, and the named tuple it expects is constructed lazily so this module
    stays importable with no Azure package installed.
    """

    def __init__(self, scopes: tuple[str, ...] = ()) -> None:
        #: Storage's resource scope. Overridable for sovereign clouds.
        self.scopes = scopes or ("https://storage.azure.com/.default",)

    def get_token(self, *scopes: str, **kwargs: Any) -> Any:
        """An ``AccessToken(token, expires_on)`` for *scopes*.

        The SDK passes scopes positionally and a bag of keyword arguments
        (``tenant_id``, ``claims``, ...) this chain has no use for; they are
        accepted and ignored so the signature stays compatible.
        """
        from app_config.azure import get_azure_client

        wanted = tuple(scopes) or self.scopes
        client = get_azure_client()
        token = client.get_token(wanted)
        expires_on = _expiry_for(client, wanted)
        try:
            from azure.core.credentials import AccessToken

            return AccessToken(token, expires_on)
        except ImportError:
            # A stand-in with the same two fields, so tests (and any caller that
            # only reads the token) work without azure-core installed.
            from collections import namedtuple

            return namedtuple("AccessToken", "token expires_on")(token, expires_on)

    def __str__(self) -> str:
        return f"EntraCredential({' '.join(self.scopes)})"


def _expiry_for(client: Any, scopes: tuple[str, ...]) -> int:
    """The cached expiry epoch for *scopes*, or a conservative near-future one.

    :class:`app_config.azure.AzureAuthClient` keeps ``(token, expires_at)`` per
    scope set; reading it avoids re-deriving a lifetime this module does not
    know. When the shape is not what is expected, a short expiry is safer than a
    long one — the SDK will simply ask again.
    """
    import time

    cached = getattr(client, "_tokens", {}).get(scopes)
    if cached and len(cached) == 2:
        return int(cached[1])
    return int(time.time() + 60)
