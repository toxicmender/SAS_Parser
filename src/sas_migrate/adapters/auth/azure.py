"""Lazy MSAL client-credential adapter behind the access-token port."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol, cast

from pydantic import SecretStr

from sas_migrate.application.ports import (
    AccessToken,
    CredentialProvider,
    CredentialProviderUnavailable,
)
from sas_migrate.config import AzureSettings
from sas_migrate.observability import redact_text


class MsalApplication(Protocol):
    def acquire_token_for_client(self, scopes: list[str]) -> dict[str, Any]: ...


MsalApplicationFactory = Callable[[AzureSettings, str], MsalApplication]


def _msal_application(settings: AzureSettings, client_secret: str) -> MsalApplication:
    if settings.client_id is None or settings.tenant_id is None:
        raise CredentialProviderUnavailable(
            "Azure client_id and tenant_id are required for client credentials"
        )
    try:
        import msal
    except ImportError as exc:
        raise CredentialProviderUnavailable(
            "msal is required for Entra ID authentication; install "
            "'sas-parser[azure]'"
        ) from exc
    authority = f"{settings.authority_host.rstrip('/')}/{settings.tenant_id}"
    return cast(
        MsalApplication,
        msal.ConfidentialClientApplication(
            settings.client_id,
            client_credential=client_secret,
            authority=authority,
        ),
    )


class MsalAccessTokenProvider:
    def __init__(
        self,
        settings: AzureSettings,
        credentials: CredentialProvider,
        *,
        credential_name: str = "azure_client_secret",
        application_factory: MsalApplicationFactory = _msal_application,
    ) -> None:
        self._settings = settings
        self._credentials = credentials
        self._credential_name = credential_name
        self._application_factory = application_factory
        self._application: MsalApplication | None = None

    async def _application_client(self) -> MsalApplication:
        if self._application is not None:
            return self._application
        if self._settings.flow != "client_credentials":
            raise CredentialProviderUnavailable(
                "MsalAccessTokenProvider supports client_credentials only"
            )
        credential = await self._credentials.get(self._credential_name)
        if credential is None:
            raise CredentialProviderUnavailable(
                f"credential {self._credential_name!r} is not configured"
            )
        self._application = self._application_factory(
            self._settings,
            credential.value.get_secret_value(),
        )
        return self._application

    async def get_token(self, scopes: tuple[str, ...] = ()) -> AccessToken:
        requested = scopes or self._settings.scopes
        if not requested:
            raise CredentialProviderUnavailable("at least one Azure scope is required")
        application = await self._application_client()
        response = application.acquire_token_for_client(scopes=list(requested))
        value = response.get("access_token")
        if not isinstance(value, str) or not value:
            error = redact_text(str(response.get("error") or "token request failed"))
            description = redact_text(str(response.get("error_description") or ""))
            detail = f": {description}" if description else ""
            raise CredentialProviderUnavailable(f"Azure token error {error}{detail}")
        expires_in = response.get("expires_in")
        expires_at = (
            int(time.time()) + int(expires_in)
            if isinstance(expires_in, (int, str)) and str(expires_in).isdigit()
            else None
        )
        return AccessToken(
            value=SecretStr(value),
            source="azure:msal-client-credentials",
            expires_at_epoch=expires_at,
        )


__all__ = ["MsalAccessTokenProvider", "MsalApplication", "MsalApplicationFactory"]
