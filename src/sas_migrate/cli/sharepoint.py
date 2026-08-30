"""Credential composition for operational SharePoint CLI commands."""

from __future__ import annotations

from collections.abc import Mapping

from sas_migrate.application.ports import (
    AccessToken,
    AccessTokenProvider,
    CredentialProviderUnavailable,
)
from sas_migrate.config import InfrastructureSettings


class DatabricksSharePointTokenProvider:
    """Resolve a dedicated SharePoint principal from a Databricks secret scope."""

    def __init__(self, settings: InfrastructureSettings) -> None:
        self._settings = settings
        self._resolved: AccessTokenProvider | None = None

    async def get_token(self, scopes: tuple[str, ...] = ()) -> AccessToken:
        from sas_migrate.adapters.auth import MsalAccessTokenProvider
        from sas_migrate.adapters.credentials import (
            DatabricksSecretCredentialProvider,
            DatabricksSecretReference,
        )

        if self._resolved is None:
            sharepoint = self._settings.sharepoint
            scope = sharepoint.secret_scope
            if scope is None:
                raise CredentialProviderUnavailable(
                    "SharePoint secret scope is not configured"
                )
            names = {
                "sharepoint_tenant_id": sharepoint.tenant_id_key,
                "sharepoint_client_id": sharepoint.client_id_key,
                "sharepoint_client_secret": sharepoint.client_secret_key,
            }
            credentials = DatabricksSecretCredentialProvider(
                {
                    name: DatabricksSecretReference(scope=scope, key=key)
                    for name, key in names.items()
                }
            )
            tenant = await credentials.get("sharepoint_tenant_id")
            client = await credentials.get("sharepoint_client_id")
            if tenant is None or client is None:
                missing = [
                    name
                    for name, value in (
                        ("tenant id", tenant),
                        ("client id", client),
                    )
                    if value is None
                ]
                raise CredentialProviderUnavailable(
                    "SharePoint secret scope lacks " + " and ".join(missing)
                )
            azure = self._settings.azure.model_copy(
                update={
                    "tenant_id": tenant.value.get_secret_value(),
                    "client_id": client.value.get_secret_value(),
                    "flow": "client_credentials",
                    "scopes": sharepoint.scopes,
                }
            )
            self._resolved = MsalAccessTokenProvider(
                azure,
                credentials,
                credential_name="sharepoint_client_secret",
            )
        return await self._resolved.get_token(scopes)


def sharepoint_token_provider(
    settings: InfrastructureSettings,
    *,
    environ: Mapping[str, str] | None = None,
) -> AccessTokenProvider:
    """Select secret-scope or environment-backed SharePoint authentication."""

    if settings.sharepoint.secret_scope is not None:
        return DatabricksSharePointTokenProvider(settings)

    from sas_migrate.adapters.auth import MsalAccessTokenProvider
    from sas_migrate.adapters.credentials import EnvironmentCredentialProvider

    azure = settings.azure.model_copy(
        update={"scopes": settings.sharepoint.scopes}
    )
    return MsalAccessTokenProvider(
        azure,
        EnvironmentCredentialProvider(
            {"azure_client_secret": "AZURE_CLIENT_SECRET"},
            environ=environ,
        ),
    )


__all__ = ["DatabricksSharePointTokenProvider", "sharepoint_token_provider"]
