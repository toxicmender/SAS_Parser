"""Lazy HashiCorp Vault credential adapter with token, AppRole, or JWT auth."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import Field, SecretStr

from sas_migrate.application.ports import (
    AccessTokenProvider,
    CredentialProvider,
    CredentialProviderUnavailable,
    CredentialValue,
)
from sas_migrate.config import VaultSettings
from sas_migrate.core.models import ContractModel
from sas_migrate.observability import redact_text


class VaultSecretReference(ContractModel):
    path: str = Field(min_length=1)
    key: str = Field(min_length=1)
    mount_point: str | None = None


VaultClientFactory = Callable[[VaultSettings], Any]


def _hvac_client(settings: VaultSettings) -> Any:
    if settings.address is None:
        raise CredentialProviderUnavailable("Vault address is not configured")
    try:
        import hvac
    except ImportError as exc:
        raise CredentialProviderUnavailable(
            "hvac is required for Vault credentials; install 'sas-parser[vault]'"
        ) from exc
    return hvac.Client(
        url=settings.address,
        namespace=settings.namespace,
        verify=settings.verify,
        # hvac's stubs say int, but requests accepts fractional seconds.
        timeout=settings.timeout,  # pyright: ignore[reportArgumentType]
    )


class VaultCredentialProvider:
    def __init__(
        self,
        settings: VaultSettings,
        references: Mapping[str, VaultSecretReference],
        *,
        bootstrap: CredentialProvider,
        azure_tokens: AccessTokenProvider | None = None,
        client_factory: VaultClientFactory = _hvac_client,
    ) -> None:
        self._settings = settings
        self._references = dict(references)
        self._bootstrap = bootstrap
        self._azure_tokens = azure_tokens
        self._client_factory = client_factory
        self._client: Any | None = None

    async def _authenticated_client(self) -> Any:
        if self._client is not None:
            return self._client
        client = self._client_factory(self._settings)
        token = await self._bootstrap.get("vault_token")
        if token is not None:
            client.token = token.value.get_secret_value()
        else:
            role_id = await self._bootstrap.get("vault_role_id")
            secret_id = await self._bootstrap.get("vault_secret_id")
            if role_id is not None and secret_id is not None:
                client.auth.approle.login(
                    role_id=role_id.value.get_secret_value(),
                    secret_id=secret_id.value.get_secret_value(),
                    use_token=True,
                )
            elif self._azure_tokens is not None and self._settings.app_name:
                access = await self._azure_tokens.get_token(
                    self._settings.azure_scopes
                )
                client.auth.jwt.jwt_login(
                    role=self._settings.app_name,
                    jwt=access.value.get_secret_value(),
                    path=self._settings.auth_path,
                    use_token=True,
                )
            else:
                raise CredentialProviderUnavailable(
                    "Vault needs vault_token, vault_role_id + vault_secret_id, "
                    "or an Azure token provider with vault.app_name"
                )
        self._client = client
        return client

    async def get(self, name: str) -> CredentialValue | None:
        reference = self._references.get(name)
        if reference is None:
            return None
        client = await self._authenticated_client()
        mount = reference.mount_point or self._settings.mount_point
        try:
            if self._settings.kv_version == 2:
                response = client.secrets.kv.v2.read_secret_version(
                    path=reference.path,
                    mount_point=mount,
                    raise_on_deleted_version=True,
                )
                data = response["data"]["data"]
            else:
                response = client.secrets.kv.v1.read_secret(
                    path=reference.path,
                    mount_point=mount,
                )
                data = response["data"]
            value = data.get(reference.key)
        except Exception as exc:
            detail = redact_text(str(exc))
            raise CredentialProviderUnavailable(
                f"could not read Vault credential {name!r} from "
                f"{mount}/{reference.path}: {type(exc).__name__}: {detail}"
            ) from exc
        if value is None:
            return None
        if not isinstance(value, str):
            raise CredentialProviderUnavailable(
                f"Vault credential {name!r} must be text, got {type(value).__name__}"
            )
        return CredentialValue(
            name=name,
            value=SecretStr(value),
            source=f"vault:{mount}/{reference.path}#{reference.key}",
        )


__all__ = ["VaultCredentialProvider", "VaultSecretReference"]
