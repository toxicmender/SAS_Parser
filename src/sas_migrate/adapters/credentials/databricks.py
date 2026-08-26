"""Lazy Databricks secret-scope credential adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from pydantic import Field, SecretStr

from sas_migrate.application.ports import CredentialProviderUnavailable, CredentialValue
from sas_migrate.core.models import ContractModel


class DatabricksSecretReference(ContractModel):
    scope: str = Field(min_length=1)
    key: str = Field(min_length=1)


class DatabricksSecretsAccessor(Protocol):
    def get(self, *, scope: str, key: str) -> str: ...


def _runtime_accessor() -> DatabricksSecretsAccessor:
    try:
        from databricks.sdk.runtime import dbutils
    except ImportError as exc:
        raise CredentialProviderUnavailable(
            "Databricks runtime secrets are unavailable; install the databricks "
            "extra and run this adapter on a Databricks cluster"
        ) from exc
    return dbutils.secrets


class DatabricksSecretCredentialProvider:
    def __init__(
        self,
        references: Mapping[str, DatabricksSecretReference],
        *,
        accessor: DatabricksSecretsAccessor | None = None,
    ) -> None:
        self._references = dict(references)
        self._configured_accessor = accessor
        self._resolved_accessor: DatabricksSecretsAccessor | None = None

    def _accessor(self) -> DatabricksSecretsAccessor:
        if self._configured_accessor is not None:
            return self._configured_accessor
        if self._resolved_accessor is None:
            self._resolved_accessor = _runtime_accessor()
        return self._resolved_accessor

    async def get(self, name: str) -> CredentialValue | None:
        reference = self._references.get(name)
        if reference is None:
            return None
        value = self._accessor().get(scope=reference.scope, key=reference.key)
        if not value:
            return None
        return CredentialValue(
            name=name,
            value=SecretStr(value),
            source=f"databricks:{reference.scope}/{reference.key}",
        )


__all__ = [
    "CredentialProviderUnavailable",
    "DatabricksSecretCredentialProvider",
    "DatabricksSecretReference",
    "DatabricksSecretsAccessor",
]
