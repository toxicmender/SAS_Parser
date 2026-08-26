"""Secret retrieval port that prevents accidental plain-text representation."""

from __future__ import annotations

from typing import Protocol

from pydantic import SecretStr

from sas_migrate.core.models import ContractModel


class CredentialProviderUnavailable(RuntimeError):
    """A configured credential backend cannot service the request."""


class CredentialValue(ContractModel):
    name: str
    value: SecretStr
    source: str


class CredentialProvider(Protocol):
    async def get(self, name: str) -> CredentialValue | None: ...


__all__ = [
    "CredentialProvider",
    "CredentialProviderUnavailable",
    "CredentialValue",
]
