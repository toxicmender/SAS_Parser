"""Short-lived access-token boundary for service adapters."""

from __future__ import annotations

from typing import Protocol

from pydantic import SecretStr

from sas_migrate.core.models import ContractModel


class AccessToken(ContractModel):
    value: SecretStr
    source: str
    expires_at_epoch: int | None = None


class AccessTokenProvider(Protocol):
    async def get_token(self, scopes: tuple[str, ...] = ()) -> AccessToken: ...


__all__ = ["AccessToken", "AccessTokenProvider"]
