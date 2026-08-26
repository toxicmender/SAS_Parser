"""Ordered credential fallback without hiding configured-provider failures."""

from __future__ import annotations

from collections.abc import Iterable

from sas_migrate.application.ports import CredentialProvider, CredentialValue


class ChainedCredentialProvider:
    def __init__(self, providers: Iterable[CredentialProvider]) -> None:
        self._providers = tuple(providers)

    async def get(self, name: str) -> CredentialValue | None:
        for provider in self._providers:
            value = await provider.get(name)
            if value is not None:
                return value
        return None


__all__ = ["ChainedCredentialProvider"]
