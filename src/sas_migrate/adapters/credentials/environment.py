"""Environment-backed credentials for local and CI composition roots."""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import SecretStr

from sas_migrate.application.ports import CredentialValue


class EnvironmentCredentialProvider:
    def __init__(
        self,
        names: Mapping[str, str],
        *,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._names = dict(names)
        self._environ = os.environ if environ is None else environ

    async def get(self, name: str) -> CredentialValue | None:
        environment_name = self._names.get(name)
        if environment_name is None:
            return None
        value = self._environ.get(environment_name)
        if value is None or not value.strip():
            return None
        return CredentialValue(
            name=name,
            value=SecretStr(value),
            source=f"environment:{environment_name}",
        )


__all__ = ["EnvironmentCredentialProvider"]
