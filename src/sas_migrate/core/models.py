"""Base models and versioned JSON serialization for v2 contracts."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    """Strict immutable base for values crossing a v2 package boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class VersionedContract(ContractModel):
    """A JSON contract whose wire schema is explicitly versioned."""

    schema_version: Literal[2] = 2

    def to_json(self) -> str:
        """Serialize using the canonical Pydantic JSON representation."""

        return self.model_dump_json()

    @classmethod
    def from_json(cls, value: str | bytes) -> Self:
        """Validate and deserialize a versioned wire value."""

        return cls.model_validate_json(value)
