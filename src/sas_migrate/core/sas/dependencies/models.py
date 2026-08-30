"""Public dependency-edge facts independent of the batching implementation."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from ...models import ContractModel


class DependencyEdgeFamily(StrEnum):
    DATASET = "dataset"
    MACRO = "macro"
    MACRO_VARIABLE = "macro_variable"
    CONTEXT = "context"


class DependencyEdge(ContractModel):
    family: DependencyEdgeFamily
    kind: str
    from_chunk_id: str = Field(min_length=1)
    to_chunk_id: str = Field(min_length=1)
    reason: str
    cross_file: bool = False
