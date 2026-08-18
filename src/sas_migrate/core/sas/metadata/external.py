"""External path and engine metadata concern."""

from __future__ import annotations

from ..models import SasChunkMetadata, SasEngineRef, SasPathRef


def path_references(metadata: SasChunkMetadata) -> tuple[SasPathRef, ...]:
    return tuple(metadata.external_refs)


def engine_references(metadata: SasChunkMetadata) -> tuple[SasEngineRef, ...]:
    return tuple(metadata.engine_refs)
