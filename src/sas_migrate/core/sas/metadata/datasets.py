"""Dataset metadata concern, projected from the shared extraction result."""

from __future__ import annotations

from ..models import SasChunkMetadata


def dataset_inputs(metadata: SasChunkMetadata) -> tuple[str, ...]:
    return tuple(metadata.input_datasets)


def dataset_outputs(metadata: SasChunkMetadata) -> tuple[str, ...]:
    return tuple(metadata.output_datasets)


def referenced_datasets(metadata: SasChunkMetadata) -> tuple[str, ...]:
    return tuple(metadata.referenced_datasets)
