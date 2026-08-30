"""Macro metadata concern, projected without rescanning source text."""

from __future__ import annotations

from ..models import SasChunkMetadata


def defined_macros(metadata: SasChunkMetadata) -> tuple[str, ...]:
    return tuple(metadata.defines_macros)


def invoked_macros(metadata: SasChunkMetadata) -> tuple[str, ...]:
    return tuple(metadata.invokes_macros)


def macro_variable_inputs(metadata: SasChunkMetadata) -> tuple[str, ...]:
    return tuple(metadata.consumes_macrovars)


def macro_variable_outputs(metadata: SasChunkMetadata) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (*metadata.produces_macrovars, *metadata.declared_macro_vars)
        )
    )
