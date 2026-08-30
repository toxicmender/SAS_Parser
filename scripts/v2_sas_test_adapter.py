"""Pytest-only aliases that run the legacy SAS suites against the v2 core.

Loaded with ``pytest -p scripts.v2_sas_test_adapter`` before test collection.
No production module imports the legacy namespace through this adapter.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sas_migrate.core.tokens import TokenBudgetPolicy

_sas = importlib.import_module("sas_migrate.core.sas")
_batching = importlib.import_module("sas_migrate.core.sas.batching")
_chunking = importlib.import_module("sas_migrate.core.sas.chunking")

# Preserve legacy logger names for assertions that treat diagnostics as part of
# the compatibility contract. Function globals still point at these modules.
_batching.__dict__["logger"] = logging.getLogger("chunker.batcher")
_chunking.__dict__["logger"] = logging.getLogger("chunker.chunker")


class _LegacySasSemanticChunker(_chunking.SasSemanticChunker):
    """Translate legacy config-sentinel values without importing app_config."""

    def __init__(
        self,
        *,
        min_words: int | None = None,
        max_words: int | None = None,
        timeout: float | None = 60.0,
    ) -> None:
        super().__init__(
            min_words=300 if min_words is None else min_words,
            max_words=700 if max_words is None else max_words,
            timeout=timeout,
        )


def _legacy_coalesce(
    items: list[Any],
    *,
    max_chunks: int = 8,
    max_tokens: int | None = None,
    item_cost: Callable[[Any], int] | None = None,
) -> list[Any]:
    policy = (
        None
        if max_tokens is None
        else TokenBudgetPolicy(
            max_input_tokens=max_tokens,
            reserved_output_tokens=0,
            safety_margin_tokens=0,
        )
    )
    return _batching.coalesce_into_batches(
        items,
        max_chunks=max_chunks,
        policy=policy,
        item_cost=item_cost,
    )


def _module_shim(name: str, source: types.ModuleType) -> types.ModuleType:
    shim = types.ModuleType(name)
    shim.__dict__.update(source.__dict__)
    shim.__name__ = name
    return shim


_batching_shim = _module_shim("chunker.batcher", _batching)
_batching_shim.__dict__["coalesce_into_batches"] = _legacy_coalesce

_chunking_shim = _module_shim("chunker.chunker", _chunking)
_chunking_shim.__dict__["SasSemanticChunker"] = _LegacySasSemanticChunker

_root = _module_shim("chunker", _sas)
_root.__path__ = []
_root.__dict__["SasSemanticChunker"] = _LegacySasSemanticChunker

sys.modules.update(
    {
        "chunker": _root,
        "chunker.batcher": _batching_shim,
        "chunker.chunker": _chunking_shim,
        "chunker.keywords": importlib.import_module("sas_migrate.core.sas.keywords"),
        "chunker.metadata": importlib.import_module(
            "sas_migrate.core.sas.metadata.extraction"
        ),
        "chunker.models": importlib.import_module("sas_migrate.core.sas.models"),
        "chunker.paths": importlib.import_module("sas_migrate.core.sas.paths"),
        "chunker.scanner": importlib.import_module("sas_migrate.core.sas.scanner"),
    }
)
