"""Embedding caches with no import-time numerical dependency."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from sas_migrate.application.ports.knowledge import EmbeddingVector

logger = logging.getLogger(__name__)


class InMemoryEmbeddingCache:
    def __init__(self) -> None:
        self._values: dict[str, EmbeddingVector] = {}

    def get_many(self, keys: tuple[str, ...]) -> dict[str, EmbeddingVector]:
        return {key: self._values[key] for key in keys if key in self._values}

    def put_many(self, values: dict[str, EmbeddingVector]) -> None:
        self._values.update(values)


class NpzEmbeddingCache:
    """Atomic NumPy cache keyed by the ranker's provider-scoped content hash."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def get_many(self, keys: tuple[str, ...]) -> dict[str, EmbeddingVector]:
        values = self._load()
        return {key: values[key] for key in keys if key in values}

    def put_many(self, values: dict[str, EmbeddingVector]) -> None:
        if not values:
            return
        current = self._load()
        current.update(values)
        dimensions = {len(vector) for vector in current.values()}
        if len(dimensions) != 1 or 0 in dimensions:
            raise ValueError("embedding cache vectors must share a non-zero dimension")

        import numpy as np

        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            dir=self._path.parent,
            prefix=f".{self._path.name}.",
            suffix=".npz",
        )
        os.close(handle)
        try:
            ordered_keys = sorted(current)
            vectors = np.asarray([current[key] for key in ordered_keys], dtype=np.float32)
            np.savez(
                temporary,
                keys=np.asarray(ordered_keys, dtype=str),
                vectors=vectors,
            )
            os.replace(temporary, self._path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _load(self) -> dict[str, EmbeddingVector]:
        if not self._path.exists():
            return {}
        try:
            import numpy as np

            with np.load(self._path, allow_pickle=False) as data:
                keys = data["keys"]
                vectors = data["vectors"]
                if vectors.ndim != 2 or len(keys) != len(vectors):
                    raise ValueError("invalid embedding cache shape")
                return {
                    str(key): tuple(float(value) for value in vectors[index])
                    for index, key in enumerate(keys)
                }
        except (KeyError, OSError, ValueError) as exc:
            logger.warning("Ignoring unreadable embedding cache %s: %s", self._path, exc)
            return {}


__all__ = ["InMemoryEmbeddingCache", "NpzEmbeddingCache"]
