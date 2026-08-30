"""Semantic identifiers used by run, item, thread and call contracts."""

from __future__ import annotations

from typing import Annotated

from pydantic import StringConstraints

type NonEmptyId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
]

type RunId = NonEmptyId
type ThreadId = NonEmptyId
type ItemId = NonEmptyId
type ChunkId = NonEmptyId
type CallId = NonEmptyId
type ArtifactId = NonEmptyId

__all__ = [
    "ArtifactId",
    "CallId",
    "ChunkId",
    "ItemId",
    "NonEmptyId",
    "RunId",
    "ThreadId",
]
