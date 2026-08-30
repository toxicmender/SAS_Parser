"""Ports for assessment profiles and optional LLM review."""

from __future__ import annotations

from typing import Protocol


class AssessmentProfileRepository(Protocol):
    def load(self, name: str) -> dict[str, object]: ...

    def names(self) -> tuple[str, ...]: ...


class AssessmentReviewer(Protocol):
    async def review(self, prompt: str) -> str: ...


__all__ = ["AssessmentProfileRepository", "AssessmentReviewer"]
