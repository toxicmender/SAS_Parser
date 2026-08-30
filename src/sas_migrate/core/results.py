"""Typed success/failure results for ports that should not throw."""

from __future__ import annotations

from typing import Literal

from .models import ContractModel


class Success[T](ContractModel):
    """Successful operation carrying a typed value."""

    ok: Literal[True] = True
    value: T


class Failure[E](ContractModel):
    """Expected failure carrying a typed error value."""

    ok: Literal[False] = False
    error: E


type Result[T, E] = Success[T] | Failure[E]

__all__ = ["Failure", "Result", "Success"]
