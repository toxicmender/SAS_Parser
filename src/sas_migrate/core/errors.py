"""Stable exception hierarchy for v2 public boundaries."""

from __future__ import annotations


class SasMigrateError(Exception):
    """Base class for expected v2 application failures."""


class ContractError(SasMigrateError, ValueError):
    """Serialized data or a boundary value violates a v2 contract."""


class TargetResolutionError(ContractError):
    """A public boundary requested an unsupported translation target."""


class ResponseContractError(ContractError):
    """A provider response cannot be represented by the response contract."""


class TokenBudgetError(ContractError):
    """Prompt composition exceeds a configured token policy."""


class ArchitectureError(SasMigrateError):
    """A source import violates the v2 dependency direction."""
