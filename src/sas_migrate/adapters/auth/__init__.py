"""Authentication adapters."""

from .azure import MsalAccessTokenProvider, MsalApplication, MsalApplicationFactory

__all__ = ["MsalAccessTokenProvider", "MsalApplication", "MsalApplicationFactory"]
