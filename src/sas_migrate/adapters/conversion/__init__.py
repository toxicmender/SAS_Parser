"""Local and SharePoint conversion adapters."""

from .local import LocalConversionRequestRepository, LocalConversionSourceRepository
from .sharepoint import (
    SharePointConversionConfig,
    SharePointConversionRequestRepository,
    SharePointConversionSourceRepository,
    SharePointConversionTransport,
    preference_from_row,
    request_from_row,
)

__all__ = [
    "LocalConversionRequestRepository",
    "LocalConversionSourceRepository",
    "SharePointConversionConfig",
    "SharePointConversionRequestRepository",
    "SharePointConversionSourceRepository",
    "SharePointConversionTransport",
    "preference_from_row",
    "request_from_row",
]
