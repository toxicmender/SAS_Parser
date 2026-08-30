"""XREF mapping-source boundaries."""

from __future__ import annotations

from typing import Protocol

from sas_migrate.application.xref.models import XrefMappings


class XrefMappingSource(Protocol):
    """Load mappings for one application from a concrete source."""

    def load(self, application_name: str) -> XrefMappings: ...


class XrefListTransport(Protocol):
    def list_items(self, list_id: str) -> list[dict[str, object]]: ...


class XrefFileTransport(Protocol):
    def read_file(self, path: str) -> bytes: ...


__all__ = ["XrefFileTransport", "XrefListTransport", "XrefMappingSource"]
