"""CSV-backed XREF mapping sources."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

from sas_migrate.application.ports.xref import XrefFileTransport
from sas_migrate.application.xref.mapping import classify_rows
from sas_migrate.application.xref.models import XrefMappings, XrefRow
from sas_migrate.core.sas import parse_databricks_mapping_csv


def parse_mapping_csv(text: str) -> XrefMappings:
    """Parse the compatible two-column SAS-to-Databricks CSV shape."""

    flat = parse_databricks_mapping_csv(text)
    return classify_rows(
        XrefRow(source=source, target=target)
        for source, target in flat.items()
    )


class CsvXrefSource:
    """Local CSV source; file I/O occurs only when ``load`` is invoked."""

    def __init__(
        self,
        path: str | pathlib.Path,
        *,
        read_bytes: Callable[[pathlib.Path], bytes] | None = None,
    ) -> None:
        self._path = pathlib.Path(path)
        self._read_bytes = read_bytes or pathlib.Path.read_bytes

    def load(self, application_name: str) -> XrefMappings:
        del application_name  # the compatible CSV shape is application-scoped
        body = self._read_bytes(self._path)
        mappings = parse_mapping_csv(body.decode("utf-8-sig"))
        if not mappings:
            raise ValueError(f"XREF CSV {self._path} parsed to zero entries")
        return mappings


class TransportCsvXrefSource:
    """CSV source over an injected file transport, including SharePoint."""

    def __init__(self, transport: XrefFileTransport, path: str) -> None:
        self._transport = transport
        self._path = path

    def load(self, application_name: str) -> XrefMappings:
        del application_name
        body = self._transport.read_file(self._path)
        mappings = parse_mapping_csv(body.decode("utf-8-sig"))
        if not mappings:
            raise ValueError(f"XREF CSV {self._path} parsed to zero entries")
        return mappings


__all__ = ["CsvXrefSource", "TransportCsvXrefSource", "parse_mapping_csv"]
