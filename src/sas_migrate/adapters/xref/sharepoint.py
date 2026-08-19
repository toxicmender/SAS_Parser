"""SharePoint-list XREF mapping source over an injected transport."""

from __future__ import annotations

from sas_migrate.application.ports.xref import XrefListTransport
from sas_migrate.application.xref.mapping import classify_rows
from sas_migrate.application.xref.models import XrefMappings, XrefRow

XREF_FIELDS: dict[str, str] = {
    "marker": "Title",
    "application_name": "Application",
    "source": "OriginalValue",
    "target": "NewValue",
}


def _text(value: object) -> str:
    return str(value).strip() if value is not None else ""


class SharePointXrefSource:
    """Load and filter XREF list rows without owning authentication or config."""

    def __init__(self, transport: XrefListTransport, list_id: str) -> None:
        self._transport = transport
        self._list_id = list_id

    def load(self, application_name: str) -> XrefMappings:
        items = self._transport.list_items(self._list_id)
        wanted = application_name.strip().casefold()
        rows: list[XrefRow] = []
        for item in items:
            raw_fields = item.get("fields")
            if not isinstance(raw_fields, dict):
                continue
            projected = {
                name: _text(raw_fields.get(column))
                for name, column in XREF_FIELDS.items()
            }
            if projected["application_name"].casefold() != wanted:
                continue
            if not projected["source"] or not projected["target"]:
                continue
            rows.append(XrefRow(**projected))
        return classify_rows(rows)


__all__ = ["XREF_FIELDS", "SharePointXrefSource"]
