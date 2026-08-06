"""The SharePoint request and conversion lists, and what their columns mean.

Two lists drive a run:

``sharepoint.list_id_sas_requests``
    One row per *application* to convert — its name, the source and
    destination languages, whether a macro file accompanies it, whether
    validation documents are wanted, and a ``Status`` this module writes back.
``sharepoint.list_id_sas_conversions``
    One row per *script* within a request — its name, the model chosen for it,
    and its own status.

Column names
------------
:data:`REQUEST_FIELDS` and :data:`CONVERSION_FIELDS` are the whole mapping,
in one place, from the attribute names the rest of the code uses to the
SharePoint **internal** column names. Those are not the display names, and
they are not tidy:

    'Macro File Name x003f '          <- a display name ending in "?"
    'Validation x0020 Documents x0020 '

The encoded characters and the trailing spaces are literal and load-bearing —
SharePoint derives an internal name once, from the column's *original* display
name, and then keeps it forever. Getting one wrong yields ``None`` rather than
an error, so they are transcribed here exactly and never rebuilt from a
display name.

Logger name: ``conversion.requests``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app_config.sharepoint import (
    SharePointConfig,
    SharePointError,
    field_text,
    project_rows,
    resolve_client,
    resolve_config,
)

logger = logging.getLogger(__name__)

# attribute -> SharePoint internal column name. See the module docstring on
# why these are transcribed rather than derived.
REQUEST_FIELDS: dict[str, str] = {
    "application_name": "Application Name",
    "input_language": "Source Language",
    "output_language": "Destination Language",
    "macro_file_name": "Macro File Name x003f ",
    "is_validation_required": "Validation x0020 Documents x0020 ",
    # Not part of the reference's projection, but the column exists — it is
    # what update_request_status writes — and pending_requests needs to read it.
    "status": "Status",
}

CONVERSION_FIELDS: dict[str, str] = {
    "app_request_id": "Request_ID",
    "script_name": "Script_Name",
    "preferred_llm": "Model",
    "status": "Status",
}

# The column update_request_status writes. Named separately from the mapping
# so the write and the read cannot drift apart.
STATUS_FIELD = REQUEST_FIELDS["status"]

# Statuses meaning "this row has been dealt with". Compared case-insensitively
# after stripping; anything else (including blank) is outstanding.
COMPLETED_STATUSES = frozenset({"completed", "complete", "done", "finished"})

# Strings a SharePoint Yes/No column may surface as through Graph's untyped
# additional_data. Compared case-insensitively after stripping.
_TRUE_STRINGS = frozenset({"true", "1", "yes", "y", "on"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "n", "off", ""})


@dataclass
class ConversionRequest:
    """One row of the requests list.

    Attributes
    ----------
    item_id : str | None
        The SharePoint list-item id, which is what
        :func:`update_request_status` writes back against.
    application_name : str
        The application to convert — also the folder its scripts live under
        (see :mod:`conversion.paths`).
    input_language, output_language : str | None
        Source and destination languages as the row names them.
    macro_file_name : str | None
        An accompanying macro file, when the row names one.
    is_validation_required : bool
        Whether validation documents are wanted for this application.
    status : str | None
        The row's current status, as last written.
    """

    item_id: str | None
    application_name: str
    input_language: str | None = None
    output_language: str | None = None
    macro_file_name: str | None = None
    is_validation_required: bool = False
    status: str | None = None

    @property
    def is_pending(self) -> bool:
        """True while this row has not been marked done — see
        :data:`COMPLETED_STATUSES`."""
        return (self.status or "").strip().lower() not in COMPLETED_STATUSES


@dataclass
class ConversionItem:
    """One row of the conversions list: a single script and its chosen model.

    Attributes
    ----------
    item_id : str | None
        The SharePoint list-item id.
    app_request_id : int | None
        Which request row this script belongs to. ``None`` when the column is
        blank or unparseable — a malformed row is reported, not fatal.
    script_name : str | None
        The source file's name.
    preferred_llm : str | None
        The model as the operator chose it, in the gateway's
        ``"provider: model"`` spelling. Parse it with
        :func:`llm_client.client._split_model` — the same parser the client
        uses, not a second one.
    status : str | None
        This script's own status.
    """

    item_id: str | None
    app_request_id: int | None = None
    script_name: str | None = None
    preferred_llm: str | None = None
    status: str | None = None


def _flag(fields: dict[str, Any], column: str, *, default: bool = False) -> bool:
    """
    A SharePoint Yes/No column coerced to ``bool``.

    Graph surfaces list fields as untyped ``additional_data``: the same column
    may arrive as a Python ``bool``, an ``int``, or a string
    (``"true"`` / ``"0"`` / ``"Yes"``). An unrecognised value falls back to
    *default* with a WARNING rather than being guessed at.
    """
    value = fields.get(column)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):  # bool is handled above
        return value != 0
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_STRINGS:
            return True
        if token in _FALSE_STRINGS:
            return False
    logger.warning(
        f"_flag: unrecognised boolean {value!r} in column {column!r}; "
        f"using default {default}"
    )
    return default


def format_request_item_params(item: dict[str, Any]) -> ConversionRequest:
    """
    One raw list item (``{id, web_url, fields}``) as a
    :class:`ConversionRequest`.

    Raises
    ------
    SharePointError
        The row has no application name, which is the one field nothing
        downstream can proceed without — it names both the folder to read and
        the folder to write.
    """
    fields = item.get("fields") or {}
    application_name = field_text(fields, REQUEST_FIELDS["application_name"])
    if not application_name:
        raise SharePointError(
            f"request row {item.get('id')!r} has no "
            f"{REQUEST_FIELDS['application_name']!r}; it names both the input "
            f"and the output folder, so the row cannot be processed"
        )
    return ConversionRequest(
        item_id=item.get("id"),
        application_name=application_name,
        input_language=field_text(fields, REQUEST_FIELDS["input_language"]),
        output_language=field_text(fields, REQUEST_FIELDS["output_language"]),
        macro_file_name=field_text(fields, REQUEST_FIELDS["macro_file_name"]),
        is_validation_required=_flag(
            fields, REQUEST_FIELDS["is_validation_required"]
        ),
        status=field_text(fields, REQUEST_FIELDS["status"]),
    )


def format_conversion_item_params(item: dict[str, Any]) -> ConversionItem:
    """One raw list item as a :class:`ConversionItem`.

    A ``Request_ID`` that is not an integer is reported at WARNING and left
    ``None``: one malformed row must not abort the whole listing.
    """
    fields = item.get("fields") or {}
    raw_request_id = field_text(fields, CONVERSION_FIELDS["app_request_id"])
    app_request_id: int | None = None
    if raw_request_id is not None:
        try:
            app_request_id = int(raw_request_id)
        except ValueError:
            logger.warning(
                f"format_conversion_item_params: conversion row "
                f"{item.get('id')!r} has a non-numeric "
                f"{CONVERSION_FIELDS['app_request_id']!r}={raw_request_id!r}; "
                f"leaving it unset"
            )
    return ConversionItem(
        item_id=item.get("id"),
        app_request_id=app_request_id,
        script_name=field_text(fields, CONVERSION_FIELDS["script_name"]),
        preferred_llm=field_text(fields, CONVERSION_FIELDS["preferred_llm"]),
        status=field_text(fields, CONVERSION_FIELDS["status"]),
    )


def requests(
    *, client: Any | None = None, config: SharePointConfig | None = None
) -> list[ConversionRequest]:
    """
    Every row of the requests list, projected.

    A row without an application name is skipped with a WARNING rather than
    failing the read: one bad row must not hide every good one.
    """
    resolved = resolve_config(config)
    rows = resolve_client(client).list_items(resolved.list_id("requests"))
    out = project_rows(rows, format_request_item_params, label="requests")
    logger.info(f"requests: read {len(out)} request row(s) of {len(rows)}")
    return out


def pending_requests(
    *, client: Any | None = None, config: SharePointConfig | None = None
) -> list[ConversionRequest]:
    """The rows of :func:`requests` not yet marked done
    (:data:`COMPLETED_STATUSES`)."""
    outstanding = [row for row in requests(client=client, config=config) if row.is_pending]
    logger.info(f"pending_requests: {len(outstanding)} row(s) outstanding")
    return outstanding


def conversion_items(
    *, client: Any | None = None, config: SharePointConfig | None = None
) -> list[ConversionItem]:
    """Every row of the conversions list, projected."""
    resolved = resolve_config(config)
    rows = resolve_client(client).list_items(resolved.list_id("conversions"))
    return [format_conversion_item_params(row) for row in rows]


def update_request_status(
    item_id: str | int,
    new_status: str,
    *,
    client: Any | None = None,
    config: SharePointConfig | None = None,
) -> dict[str, Any]:
    """
    Write *new_status* to the ``Status`` column of one request row.

    The only write this package makes to a list. Returns the row's updated
    column values.

    Raises
    ------
    SharePointError
        The requests list is not configured, or the write failed.
    """
    resolved = resolve_config(config)
    return resolve_client(client).update_list_item(
        resolved.list_id("requests"), item_id, {STATUS_FIELD: new_status}
    )
