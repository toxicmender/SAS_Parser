"""Power Apps run-request settings and list-item parsing.

Submodule of :mod:`app_config`. A Power Apps canvas app writes one *conversion
request* per row into a SharePoint list; ``demo_run.py sharepoint`` reads that
list (through :mod:`app_config.sharepoint`), picks the requested row, and turns
it into the handful of parameters a pipeline run needs:

* whether inline **live validation** is on,
* the **application name** — also the document-library folder the input ``.sas``
  scripts live under,
* the **request id** — used as the pipeline ``thread_id`` so the run's
  conversation memory, run facts, and validation verdicts are keyed to it,
* the **model** the operator selected (from an enumerated list), and
* a **timestamp** that names the run's output folder.

This module holds only *configuration and pure parsing* — no network, no SDK.
The actual list read, file download, and result upload live in ``demo_run.py``
on top of :class:`app_config.sharepoint.SharePointClient`. Keeping the parsing
here makes it unit-testable without SharePoint and keeps ``app_config`` the
dependency-free leaf the rest of the package relies on.

Split of concerns
-----------------
* **Non-secret settings** — the list name and the five column *internal* names —
  resolve through :meth:`PowerAppsConfig.from_env`, which reads the
  ``POWERAPPS_*`` environment variables first and falls back to the optional
  ``powerapps`` section of ``config.json`` (via :func:`app_config.get_value`).
* **Secrets** — there are none here. Reaching SharePoint is authenticated by the
  Entra ID service principal that :mod:`app_config.azure` /
  :mod:`app_config.sharepoint` already use.

Column names
------------
SharePoint columns have an *internal* name that often differs from the display
name shown in the list UI (spaces become ``_x0020_``, a renamed column keeps its
original internal name, and so on). The defaults here — ``RequestId``,
``ApplicationName``, ``LiveValidation``, ``Model``, ``Timestamp`` — assume the
list was built with those internal names; override any that differ via
``POWERAPPS_*_FIELD`` / the ``powerapps`` section of ``config.json``.

Logger name: ``app_config.powerapps``.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from . import get_value, is_accessible_model

logger = logging.getLogger(__name__)

# Column internal-name defaults, applied when neither POWERAPPS_*_FIELD nor the
# powerapps section of config.json overrides them.
DEFAULT_REQUEST_ID_FIELD = "RequestId"
DEFAULT_APPLICATION_NAME_FIELD = "ApplicationName"
DEFAULT_LIVE_VALIDATION_FIELD = "LiveValidation"
DEFAULT_MODEL_FIELD = "Model"
DEFAULT_TIMESTAMP_FIELD = "Timestamp"

# Strings a SharePoint Yes/No column may surface as through Graph's untyped
# additional_data. Compared case-insensitively after stripping.
_TRUE_STRINGS = frozenset({"true", "1", "yes", "y", "on"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "n", "off", ""})


class PowerAppsError(RuntimeError):
    """A Power Apps request is missing, ambiguous, or malformed.

    A single error type so ``demo_run`` can wrap request resolution in one
    ``except PowerAppsError`` regardless of which stage failed; the message
    says which — no matching row, more than one, a missing required field, or
    a model the deployment cannot reach.
    """


@dataclass
class PowerAppsConfig:
    """Which SharePoint list, and the internal names of its columns.

    Construct it directly to pin values explicitly, or call :meth:`from_env`
    for the standard environment-then-``config.json`` resolution. There are no
    secret fields.

    Attributes
    ----------
    list_name : str | None
        The SharePoint list's id or display name.
        ``POWERAPPS_LIST_NAME`` / ``config.json`` ``powerapps.list_name``.
        Required at run time — resolving a request without it raises
        :class:`PowerAppsError`.
    request_id_field, application_name_field, live_validation_field,
    model_field, timestamp_field : str
        The list columns' *internal* names, each read from its
        ``POWERAPPS_<NAME>_FIELD`` env var / ``powerapps.<name>_field`` config
        key, defaulting to the ``DEFAULT_*_FIELD`` constants.
    """

    list_name: str | None = None
    request_id_field: str = DEFAULT_REQUEST_ID_FIELD
    application_name_field: str = DEFAULT_APPLICATION_NAME_FIELD
    live_validation_field: str = DEFAULT_LIVE_VALIDATION_FIELD
    model_field: str = DEFAULT_MODEL_FIELD
    timestamp_field: str = DEFAULT_TIMESTAMP_FIELD

    @classmethod
    def from_env(cls) -> "PowerAppsConfig":
        """Resolve settings from ``POWERAPPS_*`` env vars, then ``config.json``.

        Each column name falls back to its ``DEFAULT_*_FIELD`` when neither the
        environment nor the ``powerapps`` config section supplies one.
        """

        def field(env: str, key: str, default: str) -> str:
            return os.environ.get(env) or get_value("powerapps", key, default)

        return cls(
            list_name=(
                os.environ.get("POWERAPPS_LIST_NAME")
                or get_value("powerapps", "list_name")
            ),
            request_id_field=field(
                "POWERAPPS_REQUEST_ID_FIELD",
                "request_id_field",
                DEFAULT_REQUEST_ID_FIELD,
            ),
            application_name_field=field(
                "POWERAPPS_APPLICATION_NAME_FIELD",
                "application_name_field",
                DEFAULT_APPLICATION_NAME_FIELD,
            ),
            live_validation_field=field(
                "POWERAPPS_LIVE_VALIDATION_FIELD",
                "live_validation_field",
                DEFAULT_LIVE_VALIDATION_FIELD,
            ),
            model_field=field(
                "POWERAPPS_MODEL_FIELD", "model_field", DEFAULT_MODEL_FIELD
            ),
            timestamp_field=field(
                "POWERAPPS_TIMESTAMP_FIELD",
                "timestamp_field",
                DEFAULT_TIMESTAMP_FIELD,
            ),
        )


@dataclass
class RunRequest:
    """One normalised conversion request, parsed from a SharePoint list row.

    Attributes
    ----------
    request_id : str
        The Power Apps request id, used verbatim as the pipeline ``thread_id``.
    application_name : str
        The input folder under the document library and the output-tree root.
    model : str
        The selected chat-model string (already checked accessible).
    live_validation : bool
        Whether to attach the inline ``LiveValidator``.
    timestamp : str
        A path-safe stamp naming the run's output folder.
    item_id : str | None
        The SharePoint list-item id the request was read from (for logging).
    """

    request_id: str
    application_name: str
    model: str
    live_validation: bool
    timestamp: str
    item_id: str | None = None


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    """A SharePoint Yes/No value coerced to ``bool``.

    Graph surfaces list fields as untyped ``additional_data``: a Yes/No column
    may arrive as a Python ``bool``, an ``int`` (``0``/``1``), or a string
    (``"true"``/``"0"``/``"Yes"`` …). Unrecognised strings and ``None`` fall
    back to *default* with a WARNING rather than guessing.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):  # bool already handled above
        return value != 0
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_STRINGS:
            return True
        if token in _FALSE_STRINGS:
            return False
    logger.warning(
        f"_coerce_bool: unrecognised boolean {value!r}; using default {default}"
    )
    return default


def _sanitise_segment(value: str) -> str:
    """*value* reduced to a safe single path segment (for the output folder).

    Anything outside ``[A-Za-z0-9._-]`` becomes ``-`` and runs collapse, so a
    timestamp like ``2026-07-19T08:30:00Z`` or a stray separator can name a
    SharePoint folder without splitting the path or tripping Graph.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    return cleaned or "run"


def _generated_timestamp() -> str:
    """A path-safe UTC stamp (``YYYYMMDDTHHMMSSZ``) for when the list has none."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def select_request(
    rows: list[dict[str, Any]], request_id: str, config: PowerAppsConfig
) -> dict[str, Any]:
    """The single list row whose request-id field equals *request_id*.

    Matching is done here in Python rather than as a Graph ``$filter``: the
    ``SharePointClient.read_list_items`` filter path needs a non-indexed-column
    ``Prefer`` header the client does not set, and a Power Apps request list is
    small enough to scan.

    Raises
    ------
    PowerAppsError
        No row matches *request_id*, or more than one does.
    """
    field = config.request_id_field
    matches = [
        row
        for row in rows
        if str(row.get("fields", {}).get(field, "")).strip() == request_id.strip()
    ]
    if not matches:
        raise PowerAppsError(
            f"no Power Apps request with {field}={request_id!r} in the list "
            f"(scanned {len(rows)} item(s))"
        )
    if len(matches) > 1:
        raise PowerAppsError(
            f"{len(matches)} Power Apps requests have {field}={request_id!r}; "
            f"expected exactly one"
        )
    return matches[0]


def parse_run_request(
    row: dict[str, Any], config: PowerAppsConfig
) -> RunRequest:
    """Normalise a SharePoint list *row* into a :class:`RunRequest`.

    Pulls each column out of ``row["fields"]`` by its configured internal name,
    coerces the live-validation flag, checks the selected model is one this
    deployment can reach (:func:`app_config.is_accessible_model`), and resolves
    the output timestamp (the list value if present, else a generated one),
    sanitised for use as a folder name.

    Raises
    ------
    PowerAppsError
        ``application_name`` or ``request_id`` is blank, or the model is not
        accessible.
    """
    fields = row.get("fields") or {}

    request_id = str(fields.get(config.request_id_field, "")).strip()
    if not request_id:
        raise PowerAppsError(
            f"Power Apps request is missing {config.request_id_field!r}"
        )

    application_name = str(fields.get(config.application_name_field, "")).strip()
    if not application_name:
        raise PowerAppsError(
            f"Power Apps request {request_id!r} is missing "
            f"{config.application_name_field!r} (the input/output folder name)"
        )

    model = str(fields.get(config.model_field, "")).strip()
    if not model:
        raise PowerAppsError(
            f"Power Apps request {request_id!r} is missing {config.model_field!r}"
        )
    if not is_accessible_model(model):
        raise PowerAppsError(
            f"Power Apps request {request_id!r} selected model {model!r}, which "
            f"is not an accessible model for this deployment"
        )

    live_validation = _coerce_bool(fields.get(config.live_validation_field))

    raw_ts = str(fields.get(config.timestamp_field, "")).strip()
    timestamp = _sanitise_segment(raw_ts) if raw_ts else _generated_timestamp()

    return RunRequest(
        request_id=request_id,
        application_name=application_name,
        model=model,
        live_validation=live_validation,
        timestamp=timestamp,
        item_id=row.get("id"),
    )
