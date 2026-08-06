"""Reading XREF mappings for one application out of the SharePoint list.

Row shape (SharePoint internal column names, :data:`XREF_FIELDS`)::

    Title         -> bsnId            an optional TYPE MARKER; see below
    Application   -> application_name which application the row belongs to
    OriginalValue -> sourceTable      the SAS-side name
    NewValue      -> destinationTable the Databricks target

``Title`` as a discriminator
----------------------------
The column is free to carry a type marker, so :class:`XrefMappings` has three
slots from the outset:

``exact``
    ``schema.table -> catalog.schema.table``. What every current row is.
``by_libref``
    ``libref -> catalog.schema``, for a whole library at once.
``by_path``
    A physical path to remap (``'/data/in.csv' -> ...``). **Populated but not
    yet consumed** — see below.

A row whose ``Title`` is absent, empty, or unrecognised is a *table* mapping,
which is what makes the marker backward-compatible: every existing row keeps
working and no backfill is needed. A recognised path marker
(:data:`PATH_MARKERS`, case-insensitively) routes the row to ``by_path``; an
*unrecognised* marker is warned about, so a mistyped row is visible now rather
than silently becoming a table mapping later.

``exact`` and ``by_libref`` are exactly the two key shapes
:func:`chunker.batcher._split_databricks_mapping` already classifies — a
dotted key is an exact dataset name, a bare one a libref prefix — so
:func:`xref.apply.apply_pre` hands them straight to
:func:`chunker.batcher.replace_dataset_names` and that module needs no change.
Enabling ``by_path`` later needs only the quoted-literal guard in
``chunker.batcher._map_ds`` lifted behind an argument, plus the rewriter
extended to ``LIBNAME`` / ``INFILE`` / ``%include`` targets: no config, list
schema, or transport change.

Logger name: ``xref.sourcing``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app_config.sharepoint import SharePointConfig

logger = logging.getLogger(__name__)

# attribute -> SharePoint internal column name.
XREF_FIELDS: dict[str, str] = {
    "bsnId": "Title",
    "application_name": "Application",
    "sourceTable": "OriginalValue",
    "destinationTable": "NewValue",
}

# Title values marking a row as a physical-path mapping rather than a table
# one. Matched case-insensitively after stripping.
PATH_MARKERS = frozenset({"path", "physical_path", "file"})

# Title values that explicitly mark the default (a table mapping), accepted so
# an operator can be explicit without tripping the unrecognised-marker warning.
TABLE_MARKERS = frozenset({"table", "dataset", "ds"})


@dataclass
class XrefMappings:
    """The mappings for one application, split by what they address.

    Attributes
    ----------
    exact : dict[str, str]
        ``schema.table -> catalog.schema.table``.
    by_libref : dict[str, str]
        ``libref -> catalog.schema``.
    by_path : dict[str, str]
        Physical path remappings. Populated when marked rows appear;
        **nothing consumes it yet** — see the module docstring.
    """

    exact: dict[str, str] = field(default_factory=dict)
    by_libref: dict[str, str] = field(default_factory=dict)
    by_path: dict[str, str] = field(default_factory=dict)

    @property
    def dataset_mapping(self) -> dict[str, str]:
        """``exact`` and ``by_libref`` as the single flat dict
        :func:`chunker.batcher.replace_dataset_names` takes. The two key
        shapes are distinguishable by dot count, which is exactly how that
        function tells them apart, so no information is lost by flattening."""
        return {**self.by_libref, **self.exact}

    def __bool__(self) -> bool:
        return bool(self.exact or self.by_libref or self.by_path)

    def __len__(self) -> int:
        return len(self.exact) + len(self.by_libref) + len(self.by_path)


def _format_xref_item(item: dict[str, Any]) -> dict[str, Any]:
    """One raw list item projected onto the :data:`XREF_FIELDS` names."""
    fields = item.get("fields") or {}
    return {
        attribute: fields.get(column)
        for attribute, column in XREF_FIELDS.items()
    }


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _client(client: Any | None) -> Any:
    if client is not None:
        return client
    from app_config.sharepoint import get_sharepoint_client

    return get_sharepoint_client()


def _config(config: SharePointConfig | None) -> SharePointConfig:
    return config if config is not None else SharePointConfig.from_env()


def classify(rows: list[dict[str, Any]]) -> XrefMappings:
    """
    Projected XREF rows sorted into the three slots of :class:`XrefMappings`.

    Pure: takes already-read rows, so a second backend (the reference's
    file-based ``sftp_config.xref_file_path``) can reuse the classification by
    projecting onto the same four names.

    Rows with either side blank are skipped — a half-filled row expresses no
    mapping — as is a row whose ``Title`` carries an unrecognised marker,
    which is warned about rather than guessed at.
    """
    result = XrefMappings()
    for row in rows:
        source = _text(row.get("sourceTable"))
        target = _text(row.get("destinationTable"))
        if not source or not target:
            logger.debug(
                f"classify: skipping half-filled row "
                f"{source!r} -> {target!r}"
            )
            continue
        marker = _text(row.get("bsnId")).lower()
        if marker in PATH_MARKERS:
            result.by_path[source] = target
            continue
        if marker and marker not in TABLE_MARKERS:
            logger.warning(
                f"classify: XREF row {source!r} has an unrecognised Title "
                f"marker {marker!r}; treating it as a table mapping "
                f"(recognised: {', '.join(sorted(PATH_MARKERS | TABLE_MARKERS))})"
            )
        key = source.lower()
        if "." in key:
            result.exact[key] = target
        else:
            result.by_libref[key] = target
    return result


def mappings(
    application_name: str,
    *,
    client: Any | None = None,
    config: SharePointConfig | None = None,
) -> XrefMappings:
    """
    The XREF mappings configured for *application_name*.

    The list holds every application's rows, so it is read whole and filtered
    on ``Application`` here — the same thing the reference does, and a
    ``$filter`` would need a ``Prefer`` header the transport does not set for a
    non-indexed column.

    The signature stays source-agnostic on purpose: a file-based XREF backend
    (the reference's ``sftp_config.xref_file_path``) should slot in behind this
    name rather than beside it.

    Raises
    ------
    app_config.sharepoint.SharePointError
        The XREF list is not configured, or the read failed.
    """
    resolved = _config(config)
    rows = _client(client).list_items(resolved.list_id("xref"))
    wanted = application_name.strip().casefold()
    projected = [
        row
        for row in (_format_xref_item(item) for item in rows)
        if _text(row.get("application_name")).casefold() == wanted
    ]
    result = classify(projected)
    logger.info(
        f"mappings: {application_name!r} has {len(result.exact)} exact, "
        f"{len(result.by_libref)} libref and {len(result.by_path)} path "
        f"mapping(s) of {len(rows)} row(s) in the list"
    )
    return result


def load_databricks_mapping_sharepoint(path: str) -> dict[str, str]:
    """
    Read the SAS→Databricks mapping CSV at *path* in the configured
    SharePoint document library and parse it with
    :func:`chunker.batcher.parse_databricks_mapping_csv`.

    The **file** backend, beside :func:`mappings`' list backend: some
    deployments keep the cross-reference as a two-column CSV in the library
    rather than as list rows. Both produce the flat dict
    :func:`chunker.batcher.replace_dataset_names` takes, so a caller merges or
    chooses between them in one line.

    It lives here rather than in ``chunker.batcher`` (where it used to) because
    reading it is I/O against SharePoint, and ``chunker`` stays network-free —
    the parser it delegates to is pure and correctly stays there.

    ``utf-8-sig`` decoding strips the BOM Excel stamps on exported CSVs.
    An unreadable file (``SharePointError``) propagates, and a file that
    parses to zero entries raises ``ValueError`` — both mean the operator
    asked for renaming that cannot happen, which should stop the run rather
    than silently produce SAS-named output.
    """
    from chunker.batcher import parse_databricks_mapping_csv

    logger.info(
        f"load_databricks_mapping_sharepoint: reading mapping CSV from "
        f"SharePoint '{path}'"
    )
    body = _client(None).read_file(path)
    mapping = parse_databricks_mapping_csv(body.decode("utf-8-sig"))
    if not mapping:
        raise ValueError(
            f"SharePoint Databricks mapping '{path}' parsed to zero entries; "
            f"expected CSV rows of <sas_libref_or_dataset>,<databricks_target>"
        )
    logger.info(
        f"load_databricks_mapping_sharepoint: loaded {len(mapping)} mapping entries"
    )
    return mapping
