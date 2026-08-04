"""Writing converted scripts and validation artefacts back to SharePoint.

Two destinations, both from :mod:`conversion.paths`: a run's converted scripts
go to ``{converted}/{model}/{timestamp}``, and validation output to
``{converted}/validation``.

Notebook rendering is **delegated** to :mod:`pipeline.notebook`, which already
knows how to turn a translation into an ``.ipynb`` — which fences are runnable
code, what the kernelspec is for the target, how cell ids are stamped. Doing it
again here would be a second implementation to keep in step, so this module
only decides *whether* a notebook is what should be uploaded.

Logger name: ``conversion.upload``.
"""

from __future__ import annotations

import logging
from typing import Any

from app_config.sharepoint import SharePointConfig
from target_language import resolve_target_language

from .paths import upload_target, validation

logger = logging.getLogger(__name__)

# Extension per output type, for a file name that arrives without one (or with
# the source's). Types that render as a notebook take .ipynb.
FILE_EXTENSIONS: dict[str, str] = {
    "pyspark": "ipynb",
    "sparksql": "ipynb",
    "sparkscala": "ipynb",
    "python": "py",
    "py": "py",
    "sql": "sql",
    "scala": "scala",
    "txt": "txt",
}

# Output types delivered as a runnable notebook rather than a flat file: on
# Databricks that is what an operator actually runs.
NOTEBOOK_TYPES = frozenset({"pyspark", "sparksql", "sparkscala"})

DEFAULT_EXTENSION = "txt"


def set_file_extension(file_name: str, file_type: str) -> str:
    """
    *file_name* carrying the extension *file_type* implies.

    An existing extension is replaced, not appended: the name usually arrives
    as the SAS source's (``etl.sas``), and ``etl.sas.ipynb`` would be wrong in
    a way that only shows up in the library listing.
    """
    extension = FILE_EXTENSIONS.get(file_type.strip().lower(), DEFAULT_EXTENSION)
    stem = file_name.strip().rsplit(".", 1)[0] if "." in file_name.strip() else (
        file_name.strip()
    )
    return f"{stem or 'output'}.{extension}"


def _client(client: Any | None) -> Any:
    if client is not None:
        return client
    from app_config.sharepoint import get_sharepoint_client

    return get_sharepoint_client()


def _as_notebook(contents: str, file_type: str) -> str:
    """*contents* rendered as ``.ipynb`` JSON for the *file_type* target."""
    from pipeline.notebook import build_notebook, markdown_to_cells, notebook_to_json

    target = resolve_target_language(file_type)
    cells = markdown_to_cells(contents, output_language=target)
    return notebook_to_json(build_notebook(cells, output_language=target))


def upload_converted_script(
    application: str,
    file_name: str,
    file_type: str,
    file_contents: str,
    model: str,
    timestamp: str,
    *,
    client: Any | None = None,
    config: SharePointConfig | None = None,
) -> str:
    """
    Upload one converted script into ``{converted}/{model}/{timestamp}``,
    returning the drive-relative path it landed at.

    *file_type* names the output language and decides two things: the
    extension (:func:`set_file_extension`) and whether the content is wrapped
    into a notebook (:data:`NOTEBOOK_TYPES`). The folder is created first —
    Graph's simple upload does not create missing parents — and creating it is
    idempotent, so a second script in the same run is not a conflict.

    Raises
    ------
    SharePointError
        The folder could not be created, or the upload failed.
    """
    kind = file_type.strip().lower()
    folder = upload_target(application, model, timestamp, config=config)
    name = set_file_extension(file_name, kind)
    contents = _as_notebook(file_contents, kind) if kind in NOTEBOOK_TYPES else (
        file_contents
    )
    transport = _client(client)
    transport.create_folder(folder)
    transport.upload_file(folder, name, contents)
    path = f"{folder}/{name}"
    logger.info(f"upload_converted_script: wrote {path!r} ({kind})")
    return path


def upload_validation_file(
    application: str,
    file_name: str,
    file_contents: str | bytes,
    *,
    client: Any | None = None,
    config: SharePointConfig | None = None,
) -> str:
    """
    Upload one validation artefact into ``{converted}/validation``, returning
    the path it landed at.

    Takes ``bytes`` as well as ``str`` because the validation report is also
    written as a PDF.

    Raises
    ------
    SharePointError
        The folder could not be created, or the upload failed.
    """
    folder = validation(application, config=config)
    transport = _client(client)
    transport.create_folder(folder)
    transport.upload_file(folder, file_name, file_contents)
    path = f"{folder}/{file_name}"
    logger.info(f"upload_validation_file: wrote {path!r}")
    return path
