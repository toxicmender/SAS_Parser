"""Finding and reading an application's source scripts in SharePoint.

Sourcing is separate from conversion on purpose: the chunker takes text, not
paths (:meth:`chunker.chunker.SasSemanticChunker.chunk_text` with an explicit
``source_id``), so a SharePoint-hosted corpus needs no temporary files at all.
What this module produces is exactly what that method wants — a drive-relative
path to name the source, and its text.

Logger name: ``conversion.sources``.
"""

from __future__ import annotations

import logging
from typing import Any

from app_config.sharepoint import SharePointConfig, SharePointError

from .paths import original_scripts

logger = logging.getLogger(__name__)

# Which file extensions count as source, per input language. The reference
# accepts .txt beside .sas because scripts arrive from systems that will not
# hand over a .sas file.
FILE_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "sas": ("sas", "txt"),
}


def applicable_extensions(file_type: str) -> tuple[str, ...]:
    """
    The extensions that count as source for *file_type* (case-insensitive).

    Raises
    ------
    SharePointError
        The type is not one this flow knows how to source. Raising beats
        defaulting: silently scanning for ``.sas`` when the request asked for
        something else would convert the wrong files.
    """
    try:
        return FILE_EXTENSIONS[file_type.strip().lower()]
    except KeyError:
        raise SharePointError(
            f"no source extensions known for input type {file_type!r}; "
            f"expected one of {', '.join(sorted(FILE_EXTENSIONS))}"
        ) from None


def _client(client: Any | None) -> Any:
    if client is not None:
        return client
    from app_config.sharepoint import get_sharepoint_client

    return get_sharepoint_client()


def source_files(
    application: str,
    file_type: str = "sas",
    *,
    client: Any | None = None,
    config: SharePointConfig | None = None,
) -> list[str]:
    """
    The drive-relative paths of *application*'s source scripts, sorted.

    Sorted because cross-file batching resolves references across the corpus:
    a stable order makes a re-run reproducible.

    Raises
    ------
    SharePointError
        *file_type* is unknown, or the folder cannot be listed.
    """
    folder = original_scripts(application, config=config)
    entries = _client(client).list_files(folder, applicable_extensions(file_type))
    paths = sorted(entry["path"] for entry in entries)
    logger.info(
        f"source_files: {len(paths)} {file_type} file(s) under {folder!r}"
    )
    return paths


def load(
    path: str, *, client: Any | None = None
) -> str:
    """The text of one source file.

    Raises
    ------
    SharePointError
        The file is absent or could not be downloaded.
    """
    return _client(client).download_file_as_text(path)
