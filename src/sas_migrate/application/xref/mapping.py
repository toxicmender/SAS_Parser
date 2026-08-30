"""Pure XREF classification and longest-prefix path resolution."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from .models import XrefMappings, XrefRow

logger = logging.getLogger(__name__)

PATH_MARKERS = frozenset({"path", "physical_path", "file"})
TABLE_MARKERS = frozenset({"table", "dataset", "ds"})


def ordered_path_keys(by_path: dict[str, str]) -> tuple[str, ...]:
    """Return stable mapping keys with the most specific path first."""

    return tuple(sorted(by_path, key=lambda key: (-len(key), key)))


def resolve_path(
    value: str,
    by_path: dict[str, str],
    keys: tuple[str, ...] | None = None,
) -> str | None:
    """Resolve an exact or longest directory-prefix path mapping."""

    stripped = value.strip()
    folded = stripped.casefold()
    if not folded:
        return None
    exact = by_path.get(folded)
    if exact is not None:
        return exact
    for key in keys or ordered_path_keys(by_path):
        prefix = key.rstrip("/")
        if folded.startswith(prefix + "/"):
            return f"{by_path[key].rstrip('/')}{stripped[len(prefix):]}"
    return None


def classify_rows(rows: Iterable[XrefRow]) -> XrefMappings:
    """Classify source-neutral rows without invoking any source adapter."""

    exact: dict[str, str] = {}
    by_libref: dict[str, str] = {}
    by_path: dict[str, str] = {}
    for row in rows:
        marker = row.marker.casefold()
        source = row.source.strip()
        if marker in PATH_MARKERS:
            by_path[source] = row.target
            continue
        if marker and marker not in TABLE_MARKERS:
            logger.warning(
                "XREF row %r has unrecognised marker %r; treating it as a table mapping",
                source,
                marker,
            )
        target = exact if "." in source else by_libref
        target[source] = row.target
    return XrefMappings(exact=exact, by_libref=by_libref, by_path=by_path)


__all__ = [
    "PATH_MARKERS",
    "TABLE_MARKERS",
    "classify_rows",
    "ordered_path_keys",
    "resolve_path",
]
