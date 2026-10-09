"""Where the scripts a corpus ``%INCLUDE``s are: beside it, in SharePoint, or nowhere.

``%include "/sas/prod/macros/util.sas";`` names another script by where it lived
on the SAS server — a path that means nothing here. A migration needs that
script all the same, so it is looked for by **file name**: ``util.sas`` in the
local corpus directory and in the application's SharePoint scripts folder, at
any depth, ignoring case (a SAS server on Windows would). Found in neither, it
is a dependency the corpus is missing.

The input is the reference inventory (:mod:`data_hydration.inventory`): its
``%INCLUDE`` rows carry the place SAS reads (``value``, macro variables
expanded and filerefs followed) and the spelling (``raw``).
:func:`match_includes` returns the rows with ``found_local`` /
``found_sharepoint`` filled in, so the inventory table records the answer
beside the reference; :func:`include_checks` folds them into one
:class:`IncludeCheck` per script for a report.

The local listing is a directory walk and the SharePoint one a Graph folder
walk through :mod:`app_config.sharepoint`, imported when it is used — nothing
here reaches the network unless asked to.

Logger name: ``data_hydration.includes``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .inventory import InventoryRow, RefKind

logger = logging.getLogger(__name__)

#: A file name, lowercased, and every place a script by that name was found.
FileIndex = Mapping[str, tuple[str, ...]]

#: How many folders a SharePoint walk visits before it stops and says so.
MAX_SHAREPOINT_FOLDERS = 1_000

# A fileref member as an %INCLUDE spells it: src(util), src('util.sas').
_MEMBER_RE = re.compile(r"[\w&]+\s*\(\s*(.+?)\s*\)")


def is_include(row: InventoryRow) -> bool:
    """Whether *row* is a script an ``%INCLUDE`` pulls in."""
    return row.kind is RefKind.PATH and row.statement == "include"


def include_file_name(row: InventoryRow) -> str | None:
    """The file name *row*'s ``%INCLUDE`` opens: ``util.sas`` for
    ``/sas/macros/util.sas`` or ``C:\\sas\\util.sas``, and for a member of a
    fileref's directory, ``src(util)``, the member with the ``.sas`` SAS adds.

    ``None`` when no file name is known: the last segment is still a macro
    reference (``%include &f;``), or a fileref no FILENAME binds names a whole
    file (``%include setup;``) whose name is the FILENAME's to say.
    """
    text = row.value.strip().strip("'\"")
    member = _MEMBER_RE.fullmatch(text)
    if member:
        name = member.group(1).strip("'\"")
        if "." not in name:
            name += ".sas"
    elif row.libref and text.lower() == row.libref:
        return None  # an unbound fileref: no file name yet
    else:
        name = re.split(r"[\\/]", text)[-1]
    if not name or "&" in name:
        return None
    return name


def local_index(root: Path) -> dict[str, tuple[str, ...]]:
    """Every file under *root*, by lowercased name, as paths relative to it."""
    found: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            found.setdefault(path.name.lower(), []).append(path.relative_to(root).as_posix())
    logger.info(f"local_index: {sum(map(len, found.values()))} file(s) under {root}")
    return {name: tuple(paths) for name, paths in found.items()}


def sharepoint_index(folder: str, *, client: Any = None) -> dict[str, tuple[str, ...]]:
    """Every file under the document-library *folder*, at any depth, by
    lowercased name, as drive-relative paths.

    *client* is a :class:`app_config.sharepoint.SharePointClient`, the shared
    one when ``None``. A walk stops after :data:`MAX_SHAREPOINT_FOLDERS`
    folders, with a warning, rather than list a whole library by mistake.

    Raises
    ------
    app_config.sharepoint.SharePointError
        *folder* is absent, or a listing fails.
    """
    from app_config.sharepoint import resolve_client

    sp = resolve_client(client)
    found: dict[str, list[str]] = {}
    pending = [folder.strip().strip("/")]
    visited = 0
    while pending:
        if visited == MAX_SHAREPOINT_FOLDERS:
            logger.warning(
                f"sharepoint_index: stopped after {visited} folder(s) under "
                f"{folder!r}; {len(pending)} left unlisted"
            )
            break
        current = pending.pop()
        visited += 1
        for entry in sp.list_directory(current):
            name = entry.get("name") or ""
            child = f"{current}/{name}" if current else name
            if entry.get("is_folder"):
                pending.append(child)
            elif name:
                found.setdefault(name.lower(), []).append(child)
    logger.info(
        f"sharepoint_index: {sum(map(len, found.values()))} file(s) in "
        f"{visited} folder(s) under {folder!r}"
    )
    return {name: tuple(sorted(paths)) for name, paths in found.items()}


def match_includes(
    rows: Iterable[InventoryRow],
    *,
    local: FileIndex | None = None,
    sharepoint: FileIndex | None = None,
) -> list[InventoryRow]:
    """*rows*, each ``%INCLUDE`` with the scripts of its file name found in
    *local* and in *sharepoint* (an index ``None`` is not looked in, and its
    column stays ``None``). Every other row is returned as it is.

    A script with no known file name (:func:`include_file_name`) is found
    nowhere: its columns are empty, not ``None``, since the places were
    searched and the name was what was missing.
    """
    matched: list[InventoryRow] = []
    for row in rows:
        if not is_include(row):
            matched.append(row)
            continue
        name = include_file_name(row)
        key = name.lower() if name else None
        update: dict[str, tuple[str, ...]] = {}
        for column, index in (("found_local", local), ("found_sharepoint", sharepoint)):
            if index is not None:
                update[column] = tuple(index.get(key, ())) if key else ()
        matched.append(row.model_copy(update=update) if update else row)
    return matched


@dataclass(frozen=True)
class IncludeCheck:
    """One script the corpus includes, and where a copy of it was found.

    Attributes
    ----------
    file_name
        The script's file name, as the ``%INCLUDE`` spells it; ``None`` when
        none is known (see :func:`include_file_name`), and then :attr:`spelled`
        is what the statement wrote.
    spelled
        The include as written, the first time it was met.
    included_by
        ``(source_id, line)`` of every ``%INCLUDE`` of it, in corpus order.
    local, sharepoint
        The copies found locally and in SharePoint; ``None`` where nobody
        looked.
    """

    file_name: str | None
    spelled: str
    included_by: tuple[tuple[str, int], ...]
    local: tuple[str, ...] | None
    sharepoint: tuple[str, ...] | None

    @property
    def found(self) -> bool:
        """A copy was found somewhere."""
        return bool(self.local) or bool(self.sharepoint)

    @property
    def status(self) -> str:
        """``unchecked`` (nowhere was looked), ``unnamed`` (no file name to
        look for), ``found`` or ``missing``."""
        if self.local is None and self.sharepoint is None:
            return "unchecked"
        if self.file_name is None:
            return "unnamed"
        return "found" if self.found else "missing"


def include_checks(rows: Iterable[InventoryRow]) -> list[IncludeCheck]:
    """One :class:`IncludeCheck` per script *rows* include, in the order the
    corpus first includes it. Scripts are one by file name, ignoring case;
    an include with no file name is one by its spelling."""
    found: dict[str, IncludeCheck] = {}
    for row in sorted(rows, key=lambda r: (r.file_order, r.ref_order)):
        if not is_include(row):
            continue
        name = include_file_name(row)
        key = name.lower() if name else f"&{row.raw}"
        site = (row.source_id, row.start_line)
        seen = found.get(key)
        if seen is None:
            found[key] = IncludeCheck(
                file_name=name,
                spelled=row.raw,
                included_by=(site,),
                local=row.found_local,
                sharepoint=row.found_sharepoint,
            )
        elif site not in seen.included_by:
            found[key] = IncludeCheck(
                file_name=seen.file_name,
                spelled=seen.spelled,
                included_by=(*seen.included_by, site),
                local=seen.local,
                sharepoint=seen.sharepoint,
            )
    return list(found.values())
