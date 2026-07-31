"""How a source file is *named* in a report, as opposed to identified.

A ``source_id`` is a path — usually an absolute one, because the CLI hands the
chunker ``str(path)`` for whatever directory it was pointed at. That is the
right identity (it is unique, and it says where the file actually is) and the
wrong label: a table of twenty rows of
``D:\\corp\\migration\\sas\\etl\\load_customers.sas`` is a column of shared
prefix with the informative part pushed off the edge. Reports print the name;
the path stays in the model.

Trimming to the basename alone would be lossy, and this package is precise
about exactly this kind of collision elsewhere (see
:func:`complexity.report.source_stems`, which faces the same problem for output
*filenames*): two ``load.sas`` scripts in different directories are two
different files with two different scores, and printing both as ``load.sas``
would make the report a puzzle. So :func:`display_names` shortens each path to
its **last segment, then as many parent segments as it takes to stay unique**
within the corpus it is rendered with — ``etl/load.sas`` and ``adhoc/load.sas``
when both exist, plain ``load.sas`` when only one does.

Names are for display only. Nothing keys off them: every lookup, mapping, and
model field in this package still uses the full ``source_id``.

Logger name: ``complexity.naming``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping

logger = logging.getLogger(__name__)

#: Both separators, always: a corpus can be analysed on one platform and its
#: report read on another, and a Windows ``source_id`` must not render as one
#: long unsplit segment because the reader is on POSIX.
_SEPARATORS = re.compile(r"[\\/]+")


def _segments(source_id: str) -> list[str]:
    """*source_id* split on either separator, with empty segments dropped."""
    return [part for part in _SEPARATORS.split(source_id.strip()) if part]


def display_name(source_id: str) -> str:
    """*source_id*'s last path segment — its file name.

    The context-free form, for a caller with no corpus to check uniqueness
    against. Prefer :func:`display_names` when the whole set is in hand: it is
    the one that can tell two same-named files apart.
    """
    segments = _segments(source_id)
    return segments[-1] if segments else source_id


def display_names(source_ids: Iterable[str]) -> dict[str, str]:
    """``source_id -> the shortest trailing path that is unique in this corpus``.

    Every name is at least the file name, and grows a parent directory at a
    time only for the ids that would otherwise be ambiguous — so one collision
    between two files does not lengthen the label of every other file in the
    report.

    Ids that stay ambiguous even at full length (two spellings of one path, say)
    keep their full ``source_id``: an honest long label beats a short one that
    claims two files are the same.
    """
    ids = list(dict.fromkeys(source_ids))
    parts = {source_id: _segments(source_id) for source_id in ids}
    # Every id starts at its file name and only the ambiguous ones grow.
    depth = {source_id: 1 for source_id in ids}
    longest = max((len(p) for p in parts.values()), default=1)

    def label(source_id: str) -> str:
        segments = parts[source_id]
        if not segments:
            return source_id
        return "/".join(segments[-depth[source_id] :])

    for _ in range(longest):
        clashes: dict[str, list[str]] = {}
        for source_id in ids:
            clashes.setdefault(label(source_id), []).append(source_id)
        # A group of one is already unambiguous; a group whose members have no
        # parent left to add cannot be separated by widening it further.
        widened = False
        for group in clashes.values():
            if len(group) < 2:
                continue
            for source_id in group:
                if depth[source_id] < len(parts[source_id]):
                    depth[source_id] += 1
                    widened = True
        if not widened:
            break

    names = {source_id: label(source_id) for source_id in ids}
    taken: dict[str, list[str]] = {}
    for source_id, name in names.items():
        taken.setdefault(name, []).append(source_id)
    for name, group in taken.items():
        if len(group) > 1:
            logger.warning(
                f"display_names: {len(group)} source ids still share the name "
                f"{name!r} at full path length; printing their full ids instead"
            )
            for source_id in group:
                names[source_id] = source_id
    return names


def resolve_name(source_id: str, names: Mapping[str, str] | None) -> str:
    """*source_id*'s name from *names*, falling back to its file name.

    What the renderers call: they take the corpus-wide mapping as an optional
    argument, and a file the mapping does not cover — a dependency on a peer
    outside the analysed set, say — still prints as a name rather than a path.
    """
    if names:
        found = names.get(source_id)
        if found:
            return found
    return display_name(source_id)
