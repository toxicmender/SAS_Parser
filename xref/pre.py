"""The half of the pre-conversion substitution that dataset renaming cannot reach.

:func:`chunker.batcher.replace_dataset_names` owns *dataset names* — it rewrites
the chunk metadata's dataset lists and the ``%let`` values that carry one. That
covers every reference the SAS parser recognised as a dataset, and deliberately
covers nothing else: :func:`chunker.batcher._map_ds` early-returns on quoted
physical paths and on names still carrying a ``&``, because neither is a library
member the dataset vocabulary can address.

Three constructs therefore go untouched by it, and they are exactly where a
*physical path* appears:

``LIBNAME``
    ``libname raw '/data/in';`` — the library's location, not a dataset.
``INFILE`` / ``FILE``
    ``infile '/data/in/customers.csv';`` — a flat file being read or written.
``%INCLUDE``
    ``%include '/code/macros/common.sas';`` — source pulled in from elsewhere.

This module rewrites those, keyed off :attr:`~xref.sourcing.XrefMappings.by_path`
— the slot XREF rows carrying a ``path`` marker in ``Title`` populate. It runs
over the **raw source text, before chunking**, which is both the same position
the reference system applies its substitution at and the position that keeps
``chunker`` and ``pipeline`` free of any XREF knowledge: by the time the chunker
sees the text, the paths are already the target's.

What this is not
----------------
The reference substitutes with one case-insensitive regex sweep over the whole
script, sorted longest-key-first. That reaches everything, including things that
are not paths at all — a table name inside a comment, a string that merely looks
like one. This module matches **by statement** instead, so a rewrite only lands
where SAS syntax says a path belongs. The one property of the reference's
approach worth keeping is the ordering: a path is routinely a prefix of a longer
path, so keys are tried longest-first (:func:`_ordered_keys`) and the most
specific mapping wins.

Unresolved macro references (``libname raw "&root/in";``) are **not**
substituted — the value is not knowable without running SAS — but they are
counted and reported, so a path that silently kept its SAS-side value is visible
in the log rather than only in the output.

Logger name: ``xref.pre``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle with sourcing
    from .sourcing import XrefMappings

logger = logging.getLogger(__name__)

# Statements whose quoted argument is a physical path. Each pattern captures the
# quote in group "q" and the path in group "path", so one substitution helper
# serves all of them. Case-insensitive and DOTALL-free on purpose: a SAS
# statement's quoted literal does not span lines.
_PATH_STATEMENTS: dict[str, re.Pattern[str]] = {
    # libname <libref> [<engine>] '<path>'
    "LIBNAME": re.compile(
        r"(?P<head>\blibname\s+[A-Za-z_]\w*\s+(?:[A-Za-z_]\w*\s+)?)"
        r"(?P<q>['\"])(?P<path>[^'\"\n]*)(?P=q)",
        re.IGNORECASE,
    ),
    # infile '<path>' / file '<path>'
    "INFILE": re.compile(
        r"(?P<head>\b(?:infile|file)\s+)"
        r"(?P<q>['\"])(?P<path>[^'\"\n]*)(?P=q)",
        re.IGNORECASE,
    ),
    # %include '<path>'
    "%INCLUDE": re.compile(
        r"(?P<head>%include\s+)"
        r"(?P<q>['\"])(?P<path>[^'\"\n]*)(?P=q)",
        re.IGNORECASE,
    ),
}


@dataclass
class PreStats:
    """What the pass did, and what it deliberately did not do.

    Attributes
    ----------
    rewritten : dict[str, str]
        The substitutions that landed, ``old -> new``.
    unresolved : list[str]
        Path values carrying a ``&`` macro reference. Left exactly as written —
        their real value is not knowable without running SAS — but recorded,
        because a path that kept its SAS-side value is otherwise invisible.
    """

    rewritten: dict[str, str] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.rewritten)

    def __bool__(self) -> bool:
        return bool(self.rewritten or self.unresolved)


def _ordered_keys(by_path: dict[str, str]) -> list[str]:
    """*by_path*'s keys, longest first.

    A path is routinely a prefix of a longer path (``/data`` and
    ``/data/inbound``), so trying the longest first is what makes the most
    specific mapping win. Ties break alphabetically so the order is stable and
    a run is reproducible.
    """
    return sorted(by_path, key=lambda k: (-len(k), k))


def _map_path(value: str, by_path: dict[str, str], keys: list[str]) -> str | None:
    """*value* remapped, or ``None`` when no mapping addresses it.

    An exact match wins; failing that the longest key that *value* sits under
    is treated as a directory prefix and replaced, so mapping one folder moves
    everything beneath it. Comparison is case-insensitive because
    :func:`xref.sourcing.classify` lowercases its keys, matching the repo-wide
    "names lowercased at extraction" convention.
    """
    folded = value.strip().lower()
    if not folded:
        return None
    exact = by_path.get(folded)
    if exact is not None:
        return exact
    for key in keys:
        # Directory prefix: the key must end at a path separator in `value`,
        # so `/data/in` never matches `/data/inbound`.
        if folded.startswith(key.rstrip("/") + "/"):
            tail = value.strip()[len(key.rstrip("/")) :]
            return f"{by_path[key].rstrip('/')}{tail}"
    return None


def rewrite_source_text(
    text: str,
    mappings: "XrefMappings",
    *,
    source_id: str | None = None,
) -> tuple[str, PreStats]:
    """
    *text* with its ``LIBNAME`` / ``INFILE`` / ``%INCLUDE`` paths remapped,
    plus a :class:`PreStats` describing what happened.

    Returns *text* unchanged (and empty stats) when there are no ``by_path``
    mappings, which is the common case — the slot is only populated by XREF
    rows explicitly marked as paths.

    Parameters
    ----------
    text : str
        Raw SAS source, before chunking.
    mappings : XrefMappings
        Only :attr:`~xref.sourcing.XrefMappings.by_path` is read here; the
        dataset slots are applied by
        :func:`chunker.batcher.replace_dataset_names` through the batchers.
    source_id : str | None
        Names the file in the log lines.
    """
    by_path = mappings.by_path
    if not by_path or not text:
        return text, PreStats()

    keys = _ordered_keys(by_path)
    stats = PreStats()
    label = source_id or "<inline>"

    def _substitute(match: re.Match[str]) -> str:
        raw = match.group("path")
        if "&" in raw:
            stats.unresolved.append(raw)
            return match.group(0)
        mapped = _map_path(raw, by_path, keys)
        if mapped is None:
            return match.group(0)
        stats.rewritten[raw] = mapped
        quote = match.group("q")
        return f"{match.group('head')}{quote}{mapped}{quote}"

    rewritten = text
    for statement, pattern in _PATH_STATEMENTS.items():
        before = rewritten
        rewritten = pattern.sub(_substitute, rewritten)
        if before != rewritten and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"rewrite_source_text: {label}: {statement} path(s) remapped")

    if stats.rewritten:
        logger.info(
            f"rewrite_source_text: {label}: remapped {len(stats.rewritten)} "
            f"physical path(s)"
        )
    if stats.unresolved:
        logger.warning(
            f"rewrite_source_text: {label}: {len(stats.unresolved)} path(s) carry "
            f"an unresolved macro reference and were left as written "
            f"({', '.join(sorted(set(stats.unresolved))[:3])}"
            f"{', ...' if len(set(stats.unresolved)) > 3 else ''}); "
            f"an XREF path mapping cannot address them"
        )
    return rewritten, stats
