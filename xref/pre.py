"""The half of the pre-conversion substitution that dataset renaming cannot reach.

:func:`chunker.batcher.replace_dataset_names` owns *dataset names* — it rewrites
the chunk metadata's dataset lists and the ``%let`` values that carry one. That
covers every reference the SAS parser recognised as a dataset, and deliberately
covers nothing else: :func:`chunker.batcher._map_ds` early-returns on quoted
physical paths and on names still carrying a ``&``, because neither is a library
member the dataset vocabulary can address.

A whole family of constructs therefore goes untouched by it, and they are exactly
where a *physical path* appears — ``LIBNAME``, ``FILENAME``, ``INFILE`` / ``FILE``,
``%INCLUDE``, PROC IMPORT/EXPORT's ``datafile=`` / ``outfile=``, ODS destinations,
and ``options sasautos=``.

This module rewrites those, keyed off :attr:`~xref.sourcing.XrefMappings.by_path`
— the slot XREF rows carrying a ``path`` marker in ``Title`` populate. It runs
over the **raw source text, before chunking**, which is both the same position
the reference system applies its substitution at and the position that keeps
``chunker`` and ``pipeline`` free of any XREF knowledge: by the time the chunker
sees the text, the paths are already the target's.

Where the grammar lives
-----------------------
Not here. :mod:`chunker.paths` owns the definition of where a path appears in SAS
syntax, because the chunker also *records* these paths as chunk metadata and two
modules maintaining that grammar separately is the drift Architecture.md invariant
12 exists to prevent. This module imports :data:`~chunker.paths.PATH_STATEMENTS`
and supplies only the substitution. The import is lazy, matching
:mod:`xref.apply`, so ``xref`` stays usable without pulling the chunker in.

Only filesystem locations are rewritten. ``FILENAME`` shares its syntax with
device forms — ``filename x pipe 'ls -l'``, ``filename m email 'to@host'`` — whose
quoted argument is a command or an address, not somewhere an XREF path mapping
could sensibly point.

What this is not
----------------
The reference substitutes with one case-insensitive regex sweep over the whole
script, sorted longest-key-first. That reaches everything, including things that
are not paths at all — a table name inside a comment, a string that merely looks
like one. This module matches **by statement** instead, so a rewrite only lands
where SAS syntax says a path belongs. The one property of the reference's
approach worth keeping is the ordering: a path is routinely a prefix of a longer
path, so keys are tried longest-first and the most specific mapping wins. That
resolution lives in :mod:`xref.mapping`, shared with the post pass so the two
halves can never rewrite one path to two different targets.

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
from functools import partial
from typing import TYPE_CHECKING

from .mapping import map_path, ordered_keys

if TYPE_CHECKING:  # avoid a runtime import cycle with sourcing
    from chunker.paths import PathSpec

    from .sourcing import XrefMappings

logger = logging.getLogger(__name__)



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

    # Lazy, like xref.apply's: this package stays importable without the chunker,
    # and the cost lands only on a run that actually has path mappings to apply.
    from chunker.models import PathLocation
    from chunker.paths import PATH_STATEMENTS

    keys = ordered_keys(by_path)
    stats = PreStats()
    label = source_id or "<inline>"

    def _substitute(match: re.Match[str], *, spec: "PathSpec") -> str:
        # A device form's quoted argument is a command line or an address, not a
        # location a path mapping can address. See the module docstring;
        # classifying it is chunker.paths' call, not ours.
        if spec.location_for(match) is not PathLocation.FILESYSTEM:
            return match.group(0)
        raw = match.group("path")
        if "&" in raw:
            stats.unresolved.append(raw)
            return match.group(0)
        mapped = map_path(raw, by_path, keys)
        if mapped is None:
            return match.group(0)
        stats.rewritten[raw] = mapped
        quote = match.group("q")
        return f"{match.group('head')}{quote}{mapped}{quote}"

    rewritten = text
    for spec in PATH_STATEMENTS:
        before = rewritten
        rewritten = spec.pattern.sub(partial(_substitute, spec=spec), rewritten)
        if before != rewritten and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"rewrite_source_text: {label}: {spec.statement} path(s) remapped"
            )

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
