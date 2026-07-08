"""REPL conveniences for eyeballing chunker/batcher runs (imported by nothing)."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import TextIO


def print_iterable(
    items: Iterable[object],
    *,
    label: str | None = None,
    numbered: bool = True,
    stream: TextIO | None = None,
) -> None:
    """Print an iterable of objects one-per-line via each item's ``str()``.

    A convenience for eyeballing a chunker or batcher run at the REPL —
    e.g. the ``chunks`` / ``diagnostics`` of a
    :class:`~chunker.models.SasChunkResult`, the ``batches`` /
    ``singletons`` of a :class:`~chunker.models.SasBatchResult`, or the
    internal ``_Unit`` / ``_Region`` / ``_Edge`` lists while debugging.
    Every element is rendered with ``str()``; the models and the internal
    dataclasses all provide concise readable forms.

    Parameters
    ----------
    items
        Any iterable.  Materialised to a list once so its length can be
        reported (safe to pass a generator).
    label
        Header line printed before the items.  Defaults to ``"N item(s)"``.
    numbered
        Prefix each line with a right-aligned ``[i]`` index (default True).
    stream
        Destination text stream; defaults to ``sys.stdout``.
    """
    materialised = list(items)
    out = stream if stream is not None else sys.stdout
    header = label if label is not None else f"{len(materialised)} item(s)"
    print(f"{header}:", file=out)
    if not materialised:
        print("  <empty>", file=out)
        return
    width = len(str(len(materialised)))
    for i, item in enumerate(materialised, 1):
        prefix = f"  [{i:>{width}}] " if numbered else "  "
        print(f"{prefix}{item}", file=out)
