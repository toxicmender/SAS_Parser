"""Which XREF path mapping wins, and what it rewrites a value to.

The resolver both halves of the path substitution share. :mod:`xref.pre` applies
it to SAS source before conversion; :mod:`xref.rewrite` applies it to generated
PySpark and Spark SQL after. They must agree: the same ``by_path`` mapping
reaching the same location by two routes has to produce one target, or a run in
``"both"`` mode rewrites a path twice to two different places and the second
answer wins for no reason anybody can see.

That is the same reason ``chunker/paths.py`` owns the SAS path *grammar* — one
definition, several consumers (Architecture.md invariant 12). This module owns
the *resolution*: exact match first, then longest directory prefix.

Pure and dependency-free — no logging, no config, no imports from the rest of
this package.
"""

from __future__ import annotations


def ordered_keys(by_path: dict[str, str]) -> list[str]:
    """*by_path*'s keys, longest first.

    A path is routinely a prefix of a longer path (``/data`` and
    ``/data/inbound``), so trying the longest first is what makes the most
    specific mapping win. Ties break alphabetically so the order is stable and
    a run is reproducible.
    """
    return sorted(by_path, key=lambda k: (-len(k), k))


def map_path(value: str, by_path: dict[str, str], keys: list[str]) -> str | None:
    """*value* remapped, or ``None`` when no mapping addresses it.

    An exact match wins; failing that the longest key that *value* sits under
    is treated as a directory prefix and replaced, so mapping one folder moves
    everything beneath it. Comparison is case-insensitive because
    :func:`xref.sourcing.classify` lowercases its keys, matching the repo-wide
    "names lowercased at extraction" convention.

    *keys* is :func:`ordered_keys`' output, passed in rather than derived here
    so a caller sweeping many values sorts the mapping once.
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
