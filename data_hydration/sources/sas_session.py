"""Reading through a live SAS session — the only route to SPD Engine data.

An SPD Engine library stores each dataset across numbered ``.dpf`` component
files beside an ``.mdf`` metadata file, and there is **no open-source reader for
that layout**. ``pyreadstat`` reads ``.sas7bdat`` and does not read ``.dpf``;
nothing else does either. Writing one would mean reverse-engineering a
proprietary partitioned storage format, which is a project, not a function.

So the split this module represents is:

* **Planning** an SPD Engine load needs nothing installed — the partitions are
  files, and :func:`data_hydration.partition.spde_partitions` finds them with a
  directory listing. A complexity report can show the whole shape of the load.
* **Reading** it needs SAS. ``saspy`` attaches to a SAS 9 workspace or a Viya
  compute context, assigns the library with its real engine, and hands rows back
  as a DataFrame.

When no session is configured the planner marks those items
``needs_operator_input`` rather than letting them fail at write time, so the
report says "this needs a SAS session" instead of the run dying halfway.

Logger name: ``data_hydration.sources.sas_session``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from .base import SourceInfo, UnsupportedSource

if TYPE_CHECKING:
    from ..config import HydrationConfig
    from ..models import HydrationItem

logger = logging.getLogger(__name__)


def open_session(config: "HydrationConfig") -> Any:
    """A ``saspy`` session from the configured coordinates.

    Raises
    ------
    UnsupportedSource
        ``saspy`` is absent or no host is configured. Both are operator actions,
        and the message says which.
    """
    if not config.sas_host:
        raise UnsupportedSource(
            "no SAS session configured: set data_hydration.sas_host (and "
            "sas_port / sas_context) to read SPD Engine libraries"
        )
    try:
        import saspy
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise UnsupportedSource(
            "saspy is required to read SPD Engine data; install it with "
            "'pip install \"sas-parser[saspy]\"'"
        ) from exc

    kwargs: dict[str, Any] = {"java": "java", "iomhost": config.sas_host}
    if config.sas_port:
        kwargs["iomport"] = config.sas_port
    if config.sas_context:
        kwargs["context"] = config.sas_context
    logger.info(f"open_session: saspy -> {config.sas_host}:{config.sas_port or ''}")
    return saspy.SASsession(**kwargs)


class SpdeReader:
    """One SPD Engine dataset, read whole through SAS.

    Whole and not per-component, deliberately. The planner *counts* the ``.dpf``
    components — that number is worth reporting — but does not fan them out into
    an item each, because a component cannot be read in isolation: doing so means
    bypassing the engine that owns the partitioning scheme, and ``saspy`` offers
    no way to ask for one. An item per component would read the entire dataset
    once per component and write it as many times.
    """

    def __init__(self, item: "HydrationItem", config: "HydrationConfig") -> None:
        self._item = item
        self._config = config
        self._session: Any = None

    @property
    def session(self) -> Any:
        if self._session is None:
            self._session = open_session(self._config)
        return self._session

    def _assign_library(self) -> str:
        """Assign the SPD Engine library in the session; return the libref."""
        source = self._item.source
        libref = (source.libref or "spdelib")[:8]
        self.session.submit(
            f"libname {libref} spde '{source.locator}';"
        )
        return libref

    def info(self) -> SourceInfo:
        libref = self._assign_library()
        frame = self.session.sasdata2dataframe(
            table=self._item.source.object_name, libref=libref, obs=1
        )
        columns = tuple(frame.columns) if frame is not None else ()
        return SourceInfo(columns=columns)

    def batches(self) -> Iterator[Any]:
        libref = self._assign_library()
        table = self._item.source.object_name
        logger.info(f"SpdeReader: {libref}.{table} via saspy")
        frame = self.session.sasdata2dataframe(table=table, libref=libref)
        if frame is not None:
            yield frame

    def close(self) -> None:
        if self._session is not None:
            self._session.endsas()
            self._session = None
