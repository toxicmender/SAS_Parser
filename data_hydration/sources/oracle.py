"""Reading an Oracle table, whole or one partition at a time.

``oracledb`` in **thin mode** — no Oracle Instant Client to install, which is
what makes this runnable from a Databricks job cluster without a custom image.

The partition predicate the planner chose arrives on the item and is spliced
into the ``SELECT``: a native partition becomes ``PARTITION (p_2024_01)`` after
the table name, a column range becomes a ``WHERE`` clause. Building the SQL here
rather than in the planner keeps dialect out of the planning layer, which has to
stay importable with no driver present.

Logger name: ``data_hydration.sources.oracle``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from ..secrets import resolve_secret
from .base import SourceInfo

if TYPE_CHECKING:
    from ..config import HydrationConfig
    from ..models import HydrationItem

logger = logging.getLogger(__name__)


class OracleProbe:
    """Answers :class:`~data_hydration.partition.SourceProbe` from the data dictionary.

    Held separate from :class:`OracleReader` because it has a different
    lifetime: the planner needs it once per table before any item exists, and
    :mod:`complexity` never builds one at all.

    Every method returns ``None`` rather than raising when the dictionary views
    are not readable — a schema-only account frequently cannot see
    ``ALL_TAB_PARTITIONS``, and that should downgrade the strategy rather than
    fail the run.
    """

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def _one(self, sql: str, **binds: Any) -> Any:
        with self._connection.cursor() as cursor:
            cursor.execute(sql, **binds)
            row = cursor.fetchone()
            return row[0] if row else None

    def native_partitions(self, owner: str, table: str) -> list[str] | None:
        """Partition names from ``ALL_TAB_PARTITIONS``, in position order."""
        with self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT partition_name FROM all_tab_partitions "
                "WHERE table_owner = :owner AND table_name = :table "
                "ORDER BY partition_position",
                owner=owner.upper(),
                table=table.upper(),
            )
            names = [row[0] for row in cursor.fetchall()]
        return names or None

    def row_count(self, owner: str, table: str) -> int | None:
        """``NUM_ROWS`` from the optimiser statistics, not ``COUNT(*)``.

        An estimate is the right trade here: it decides how many slices to cut,
        where being 10% out costs nothing, and a real count on a large table
        costs a full scan before the load has even started.
        """
        return self._one(
            "SELECT num_rows FROM all_tables "
            "WHERE owner = :owner AND table_name = :table",
            owner=owner.upper(),
            table=table.upper(),
        )

    def range_column(self, owner: str, table: str) -> tuple[str, float, float] | None:
        """An indexed NUMBER/DATE column and its bounds, or ``None``.

        Indexed specifically: splitting on an unindexed column turns one table
        scan into *n* table scans, which is slower than not partitioning at all.
        """
        with self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT c.column_name FROM all_tab_columns c "
                "JOIN all_ind_columns i ON i.table_owner = c.owner "
                " AND i.table_name = c.table_name "
                " AND i.column_name = c.column_name "
                "WHERE c.owner = :owner AND c.table_name = :table "
                " AND c.data_type IN ('NUMBER', 'DATE') "
                "ORDER BY i.column_position "
                "FETCH FIRST 1 ROWS ONLY",
                owner=owner.upper(),
                table=table.upper(),
            )
            row = cursor.fetchone()
        if not row:
            return None
        column = row[0]
        with self._connection.cursor() as cursor:
            cursor.execute(
                f'SELECT MIN("{column}"), MAX("{column}") FROM "{owner}"."{table}"'
            )
            bounds = cursor.fetchone()
        if not bounds or bounds[0] is None or bounds[1] is None:
            return None
        try:
            return (column, float(bounds[0]), float(bounds[1]))
        except (TypeError, ValueError):
            # A DATE column: real range splitting over dates needs date
            # arithmetic in the predicate, which this does not do yet.
            logger.debug(f"range_column: '{column}' is not numerically splittable")
            return None


def connect(item: "HydrationItem", config: "HydrationConfig") -> Any:
    """An ``oracledb`` connection for the item's LIBNAME options.

    The DSN comes from the SAS ``path=`` option when it had one, else the
    configured default. The password never comes from either — it resolves
    through the credential chain, keyed on the libref so one corpus can reach
    several databases with different accounts.
    """
    try:
        import oracledb
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(
            "oracledb is required to read Oracle; install it with "
            "'pip install \"sas-parser[oracle]\"'"
        ) from exc

    options = item.source.option_map
    dsn = options.get("path") or config.oracle_dsn
    user = options.get("user") or config.oracle_user
    libref = item.source.libref or "oracle"
    password = resolve_secret(f"oracle_password_{libref}", scope=config.secret_scope)

    if not dsn:
        raise ValueError(
            f"no Oracle DSN for libref '{libref}': the LIBNAME carried no "
            f"path= option and data_hydration.oracle_dsn is unset"
        )
    logger.info(f"connect: oracle {user}@{dsn} (libref {libref})")
    return oracledb.connect(user=user, password=password, dsn=dsn)


class OracleReader:
    """Streams one item's rows as DataFrames."""

    def __init__(self, item: "HydrationItem", config: "HydrationConfig") -> None:
        self._item = item
        self._config = config
        self._connection: Any = None

    @property
    def connection(self) -> Any:
        """The connection, opened on first use."""
        if self._connection is None:
            self._connection = connect(self._item, self._config)
        return self._connection

    def _sql(self) -> str:
        """The ``SELECT`` for this item, with its partition spliced in."""
        source = self._item.source
        schema = source.object_name
        table = self._item.target_table.rsplit(".", 1)[-1]
        # The target table name has been through naming.sanitise_part, so it is
        # the source object that must supply the real name.
        table = source.option_map.get("table", table)
        qualified = f'"{schema.upper()}"."{table.upper()}"' if schema else f'"{table.upper()}"'

        partition = self._item.partition
        if partition is None or not partition.predicate:
            return f"SELECT * FROM {qualified}"
        if partition.predicate.upper().startswith("PARTITION"):
            return f"SELECT * FROM {qualified} {partition.predicate}"
        return f"SELECT * FROM {qualified} WHERE {partition.predicate}"

    def info(self) -> SourceInfo:
        with self.connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM ({self._sql()}) WHERE ROWNUM < 1")
            columns = tuple(d[0] for d in cursor.description)
        return SourceInfo(columns=columns)

    def batches(self) -> Iterator[Any]:
        import pandas as pd

        sql = self._sql()
        logger.info(f"OracleReader: {sql}")
        with self.connection.cursor() as cursor:
            cursor.arraysize = self._config.fetch_size
            cursor.execute(sql)
            # An Index rather than a plain list: pandas accepts both, but only
            # this form matches the stubs, and building it once keeps it out of
            # the fetch loop.
            columns = pd.Index([d[0] for d in cursor.description])
            while True:
                rows = cursor.fetchmany(self._config.fetch_size)
                if not rows:
                    break
                yield pd.DataFrame(rows, columns=columns)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
