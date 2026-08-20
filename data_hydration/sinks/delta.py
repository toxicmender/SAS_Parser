"""Writing an item's rows into a managed Delta table.

Managed, not external: the target is ``catalog.schema.table`` with no
``LOCATION``, so Unity Catalog owns the files and the governance that comes with
them. That is the decision the user made when hydration was specified, and it is
why nothing here takes a storage path.

The write is done through Spark rather than the SQL connector because the input
is a stream of DataFrames of unknown total size — ``createDataFrame`` plus
``saveAsTable`` handles that without materialising the whole table client-side,
and a batch of ``INSERT`` statements does not.

⚠️ **This module cannot run in the repo's local ``.venv``**, where ``pyspark`` is
shadowed by ``databricks-connect``. Verify changes here in Docker
(``docker/spark``), the same rule ``memory.store``'s Delta backend follows.

Logger name: ``data_hydration.sinks.delta``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..models import WriteMode
from ..sources.base import reader_for

if TYPE_CHECKING:
    from ..config import HydrationConfig
    from ..models import HydrationItem

logger = logging.getLogger(__name__)


def get_session() -> Any:
    """The active :class:`SparkSession`, or a new one against the configured master.

    Inside a Databricks job the session already exists and must be reused;
    outside one this is what a local test cluster gets. ``app_config.spark``
    owns both halves of that decision — see
    :func:`app_config.spark.active_or_new_session`.
    """
    from app_config.spark import active_or_new_session

    return active_or_new_session("sas-parser-hydration")


def _ensure_schema(spark: Any, table: str) -> None:
    """Create the target schema when it does not exist.

    The *schema*, never the catalog: creating a catalog is a governance act with
    storage and permission consequences that a data load has no business
    performing. A missing catalog fails, and the message says so.
    """
    catalog, schema, _ = table.split(".")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")


def _clustering_columns(item: "HydrationItem", known: set[str]) -> tuple[str, ...]:
    """The SAS index's columns, if they can be read and are real columns.

    The planner cannot supply these — the names are inside the ``.sas7bndx``
    binary — so they are recovered here, where the file is at hand. Recovery is
    best-effort by design, so the result is intersected with the columns that
    actually landed: an index parse that guessed wrong must not reach
    ``ALTER TABLE ... CLUSTER BY`` naming a column the table does not have.
    """
    if item.cluster_by:
        return tuple(c for c in item.cluster_by if c.lower() in known)
    from ..models import SourceKind

    if item.source.kind is not SourceKind.SAS_DATASET:
        return ()
    from pathlib import Path

    from ..sources.sas_files import DATASET_SUFFIX, index_columns

    path = Path(item.source.locator) / f"{item.source.object_name}{DATASET_SUFFIX}"
    recovered = index_columns(path)
    keep = tuple(c for c in recovered if c.lower() in known)
    if recovered and not keep:
        logger.info(
            f"write_item: the index beside {path.name} yielded no name matching "
            f"a column of {item.target_table}; not clustering"
        )
    return keep


def _cluster_by(spark: Any, table: str, columns: tuple[str, ...]) -> None:
    """Apply the SAS index hint as Delta clustering.

    Only reached when ``data_hydration.apply_index_clustering`` is on. A failure
    is logged and swallowed: clustering is an optimisation, and a table that
    landed correctly but is not clustered is a far better outcome than a failed
    load.
    """
    if not columns:
        return
    names = ", ".join(f"`{c}`" for c in columns)
    try:
        spark.sql(f"ALTER TABLE {table} CLUSTER BY ({names})")
        logger.info(f"write_item: clustered {table} by {names}")
    except Exception as exc:
        logger.warning(f"write_item: could not cluster {table} by {names}: {exc}")


def write_item(item: "HydrationItem", config: "HydrationConfig") -> int:
    """Read one item and write it to its target table; return the row count.

    The write mode is the plan's, not a runtime decision: the first item for a
    table overwrites and the rest append, so re-running a plan is idempotent
    however the items are ordered.

    Batches are unioned into one DataFrame before the write rather than written
    one at a time, because an append per batch would leave a partially-written
    table behind on a mid-stream failure — with one write, the table either has
    the item's rows or it does not.
    """
    spark = get_session()
    reader = reader_for(item, config)
    try:
        frames = [frame for frame in reader.batches() if frame is not None]
    finally:
        reader.close()

    rows = sum(len(frame) for frame in frames)
    if not frames or rows == 0:
        logger.warning(f"write_item: {item.source} yielded no rows; nothing written")
        return 0

    dataset = spark.createDataFrame(frames[0])
    for frame in frames[1:]:
        dataset = dataset.unionByName(spark.createDataFrame(frame))

    _ensure_schema(spark, item.target_table)
    mode = "overwrite" if item.write_mode is WriteMode.OVERWRITE else "append"
    logger.info(f"write_item: {mode} {rows} row(s) -> {item.target_table}")
    writer = dataset.write.format("delta").mode(mode)
    if mode == "overwrite":
        # The source schema is authoritative on a full replace; without this a
        # column added upstream fails the write instead of landing.
        writer = writer.option("overwriteSchema", "true")
    writer.saveAsTable(item.target_table)

    if config.apply_index_clustering:
        landed = {str(name).lower() for name in dataset.columns}
        _cluster_by(spark, item.target_table, _clustering_columns(item, landed))
    return rows
