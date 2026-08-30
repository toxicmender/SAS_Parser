"""Managed Delta sink with a lazily resolved Spark runtime."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from sas_migrate.application.hydration import HydrationItem, WriteMode

LOGGER = logging.getLogger(__name__)


def _default_session() -> Any:
    from pyspark.sql import SparkSession

    return SparkSession.getActiveSession() or SparkSession.builder.appName(
        "sas-parser-hydration"
    ).getOrCreate()


class DeltaHydrationSink:
    def __init__(
        self,
        *,
        session_factory: Callable[[], Any] | None = None,
        apply_index_clustering: bool = False,
        index_columns: Callable[[HydrationItem], tuple[str, ...]] | None = None,
    ) -> None:
        self._session_factory = session_factory or _default_session
        self._apply_index_clustering = apply_index_clustering
        self._index_columns = index_columns

    def write(self, item: HydrationItem, batches: Iterable[Any]) -> int:
        frames = tuple(frame for frame in batches if frame is not None)
        rows = sum(len(frame) for frame in frames)
        if not frames or rows == 0:
            return 0

        spark = self._session_factory()
        dataset = spark.createDataFrame(frames[0])
        for frame in frames[1:]:
            dataset = dataset.unionByName(spark.createDataFrame(frame))

        catalog, schema, _table = item.target_table.split(".")
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
        mode = "overwrite" if item.write_mode is WriteMode.OVERWRITE else "append"
        writer = dataset.write.format("delta").mode(mode)
        if mode == "overwrite":
            writer = writer.option("overwriteSchema", "true")
        writer.saveAsTable(item.target_table)

        if self._apply_index_clustering:
            requested = item.cluster_by
            if not requested and self._index_columns is not None:
                requested = self._index_columns(item)
            known = {str(column).casefold() for column in dataset.columns}
            columns = tuple(column for column in requested if column.casefold() in known)
            if columns:
                names = ", ".join(f"`{column}`" for column in columns)
                try:
                    spark.sql(f"ALTER TABLE {item.target_table} CLUSTER BY ({names})")
                except Exception as exc:  # noqa: BLE001 - optional optimisation
                    LOGGER.warning("could not cluster %s: %s", item.target_table, exc)
        return rows


__all__ = ["DeltaHydrationSink"]
