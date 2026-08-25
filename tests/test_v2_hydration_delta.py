"""Real containerized PySpark/Delta contract for the v2 hydration sink."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from sas_migrate.adapters.hydration import DeltaHydrationSink
from sas_migrate.application.hydration import (
    HydrationItem,
    HydrationSource,
    SourceKind,
    WriteMode,
)

pytestmark = pytest.mark.spark_delta


def test_real_delta_hydration_sink_overwrites_then_appends(delta_spark: Any) -> None:
    schema = f"hydration_{uuid4().hex}"
    table = f"spark_catalog.{schema}.sales"
    source = HydrationSource(kind=SourceKind.FILE, object_name="sales")
    sink = DeltaHydrationSink(session_factory=lambda: delta_spark)
    overwrite = HydrationItem(source=source, target_table=table)
    append = overwrite.model_copy(update={"write_mode": WriteMode.APPEND})

    try:
        assert sink.write(overwrite, ([{"id": 1}, {"id": 2}],)) == 2
        assert sink.write(append, ([{"id": 3}],)) == 1
        rows = delta_spark.table(table).orderBy("id").collect()
        assert [row["id"] for row in rows] == [1, 2, 3]
        detail = delta_spark.sql(f"DESCRIBE DETAIL {table}").first()
        assert detail.format == "delta"
    finally:
        delta_spark.sql(f"DROP SCHEMA IF EXISTS spark_catalog.{schema} CASCADE")
