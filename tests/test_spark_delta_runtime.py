"""Runtime contracts for the containerized PySpark and Delta Lake stack.

These tests intentionally use the public PySpark DataFrame and ``DeltaTable``
APIs directly. The application-level memory contracts prove our adapters; this
module separately proves that the Java/Python/Maven artifacts assembled by the
CI image can execute the engine operations those adapters ultimately require.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.spark_delta


def test_pyspark_dataframe_transformations(delta_spark) -> None:
    """The Python client can plan and execute DataFrame work on the JVM."""
    from pyspark.sql import functions as F

    frame = delta_spark.createDataFrame(
        [(1, "north", 10), (2, "north", 15), (3, "south", 7)],
        "id LONG, region STRING, amount LONG",
    )

    rows = (
        frame.where(F.col("amount") >= 10)
        .groupBy("region")
        .agg(F.sum("amount").alias("total"))
        .orderBy("region")
        .collect()
    )

    assert [(row["region"], row["total"]) for row in rows] == [("north", 25)]


def test_delta_table_python_api_merge_update_delete(delta_spark, tmp_path) -> None:
    """The Python DeltaTable API can mutate and read a real Delta log."""
    from delta.tables import DeltaTable
    from pyspark.sql import functions as F

    table_path = str(tmp_path / "delta-table-api")
    delta_spark.createDataFrame(
        [(1, "one"), (2, "two")], "id LONG, value STRING"
    ).write.format("delta").mode("overwrite").save(table_path)

    table = DeltaTable.forPath(delta_spark, table_path)
    changes = delta_spark.createDataFrame(
        [(2, "TWO"), (3, "three")], "id LONG, value STRING"
    )
    (
        table.alias("target")
        .merge(changes.alias("source"), "target.id = source.id")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
    table.update(condition=F.col("id") == 1, set={"value": F.lit("ONE")})
    table.delete(condition=F.col("id") == 2)

    rows = table.toDF().orderBy("id").collect()
    assert [(row["id"], row["value"]) for row in rows] == [
        (1, "ONE"),
        (3, "three"),
    ]

    operations = {row["operation"] for row in table.history(10).collect()}
    assert {"WRITE", "MERGE", "UPDATE", "DELETE"} <= operations
