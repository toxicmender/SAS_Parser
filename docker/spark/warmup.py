"""Throwaway build-time job that forces Spark to resolve the Delta jars.

Run once by docker/spark/install_delta.sh via ``spark-submit --master local[1]``,
which reads $SPARK_CONF_DIR/spark-defaults.conf — so the ``spark.jars.packages``
coordinate and the ``spark.jars.ivy`` cache directory written there are what
get exercised. Real Delta work is done (not just a session start) because
resolving the jars and *loading* the catalog extension are separate failures,
and a build that only proves the first is worth little.

Both Delta APIs are exercised, and the second one is the point:

``path``
    ``df.write.format("delta").save(path)`` — the DataFrame writer.
``catalog``
    ``CREATE TABLE ... USING DELTA`` — goes through Spark's session catalog
    and ``CatalogStorageFormat``, which the path-based writer never touches.

They fail independently. A delta-spark built against a different Spark minor
than the installed pyspark can serve path-based writes perfectly while every
catalog statement dies on ``NoSuchMethodError: CatalogStorageFormat.copy`` —
which is the API ``memory.store._DeltaBackend._ensure_table`` uses, so the
image would build clean and the application would be entirely broken. Proving
only the path API is what let exactly that ship; do not drop the catalog half.

Everything is written under a throwaway tmpdir (the warehouse is pointed there
too, so no managed table leaks into /data/warehouse in the build layer).
"""

from __future__ import annotations

import tempfile

from pyspark.sql import SparkSession

_TABLE = "delta_warmup_catalog"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        spark = (
            SparkSession.builder.appName("delta-warmup")
            # Overrides spark-defaults' /data/warehouse: a managed table
            # created here must not survive into the image layer.
            .config("spark.sql.warehouse.dir", f"{tmp}/warehouse")
            .getOrCreate()
        )
        try:
            path = f"{tmp}/warmup"
            spark.range(1).write.format("delta").mode("overwrite").save(path)
            count = spark.read.format("delta").load(path).count()
            print(f"delta warmup: path-based write/read ok ({count} row)")

            spark.sql(f"CREATE TABLE IF NOT EXISTS {_TABLE} (id BIGINT) USING DELTA")
            spark.sql(f"INSERT INTO {_TABLE} VALUES (1)")
            rows = spark.table(_TABLE).count()
            spark.sql(f"DROP TABLE IF EXISTS {_TABLE}")
            print(f"delta warmup: catalog table create/insert/read ok ({rows} row)")
        finally:
            spark.stop()


if __name__ == "__main__":
    main()
