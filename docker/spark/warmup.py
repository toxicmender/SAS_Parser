"""Throwaway build-time job that forces Spark to resolve the Delta jars.

Run once by docker/spark/install_delta.sh via ``spark-submit --master local[1]``,
which reads $SPARK_CONF_DIR/spark-defaults.conf — so the ``spark.jars.packages``
coordinate and the ``spark.jars.ivy`` cache directory written there are what
get exercised. A real Delta write is done (not just a session start) because
resolving the jars and *loading* the catalog extension are separate failures,
and a build that only proves the first is worth little.

The table is written under /tmp and left behind in the build layer's tmpdir,
which docker discards.
"""

from __future__ import annotations

import tempfile

from pyspark.sql import SparkSession


def main() -> None:
    spark = SparkSession.builder.appName("delta-warmup").getOrCreate()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/warmup"
            spark.range(1).write.format("delta").mode("overwrite").save(path)
            count = spark.read.format("delta").load(path).count()
            print(f"delta warmup ok: read back {count} row(s)")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
