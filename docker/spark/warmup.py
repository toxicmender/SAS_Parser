"""Throwaway build-time job that forces Spark to resolve the Delta jars.

Run once by docker/spark/install_delta.sh via ``spark-submit --master local[1]``,
which reads $SPARK_CONF_DIR/spark-defaults.conf — so the ``spark.jars.packages``
coordinate and the ``spark.jars.ivy`` cache directory written there are what
get exercised. Real Delta work is done (not just a session start) because
resolving the jars and *loading* the catalog extension are separate failures,
and a build that only proves the first is worth little.

Three Delta APIs are exercised, and each later one is the point of the one
before it:

``path``
    ``df.write.format("delta").save(path)`` — the DataFrame writer.
``catalog``
    ``CREATE TABLE ... USING DELTA`` — goes through Spark's session catalog
    and ``CatalogStorageFormat``, which the path-based writer never touches.
``properties``
    ``ALTER TABLE ... SET TBLPROPERTIES`` — the property writer, which reaches
    catalyst internals neither of the above does.

They fail independently, and each boundary has already been crossed in
production. A delta-spark built against a different Spark minor than the
installed pyspark can serve path-based writes perfectly while every catalog
statement dies on ``NoSuchMethodError: CatalogStorageFormat.copy``; proving
only the path API is what let exactly that ship.

The properties half is here for the same reason, learned the same way: with an
unsupported pyspark 4.1.3 / delta-spark 4.1.0 pair both halves above pass and
``ALTER TABLE ... SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')``
dies on ``ClassNotFoundException:
org.apache.spark.sql.catalyst.plans.logical.IgnoreCachedData`` — a trait Spark
3.x had and Spark 4.x does not. ``memory.store._DeltaBackend`` depends on that
statement, so the image built clean and every one of its 14 contract tests
failed. Do not drop any of the three.

Everything is written under a throwaway tmpdir (the warehouse is pointed there
too, so no managed table leaks into /data/warehouse in the build layer).
"""

from __future__ import annotations

import tempfile

from pyspark.sql import SparkSession

_TABLE = "delta_warmup_catalog"


def _check_property_writes(spark: SparkSession) -> None:
    """Exercise ``ALTER TABLE ... SET TBLPROPERTIES``, and report what it does.

    Deliberately an ALTER on an existing table rather than a ``TBLPROPERTIES``
    clause on the ``CREATE``: the two take different code paths inside Delta,
    and only this one reaches the catalyst internals that break across a Spark
    minor.

    **A warning, not a build failure** — unlike the two checks above, which are
    fatal. The difference is what the application can still do. Path and
    catalog writes are unconditional: without them ``memory.store`` cannot
    create its table at all, and a green build would be a lie.
    ``SET TBLPROPERTIES`` is not — the ``CREATE`` already enables CDF on a table
    it creates, so a *fresh* deployment never issues this statement and works
    completely. Only migrating a table that predates CDF needs it.

    Failing the build here would therefore make the image unbuildable for
    everyone in order to flag a limitation that affects upgrades alone. The
    warning names the exact statement and the exact consequence instead, so an
    operator with a pre-CDF table knows before they meet it at run time.
    """
    try:
        spark.sql(
            f"ALTER TABLE {_TABLE} SET TBLPROPERTIES "
            "('delta.enableChangeDataFeed' = 'true')"
        )
    except Exception as exc:  # noqa: BLE001 - surface any JVM/Py4J limitation
        print(
            f"delta warmup: WARNING — ALTER TABLE SET TBLPROPERTIES failed "
            f"({type(exc).__name__}). This pyspark/delta-spark pair cannot "
            f"write table properties. A NEW deployment is unaffected: "
            f"memory.store's CREATE enables Change Data Feed on the table it "
            f"creates, and it only issues this ALTER when the property is "
            f"genuinely missing. UPGRADING a table that predates CDF will fail "
            f"until a compatible pair exists (see pyproject.toml's spark extra)."
        )
        return

    enabled = {
        row["key"]: row["value"]
        for row in spark.sql(f"SHOW TBLPROPERTIES {_TABLE}").collect()
    }.get("delta.enableChangeDataFeed")
    if str(enabled).lower() != "true":
        # Reported success and did nothing: worse than an exception, because
        # nothing downstream would ever notice.
        raise SystemExit(
            f"delta warmup: ALTER TABLE SET TBLPROPERTIES reported success but "
            f"delta.enableChangeDataFeed reads {enabled!r} — the property did "
            f"not stick."
        )
    print("delta warmup: table property write/read ok (CDF enabled)")


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
            spark.sparkContext.setLogLevel("WARN")
            path = f"{tmp}/warmup"
            spark.range(1).write.format("delta").mode("overwrite").save(path)
            count = spark.read.format("delta").load(path).count()
            print(f"delta warmup: path-based write/read ok ({count} row)")

            spark.sql(f"CREATE TABLE IF NOT EXISTS {_TABLE} (id BIGINT) USING DELTA")
            spark.sql(f"INSERT INTO {_TABLE} VALUES (1)")
            rows = spark.table(_TABLE).count()
            print(f"delta warmup: catalog table create/insert/read ok ({rows} row)")

            _check_property_writes(spark)
            spark.sql(f"DROP TABLE IF EXISTS {_TABLE}")
        finally:
            spark.stop()


if __name__ == "__main__":
    main()
