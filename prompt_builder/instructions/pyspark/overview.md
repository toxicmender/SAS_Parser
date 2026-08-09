## [kind: DATA_STEP, PROC_STEP] [lang: pyspark] Databricks PySpark target

Emit executable PySpark DataFrame code for a Databricks notebook. Import
`pyspark.sql.functions as F` and use the existing `spark` session; do not
create a local SparkSession. Prefer DataFrame transformations for DATA steps
and use `spark.sql(...)` only where a SQL expression is clearer.

Preserve SAS missing-value, ordering, and duplicate-handling semantics
explicitly. Record every approximation or unsupported PROC in the risks.
