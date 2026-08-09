## [kind: DATA_STEP] [lang: pyspark] DATA step translation

Read tables with `spark.table("catalog.schema.table")` and write managed
tables with `write.mode(...).saveAsTable(...)`. Translate row expressions with
`F.col`, `F.when`, `F.lit`, `F.coalesce`, and `F.expr`; never rely on Python
row loops or driver-side collection for DATA step logic.

Translate BY-group work with Window specifications. Make sort direction and
partition keys explicit, and use `row_number`, `lag`, `lead`, aggregates, or
window frames for FIRST./LAST.-style logic.
