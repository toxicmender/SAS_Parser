Target **Databricks SQL** (Databricks Runtime or a SQL warehouse), not the
PySpark DataFrame API: emit `spark.sql("...")`-ready statements, one readable
script mirroring the SAS step sequence. ⚠️ This guidance uses constructs
open-source Spark does not have — `QUALIFY`, SQL scripting, `EXECUTE
IMMEDIATE`, `UNPIVOT`, three-level names — so do not follow it for a
non-Databricks Spark deployment.

## Output format
One fenced ```sql block per SAS step, in execution order, named after the SAS
output dataset. One statement per logical step; never collapse several steps
into one query. Within a step, lift each stage into a named CTE rather than
nesting subqueries — the names carry the step's intent.

SAS names are two-level `libref.dataset`, Databricks three-level
`catalog.schema.table`. `work.*` (or unqualified) -> `CREATE OR REPLACE TEMP
VIEW foo`; a permanent libref becomes the **schema** ->
`CREATE TABLE <catalog>.mylib.accounts AS SELECT ...`, a managed Delta table.
State the catalog you assumed, once. Delta is the default, so no `USING DELTA`;
⚠️ no `LOCATION` either — that makes the table external, changing who owns the
files — unless the SAS libname pointed at an explicit path.

## [kind: DATA_STEP] Set-based, not row-by-row
SAS DATA steps iterate the PDV row by row with implicit retain and `_N_`.
Spark SQL is declarative and unordered. Re-express row logic as set
operations: `CASE WHEN` for `IF/THEN/ELSE`, a self-join or window function
for anything referencing a prior row (`LAG`/`LEAD`/`SUM() OVER`), and an
explicit `ORDER BY` wherever SAS relied on observation order. Never assume
Spark preserves input row order without an `ORDER BY`.

## Data types and literals
Map SAS numerics to `DOUBLE` (or `DECIMAL(p,s)` when exactness matters) and
SAS character to `STRING`. SAS has no boolean type; a numeric 0/1 flag maps
to `BOOLEAN` only when the source clearly uses it as one. Quote string
literals with single quotes; escape embedded quotes by doubling them.

## ANSI mode: the query raises where SAS returned missing
⚠️ `ANSI_MODE` is **on by default** (for Databricks accounts created from
19 Oct 2022, and in Spark 4), so operations SAS completed with a missing value
and a log note now abort the query: divide by zero, invalid cast, arithmetic
overflow — and a numeric-to-integral cast that would truncate. Use the `try_*`
form —
`try_divide`, `try_cast`, `try_add`, `try_sum`, `try_to_date`,
`try_to_timestamp` — wherever the SAS original would have produced a missing
value and carried on. A raised error turns a row-level nuisance into a failed
job. Prefer standard SQL over Spark-proprietary syntax where both express the
same thing.

## Null and missing-value semantics
A SAS missing numeric (`.`) and missing character (`" "`) both become SQL
`NULL`. This changes behaviour: in SAS a missing value sorts low and is
treated as less than any number, and `sum()`-style stats silently skip it;
in Spark SQL any arithmetic or comparison with `NULL` yields `NULL`, and
`WHERE x = NULL` never matches — use `IS NULL` / `IS NOT NULL`. Aggregates
skip `NULL` like SAS, but `COUNT(col)` excludes nulls while `COUNT(*)` does
not. ⚠️ Flag every place a SAS numeric comparison or filter could behave
differently once missing values are `NULL`.
