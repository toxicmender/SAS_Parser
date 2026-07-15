Target Spark SQL (ANSI dialect), not the PySpark DataFrame API. Emit
`spark.sql("...")`-ready statements — `CREATE OR REPLACE TEMP VIEW` /
`CREATE TABLE ... AS SELECT` for each SAS step's output dataset — so the
translation is one readable SQL script that mirrors the SAS step sequence.

## Output format
One fenced ```sql block per SAS step, in execution order. Name each result
after the SAS output dataset (`work.foo` -> a view/table `foo`). Keep one
statement per logical step; do not collapse several SAS steps into a single
opaque query unless they are a trivial rename.

## Set-based, not row-by-row
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

## Null and missing-value semantics
A SAS missing numeric (`.`) and missing character (`" "`) both become SQL
`NULL`. This changes behaviour: in SAS a missing value sorts low and is
treated as less than any number, and `sum()`-style stats silently skip it;
in Spark SQL any arithmetic or comparison with `NULL` yields `NULL`, and
`WHERE x = NULL` never matches — use `IS NULL` / `IS NOT NULL`. Aggregates
skip `NULL` like SAS, but `COUNT(col)` excludes nulls while `COUNT(*)` does
not. ⚠️ Flag every place a SAS numeric comparison or filter could behave
differently once missing values are `NULL`.
