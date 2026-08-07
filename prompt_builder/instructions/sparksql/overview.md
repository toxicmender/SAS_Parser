Target Spark SQL (ANSI dialect), not the PySpark DataFrame API. Emit
`spark.sql("...")`-ready statements — `CREATE OR REPLACE TEMP VIEW` /
`CREATE TABLE ... AS SELECT` for each SAS step's output dataset — so the
translation is one readable SQL script that mirrors the SAS step sequence.

## Output format
One fenced ```sql block per SAS step, in execution order. Name each result
after the SAS output dataset (`work.foo` -> a view/table `foo`). Keep one
statement per logical step; do not collapse several SAS steps into a single
opaque query unless they are a trivial rename.

Pick the statement by where the SAS wrote: a temporary dataset (`work.*`, or
an unqualified name) -> `CREATE OR REPLACE TEMP VIEW`; a permanent libref
-> `CREATE TABLE ... AS SELECT`. Inside one step, lift each stage of a
multi-part transformation into its own named CTE rather than nesting
subqueries — a long `CASE` chain or a repeated aggregate reads better named
once, and the CTE names carry the SAS step's intent into the SQL.

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

## ANSI mode: Spark raises where SAS returned missing
⚠️ Spark 4 enables `spark.sql.ansi.enabled` by **default**. Three things SAS
does silently now abort the query:
- **Divide by zero.** SAS yields a missing value and writes a log note; Spark
  raises. Use `try_divide(a, b)` (or `a / NULLIF(b, 0)`) wherever the
  denominator is not provably non-zero.
- **Invalid cast.** `CAST('abc' AS DOUBLE)` raises; SAS `INPUT` yields missing
  and notes it. Use `try_cast`.
- **Arithmetic overflow.** Raises instead of wrapping; `try_add`, `try_sum`,
  `try_multiply` return `NULL` instead.
`try_to_date` / `try_to_timestamp` do the same for parsing. Reach for the
`try_*` form whenever the SAS original would have produced a missing value and
carried on — a raised error turns a row-level nuisance into a failed job.

Prefer standard SQL constructs over Spark-proprietary ones where both express
the same thing, so the output ports; this is about *dialect*, not about
avoiding built-ins — a built-in is still better than a hand-rolled `CASE` or
regex (see the function guidance).

## Null and missing-value semantics
A SAS missing numeric (`.`) and missing character (`" "`) both become SQL
`NULL`. This changes behaviour: in SAS a missing value sorts low and is
treated as less than any number, and `sum()`-style stats silently skip it;
in Spark SQL any arithmetic or comparison with `NULL` yields `NULL`, and
`WHERE x = NULL` never matches — use `IS NULL` / `IS NOT NULL`. Aggregates
skip `NULL` like SAS, but `COUNT(col)` excludes nulls while `COUNT(*)` does
not. ⚠️ Flag every place a SAS numeric comparison or filter could behave
differently once missing values are `NULL`.
