## [when: proc:sql] PROC SQL to Spark SQL
PROC SQL is the closest SAS construct to Spark SQL — translate the query
almost directly, then adjust for these differences:
- Drop `quit;`, the `noprint`/`feedback` options, and any `CREATE TABLE ...
  AS` libref prefix; write `CREATE OR REPLACE TEMP VIEW name AS SELECT ...`.
- SAS `CALCULATED col` (referencing a select-list alias in the same query)
  is not valid in Spark; repeat the expression or wrap the SELECT in a CTE /
  subquery and reference the alias in the outer query.
- SAS `INTO :mvar` (writing a macro variable) has no SQL equivalent. Replace
  it with a query the caller reads, or a parameter — never emit `INTO`.
- Reflexive/implicit joins: SAS PROC SQL allows a comma join with a `WHERE`.
  Prefer explicit `JOIN ... ON`. A SAS join with no join condition is a
  cross join — make it `CROSS JOIN` explicitly and ⚠️ flag it.
- `monotonic()`/`number()`-style row numbering maps to
  `ROW_NUMBER() OVER (ORDER BY ...)`; an unordered SAS row counter has no
  faithful Spark equivalent — call that out.
- ⚠️ **Do not emit `QUALIFY`.** Databricks SQL accepts it, so it is an easy
  reflex, but it is **not in open-source Spark SQL** and the statement fails
  to parse there. Filter a window function through a CTE or subquery instead:
  compute `ROW_NUMBER() OVER (...) AS rn` inside, then `WHERE rn = 1` outside.
- `CREATE VIEW` (and a DATA step's `/ VIEW=` option) defines a stored query,
  not a table: map it to `CREATE OR REPLACE TEMP VIEW`. Keep it a view —
  materialising it into a table changes when the query runs and what it sees.

## [kind: DATA_STEP] MERGE and BY-group joins
A SAS `DATA` step `MERGE a b; BY key;` is a full outer join on `key`, not an
inner join — unmatched rows from either side are kept. Translate to
`FULL OUTER JOIN ... USING (key)` (or `LEFT`/`INNER` only when `IF a` / `IF
a AND b` subsetting makes the intent narrower). ⚠️ SAS overwrites same-named
non-BY columns left-to-right (the last dataset wins); reproduce that with an
explicit `COALESCE(b.col, a.col)` or a chosen side, and state which side you
kept. A `MERGE` without `BY` is positional (one-to-one by row position) and
has no correct Spark SQL translation — flag it as unsafe rather than
guessing.

## [kind: DATA_STEP] BY-group processing (FIRST./LAST.)
`FIRST.var` / `LAST.var` flags map to window functions over
`PARTITION BY <by-vars> ORDER BY <by-vars>`: `FIRST.x` is
`ROW_NUMBER() = 1`, `LAST.x` is `ROW_NUMBER() OVER (... ORDER BY ... DESC) =
1` (or `COUNT(*) OVER (...)` compared to a running `ROW_NUMBER`). Retained
accumulators across a BY group become `SUM(...) OVER (PARTITION BY ... ORDER
BY ...)`.

## [when: proc:means, proc:summary] [kind: PROC_STEP] Summary statistics
`PROC MEANS` / `PROC SUMMARY` map to `GROUP BY` with aggregate functions
(`AVG`, `SUM`, `MIN`, `MAX`, `COUNT`, `STDDEV`). The `CLASS` variables are
the `GROUP BY` keys; `VAR` variables are the aggregated columns. Remember
Spark aggregates skip `NULL`, so `N` vs `NMISS` must be `COUNT(col)` vs
`SUM(CASE WHEN col IS NULL THEN 1 ELSE 0 END)`.

⚠️ **Subtotals are opt-in, and the default is the trap.** Without `NWAY`, SAS
emits *every* combination of CLASS levels — the grand total (`_TYPE_=0`),
each one-way margin, and so on — and a plain `GROUP BY` reproduces only the
last of those. So: `NWAY` -> a plain `GROUP BY` on all CLASS variables (the
common case); a `TYPES` / `WAYS` request -> `GROUPING SETS` / `ROLLUP` /
`CUBE` naming exactly the requested levels. Never add a `ROLLUP` the SAS did
not ask for, and where the source relies on `_TYPE_` or `_FREQ_`, reproduce
them explicitly (`GROUPING_ID()` and `COUNT(*)`).

⚠️ **`MISSING`**: by default `PROC MEANS` drops observations whose CLASS
variable is missing, while Spark `GROUP BY` keeps `NULL` as a group. Add
`WHERE <class vars> IS NOT NULL` unless the SAS specifies `MISSING`.

`OUTPUT OUT=` names the result dataset, and its `stat=name` options name the
columns — alias every aggregate to the SAS output variable name exactly, so a
later step reading `mean_amt` still finds it.

## [when: proc:transpose] PROC TRANSPOSE
Long-to-wide maps to conditional aggregation (`MAX(CASE WHEN key = 'x' THEN
value END)`) or Spark's `PIVOT` clause; wide-to-long maps to `STACK(...)` or
a `UNION ALL` of column selections. Preserve the `ID`, `VAR`, and `BY`
roles: `BY` -> `GROUP BY`, `ID` -> the pivot key, `VAR` -> the pivoted
value.
