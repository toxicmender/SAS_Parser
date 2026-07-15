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

## MERGE and BY-group joins
A SAS `DATA` step `MERGE a b; BY key;` is a full outer join on `key`, not an
inner join — unmatched rows from either side are kept. Translate to
`FULL OUTER JOIN ... USING (key)` (or `LEFT`/`INNER` only when `IF a` / `IF
a AND b` subsetting makes the intent narrower). ⚠️ SAS overwrites same-named
non-BY columns left-to-right (the last dataset wins); reproduce that with an
explicit `COALESCE(b.col, a.col)` or a chosen side, and state which side you
kept. A `MERGE` without `BY` is positional (one-to-one by row position) and
has no correct Spark SQL translation — flag it as unsafe rather than
guessing.

## BY-group processing (FIRST./LAST.)
`FIRST.var` / `LAST.var` flags map to window functions over
`PARTITION BY <by-vars> ORDER BY <by-vars>`: `FIRST.x` is
`ROW_NUMBER() = 1`, `LAST.x` is `ROW_NUMBER() OVER (... ORDER BY ... DESC) =
1` (or `COUNT(*) OVER (...)` compared to a running `ROW_NUMBER`). Retained
accumulators across a BY group become `SUM(...) OVER (PARTITION BY ... ORDER
BY ...)`.

## [when: proc:means, proc:summary] Summary statistics
`PROC MEANS` / `PROC SUMMARY` map to `GROUP BY` with aggregate functions
(`AVG`, `SUM`, `MIN`, `MAX`, `COUNT`, `STDDEV`). The `CLASS` variables are
the `GROUP BY` keys; `VAR` variables are the aggregated columns; a `TYPES` /
`WAYS` request for multiple subtotal levels maps to `GROUPING SETS`,
`ROLLUP`, or `CUBE`. Remember Spark aggregates skip `NULL`, so `N` vs
`NMISS` must be `COUNT(col)` vs `SUM(CASE WHEN col IS NULL THEN 1 ELSE 0
END)`.

## [when: proc:transpose] PROC TRANSPOSE
Long-to-wide maps to conditional aggregation (`MAX(CASE WHEN key = 'x' THEN
value END)`) or Spark's `PIVOT` clause; wide-to-long maps to `STACK(...)` or
a `UNION ALL` of column selections. Preserve the `ID`, `VAR`, and `BY`
roles: `BY` -> `GROUP BY`, `ID` -> the pivot key, `VAR` -> the pivoted
value.
