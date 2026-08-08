## [when: proc:freq] [kind: PROC_STEP] PROC FREQ
A `TABLES` statement is a `GROUP BY` with `COUNT(*)`. One variable is a
one-way frequency; `a*b` is a cross-tabulation, which becomes `GROUP BY a, b`
— produce it in long form (one row per cell) rather than pivoting into a
matrix, unless the SAS output is consumed as a matrix.

```sql
CREATE OR REPLACE TEMP VIEW freq_region AS
SELECT region, COUNT(*) AS count
FROM customers
GROUP BY region
ORDER BY region;
```

`OUT=` / `OUTPUT OUT=` names a dataset: emit it as a view. SAS writes the
count into a column named `COUNT` and, for `OUT=`, a `PERCENT` column —
alias to those names when a later step reads them.

⚠️ **Emit only what the SAS asked for.** `PROC FREQ` prints percentages,
cumulative counts, and row/column percentages by default *in its report*, but
a translation targets the data, not the printout. Do not add `PERCENT`,
cumulative, or `NOPERCENT`-suppressed columns unless the SAS requests them via
`OUT=`/`OUTPUT` or an option. Likewise, do not generate `CHISQ`, `FISHER`, or
any other statistical test unless the SAS source names it — those are not
expressible as a `GROUP BY` and, where genuinely requested, warrant the
non-convertible marker.

⚠️ **Missing values.** `PROC FREQ` **excludes** missing values from its
frequency table unless `MISSING` is specified, whereas `GROUP BY` in Spark
treats `NULL` as its own group. Without the `MISSING` option, add an explicit
`WHERE col IS NOT NULL` to match SAS; with it, the plain `GROUP BY` is right.
This changes both the row count and the totals, so state which you applied.

`ORDER=FREQ` maps to `ORDER BY count DESC`; the default `ORDER=INTERNAL` is
`ORDER BY` the variable itself.
