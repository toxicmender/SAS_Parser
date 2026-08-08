## [when: proc:univariate] [kind: PROC_STEP] PROC UNIVARIATE
Translate to an aggregate query over the `VAR` variables, grouped by the `BY`
variables. The distributional statistics map directly: `MEAN` -> `avg`, `STD`
-> `stddev` (sample; `stddev_pop` is the population form), `VAR` ->
`var_samp`, `MIN`/`MAX`/`N` -> `min`/`max`/`count`, `SUM` -> `sum`, `MEDIAN`
-> `median`, `MODE` -> `mode`, `SKEWNESS`/`KURTOSIS` -> `skewness`/`kurtosis`,
`NMISS` -> `SUM(CASE WHEN x IS NULL THEN 1 ELSE 0 END)`, `RANGE` ->
`max(x) - min(x)`.

Percentiles (`P1`, `Q1`, `P50`, `Q3`, `P99`, the `PCTLPTS=` list) ->
`percentile_approx(x, p)`. ⚠️ That function is **approximate**: its default
accuracy of 10000 is close but not exact, so a value compared against SAS
output can differ in the last places. Use `percentile(x, p)` when exactness
matters more than cost, and say which you chose.

```sql
CREATE OR REPLACE TEMP VIEW income_stats AS
SELECT region,
       COUNT(income)                        AS n,
       AVG(income)                          AS mean,
       STDDEV(income)                       AS std,
       MIN(income)                          AS min,
       percentile_approx(income, 0.25)      AS q1,
       percentile_approx(income, 0.50)      AS median,
       percentile_approx(income, 0.75)      AS q3,
       MAX(income)                          AS max
FROM applicants
GROUP BY region;
```

`OUTPUT OUT=` names the output dataset and its `stat=name` options name the
columns — alias each aggregate to the SAS output variable name exactly, since
a later step reads it by that name. With no `OUTPUT` statement the PROC only
prints, so the translation is whatever a downstream step actually consumes.

⚠️ Do not attempt the distribution machinery: `HISTOGRAM`, `QQPLOT`,
`PROBPLOT`, `CDFPLOT`, `NORMAL`-test output, and extreme-observation tables
have no Spark SQL equivalent. Where the SAS source genuinely depends on one,
emit the non-convertible marker rather than approximating it.
