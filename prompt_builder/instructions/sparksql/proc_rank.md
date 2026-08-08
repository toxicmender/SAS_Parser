## [when: proc:rank] [kind: PROC_STEP] PROC RANK
Ranking within `BY` groups is a window function: `BY` -> `PARTITION BY`, the
`VAR` list -> one ranking expression per variable, `DESCENDING` -> `ORDER BY
... DESC`.

⚠️ **The default tie rule has no single-function equivalent.** SAS
`TIES=MEAN` — the default — gives tied rows the *average* of the ranks they
span, which no Spark ranking function does:

| SAS | Databricks SQL |
|---|---|
| `TIES=LOW` | `RANK()` |
| `TIES=DENSE` | `DENSE_RANK()` |
| `TIES=HIGH` | `COUNT(*) OVER (PARTITION BY g ORDER BY x)` |
| `TIES=MEAN` *(default)* | `AVG(rnk) OVER (PARTITION BY g, x)` over a `RANK()` computed in a CTE |

Reaching for `RANK()` on a step that never named `TIES=` silently changes every
tied row. If the data has no ties the two agree — say so if you assume it.

⚠️ **`GROUPS=n` is zero-based; `NTILE(n)` is one-based.** `GROUPS=10` produces
0–9, `NTILE(10)` produces 1–10, so the mapping is `NTILE(n) OVER (...) - 1`.
Getting this wrong shifts every decile by one and the result still looks
plausible.

`PERCENT` -> `PERCENT_RANK() OVER (...) * 100`; `FRACTION` ->
`PERCENT_RANK()`; `NORMAL=`/`SAVAGE`/`BLOM`/`TUKEY` are normal scores with no
built-in equivalent — emit the non-convertible marker rather than approximating
a distribution.

⚠️ **Without a `RANKS` statement, the ranks *replace* the analysis variables**
in the output dataset — the original values are gone. With one, they are added
under the new names. Reproduce whichever the SAS did; overwriting a value
column by accident is a silent data change.

## [when: proc:standard] [kind: PROC_STEP] PROC STANDARD
Standardises the `VAR` variables to a target mean and standard deviation
(`MEAN=0 STD=1` unless stated, i.e. a z-score), within `BY` groups:

```sql
CREATE OR REPLACE TEMP VIEW standardised AS
SELECT t.*,
       (amt - AVG(amt) OVER (PARTITION BY region))
         / NULLIF(STDDEV_SAMP(amt) OVER (PARTITION BY region), 0) AS amt_z
FROM txns t;
```

Map `MEAN=m STD=s` as `... * s + m` on top of that expression.

- ⚠️ SAS uses the **sample** standard deviation (n−1), so `STDDEV_SAMP` — which
  is what Spark's bare `STDDEV` is — never `STDDEV_POP`.
- ⚠️ `NULLIF(..., 0)` matters: a BY group with one row, or with no variation,
  has a zero standard deviation, and under ANSI mode dividing by it **raises**
  rather than yielding missing as SAS does.
- `REPLACE` fills missing values with the (group) mean instead of leaving them
  missing -> `COALESCE(amt, AVG(amt) OVER (PARTITION BY region))`. Without it,
  missing stays missing.
- Like `PROC RANK`, the standardised values **replace** the originals in the
  output unless the step says otherwise.
