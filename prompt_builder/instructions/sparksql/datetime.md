## Date and datetime epoch
⚠️ SAS date values count days from 1960-01-01; Spark SQL `DATE`/`TIMESTAMP`
count from 1970-01-01. A raw SAS numeric date is off by 3653 days if cast
directly. When the source column is a true SAS date, convert with
`DATE_ADD(DATE'1960-01-01', CAST(sas_num AS INT))` rather than casting the
number to a date. SAS datetime values are *seconds* since 1960-01-01
00:00:00; convert with `TIMESTAMP'1960-01-01 00:00:00' + MAKE_INTERVAL(0,0,
0,0,0,0, sas_dt_seconds)` or arithmetic on `unix_timestamp`. State the epoch
assumption in Mapping.

## [when: function:intnx] INTNX -> date arithmetic
`INTNX('interval', start, n, 'alignment')` advances a date/datetime by `n`
intervals. Map the interval:
- `MONTH`/`QTR`/`YEAR` -> `ADD_MONTHS(start, n * k)` (k = 1/3/12), which is
  the correct end-of-month-aware shift.
- `DAY`/`WEEKDAY` -> `DATE_ADD(start, n)` (weekday needs extra filtering).
- `WEEK` -> `DATE_ADD(start, n * 7)`.
⚠️ SAS default alignment is `BEGINNING` (the first day of the resulting
interval), so `INTNX('MONTH', d, 1)` returns the first of next month, not
`d + 1 month`. Reproduce alignment explicitly with `TRUNC(..., 'MM')`,
`LAST_DAY(...)`, etc. Do not translate `INTNX` as a plain `DATE_ADD` of days.

## [when: function:intck] INTCK -> date difference
`INTCK('interval', from, to)` counts interval *boundaries* crossed, not
elapsed units. `INTCK('MONTH', a, b)` -> `MONTHS_BETWEEN(TRUNC(b,'MM'),
TRUNC(a,'MM'))` (integer), `INTCK('DAY', a, b)` -> `DATEDIFF(b, a)`,
`INTCK('YEAR', a, b)` -> difference of `YEAR()` parts. ⚠️ Because it counts
boundaries, `INTCK('MONTH','31JAN','01FEB')` is 1, not ~0 — do not
substitute a fractional `MONTHS_BETWEEN` without `TRUNC`.

## [when: function:today, function:date, function:datetime] Current date/time
`TODAY()`/`DATE()` -> `CURRENT_DATE()`; `DATETIME()` -> `CURRENT_TIMESTAMP()`;
`TIME()` -> `DATE_FORMAT(CURRENT_TIMESTAMP(), 'HH:mm:ss')` or the relevant
extraction. Note these are evaluated per query in Spark.

## [when: function:put, function:input] PUT/INPUT with date formats
`PUT(date, yymmddn8.)` and friends format a value to text; map to
`DATE_FORMAT(date, 'yyyyMMdd')` using Spark's datetime pattern letters
(`yyyy`, `MM`, `dd`, `HH`, `mm`, `ss`). `INPUT(str, yymmdd10.)` parses text
to a date; map to `TO_DATE(str, 'yyyy-MM-dd')` / `TO_TIMESTAMP(...)`. The
SAS informat/format name determines the pattern — translate the specific
width and layout, not a generic default.
