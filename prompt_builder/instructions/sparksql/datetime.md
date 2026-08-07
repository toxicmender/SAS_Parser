## [category: date_time] Date and datetime epoch
⚠️ SAS counts from **1960-01-01**; Spark SQL `DATE`/`TIMESTAMP` count from
1970-01-01. The offset is 3653 days, or 315 619 200 seconds. A raw SAS numeric
cast directly to a date is wrong by exactly that.
- SAS **date** (days) -> `DATE_ADD(DATE '1960-01-01', CAST(sas_num AS INT))`.
- SAS **datetime** (seconds) -> `TIMESTAMP_SECONDS(sas_dt - 315619200)`. One
  function, no interval arithmetic.
- SAS **time** (seconds since midnight) -> `MAKE_INTERVAL(0,0,0,0,0,0, sas_t)`
  added to a date, or format it directly.

⚠️ **Timezone.** A SAS datetime is naive — it carries no zone. Spark's
`TIMESTAMP` is `TIMESTAMP_LTZ`: it is *interpreted* in
`spark.sql.session.timeZone`, so the same stored value reads differently under
a different session setting, and a round trip can shift the wall-clock time.
When the SAS value is a wall-clock reading rather than an instant, target
`TIMESTAMP_NTZ` (`CAST(... AS TIMESTAMP_NTZ)`) to preserve it exactly. State
the epoch *and* the timezone assumption in Mapping.

## [category: date_time] Date literals and date parts
**Literals.** A SAS date constant is a quoted string with a type suffix; it is
not a string in Spark:
`'01JAN2020'd` -> `DATE '2020-01-01'`,
`'01JAN2020:09:30:00'dt` -> `TIMESTAMP '2020-01-01 09:30:00'`,
`'09:30:00't` -> a time value or an interval, depending on use. Never leave
the SAS spelling quoted — `'01JAN2020'` compared against a `DATE` column is a
string-vs-date comparison that ANSI mode rejects outright.

**Extractors** are one-to-one: `YEAR`/`MONTH`/`DAY`/`HOUR`/`MINUTE`/`SECOND`
-> `year`/`month`/`day`/`hour`/`minute`/`second`, `QTR` -> `quarter`,
`DATEPART(dt)` -> `CAST(dt AS DATE)`, `TIMEPART(dt)` -> `DATE_FORMAT(dt,
'HH:mm:ss')`. Constructors: `MDY(m,d,y)` -> `make_date(y, m, d)` (⚠️ **reversed
argument order**), `DHMS(d,h,m,s)` -> `make_timestamp(...)` or the epoch
arithmetic above, `HMS(h,m,s)` -> `make_interval(0,0,0,0,h,m,s)`.

⚠️ **`WEEKDAY` maps to `dayofweek`, not `weekday`.** SAS `WEEKDAY()` returns
1=Sunday … 7=Saturday. Spark's `dayofweek` uses exactly that numbering;
Spark's similarly-named `weekday` returns 0=Monday … 6=Sunday. Reaching for
the name-alike shifts every value and changes which rows a weekend filter
selects.

⚠️ `HOLIDAY`, `HOLIDAYCK`, and `HOLIDAYCOUNT` read SAS's built-in holiday
calendar, which **has no Spark equivalent**. Emit the non-convertible marker
and translate against an explicit calendar/holiday lookup table joined into
the query — never approximate a holiday as "not a weekend".

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

⚠️ **`TRUNC` is date-only.** `TRUNC(d, 'MM')` takes a `DATE`. For a
`TIMESTAMP` the function is `DATE_TRUNC('MONTH', ts)` — a different name *and*
the **reverse argument order** (unit first). Mixing them up is a silent type
error at best.

## [when: function:intck] INTCK -> date difference
`INTCK('interval', from, to)` counts interval *boundaries* crossed, not
elapsed units. ⚠️ Because it counts boundaries, `INTCK('MONTH','31JAN','01FEB')`
is 1, not ~0 — never substitute a bare `MONTHS_BETWEEN`.
- `INTCK('DAY', a, b)` -> `DATEDIFF(b, a)`.
- `INTCK('MONTH', a, b)` -> `CAST(MONTHS_BETWEEN(TRUNC(b,'MM'),
  TRUNC(a,'MM')) AS INT)`. ⚠️ `MONTHS_BETWEEN` returns a **DOUBLE**; without
  the cast the result is `1.0`, not `1`, and any downstream integer comparison
  or join key drifts.
- `INTCK('YEAR', a, b)` -> `YEAR(b) - YEAR(a)`.

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

⚠️ **Parsing is strict.** Since Spark 3.0 these use the Java
`DateTimeFormatter`, which *raises* on text that does not match the pattern;
SAS `INPUT` yields a missing value and a log note. Use `TRY_TO_DATE` /
`TRY_TO_TIMESTAMP` whenever the source text is not provably clean — that is
the faithful translation of SAS's behaviour, and it keeps one bad row from
failing the job. Note also that pattern `yyyy` means *year-of-era* and `uuuu`
means *proleptic year*; they differ only before 1 CE, but `uuuu` is the safer
letter when a pattern is used for both parsing and formatting.
