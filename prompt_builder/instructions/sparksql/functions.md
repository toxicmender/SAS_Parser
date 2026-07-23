SAS DATA-step and macro-time functions map to Spark SQL **built-in functions**
(see the [Spark SQL functions
reference](https://spark.apache.org/docs/latest/api/sql/index.html)). Reach for
a built-in before hand-rolling a `CASE`/regex; most SAS scalar functions have a
direct Spark equivalent. The gotchas below are where the *semantics* differ —
translate the intent, not the name. Date/time functions (`INTNX`, `INTCK`,
`PUT`/`INPUT` with date formats, `TODAY`, …) are covered separately in the
datetime guidance; this section is string, numeric, and null/conditional
functions.

## [when: function:coalesce, function:coalescec] COALESCE
`COALESCE(a, b, …)` (and character `COALESCEC`) both map directly to Spark
`coalesce(a, b, …)` — first non-null wins. Once SAS missing values have become
`NULL` this is exact. `IFNULL(a, b)` / `NVL(a, b)` are the two-argument form.

## [when: function:substr] SUBSTR
`SUBSTR(s, pos, len)` -> `substr(s, pos, len)`; both are **1-based** and `len`
is optional. ⚠️ SAS `SUBSTR` on the left of an assignment (the *SUBSTR
pseudo-function*, replacing characters in place) has no Spark equivalent —
re-express it as a `CONCAT` of the kept pieces (`concat(substr(s,1,p-1), new,
substr(s,p+length(new)))`).

## [when: function:scan] SCAN
`SCAN(s, n, delims)` returns the n-th word. Map to `split_part(s, delim, n)`
(1-based) for a single delimiter, or `element_at(split(s, regex), n)` when SAS
used several delimiters (`split` takes a **regex**, so escape/`[]`-group the
delimiter set). ⚠️ SAS `SCAN` treats runs of delimiters as one and accepts a
**negative n** to count from the end (`element_at(..., n)` with negative n does
this in Spark); a naive `split_part` does neither. Default SAS delimiters are a
large punctuation set, not just space — reproduce the exact set.

## [when: function:index, function:find] INDEX / FIND
`INDEX(s, sub)` -> `instr(s, sub)` or `locate(sub, s)` — position of the first
match, **1-based, 0 when absent** (same convention as SAS). `FIND(s, sub, start)`
with a start position -> `locate(sub, s, start)`. Case-insensitive `FIND(...,
'i')` -> `locate(lower(sub), lower(s))`.

## [when: function:tranwrd] TRANWRD
`TRANWRD(s, from, to)` is a **literal** substring replace -> `replace(s, from,
to)`. Use `regexp_replace(s, pattern, repl)` only when the SAS source clearly
intended a pattern. `TRANSLATE(s, to, from)` (character-for-character) ->
Spark `translate(s, from, to)` — note Spark's argument order is `(input, from,
to)`, the reverse of SAS.

## [when: function:trim, function:strip, function:left, function:compress] Trimming and stripping
- `STRIP(s)` (both ends) -> `trim(s)`.
- `TRIM(s)` (trailing blanks only) -> `rtrim(s)`.
- `LEFT(s)` (left-align, drop leading blanks) -> `ltrim(s)`.
- `COMPRESS(s)` (remove all blanks) -> `replace(s, ' ', '')`;
  `COMPRESS(s, chars)` -> `regexp_replace(s, '[chars]', '')` (regex-escape the
  set); `COMPRESS(s, chars, 'k')` (keep-only) -> `regexp_replace(s,
  '[^chars]', '')`. ⚠️ Spark `STRING` has no fixed width, so SAS blank-padding
  differences mostly disappear — but check any comparison that relied on
  trailing blanks.

## [when: function:cat, function:cats, function:catx, function:catt] CAT family
`CAT`/`CATT`/`CATS`/`CATX` concatenate. `CATX(sep, a, b, …)` -> `concat_ws(sep,
a, b, …)`; `CATS(a, b, …)` (strip each, no separator) -> `concat(trim(a),
trim(b), …)`. ⚠️ SAS treats a missing value as an empty string when
concatenating, and `concat_ws` **skips nulls** (close), but bare `concat`
returns `NULL` if **any** argument is null — wrap each with `coalesce(x, '')`
to reproduce SAS behaviour.

## [when: function:upcase, function:lowcase, function:propcase] Case functions
`UPCASE` -> `upper`, `LOWCASE` -> `lower`, `PROPCASE` -> `initcap`. `initcap`
capitalises after every whitespace run, which matches `PROPCASE`'s default word
break; if the SAS call passed custom delimiters, reproduce them explicitly.

## [when: function:length, function:lengthn, function:lengthc] LENGTH
⚠️ SAS `LENGTH(s)` ignores trailing blanks **and returns 1 for a blank/missing
string** -> `greatest(length(rtrim(s)), 1)` when that edge case matters.
`LENGTHN(s)` returns 0 for blank -> `length(rtrim(s))`. `LENGTHC(s)` (with
trailing blanks) -> `length(s)`. Because Spark `STRING` is not blank-padded,
prefer the `LENGTHN` semantics unless the SAS code depended on the padded width.

## [when: function:put, function:input] PUT / INPUT (non-date)
For **non-date** conversions: numeric-to-character `PUT(n, best12.)` -> `cast(n
AS STRING)` (or `format_string('%d', n)` / `format_number(n, d)` for a specific
width/decimals); character-to-numeric `INPUT(s, 8.)` -> `cast(s AS DOUBLE)` or
`try_cast(s AS DOUBLE)` when a non-numeric string should yield `NULL` rather
than error. Date/time formats and informats are handled in the datetime
guidance — do not treat those as plain casts.

## [when: function:round, function:ceil, function:floor, function:int] Rounding and truncation
- `ROUND(x)` -> `round(x)`; `ROUND(x, u)` rounds to the nearest multiple of the
  **rounding unit** `u`, which is *not* the same as Spark `round(x, d)`
  (decimal places). `ROUND(x, 0.01)` -> `round(x, 2)` only because `0.01` is a
  power of ten; for a general unit use `round(x / u) * u`. ⚠️ Flag any
  `ROUND` whose second argument is not a power of ten.
- `CEIL(x)` -> `ceil(x)`, `FLOOR(x)` -> `floor(x)`.
- `INT(x)` truncates **toward zero** -> `cast(x AS BIGINT)` (also toward zero),
  *not* `floor` (which differs for negatives).

## [when: function:mod] MOD
`MOD(a, b)` -> `mod(a, b)` (or the `%` operator); both take the sign of the
dividend, so they agree. Use Spark `pmod(a, b)` only if you actually want a
non-negative result — that is *not* SAS `MOD` behaviour.

## [when: function:abs, function:max, function:min] ABS / MAX / MIN (across arguments)
`ABS(x)` -> `abs(x)`. Multi-argument `MAX(a, b, …)` / `MIN(a, b, …)` (across
columns, not the aggregate) -> `greatest(a, b, …)` / `least(a, b, …)`; like SAS
these skip `NULL`/missing and return `NULL` only when every argument is null. Do
**not** use the aggregate `MAX`/`MIN` (those reduce over rows).

## [when: function:sum, function:mean, function:n, function:nmiss] SUM / MEAN across arguments
⚠️ `SUM(a, b, c)` / `MEAN(a, b, c)` across variables **ignore missing values**,
but Spark arithmetic `a + b + c` yields `NULL` if any operand is null. Reproduce
SAS by summing `coalesce(x, 0)` — but note this makes an all-missing row `0`
where SAS returns missing, so guard with a `CASE` on the non-null count if that
distinction matters. `N(of a b c)` -> the non-null count; `NMISS(of a b c)` ->
the null count (`SUM(CASE WHEN x IS NULL THEN 1 ELSE 0 END)` summed per column).

## [when: function:missing] MISSING
`MISSING(x)` -> `x IS NULL` (numeric) once missing values map to `NULL`; for a
character argument also treat a blank as missing: `(x IS NULL OR trim(x) =
'')`.

## [when: function:ifn, function:ifc] IFN / IFC
`IFN(cond, t, f)` / `IFC(cond, t, f)` -> Spark `if(cond, t, f)` or `CASE WHEN
cond THEN t ELSE f END`. The optional third-missing argument of `IFN(cond, t,
f, m)` (value when `cond` is missing) becomes a nested `CASE` on `cond IS
NULL`.

## [when: function:lag, function:dif] LAG / DIF
⚠️ `LAG(x)` / `DIF(x)` are **not** simply "previous row". SAS `LAGn` reads from a
FIFO queue updated only when the function is *executed*, so under conditional
execution it does not line up with the previous observation. When the intent is
genuinely the prior row in an order, map to a window: `LAG(x) OVER (ORDER BY
…)` and `x - LAG(x) OVER (ORDER BY …)` for `DIF`. State the assumed ordering
and flag the queue-vs-row-offset gap when the SAS `LAG` sat inside an `IF`.
