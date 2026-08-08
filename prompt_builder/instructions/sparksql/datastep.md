## [kind: DATA_STEP] Sequential IF blocks that assign the same variable
A DATA step executes top to bottom per row, so several `IF`s assigning one
variable are **successive overwrites — the last one that fires wins**:

```sas
data work.scored;
  set work.acct;
  tier = 'C';
  if balance > 1000 then tier = 'B';
  if balance > 5000 then tier = 'A';   /* wins for balance > 5000 */
run;
```

Collapse them into a single `CASE`, and ⚠️ **reverse the order**: the *last*
SAS block becomes the *first* `WHEN`, because `CASE` stops at its first match
while SAS kept going and let the last write stand.

```sql
CREATE OR REPLACE TEMP VIEW scored AS
SELECT acct.*,
       CASE WHEN balance > 5000 THEN 'A'    -- last IF, so highest priority
            WHEN balance > 1000 THEN 'B'
            ELSE 'C'                        -- the unconditional default
       END AS tier
FROM acct;
```

⚠️ **The reversal is only valid when the blocks are independent.** Check every
later condition: if one *reads* a variable an earlier block *wrote* — including
the target itself, as in `if tier = 'B' then ...` — the blocks are a dependent
chain, not a priority list, and flattening them changes the answer. Keep those
sequential, as nested `CASE`s or successive CTEs, one per stage. State which
form you applied and why.

## [when: statement:retain] RETAIN and accumulators
A `RETAIN`ed variable keeps its value across rows, and `x + expr;` (the sum
statement) is an implicit retained accumulator that also ignores missing
values. Both are running aggregates -> a window function with an explicit
frame: `SUM(x) OVER (PARTITION BY grp ORDER BY seq ROWS BETWEEN UNBOUNDED
PRECEDING AND CURRENT ROW)`. ⚠️ Spark's default window frame changes with the
`ORDER BY`, so state the frame rather than relying on it, and name the
ordering column the SAS relied on — if the DATA step depended on physical
input order with no sort key, there is no faithful translation; say so.

## [when: statement:subsetting_if, statement:where, statement:output] Subsetting IF, WHERE, and OUTPUT
- A **subsetting `IF`** (`if region = 'N';` with no `then`) drops rows -> a
  `WHERE` clause. ⚠️ It filters *after* the assignments above it, so a
  computed column may be referenced; `WHERE` cannot see select-list aliases,
  so wrap the computation in a CTE and filter outside it.
- `WHERE` in SAS filters at read time, before the PDV is built, so it cannot
  see computed columns — that one maps to a plain `WHERE`.
- An explicit **`OUTPUT`** writes a row at that point. Several `OUTPUT`s in
  one step emit several rows per input row -> `UNION ALL` of the branches, or
  `explode` over an array built per row. `OUTPUT a; OUTPUT b;` writing to
  *different* datasets becomes one view per dataset, each with the branch's
  own filter.
- `_N_` is the iteration counter -> `ROW_NUMBER() OVER (ORDER BY ...)`, which
  needs an explicit order; `END=eof` marks the last row -> compare
  `ROW_NUMBER()` against `COUNT(*) OVER ()`.

## [when: statement:array, statement:do] ARRAY and DO loops
An `ARRAY` names a set of existing columns and a `DO i = 1 TO n` loop walks
them — together they are a wide-to-long reshape. Map to `STACK(...)` or a
`UNION ALL` of the column selections, apply the loop body once against the
long form, and pivot back with conditional aggregation if the step's output is
still wide. A `DO WHILE`/`DO UNTIL` whose trip count depends on data has no
set-based equivalent — flag it rather than guessing an iteration count.
