## [when: proc:sort] [kind: PROC_STEP] PROC SORT
A `PROC SORT` names a dataset, so translate it to a named object: a view whose
definition carries the `ORDER BY`. The step keeps its identity in the DAG, and
the sort keys stay written down once instead of being copied into every
consumer.

```sql
-- proc sort data=work.txns out=work.txns_ord; by cust_id descending txn_dt;
CREATE OR REPLACE TEMP VIEW txns_ord AS
SELECT * FROM txns ORDER BY cust_id, txn_dt DESC;
```

`work.*` gives a `TEMP VIEW`, a permanent libref a
`CREATE OR REPLACE VIEW <catalog>.mylib.txns_ord` — a view either way, since
PROC SORT adds no columns and computes nothing.

- `BY DESCENDING v` flips *that* column only: `by a descending b` is
  `ORDER BY a, b DESC`, never `ORDER BY a DESC, b DESC`.
- **Never invent a row-order column.** The ordered view is what preserves SAS
  observation order; a synthetic `_row_id` or `MONOTONICALLY_INCREASING_ID()` is
  neither reproducible nor equivalent. Where a later step needs the position,
  take it from a `ROW_NUMBER()` over the *same* keys this view orders by.
- ⚠️ **A view's `ORDER BY` orders that view, not its consumers.** An outer query
  that joins, groups or unions over it may return rows in any order. So wherever
  the order is load-bearing — a window frame, a `FIRST.`/`LAST.` emulation, a
  `LIMIT`, a report — restate it there. The view tells you what to write in that
  `ORDER BY`; it does not excuse it. A sort that merely feeds a following `BY`
  step is exactly this case: emit the view, and let the join or window state its
  own keys.
- ⚠️ **In-place sort (no `OUT=`)**: input and output are the same dataset. Never
  emit `CREATE OR REPLACE VIEW x AS SELECT * FROM x ORDER BY ...` — a view
  cannot select from itself. Name the ordered view differently and repoint the
  consumers, or use `CREATE OR REPLACE TABLE x AS SELECT ...` where the SAS
  genuinely replaced a stored dataset, and say which under Risks.

Ordering semantics to preserve:

- **Missing values.** SAS sorts a missing numeric below every number, and
  Spark's defaults already agree (`ASC` is `NULLS FIRST`, `DESC` is
  `NULLS LAST`). ⚠️ Do not add `NULLS LAST` to an ascending sort "to be safe" —
  that changes which rows come first, and under `NODUPKEY` which row survives.
- **Stability.** `EQUALS` is the default, so SAS is a *stable* sort: rows tied
  on the BY keys keep input order. Spark has neither stability nor an input
  order to keep, so where ties decide an outcome add the tiebreaker to the
  `ORDER BY` rather than assuming one. `NOEQUALS` says the SAS gave that
  guarantee up already.
- **Collation.** SAS's default on Windows and UNIX is ASCII, which Spark's
  binary `STRING` comparison reproduces. ⚠️ `SORTSEQ=LINGUISTIC`, a `SORTSEQ=`
  translation table, `EBCDIC` (the z/OS default), and
  `DANISH`/`SWEDISH`/`NATIONAL`/`REVERSE` do not — those need a `COLLATE`
  clause, so flag them rather than emitting a binary sort in their place.
- `THREADS`/`NOTHREADS`, `SORTSIZE=`, `TAGSORT`, `PRESORTED`, `FORCE`,
  `OVERWRITE`, `DATECOPY` tune the SAS sort, not its result. They translate to
  **nothing**: note the drop once under Risks and emit no substitute.

De-duplication is the part that changes the rows:

- **`NODUPKEY`** keeps the first row per `BY` key. That is a window dedup, not
  `DISTINCT`:
  ```sql
  CREATE OR REPLACE TEMP VIEW dedup AS
  SELECT * FROM txns
  QUALIFY ROW_NUMBER() OVER (PARTITION BY cust_id ORDER BY load_dt) = 1;
  ```
  ⚠️ "First" is only defined once you say by what. SAS takes the first row in
  the sort order it just applied; the `ORDER BY` inside the window must
  reproduce that order exactly, and if the SAS sort keys do not break ties the
  choice is arbitrary in both systems — state that in Risks.
- **`NODUPRECS`** (or `NODUP`) removes adjacent rows identical across *all*
  columns -> `SELECT DISTINCT *`. ⚠️ It compares only *adjacent* rows in SAS,
  so on unsorted input it removes less than `DISTINCT` does. Flag the
  difference rather than assuming they agree.
- `DUPOUT=` names a dataset of the removed rows: the same window with
  `QUALIFY ROW_NUMBER() OVER (...) > 1`.
- **`NOUNIQUEKEY`** (alias `NOUNIKEY`) is the opposite operation, and is often
  mistranslated as a dedup. It drops every BY group holding **exactly one** row
  and keeps the surviving groups *whole* — BY-group integrity, not one row per
  key. That is a count, not a row number:
  `QUALIFY COUNT(*) OVER (PARTITION BY cust_id) > 1`. `UNIQUEOUT=` is the
  complement (`= 1`). ⚠️ It cannot be combined with `NODUPKEY`, and `DUPOUT=`
  pairs only with `NODUPKEY` — re-read any step that appears to mix them.

Where de-duplication is present the sort chooses the surviving row, so fold the
`ORDER BY` into the window rather than ordering a view and de-duplicating
separately: **one** statement for the step.

⚠️ **Preserve de-duplication; never invent or remove it.** Keep every
`DISTINCT` the SAS specifies, and add none it does not. Where a `DISTINCT`
looks redundant because a key already guarantees uniqueness, say so under
Risks and leave it in place — a wrong uniqueness assumption silently changes
the row count, and the SAS output is the reference.

## [when: proc:append, statement:set_multi, statement:dataset_option] Stacking, appending, and dataset options
`PROC APPEND BASE=a DATA=b` and a DATA step's `SET a b;` both concatenate ->
`SELECT ... FROM a UNION ALL SELECT ... FROM b`. Use `UNION ALL`, never
`UNION`: plain `UNION` de-duplicates, which SAS does not.

⚠️ Spark's `UNION ALL` matches columns **by position**; SAS matches them **by
name** and fills a column missing from one input with missing values. So list
the columns explicitly in each branch, in one agreed order, adding
`CAST(NULL AS <type>) AS missing_col` where an input lacks one. A positional
`SELECT *` union across differently-shaped inputs is a silent column swap.
(`UNION ALL BY NAME` matches by name where your Spark version has it.)

Dataset options become parts of the `SELECT`:
- `KEEP=`/`DROP=` -> the select list (or `SELECT * EXCEPT (...)`).
- `RENAME=(old=new)` -> `old AS new`.
- `WHERE=` -> a `WHERE` clause; `OBS=`/`FIRSTOBS=` -> `LIMIT` (⚠️ meaningless
  without an `ORDER BY`, since Spark has no inherent row order).
- `IN=` sets a flag for which input a row came from. In a join, that is
  `b.key IS NOT NULL`; in a `UNION ALL`, add a literal
  `'a' AS source` to each branch.
