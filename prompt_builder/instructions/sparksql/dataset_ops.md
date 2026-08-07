## [when: proc:sort] [kind: PROC_STEP] PROC SORT
A bare `PROC SORT` with no `NODUPKEY`/`NODUPRECS` only orders rows. Spark is
unordered, so a sort that merely feeds the next step usually **translates to
nothing** — carry the intent into the consuming query's `ORDER BY` (or the
`ORDER BY` of a window) instead of emitting a standalone sorted view, which
Spark is free to reorder anyway.

De-duplication is the part that carries meaning:

- **`NODUPKEY`** keeps the first row per `BY` key. That is a window dedup, not
  `DISTINCT`:
  ```sql
  CREATE OR REPLACE TEMP VIEW dedup AS
  SELECT * EXCEPT (rn) FROM (
    SELECT t.*, ROW_NUMBER() OVER (PARTITION BY cust_id ORDER BY load_dt) AS rn
    FROM txns t
  ) WHERE rn = 1;
  ```
  ⚠️ "First" is only defined once you say by what. SAS takes the first row in
  the sort order it just applied; the `ORDER BY` inside the window must
  reproduce that order exactly, and if the SAS sort keys do not break ties the
  choice is arbitrary in both systems — state that in Risks.
- **`NODUPRECS`** (or `NODUP`) removes adjacent rows identical across *all*
  columns -> `SELECT DISTINCT *`. ⚠️ It compares only *adjacent* rows in SAS,
  so on unsorted input it removes less than `DISTINCT` does. Flag the
  difference rather than assuming they agree.
- `DUPOUT=` names a dataset of the removed rows: `WHERE rn > 1` over the same
  window.

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
