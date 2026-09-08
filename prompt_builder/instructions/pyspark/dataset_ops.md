## [when: proc:sort] [kind: PROC_STEP] [lang: pyspark] PROC SORT
A `PROC SORT` produces a named dataset, so give it a named DataFrame and — when
the SAS wrote to `OUT=` — register that name, so the SAS dataset still exists as
something later steps and a reviewer can refer to:

```python
txns_ord = txns.orderBy(F.col("cust_id").asc(), F.col("txn_dt").desc())
txns_ord.createOrReplaceTempView("txns_ord")
```

- `BY DESCENDING v` flips *that* column only. `by a descending b` is
  `.orderBy(F.col("a").asc(), F.col("b").desc())`.
- **Do not invent a row-order column.** `F.monotonically_increasing_id()` is not
  a row number, is not stable across runs, and does not preserve SAS
  observation order. The named ordered DataFrame is the mechanism; where a later
  step needs a position, take it from `F.row_number()` over the same keys.
- ⚠️ **`orderBy` orders that DataFrame, not what you build from it.** A later
  `groupBy`, `join`, `union` or `repartition` discards the order. Restate it in
  the `Window.orderBy` — or the final write/display — that actually depends on
  it. The sorted DataFrame records the intent; the window enforces it.
- ⚠️ `sortWithinPartitions` is a shuffle-avoidance tuning knob, not a
  translation of PROC SORT: it orders each partition independently and gives no
  global order. Never substitute it for `orderBy`.
- ⚠️ **In-place sort (no `OUT=`)**: rebinding the same Python name is fine, but
  if the SAS replaced a *stored* dataset the equivalent is an overwrite of that
  table — state which you did.

Ordering semantics to preserve:

- **Missing values.** SAS sorts a missing numeric below every number, and
  PySpark's defaults already match: `.asc()` is `asc_nulls_first`, `.desc()` is
  `desc_nulls_last`. ⚠️ Do not reach for `asc_nulls_last()` unless the SAS asked
  for it — under `NODUPKEY` it changes which row survives.
- **Stability.** `EQUALS` is the default, so SAS keeps tied rows in input order.
  Spark offers no such guarantee and has no input order to keep: add the
  tiebreaker where ties decide the outcome rather than assuming one.
- **Collation.** SAS's default on Windows and UNIX is ASCII, which Spark's
  binary string comparison reproduces. ⚠️ `SORTSEQ=LINGUISTIC`, `EBCDIC` and the
  national collating sequences do not — flag them rather than emitting a binary
  sort in their place.
- `THREADS`, `SORTSIZE=`, `TAGSORT`, `PRESORTED`, `FORCE`, `OVERWRITE`,
  `DATECOPY` tune the SAS sort, not its result: they translate to **nothing**.

**De-duplication.** `NODUPKEY` keeps the first row per BY key *in the sort order
just applied* — a window, not `dropDuplicates`:

```python
from pyspark.sql.window import Window

w = Window.partitionBy("cust_id").orderBy(F.col("txn_dt").desc())
latest = txns.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")
```

⚠️ `dropDuplicates(["cust_id"])` keeps an **arbitrary** row, not the first in
any order — it is not a translation of `NODUPKEY`. Reserve `.distinct()` for
`NODUPRECS`, and flag that SAS compares only *adjacent* rows there, so on
unsorted input it removes less than `distinct()` does. `DUPOUT=` is the same
window filtered `_rn > 1`; `NOUNIQUEKEY` keeps whole BY groups of two or more
rows — `F.count("*").over(Window.partitionBy(keys)) > 1` — and `UNIQUEOUT=` is
its complement.

## [when: proc:sort, proc:datasets, proc:spdo] [kind: PROC_STEP] [lang: pyspark] Physical layout: clusterBy, not OPTIMIZE
A stored sort order, an index (`INDEX CREATE`, `CREATE INDEX`), or a
`SORTEDBY=` is SAS stating **physical layout**. On Databricks that is liquid
clustering, declared on the table:

```python
(df.write.format("delta")
   .clusterBy("acct_id", "open_dt")
   .mode("overwrite")
   .saveAsTable("cat.mylib.accounts"))
```

For a table that already exists, change the keys in SQL:

```python
spark.sql("ALTER TABLE cat.mylib.accounts CLUSTER BY (acct_id, open_dt)")
spark.sql("OPTIMIZE FULL cat.mylib.accounts")   # once, after enabling/changing keys
```

- ⚠️ **Never `.partitionBy(...)` and never a ZORDER call** for this. Clustering
  replaces both, is incompatible with them on the same table, and can be
  redefined later without rewriting data.
- ⚠️ `clusterBy` can only be set on **create or overwrite**, never on
  `mode("append")`. To change keys while appending, use the `ALTER TABLE` form
  above, separately from the write.
- `.option("clusterByAuto", "true")` lets predictive optimization choose and
  adapt the keys — the better choice where the SAS sort keys are a guess about
  access patterns rather than a statement of them.
- ⚠️ **Clustering is not row order.** It colocates rows so scans skip files; it
  never makes a read come back sorted, and so can never satisfy a SAS ordering
  requirement. Order is `orderBy` plus the window that needs it; layout is
  `clusterBy`.
- Emit it only for **permanent tables**. A `work.*` sort is a temp view with no
  files to lay out.
- Emit **no** scheduled `OPTIMIZE` or `VACUUM`: predictive optimization runs
  them on Unity Catalog managed tables. Recurring maintenance is a table
  setting, not a line of translated SAS — say so under Risks.
