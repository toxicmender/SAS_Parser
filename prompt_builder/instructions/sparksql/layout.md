## [when: proc:sort, proc:datasets, proc:spdo] [kind: PROC_STEP] Physical layout: CLUSTER BY, not OPTIMIZE
SAS says "make this dataset fast to reach by these columns" three ways — a
stored sort order, an index, or an SPD Server layout directive. All three are
**physical layout**, and on Databricks layout is declared once, on the table,
as liquid clustering.

| SAS | Databricks SQL |
|---|---|
| `PROC SORT DATA=lib.t; BY a b;` (in place, permanent libref) | `ALTER TABLE cat.lib.t CLUSTER BY (a, b);` |
| `PROC SORT DATA=lib.s OUT=lib.t; BY a b;` | `CREATE OR REPLACE TABLE cat.lib.t CLUSTER BY (a, b) AS SELECT * FROM cat.lib.s;` |
| `MODIFY t; INDEX CREATE a;` / `CREATE INDEX i ON t(a);` | `ALTER TABLE cat.lib.t CLUSTER BY (a);` |
| `MODIFY t (SORTEDBY=a b);` | `ALTER TABLE cat.lib.t CLUSTER BY (a, b);` |
| `INDEX DELETE a;` / `SORTEDBY=_NULL_` | `ALTER TABLE cat.lib.t CLUSTER BY NONE;` |

⚠️ **Never emit `OPTIMIZE ... ZORDER BY`, `PARTITIONED BY`, or a bucketing
clause for this.** Clustering replaces all three, is incompatible with them on
the same table, and unlike them can be redefined later without rewriting data.
A translation reaching for `ZORDER BY (a)` should have written `CLUSTER BY (a)`.

`CLUSTER BY` binds to the **table**, so on a CTAS it goes after the table name
and before `AS` — never inside the `SELECT`:

```sql
CREATE TABLE cat.lib.t (acct_id BIGINT, open_dt DATE) CLUSTER BY (acct_id);
CREATE TABLE cat.lib.t CLUSTER BY (acct_id) AS SELECT * FROM cat.lib.s;
ALTER TABLE cat.lib.t CLUSTER BY (acct_id, open_dt);   -- existing, unpartitioned
```

**OPTIMIZE is maintenance, not translation.** Predictive optimization runs
`OPTIMIZE`, `ANALYZE` and `VACUUM` on Unity Catalog managed tables by itself.
Emit `OPTIMIZE <table> FULL;` **once**, right after first enabling clustering or
changing keys — declaring keys does not rewrite rows already written, and plain
`OPTIMIZE` is incremental so it will not re-cluster them. ⚠️ Emit no scheduled
or repeated `OPTIMIZE`/`VACUUM`: that is a maintenance job smuggled into an ETL
script. Say under Risks that upkeep is a table setting and leave it there.

⚠️ **`CLUSTER BY` is not row order.** It colocates related rows so scans skip
files; `SELECT * FROM t` still returns rows in no guaranteed order, so it can
never satisfy a SAS ordering requirement. Keep the two apart: the SAS **order**
becomes an ordered view plus the consuming `ORDER BY` (see the PROC SORT
guidance); the SAS **layout** becomes `CLUSTER BY`.

When to emit it at all:

- Only for **permanent, stored tables** — a `work.*` sort is a temp view, and a
  view has no files to lay out.
- Only where the SAS said something durable about access: an index, a
  `SORTEDBY=`, or an in-place sort of a permanent dataset that later steps
  filter or join on. A one-off sort feeding the next step is not layout intent;
  do not manufacture clustering keys the SAS never stated.
- `CLUSTER BY AUTO` (Unity Catalog managed tables, predictive optimization on)
  lets Databricks pick keys from the real query workload and adapt as it
  changes — prefer it where the SAS sort keys read as a guess about how the
  data gets queried rather than a statement of it, and note the choice.

⚠️ Constraints to flag rather than paper over: keys must be columns carrying
statistics, and clustering keys support `DATE`, `TIMESTAMP`, `TIMESTAMP_NTZ`,
`STRING`, the integer family, `FLOAT`/`DOUBLE`/`DECIMAL` — a struct *field*
works by dot notation, but `STRUCT`/`MAP`/`ARRAY` themselves do not, so a SAS
index over one has no direct translation. Clustering also raises the table to
Delta writer version 7 / reader version 3, irreversibly: non-Databricks Delta
clients may lose read access. And an index never enforced anything — where the
migration needs the uniqueness of `INDEX CREATE ... /UNIQUE`, say that
clustering does not provide it.
