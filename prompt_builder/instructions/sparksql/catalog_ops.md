## [when: proc:datasets, proc:delete, proc:copy] [kind: PROC_STEP] Library management
These PROCs manage the library rather than compute anything, so they translate
to **DDL** — the one place a translation legitimately emits destructive
statements. Flag each under Risks: a reviewer should see a `DROP` before it
runs, not after.

| SAS | Databricks SQL |
|---|---|
| `PROC DATASETS; DELETE a b;` / `PROC DELETE DATA=a;` | `DROP TABLE IF EXISTS a;` |
| `CHANGE old=new;` | `ALTER TABLE old RENAME TO new;` |
| `MODIFY t; RENAME v=w;` | `ALTER TABLE t RENAME COLUMN v TO w;` |
| `MODIFY t (LABEL='...');` | `COMMENT ON TABLE t IS '...';` |
| `APPEND BASE=a DATA=b;` | `INSERT INTO a SELECT * FROM b;` |
| `PROC COPY IN=x OUT=y;` | `CREATE TABLE y.t DEEP CLONE x.t;` |

⚠️ `PROC DATASETS KILL` deletes **every** member of the library. Do not
translate it to a generated list of `DROP` statements — emit the
non-convertible marker and make a human decide, because the SAS meaning
("whatever happens to be there") is not something a static translation can
safely enumerate.

`PROC COPY` is a real copy of the data, so `DEEP CLONE` is the faithful mapping
— it copies the files and the history. `SHALLOW CLONE` is a metadata-only
pointer and is *not* equivalent: dropping the source breaks it. Use
`CREATE TABLE ... AS SELECT` where only the current rows are wanted and the
history is not.

## [when: proc:contents] [kind: PROC_STEP] PROC CONTENTS
Describes a dataset's structure. `OUT=` writes that description as data —
column name, type, length, label, format, position — which downstream steps
sometimes read to drive dynamic code.

Query the catalog rather than the data: `DESCRIBE TABLE EXTENDED t` for a
one-off look, or `information_schema.columns` when the result is consumed:

```sql
CREATE OR REPLACE TEMP VIEW contents_out AS
SELECT column_name AS name, data_type AS type, ordinal_position AS varnum,
       comment AS label
FROM main.mylib.information_schema.columns
WHERE table_name = 'accounts';
```

⚠️ The SAS columns do not survive one-to-one: `LENGTH` is storage width, which
a `STRING` does not have; `FORMAT`/`INFORMAT` are display rules with no
catalog equivalent (see the metadata guidance); `TYPE` is SAS's 1/2 numeric
code, not a type name. Where a later step *branches* on one of these, that
logic needs rethinking rather than translating — flag it.
