## [when: proc:datasets, proc:delete, proc:copy] [kind: PROC_STEP] Library management
These PROCs manage the library rather than compute anything, so they translate
to **DDL** — the one place a translation legitimately emits destructive
statements. Flag each under Risks: a reviewer should see a `DROP` before it
runs, not after.

| SAS | Databricks SQL |
|---|---|
| `PROC DATASETS; DELETE a b;` / `PROC DELETE DATA=a;` | `DROP TABLE IF EXISTS IDENTIFIER(session.A);` |
| `CHANGE old=new;` | `ALTER TABLE ... RENAME TO ...`, through `EXECUTE IMMEDIATE` (below) |
| `MODIFY t; RENAME v=w;` | `ALTER TABLE IDENTIFIER(session.T) RENAME COLUMN v TO w;` |
| `MODIFY t (LABEL='...');` | `COMMENT ON TABLE IDENTIFIER(session.T) IS '...';` |
| `APPEND BASE=a DATA=b;` | `INSERT INTO IDENTIFIER(session.A) SELECT * FROM IDENTIFIER(session.B);` |
| `PROC COPY IN=x OUT=y;` | `CREATE TABLE ... DEEP CLONE ...` per member, through `EXECUTE IMMEDIATE` (below) |

Every table is its session variable (see the session-variable rule). The new
name of `RENAME TO` and a `CLONE` source take no `IDENTIFIER`, so build those
statements from the variables and run them:

```sql
DECLARE OR REPLACE VARIABLE STMT STRING;
SET VAR STMT = CONCAT('ALTER TABLE ', session.OLD, ' RENAME TO ', session.NEW);
EXECUTE IMMEDIATE STMT;
SET VAR STMT = CONCAT('CREATE TABLE ', session.Y_T, ' DEEP CLONE ', session.X_T);
EXECUTE IMMEDIATE STMT;
```

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

Query the catalog rather than the data: `DESCRIBE TABLE EXTENDED
IDENTIFIER(session.ACCOUNTS)` for a one-off look, or `information_schema.columns`
when the result is consumed — `system.information_schema` spans every catalog,
so the library's prefix variable picks the schema:

```sql
CREATE OR REPLACE TEMP VIEW contents_out AS
SELECT column_name AS name, data_type AS type, ordinal_position AS varnum,
       comment AS label
FROM system.information_schema.columns
WHERE CONCAT(table_catalog, '.', table_schema) = session.MYLIB_PREFIX
  AND table_name = 'accounts';
```

⚠️ The SAS columns do not survive one-to-one: `LENGTH` is storage width, which
a `STRING` does not have; `FORMAT`/`INFORMAT` are display rules with no
catalog equivalent (see the metadata guidance); `TYPE` is SAS's 1/2 numeric
code, not a type name. Where a later step *branches* on one of these, that
logic needs rethinking rather than translating — flag it.
