## [meta: permanent_tables] Table names through session variables
Name every permanent table — a dataset in a SAS library other than WORK, or a
database table — through session variables, so its catalog and schema are set
in one place:

1. One prefix per SAS library, `<LIBREF>_PREFIX`, holding `'catalog.schema'`:
   the libref is the schema, the catalog the one you assumed (say which under
   Mapping). Where the item already gives a dataset's Databricks name
   (`catalog.schema.table`), the prefix holds that `catalog.schema`; a library
   mapped to several schemas takes a prefix per schema, named after it.
2. One variable per table, named after it (`RAW_CUSTOMERS`, `MART_CUSTOMERS`
   when two libraries share a name), built from its prefix with a leading dot.
3. `IDENTIFIER(session.<VAR>)` wherever the table is read or written.

```sql
DECLARE OR REPLACE VARIABLE MART_PREFIX STRING DEFAULT 'catalog_name.mart';
DECLARE OR REPLACE VARIABLE RAW_DEMO STRING DEFAULT CONCAT(session.MART_PREFIX, '.raw_demo');
DECLARE OR REPLACE VARIABLE DEMO_TABLE STRING DEFAULT CONCAT(session.MART_PREFIX, '.demo_table');

CREATE OR REPLACE TABLE IDENTIFIER(session.DEMO_TABLE) AS
SELECT * FROM IDENTIFIER(session.RAW_DEMO);
```

Declare every variable an item uses at the top of its translation —
`DECLARE OR REPLACE` makes a re-run safe. WORK datasets stay temporary views.
⚠️ Never a literal `catalog.schema.table`, nor `IDENTIFIER('a.b.c')`. A
statement that takes no `IDENTIFIER` — `RENAME TO`'s new name, a `CLONE`
source, `OPTIMIZE`, a stored view's query (which may not read a session
variable) — is built from the same variables into a `STRING` variable and run
with `EXECUTE IMMEDIATE`. Session variables need Runtime 14.1 or above,
`EXECUTE IMMEDIATE` 14.3; say so under Risks.
