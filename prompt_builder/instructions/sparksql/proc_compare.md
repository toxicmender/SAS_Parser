## [when: proc:compare] [kind: PROC_STEP] PROC COMPARE
`PROC COMPARE BASE=a COMPARE=b` reconciles two datasets. In a migration it is
often the *most* important step to translate faithfully, because it is what
proves the migration itself — so keep its three distinct questions separate
rather than collapsing them into one query.

**Rows on one side only** — `EXCEPT` between the two, both directions:

```sql
SELECT * FROM base   EXCEPT ALL SELECT * FROM compare;  -- in base, not compare
SELECT * FROM compare EXCEPT ALL SELECT * FROM base;    -- and the reverse
```

⚠️ `EXCEPT` de-duplicates, `EXCEPT ALL` does not. SAS compares observations,
duplicates included, so `EXCEPT ALL` is the faithful one; plain `EXCEPT` hides
a duplicated row, which is exactly the kind of difference a reconciliation
exists to find.

**Value differences per key** — the `ID` statement is the join key, so join on
it and compare column by column:

```sql
SELECT COALESCE(a.id, b.id) AS id, 'amt' AS column,
       a.amt AS base_value, b.amt AS compare_value
FROM base a FULL OUTER JOIN compare b USING (id)
WHERE a.amt IS DISTINCT FROM b.amt;
```

⚠️ **Use `IS DISTINCT FROM`, never `<>`.** `a.amt <> b.amt` is `NULL` when
either side is missing, so `WHERE` drops the row and the difference is never
reported — a missing-on-one-side mismatch, which is the commonest kind, would
be silently declared equal. `IS DISTINCT FROM` treats NULL as a comparable
value.

**Structural differences** — variables or types present on one side only. That
is a metadata question; see the catalog guidance rather than querying the rows.

Other options worth carrying over:
- `CRITERION=` is a fuzzy numeric match -> `abs(a.x - b.x) > <criterion>`
  rather than an equality test, guarded for NULL the same way.
- `VAR`/`WITH` compare differently-named columns -> just name them in the
  comparison.
- ⚠️ Without an `ID` statement, SAS compares by **observation number**. Spark
  has no row order, so there is no faithful translation — say so and ask for
  the key, rather than inventing one with `ROW_NUMBER()`.

`OUT=` with `OUTDIF`/`OUTPERCENT` writes the differences as data; emit that as
the view. With no `OUT=`, the PROC only prints, so what a later step consumes
is the whole of what needs translating.
