## [when: component_object:hash, component_object:hiter] Hash-object lookups
A `DECLARE HASH h(dataset:'work.lk')` loaded from a dataset is an in-memory
lookup table, and the DATA step around it is a **join** — that is the whole
construct, and it should read as one:

```sas
data work.enriched;
  set work.txns;
  if _n_ = 1 then do;
    declare hash h(dataset:'work.rates');
    h.definekey('ccy'); h.definedata('rate'); h.definedone();
  end;
  rc = h.find();
  if rc = 0 then amt_usd = amt * rate;
run;
```

```sql
CREATE OR REPLACE TEMP VIEW enriched AS
SELECT t.*,
       CASE WHEN r.ccy IS NOT NULL THEN t.amt * r.rate END AS amt_usd
FROM txns t
LEFT JOIN rates r ON t.ccy = r.ccy;
```

Read the method calls as join parts: `definekey` is the `ON` clause,
`definedata` is what the join brings back, `find()` returning 0 is a **match**
(so `rc = 0` is `r.<key> IS NOT NULL` after a `LEFT JOIN`), and `check()` tests
existence without retrieving — a semi-join, `WHERE EXISTS (...)` or
`LEFT SEMI JOIN`, not a `LEFT JOIN`.

⚠️ **A hash never multiplies rows; a join does.** SAS keeps only the *first*
record per key unless the hash is declared `multidata:'y'`, so a duplicate key
in the lookup dataset is silently ignored. A `LEFT JOIN` on that same
non-unique key fans out and the step returns more rows than SAS produced.
De-duplicate the right side before joining —
`QUALIFY ROW_NUMBER() OVER (PARTITION BY key ORDER BY ...) = 1` — and state
which row you kept. With `multidata:'y'` the fan-out *is* the intent, so join
straight through.

Two hash uses that are **not** joins:

- **`h.add()` inside the row loop**, accumulating as the step reads, is a
  running de-duplication or aggregation over the input — a window or
  `GROUP BY`, not a second table. `h.find()` immediately before an `add()` is
  the classic "have I seen this key?" idiom -> `ROW_NUMBER() ... = 1`.
- **`h.output(dataset:'x')`** writes the hash's contents out, so it is that
  step's *result*: emit the accumulated set as its own view.

`DECLARE HITER` walks the hash in key order; that is an ordered scan, so it
maps to a window with an explicit `ORDER BY`, never to Spark's unordered scan.

The SAS author asserted the lookup fits in memory by choosing a hash at all —
useful context for a reviewer, and worth stating — but leave broadcast hints to
the operator rather than emitting them.
