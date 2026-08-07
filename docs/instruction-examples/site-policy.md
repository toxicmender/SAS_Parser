# Site-policy instructions — an example to copy, not a shipped default

These are *house rules*, not SAS→Spark semantics: they encode how one
organisation wants its migrated SQL to look, and they would be wrong for
someone else. That is why they live here rather than in
`prompt_builder/instructions/`, which ships only guidance that is true of the
languages themselves.

**To use them:** copy this file into your own instructions directory — under
`_common/` if it should apply to every target, or under `sparksql/` if it is
SQL-specific — edit the specifics, and point `config.json`
`user_instructions.dir` at that directory. Drop the `# ` heading level so the
`##` sections below become top-level instruction sections.

Everything below this line is the instruction text.

---

## [lang: sparksql] Parameterise hardcoded filters
Replace a hardcoded geography, entity, or reporting-year filter with a session
variable so the same script serves every partition of the work. Spark 4 has
first-class SQL session variables:

```sql
DECLARE OR REPLACE VARIABLE EXCLUDED_REGION STRING DEFAULT 'XX';
DECLARE OR REPLACE VARIABLE EXCLUDED_YEAR   INT    DEFAULT 2019;

CREATE OR REPLACE TEMP VIEW active AS
SELECT * FROM accounts
WHERE region <> session.EXCLUDED_REGION
  AND YEAR(open_dt) <> session.EXCLUDED_YEAR;
```

Qualify the reference as `session.NAME` so it can never be shadowed by a
column of the same name. Where the caller drives the value instead, bind it as
a query parameter (`spark.sql(query, args={...})` with `:name` markers) — the
same substitution a SAS macro variable was doing, moved to a place the reader
can see. A value used once and never varying stays a literal; parameterising
everything is its own kind of noise.

⚠️ Session variables are not visible inside a **persisted** view — only a
temp view or a direct query. Keep the logic equivalent: a filter must not
quietly become a different filter because a default changed.

## [lang: sparksql] Leave physical layout alone
Do not emit `OPTIMIZE`, `ZORDER`, `VACUUM`, partitioning clauses, bucketing,
or any other storage directive unless the SAS source explicitly asked for the
equivalent. Physical layout is a platform decision made against real data
volumes and query patterns, and a translation has no basis for it. The same
goes for broadcast hints and cache statements — flag a suspected performance
problem under Risks and leave the choice to the operator.

## [lang: sparksql] Close with equivalence checks
After the translated steps, emit validation queries that let a reviewer
compare the Spark output against the SAS output: a row count per output
dataset, and a key aggregate (sum or distinct count) over the columns the
business actually reconciles on.

```sql
SELECT 'summary' AS dataset, COUNT(*) AS row_count,
       SUM(total_amt) AS total_amt, COUNT(DISTINCT cust_id) AS cust_n
FROM summary;
```

Keep them in a separate final block, clearly labelled, so they are easy to
strip before deployment. Where a column is nullable and the SAS relied on
missing-value behaviour, check it explicitly — `COUNT(col)` against
`COUNT(*)`, or a `COALESCE` in the aggregate — rather than assuming the null
handling matched.
