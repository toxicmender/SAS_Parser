## [example: proc:sql] PROC SQL join with a computed column
A worked SAS -> Spark SQL pair showing the target response shape.

SAS:
```sas
proc sql;
  create table work.summary as
  select a.cust_id,
         a.region,
         sum(b.amount) as total_amt,
         calculated total_amt / count(b.txn_id) as avg_amt
  from work.customers as a
  left join work.txns as b
    on a.cust_id = b.cust_id
  group by a.cust_id, a.region;
quit;
```

Spark SQL:
```sql
CREATE OR REPLACE TEMP VIEW summary AS
WITH agg AS (
  SELECT a.cust_id,
         a.region,
         SUM(b.amount)              AS total_amt,
         COUNT(b.txn_id)            AS txn_count
  FROM customers a
  LEFT JOIN txns b
    ON a.cust_id = b.cust_id
  GROUP BY a.cust_id, a.region
)
SELECT cust_id,
       region,
       total_amt,
       total_amt / NULLIF(txn_count, 0) AS avg_amt  -- guard divide-by-zero
FROM agg;
```
Notes: `CALCULATED total_amt` is not valid in Spark, so the aggregate moves
into a CTE and the ratio is computed in the outer query. `NULLIF(...,0)`
replaces SAS's silent missing-on-divide-by-zero with an explicit `NULL`.

## [example: function:intnx] INTNX month shift
SAS:
```sas
data work.next_month;
  set work.acct;
  due_date = intnx('month', open_date, 1);  /* first day of next month */
  format due_date yymmdd10.;
run;
```

Spark SQL:
```sql
CREATE OR REPLACE TEMP VIEW next_month AS
SELECT acct.*,
       -- INTNX default alignment is BEGINNING: first day of next month,
       -- not open_date + 1 month.
       TRUNC(ADD_MONTHS(open_date, 1), 'MM') AS due_date
FROM acct;
```
