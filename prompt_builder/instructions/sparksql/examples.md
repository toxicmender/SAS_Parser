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

## [example: statement:merge] MERGE with BY-group flags
SAS:
```sas
data work.enriched;
  merge work.orders (in=a) work.customers (in=b);
  by cust_id;
  if a;
  if first.cust_id then seq = 1;
  else seq + 1;
run;
```

Spark SQL:
```sql
CREATE OR REPLACE TEMP VIEW enriched AS
SELECT o.*,
       c.cust_name,
       c.region,
       -- FIRST./LAST. + retained counter -> a window over the BY key.
       ROW_NUMBER() OVER (PARTITION BY o.cust_id ORDER BY o.order_dt) AS seq
FROM orders o
LEFT JOIN customers c          -- `if a;` subsets to the left side, so the
  ON o.cust_id = c.cust_id;    -- full outer MERGE narrows to a LEFT JOIN
```
Notes: a bare `MERGE ... BY` is a **full outer** join; `if a;` restricts it to
rows present in `orders`, which is a `LEFT JOIN`. ⚠️ The `ORDER BY` inside the
window states an order SAS took from its sort; without a sort key the counter
is not reproducible and that must be flagged.

## [example: call_routine:symput] CALL SYMPUT read in a later step
SAS:
```sas
data _null_;
  set work.txns end=eof;
  total + amount;
  if eof then call symput('grand_total', put(total, best12.));
run;

data work.share;
  set work.txns;
  pct = amount / &grand_total;
run;
```

Spark SQL:
```sql
-- The macro variable is written at the END of step 1 and read in step 2,
-- so it is a scalar aggregate, not a per-row value.
CREATE OR REPLACE TEMP VIEW grand_total AS
SELECT SUM(amount) AS grand_total FROM txns;

CREATE OR REPLACE TEMP VIEW share AS
SELECT t.*,
       -- ANSI mode raises on divide-by-zero; SAS returned missing.
       TRY_DIVIDE(t.amount, g.grand_total) AS pct
FROM txns t
CROSS JOIN grand_total g;
```
Notes: ⚠️ `CALL SYMPUT` inside a DATA step publishes at the *step boundary* —
every later step sees the final accumulated value, not a running one. The
`CROSS JOIN` against a one-row view reproduces that read-after-write ordering
explicitly. Passing `grand_total` as a bound parameter is equally correct.

## [example: proc:sort] PROC SORT NODUPKEY
SAS:
```sas
proc sort data=work.txns out=work.latest nodupkey;
  by cust_id descending txn_dt;
run;
```

Spark SQL:
```sql
CREATE OR REPLACE TEMP VIEW latest AS
SELECT * EXCEPT (rn) FROM (
  SELECT t.*,
         ROW_NUMBER() OVER (PARTITION BY cust_id
                            ORDER BY txn_dt DESC) AS rn
  FROM txns t
) WHERE rn = 1;
```
Notes: `NODUPKEY` keeps the first row per BY key *in the sort order just
applied*, so the window's `ORDER BY` must repeat the PROC SORT keys exactly —
here `descending txn_dt`. ⚠️ `txn_dt` alone may not break ties; where it does
not, the surviving row is arbitrary in both systems.

## [example] [kind: MACRO_DEFINITION, MACRO_CALL] Macro loop generating repeated steps
SAS:
```sas
%macro load_region(regions);
  %do i = 1 %to %sysfunc(countw(&regions));
    %let r = %scan(&regions, &i);
    data work.sales_&r;
      set work.sales;
      where region = "&r";
    run;
  %end;
%mend;
%load_region(north south);
```

Spark SQL:
```sql
-- The %DO loop generates one step per region; expanded for the call
-- `%load_region(north south)`, that is two concrete views.
CREATE OR REPLACE TEMP VIEW sales_north AS
SELECT * FROM sales WHERE region = 'north';

CREATE OR REPLACE TEMP VIEW sales_south AS
SELECT * FROM sales WHERE region = 'south';
```
Notes: the macro is a code generator, so the translation is what it
*generates*, expanded against the actual call arguments — not a runtime loop.
⚠️ If the argument list is not known at translation time, the expansion cannot
be resolved; say so and either parameterise a single view on `region` or state
the assumed list.
