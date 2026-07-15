## Macros and macro variables
Spark SQL has no macro processor. A SAS macro variable reference (`&name`)
resolves *before* the SQL runs, so translate it to a query parameter or a
literal substituted at generation time — never emit `&name` into SQL. State
which macro variables became parameters so the caller can bind them
(`spark.sql(query, args={...})` with named `:param` markers).

## [when: call_routine:symput, call_routine:symputx] CALL SYMPUT
⚠️ `CALL SYMPUT`/`SYMPUTX` writes a macro variable at DATA-step run time; the
value is not available until the step boundary, and any later step reads the
*final* value. There is no equivalent side channel in Spark SQL. Translate
the intent: if the macro variable feeds a later query as a scalar, compute
it with a small aggregating query and pass it as a parameter (or inline it
via a CTE/`CROSS JOIN` of a one-row value). Make the read-after-write
ordering explicit and flag it — a naive translation that reads the variable
too early is a silent-error class.

## [when: macro_function:sysfunc] %SYSFUNC
`%SYSFUNC(fn(args))` calls a DATA-step function at macro-resolution time,
before any query runs. Evaluate it at generation time where possible (e.g.
`%SYSFUNC(today())` -> a bound date parameter), or map it to the equivalent
Spark SQL function inside the query when it genuinely depends on row data.
Do not leave `%SYSFUNC` in the emitted SQL.
