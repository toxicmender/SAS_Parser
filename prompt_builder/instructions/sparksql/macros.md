## [meta: invokes_macros, defines_macros, produces_macrovars] Macros and macro variables
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

## [meta: invokes_macros] %SYSFUNC and macro-time function calls
`%SYSFUNC(fn(args))` runs a DATA-step function at macro-resolution time,
before any query. Evaluate it at generation time where you can
(`%SYSFUNC(today())` -> a bound date), or map it to the Spark function inside
the query when it genuinely depends on row data. Never leave `%SYSFUNC` in the
emitted SQL.

## [kind: MACRO_DEFINITION, MACRO_CALL] Decomposing a macro
A macro is a *code generator*, not a function: it emits SAS text that is then
compiled. Translate what it generates, not the generator.

Work through it in order: list the parameters (positional and keyword, with
defaults) and every macro variable the body reads or writes; expand the
control flow (`%IF`, `%DO` loops) to see what steps actually result; then
translate each resulting step, chaining them as CTEs where they feed one
another and as separate views where a later SAS step reads them by name.

- A `%DO i = 1 %TO n` loop that generates *n* similar steps becomes a
  `UNION ALL` over the *n* variants, or one query parameterised by the loop
  variable — not a loop.
- A `%IF` on a parameter chooses which code exists at all. Resolve it against
  the known argument at generation time and emit only the surviving branch;
  do not translate it to a runtime `CASE`, which changes when the decision
  happens. ⚠️ When the argument is not known, say so and state which branch
  you assumed.
- Macro parameters become query parameters (`spark.sql(query, args={...})`) or
  substituted literals. Report the mapping from macro parameter to query
  parameter so the caller can bind them.
- A macro invoked several times with different arguments produces several
  concrete step sequences. Name each output distinctly rather than emitting
  one view that silently overwrites the previous call's result.

## [kind: MACRO_DEFINITION, MACRO_CALL, MACRO_CONTROL_FLOW] When to use SQL scripting instead
Databricks SQL has real procedural statements — `BEGIN ... END`, `IF/THEN/ELSE`,
`CASE`, `FOR`, `WHILE`, `LOOP`, `REPEAT`, `LEAVE`, `ITERATE`, `SIGNAL`,
`GET DIAGNOSTICS` — so a SAS control-flow construct is no longer automatically
untranslatable.

**Expansion stays the default.** Most SAS macros are code generators whose loop
bounds and branches are known at translation time; expanding them yields the
one-view-per-step shape the output format asks for, and a reviewer can diff it
against the SAS. Reach for scripting only when the control flow genuinely
depends on data read at run time — then the mapping is:

| SAS | Databricks SQL |
|---|---|
| `%DO i = 1 %TO n` over a data-driven `n` | `FOR ... DO ... END FOR` over a query |
| `%DO %WHILE` / `%DO %UNTIL` | `WHILE ... DO` / `REPEAT ... UNTIL` |
| `%IF ... %THEN` on run-time data | `IF ... THEN ... ELSE ... END IF` |
| `%GOTO` out of a loop | `LEAVE` (or `ITERATE` to skip to the next pass) |
| `%ABORT` / `%ABORT CANCEL` | `SIGNAL` with a condition and message |

⚠️ Scripting runs statement by statement, so it gives up the whole-script
optimisation a set-based translation gets. Say in Risks why the loop could not
be expanded.

## [meta: invokes_macros, produces_macrovars] Dynamic SQL and generated identifiers
Where a macro builds SQL *text* rather than parameter values:

- `CALL EXECUTE('...')` and a `%SYSFUNC`-assembled statement -> `EXECUTE
  IMMEDIATE`, which also takes `USING` arguments and can write results `INTO`
  variables. Prefer binding values over concatenating them.
- A macro-variable-built **object name** (`&lib..&ds`, `work.&prefix._out`) ->
  the `IDENTIFIER()` clause: `SELECT * FROM IDENTIFIER(:tbl)`. A parameter
  marker cannot stand where a table name goes, and string-concatenating the
  name into the statement is an injection risk; `IDENTIFIER()` is the construct
  for exactly this.
- ⚠️ Prefer resolving the name at translation time when the macro variable is
  known — a literal name is reviewable, a dynamic one is not.
