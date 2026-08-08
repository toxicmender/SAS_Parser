## [when: proc:print, proc:report, proc:tabulate] [kind: PROC_STEP] Reporting PROCs
These produce a **printed report**, not a dataset. The rule is the same for all
three: translate the data the report is built from, and nothing else.

⚠️ **Do not invent a table for a report.** A `PROC PRINT` that only displays
rows translates to a `SELECT`, or — when nothing downstream consumes it — to
nothing at all, noted as a dropped display step. Creating a view named after
the printout adds an artefact the SAS never had, and it will be mistaken for
data by the next step.

Where they *do* produce data, it is through `OUT=`, and that is what to emit:

- **`PROC TABULATE ... OUT=`** is a summary dataset: `CLASS` -> `GROUP BY`,
  the `TABLE` statement's dimensions -> the grouping combinations. Multiple
  crossings become `GROUPING SETS`, and the `_TYPE_` column that distinguishes
  them is `GROUPING_ID()`. Same discipline as PROC MEANS: emit only the levels
  the `TABLE` statement asks for.
- **`PROC REPORT ... OUT=`** gives the computed report as rows. `DEFINE ...
  /GROUP` are the grouping columns, `/ANALYSIS` the aggregates, `/ACROSS` a
  pivot, and `/COMPUTED` a derived column — so a `PIVOT` plus expressions.
  ⚠️ `COMPUTE` blocks are procedural and can reference the *previous* report
  line; those have no set-based form and need flagging, not guessing.

Everything else these PROCs carry — `TITLE`, `FOOTNOTE`, `LABEL`, `FORMAT`,
`BY` page breaks, `SPLIT=`, ordering for display, styles — is presentation. It
has no target equivalent and is not a loss worth working around: note it once
under Risks and move on.

## [when: proc:fedsql] [kind: PROC_STEP] PROC FEDSQL
FedSQL is ANSI SQL, so it is *closer* to Databricks SQL than PROC SQL is —
translate the query almost verbatim and check only the edges:

- No `CALCULATED` keyword (it is not SAS SQL), so select-list aliases were
  already being repeated or wrapped — keep whatever the source did.
- FedSQL has its own type names (`DOUBLE`, `BIGINT`, `VARCHAR(n)`) which map
  directly; SAS-format-based conversions do not appear.
- Dataset references still need the two-level to three-level name change.
- ⚠️ FedSQL runs with its own missing-value and division semantics; the
  `NULL`-versus-missing and ANSI divide-by-zero guidance still applies.
