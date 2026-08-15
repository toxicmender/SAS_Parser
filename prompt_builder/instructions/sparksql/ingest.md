## [when: proc:import, proc:export] PROC IMPORT / PROC EXPORT
`PROC IMPORT` reads a file into a dataset. Two targets, and the choice is about
repeatability:

- **`COPY INTO`** for a load that runs again: it is idempotent, tracking which
  files it has already ingested, so a re-run does not double the data. This is
  what a scheduled SAS job's import should become.
- **`read_files(...)`** (or `SELECT * FROM csv.\`path\``) for a one-off read or
  when the result feeds straight into a query.

```sql
COPY INTO main.stage.customers
FROM '/Volumes/main/stage/landing/customers/'
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'delimiter' = ',', 'inferSchema' = 'true');
```

Map the statement's options rather than assuming defaults: `DBMS=CSV|DLM|XLSX`
-> `FILEFORMAT`, `DELIMITER=` -> `'delimiter'`, `GETNAMES=YES` -> `'header'`,
`GUESSINGROWS=` -> `'inferSchema'` with the caveat below, `DATAROW=` -> a
skip-rows option or a filtering `SELECT`.

⚠️ **`GUESSINGROWS` and schema inference are not the same promise.** SAS scans
a bounded number of rows and picks types; `inferSchema` scans differently and
can land on a different type for the same file. Where the SAS pinned types with
an `INPUT` statement or informats, state the schema explicitly in the `CREATE
TABLE` rather than inferring — a column that silently arrives as `STRING`
instead of `DOUBLE` breaks every arithmetic downstream.

⚠️ **A source that was hydrated ahead of the run needs no ingestion at all.**
Where the data has already been copied into the lakehouse — a `LIBNAME` on a
database engine, or a SAS library migrated as tables — the translation reads
`<catalog>.<schema>.<table>` directly and emits **no** `COPY INTO` and no
`read_files`. Emitting a load for data that is already there re-ingests it, and
against `COPY INTO`'s idempotence tracking the two disagree about what has been
loaded. Ingest only what the SAS itself reads from a *file*.

`PROC EXPORT` writes a dataset out. Prefer leaving the data as a table and
letting the consumer read it; where a file genuinely must be produced, write to
a volume path. ⚠️ Spark writes a *directory* of part files, not one file — if
something downstream expects a single named CSV, that is a real behavioural
difference and belongs under Risks.

## [when: statement:infile] INFILE / INPUT read raw text
A DATA step with `INFILE` plus an `INPUT` statement is a parser, not a query.
The file becomes a `read_files(...)` source or a `COPY INTO`, and the `INPUT`
statement becomes the schema.

How hard that is depends on the `INPUT` form:

- **List input** (`input id name $ amt;`) with a `DLM=`/`DSD` delimiter is a
  plain delimited read — straightforward.
- ⚠️ **Column and formatted input** (`input id 1-5 name $ 6-25;`, `@` pointers,
  informats like `input dt yymmdd10.;`) is **fixed-width parsing**, which has
  no delimited-read equivalent. Read the line as a single `STRING` column and
  cut it with `substr`, then cast — and say that is what you did.
- ⚠️ `INFILE` options that change the reading loop itself — `FIRSTOBS=`, `OBS=`,
  `MISSOVER`/`TRUNCOVER`, `END=`, trailing `@`/`@@` line-hold — control a
  row-by-row reader Spark does not have. `MISSOVER` and `TRUNCOVER` decide
  whether a short line yields missing values or reads on to the next line;
  reproduce the intent explicitly and flag it, because the two produce
  different data from the same file.

`FILE` and `PUT` statements writing a text file are the export path in reverse
— the same directory-of-part-files caveat applies.
