## [when: proc:import, proc:export] PROC IMPORT / PROC EXPORT
`PROC IMPORT` reads a file into a dataset. In PySpark that is a reader call, and
the choice is about repeatability:

- **Auto Loader** (`spark.readStream.format("cloudFiles")`) for a load that runs
  again: it tracks which files it has already seen, so a re-run does not double
  the data.
- **`spark.read`** for a one-off read or when the result feeds straight on.

```python
df = (spark.read
      .option("header", True)
      .option("inferSchema", True)
      .csv("/Volumes/main/stage/landing/customers/"))
```

Map the statement's options rather than assuming defaults: `DBMS=` picks the
reader, `DELIMITER=` -> `.option("sep", ...)`, `GETNAMES=YES` -> `header`,
`GUESSINGROWS=` -> `inferSchema`.

⚠️ **`GUESSINGROWS` and `inferSchema` are not the same promise.** They scan
differently and can land on different types for the same file. Where the SAS
pinned types with an `INPUT` statement or informats, build an explicit
`StructType` instead — a column that silently arrives as `string` breaks every
arithmetic downstream.

⚠️ **A source that was hydrated ahead of the run needs no read at all.** Where
the data is already in the lakehouse, use `spark.table(...)` and emit no file
read; re-ingesting data that is already there duplicates it and confuses Auto
Loader's own record of what it has loaded.

`PROC EXPORT` writes a dataset out. Prefer leaving it as a table; where a file
must be produced, write to a volume path — and note that Spark writes a
*directory* of part files, not one named file. Where something downstream
expects a single CSV, that is a real behavioural difference and belongs under
Risks.
