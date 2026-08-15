## [when: global_statement:libname] LIBNAME binds a name to a place
A `LIBNAME` produces no rows, so it usually becomes **no code at all** — what it
does is fix how every later `mylib.table` reference is written. Resolve it and
say so in Mapping rather than emitting anything for it.

`libname mylib '/mnt/data';` -> the libref becomes the **schema**, so
`mylib.accounts` reads as `spark.table("<catalog>.mylib.accounts")`. Name the
catalog you assumed once and use it consistently.

The engine decides whether the data is even in the lakehouse:

- **Base engine on a path** — an ordinary schema of managed tables.
- **A database engine** (`libname x oracle ...`, `odbc`, `teradata`) — a foreign
  system. Either it was **hydrated** into the lakehouse, in which case
  `spark.table("<catalog>.<libref>.<table>")` reads it like any other table, or
  it is **federated** and read through a foreign catalog by the same three-level
  name. ⚠️ Either way, do **not** emit a JDBC read carrying the connection:
  `spark.read.format("jdbc").option("user", ...)` reproduces a credential the
  SAS spelled as a macro variable. State which you assumed and flag it.
- **The SPD Engine** (`libname x spde '/path'`) — a quoted path that is *not* an
  ordinary directory; it is SAS's own partitioned storage, unreadable outside
  SAS. Treat it as a library to migrate, not a path to read, and flag it.
- `libname x clear;` -> nothing at all.

## [when: global_statement:filename] FILENAME points at files, not tables
A filesystem path becomes a **Unity Catalog volume** path
(`/Volumes/<catalog>/<schema>/<volume>/...`); state the volume you assumed.

⚠️ `filename x pipe '...'` runs an operating-system command, and
`url`/`email`/`ftp` reach off the cluster. There is no DataFrame equivalent for
any of them — emit the non-convertible marker rather than guessing an intent.
