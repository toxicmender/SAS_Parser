## [when: global_statement:libname] LIBNAME binds a name to a place
A `LIBNAME` is a *binding*, not a step — it gives a two-level prefix somewhere
to resolve to. It produces no rows, so it usually translates to **no
statement at all**; what it does is fix how every later `mylib.table` name is
written. Resolve it and say so in Mapping rather than emitting DDL for it.

`libname mylib '/mnt/data';` -> the libref becomes the **schema** in
`<catalog>.mylib.<table>`. Name the catalog you assumed once, and use it
consistently for every table in the translation.

⚠️ Do not emit `CREATE SCHEMA` unless the SAS creates the library. A `LIBNAME`
pointing at an existing directory asserts the location already exists; a
migration that silently creates schemas is doing governance work nobody asked
for. Where the schema may genuinely not exist, say so under Risks.

The engine matters, because it decides whether the data is even in the
lakehouse:

- **Base/default engine on a path** — an ordinary schema of managed tables.
  Only if the SAS is explicitly reading files in place does an external table
  or a `LOCATION` become right.
- **A database engine** (`libname x oracle ...`, `odbc`, `teradata`, `sqlsvr`)
  — this is a foreign system, not a Databricks schema. Map it to a Lakehouse
  Federation foreign catalog and ⚠️ flag it: the connection, credentials and
  the decision to federate rather than ingest are the operator's, not the
  translation's.
- `libname x clear;` and `libname _all_ list;` -> nothing at all.

## [when: global_statement:filename] FILENAME points at files, not tables
A `FILENAME` names a file or pipe for `INFILE`/`FILE` to read or write. A
filesystem path becomes a **Unity Catalog volume** path
(`/Volumes/<catalog>/<schema>/<volume>/...`), which is where a governed
workspace keeps non-tabular files; state the volume you assumed.

⚠️ `filename x pipe '...'` runs an operating-system command. There is no SQL
equivalent, and it is the SAS construct most likely to be doing something the
lakehouse deliberately cannot — emit the non-convertible marker rather than
guessing an intent. The same goes for `filename x url|email|ftp`.
