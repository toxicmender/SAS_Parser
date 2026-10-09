# `data_hydration`

Import the **data** a SAS corpus reads into Databricks, as managed Delta tables.

The rest of this repo converts SAS *code*. A converted job that says
`libname edwprod oracle path=EDWPRO_READ_ONLY schema=FR_DM_Pro;` produces Spark
SQL against a table that does not exist in the lakehouse yet. This package reads
the sources the chunker found — Oracle, sFTP, ADLS Gen2, Azure Blob, SAS data
files, SPD Engine libraries — and lands each one as
`catalog.schema.table`.

## Quick start

Plan a corpus without connecting to anything:

```bash
python -m data_hydration path/to/sas --dry-run --stage bronze
```

The same plan, inside the complexity report:

```bash
python -m complexity path/to/sas --hydration --out-dir reports
```

Then run it:

```bash
python -m data_hydration path/to/sas --stage bronze
```

Keep the corpus's reference inventory in a Delta table, and plan from it later
without the SAS source:

```bash
python -m data_hydration path/to/sas --dry-run --inventory-table main.meta.sas_refs
python -m data_hydration --from-inventory --inventory-table main.meta.sas_refs
```

## Plan, then execute

Two layers, and the split is the design:

- **`planner.build_corpus_plan`** turns chunk metadata into a `HydrationPlan`.
  It opens no socket, imports no driver, and reads no file (SPD Engine component
  listing aside). Everything it cannot decide it *records* — an unresolved
  `&macro` password becomes a blocker on the item, not a guess.
- **`runner.execute`** walks the plan and moves the bytes.

That purity is not decoration. It is what lets `complexity` build a plan purely
to print it: a report renderer must not be able to open a database connection,
so `complexity` always passes `probe=None`.

### Database tables

`build_corpus_plan(..., db_tables=...)` takes the chunker's
`SasChunkMetadata.db_tables` — the tables a corpus reaches inside a database,
through SQL pass-through or as members of a database LIBNAME — and plans each
table **read** as its own item: `object_name` is `owner.table`, the connection
options are the ones that table was read through, and the target schema
defaults to the libref, else the owner (`edw_export.current_nonip` →
`<catalog>.edw_export.current_nonip`). Writes are what the converted job
produces, never sources.

- **One item per table.** A table read by five files is planned once, under the
  first file that reads it; appending one copy per reader would load its rows
  five times.
- **A LIBNAME whose tables are named is planned per table.** Its schema-level
  item — the stand-in when no table is known — is dropped; a LIBNAME nothing
  names keeps it.
- **Blocked, not guessed:** a connection whose engine is unknown (its `CONNECT`
  made by a macro call), a `@dblink` (the table lives in the *linked* database),
  and an unresolved `&macro` in the table name, alongside the usual option
  check. A database other than Oracle (Teradata, DB2, …) is planned through
  the SQL path so the plan lists it, and blocked: the one SQL reader speaks
  Oracle.
- **A list of tables is one item, named by the operator.** `set edw.acct_:;`
  reads every table whose name starts `acct_`, and `proc copy in=edw` every
  table there is; which those are only the database knows. The list is planned
  as it is written, with no target and a blocker saying what it covers.
- **Credentials** are keyed on the libref, or for pass-through on the
  connection alias (`oracle_password_<alias>`).
- **Names arrive resolved** when the corpus says what their macro variables
  hold — `%LET`, `CALL SYMPUTX` literals, a utility macro's call arguments.
  A table a `%MACRO` body names by its own parameters is a template and is
  never planned: each call is planned instead, with the table it reads.
- **Paths arrive resolved the same way.** `libname raw "&root/in";` after
  `%let root = /SASData;` is planned at `/SASData/in` (the reference's
  `effective_path`, case kept). Only a reference still unresolved becomes a
  blocker.

Pass metadata resolved across the corpus — `chunker.resolve_corpus_references`
— or a LIBNAME in a setup file cannot reach the reads in the files after it.
The CLI and `complexity --hydration` both do.

### SAS data libraries

`build_corpus_plan(..., datasets=...)` takes `chunk.metadata.dataset_refs`, and
plans a **directory LIBNAME per dataset**, the way a database LIBNAME is planned
per table. `set raw.customers;` after `libname raw '/data/raw';` is a
`sas7bdat` item for `/data/raw/customers.sas7bdat`, target
`<catalog>.raw.customers`, owned by the first file that reads it.

- **Only what the job needs from outside.** A dataset some step creates is the
  job's own and is never loaded; one only updated in place (`proc append
  base=`) must already exist, so it is.
- **The LIBNAME in force** is the latest one before the read, in corpus order
  (a file's LIBNAMEs before its reads). A read before any LIBNAME binds its
  libref is not planned.
- **The library item stays** where no member is named, blocked as before, and
  where a list (`raw.sales_:`) is read from it: which members that covers, only
  a listing knows.
- A member spelled through an unresolved macro variable (`raw.&tbl`) is an
  item, blocked; an SPD Engine library is not split (see below).

## The reference inventory

`inventory.py` keeps **every reference the corpus makes**, resolved or not, as
rows: each path (`LIBNAME`, `FILENAME`, `INFILE`, `%INCLUDE`, ...), each SAS
dataset a step reads or writes, each database table and each database LIBNAME.
`resolved` says which still hold an unresolved `&macro`; `raw` is the SAS
spelling, `value` the place or name SAS reads, `name` the comparison key.

- **`inventory_rows(file_results)`** builds the rows from the chunker's results
  (top-level chunks only, so a chunk split for size is not read twice).
- **`write_inventory(rows, table)`** appends them to a Delta table as one *run*:
  `run_id` is `<UTC stamp>-<8 hex>`, so the latest run sorts last and the table
  keeps the history. The table and its schema are created when missing, never
  its catalog. **`read_inventory(table)`** reads the latest run back, or a
  given `run_id`.
- **`plan_from_inventory(rows)`** plans from the rows alone. The CLI and
  `complexity --hydration` plan this way, so the inventory is the planner's
  one input, and `--from-inventory` plans from the table with no SAS source
  and no chunker.
- **No secret is stored.** A connection option whose key names a password
  (`pass=`, `password=`, `pwd=`, ...) is stored as `<redacted>`, in `options`
  and in the statement's `raw`, and so is a `PWD=` inside a connection string.
  A `&macro` reference is kept: it is no secret, and the plan's blocker names
  it. No reader takes a password from these options anyway (invariant 6).

The CLI writes the inventory when `--inventory-table` or
`data_hydration.inventory_table` names a table, with `--dry-run` too: it is a
record of the corpus, not a load. A table that cannot be written fails the run's
exit status, never its plan.

### Included scripts

`includes.py` answers where the scripts a corpus `%INCLUDE`s are. The path an
`%INCLUDE` names is the SAS server's, so each script is looked for by **file
name** — the one SAS opens, macro variables expanded and filerefs followed
(`src(util)` opens `util.sas`) — ignoring case and at any depth:

```bash
python -m data_hydration path/to/sas --dry-run --check-includes
python -m data_hydration path/to/sas --dry-run --sharepoint-app MyApp
```

`--check-includes` looks in the source directory; `--sharepoint-app` also looks
in the application's SharePoint scripts folder (`{base}/{app}/scripts_original`,
`conversion.paths`) and implies it. Each `%INCLUDE` row of the inventory records
the answer in `found_local` / `found_sharepoint` (`None` where nobody looked, an
empty list where the script is missing), and the CLI prints one line per
script. A folder SharePoint cannot list fails the run's exit status, never the
rest of the check. `python -m complexity --check-includes` uses the same
matching for its report.

An `%INCLUDE` also says what a FILENAME is: `filename src '/code/macros';` read
only through `%include src(util);` names SAS source, so the plan leaves it out,
as it leaves out an `%INCLUDE` of a quoted path. A fileref INFILE or FILE reads
too holds data, and stays.

## Package layout

| File | Role |
|---|---|
| `models.py` | `HydrationSource` / `Item` / `Plan` / `Report` — all inert |
| `config.py` | `HydrationConfig` + `from_env()`. No secret is a field here |
| `secrets.py` | The one credential chain, and the Entra ID adapter |
| `naming.py` | The target-name template |
| `planner.py` | Refs → plan. Pure |
| `inventory.py` | Refs → inventory rows → plan; the inventory's Delta table |
| `includes.py` | Where each `%INCLUDE`d script is: local, SharePoint, or missing |
| `partition.py` | Which partitioning strategy, and why |
| `runner.py` | Executes a plan, one item at a time |
| `rawio.py` | `RangedRawIO` — object storage as a file object |
| `sources/` | One reader per system; every driver imported lazily |
| `sinks/delta.py` | The managed-table writer |
| `__main__.py` | `python -m data_hydration` |

## Load-bearing invariants

1. **This package imports only `app_config` at run time.** Chunker types are
   `TYPE_CHECKING`-only annotations, so `import data_hydration` never pulls in
   `chunker`, `pipeline`, or any driver. `tests/test_data_hydration.py` asserts
   this directly, because the failure mode is silent: an import added for
   convenience turns a decoupled module into part of the conversion stack.
   Direction is one-way — `complexity` imports *this*, never the reverse.

2. **Every driver import is lazy.** `import data_hydration` must succeed with
   none of `oracledb`, `paramiko`, `pyreadstat`, `saspy` or the Azure SDKs
   installed — the same rule Architecture.md invariant 8 sets for pyspark. A
   plan needs none of them; only reading does.

3. **The run date is rendered once, on the plan.** `HydrationPlan.run_date` is
   fixed when the plan is built and every target name uses it. Re-deriving it
   per item means a run starting at 23:59 writes half its partitions into
   yesterday's table.

4. **Write mode is a planning decision, not a runtime one.** The first item for
   a table overwrites, the rest append. Deciding at execution time would make
   the result depend on the order items happened to run in, and two files
   declaring the same LIBNAME would each think they were first —
   which is why `build_corpus_plan` builds the whole corpus in one pass rather
   than merging per-file plans.

5. **A bad template raises; a missing value blocks one item.**
   `validate_template` failing is a broken configuration and stops the run.
   A *source* that cannot fill a placeholder — an `INFILE` with no libref, so no
   schema — gets `target_table = "<unresolved>"` plus a blocker, and the other
   forty tables still appear in the report.

6. **Secrets never come from `config.json`.** The chain is the Databricks secret
   scope, then Vault, then the environment; Azure storage uses an Entra ID token
   through `app_config.azure` and has no key at all. `HydrationConfig` has no
   secret field, so there is nowhere for one to be written by accident.

## What the SAS formats actually are

Three things that look alike and are not:

- **`.sas7bdat` — data.** Read with `pyreadstat`, no SAS installation.
  `metadataonly=True` gives the schema and row count for free, and
  `row_offset`/`row_limit` implement row-range partitioning directly.
- **`.sas7bndx` — an index, not data.** Detected by convention (`<stem>.sas7bndx`
  beside the dataset) and never read for rows: its layout is undocumented. What
  survives is a *hint* — a SAS index and Delta clustering answer the same
  question, so the columns become a candidate `CLUSTER BY`, applied only when
  `apply_index_clustering` is on. Column recovery is best-effort and usually
  returns nothing; the file's presence is the reliable part.
- **SPD Engine (`libname x spde '/path'`) — a partitioned directory.** Planning
  is static: the `.dpf` components are counted from a directory listing, with no
  SAS needed. **Reading requires `saspy`** — there is no open-source `.dpf`
  parser and writing one is not in scope. The components are counted but *not*
  fanned out into an item each, because they cannot be read individually; doing
  so would read the whole dataset once per component.

## Logging

Logger names follow `data_hydration.*` (`data_hydration.planner`,
`data_hydration.rawio`, `data_hydration.sources.oracle`, ...). f-string messages
throughout, per-iteration debug guarded with `isEnabledFor`. The CLI configures
logging through `app_config.logging_setup.configure_logging`, never
`basicConfig`.

## Testing

`tests/test_data_hydration*.py` run with no network, no JVM and no driver
installed: sources are hand-written fakes recording their calls, and
`RangedRawIO` is exercised against an in-memory byte source.

⚠️ **`sinks/delta.py` and the inventory's table cannot be exercised in the
local `.venv`**, where `pyspark` is shadowed by `databricks-connect`. Verify them
in Docker (`docker/spark`), the same rule `memory.store`'s Delta backend
follows: `tests/test_data_hydration_delta.py` writes and reads both through a
real Delta session there. `tests/test_data_hydration_inventory.py` covers the
rest of the inventory with no Spark at all.
