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
python -m data_hydration --from-inventory --inventory-table main.meta.sas_refs \
    --corpus "$(realpath path/to/sas)"
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

Pass metadata resolved across the corpus — `chunker.resolve_corpus_datasets`,
the passes `MultiFileBatcher` runs — or a LIBNAME in a setup file cannot reach
the reads in the files after it, and a macro call names none of the datasets
its macro reads. The CLI and `complexity --hydration` both do.

### SAS data libraries

`build_corpus_plan(..., datasets=...)` takes each chunk's
`metadata.dataset_refs`, chunk by chunk, and plans a **directory LIBNAME per
dataset**, the way a database LIBNAME is planned per table. `set raw.customers;`
after `libname raw '/data/raw';` is a `sas7bdat` item for
`/data/raw/customers.sas7bdat`, target `<catalog>.raw.customers`, owned by the
first file that reads it. A macro call's reads count:
`chunker.resolve_corpus_datasets` gives `%load(raw.customers)` the read its
macro's `set &ds;` makes.

- **Only what the job needs from outside.** A dataset an earlier step of the
  same file creates, or a step of another file (taken to run first, as the
  batcher links it), is the job's own and is never loaded. The step that reads
  a table and rewrites it — `data raw.a; set raw.a;`, an in-place `proc sort` —
  needs the one that was there, so it is loaded; so is one only updated in
  place (`proc append base=`), which must already exist. A step's writes land
  when it ends, save a rename or a copy (`proc datasets; change old=new;
  modify new;`), which serves the statements after it.
- **The LIBNAME in force** is the latest one before the read, in corpus order
  and within a file: `libname raw '/data/2025'; set raw.x; libname raw
  '/data/2026'; set raw.y;` reads `/data/2025/x` and `/data/2026/y`. A read
  before any LIBNAME binds its libref is not planned. A CREATE VIEW's `using
  libname raw '/other'` serves the view's own query alone: the view's
  `raw.t` is planned from `/other`, and the next step's `raw.x` from the
  session's `raw`. A libref only a view assigns is not assigned in the
  session.
- **The library item stays** where no member is named, blocked as before;
  where a list (`raw.sales_:`) is read from it, since which members that covers
  only a listing knows; and while any read names its library through a macro
  variable nothing resolved (`&where..t`), since that may be this library.
- A member spelled through an unresolved macro variable (`raw.&tbl`) is an
  item, blocked; an SPD Engine library is not split (see below).
- **An environment variable is a libref on UNIX.** After `x setenv EDUC
  /users/mydir/educ;` or `options set=EDUC '/users/mydir/educ';`, `set
  educ.class;` reads `/users/mydir/educ/class.sas7bdat`, and `infile ined;` the
  file a variable INED holds. A LIBNAME or FILENAME of the same name wins, as
  in SAS, and a variable nothing reads through is not planned.
- **A library the corpus never assigns is one item, blocked.** A read from
  `xwrk.batch_med` when no LIBNAME anywhere in the corpus assigns `xwrk` comes
  from a library the SAS server assigns itself — an `autoexec.sas` (on UNIX in
  the current, home or `!SASROOT` directory), a configuration file, or an
  environment variable `XWRK`. It is planned as a `sas_session` item naming
  the datasets read, so it no longer drops out of the plan. SAS's own librefs
  (WORK, SASHELP, SASUSER, MAPS) need none, a dataset the job makes first is no
  input, and while some LIBNAME's libref is an unresolved `&lib` nothing is
  claimed unassigned.
- **So is a fileref the corpus never assigns.** `infile qtr;` with no FILENAME
  or environment variable for `qtr` anywhere in the corpus is one
  `sas_session` item, blocked, naming the reads and — for a bare word — the
  file SAS on UNIX reads when no fileref of that name exists: `qtr.dat` in the
  current directory, placed where a cd says which. Before, the read dropped out
  of the plan. A fileref a FILENAME assigns to a device (`filename t temp;`) is
  assigned, to no place, and needs no item.
- **A path the SAS server completes is blocked.** On UNIX `~/` is `$HOME/`,
  `~name/` that user's home, `!sasroot` the directory SAS is installed in, and
  `$VARIABLE` an environment variable. Where those point only the server
  knows, so an item whose path holds one says which, the variable's case
  kept (`$PROJ`), and asks for the place it names.
- **A relative path is read from the current directory.** SAS looks for
  one in its current directory, which a lone `cd` changes — `x cd
  /users/smith;`, `x 'cd /users/smith';`, `call system('cd /users/smith');`,
  `%sysexec cd /users/smith;` — and a `cd` among other commands, run in a
  child shell, does not. So `infile 'in/x.csv';` after one is planned from
  `/users/smith/in/x.csv`, with a note, and a LIBNAME or a data set named by
  a relative path likewise; the current directory carries on into the next
  file, in corpus order. Before any `cd`, SAS reads from the directory it
  was started in, which the corpus does not say: such an item is blocked.
- **A read of a shell command is one blocked item.** `filename f pipe 'gunzip
  -c /data/x.gz'; infile f;` (or `infile '...' pipe;`) reads what the command
  prints, which only the SAS server can produce. Each command read is a
  `sas_session` item naming it, and the paths it names, which are often the
  data itself. Output sent to a command (`file f;`) is no source.
- **A wildcard is blocked.** On UNIX, FILENAME and INFILE may name every file
  a pattern matches — `'/data/in/wild*.dat'`, `'/data/*/*'`, `[0-9]` — read
  one after another. Which files those are only a listing on the SAS server
  says, so such an item is blocked, for that reason alone.
- **A data set read by its path is an item too.** `set
  '/u/2011/budgets/may';` reads `may.sas7bdat` there (SAS adds the
  extension), so it is a `sas7bdat` item, the path's case kept; with no
  libref, its target needs a configured schema. A path that is not a full
  one (`merge 'jan' 'feb';`) is read from the directory SAS runs in, which
  the corpus does not say, so its item is blocked. A name literal
  (`'my data'n`) is a member, not a path.
- **A concatenated library is searched in order.** `libname x ('/a' '/b');` is
  one library: SAS reads a member from the first directory that holds it.
  Only a listing says which that is, so `set x.m;` is planned from `/a`, with
  a note naming the search order; if `m` is not there, the load fails rather
  than read the wrong file. The directories searched after the first need no
  items of their own, unless a list is read through the library. A library
  named by its libref (`libname all (v9 '/x');`) is taken as that libref is
  assigned at that point, so `all.m` is planned from v9's directory, as a v9
  member. One the corpus never assigns, searched first, is a `sas_session`
  item naming the read (`all.m`). A concatenated fileref
  (`filename f ('/q1.csv' '/q2.csv');`) plans each file.

## The reference inventory

`inventory.py` keeps **every reference the corpus makes**, resolved or not, as
rows: each path (`LIBNAME`, `FILENAME`, `INFILE`, `%INCLUDE`, ...), each SAS
dataset a step reads or writes, each database table and each database LIBNAME.
`resolved` says which still hold an unresolved `&macro`; `raw` is the SAS
spelling, `value` the place or name SAS reads, `name` the comparison key.

- **`inventory_rows(file_results, corpus=...)`** builds the rows from the
  chunker's results, resolved with `chunker.resolve_corpus_datasets` (top-level
  chunks only, so a chunk split for size is not read twice). Every row names
  its `corpus`, so one table can hold several.
- **`write_inventory(rows, table)`** appends them to a Delta table as one *run*:
  `run_id` is `<UTC stamp>-<8 hex>`, so the latest run sorts last and the table
  keeps the history. The table and its schema are created when missing, never
  its catalog, and a table created before a column was added (`concat_index`,
  a path's place in a concatenation) gains it on the next write, null in the
  earlier runs. **`read_inventory(table, corpus=...)`** reads the corpus's
  latest run back, or a given `run_id`.
- **`plan_from_inventory(rows)`** plans from the rows alone. The CLI and
  `complexity --hydration` plan this way, so the inventory is the planner's
  one input, and `--from-inventory` plans from the table with no SAS source
  and no chunker.
- **No secret is stored.** A connection option keeps its value only when its
  key says where the connection goes or as whom (`path`, `server`, `schema`,
  `user`, `dsn`, ...); every other value is stored as `<redacted>`, and a
  connection string (`noprompt=`, `url=`) is redacted pair by pair, credentials
  in a URL too. SAS/ACCESS spells a password many ways — `PASSWORD=`, `PASS=`,
  `PW=`, `PWD=`, `USING=` — so a list of the safe keys holds where a list of
  the secret ones leaks. A database LIBNAME's `raw` is rebuilt from those
  options, since the statement as written carries every secret they do. A
  `&macro` reference is kept: it is no secret, and the plan's blocker names
  it. No reader takes a password from these options anyway (invariant 6).

The CLI writes the inventory when `--inventory-table` or
`data_hydration.inventory_table` names a table, under the name `--corpus` gives
(the source directory's absolute path by default). A `--dry-run` writes only
when `--inventory-table` is on its own command line: a table from config is the
deployment's, and a trial plan of another corpus does not belong in it.
`--from-inventory` plans the latest run of `--corpus`, or `--run-id`'s. A table
that cannot be written fails the run's exit status, never its plan.

### Included scripts

`includes.py` answers where the scripts a corpus `%INCLUDE`s are, and the
Python files PROC PYTHON's `INFILE=` runs (`proc python infile=py;`). The path
either names is the SAS server's, so each script is looked for by **file
name** — the one SAS opens, macro variables expanded and filerefs followed
(`src(UTIL)` opens `util.sas`, `src("Util")` opens `Util`) — at any depth:
exactly in the source directory, since UNIX file names are case sensitive,
and ignoring case in SharePoint, as SharePoint does:

```bash
python -m data_hydration path/to/sas --dry-run --check-includes
python -m data_hydration path/to/sas --dry-run --sharepoint-app MyApp
```

`--check-includes` looks in the source directory; `--sharepoint-app` also
looks in the application's SharePoint scripts folder
(`{base}/{app}/scripts_original`, `conversion.paths`) and implies it. Each
`%INCLUDE` and PROC PYTHON row of the inventory records the answer in
`found_local` / `found_sharepoint` (`None` where nobody looked, an empty list
where the script is missing), and the CLI prints one line per script. A folder
SharePoint cannot list fails the run's exit status, never the rest of the
check. `python -m complexity --check-includes` uses the same matching for its
report. A wildcard (`%include '/code/macros/util*.sas';`) finds every script
it matches, as SAS on UNIX reads one: `*` is one or more characters and never
a leading period, `?` one, `[ ]` one of a set.

An `%INCLUDE` also says what a FILENAME is: `filename src '/code/macros';` read
only through `%include src(util);` names SAS source, so the plan leaves it out,
as it leaves out an `%INCLUDE` of a quoted path. So does PROC PYTHON: a
FILENAME only its `INFILE=` reads names a Python script. A fileref INFILE or
FILE reads too holds data, and stays.

## Package layout

| File | Role |
|---|---|
| `models.py` | `HydrationSource` / `Item` / `Plan` / `Report` — all inert |
| `config.py` | `HydrationConfig` + `from_env()`. No secret is a field here |
| `secrets.py` | The one credential chain, and the Entra ID adapter |
| `naming.py` | The target-name template |
| `planner.py` | Refs → plan. Pure |
| `inventory.py` | Refs → inventory rows → plan; the inventory's Delta table |
| `includes.py` | Where each `%INCLUDE`d or PROC PYTHON script is: local, SharePoint, or missing |
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

A flat file (CSV, TSV, Parquet, Excel) is opened by the name SAS reads:
`object_name` keeps the file name's case and suffix. On AIX, Linux and object
storage `Sales.CSV` and `sales.csv` are two different files, and the reader
takes the format from the suffix. The target table is named from the stem
(`Sales.CSV` → `sales`).

Four things that look alike and are not:

- **`.sas7bdat` — data.** Read with `pyreadstat`, no SAS installation.
  `metadataonly=True` gives the schema and row count for free, and
  `row_offset`/`row_limit` implement row-range partitioning directly.
- **`.sas7bndx` — an index, not data.** Detected by convention (`<stem>.sas7bndx`
  beside the dataset) and never read for rows: its layout is undocumented. What
  survives is a *hint* — a SAS index and Delta clustering answer the same
  question, so the columns become a candidate `CLUSTER BY`, applied only when
  `apply_index_clustering` is on. Column recovery is best-effort and usually
  returns nothing; the file's presence is the reliable part.
- **A CPORT transport file (PROC CIMPORT's `infile=`) — data no reader here
  opens.** It is planned, blocked, once (through its FILENAME when a fileref
  names it): import it with PROC CIMPORT in SAS and load the data sets it
  creates.
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
