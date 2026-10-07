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
  check.
- **Credentials** are keyed on the libref, or for pass-through on the
  connection alias (`oracle_password_<alias>`).
- **Names arrive resolved** when the corpus says what their macro variables
  hold — `%LET`, `CALL SYMPUTX` literals, a utility macro's call arguments.
  A table a `%MACRO` body names by its own parameters is a template and is
  never planned: each call is planned instead, with the table it reads.

Pass metadata resolved across the corpus — `chunker.resolve_corpus_references`
— or a LIBNAME in a setup file cannot reach the reads in the files after it.
The CLI and `complexity --hydration` both do.

## Package layout

| File | Role |
|---|---|
| `models.py` | `HydrationSource` / `Item` / `Plan` / `Report` — all inert |
| `config.py` | `HydrationConfig` + `from_env()`. No secret is a field here |
| `secrets.py` | The one credential chain, and the Entra ID adapter |
| `naming.py` | The target-name template |
| `planner.py` | Refs → plan. Pure |
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

⚠️ **`sinks/delta.py` cannot be exercised in the local `.venv`**, where `pyspark`
is shadowed by `databricks-connect`. Verify it in Docker (`docker/spark`), the
same rule `memory.store`'s Delta backend follows.
