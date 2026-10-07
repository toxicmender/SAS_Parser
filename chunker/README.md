# chunker

SAS semantic chunker and dependency batcher — the two layers that turn Base
SAS source into LLM-ready work items. Each layer is usable on its own.

1. **Chunker** — splits SAS source into source-preserving semantic chunks
   (DATA steps, PROC steps, macro definitions, …) with extracted metadata.
2. **Batcher** — discovers dataset / macro / macro-variable dependencies
   between chunks (within and across files) and groups inter-dependent chunks
   into batches that must be translated together.

The LLM orchestration layer that consumes these work items lives in the
top-level [`pipeline` package](../pipeline/README.md), which is where it moved
out of this package to.
For the whole-system view (including `llm_client` and `memory`), see the
repository [Architecture.md](../Architecture.md).

## Quick start

Single file:

```python
from chunker import SasSemanticChunker, SasChunkBatcher

chunker = SasSemanticChunker()
result  = chunker.chunk_file("program.sas")

batcher = SasChunkBatcher()
batches = batcher.batch(result)
```

Multiple files (cross-file dependencies resolved):

```python
from chunker import SasSemanticChunker, SasCorpus
from chunker.batcher import MultiFileBatcher

chunker = SasSemanticChunker()
corpus  = SasCorpus(file_results=[
    chunker.chunk_file("macros.sas"),
    chunker.chunk_file("etl.sas"),
    chunker.chunk_file("reports.sas"),
])
result = MultiFileBatcher().batch(corpus)

# Or the convenience factory:
corpus, result = MultiFileBatcher.from_files(["macros.sas", "etl.sas", "reports.sas"])

for item in result.all_ordered_items:
    ...  # SasBatch or SasChunk, cross-file batches included
```

Databricks target names (opt-in): pass `databricks_mapping` to either batcher
(or call `replace_dataset_names` on an existing result) to rewrite the emitted
SAS dataset names to Unity Catalog `catalog.schema.table` names. Batching runs
entirely on the SAS names — grouping, `reason` strings, and `required_librefs`
are identical with or without a mapping.

```python
from chunker import SasChunkBatcher

batcher = SasChunkBatcher(databricks_mapping={
    "work":         "dev.staging",             # libref → catalog.schema
    "sales":        "prod.sales",
    "sales.orders": "prod.sales.orders_v2",    # exact override, wins over libref
})
result = batcher.batch(chunk_result)
result.batches[0].output_datasets  # ['dev.staging.clean', ...]
```

Names created via DATA headers, SET/MERGE renames, and PROC `OUT=`/`OUTPUT
OUT=` all reach the mapper through the extracted metadata — including the ones
written as `&refs`, which are resolved to real names before batching (see
[Names spelled through macro variables](#names-spelled-through-macro-variables)).
A dataset name *stored* in a macro variable (`%let ds = mylib.orders;`,
including the `%global`/`%local` + `%let` pattern) is additionally rewritten in
the chunk *text*, since the rewritten metadata fields hold the names steps
read and write, not the `%let` values they were assembled from. A name that
never resolved keeps its `&` and is never mapped: the mapping vocabulary
cannot address it.

The mapping can also come from a two-column CSV (`sas_name,databricks_name` —
librefs or exact `libref.member` names) via `parse_databricks_mapping_csv`.
That parser is pure and lives here; *fetching* a mapping is `xref`'s job, since
this package stays network-free — `xref.sourcing.mappings(app)` reads the
SharePoint XREF list and `xref.sourcing.load_databricks_mapping_sharepoint(path)`
a CSV in the document library. Pass the resulting dict to either batcher or to
`pipeline.SasLLMPipeline(databricks_mapping=...)`; merging a loaded CSV under
explicit overrides is the caller's one-liner (`{**loaded, **overrides}`).

For running the work items end-to-end through an LLM, see the
[`pipeline` README](../pipeline/README.md).

## Package layout

| File | Role |
|------|------|
| `models.py` | Pydantic models: `SasChunk` (+`Kind`), `SasChunkMetadata`, `SasChunkResult`, `SasCorpus`, `SasBatch`, `SasBatchResult`, `SasDiagnostic` (+`Severity`), `SasPathRef` (+`PathLocation`), `SasEngineRef`, `SasDbTableRef` (+`DbTableAccess`, `DbTableVia`). |
| `paths.py` | Where a physical path appears in SAS syntax — `PATH_STATEMENTS`, `classify_location`, `extract_paths`. The **single owner** of that grammar: `xref.pre` imports it to rewrite the same statements. |
| `keywords.py` | SAS keyword catalogues transcribed from the SAS docs (reserved macro words, autocall macros, function / CALL-routine dictionaries, and `SAS_FUNCTION_CATEGORIES`) + the patterns compiled from them. Pure data; no package imports, no logging. |
| `scanner.py` | Lexical layer: `_Unit` / `_Region` parse primitives, the statement classifier (`_classify`), text normalisation / sanitisation, line-offset helpers, and the `_Deadline` / `_ParseWatchdog` stuck-parser machinery. |
| `macro_vars.py` | Macro-variable values and reference expansion: `let_values` (the `%LET` symbol table), `resolve_refs` (`&name` / `&name.` / `&&name&i`), and `DS_REF_TOKEN` — the single definition of a dataset token that may embed `&refs`. Pure; no package imports. |
| `passthrough.py` | SQL pass-through — `CONNECT TO` / `CONNECTION TO` / `EXECUTE … BY` / `DISCONNECT` and the native-SQL table scan: `scan_pass_through` (tables + the spans to mask), `mask`, `db_table_ref` (the one `SasDbTableRef` builder). The **single owner** of that grammar. |
| `metadata.py` | Per-chunk semantic extraction: `_metadata_for`, `_io_for` (directed dataset I/O), `_macro_body_io` (literal vs parameterised body refs), symput / SQL-INTO / CALL EXECUTE extractors, `_merge_meta`, the extraction regex catalogue, and the whole-list resolution passes — `resolve_macro_var_refs`, `resolve_db_librefs`, composed in order by `resolve_references` (and across files by `resolve_corpus_references`). |
| `chunker.py` | `SasSemanticChunker` orchestration (scan → group → build chunks, oversized-split with overlap). |
| `batcher.py` | `_EdgeDiscovery` + Union-Find grouping, weak-edge resolution, context absorption, batch construction. `SasChunkBatcher` is a one-file convenience over `MultiFileBatcher`. |
| `_repl.py` | `print_iterable` REPL helper (imported by nothing). |
| `pipeline.py`, `pipeline_constants.py`, `response_models.py`, `notebook.py` | **Deprecated shims** re-exporting from the top-level `pipeline` package, where these modules now live. |

> **Budgets here stay in words.** `sas_chunker.min_words`/`max_words` size SAS
> *source* into semantic units — a question about where a step ends, not about
> what a prompt costs. `prompt_builder` budgets in tokens because it is filling
> a prompt; `chunker.batcher` then packs these units by token cost
> (`pipeline.max_merged_tokens`) on top. Two different questions, two units.

**Import direction is strictly downward:** `keywords`, `macro_vars` and `models`
import nothing from the package; `scanner` and `paths` import from them;
`passthrough` imports from those; `metadata` imports from all of them;
`chunker.py` imports from all of them; `batcher` imports from `keywords`,
`metadata`, `models`.
The package imports nothing from `memory`, `llm_client`, `prompt_builder`, or
`pipeline` — it is a leaf the `pipeline` package builds on.

## Chunking model

The chunker is deliberately a **statement scanner + regex extractor**, not a
grammar-driven parser. It degrades gracefully on malformed source (emitting
`SasDiagnostic`s such as `UNCLOSED_MACRO`, `UNRECOGNIZED_SOURCE_REGION`,
`PARSER_TIMEOUT`) instead of failing, which a strict parser would not.
Replacing it with a full SAS grammar would be a rewrite, not a simplification —
this is a considered decision, not an accident.

- **Block collection rule:** only a new DATA / PROC / `%MACRO` header or an
  explicit `RUN;` / `QUIT;` / `%MEND` closes the current block. FORMAT, OPTIONS,
  LIBNAME, ODS, etc. inside a block body are collected, never treated as
  boundaries. A `%MACRO` block closes only on its own (nesting-balanced)
  `%MEND`.
- **Oversized splits:** a region exceeding `max_words` yields a *parent* chunk
  (full text) plus overlapping *child* chunks (`parent_id` set). The
  parent/child text redundancy is intentional context for the LLM. Child
  metadata is merged with the parent's via `_merge_meta`.
- **Stuck-parser protection** (`SasSemanticChunker(timeout=...)`): a wall-clock
  **deadline** gives a graceful partial-result exit at statement boundaries; a
  background **watchdog** thread names the stuck phase in the logs (WARNING →
  ERROR) for the one case the deadline cannot cover — a hang inside a single
  un-interruptible C-level regex call (catastrophic backtracking on hostile
  source). Pass `timeout=None` to disable both and parse unbounded.

### Metadata: stored vs computed

`SasChunkMetadata` stores one field per concept. Five views are **computed
fields** derived at access time, not stored:

- `referenced_automatic_vars` — the `&sys*` subset of `referenced_macro_vars`
  (all SAS automatic variables carry the reserved `SYS` prefix; see
  `models._is_automatic_macro_var`).
- `consumes_macrovars` — `referenced_macro_vars` minus automatics minus the
  macro's own `macro_param_names` (call-site-resolved, so never a corpus-level
  dependency).
- `physical_paths` / `remote_paths` / `email_refs` — the `external_refs` entries
  whose `location` is `FILESYSTEM` / `REMOTE` / `EMAIL`.
- `unresolved_dataset_refs` — the dataset names across `referenced_datasets`,
  the I/O lists and `body_literal_*` that still hold a `&` (see below).

Both appear in `model_dump()` but are silently ignored as constructor kwargs,
and they do not appear in `__str__`. `defines_macros` / `invokes_macros` are the
single authoritative macro fields (`invokes_macros` includes CALL
EXECUTE-invoked macros).

Names are lowercased at extraction; quoted physical paths keep a leading `'` so
they can never collide with identifiers.

### External references

`external_refs` is one stored list of `SasPathRef` records — every location a
chunk's statements name, whatever kind of place it is. `paths.py` recognises
`LIBNAME`, `FILENAME`, `INFILE` / `FILE`, `%INCLUDE`, PROC IMPORT/EXPORT's
`datafile=` / `outfile=`, ODS `file=` / `path=`, and `options sasautos=`.

A `FILENAME` device keyword redirects the same syntax somewhere that is not the
filesystem, so each record carries a `PathLocation` — `FILESYSTEM`, `REMOTE`
(FTP, URL, …), `EMAIL`, `PIPE` (a shell command), or `DEVICE` for a keyword this
module does not know. An unknown device is never silently treated as a path.

One list rather than one per kind: one scan to keep correct, one merge rule to
keep honest, and the per-kind views above for consumers. `includes` is the
`%INCLUDE` slice of the same scan, not a second definition of where an include
path lives.

### Names spelled through macro variables

Production SAS names libraries and tables with macro variables far more often
than it writes them out:

```sas
%let lname  = xwrk;
%let suf    = batch_med;
%let table1 = &suf;

data &table1;
  set &lname..&table1;
run;
```

Every dataset position is scanned with `macro_vars.DS_REF_TOKEN`, which admits
`&refs`, so those names are seen at all; `metadata.resolve_macro_var_refs` then
walks the built chunks in source order, accumulating each `%LET` value and
expanding the references of the chunks that follow. The step above reports
`work.batch_med` out, `xwrk.batch_med` in, and `xwrk` as a referenced libref.
The delimiter dot is SAS's: in `&lname..&table1` one dot ends the reference and
the other separates libref from member. Chains (`&table1` → `&suf` →
`batch_med`) resolve in full, a `%LET` value is expanded where it stands (so
`%let x = &x.b;` appends rather than recursing), and the indirect `&&ds&i`
idiom is rescanned when — and only when — the rescan resolves it completely.

**What does not resolve is reported exactly as written**, never dropped and
never guessed at:

```sas
%let table_reg_excl_spd = &lib_out_spd..cia_hso_excl;   /* &lib_out_spd unknown */
```

`referenced_datasets` gains `&lib_out_spd..cia_hso_excl` and
`referenced_librefs` gains `&lib_out_spd`; `SasChunkMetadata.unresolved_dataset_refs`
and `SasBatch.unresolved_dataset_refs` are the views that separate those from
the resolved names. A reference whose value only exists at run time is still a
dependency, and saying "this batch reads a library called `&lib_out_spd`" is
strictly better than reporting no library at all. For the same reason a name
holding a `&` is never `work.`-canonicalised — `&suf` may well resolve to a
two-level name — and the Databricks mapping skips it.

A `%LET` whose value is *shaped* like a dataset reference
(`%let table_demogr = datacia.member_demographic;`) contributes to
`referenced_datasets` / `referenced_librefs` on sight. It is provenance only,
never I/O: a `%LET` reads and writes nothing; the step that uses
`&table_demogr` does.

Two scope rules keep this from over-reaching. A macro's own parameters shadow
the table, so `&ds` inside `%macro m(ds);` stays a `body_param_*` entry the
batcher resolves per call site rather than picking up a corpus-level
`%let ds = ...;`. And a `%LET` inside a `%MACRO` body stays local to that chunk,
since whether it ever executes depends on a call. Resolution runs once per file
in `chunk_text` and again over the flattened corpus in `MultiFileBatcher`,
which is what lets a `%LET` in one file name a dataset another file reads.

### Database tables (SQL pass-through and database LIBNAMEs)

SAS reaches a database's own tables two ways, and both produce
`SasDbTableRef` records on `SasChunkMetadata.db_tables` (rolled up as
`SasBatch.db_tables`) — the table in the database's terms, linked to the SAS
dataset the read lands in:

```sas
proc sql;
connect to oracle (user=&ora_user password=&ora_pass path=&ora_path);
create table nonip as select * from connection to oracle
(select cov_month from edw_export.current_nonip where table_cd='MED');
quit;
/* → oracle:edw_export.current_nonip → work.nonip (read via connection_to oracle) */

libname edw oracle path=EDWPRO schema=fr_dm;
data work.accts; set edw.accounts; run;
/* → oracle:fr_dm.accounts → work.accts (read via libname edw) */
```

**Explicit pass-through** is `passthrough.py`'s grammar: `CONNECT TO` (alias →
engine and options, in statement order), `CONNECT USING`, `(FROM|JOIN)
CONNECTION TO` (reads), `EXECUTE … BY` (writes and reads), `DISCONNECT`. The
native SQL gets its own token walk — comma FROM lists, joins, subqueries, table
functions, CTE names and `DUAL` excluded, `EXTRACT(… FROM …)`, Oracle `'…'`
literals and `--` comments, `"QUOTED"` identifiers, `@dblink`, and every
DDL/DML write form. Every *dataset* scan in `_metadata_for` runs on text where
the pass-through spans are **masked**, so `from connection to oracle` stops
being the dataset `work.connection`, `disconnect from oracle` stops being
`work.oracle`, and the Oracle owner stops being a SAS libref a batch then
"requires". Macro invocations are still read from the unmasked text: SAS
resolves them before the native SQL is sent.

**Database LIBNAMEs** are `metadata.resolve_db_librefs`: a source-order walk
keeping the engine LIBNAMEs in force (`schema=` resolved against `%LET` where
the LIBNAME stands; `libname x clear;` or a path rebind ends one; one inside a
`%MACRO` body *does* bind, since connection macros are how production SAS hides
credentials). A SAS name under such a libref keeps its place in the I/O lists —
SAS code does name `edw.accounts` — and also gains a `via=libname` record.
`CONNECT USING` records get their engine and options here.

`resolve_references` runs the macro pass, then this one — the one order —
from `chunk_text` (per file) and `MultiFileBatcher` (corpus-wide).
`resolve_corpus_references(corpus)` does the same for callers that do not
batch, which is how `data_hydration` and `complexity --hydration` see a LIBNAME
in `setup.sas` reach the reads in `job.sas`.

Options ride on each record, so a consumer never joins by alias. The one-line
`str()` form — what reaches the LLM prompt — leaves them out, because that is
where credentials live. No batching edges are added: two reads of one Oracle
table are not a dependency between them.

## Batching model

`_EdgeDiscovery` builds producer indices, then walks the flattened corpus once,
emitting typed edges:

| Edge kind | Tier | Meaning |
|-----------|------|---------|
| `dataset_flow` | strong | chunk reads a dataset a preceding chunk wrote |
| `macro_body_dataset` | strong | call-site-resolved parameterised macro-body I/O |
| `macro_invocation` | weak | chunk invokes a macro defined elsewhere |
| `macro_var_flow` | weak | chunk reads `&name` a preceding chunk created |
| `macro_arg_dataset` | weak | dataset name appears in a macro call's argument |

Strong edges union their endpoints in a Union-Find immediately. Weak edges are
resolved afterwards at *component* granularity: a producer feeding exactly one
component is absorbed into it; a producer feeding two or more otherwise-independent
components is promoted into a single **global-context batch**, emitted first
(`is_global_context=True`) — so one widely-used `%let` or utility macro cannot
fuse the whole corpus into one mega-batch. OPTIONS / GLOBAL_STATEMENT (and
optionally comment) chunks are then absorbed into the following substantive
chunk's component, same-file only.

Dataset names are canonicalised (`_canon_ds`): one-level names become
`work.<name>` (a `USER_LIBRARY_ASSIGNED` diagnostic flags the case where that
rewrite is inexact); a name still holding a `&` is left alone, since its libref
is not knowable yet. Consumers link to the **nearest preceding producer** in
corpus order — the state a sequential SAS session would actually read — so
unrelated jobs reusing `work.tmp` stay separate.

## Load-bearing invariants

Things that look like implementation details but are contracts. Breaking any of
these silently changes behavior.

1. **Edge discovery is one walk, in corpus order.**
   `_EdgeDiscovery._resolve_macro_body` mutates `produces_ds` mid-walk: a macro
   call site's resolved outputs are registered as producers at the moment the
   call is visited, which implements "a macro's output exists only once the call
   has executed" under nearest-preceding-producer bisection. Splitting the edge
   families into separate corpus walks would let a consumer link to a producer
   that does not exist yet at its position — or miss one that does.
2. **Producer lists stay sorted by global index.** The nearest-preceding lookups
   are `bisect_left` over `produces_ds[name]`; mid-walk registration therefore
   uses `insort`, never `append`.
3. **`output_datasets` is insertion-ordered, never sorted.**
   `_resolve_implicit_datasets` treats `output_datasets[-1]` as "the last
   dataset named" when resolving `_LAST_` / `_DATA_` / missing-`data=` references.
   (The list-merge in `_merge_meta` is the deliberate exception.)
4. **Every `SasChunkMetadata` field must have a merge rule.** `_merge_meta`
   dispatches on field annotation (`list[str]` → sorted union,
   `list[SasPathRef]` → union ordered by `_path_ref_sort_key`,
   `dict[str, str]` → merged with the child's entry winning, `bool` → OR,
   `str | None` → child-or-parent, `_MERGE_PARENT_WINS` → parent's value —
   which includes `db_tables`, since a `CONNECT` and the `CONNECTION TO` using
   its alias can land in different split slices) and
   raises `TypeError` for anything else. The default-instance test in
   `tests/test_chunker.py` trips the guard for every stored field, so a new field
   shape cannot ship without a conscious decision.
5. **`SasBatch.reason` strings and item ordering are pinned by tests.**
   Edge-emission order is observable output, not an implementation detail.
6. **`_RESERVED_WORDS` is Appendix 1 verbatim (94 words).** Genuine macro
   functions missing from Appendix 1 go in `_ADDITIONAL_MACRO_FUNCTION_WORDS`,
   and SAS-provided autocall macros in `_STANDARD_AUTOCALL_MACROS` — the three
   sets have distinct, citable identities and distinct consumers; do not fold
   them together.
7. **`data_step_statements` is advisory and DATA-step-only.** It reports which
   statements a step uses — logic statements (`merge`, `by`, `retain`, `array`,
   `output`, `set`, `update`, `modify`, `where`, `infile`, `do`) and
   declaration statements (`label`, `format`, `length`, which carry the
   documentation a migration should preserve) — plus four constructs the scan
   derives rather than reads: `retain` for a sum statement, `subsetting_if` for
   an `if <expr>;` that drops rows, `set_multi` for a concatenating
   `SET a b;`, and `dataset_option` for `keep=` and friends. A PROC's statements are already identified by `proc_name`, so the
   scan skips non-DATA chunks. Never gates chunking or batching; the consumer
   is `prompt_builder`'s `[when: statement:...]` scope, which is what lets
   guidance fire on the steps that raise a problem instead of on every step.
   `SAS_DATA_STEP_STATEMENT_TOKENS` publishes the full vocabulary.
8. **`global_statement_keyword` names the statement that opened the chunk.**
   `SAS_GLOBAL_STATEMENT_TOKENS` publishes its full vocabulary, for the same
   reason: it is what `[when: global_statement:...]` scopes on. Alongside the
   macro-variable and output statements it carries the **host-command escapes**
   (`SAS_HOST_COMMAND_TOKENS`: `x`, `systask`, `sysexec`, `waitfor`), which are
   grouped by the translation question they raise rather than by syntax class —
   `%SYSEXEC` is a macro statement and `SYSTASK`/`WAITFOR` are host-documented,
   but they all concern handing a string to the operating system and waiting on
   it. ⚠️ `X` is recognised only with its quoted
   argument or as the whole bare statement, because `x` is also one of the
   commonest SAS variable names; `x = 1;` and `x + 1;` must stay DATA step body.
9. **`SAS_FUNCTION_CATEGORIES` is advisory and deliberately partial.** It maps
   a function or routine name to its family in *SAS Functions and CALL Routines
   by Category*, and its only consumer is `prompt_builder`'s `[category: ...]`
   instruction scope (reached via `pipeline.prompting._constructs_for_item`, so
   `prompt_builder` still imports nothing from here). Only families that carry
   translation guidance are mapped — an unmapped name contributes no category
   key, which is the same as having no rule for it. Every mapped name must
   exist in `_SAS_FUNCTIONS` or `_SAS_CALL_ROUTINES`;
   `tests/test_bundled_instructions.py` enforces both ends, so a typo cannot
   silently create a category nothing can ever match.
10. **An unresolved macro reference stays verbatim, and stays reported.**
   `resolve_macro_var_refs` never invents a name for a reference the corpus
   does not assign, never drops it, and never rewrites the spelling of the part
   it could not resolve — which is why expansion substitutes over the whole
   string instead of re-rendering it from tokens (re-rendering `&lname` before
   a `.` would have to re-escape SAS's delimiter dot, and `&lname.batch_med`
   means something else than `&lname..batch_med`). The two directions this can
   fail are both silent: guessing produces a dataset name nothing in the corpus
   has, and dropping produces a step that appears to read nothing.
   `_canon_ds` therefore leaves `&`-bearing names alone, and `_map_ds` refuses
   to map them.
11. **Native SQL is never scanned as SAS.** Every dataset scan in
   `_metadata_for` (`_DATASET_RE`, `_SQL_*`, `_io_for`, `_macro_body_io`, …)
   reads the text with `scan_pass_through`'s spans masked. A new SAS-side
   dataset scan must read the masked `mt_ds`/`cf_ds` too, or the
   `work.connection` / Oracle-schema-as-libref misreadings come straight back —
   silently, since the chunk still reports *a* dataset. Scans for macro names,
   macro variables and functions keep the unmasked text on purpose.

## Logging

f-string messages everywhere (never lazy `%`-style). Per-iteration debug logs
inside parse/batch loops are guarded with `if logger.isEnabledFor(logging.DEBUG):`
so the f-string is never built when DEBUG is off; per-call entry/exit logs
are unguarded. Logger names follow modules:
`chunker.chunker`, `chunker.scanner`, `chunker.metadata`, `chunker.batcher`.
