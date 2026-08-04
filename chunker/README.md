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
OUT=` all reach the mapper through the extracted metadata; a dataset name
stored in a macro variable (`%let ds = mylib.orders;`, including the
`%global`/`%local` + `%let` pattern) is additionally rewritten in the chunk
*text*, since a `%let` value never appears in the metadata dataset lists.

The mapping can also come from a two-column CSV (`sas_name,databricks_name` —
librefs or exact `libref.member` names) via `parse_databricks_mapping_csv`,
or straight from the configured SharePoint document library (see
`app_config.sharepoint`) via `load_databricks_mapping_sharepoint(path)` —
the SharePoint client is imported lazily inside that call. Pass the resulting
dict to either batcher or to `pipeline.SasLLMPipeline(databricks_mapping=...)`;
merging a loaded CSV under explicit overrides is the caller's one-liner
(`{**loaded, **overrides}`).

For running the work items end-to-end through an LLM, see the
[`pipeline` README](../pipeline/README.md).

## Package layout

| File | Role |
|------|------|
| `models.py` | Pydantic models: `SasChunk` (+`Kind`), `SasChunkMetadata`, `SasChunkResult`, `SasCorpus`, `SasBatch`, `SasBatchResult`, `SasDiagnostic` (+`Severity`). |
| `keywords.py` | SAS keyword catalogues transcribed from the SAS docs (reserved macro words, autocall macros, function / CALL-routine dictionaries) + the patterns compiled from them. Pure data; no package imports, no logging. |
| `scanner.py` | Lexical layer: `_Unit` / `_Region` parse primitives, the statement classifier (`_classify`), text normalisation / sanitisation, line-offset helpers, and the `_Deadline` / `_ParseWatchdog` stuck-parser machinery. |
| `metadata.py` | Per-chunk semantic extraction: `_metadata_for`, `_io_for` (directed dataset I/O), `_macro_body_io` (literal vs parameterised body refs), symput / SQL-INTO / CALL EXECUTE extractors, `_merge_meta`, and the extraction regex catalogue. |
| `chunker.py` | `SasSemanticChunker` orchestration (scan → group → build chunks, oversized-split with overlap). |
| `batcher.py` | `_EdgeDiscovery` + Union-Find grouping, weak-edge resolution, context absorption, batch construction. `SasChunkBatcher` is a one-file convenience over `MultiFileBatcher`. |
| `_repl.py` | `print_iterable` REPL helper (imported by nothing). |
| `pipeline.py`, `pipeline_constants.py`, `response_models.py`, `notebook.py` | **Deprecated shims** re-exporting from the top-level `pipeline` package, where these modules now live. |

**Import direction is strictly downward:** `keywords` and `models` import
nothing from the package; `scanner` and `metadata` import from them; `chunker.py`
imports from all four; `batcher` imports from `keywords`, `metadata`, `models`.
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

`SasChunkMetadata` stores one field per concept. Two views are **computed
fields** derived at access time, not stored:

- `referenced_automatic_vars` — the `&sys*` subset of `referenced_macro_vars`
  (all SAS automatic variables carry the reserved `SYS` prefix; see
  `models._is_automatic_macro_var`).
- `consumes_macrovars` — `referenced_macro_vars` minus automatics minus the
  macro's own `macro_param_names` (call-site-resolved, so never a corpus-level
  dependency).

Both appear in `model_dump()` but are silently ignored as constructor kwargs,
and they do not appear in `__str__`. `defines_macros` / `invokes_macros` are the
single authoritative macro fields (`invokes_macros` includes CALL
EXECUTE-invoked macros).

Names are lowercased at extraction; quoted physical paths keep a leading `'` so
they can never collide with identifiers.

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
rewrite is inexact). Consumers link to the **nearest preceding producer** in
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
   dispatches on field annotation (`list[str]` → sorted union, `bool` → OR,
   `str | None` → child-or-parent, `_MERGE_PARENT_WINS` → parent's value) and
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

## Logging

f-string messages everywhere (never lazy `%`-style). Per-iteration debug logs
inside parse/batch loops are guarded with `if logger.isEnabledFor(logging.DEBUG):`
so the f-string is never built when DEBUG is off; per-call entry/exit logs
are unguarded. Logger names follow modules:
`chunker.chunker`, `chunker.scanner`, `chunker.metadata`, `chunker.batcher`.
