# chunker

SAS semantic chunker, dependency batcher, and LangChain pipeline — the three
layers that turn Base SAS source into LLM-ready work items. Each layer is
usable on its own.

1. **Chunker** — splits SAS source into source-preserving semantic chunks
   (DATA steps, PROC steps, macro definitions, …) with extracted metadata.
2. **Batcher** — discovers dataset / macro / macro-variable dependencies
   between chunks (within and across files) and groups inter-dependent chunks
   into batches that must be translated together.
3. **Pipeline** — feeds work items, in dependency order, through a
   LangChain/LangGraph chat model with per-run conversational memory. Every LLM
   call is made per `SasBatch`: before a run the batcher's ordered items are
   coalesced (`coalesce_into_batches`) so each dependency batch is one call and
   consecutive independent singletons are packed into synthetic `merged-NNN`
   batches (≤ `max_merged_chunks` members) — fewer, larger requests.

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
librefs or exact `libref.member` names) via `parse_databricks_mapping_csv`;
`SasLLMPipeline` accepts `databricks_mapping` directly, or
`databricks_mapping_sharepoint="path/in/library.csv"` to load that CSV from
the configured SharePoint document library (see `app_config.sharepoint`) at
construction, with the explicit dict winning per key when both are given.

End-to-end through an LLM (`SasLLMPipeline` is imported lazily so `langchain`
is only needed when you use it):

```python
from chunker import SasLLMPipeline

pipeline = SasLLMPipeline(model="claude-sonnet-4-5")
outputs  = pipeline.run_files(["macros.sas", "etl.sas", "reports.sas"])
```

## Package layout

| File | Role |
|------|------|
| `models.py` | Pydantic models: `SasChunk` (+`Kind`), `SasChunkMetadata`, `SasChunkResult`, `SasCorpus`, `SasBatch`, `SasBatchResult`, `SasDiagnostic` (+`Severity`). |
| `keywords.py` | SAS keyword catalogues transcribed from the SAS docs (reserved macro words, autocall macros, function / CALL-routine dictionaries) + the patterns compiled from them. Pure data; no package imports, no logging. |
| `scanner.py` | Lexical layer: `_Unit` / `_Region` parse primitives, the statement classifier (`_classify`), text normalisation / sanitisation, line-offset helpers, and the `_Deadline` / `_ParseWatchdog` stuck-parser machinery. |
| `metadata.py` | Per-chunk semantic extraction: `_metadata_for`, `_io_for` (directed dataset I/O), `_macro_body_io` (literal vs parameterised body refs), symput / SQL-INTO / CALL EXECUTE extractors, `_merge_meta`, and the extraction regex catalogue. |
| `chunker.py` | `SasSemanticChunker` orchestration (scan → group → build chunks, oversized-split with overlap). |
| `batcher.py` | `_EdgeDiscovery` + Union-Find grouping, weak-edge resolution, context absorption, batch construction. `SasChunkBatcher` is a one-file convenience over `MultiFileBatcher`. |
| `pipeline.py` | `SasLLMPipeline`: chunk/batch prompt formatting and the LangGraph `StateGraph` wiring. |
| `pipeline_constants.py` | Prompt templates (importable without langchain installed). |
| `_repl.py` | `print_iterable` REPL helper (imported by nothing). |

**Import direction is strictly downward:** `keywords` and `models` import
nothing from the package; `scanner` and `metadata` import from them; `chunker.py`
imports from all four; `batcher` imports from `keywords`, `metadata`, `models`;
`pipeline` sits on top and is the **only** module that imports `memory.store`,
`memory.relevance`, and `llm_client`.

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

## Pipeline and memory

`SasLLMPipeline` compiles a one-node LangGraph `StateGraph(MessagesState)`. The
model node loads the thread's history from `KVChatMessageHistory`, runs
`_trim | prompt | LLMClient`, and persists exactly the prompted message plus the
response in one bulk `add_messages` write (trimming only limits what is
*prompted*; storage keeps every turn).

Prompted-history trimming has two modes: the default `window_k` recency window,
or — when a `memory.relevance.RelevantHistorySelector` is passed as
`history_selector` — relevance-based selection (see the [memory README](../memory/README.md)).
All items of one `run_file` / `run_text` / `run_files` call share one thread
(`thread_id = "run::<source ids>"`), so the LLM sees the run's accumulated
context batch by batch. Those calls send `SasBatch` objects only:
`coalesce_into_batches` first merges the run's standalone singleton chunks into
`merged-NNN` batches (capped at `max_merged_chunks`), so the model is never
prompted with a bare `SasChunk`. The mapping is deterministic, so resume and
`fork_run` reproduce the same batch ids. `llm_client.LLMClient` owns model
construction (temperature, output-token cap, an optional proactive rate
limiter — on for the `from_ai_gateway` credential path) and invocation
(input-token budget, transient-error retry that honors a gateway
`Retry-After`); an injected `llm` still gets the retry / budget layers.

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
5. **The LangGraph graph is compiled *without* a checkpointer, on purpose.**
   Durable per-thread persistence lives in the KV `msg::` row schema that
   `snapshot()`, `prune_before()`, and `list_threads()` depend on. A
   `BaseCheckpointSaver` would store full state blobs per turn (O(n²) growth in
   the Delta table) and duplicate the canonical store. One graph invocation is
   one conversational turn — the node prompts with, and persists, exactly the
   last state message.
6. **`SasBatch.reason` strings and item ordering are pinned by tests.**
   Edge-emission order is observable output, not an implementation detail.
7. **`_RESERVED_WORDS` is Appendix 1 verbatim (94 words).** Genuine macro
   functions missing from Appendix 1 go in `_ADDITIONAL_MACRO_FUNCTION_WORDS`,
   and SAS-provided autocall macros in `_STANDARD_AUTOCALL_MACROS` — the three
   sets have distinct, citable identities and distinct consumers; do not fold
   them together.

## Logging

f-string messages everywhere (never lazy `%`-style). Per-iteration debug logs
inside parse/batch loops are guarded with `if logger.isEnabledFor(logging.DEBUG):`
so the f-string is never built when DEBUG is off; per-call entry/exit and
LLM-paced logs are unguarded. Logger names follow modules: `chunker.chunker`,
`chunker.scanner`, `chunker.metadata`, `chunker.batcher`, `chunker.pipeline`.
