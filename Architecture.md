# Architecture

SAS_Parser turns Base SAS source into LLM-ready work items. It has three
layers, each usable on its own:

1. **Chunker** — splits SAS source into source-preserving semantic chunks
   (DATA steps, PROC steps, macro definitions, …) with extracted metadata.
2. **Batcher** — discovers dataset/macro/macro-variable dependencies between
   chunks (within and across files) and groups inter-dependent chunks into
   batches that must be translated together.
3. **Pipeline** — feeds batches and singleton chunks, in dependency order,
   through a LangChain/LangGraph chat model with per-run conversational
   memory persisted to a KV store (in-memory dict locally, Databricks Delta
   in production).

```
                 +----------------------+
  SAS source(s) ─▶ SasSemanticChunker   │──▶ SasChunkResult (per file)
                 +----------------------+
                          │  SasCorpus wraps N file results
                          ▼
                 +----------------------+
                 │ MultiFileBatcher /   │──▶ SasBatchResult
                 │ SasChunkBatcher      │    (batches + singletons)
                 +----------------------+
                          │  all_ordered_items
                          ▼
                 +----------------------+       +--------------------+
                 │ SasLLMPipeline       │──────▶│ memory.short_mem   │
                 │ (LangGraph graph,    │ turns │ (KV chat history)  │
                 │  one thread per run) │       +--------------------+
                 +----------------------+
                          ▼
                LLM responses, one per batch/singleton
```

## Package layout

```
chunker/
  models.py             Pydantic models: SasChunk(+Kind), SasChunkMetadata,
                        SasChunkResult, SasCorpus, SasBatch, SasBatchResult,
                        SasDiagnostic(+Severity)
  keywords.py           SAS keyword catalogues transcribed from the SAS docs
                        (reserved macro words, autocall macros, function /
                        CALL-routine dictionaries) + patterns compiled from
                        them. Pure data; no package imports, no logging.
  scanner.py            Lexical layer: _Unit/_Region parse primitives, the
                        statement classifier (_classify), text normalisation
                        and sanitisation, line-offset helpers, and the
                        _Deadline/_ParseWatchdog stuck-parser machinery.
  metadata.py           Per-chunk semantic extraction: _metadata_for, _io_for
                        (directed dataset I/O), _macro_body_io (literal vs
                        parameterised body refs), symput / SQL-INTO / CALL
                        EXECUTE extractors, _merge_meta, and the extraction
                        regex catalogue.
  chunker.py            SasSemanticChunker orchestration (scan → group →
                        build chunks, oversized-split with overlap).
  batcher.py            _EdgeDiscovery + Union-Find grouping, weak-edge
                        resolution, context absorption, batch construction.
                        SasChunkBatcher is a one-file convenience over
                        MultiFileBatcher.
  pipeline.py           SasLLMPipeline: formatting of chunk/batch prompts and
                        the LangGraph StateGraph wiring.
  pipeline_constants.py Prompt templates.
  _repl.py              print_iterable REPL helper (imported by nothing).

memory/
  short_mem.py          SparkKVStore façade over two backends
                        (_InMemoryBackend dict / _DeltaBackend Spark+Delta),
                        KVChatMessageHistory (BaseChatMessageHistory),
                        ThreadMemoryManager, KVMemoryStore, and the
                        DatabricksMemory entry-point façade.
```

Import direction is strictly downward: `keywords` and `models` import
nothing from the package; `scanner` and `metadata` import from them;
`chunker.py` imports from all four; `batcher` imports from `keywords`,
`metadata`, `models`; `pipeline` sits on top and is the **only** module that
imports `memory.short_mem`. `memory` never imports `chunker`.

## Chunking model

The chunker is deliberately a **statement scanner + regex extractor**, not a
grammar-driven parser. It degrades gracefully on malformed source (emitting
`SasDiagnostic`s such as `UNCLOSED_MACRO`, `UNRECOGNIZED_SOURCE_REGION`,
`PARSER_TIMEOUT`) instead of failing, which a strict parser would not.
Replacing it with a full SAS grammar would be a rewrite, not a
simplification — this is a considered decision, not an accident.

- **Block collection rule:** only a new DATA/PROC/%MACRO header or an
  explicit RUN;/QUIT;/%MEND closes the current block. FORMAT, OPTIONS,
  LIBNAME, ODS, etc. inside a block body are collected, never treated as
  boundaries. A %MACRO block closes only on its own (nesting-balanced)
  %MEND.
- **Oversized splits:** a region exceeding `max_words` yields a *parent*
  chunk (full text) plus overlapping *child* chunks (`parent_id` set). The
  parent/child text redundancy is intentional context for the LLM. Child
  metadata is merged with the parent's via `_merge_meta` (see invariants).
- **Stuck-parser protection:** a wall-clock deadline gives a graceful
  partial-result exit at statement boundaries; a watchdog thread names the
  stuck phase in the logs when the parser is wedged inside a C-level regex
  call it cannot preempt.

### Metadata: stored vs computed

`SasChunkMetadata` stores one field per concept. Two views are
**computed fields** derived at access time, not stored:

- `referenced_automatic_vars` — the `&sys*` subset of
  `referenced_macro_vars` (all SAS automatic variables carry the reserved
  `SYS` prefix; see `models._is_automatic_macro_var`).
- `consumes_macrovars` — `referenced_macro_vars` minus automatics minus the
  macro's own `macro_param_names` (call-site-resolved, so never a
  corpus-level dependency).

Consequences: both appear in `model_dump()` but are silently ignored as
constructor kwargs, and they do not appear in `__str__` (which walks
`__dict__`). `defines_macros` / `invokes_macros` are the single
authoritative macro fields (`invokes_macros` includes CALL EXECUTE-invoked
macros).

## Batching model

`_EdgeDiscovery` builds producer indices, then walks the flattened corpus
once, emitting typed edges:

| Edge kind            | Tier   | Meaning |
|----------------------|--------|---------|
| `dataset_flow`       | strong | chunk reads a dataset a preceding chunk wrote |
| `macro_body_dataset` | strong | call-site-resolved parameterised macro-body I/O |
| `macro_invocation`   | weak   | chunk invokes a macro defined elsewhere |
| `macro_var_flow`     | weak   | chunk reads `&name` a preceding chunk created |
| `macro_arg_dataset`  | weak   | dataset name appears in a macro call's argument |

Strong edges union their endpoints in a Union-Find immediately. Weak edges
are resolved afterwards at *component* granularity: a producer feeding
exactly one component is absorbed into it; a producer feeding two or more
otherwise-independent components is promoted into a single **global-context
batch**, emitted first (`is_global_context=True`) — so one widely-used
`%let` or utility macro cannot fuse the whole corpus into one mega-batch.
OPTIONS/GLOBAL_STATEMENT (and optionally comment) chunks are then absorbed
into the following substantive chunk's component, same-file only.

Dataset names are canonicalised (`_canon_ds`): one-level names become
`work.<name>` (a `USER_LIBRARY_ASSIGNED` diagnostic flags the case where
that rewrite is inexact). Consumers link to the **nearest preceding
producer** in corpus order — the state a sequential SAS session would
actually read — so unrelated jobs reusing `work.tmp` stay separate.

## Pipeline and memory

`SasLLMPipeline` compiles a one-node LangGraph `StateGraph(MessagesState)`.
The model node loads the thread's history from `KVChatMessageHistory`, runs
`_trim | prompt | llm` (trimming only limits what is *prompted*; storage
keeps every turn), and persists exactly the prompted message plus the
response in one bulk `add_messages` write. All items of one
`run_file`/`run_text`/`run_files` call share one thread
(`thread_id = "run::<source ids>"`), so the LLM sees the run's accumulated
context batch by batch.

`memory.short_mem` stores everything as namespaced KV rows
(`msg::<thread>::<μs-timestamp>-<rand>` for messages). The
`SparkKVStore` façade owns all JSON (de)serialisation, tag queries, search,
and snapshot/restore; a backend only stores/retrieves/deletes raw rows.
`_InMemoryBackend` (default) is a plain dict and requires neither pyspark
nor a JVM; `_DeltaBackend` requires both and uses MERGE INTO / DELETE FROM
against a Delta table.

## Load-bearing invariants

Things that look like implementation details but are contracts. Breaking
any of these silently changes behavior.

1. **Edge discovery is one walk, in corpus order.**
   `_EdgeDiscovery._resolve_macro_body` mutates `produces_ds` mid-walk: a
   macro call site's resolved outputs are registered as producers at the
   moment the call is visited, which is what implements "a macro's output
   exists only once the call has executed" under nearest-preceding-producer
   bisection. Splitting the edge families into separate corpus walks would
   let a consumer link to a producer that does not exist yet at its
   position — or miss one that does.

2. **Producer lists stay sorted by global index.** The nearest-preceding
   lookups are `bisect_left` over `produces_ds[name]`; mid-walk
   registration therefore uses `insort`, never `append`.

3. **`output_datasets` is insertion-ordered, never sorted.**
   `_resolve_implicit_datasets` treats `output_datasets[-1]` as "the last
   dataset named" when resolving `_LAST_`/`_DATA_`/missing-`data=`
   references. Sorting it breaks that convention (list-merge in
   `_merge_meta` is the deliberate exception: split children lose ordering,
   and implicit resolution operates on unsplit metadata).

4. **Every `SasChunkMetadata` field must have a merge rule.** `_merge_meta`
   dispatches on field annotation (`list[str]` → sorted union, `bool` → OR,
   `str | None` → child-or-parent, `_MERGE_PARENT_WINS` → parent's value)
   and raises `TypeError` for anything else. The default-instance test in
   `tests/test_chunker.py` trips the guard for every stored field, so a new
   field shape cannot ship without a conscious decision. Signature-derived
   fields (`macro_param_names`, `body_param_*`) are parent-wins because only
   the split slice containing the `%MACRO` header can parse them.

5. **The LangGraph graph is compiled *without* a checkpointer, on
   purpose.** Durable per-thread persistence lives in the KV `msg::` row
   schema that `snapshot()`, `prune_before()`, and `list_threads()` depend
   on. A `BaseCheckpointSaver` would store full state blobs per turn
   (O(n²) growth in the Delta table) and duplicate the canonical store.
   Corollary: one graph invocation is one conversational turn — the node
   prompts with, and persists, exactly the last state message.

6. **In-memory mode must stay Spark-free.** `_InMemoryBackend` (and
   therefore `DatabricksMemory()` with no arguments) must import and run
   without pyspark installed; the pyspark requirement lives inside
   `_DeltaBackend.__init__` only. The pipeline never boots a SparkSession
   unless `delta_table` is set.

7. **`SasBatch.reason` strings and item ordering are pinned by tests.**
   Edge-emission order is observable output, not an implementation detail.

8. **`_RESERVED_WORDS` is Appendix 1 verbatim (94 words).** Genuine macro
   functions missing from Appendix 1 go in
   `_ADDITIONAL_MACRO_FUNCTION_WORDS`, and SAS-provided autocall macros in
   `_STANDARD_AUTOCALL_MACROS` — the three sets have distinct, citable
   identities and distinct consumers; do not fold them together.

## Conventions

- **Logging:** f-string messages everywhere (never lazy `%`-style).
  Per-iteration debug logs inside parse/batch loops are guarded with
  `if logger.isEnabledFor(logging.DEBUG):` so the f-string is never built
  when DEBUG is off; per-call entry/exit and LLM-paced logs are unguarded.
  Logger names follow modules: `chunker.chunker`, `chunker.scanner`,
  `chunker.metadata`, `chunker.batcher`, `chunker.pipeline`,
  `memory.short_mem`.
- **Names:** dataset/macro/libref names are lowercased at extraction;
  quoted physical paths keep a leading `'` so they can never collide with
  identifiers.
- **Commits:** conventional-commit style (`refactor(scope): …`), one
  logical change per commit.

## Testing

`tests/` (403 tests) runs without a JVM, network, or API keys: the memory
tests use the in-memory backend, and the pipeline tests inject
`FakeListChatModel`. The Delta backend (`_DeltaBackend`) has **no automated
coverage** in this suite — it requires a live Spark session — so changes to
it need manual verification against Databricks. Behavior-preserving
refactors of the chunker/batcher have historically been verified by
snapshotting full batcher output (batch membership, I/O fields, reason
strings, ordering) on a synthetic multi-file corpus and diffing against the
pre-change code; prefer that over trusting the suite alone for pure
code-motion changes.
