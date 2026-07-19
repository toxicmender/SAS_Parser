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

An optional fourth component, **prompt_builder**, reads reference PDFs (SAS
manuals, target-platform guides) into retrieval-ready instruction chunks and,
when passed to the pipeline, injects per-item guidance relevant to each work
item's constructs — prompted to the LLM but never persisted (see invariant 5).

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
                 │ SasLLMPipeline       │──────▶│ memory.store   │
                 │ (LangGraph graph,    │ turns │ (KV chat history)  │
                 │  one thread per run) │       +--------------------+
                 +----------------------+
                    ▲ ephemeral   │
                    │ guidance    ▼
   +----------------------+   LLM responses, one per batch/singleton
   │ prompt_builder       │◀── reference PDFs (SAS + target manuals)
   │ (PromptBuilder, opt) │
   +----------------------+
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

llm_client/
  client.py             LLMClient / LLMClientConfig: chat-model construction
                        via init_chat_model (temperature, max output tokens,
                        endpoint overrides — base_url / api_key / headers /
                        timeout / model_kwargs, proactive InMemoryRateLimiter)
                        and sync + async invocation (input-token budget ->
                        InputTokenLimitError, transient-error retry with
                        exponential backoff). Imports nothing from chunker
                        or memory.

memory/
  turns.py              Dependency-light turn grouping + approx token count,
                        shared by relevance and summarize (so summarize never
                        imports the bm25s/faiss stack). A leaf module.
  relevance.py          HybridRanker: shared BM25 (bm25s) + optional dense
                        retrieval (LangChain Embeddings + FAISS IndexFlatIP),
                        RRF fusion, optional reranker hook, content-hashed
                        embedding cache — with a stateless per-call mode and an
                        index-once/query-many static-corpus mode — plus
                        DiskCachedEmbeddings (on-disk .npz document-embedding
                        cache). RelevantHistorySelector layers history policy
                        on top: relevance-based selection of prompted history
                        turn pairs, always-keep-last tail, recency fallback,
                        optional max_tokens packing.
                        Imports nothing from chunker or memory.store.
  summarize.py          RollingSummarizer: one rolling summary per thread —
                        turns older than a recency tail fold monotonically
                        into a KV-stored summary (prompted, never persisted).
                        Store is duck-typed; imports only memory.turns.
  store.py              KVStore façade over two backends
                        (_InMemoryBackend dict / _DeltaBackend Spark+Delta),
                        KVChatMessageHistory (BaseChatMessageHistory, with
                        optional after-write retention), ThreadMemoryManager
                        (incl. fork_thread), KVMemoryStore (optional injected
                        HybridRanker upgrades kv.search to hybrid retrieval),
                        and the MemoryHub entry-point façade.

prompt_builder/
  models.py             Pydantic models: InstructionChunk, DocSection,
                        InstructionDoc, InstructionDiagnostic, ConstructKey
                        (frozen), DocRole / ExtractionStrategy.
  pdf_reader.py         PdfReader: reference PDF -> DocSections via a TOC or a
                        font-heuristic strategy, with shared text cleanup,
                        SAS-title -> ConstructKey parsing, and chunker-style
                        diagnostics (never raises).
  doc_chunker.py        InstructionChunker: DocSections -> word-budgeted
                        InstructionChunks (same-parent merge, oversized
                        paragraph-window split, breadcrumb prefix).
  catalog.py            DocumentSpec, default_catalog (the bundled
                        reference_docs set), and CorpusLoader with a per-doc
                        on-disk extraction cache keyed by file SHA-256.
  selector.py           InstructionSelector: construct-key lookup (hazard-
                        first, stop-listed) + HybridRanker topical ranking
                        under a word budget; the dense stage uses
                        memory.relevance.DiskCachedEmbeddings. Imports
                        memory.relevance only.
  builder.py            PromptBuilder façade: read -> chunk -> index at
                        construction, then build(query, constructs) -> a
                        Markdown guidance block or None.
  user_instructions.py  UserInstructionSet: operator-supplied rules (plain
                        str / file) -> scoped InstructionChunks (always /
                        when:constructs / topic). Selected ahead of all
                        reference tiers, rendered as a separate "Project
                        instructions" block, fingerprinted for validation
                        run history. Degrades toward over-inclusion, never
                        raises.

app_config/
  __init__.py           Dependency-free loader for the repo-root config.json
                        (word/token limits). Precedence: explicit constructor
                        argument > config.json > hard default; JSON null means
                        "unset". Searched via SAS_PARSER_CONFIG env var, cwd,
                        then repo root. A leaf package — imports nothing.
  vault.py              HashiCorp Vault credential client (get_secret).
                        Non-secret connection settings from VAULT_* env vars >
                        config.json `vault`; token / AppRole creds from the
                        environment only. hvac imported lazily (extra: vault).
  databricks.py         Databricks workspace settings: host, warehouse/cluster,
                        Unity Catalog catalog+schema (full_table_name()), and
                        the credential. Auth: notebook (on-cluster) > pat
                        (DATABRICKS_TOKEN) > azure-ad (via azure.py). SDK /
                        SQL connector imported lazily (extra: databricks).
  azure.py              Microsoft Entra ID auth: AzureAuthClient.get_token()
                        via client_credentials (secret or certificate) or
                        device_code, with per-scope expiry-aware caching.
                        Client secret from AZURE_CLIENT_SECRET only; msal
                        imported lazily (extra: azure).

validation/
  models.py             Pydantic models: ValidationCase, CaseRun,
                        MetricResult, CaseResult, ValidationReport
                        (score/passed are computed fields; to_markdown()).
  metrics.py            Deterministic metrics + default_metrics():
                        response_coverage, dataset_fidelity, python_syntax,
                        required_terms, reference_similarity. Thresholds
                        resolve via app_config (validation.<name>_threshold).
  judge.py              LLMJudgeMetric — opt-in LLM-as-judge (any
                        LangChain-style model / llm_client.LLMClient);
                        never part of default_metrics().
  runner.py             ValidationRunner: cases -> SasLLMPipeline -> metrics
                        -> ValidationReport; fresh thread_id per case run.
  dataset.py            load_cases(): *.json case files (inline sas_source
                        or a sibling sas_path).
  tracking.py           log_report() / load_runs(): Spark-backed run history,
                        one row per (run, case, metric) — local parquet dir
                        (./validation_runs) by default, saveAsTable (Delta)
                        via `table` on Databricks. Spark boots lazily inside
                        these two functions only.
  __main__.py           CLI: python -m validation <cases_dir> [--judge-model
                        ...] [--track]; exit code gates CI.
  cases/                Sample cases. Like tests/, the package does not ship
                        in the wheel.
```

Import direction is strictly downward: `keywords` and `models` import
nothing from the package; `scanner` and `metadata` import from them;
`chunker.py` imports from all four; `batcher` imports from `keywords`,
`metadata`, `models`; `pipeline` sits on top and is the **only** module that
imports `memory.store`, `memory.relevance`, `memory.summarize`,
`llm_client`, and `prompt_builder`. `memory`, `llm_client`, and `prompt_builder` never import
`chunker` (or each other) — `prompt_builder` reuses `memory.relevance` for
retrieval, and the SAS-metadata → `(query, constructs)` mapping that feeds it
lives in `pipeline`, precisely so `prompt_builder` needs no `chunker` import.
`app_config` is a leaf every package may import (like `chunker.keywords`, it
imports nothing from this repo outside itself): `chunker`, `llm_client`, and
`prompt_builder` read their word/token-limit defaults through it. Its
credential submodules — `vault`, `azure`, `databricks` — import only the
`app_config` loader and, in `databricks`'s case, `azure`; each defers its
third-party client to a lazy import inside the call that needs it, so the
package stays dependency-free to import. `validation` sits *above* the whole
stack, beside the CLI entry points: it drives `chunker.pipeline` and may
import anything, and nothing imports it back.

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
| `macro_var_flow`     | weak   | chunk reads `&name`; links to the nearest preceding creator |
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
Macro-variable consumers follow the same rule: a chunk reading `&name`
links only to the nearest preceding `%let`/`%global`/SYMPUT/SQL-INTO
creator (the last assignment before the reference is the one whose value
SAS resolves), so a name reassigned across unrelated jobs cannot fuse
them, and a reference before any assignment gets no edge.

## Pipeline and memory

`SasLLMPipeline` compiles a one-node LangGraph `StateGraph(MessagesState)`.
The model node loads the thread's history from `KVChatMessageHistory`, runs
`_trim | prompt | LLMClient` (trimming only limits what is *prompted*;
storage keeps every turn), and persists exactly the item message plus the
response in one bulk `add_messages` write. When a `prompt_builder` is set the
prompt additionally carries a block of reference guidance (see
`prompt_builder/`), injected via an ephemeral `instructions` placeholder that
is **prompted but never persisted**. `llm_client.LLMClient` owns
model construction (temperature, output-token cap, endpoint overrides,
proactive rate limiter) and sync + async invocation (input-token budget,
transient-error retry with backoff); an injected `llm` still gets the
retry/budget layers.

Prompted-history trimming has two modes: the default `window_k` recency
window, or — when a `memory.relevance.RelevantHistorySelector` is passed as
`history_selector` — relevance-based selection: each call keeps the
`top_k` turn pairs most relevant to the current batch/chunk message (BM25
lexical retrieval, optional FAISS dense retrieval over embeddings, RRF
fusion, optional reranker), always including the most recent
`always_keep_last` pairs and preserving chronological order. Scorers with
no signal (all scores tied) are excluded from fusion; with no signal at
all, selection degrades to recency. The selector optionally packs its picks
into a `max_tokens` budget (tail exempt, oversized pairs skipped). Either
way, trimming affects only the prompt — storage keeps every turn. All items
of one `run_file`/`run_text`/`run_files` call share one thread
(`thread_id = "run::<source ids>"`), so the LLM sees the run's accumulated
context batch by batch.

Two optional layers complement trimming. A `memory.summarize.RollingSummarizer`
(passed as `summarizer=`) folds turns older than its recency tail into one
running summary per thread, prepended as a SystemMessage *after*
trimming/selection — so it is never dropped by the window and never scored
by the selector; a store-less summarizer is auto-wired to the pipeline's
`memory.kv`. And the pipeline records one small **run fact** per processed
item into the KV layer (`run::<thread>::item::<item_id>`: status, index,
timing — never the response text, which already lives in `msg::`), readable
via `get_run_facts(thread_id)`.

Run facts power two control features. **Resume**: `run_file` / `run_text` /
`run_files` accept `resume=True` — items whose fact reads `ok` on the
thread are skipped (their stored responses recovered from the thread's
turn pairs, `skipped: True` in the output; error facts are reprocessed and
overwritten), so a crashed run continues instead of replaying completed
turns. **Fork**: `fork_run(src, dst, upto_items=k)` copies the first *k*
turn pairs plus their `ok` facts onto an empty thread
(`MemoryHub.fork_thread` underneath, preserving keys/timestamps);
rerunning with `thread_id=dst, resume=True` redoes everything after item
*k* on the branched history — KV-native time travel, no checkpointer.
Storage growth is bounded, when wanted, by
`MemoryHub(retention_max_age_s=..., retention_max_messages=...)`,
applied after each write.

`memory.store` stores everything as namespaced KV rows
(`msg::<thread>::<μs-timestamp>-<rand>` for messages). The
`KVStore` façade owns all JSON (de)serialisation, tag queries, search,
and snapshot/restore; a backend only stores/retrieves/deletes raw rows.
Message reads are incremental: after one full load per
`KVChatMessageHistory` instance, `.messages` fetches only rows past the
last seen key (`records_after` — keys are time-ordered), invalidating on
clear/prune/retention/restore, so an n-item run reads O(n) message rows
instead of O(n²).
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
   persists exactly the last state message plus the response. **Ephemeral
   context is prompted but never persisted to the `msg::` history** — two
   kinds: (a) reference guidance — when a `prompt_builder` is set, per-item
   instruction chunks are injected through an `instructions` placeholder
   carried in the run config; (b) the rolling summary — when a `summarizer`
   is set, its SystemMessage is prepended after trimming/selection and its
   state lives under the KV `summary::` key. Both are re-derivable, would
   bloat the O(n) history, and must stay out of
   `RelevantHistorySelector`'s scoring — *stored = the item message;
   prompted = summary + selected history + guidance + item message*.

6. **In-memory mode must stay Spark-free.** `_InMemoryBackend` (and
   therefore `MemoryHub()` with no arguments) must import and run
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
  `memory.store`, `memory.relevance`, `memory.summarize`,
  `llm_client.client`.
- **Names:** dataset/macro/libref names are lowercased at extraction;
  quoted physical paths keep a leading `'` so they can never collide with
  identifiers.
- **Commits:** conventional-commit style (`refactor(scope): …`), one
  logical change per commit.

## Testing

`tests/` runs without a JVM, network, or API keys: the memory
tests use the in-memory backend, and the pipeline and validation tests inject
`FakeListChatModel`. The two KV backends share one behavioral contract suite
(`tests/test_backend_contract.py`): the in-memory half always runs, and the
Delta half runs the identical tests against a local delta-spark session,
skipping itself where pyspark + delta-spark + a JVM are unavailable — where
it cannot run, `_DeltaBackend` changes still need manual verification
against Databricks. Behavior-preserving
refactors of the chunker/batcher have historically been verified by
snapshotting full batcher output (batch membership, I/O fields, reason
strings, ordering) on a synthetic multi-file corpus and diffing against the
pre-change code; prefer that over trusting the suite alone for pure
code-motion changes.

The unit suite asserts *code behavior*; the *output quality* of a real model
run is the `validation` package's job — declarative cases scored by
deterministic metrics (coverage, dataset fidelity, Python syntax, required
terms, reference similarity) plus an opt-in LLM judge, with per-run history
appended via Spark (`python -m validation validation/cases --track`): local
parquet by default, a Delta table on Databricks. Like the Delta memory
backend, the Spark write path needs a JVM, so its test skips itself where no
local Spark session can start.
