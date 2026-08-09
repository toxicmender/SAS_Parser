# memory

Chat-history and KV persistence layer for the pipeline, plus relevance-based
history selection, rolling summarization, and the long-/short-term
instruction memories. No module here imports `chunker`, and the feature
modules never import each other — in LangChain context-engineering terms
they cover **write** (`store`), **select** (`relevance`), and **compress**
(`summarize`) independently, with `policy`/`thread_mem`/`extractor` adding
the **remember** channel on top of the same duck-typed KV interface.

- `store.py` — durable chat history and a tagged KV store, backed by a
  plain Python dict locally or a Databricks Delta table in production.
  Owns the Thread → Chat → Message hierarchy.
- `relevance.py` — `RelevantHistorySelector`, an alternative to recency-window
  trimming that prompts the history turns most *relevant* to the current
  request (optionally packed into a `max_tokens` budget), plus the shared
  `HybridRanker` retrieval stack and `DiskCachedEmbeddings`.
- `summarize.py` — `RollingSummarizer`, one running summary per thread:
  turns older than a recency tail are folded (monotonically, oldest first)
  into a KV-stored summary that is prompted but never persisted to the
  message history.
- `policy.py` — **long-term memory**: `TaskPolicy`, the standing
  instructions for a *task*, stable across every thread and session it
  spawns, folded into the (cacheable) system prompt.
- `thread_mem.py` — **short-term memory**: `ThreadMemory`, the notes,
  exceptions and overrides that hold for *one conversation only*, prompted
  ephemerally and expiring on a TTL.
- `context.py` — `MemoryContext.assemble()`, the seam that resolves both
  instruction channels for one turn and states their precedence.
- `extractor.py` — `MemoryExtractor`, the write path: classifies a completed
  turn into permanent (a policy proposal, held for approval) or temporary (a
  thread note, applied) memory, behind an offline cue gate.
- `turns.py` — dependency-light turn grouping and token-estimate helpers
  shared by `relevance` and `summarize` (so `summarize` never drags in
  bm25s/faiss).

`memory` is a regular package (not a PEP-420 namespace package) so packaging
tools and import machinery treat it uniformly.

## The memory model at a glance

```
Task ────────── TaskPolicy          long-term, task-scoped   (policy.py)
 └── Thread ─── ThreadMemory        short-term, thread-scoped (thread_mem.py)
      │         RollingSummarizer   compressed history        (summarize.py)
      ├── Chat  one consumer's span of the thread             (store.py)
      │    ├── Message
      │    └── Message
      └── Chat
```

One turn's prompt, in order:

```
system  = base prompt + TaskPolicy.render()        ← cacheable, stable
history = rolling summary + selected/trimmed turns ← memory.relevance / window_k
ephemeral = reference guidance + ThreadMemory notes ← prompted, never stored
human   = the item
```

Where each instruction channel lands is a caching decision. The policy is
identical for every thread and item, so it sits **inside** the
`cache_control` breakpoint on the system block and is served from the
provider cache after the run's first item. Thread notes differ per thread —
folding them into that prefix would miss the cache on every thread — so they
ride the ephemeral `instructions` placeholder *after* the breakpoint, which
is also what keeps them out of the stored history and out of
`RelevantHistorySelector`'s scoring.

---

## store — chat history + KV persistence

Persists to a Databricks Delta table in production; runs on a plain Python dict
locally, with no Spark (or JVM) required at all in that mode.

```python
from memory.store import MemoryHub

# local / CI — in-memory dict, pyspark not required
mem = MemoryHub()

# Databricks — Delta-backed
mem = MemoryHub(
    spark=spark,                          # existing Databricks SparkSession
    table="catalog.schema.langchain_mem", # Delta table (created if absent)
)

thread = mem.get_thread("user-42")
thread.add_user_message("Hello!")
thread.add_ai_message("Hi! How can I help?")

mem.kv.set("project_goal", "RAG pipeline", tags=["project"])
mem.kv.search("pipeline")

# Optional: hybrid search over the KV store (BM25 + optional dense + RRF)
from memory.relevance import HybridRanker
mem = MemoryHub(ranker=HybridRanker())
```

`MemoryHub(ranker=...)` / `KVMemoryStore(ranker=...)` upgrade
`kv.search` from the naive substring scan to the same
BM25 + optional dense + RRF stack the history selector uses (scores are the
1/rank of the fused order; no-signal queries return `[]`). The ranker is
duck-typed — `store` never imports `relevance`, so plain KV usage stays
free of the bm25s/faiss dependencies.

### Chats — the span between a thread and its messages

A **chat** is one consumer's stretch of a thread: `SasLLMPipeline` opens one
per pipeline instance, so a resumed or forked run appears as a second chat on
the same continuous thread.

```python
thread = mem.get_thread("user-42", chat_id="pipe-3f9c", chat_label="nightly run")
thread.add_user_message("...")

mem.chats("user-42")              # [{chat_id, label, first_key, last_key, ...}]
thread.messages_in_chat("pipe-3f9c")
```

The chat id is deliberately **not** part of the message key. Thread ids are
recovered from `msg::{thread}::{tick}` keys by splitting off the last segment
(`list_threads`), `fork_thread` copies a key-prefix range, and the
incremental read frontier is a key comparison — a third segment would break
all three at once, and every legacy row with them. Instead one index record
per chat (`chat::{thread}::{chat_id}`) is written **in the same batch** as
the messages it indexes, so the index can never describe a write that did not
land. `get_thread(..., chat_id=...)` re-stamps the active chat on every call
rather than fixing it at creation, because thread objects are cached per id
and two consumers sharing a manager would otherwise inherit each other's
chat. Forking clamps chats to the copied slice: one that starts after the
fork point is dropped, one that straddles it keeps only the copied part.

### Thread forking

`mem.fork_thread(src, dst, upto_messages=..., upto_ts=...)` copies a
thread's oldest messages onto an empty destination thread — rows keep their
key suffixes, timestamps, and payloads (one batched write); only the
session-id tag is rewritten, and the source is untouched. This is the
KV-native half of "time travel": rewind a conversation to a point and
continue it under a new id. The pipeline builds `fork_run` (fork at an
*item* boundary + copy run facts, enabling `resume=True` on the branch) on
top of it.

### Retention

`MemoryHub(retention_max_age_s=..., retention_max_messages=...)`
bounds the *stored* thread, applied opportunistically after every write:
messages older than the age limit are pruned, then the oldest beyond the
count limit. Both default to off (keep everything). This automates the
manual `prune_before` / `prune_to_count` calls; prompt-side trimming stays
a separate concern.

### Architecture

```
KVStore                      ← façade over one of two interchangeable backends
│   ├── _InMemoryBackend     ← plain dict  (local / CI; pyspark not required)
│   └── _DeltaBackend        ← Spark DataFrame / Databricks Delta table
│
├── KVChatMessageHistory     ← BaseChatMessageHistory for one thread/session
├── ThreadMemoryManager      ← manages many independent threads
├── KVMemoryStore            ← tagged KV store with search
└── MemoryHub                ← unified façade (recommended entry point)
```

The `KVStore` façade owns all JSON (de)serialisation, tag queries, search,
and snapshot/restore. A backend only stores, retrieves, and deletes raw rows,
which both speak in the same Delta-schema column order:

```
(key, value_json, tags_json | None, created_at, updated_at, source)
```

### LangChain integration

`KVChatMessageHistory` implements `langchain_core.chat_history.BaseChatMessageHistory`
(overriding bulk `add_messages`, as the base class recommends) — a current,
supported API in LangChain v1. It is the durable backing store behind a
LangGraph `StateGraph`: the graph's model node loads `history.messages` before
each LLM call and persists the new turn with `add_messages` (see
`pipeline.engine` for the wiring). The legacy `BaseMemory` / `ConversationChain`
layer was removed from LangChain in v1, and `RunnableWithMessageHistory` is
deprecated in favour of LangGraph persistence, so this module ships neither
adapter.

### Storage schema (one row per KV entry)

| Column | Type | Meaning |
|--------|------|---------|
| `kv_key` | STRING NOT NULL | namespaced key, e.g. `msg::thread-1::0001783440000000-9f3a` |
| `value` | STRING NOT NULL | JSON-serialised payload |
| `tags` | STRING | JSON array of tag strings |
| `created_at` | DOUBLE | Unix timestamp (float) |
| `updated_at` | DOUBLE | Unix timestamp (float) |
| `source` | STRING | optional provenance label |

Keys follow a `namespace::subkey` convention so multiple logical stores share
one physical table without collision (`msg::…` chat messages, `kv::…`
`KVMemoryStore` entries, `idx::…` legacy sequence counters). Message keys embed
a zero-padded microsecond timestamp plus a short random suffix, so they are
collision-free without any read-modify-write sequence counter and sort
lexicographically in time order (legacy `{seq:08d}` keys sort before them). On
Databricks the Delta backend uses `MERGE INTO` for upserts (`set_many` batches
several entries into one MERGE) and `DELETE FROM` for deletes; every value the
SQL sees goes through Spark parameter markers (`spark.sql(sql, args)`, Spark ≥
3.4), never string interpolation. `restore()` is a single `INSERT OVERWRITE`
commit, so a crash mid-restore cannot leave the table emptied.

Message values carry the full LangChain `message_to_dict` payload
(`{"message": …, "ts": …}`), so tool calls, `usage_metadata`,
`response_metadata`, names, and ids round-trip losslessly;
`get_session_metadata()` sums the persisted `usage_metadata` into per-thread
`total_usage` token counts. Rows written by the pre-lossless schema
(`{"role", "content", "meta", "ts"}`) are still readable.

### Incremental reads

Message keys are time-ordered, so `KVChatMessageHistory.messages` performs
a full prefix scan only once per instance; every later call fetches just
the rows whose key sorts after the last one seen
(`KVStore.records_after`) and appends them to an in-instance cache — an
n-item pipeline run reads O(n) rows instead of O(n²). The cache is
invalidated by anything that deletes messages through the instance
(`clear`, `prune_*`, retention) and by `MemoryHub.restore()`; appends from
*other* writers are still picked up (their keys sort after the cache
frontier), but out-of-band deletes or backdated keys are not seen until
the next invalidation.

### Delta change feed, audit, and cross-process caches

For a shared Delta deployment, configure both a per-process consumer id and
an **independent** audit table. `MemoryHub.sync_changes()` establishes a
snapshot boundary on its first call (invalidating all local thread caches),
then consumes later committed CDF versions and invalidates only cached threads
whose `msg::` or `chat::` rows changed. The consumer checkpoint lives in the
audit table, never in the source memory table, so processing it cannot create
an endless CDF feedback loop.

```python
mem = MemoryHub(
    spark=spark,
    table="catalog.schema.langchain_mem",
    cdf_consumer_id="pipeline_worker_1",
    cdf_audit_table="catalog.schema.langchain_mem_audit",
)
mem.sync_changes()  # once at each long-lived worker's request boundary
```

The two table names may instead be declared once under config.json
`memory.delta_table` and `memory.cdf_audit_table`. They are validated as
fully-qualified `Catalog.Schema.Table` identifiers; explicit `MemorySetup`
values take precedence. `cdf_consumer_id` remains an explicit per-worker
setting so two workers never share a checkpoint.

CDF begins only after the feature is enabled and its files follow the source
table's VACUUM retention. The first call is therefore a baseline rather than
a false historical audit; later CDF events are upserted into the separately
retained audit Delta table with the durable checkpoint. Keep source retention
longer than the longest expected CDF consumer outage.

### Delta maintenance

Maintenance is explicit and never runs in the request path. Use
`DeltaMemoryMaintenance` from a scheduled job to inspect history, compact, or
vacuum a memory table. `VacuumPolicy` rejects retention below seven days or
above four 30-day months, and requires it to exceed the configured worst-case
CDF consumer outage. `vacuum()` is a dry run by default.

```python
from memory import DeltaMemoryMaintenance, VacuumPolicy

ops = DeltaMemoryMaintenance(
    spark, "catalog.schema.langchain_mem",
    policy=VacuumPolicy(retention_hours=30 * 24, max_cdf_outage_hours=7 * 24),
)
ops.optimize()
ops.vacuum()  # dry run
```

### Optional Databricks AI Bridge

Install `sas-parser[databricks-ai]` only where Databricks Model Serving is
needed. The optional extra installs `databricks-langchain`; importing
`memory` stays independent of it. `memory.chat_model()` returns a
`ChatDatabricks` instance that can be passed as `SasLLMPipeline(llm=...)` or
as the model for `MemoryExtractor` / `RollingSummarizer`; `memory.embeddings()`
returns `DatabricksEmbeddings` for `HybridRanker(embeddings=...)`.

### Invariant — in-memory mode stays Spark-free

`_InMemoryBackend` (and therefore `MemoryHub()` with no arguments) must
import and run without pyspark installed; the pyspark requirement lives inside
`_DeltaBackend.__init__` only. Both backends are held to one behavioral
contract by `tests/test_backend_contract.py`: the in-memory half always runs,
and the Delta half runs the identical tests against a local delta-spark
session, skipping itself where pyspark + delta-spark + a JVM are unavailable.
Where the Delta tests cannot run, changes to `_DeltaBackend` still need manual
verification against Databricks.

---

## relevance — relevance-based history selection

An alternative to recency-only window trimming: instead of "keep the last *k*
turn pairs", keep the pairs most *relevant* to the current request. In the SAS
pipeline a later batch often depends on one specific earlier batch — the one
that defined the macro or wrote the dataset it consumes — which a recency
window may have already dropped.

Wire it into the pipeline via `SasLLMPipeline(history_selector=...)`.

### HybridRanker — the shared retrieval core

The BM25 + dense + RRF + reranker stack lives in `HybridRanker`, independent of
chat history. It has two modes:

- **Stateless per-call** (`bm25_ranking` / `dense_ranking` / `rrf_fuse` /
  `rerank`): ranks an arbitrary doc list afresh each call, for a corpus that
  changes every time — which is exactly a chat thread, so
  `RelevantHistorySelector` uses this mode.
- **Static corpus** (`index` once, then `query` many): builds one BM25 index
  and one FAISS index and reuses them across queries, for a fixed corpus where
  a per-query rebuild would dominate runtime. `query` returns an empty list on
  no signal — a fixed corpus has no recency to fall back to.

`RelevantHistorySelector` is a thin policy layer over the per-call mode.

### Two-stage retrieve-then-rerank

Over the thread's `(human, AI)` turn pairs:

1. **Retrieval.** Each candidate pair is ranked against the current request by
   two independent scorers:
   - **BM25** (`bm25s`): lexical match over a lowercased identifier
     tokenisation. Strong here because SAS prompts are full of exact
     identifiers (dataset names, macro names, librefs).
   - **Dense** (optional): cosine similarity between embeddings, searched with
     a FAISS `IndexFlatIP` over L2-normalised vectors. Enabled by passing an
     `embeddings` model (a LangChain `Embeddings` instance, or a provider string
     resolved via `langchain.embeddings.init_embeddings`).
2. **Rerank.** The per-scorer rankings are fused with Reciprocal Rank Fusion
   (`score(d) = Σ 1 / (rrf_k + rank(d))`). RRF is rank-based, so BM25's
   unbounded scores and cosine's `[-1, 1]` fuse without calibration. A scorer
   whose scores are all identical carries no signal and is dropped from fusion;
   if no scorer has signal, selection falls back to recency. An optional
   `reranker` callable (a cross-encoder or LLM judge) then re-orders the fused
   shortlist.

The most recent `always_keep_last` pairs are always kept regardless of score,
and the selected pairs are returned in their original chronological order —
relevance decides *which* pairs are prompted, never their order.

### Token budget (`max_tokens`)

`top_k` counts pairs regardless of size; passing `max_tokens` additionally
packs the ranked pairs into a token envelope: pairs are taken best-first
while they fit, an oversized pair is *skipped* (not a stopping point) so
smaller relevant pairs behind it can still use the budget, and the
always-kept tail is exempt — it is included even when it alone exceeds the
budget. With a budget set, selection also runs on histories short enough
that `top_k` alone would pass through whole. Token counting defaults to the
offline ~4-chars/token estimate (`memory.turns.approx_token_count`); pass
`token_counter` for a real tokenizer.

### Notes

- **FAISS index choice:** `IndexFlatIP` is exact brute-force search, identical
  to a numpy dot product. Approximate indexes (IVF/HNSW) only pay off at ~10^5+
  vectors and one thread's history is at most a few hundred pairs, so the flat
  index is right here; the seam to swap in an approximate one is `_dense_ranking`.
- **Embedding cache:** embeddings are cached per pair text (keyed by content
  hash) for the selector's lifetime, so each turn pair is embedded once per run
  even though `select` runs before every LLM call.

### Logging

Logger name: `memory.relevance`

| Level | When emitted |
|-------|--------------|
| DEBUG | Per-select selection summary; embedding-cache misses; no-signal recency fallback; over-budget skips |
| INFO | Selector construction |
| WARNING | Query produced no tokens (BM25 stage skipped) |

---

## summarize — rolling thread summarization

The **compress** channel: selection decides which turns are prompted
verbatim; `RollingSummarizer` guarantees a floor of information about
everything else. Once the turns older than a `keep_last_turns` recency tail
jointly reach `trigger_tokens`, they are folded — monotonically, oldest
first, one LLM call per fold — into a single running summary per thread,
stored under `summary::{thread_id}` in a KV store and returned as one
`SystemMessage` to prepend to the prompt.

```python
from memory.summarize import RollingSummarizer

summarizer = RollingSummarizer(model)          # any .invoke model or str->str callable
pipeline = SasLLMPipeline(summarizer=summarizer)  # store auto-wired to mem.kv
```

Design points:

- **Coverage is positional, not selection-based.** What the relevance
  selector drops varies per query; summarizing "dropped" turns would
  re-summarize the same content endlessly. A monotonic prefix is summarized
  exactly once, and the selector remains free to surface any covered turn
  verbatim when it becomes relevant again.
- **Prompted, never persisted.** The summary lives in the KV layer, not the
  `msg::` history — it is re-derivable from the full stored thread, and it
  must not pollute relevance scoring (the pipeline prepends it *after*
  trimming/selection).
- **Self-healing.** If a thread shrinks below the covered turn count
  (cleared or forked), the stale summary is discarded and rebuilt.
- The `store` is duck-typed (`get`/`set`/`delete`) — `KVMemoryStore` fits,
  and `SasLLMPipeline` injects its own `memory.kv` into a store-less
  summarizer. Without any store, state is process-local.

Logger name: `memory.summarize` (INFO on construction and each fold,
WARNING on a shrunken-thread reset).

---

## policy — long-term (task) memory

Standing instructions for a *task*, stable across every thread and session it
spawns. Stored as one record per task, `policy::{task_id}`.

```python
from memory.policy import TaskPolicy
from pipeline import MemorySetup

policy = TaskPolicy("customer_support", store=mem.kv)
policy.add("Prefer concise responses.")
policy.add("Escalate refund requests above $500.", overridable=False)

# Memory wiring goes in as one MemorySetup.
pipeline = SasLLMPipeline(
    memory_setup=MemorySetup(task_id="customer_support")
)   # loads and prompts it
```

- **Overridable vs fixed.** Each instruction declares whether a
  conversation-specific note may override it. Preferences are overridable;
  guardrails are not, and the rendered block says so inline — which is what
  makes short-term overrides safe to enable at all.
- **Snapshot, not live read.** The pipeline folds `render()` into its system
  prompt at construction and records `fingerprint` for that text. Editing the
  policy afterwards does not change the running pipeline (that is what keeps
  the cached prefix stable, and what makes the fingerprint an honest record
  of what was prompted) — the next pipeline, i.e. the next chat, picks it up.
- **Defaults seed, they do not overwrite.** `default_instructions=` applies
  only to a task with no stored policy, so re-running a program cannot
  silently revert an operator's edit.

## thread_mem — short-term (conversation) memory

The exceptions, preferences and overrides that hold for one thread only.
Stored as one record per thread, `tmem::{thread_id}`.

```python
from memory.thread_mem import ThreadMemory
from pipeline import MemorySetup

notes = ThreadMemory(store=mem.kv)
notes.add("thread-1", "Customer prefers email.", kind="preference")
notes.add("thread-1", "Discount approved.", kind="exception",
          source="supervisor", ttl_s=3600)

pipeline = SasLLMPipeline(memory_setup=MemorySetup(thread_memory=notes))
pipeline.remember("thread-1", "Do not offer the standard discount.")
```

Notes are **prompted, never persisted** to the `msg::` history — the same
rule reference guidance and the rolling summary follow — and they expire:
`ttl_s` bounds an exception's life so a stale override cannot outlive the
approval that justified it. `fork()` copies live notes onto a branch, which
`SasLLMPipeline.fork_run` calls, because losing "exception approved by
supervisor" on a rewind would silently change what the branch may do.

## context — assembling the two channels

```python
from memory.context import MemoryContext

ctx = MemoryContext(policy=policy, thread_memory=notes)
assembled = ctx.assemble("thread-1")
assembled.system_suffix   # -> append to the system prompt (cacheable)
assembled.ephemeral       # -> prompt after the cache breakpoint
```

The only place that sees both channels, and therefore the only place that can
state their precedence: a note overrides an overridable instruction, and
never a fixed one. That sentence is rendered with the notes, not with the
policy, so it costs nothing on threads that have no notes.

## extractor — the write path

```python
from memory.extractor import MemoryExtractor
from pipeline import MemorySetup

extractor = MemoryExtractor(model, policy=policy, thread_memory=notes)
pipeline = SasLLMPipeline(
    memory_setup=MemorySetup(memory_extractor=extractor)
)   # observes kept turns

extractor.pending()          # permanent candidates awaiting approval
extractor.approve(candidate_id, overridable=False)
```

Two asymmetries carry the safety story:

- **The cue gate runs before the model.** A turn with no instruction-shaped
  language ("from now on", "never", "exception", "approved", …) is skipped
  offline, so an ordinary translation run pays no extra LLM calls.
- **Permanent memories are proposed, not written.** A thread note expires and
  dies with its thread; a policy instruction applies to every future thread
  of the task, so it needs `approve()` (or `auto_apply_permanent=True`, off
  by default, which logs a warning).

Extraction runs on the turn that was *kept* — never on an attempt inline
validation rolled back — and swallows every error: a memory write must not be
able to fail a translation run.

Logger names: `memory.policy`, `memory.thread_mem`, `memory.context`,
`memory.extractor`.
