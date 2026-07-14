# memory

Chat-history and KV persistence layer for the pipeline, plus relevance-based
history selection and rolling summarization. No module here imports
`chunker`, and the three feature modules never import each other — in
LangChain context-engineering terms they cover **write** (`store`),
**select** (`relevance`), and **compress** (`summarize`) independently.

- `store.py` — durable chat history and a tagged KV store, backed by a
  plain Python dict locally or a Databricks Delta table in production.
- `relevance.py` — `RelevantHistorySelector`, an alternative to recency-window
  trimming that prompts the history turns most *relevant* to the current
  request (optionally packed into a `max_tokens` budget), plus the shared
  `HybridRanker` retrieval stack and `DiskCachedEmbeddings`.
- `summarize.py` — `RollingSummarizer`, one running summary per thread:
  turns older than a recency tail are folded (monotonically, oldest first)
  into a KV-stored summary that is prompted but never persisted to the
  message history.
- `turns.py` — dependency-light turn grouping and token-estimate helpers
  shared by `relevance` and `summarize` (so `summarize` never drags in
  bm25s/faiss).

`memory` is a regular package (not a PEP-420 namespace package) so packaging
tools and import machinery treat it uniformly.

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
├── KVMemoryStore            ← tagged KV store with search + text ingestion
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
`chunker.pipeline` for the wiring). The legacy `BaseMemory` / `ConversationChain`
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
