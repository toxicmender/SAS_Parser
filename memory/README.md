# memory

Chat-history and KV persistence layer for the pipeline, plus relevance-based
history selection. The two modules are independent and never import `chunker`
or each other.

- `short_mem.py` — durable chat history and a tagged KV store, backed by a
  plain Python dict locally or a Databricks Delta table in production.
- `relevance.py` — `RelevantHistorySelector`, an alternative to recency-window
  trimming that prompts the history turns most *relevant* to the current
  request.

`memory` is a regular package (not a PEP-420 namespace package) so packaging
tools and import machinery treat it uniformly.

---

## short_mem — chat history + KV persistence

Persists to a Databricks Delta table in production; runs on a plain Python dict
locally, with no Spark (or JVM) required at all in that mode.

```python
from memory.short_mem import DatabricksMemory

# local / CI — in-memory dict, pyspark not required
mem = DatabricksMemory()

# Databricks — Delta-backed
mem = DatabricksMemory(
    spark=spark,                          # existing Databricks SparkSession
    table="catalog.schema.langchain_mem", # Delta table (created if absent)
)

thread = mem.get_thread("user-42")
thread.add_user_message("Hello!")
thread.add_ai_message("Hi! How can I help?")

mem.kv.set("project_goal", "RAG pipeline", tags=["project"])
mem.kv.search("pipeline")
```

### Architecture

```
SparkKVStore                 ← façade over one of two interchangeable backends
│   ├── _InMemoryBackend     ← plain dict  (local / CI; pyspark not required)
│   └── _DeltaBackend        ← Spark DataFrame / Databricks Delta table
│
├── KVChatMessageHistory     ← BaseChatMessageHistory for one thread/session
├── ThreadMemoryManager      ← manages many independent threads
├── KVMemoryStore            ← tagged KV store with search + text ingestion
└── DatabricksMemory         ← unified façade (recommended entry point)
```

The `SparkKVStore` façade owns all JSON (de)serialisation, tag queries, search,
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
several entries into one MERGE) and `DELETE FROM` for deletes.

### Invariant — in-memory mode stays Spark-free

`_InMemoryBackend` (and therefore `DatabricksMemory()` with no arguments) must
import and run without pyspark installed; the pyspark requirement lives inside
`_DeltaBackend.__init__` only. `_DeltaBackend` has no automated coverage in the
test suite — it requires a live Spark session — so changes to it need manual
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
| DEBUG | Per-select selection summary; embedding-cache misses; no-signal recency fallback |
| INFO | Selector construction |
| WARNING | Query produced no tokens (BM25 stage skipped) |
