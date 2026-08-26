# V2 knowledge and memory — Phase 6 migration note

Phase 6 replaces the legacy prompt-builder and conversation-memory service
APIs with attributed application contracts. Breaking import and persistence
changes are intentional; the v2 schemas write fresh records and do not migrate
legacy Delta keys in place.

Knowledge ingestion now returns typed extraction diagnostics, sections,
sources, chunks, and fingerprints. PDF loading is a lazy PyMuPDF adapter, so
application services remain independent of document libraries. Retrieval and
scoped user rules return attributed prompt components and keep reference
guidance separate from project instructions in token reports.

Phase 9 completes the advanced retrieval boundary. `KnowledgeRanker`,
`EmbeddingProvider`, `EmbeddingCache`, and `KnowledgeReranker` ports keep the
application layer free of numerical and model SDKs. The hybrid adapter imports
BM25 and FAISS only when ranking executes, combines lexical and optional dense
rankings with reciprocal-rank fusion, and can rerank a bounded fused window.
Embedding caches are keyed by both provider namespace and content digest;
the disk adapter writes atomically and treats unreadable cache files as misses.
The original deterministic lexical selector remains the default when no
advanced ranker is injected.

Conversation memory now has explicit services for accepted history, relevant
turn selection, rolling summaries, task policy, thread notes, context assembly,
and memory extraction. Temporary extracted memories become TTL notes. Permanent
memories remain pending policy proposals until explicitly approved. Classifier
failures do not fail an accepted translation.

The in-memory adapter imports and runs without Spark. The Delta adapter writes
the same contracts through the existing CDF-enabled Delta KV engine while that
engine is migrated out of the legacy namespace. It persists audit events,
supports snapshots, restore, rewind, fork, retention, accepted-response state,
and exposes CDF synchronization. A fresh v2 prefix prevents collision with
legacy records.

Delta OPTIMIZE and VACUUM are intentionally separate from request handling.
Table identifiers are validated and quoted, retention remains between one week
and four months, and retention must exceed the maximum expected CDF outage.
The dedicated CI job builds the repository-owned Spark/Delta image, exercises
direct PySpark DataFrame and Python ``DeltaTable`` mutations plus a real
catalog-backed repository, and treats a missing or incompatible runtime as a
failure rather than a skip. Its locked PySpark and Delta versions are also the
versions used by Compose and the standalone Spark image.
