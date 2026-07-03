"""
(Databricks / PySpark backend)
=====================================================
LangChain Memory Module backed by a Spark DataFrame that maps 1-to-1
onto a Databricks Delta table in production.

Architecture
------------
SparkKVStore                 ← core backend  (Spark DataFrame / Delta table)
│
├── KVChatMessageHistory     ← BaseChatMessageHistory for one thread/session
├── ThreadMemoryManager      ← manages many independent threads
├── WindowChatMemory         ← rolling-window BaseMemory for LangChain chains
├── KVMemoryStore            ← tagged KV store with search + text ingestion
└── DatabricksMemory         ← unified façade (recommended entry point)

Storage schema (one row per KV entry)
--------------------------------------
kv_key      STRING NOT NULL   — namespaced key  e.g. "msg::thread-1::00000003"
value       STRING NOT NULL   — JSON-serialised payload
tags        STRING            — JSON array of tag strings
created_at  DOUBLE            — Unix timestamp (float)
updated_at  DOUBLE            — Unix timestamp (float)
source      STRING            — optional provenance label

On Databricks the SparkKVStore writes to a Delta table and uses
MERGE INTO for upserts.  In local / CI mode it keeps state in a
plain in-memory DataFrame (no disk I/O).

Usage — Databricks notebook
----------------------------
    from persistent_memory import DatabricksMemory

    mem = DatabricksMemory(
        spark=spark,                          # existing Databricks SparkSession
        table="catalog.schema.langchain_mem", # Delta table (created if absent)
    )

    thread = mem.get_thread("user-42")
    thread.add_user_message("Hello!")
    thread.add_ai_message("Hi! How can I help?")

    chat = mem.chat_memory(session_id="session-1", k=8)
    # plug straight into ConversationChain(llm=llm, memory=chat)

    mem.kv.set("project_goal", "RAG pipeline", tags=["project"])
    mem.kv.search("pipeline")

Usage — local / CI (no Databricks)
-----------------------------------
    from persistent_memory import DatabricksMemory
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local").appName("mem").getOrCreate()
    mem   = DatabricksMemory(spark=spark)   # table=None → pure in-memory DF
"""

from __future__ import annotations

import datetime
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Guards: PySpark is optional at import time so the module can be loaded
# even when building docs or running unit tests without a Spark cluster.
# ---------------------------------------------------------------------------
try:
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        DoubleType,
        StringType,
        StructField,
        StructType,
    )

    _PYSPARK_AVAILABLE = True
except ImportError:
    _PYSPARK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).isoformat(sep=" ", timespec="seconds")


def _jdump(v: Any) -> str:
    return json.dumps(v, default=str)


def _jload(s: str) -> Any:
    return json.loads(s)


# Schema shared by every layer
_KV_SCHEMA = None  # built lazily after PySpark import check


def _schema():
    global _KV_SCHEMA
    if _KV_SCHEMA is None:
        _KV_SCHEMA = StructType(
            [
                StructField("kv_key", StringType(), nullable=False),
                StructField("value", StringType(), nullable=False),
                StructField("tags", StringType(), nullable=True),
                StructField("created_at", DoubleType(), nullable=True),
                StructField("updated_at", DoubleType(), nullable=True),
                StructField("source", StringType(), nullable=True),
            ]
        )
    return _KV_SCHEMA


# ---------------------------------------------------------------------------
# BaseMemory shim  (removed from langchain_core public API in v1)
# ---------------------------------------------------------------------------


class BaseMemory(BaseModel):
    """Minimal abstract base for custom memory objects."""

    @property
    def memory_variables(self) -> List[str]:
        raise NotImplementedError

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


# ===========================================================================
# CORE: SparkKVStore
# ===========================================================================


class SparkKVStore:
    """
    Key-value store backed by a Spark DataFrame / Databricks Delta table.

    Two modes
    ---------
    - **Delta mode** (production):
      Pass ``table="catalog.schema.table_name"``.
      The table is created with Delta format if it does not already exist.
      Upserts use ``MERGE INTO``, deletes use ``DELETE FROM``.

    - **In-memory mode** (local / CI):
      Pass ``table=None`` (default).
      All state lives in a Python dict that is wrapped in a DataFrame only
      for query operations.  No disk I/O — identical to the previous backend
      but going through the same Spark SQL interface.

    All values are JSON-serialised so any Python type (dict, list, str, …)
    can be stored.  Tags are stored as a JSON array string.

    Key layout
    ----------
    Keys follow a namespace::subkey convention so multiple logical stores
    can share one physical table without collision:

        msg::thread-42::00000003   → chat message seq 3 of thread-42
        kv::project_goal           → a KVMemoryStore entry
        idx::thread-42             → next sequence counter for a thread

    Parameters
    ----------
    spark : SparkSession
        Active Spark / Databricks session.
    table : str | None
        Fully-qualified Delta table name for persistence, or None for
        pure in-memory operation.
    """

    def __init__(
        self,
        spark: "SparkSession",
        table: Optional[str] = None,
    ) -> None:
        if not _PYSPARK_AVAILABLE:
            raise RuntimeError("pyspark is not installed. Run: pip install pyspark")
        self._spark = spark
        self._table = table
        self._mem: Dict[str, Dict] = {}  # used only when table is None

        if table:
            self._ensure_table()

    # ---- Table bootstrap (Delta only) --------------------------------------

    def _ensure_table(self) -> None:
        """Create the Delta table if it does not already exist."""
        self._spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                kv_key     STRING  NOT NULL,
                value      STRING  NOT NULL,
                tags       STRING,
                created_at DOUBLE,
                updated_at DOUBLE,
                source     STRING
            )
            USING DELTA
            TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
        """)

    # ---- Internal read helpers ---------------------------------------------

    def _df(self, prefix: str = "") -> "DataFrame":
        """Return a DataFrame view of current store contents."""
        if self._table:
            df = self._spark.table(self._table)
        else:
            rows = [
                (
                    k,
                    r["value"],
                    r["tags"],
                    r["created_at"],
                    r["updated_at"],
                    r["source"],
                )
                for k, r in self._mem.items()
            ]
            df = self._spark.createDataFrame(rows or [], schema=_schema())

        if prefix:
            df = df.filter(F.col("kv_key").startswith(prefix))
        return df

    def _row_to_dict(self, row) -> Dict:
        return {
            "value": _jload(row.value),
            "tags": _jload(row.tags) if row.tags else [],
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "source": row.source,
        }

    # ---- CRUD --------------------------------------------------------------

    def set(
        self,
        key: str,
        value: Any,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
    ) -> None:
        now = _now()
        val_json = _jdump(value)
        tags_json = _jdump(tags or [])

        if self._table:
            # MERGE INTO for upsert
            # Stage the new row as a single-row temp view
            new_row = self._spark.createDataFrame(
                [(key, val_json, tags_json, now, now, source)],
                schema=_schema(),
            )
            new_row.createOrReplaceTempView("_kv_upsert_staging")
            self._spark.sql(f"""
                MERGE INTO {self._table} AS target
                USING _kv_upsert_staging AS source
                  ON target.kv_key = source.kv_key
                WHEN MATCHED THEN UPDATE SET
                    target.value      = source.value,
                    target.tags       = COALESCE(source.tags, target.tags),
                    target.updated_at = source.updated_at,
                    target.source     = COALESCE(source.source, target.source)
                WHEN NOT MATCHED THEN INSERT *
            """)
        else:
            existing = self._mem.get(key)
            self._mem[key] = {
                "value": val_json,
                "tags": tags_json
                if tags is not None
                else (existing["tags"] if existing else _jdump([])),
                "created_at": existing["created_at"] if existing else now,
                "updated_at": now,
                "source": source or (existing["source"] if existing else None),
            }

    def get(self, key: str, default: Any = None) -> Any:
        if self._table:
            rows = (
                self._spark.table(self._table)
                .filter(F.col("kv_key") == key)
                .select("value")
                .limit(1)
                .collect()
            )
            return _jload(rows[0].value) if rows else default
        rec = self._mem.get(key)
        return _jload(rec["value"]) if rec else default

    def delete(self, key: str) -> bool:
        if self._table:
            before = (
                self._spark.table(self._table).filter(F.col("kv_key") == key).count()
            )
            if before == 0:
                return False
            self._spark.sql(f"DELETE FROM {self._table} WHERE kv_key = '{key}'")
            return True
        if key in self._mem:
            del self._mem[key]
            return True
        return False

    def keys(self, prefix: str = "") -> List[str]:
        return [row.kv_key for row in self._df(prefix).select("kv_key").collect()]

    def all_records(self, prefix: str = "") -> List[Tuple[str, Dict]]:
        rows = self._df(prefix).collect()
        return [(row.kv_key, self._row_to_dict(row)) for row in rows]

    def clear_prefix(self, prefix: str) -> int:
        if self._table:
            count = self._df(prefix).count()
            self._spark.sql(f"DELETE FROM {self._table} WHERE kv_key LIKE '{prefix}%'")
            return count
        targets = [k for k in self._mem if k.startswith(prefix)]
        for k in targets:
            del self._mem[k]
        return len(targets)

    def clear(self) -> None:
        if self._table:
            self._spark.sql(f"DELETE FROM {self._table}")
        else:
            self._mem.clear()

    # ---- Tag queries -------------------------------------------------------

    def get_by_tag(self, tag: str, prefix: str = "") -> List[Tuple[str, Dict]]:
        """Return [(key, record_dict), ...] for all entries carrying `tag`."""
        rows = self._df(prefix).collect()
        result = []
        for row in rows:
            rec = self._row_to_dict(row)
            if tag in rec["tags"]:
                result.append((row.kv_key, rec))
        return result

    def all_tags(self, prefix: str = "") -> List[str]:
        rows = self._df(prefix).select("tags").collect()
        tags: set = set()
        for row in rows:
            if row.tags:
                tags.update(_jload(row.tags))
        return sorted(tags)

    # ---- Search ------------------------------------------------------------

    def search(
        self,
        query: str,
        prefix: str = "",
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Keyword search over keys, JSON-serialised values, and tags.

        On Databricks this can be enhanced with Spark MLlib TF-IDF or
        a Vector Search index; here we use a simple LIKE / contains scan
        which is optimised by Delta's data-skipping on string predicates.
        """
        q = query.lower()
        rows = self._df(prefix).collect()
        hits: List[Tuple[str, float]] = []
        for row in rows:
            score = 0.0
            if q in row.kv_key.lower():
                score += 2.0
            if q in row.value.lower():
                score += row.value.lower().count(q) * 1.0
            tags = _jload(row.tags) if row.tags else []
            for tag in tags:
                if q in tag.lower():
                    score += 1.5
            if score > 0:
                hits.append((row.kv_key, score))
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits[:top_k]

    # ---- Snapshot / restore ------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Export full store as a plain dict (JSON-serialisable)."""
        return {
            key: {
                "value": rec["value"],
                "tags": _jload(rec["tags"])
                if isinstance(rec["tags"], str)
                else rec["tags"],
                "created_at": rec["created_at"],
                "updated_at": rec["updated_at"],
                "source": rec["source"],
            }
            for key, rec in self.all_records()
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Overwrite store contents from a snapshot dict."""
        self.clear()
        for key, data in snapshot.items():
            self.set(
                key,
                data["value"],
                tags=data.get("tags", []),
                source=data.get("source"),
            )

    # ---- Diagnostics -------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        total = self._df().count()
        return {
            "total_keys": total,
            "all_tags": self.all_tags(),
            "backend": f"Delta:{self._table}" if self._table else "Spark in-memory",
        }

    def __repr__(self) -> str:
        return f"SparkKVStore(table={self._table!r}, keys={self._df().count()})"


# ===========================================================================
# 1. KVChatMessageHistory  (BaseChatMessageHistory → SparkKVStore)
# ===========================================================================


class KVChatMessageHistory(BaseChatMessageHistory):
    """
    Chat message history for one session, stored in SparkKVStore.

    Key layout
    ----------
    msg::{session_id}::{seq:08d}  → {"role":..., "content":..., "ts":...}
    idx::{session_id}             → next sequence int (as JSON number)

    Usage
    -----
    store   = SparkKVStore(spark)
    history = KVChatMessageHistory("thread-42", store=store)
    history.add_user_message("Hello!")
    history.add_ai_message("Hi there!")
    print(history.messages)
    """

    _ROLE_MAP = {HumanMessage: "human", AIMessage: "ai", SystemMessage: "system"}
    _TYPE_MAP = {"human": HumanMessage, "ai": AIMessage, "system": SystemMessage}

    def __init__(self, session_id: str, store: SparkKVStore) -> None:
        self.session_id = session_id
        self._store = store
        self._idx_key = f"idx::{session_id}"
        self._msg_prefix = f"msg::{session_id}::"

    def _next_seq(self) -> int:
        seq = self._store.get(self._idx_key, 0)
        self._store.set(self._idx_key, seq + 1)
        return seq

    # ---- BaseChatMessageHistory --------------------------------------------

    @property
    def messages(self) -> List[BaseMessage]:
        pairs = self._store.all_records(prefix=self._msg_prefix)
        # Sort by the zero-padded sequence suffix
        pairs.sort(key=lambda kv: kv[0].split("::")[-1])
        result: List[BaseMessage] = []
        for _, rec in pairs:
            data = rec["value"]  # already deserialised by _row_to_dict
            cls = self._TYPE_MAP.get(data["role"], HumanMessage)
            result.append(
                cls(content=data["content"], additional_kwargs=data.get("meta", {}))
            )
        return result

    def add_message(self, message: BaseMessage) -> None:
        seq = self._next_seq()
        role = self._ROLE_MAP.get(type(message), "human")
        key = f"{self._msg_prefix}{seq:08d}"
        self._store.set(
            key,
            {
                "role": role,
                "content": message.content,
                "meta": message.additional_kwargs,
                "ts": _now(),
            },
            tags=["message", role, self.session_id],
        )

    def clear(self) -> None:
        self._store.clear_prefix(self._msg_prefix)
        self._store.delete(self._idx_key)

    # ---- Helpers -----------------------------------------------------------

    def get_session_metadata(self) -> Dict[str, Any]:
        pairs = self._store.all_records(prefix=self._msg_prefix)
        if not pairs:
            return {
                "session_id": self.session_id,
                "message_count": 0,
                "first_message": None,
                "last_message": None,
            }
        timestamps = [rec["value"]["ts"] for _, rec in pairs]
        return {
            "session_id": self.session_id,
            "message_count": len(pairs),
            "first_message": _iso(min(timestamps)),
            "last_message": _iso(max(timestamps)),
        }


# ===========================================================================
# 2. ThreadMemoryManager
# ===========================================================================


class ThreadMemoryManager:
    """
    Manages independent conversation threads, all sharing one SparkKVStore.

    Usage
    -----
    mgr = ThreadMemoryManager(store)
    t1  = mgr.get_thread("user-123-support")
    t1.add_user_message("My order is late.")
    """

    def __init__(self, store: SparkKVStore) -> None:
        self._store = store
        self._threads: Dict[str, KVChatMessageHistory] = {}

    def get_thread(self, thread_id: str) -> KVChatMessageHistory:
        if thread_id not in self._threads:
            self._threads[thread_id] = KVChatMessageHistory(thread_id, self._store)
        return self._threads[thread_id]

    def list_threads(self) -> List[str]:
        return [tid for tid, h in self._threads.items() if h.messages]

    def delete_thread(self, thread_id: str) -> None:
        self.get_thread(thread_id).clear()

    def thread_summary(self, thread_id: str) -> Dict[str, Any]:
        return self.get_thread(thread_id).get_session_metadata()

    def all_summaries(self) -> List[Dict[str, Any]]:
        return [self.thread_summary(tid) for tid in self._threads]


# ===========================================================================
# 3. WindowChatMemory  (BaseMemory with rolling window)
# ===========================================================================


class WindowChatMemory(BaseMemory):
    """
    Rolling-window chat memory backed by SparkKVStore.

    All turns are persisted; only the last ``k`` (human, AI) pairs are
    surfaced in the chain context.  Plug directly into ConversationChain.

    Usage
    -----
    mem   = WindowChatMemory(session_id="s1", k=10)
    mem.bind_store(store)
    chain = ConversationChain(llm=llm, memory=mem)
    """

    session_id: str = Field(default="default")
    k: int = Field(default=10)
    human_prefix: str = Field(default="Human")
    ai_prefix: str = Field(default="AI")
    memory_key: str = Field(default="history")

    _history: Optional[KVChatMessageHistory] = None
    _store: Optional[SparkKVStore] = None

    class Config:
        arbitrary_types_allowed = True

    def bind_store(self, store: SparkKVStore) -> "WindowChatMemory":
        self._store = store
        self._history = KVChatMessageHistory(self.session_id, store)
        return self

    def _get_history(self) -> KVChatMessageHistory:
        if self._history is None:
            raise RuntimeError(
                "Call bind_store(spark_kv_store) before using WindowChatMemory."
            )
        return self._history

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        msgs = self._get_history().messages
        window = msgs[-(self.k * 2) :]
        lines: List[str] = []
        for msg in window:
            if isinstance(msg, HumanMessage):
                lines.append(f"{self.human_prefix}: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"{self.ai_prefix}: {msg.content}")
        return {self.memory_key: "\n".join(lines)}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        h = self._get_history()
        inp = (
            inputs.get("input")
            or inputs.get("human_input")
            or next(iter(inputs.values()), "")
        )
        out = (
            outputs.get("output")
            or outputs.get("response")
            or next(iter(outputs.values()), "")
        )
        h.add_user_message(str(inp))
        h.add_ai_message(str(out))

    def clear(self) -> None:
        self._get_history().clear()

    def get_stats(self) -> Dict[str, Any]:
        h = self._get_history()
        msgs = h.messages
        return {
            "session_id": self.session_id,
            "total_messages": len(msgs),
            "window_size": self.k,
            "messages_in_context": min(len(msgs), self.k * 2),
            **h.get_session_metadata(),
        }


# ===========================================================================
# 4. KVMemoryStore  (tagged KV + text ingestion + search)
# ===========================================================================


class KVMemoryStore:
    """
    High-level namespaced store for config, summaries, file chunks, etc.,
    backed by SparkKVStore.

    Usage
    -----
    kv = KVMemoryStore(store, namespace="kv")
    kv.set("goal", "RAG pipeline", tags=["project"])
    kv.ingest_text("Long document text …", name="spec", chunk_size=500)
    kv.search("pipeline")
    """

    def __init__(self, store: SparkKVStore, namespace: str = "kv") -> None:
        self._store = store
        self._ns = namespace + "::"

    def _k(self, key: str) -> str:
        return f"{self._ns}{key}"

    # ---- CRUD --------------------------------------------------------------

    def set(
        self,
        key: str,
        value: Any,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
    ) -> None:
        self._store.set(self._k(key), value, tags=tags, source=source)

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(self._k(key), default)

    def delete(self, key: str) -> bool:
        return self._store.delete(self._k(key))

    def keys(self) -> List[str]:
        return [k[len(self._ns) :] for k in self._store.keys(prefix=self._ns)]

    def all_items(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": k[len(self._ns) :],
                "value": rec["value"],
                "tags": rec["tags"],
                "source": rec["source"],
                "updated_at": _iso(rec["updated_at"]),
            }
            for k, rec in self._store.all_records(prefix=self._ns)
        ]

    # ---- Tag queries -------------------------------------------------------

    def get_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        return [
            {"key": k[len(self._ns) :], "value": rec["value"], "tags": rec["tags"]}
            for k, rec in self._store.get_by_tag(tag, prefix=self._ns)
        ]

    def all_tags(self) -> List[str]:
        return self._store.all_tags(prefix=self._ns)

    # ---- Search ------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        raw = self._store.search(query, prefix=self._ns, top_k=top_k)
        return [(k[len(self._ns) :], score) for k, score in raw]

    # ---- Text ingestion ----------------------------------------------------

    def ingest_text(
        self,
        text: str,
        name: str,
        chunk_size: int = 500,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
    ) -> List[str]:
        """Chunk text and store each chunk. Returns list of short keys created."""
        words = text.split()
        step = max(1, chunk_size // 2)
        chunks = [
            " ".join(words[i : i + chunk_size])
            for i in range(0, max(1, len(words) - chunk_size + 1), step)
        ]
        base_tags = (tags or []) + ["chunk", name]
        created: List[str] = []
        for idx, chunk in enumerate(chunks):
            key = f"{name}::chunk{idx}"
            self.set(key, chunk, tags=base_tags, source=source or name)
            created.append(key)
        return created

    # ---- Snapshot ----------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        return {
            k[len(self._ns) :]: {
                "value": rec["value"],
                "tags": rec["tags"],
                "source": rec["source"],
                "created_at": rec["created_at"],
            }
            for k, rec in self._store.all_records(prefix=self._ns)
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "namespace": self._ns.rstrip("::"),
            "total_entries": len(self._store.keys(prefix=self._ns)),
            "tags": self.all_tags(),
            "backend": self._store.stats()["backend"],
        }


# ===========================================================================
# 5. DatabricksMemory  —  unified façade
# ===========================================================================


class DatabricksMemory:
    """
    Unified façade over all memory layers, all sharing one SparkKVStore.

    Parameters
    ----------
    spark : SparkSession
        Active Databricks / local Spark session.
    table : str | None
        Fully-qualified Delta table name, e.g. ``"main.ml.langchain_memory"``.
        Pass ``None`` (default) for local in-memory operation.

    Usage — Databricks
    ------------------
        mem = DatabricksMemory(
            spark=spark,
            table="main.langchain.memory",
        )
        thread = mem.get_thread("user-42")
        thread.add_user_message("Hello!")

        chain = ConversationChain(
            llm=llm,
            memory=mem.chat_memory(session_id="s1", k=10),
        )

        mem.kv.set("goal", "Build a RAG pipeline", tags=["project"])

    Usage — local
    -------------
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.master("local").appName("mem").getOrCreate()
        mem   = DatabricksMemory(spark=spark)
    """

    def __init__(
        self,
        spark: "SparkSession",
        table: Optional[str] = None,
    ) -> None:
        self._store = SparkKVStore(spark=spark, table=table)
        self.threads = ThreadMemoryManager(store=self._store)
        self.kv = KVMemoryStore(store=self._store, namespace="kv")

    def get_thread(self, thread_id: str) -> KVChatMessageHistory:
        return self.threads.get_thread(thread_id)

    def chat_memory(self, session_id: str = "default", k: int = 10) -> WindowChatMemory:
        mem = WindowChatMemory(session_id=session_id, k=k)
        mem.bind_store(self._store)
        return mem

    def snapshot(self) -> Dict[str, Any]:
        """Dump entire store to a plain dict (JSON-serialisable)."""
        return self._store.snapshot()

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Replace store contents from a snapshot dict."""
        self._store.restore(snapshot)

    def summary(self) -> Dict[str, Any]:
        return {
            "store": self._store.stats(),
            "threads": self.threads.list_threads(),
            "kv": self.kv.stats(),
        }
