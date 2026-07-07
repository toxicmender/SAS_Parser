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
kv_key      STRING NOT NULL   — namespaced key
                                e.g. "msg::thread-1::0001783440000000-9f3a"
value       STRING NOT NULL   — JSON-serialised payload
tags        STRING            — JSON array of tag strings
created_at  DOUBLE            — Unix timestamp (float)
updated_at  DOUBLE            — Unix timestamp (float)
source      STRING            — optional provenance label

Message keys embed a zero-padded microsecond timestamp plus a short random
suffix, so they are collision-free without any read-modify-write sequence
counter and sort lexicographically in time order (legacy ``{seq:08d}``
keys sort before them, i.e. before any new message).

On Databricks the SparkKVStore writes to a Delta table and uses
MERGE INTO for upserts (``set_many`` batches several entries into one
MERGE).  In local / CI mode it keeps state in a plain Python dict.

Usage — Databricks notebook
----------------------------
    from memory.short_mem import DatabricksMemory

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
    from memory.short_mem import DatabricksMemory
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local").appName("mem").getOrCreate()
    mem   = DatabricksMemory(spark=spark)   # table=None → pure in-memory DF
"""

from __future__ import annotations

import datetime
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel, ConfigDict, Field

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


logger = logging.getLogger(__name__)


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


def _sql_quote(s: str) -> str:
    """Escape a value for safe inlining inside a single-quoted SQL literal.

    Keys are internally generated today, but the DELETE statements in Delta
    mode interpolate them into SQL text; doubling single quotes prevents a
    stray quote from breaking (or injecting into) the statement.
    """
    return s.replace("'", "''")


# LIKE-pattern escape character.  '~' is used instead of the conventional
# backslash so the escaped pattern survives Spark SQL string-literal
# processing without a second layer of backslash doubling.
_LIKE_ESCAPE = "~"


def _sql_like_prefix(prefix: str) -> str:
    """Escape LIKE wildcards in *prefix* for a literal prefix match.

    ``_`` matches any single character in LIKE and ``%`` any run, so a key
    prefix like ``msg::thread_1::`` would otherwise also match
    ``msg::threadX1::…``.  The result must be used with ``ESCAPE '~'``.
    """
    return (
        prefix.replace(_LIKE_ESCAPE, _LIKE_ESCAPE * 2)
        .replace("%", _LIKE_ESCAPE + "%")
        .replace("_", _LIKE_ESCAPE + "_")
    )


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
        """Return a DataFrame view of the Delta table (Delta mode only).

        Every in-memory-mode caller has a dict fast path via
        :meth:`_mem_items`; wrapping the dict in a DataFrame here was dead
        code and would have required a live SparkSession the in-memory
        tests deliberately run without.
        """
        if not self._table:
            raise RuntimeError(
                "_df() is Delta-mode only; in-memory mode reads _mem_items()"
            )
        df = self._spark.table(self._table)
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

    def _mem_row_to_dict(self, rec: Dict) -> Dict:
        """In-memory-mode counterpart of :meth:`_row_to_dict`.

        ``rec`` is a raw dict from ``self._mem`` (``value``/``tags`` still
        JSON-encoded); this returns the same deserialised shape the DataFrame
        path produces via :meth:`_row_to_dict`.
        """
        return {
            "value": _jload(rec["value"]),
            "tags": _jload(rec["tags"]) if rec["tags"] else [],
            "created_at": rec["created_at"],
            "updated_at": rec["updated_at"],
            "source": rec["source"],
        }

    def _mem_items(self, prefix: str = ""):
        """Yield ``(key, raw_rec)`` from the in-memory dict, prefix-filtered."""
        for k, rec in self._mem.items():
            if not prefix or k.startswith(prefix):
                yield k, rec

    # ---- CRUD --------------------------------------------------------------

    def set(
        self,
        key: str,
        value: Any,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
    ) -> None:
        self.set_many([{"key": key, "value": value, "tags": tags, "source": source}])

    def set_many(self, entries: List[Dict[str, Any]]) -> None:
        """Upsert several entries at once — a single MERGE in Delta mode.

        Each entry is a dict with keys:

        - ``key`` (required), ``value`` (required)
        - ``tags``: list of tag strings, or None to preserve existing tags
          on update (both modes now agree on this — Delta previously
          overwrote tags with ``[]`` when None was passed)
        - ``source``: provenance label, or None to preserve existing
        - ``created_at`` / ``updated_at``: optional explicit timestamps
          (used by :meth:`restore` to keep snapshot history); default now.
        """
        if not entries:
            return
        now = _now()

        if self._table:
            rows = [
                (
                    e["key"],
                    _jdump(e["value"]),
                    _jdump(e["tags"]) if e.get("tags") is not None else None,
                    e.get("created_at", now),
                    e.get("updated_at", now),
                    e.get("source"),
                )
                for e in entries
            ]
            # Unique per-call view name: a fixed name would let two
            # concurrent writers on the same SparkSession clobber each
            # other's staged rows between createOrReplaceTempView and MERGE.
            view = f"_kv_upsert_{uuid.uuid4().hex}"
            staged = self._spark.createDataFrame(rows, schema=_schema())
            staged.createOrReplaceTempView(view)
            try:
                self._spark.sql(f"""
                    MERGE INTO {self._table} AS target
                    USING {view} AS source
                      ON target.kv_key = source.kv_key
                    WHEN MATCHED THEN UPDATE SET
                        target.value      = source.value,
                        target.tags       = COALESCE(source.tags, target.tags),
                        target.updated_at = source.updated_at,
                        target.source     = COALESCE(source.source, target.source)
                    WHEN NOT MATCHED THEN INSERT *
                """)
            finally:
                self._spark.catalog.dropTempView(view)
        else:
            for e in entries:
                key = e["key"]
                tags = e.get("tags")
                existing = self._mem.get(key)
                self._mem[key] = {
                    "value": _jdump(e["value"]),
                    "tags": _jdump(tags)
                    if tags is not None
                    else (existing["tags"] if existing else _jdump([])),
                    "created_at": e.get(
                        "created_at",
                        existing["created_at"] if existing else now,
                    ),
                    "updated_at": e.get("updated_at", now),
                    "source": e.get("source")
                    or (existing["source"] if existing else None),
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
            self._spark.sql(
                f"DELETE FROM {self._table} WHERE kv_key = '{_sql_quote(key)}'"
            )
            return True
        if key in self._mem:
            del self._mem[key]
            return True
        return False

    def delete_many(self, keys: List[str]) -> int:
        """Delete several keys — a single DELETE … IN in Delta mode.

        Returns the number of keys that existed and were removed.
        """
        if not keys:
            return 0
        if self._table:
            key_set = set(keys)
            existing = self._df().filter(F.col("kv_key").isin(list(key_set))).count()
            if existing:
                quoted = ", ".join(f"'{_sql_quote(k)}'" for k in key_set)
                self._spark.sql(
                    f"DELETE FROM {self._table} WHERE kv_key IN ({quoted})"
                )
            return existing
        removed = 0
        for k in keys:
            if k in self._mem:
                del self._mem[k]
                removed += 1
        return removed

    def keys(self, prefix: str = "") -> List[str]:
        if self._table is None:
            return [k for k, _ in self._mem_items(prefix)]
        return [row.kv_key for row in self._df(prefix).select("kv_key").collect()]

    def has_prefix(self, prefix: str) -> bool:
        """True if at least one key starts with *prefix*.

        Cheaper than ``bool(keys(prefix))`` in Delta mode: stops at the
        first matching row instead of collecting every key.
        """
        if self._table is None:
            return any(True for _ in self._mem_items(prefix))
        return bool(self._df(prefix).limit(1).count())

    def all_records(self, prefix: str = "") -> List[Tuple[str, Dict]]:
        if self._table is None:
            return [
                (k, self._mem_row_to_dict(rec)) for k, rec in self._mem_items(prefix)
            ]
        rows = self._df(prefix).collect()
        return [(row.kv_key, self._row_to_dict(row)) for row in rows]

    def clear_prefix(self, prefix: str) -> int:
        if self._table:
            count = self._df(prefix).count()
            # LIKE wildcards in the prefix must be escaped: '_' matches any
            # character, so "msg::thread_1::" would otherwise also delete
            # "msg::threadX1::…" — unlike every read path, which matches
            # the prefix literally via startswith.
            pattern = _sql_like_prefix(prefix)
            self._spark.sql(
                f"DELETE FROM {self._table} "
                f"WHERE kv_key LIKE '{_sql_quote(pattern)}%' ESCAPE '{_LIKE_ESCAPE}'"
            )
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
        if self._table is None:
            result = []
            for k, raw in self._mem_items(prefix):
                rec = self._mem_row_to_dict(raw)
                if tag in rec["tags"]:
                    result.append((k, rec))
            return result
        rows = self._df(prefix).collect()
        result = []
        for row in rows:
            rec = self._row_to_dict(row)
            if tag in rec["tags"]:
                result.append((row.kv_key, rec))
        return result

    def all_tags(self, prefix: str = "") -> List[str]:
        tags: set = set()
        if self._table is None:
            for _, raw in self._mem_items(prefix):
                if raw["tags"]:
                    tags.update(_jload(raw["tags"]))
            return sorted(tags)
        rows = self._df(prefix).select("tags").collect()
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
        # Unify both backends into (key, value_str, tags_list) tuples so the
        # scoring below is written once.  In-memory mode reads the dict
        # directly; Delta mode collects the DataFrame.
        if self._table is None:
            scannable = (
                (k, raw["value"], _jload(raw["tags"]) if raw["tags"] else [])
                for k, raw in self._mem_items(prefix)
            )
        else:
            scannable = (
                (row.kv_key, row.value, _jload(row.tags) if row.tags else [])
                for row in self._df(prefix).collect()
            )

        hits: List[Tuple[str, float]] = []
        for key, value_str, tags in scannable:
            score = 0.0
            if q in key.lower():
                score += 2.0
            if q in value_str.lower():
                score += value_str.lower().count(q) * 1.0
            for tag in tags:
                if q in tag.lower():
                    score += 1.5
            if score > 0:
                hits.append((key, score))
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits[:top_k]

    # ---- Snapshot / restore ------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Export full store as a plain dict (JSON-serialisable)."""
        # all_records() already deserialises value and tags
        return {
            key: {
                "value": rec["value"],
                "tags": rec["tags"],
                "created_at": rec["created_at"],
                "updated_at": rec["updated_at"],
                "source": rec["source"],
            }
            for key, rec in self.all_records()
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Overwrite store contents from a snapshot dict.

        The snapshot's ``created_at``/``updated_at`` are preserved (a
        restored store previously re-stamped every entry to restore time,
        losing the history the snapshot carried).
        """
        self.clear()
        entries = []
        for key, data in snapshot.items():
            entry: Dict[str, Any] = {
                "key": key,
                "value": data["value"],
                "tags": data.get("tags", []),
                "source": data.get("source"),
            }
            if data.get("created_at") is not None:
                entry["created_at"] = data["created_at"]
            if data.get("updated_at") is not None:
                entry["updated_at"] = data["updated_at"]
            entries.append(entry)
        self.set_many(entries)

    # ---- Diagnostics -------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        total = len(self._mem) if self._table is None else self._df().count()
        return {
            "total_keys": total,
            "all_tags": self.all_tags(),
            "backend": f"Delta:{self._table}" if self._table else "Spark in-memory",
        }

    def __repr__(self) -> str:
        count = len(self._mem) if self._table is None else self._df().count()
        return f"SparkKVStore(table={self._table!r}, keys={count})"


# ===========================================================================
# 1. KVChatMessageHistory  (BaseChatMessageHistory → SparkKVStore)
# ===========================================================================


class KVChatMessageHistory(BaseChatMessageHistory):
    """
    Chat message history for one session, stored in SparkKVStore.

    Key layout
    ----------
    msg::{session_id}::{epoch_micros:016d}-{rand4}
        → {"role":..., "content":..., "meta":..., "ts":...}

    Keys embed a microsecond timestamp plus a random suffix, so they are
    collision-free with no read-modify-write sequence counter (the old
    ``idx::{session_id}`` get-then-set raced under concurrent writers and
    silently overwrote messages) and sort lexicographically in time order.
    Legacy ``{seq:08d}`` keys sort before every timestamp key, i.e. legacy
    messages stay ahead of any new ones.

    Usage
    -----
    store   = SparkKVStore(spark)
    history = KVChatMessageHistory("thread-42", store=store)
    history.add_user_message("Hello!")
    history.add_ai_message("Hi there!")
    print(history.messages)
    """

    _TYPE_MAP = {"human": HumanMessage, "ai": AIMessage, "system": SystemMessage}

    def __init__(self, session_id: str, store: SparkKVStore) -> None:
        self.session_id = session_id
        self._store = store
        self._idx_key = f"idx::{session_id}"  # legacy counter key, cleanup only
        self._msg_prefix = f"msg::{session_id}::"
        self._last_us = 0

    def _msg_key(self, ts: float) -> str:
        """Build a time-ordered, collision-free message key for *ts*."""
        us = int(ts * 1_000_000)
        if us <= self._last_us:
            # Keep same-microsecond writes from this instance in insertion
            # order (the random suffix alone would shuffle them).
            us = self._last_us + 1
        self._last_us = us
        return f"{self._msg_prefix}{us:016d}-{uuid.uuid4().hex[:4]}"

    # ---- BaseChatMessageHistory --------------------------------------------

    @property
    def messages(self) -> List[BaseMessage]:
        pairs = self._store.all_records(prefix=self._msg_prefix)
        # Sort by the key suffix: zero-padded timestamps (and legacy
        # zero-padded sequence numbers) order lexicographically by time.
        pairs.sort(key=lambda kv: kv[0][len(self._msg_prefix) :])
        result: List[BaseMessage] = []
        for _, rec in pairs:
            data = rec["value"]  # already deserialised by _row_to_dict
            role = data.get("role", "human")
            cls = self._TYPE_MAP.get(role)
            meta = data.get("meta", {})
            if cls is not None:
                result.append(cls(content=data["content"], additional_kwargs=meta))
            else:
                # Tool/function/custom roles round-trip via ChatMessage
                # instead of being silently rewritten as human messages.
                result.append(
                    ChatMessage(role=role, content=data["content"], additional_kwargs=meta)
                )
        return result

    def add_message(self, message: BaseMessage) -> None:
        self.add_messages([message])

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Persist several messages in ONE store write (one Delta MERGE)."""
        ts = _now()
        entries = []
        for message in messages:
            # ChatMessage carries the actual role in .role; every other
            # BaseMessage subclass identifies itself via .type.
            role = getattr(message, "role", None) or message.type
            entries.append(
                {
                    "key": self._msg_key(ts),
                    "value": {
                        "role": role,
                        "content": message.content,
                        "meta": message.additional_kwargs,
                        "ts": ts,
                    },
                    "tags": ["message", role, self.session_id],
                }
            )
        self._store.set_many(entries)

    def clear(self) -> None:
        self._store.clear_prefix(self._msg_prefix)
        self._store.delete(self._idx_key)  # legacy counter, if present

    def has_messages(self) -> bool:
        """True if this thread has at least one stored message.

        Checks only for the presence of a message key — in Delta mode this
        stops at the first matching row instead of collecting every key.
        """
        return self._store.has_prefix(self._msg_prefix)

    def prune_before(self, cutoff_ts: float) -> int:
        """Delete messages whose stored ``ts`` is older than *cutoff_ts*.

        The rolling window in WindowChatMemory only limits what is
        *surfaced* to a chain; this is the complementary storage-side trim
        (one DELETE in Delta mode).  Returns the number removed.
        """
        stale = [
            key
            for key, rec in self._store.all_records(prefix=self._msg_prefix)
            if rec["value"].get("ts", 0.0) < cutoff_ts
        ]
        removed = self._store.delete_many(stale)
        logger.info(
            f"prune_before: session '{self.session_id}' removed {removed} message(s)"
        )
        return removed

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
        """Thread ids with at least one stored message.

        Derived from the store's keys — not from the threads this manager
        instance happens to have touched — so with a persistent Delta
        backend the list survives a process restart.
        """
        ids = {
            key[len("msg::") :].rsplit("::", 1)[0]
            for key in self._store.keys(prefix="msg::")
        }
        return sorted(ids)

    def delete_thread(self, thread_id: str) -> None:
        self.get_thread(thread_id).clear()

    def thread_summary(self, thread_id: str) -> Dict[str, Any]:
        return self.get_thread(thread_id).get_session_metadata()

    def all_summaries(self) -> List[Dict[str, Any]]:
        return [self.thread_summary(tid) for tid in self.list_threads()]


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

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
        # One store write for the whole turn (a single MERGE in Delta mode)
        h.add_messages(
            [HumanMessage(content=str(inp)), AIMessage(content=str(out))]
        )

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

    def prune_thread(self, thread_id: str, before_ts: float) -> int:
        """Storage-side trim: delete *thread_id* messages older than
        *before_ts* (Unix seconds).  Returns the number removed."""
        return self.get_thread(thread_id).prune_before(before_ts)

    def summary(self) -> Dict[str, Any]:
        return {
            "store": self._store.stats(),
            "threads": self.threads.list_threads(),
            "kv": self.kv.stats(),
        }
