"""LangChain chat-history and KV memory. See memory/README.md.

Persists to a Databricks Delta table in production; runs on a plain Python dict
locally, with no Spark (or JVM) required at all in that mode.

Logger name: ``memory.store``.
"""

from __future__ import annotations

import datetime
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    message_to_dict,
    messages_from_dict,
)

# PySpark is optional — required only for Delta mode; the in-memory backend
# (table=None) must import and run without it. The type checker imports these
# unconditionally so uses below aren't "possibly unbound"; _PYSPARK_AVAILABLE
# is what actually gates them at runtime.
if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        DoubleType,
        StringType,
        StructField,
        StructType,
    )

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


# LIKE-pattern escape character.  All Delta-mode SQL passes values through
# Spark's parameter markers (spark.sql(sql, args)), so there is no
# string-literal layer to survive; '~' is simply the escape char declared in
# the LIKE's ESCAPE clause.
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


# Schema for the Delta table (built lazily; Delta mode only)
_KV_SCHEMA = None


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


# Storage backends. Both speak the same raw record shape, a tuple in
# Delta-schema column order (value/tags kept JSON-serialised). Upsert rows with
# created_at=None get each backend's preservation semantics (keep existing on
# update, default to now on insert).
_RawRecord = Tuple[str, str, Optional[str], Optional[float], float, Optional[str]]


class _InMemoryBackend:
    """Plain-dict backend for local / CI use.  No pyspark, no JVM."""

    def __init__(self) -> None:
        self._mem: Dict[str, Dict[str, Any]] = {}

    def upsert(self, rows: List[_RawRecord], now: float) -> None:
        for key, value_json, tags_json, created_at, updated_at, source in rows:
            existing = self._mem.get(key)
            self._mem[key] = {
                "value": value_json,
                # tags=None / source=None on update preserve the existing
                # values (mirrors the Delta MERGE's COALESCE below).
                "tags": tags_json
                if tags_json is not None
                else (existing["tags"] if existing else _jdump([])),
                "created_at": created_at
                if created_at is not None
                else (existing["created_at"] if existing else now),
                "updated_at": updated_at,
                "source": source or (existing["source"] if existing else None),
            }

    def replace_all(self, rows: List[_RawRecord], now: float) -> None:
        """Atomically replace the whole store with *rows*."""
        fresh: Dict[str, Dict[str, Any]] = {}
        for key, value_json, tags_json, created_at, updated_at, source in rows:
            fresh[key] = {
                "value": value_json,
                "tags": tags_json if tags_json is not None else _jdump([]),
                "created_at": created_at if created_at is not None else now,
                "updated_at": updated_at,
                "source": source,
            }
        self._mem = fresh

    def get(self, key: str) -> Optional[str]:
        rec = self._mem.get(key)
        return rec["value"] if rec else None

    def records(self, prefix: str = "") -> List[_RawRecord]:
        return self.records_after(prefix, "")

    def records_after(self, prefix: str, after_key: str) -> List[_RawRecord]:
        return [
            (k, r["value"], r["tags"], r["created_at"], r["updated_at"], r["source"])
            for k, r in self._mem.items()
            if (not prefix or k.startswith(prefix)) and k > after_key
        ]

    def keys(self, prefix: str = "") -> List[str]:
        return [k for k in self._mem if not prefix or k.startswith(prefix)]

    def has_prefix(self, prefix: str) -> bool:
        return any(k.startswith(prefix) for k in self._mem)

    def delete_many(self, keys: List[str]) -> int:
        removed = 0
        for k in keys:
            if k in self._mem:
                del self._mem[k]
                removed += 1
        return removed

    def clear_prefix(self, prefix: str) -> int:
        targets = [k for k in self._mem if k.startswith(prefix)]
        for k in targets:
            del self._mem[k]
        return len(targets)

    def clear(self) -> None:
        self._mem.clear()

    def count(self) -> int:
        return len(self._mem)

    @property
    def label(self) -> str:
        return "in-memory"


class _DeltaBackend:
    """Spark DataFrame / Databricks Delta table backend (production).

    Upserts use ``MERGE INTO``, deletes use ``DELETE FROM``.  Requires
    pyspark and an active SparkSession; the table is created with Delta
    format if it does not already exist.
    """

    def __init__(self, spark: "SparkSession | None", table: str) -> None:
        if not _PYSPARK_AVAILABLE:
            raise RuntimeError("pyspark is not installed. Run: pip install pyspark")
        if spark is None:
            raise ValueError(
                f"Delta mode (table={table!r}) requires an active SparkSession"
            )
        self._spark = spark
        self._table = table
        self._ensure_table()

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

    def _df(self, prefix: str = "") -> "DataFrame":
        df = self._spark.table(self._table)
        if prefix:
            df = df.filter(F.col("kv_key").startswith(prefix))
        return df

    def _stage(self, rows: List[_RawRecord], now: float) -> "DataFrame":
        # created_at=None (not supplied) defaults to now; the MERGE's
        # WHEN MATCHED branch never touches created_at, so an existing
        # row's creation time is preserved on update.
        staged_rows = [
            (key, value_json, tags_json, created_at if created_at is not None else now, updated_at, source)
            for key, value_json, tags_json, created_at, updated_at, source in rows
        ]
        return self._spark.createDataFrame(staged_rows, schema=_schema())

    def upsert(self, rows: List[_RawRecord], now: float) -> None:
        # Unique per-call view name: a fixed name would let two concurrent
        # writers on the same SparkSession clobber each other's staged rows
        # between createOrReplaceTempView and MERGE.
        view = f"_kv_upsert_{uuid.uuid4().hex}"
        self._stage(rows, now).createOrReplaceTempView(view)
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

    def replace_all(self, rows: List[_RawRecord], now: float) -> None:
        """Atomically replace the table contents with *rows*.

        ``INSERT OVERWRITE`` is a single Delta commit, so a reader never
        observes the empty intermediate state a DELETE-then-insert would
        expose (and a crash cannot leave the table emptied).
        """
        view = f"_kv_replace_{uuid.uuid4().hex}"
        self._stage(rows, now).createOrReplaceTempView(view)
        try:
            self._spark.sql(f"INSERT OVERWRITE {self._table} SELECT * FROM {view}")
        finally:
            self._spark.catalog.dropTempView(view)

    def get(self, key: str) -> Optional[str]:
        rows = (
            self._df()
            .filter(F.col("kv_key") == key)
            .select("value")
            .limit(1)
            .collect()
        )
        return rows[0].value if rows else None

    def records(self, prefix: str = "") -> List[_RawRecord]:
        return self.records_after(prefix, "")

    def records_after(self, prefix: str, after_key: str) -> List[_RawRecord]:
        df = self._df(prefix)
        if after_key:
            df = df.filter(F.col("kv_key") > after_key)
        return [
            (row.kv_key, row.value, row.tags, row.created_at, row.updated_at, row.source)
            for row in df.collect()
        ]

    def keys(self, prefix: str = "") -> List[str]:
        return [row.kv_key for row in self._df(prefix).select("kv_key").collect()]

    def has_prefix(self, prefix: str) -> bool:
        # Stops at the first matching row instead of collecting every key.
        return bool(self._df(prefix).limit(1).count())

    def delete_many(self, keys: List[str]) -> int:
        key_list = list(set(keys))
        existing = self._df().filter(F.col("kv_key").isin(key_list)).count()
        if existing:
            # Named parameter markers (Spark >= 3.4) — key values never
            # touch the SQL text, so no quoting/escaping is needed.
            markers = ", ".join(f":k{i}" for i in range(len(key_list)))
            args = {f"k{i}": key for i, key in enumerate(key_list)}
            self._spark.sql(
                f"DELETE FROM {self._table} WHERE kv_key IN ({markers})", args
            )
        return existing

    def clear_prefix(self, prefix: str) -> int:
        count = self._df(prefix).count()
        # LIKE wildcards in the prefix must be escaped: '_' matches any
        # character, so "msg::thread_1::" would otherwise also delete
        # "msg::threadX1::…" — unlike every read path, which matches
        # the prefix literally via startswith. The pattern itself is passed
        # as a parameter marker, never inlined into the SQL text.
        pattern = _sql_like_prefix(prefix) + "%"
        self._spark.sql(
            f"DELETE FROM {self._table} "
            f"WHERE kv_key LIKE :pattern ESCAPE '{_LIKE_ESCAPE}'",
            {"pattern": pattern},
        )
        return count

    def clear(self) -> None:
        self._spark.sql(f"DELETE FROM {self._table}")

    def count(self) -> int:
        return self._df().count()

    @property
    def label(self) -> str:
        return f"Delta:{self._table}"


# ===========================================================================
# CORE: KVStore
# ===========================================================================


class KVStore:
    """
    Key-value store, backed by one of two interchangeable backends.

    Two modes
    ---------
    - **Delta mode** (production):
      Pass ``spark`` and ``table="catalog.schema.table_name"``.
      The table is created with Delta format if it does not already exist.
      Upserts use ``MERGE INTO``, deletes use ``DELETE FROM``.

    - **In-memory mode** (local / CI):
      Pass ``table=None`` (default).  All state lives in a plain Python
      dict — pyspark is not required, no Spark session is touched.

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
    spark : SparkSession | None
        Active Spark / Databricks session.  Required only when ``table``
        is given; ignored in in-memory mode.
    table : str | None
        Fully-qualified Delta table name for persistence, or None for
        pure in-memory operation.
    """

    def __init__(
        self,
        spark: "SparkSession | None" = None,
        table: Optional[str] = None,
    ) -> None:
        self._table = table
        self._backend = _DeltaBackend(spark, table) if table else _InMemoryBackend()

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
          on update
        - ``source``: provenance label, or None to preserve existing
        - ``created_at`` / ``updated_at``: optional explicit timestamps
          (used by :meth:`restore` to keep snapshot history); default now.
        """
        if not entries:
            return
        now = _now()
        rows: List[_RawRecord] = [
            (
                e["key"],
                _jdump(e["value"]),
                _jdump(e["tags"]) if e.get("tags") is not None else None,
                e.get("created_at"),  # None → backend preserves/defaults
                e.get("updated_at", now),
                e.get("source"),
            )
            for e in entries
        ]
        self._backend.upsert(rows, now)

    def get(self, key: str, default: Any = None) -> Any:
        raw = self._backend.get(key)
        return _jload(raw) if raw is not None else default

    def delete(self, key: str) -> bool:
        return self._backend.delete_many([key]) == 1

    def delete_many(self, keys: List[str]) -> int:
        """Delete several keys — a single DELETE … IN in Delta mode.

        Returns the number of keys that existed and were removed.
        """
        if not keys:
            return 0
        return self._backend.delete_many(keys)

    def keys(self, prefix: str = "") -> List[str]:
        return self._backend.keys(prefix)

    def has_prefix(self, prefix: str) -> bool:
        """True if at least one key starts with *prefix*.

        Cheaper than ``bool(keys(prefix))`` in Delta mode: stops at the
        first matching row instead of collecting every key.
        """
        return self._backend.has_prefix(prefix)

    @staticmethod
    def _decode_record(raw: _RawRecord) -> Tuple[str, Dict]:
        key, value_json, tags_json, created_at, updated_at, source = raw
        return (
            key,
            {
                "value": _jload(value_json),
                "tags": _jload(tags_json) if tags_json else [],
                "created_at": created_at,
                "updated_at": updated_at,
                "source": source,
            },
        )

    def all_records(self, prefix: str = "") -> List[Tuple[str, Dict]]:
        return [self._decode_record(r) for r in self._backend.records(prefix)]

    def records_after(self, prefix: str, after_key: str) -> List[Tuple[str, Dict]]:
        """Records under *prefix* whose key sorts strictly after *after_key*.

        The append-only tail read: message keys are time-ordered, so a
        caller that remembers the last key it saw fetches only what is new
        instead of rescanning the prefix.
        """
        return [
            self._decode_record(r)
            for r in self._backend.records_after(prefix, after_key)
        ]

    def clear_prefix(self, prefix: str) -> int:
        return self._backend.clear_prefix(prefix)

    def clear(self) -> None:
        self._backend.clear()

    # ---- Tag queries -------------------------------------------------------

    def get_by_tag(self, tag: str, prefix: str = "") -> List[Tuple[str, Dict]]:
        """Return [(key, record_dict), ...] for all entries carrying `tag`."""
        return [(k, rec) for k, rec in self.all_records(prefix) if tag in rec["tags"]]

    def all_tags(self, prefix: str = "") -> List[str]:
        tags: set = set()
        for _, _, tags_json, _, _, _ in self._backend.records(prefix):
            if tags_json:
                tags.update(_jload(tags_json))
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
        a Vector Search index; here we use a simple contains scan over
        the raw records both backends expose in the same shape.
        """
        q = query.lower()
        hits: List[Tuple[str, float]] = []
        for key, value_json, tags_json, _, _, _ in self._backend.records(prefix):
            score = 0.0
            if q in key.lower():
                score += 2.0
            if q in value_json.lower():
                score += value_json.lower().count(q) * 1.0
            for tag in _jload(tags_json) if tags_json else []:
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
        """Overwrite store contents from a snapshot dict, atomically.

        The snapshot's ``created_at``/``updated_at`` are preserved (a
        restored store previously re-stamped every entry to restore time,
        losing the history the snapshot carried). The replacement is a
        single backend operation — one Delta ``INSERT OVERWRITE`` commit —
        so a crash mid-restore cannot leave the store emptied the way the
        old clear-then-write sequence could.
        """
        now = _now()
        rows: List[_RawRecord] = []
        for key, data in snapshot.items():
            updated_at = data.get("updated_at")
            rows.append(
                (
                    key,
                    _jdump(data["value"]),
                    _jdump(data.get("tags", [])),
                    data.get("created_at"),  # None → backend defaults to now
                    updated_at if updated_at is not None else now,
                    data.get("source"),
                )
            )
        self._backend.replace_all(rows, now)

    # ---- Diagnostics -------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "total_keys": self._backend.count(),
            "all_tags": self.all_tags(),
            "backend": self._backend.label,
        }

    def __repr__(self) -> str:
        return f"KVStore(table={self._table!r}, keys={self._backend.count()})"


# ===========================================================================
# 1. KVChatMessageHistory  (BaseChatMessageHistory → KVStore)
# ===========================================================================


class KVChatMessageHistory(BaseChatMessageHistory):
    """
    Chat message history for one session, stored in KVStore.

    Key layout
    ----------
    msg::{session_id}::{epoch_micros:016d}-{rand4}
        → {"message": <langchain message_to_dict>, "ts": ...}

    Values carry the full ``message_to_dict`` payload, so tool calls,
    ``usage_metadata``, ``response_metadata``, names, and ids all
    round-trip losslessly. Legacy rows written by the pre-lossless schema
    ({"role":..., "content":..., "meta":..., "ts":...}) are still read.

    Retention
    ---------
    ``max_age_s`` / ``max_messages`` (both off by default) bound the
    *stored* thread, applied opportunistically after every write: messages
    older than ``max_age_s`` seconds are pruned, then the oldest messages
    beyond ``max_messages`` are pruned. This automates what
    :meth:`prune_before` does manually — prompt-side trimming remains a
    separate concern.

    Keys embed a microsecond timestamp plus a random suffix, so they are
    collision-free with no read-modify-write sequence counter (the old
    ``idx::{session_id}`` get-then-set raced under concurrent writers and
    silently overwrote messages) and sort lexicographically in time order.
    Legacy ``{seq:08d}`` keys sort before every timestamp key, i.e. legacy
    messages stay ahead of any new ones.

    Usage
    -----
    store   = KVStore()          # in-memory; pass spark+table for Delta
    history = KVChatMessageHistory("thread-42", store=store)
    history.add_user_message("Hello!")
    history.add_ai_message("Hi there!")
    print(history.messages)
    """

    _TYPE_MAP = {"human": HumanMessage, "ai": AIMessage, "system": SystemMessage}

    def __init__(
        self,
        session_id: str,
        store: KVStore,
        *,
        max_age_s: float | None = None,
        max_messages: int | None = None,
    ) -> None:
        if max_age_s is not None and max_age_s <= 0:
            raise ValueError(f"max_age_s must be > 0, got {max_age_s}")
        if max_messages is not None and max_messages < 1:
            raise ValueError(f"max_messages must be >= 1, got {max_messages}")
        self.session_id = session_id
        self.max_age_s = max_age_s
        self.max_messages = max_messages
        self._store = store
        self._idx_key = f"idx::{session_id}"  # legacy counter key, cleanup only
        self._msg_prefix = f"msg::{session_id}::"
        self._last_us = 0
        # Read cache: after the first full load, .messages only fetches
        # rows whose key sorts after the cache frontier (records_after) —
        # an append-only tail read instead of a per-call prefix rescan.
        # The frontier is the last cached key's bare microsecond tick, NOT
        # the full key: another writer can land on the same tick (the
        # _last_us bump is per-instance) with a random suffix that sorts
        # below ours, so each refresh re-reads the frontier tick and
        # dedupes via _cache_keys. The cache is invalidated whenever this
        # instance deletes messages (clear / prune / retention). A writer
        # bypassing this instance to *delete or backdate* rows is not
        # detected until then.
        self._msg_cache: Optional[List[BaseMessage]] = None
        self._cache_keys: set[str] = set()
        self._cache_frontier = ""

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

    @classmethod
    def _decode_message(cls, data: Dict[str, Any]) -> BaseMessage:
        if "message" in data:
            return messages_from_dict([data["message"]])[0]
        # Legacy rows from the pre-lossless schema: {"role", "content",
        # "meta"}. Kept readable so existing Delta tables keep working.
        role = data.get("role", "human")
        msg_cls = cls._TYPE_MAP.get(role)
        meta = data.get("meta", {})
        if msg_cls is not None:
            return msg_cls(content=data["content"], additional_kwargs=meta)
        # Tool/function/custom roles round-trip via ChatMessage instead of
        # being silently rewritten as human messages.
        return ChatMessage(role=role, content=data["content"], additional_kwargs=meta)

    # BaseChatMessageHistory annotates `messages` as a plain attribute, but
    # documents overriding it as a property — which is what this lazy,
    # cache-backed read needs.
    @property
    def messages(self) -> List[BaseMessage]:  # pyright: ignore[reportIncompatibleVariableOverride]
        if self._msg_cache is None:
            pairs = self._store.all_records(prefix=self._msg_prefix)
            self._msg_cache = []
            self._cache_keys = set()
        else:
            pairs = self._store.records_after(self._msg_prefix, self._cache_frontier)
            # The frontier tick is re-read in full; drop what is cached.
            pairs = [kv for kv in pairs if kv[0] not in self._cache_keys]
            if not pairs:
                return list(self._msg_cache)
        if pairs:
            # Sort by the key suffix: zero-padded timestamps (and legacy
            # zero-padded sequence numbers) order lexicographically by time.
            pairs.sort(key=lambda kv: kv[0][len(self._msg_prefix) :])
            self._msg_cache.extend(
                self._decode_message(rec["value"]) for _, rec in pairs
            )
            self._cache_keys.update(key for key, _ in pairs)
            # Frontier = the last key's bare tick (random suffix stripped):
            # it sorts before every key on that tick, so a same-tick append
            # from another writer is still fetched on the next read.
            last_suffix = pairs[-1][0][len(self._msg_prefix) :]
            self._cache_frontier = self._msg_prefix + last_suffix.split("-", 1)[0]
        return list(self._msg_cache)

    def _invalidate_cache(self) -> None:
        self._msg_cache = None
        self._cache_keys = set()
        self._cache_frontier = ""

    def add_message(self, message: BaseMessage) -> None:
        self.add_messages([message])

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
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
                        # Lossless LangChain serialisation: tool_calls,
                        # usage_metadata, response_metadata, name, id all
                        # survive the round-trip.
                        "message": message_to_dict(message),
                        "ts": ts,
                    },
                    "tags": ["message", role, self.session_id],
                }
            )
        self._store.set_many(entries)
        self._apply_retention()

    def clear(self) -> None:
        self._store.clear_prefix(self._msg_prefix)
        self._store.delete(self._idx_key)  # legacy counter, if present
        self._invalidate_cache()

    def has_messages(self) -> bool:
        """True if this thread has at least one stored message.

        Checks only for the presence of a message key — in Delta mode this
        stops at the first matching row instead of collecting every key.
        """
        return self._store.has_prefix(self._msg_prefix)

    def prune_before(self, cutoff_ts: float) -> int:
        """Delete messages whose stored ``ts`` is older than *cutoff_ts*.

        The pipeline's rolling window only limits what is *surfaced* to the
        LLM; this is the complementary storage-side trim (one DELETE in
        Delta mode).  Returns the number removed.
        """
        stale = [
            key
            for key, rec in self._store.all_records(prefix=self._msg_prefix)
            if rec["value"].get("ts", 0.0) < cutoff_ts
        ]
        removed = self._store.delete_many(stale)
        if removed:
            self._invalidate_cache()
        # DEBUG when nothing was pruned: age-based retention calls this
        # after every write, and a no-op should not spam INFO.
        logger.log(
            logging.INFO if removed else logging.DEBUG,
            f"prune_before: session '{self.session_id}' removed {removed} message(s)",
        )
        return removed

    def prune_to_count(self, max_messages: int) -> int:
        """Delete the oldest messages beyond the newest *max_messages*.

        Returns the number removed.
        """
        pairs = self._store.all_records(prefix=self._msg_prefix)
        excess = len(pairs) - max_messages
        if excess <= 0:
            return 0
        pairs.sort(key=lambda kv: kv[0][len(self._msg_prefix) :])
        removed = self._store.delete_many([key for key, _ in pairs[:excess]])
        if removed:
            self._invalidate_cache()
        logger.info(
            f"prune_to_count: session '{self.session_id}' removed {removed} "
            f"message(s) beyond the newest {max_messages}"
        )
        return removed

    def truncate_to(self, keep: int) -> int:
        """Delete the *newest* messages, keeping only the oldest *keep*.

        The tail-side counterpart to :meth:`prune_to_count` (which drops the
        oldest): a rewind primitive. Keys embed a time-ordered tick, so the
        newest rows are simply the last ones in key order. Used to roll a
        turn back — dropping the just-appended (human, AI) pair so it can be
        re-generated without leaving a duplicate — and to rewind a thread to
        an item boundary on resume. Returns the number removed.
        """
        if keep < 0:
            raise ValueError(f"keep must be >= 0, got {keep}")
        pairs = self._store.all_records(prefix=self._msg_prefix)
        if len(pairs) <= keep:
            return 0
        pairs.sort(key=lambda kv: kv[0][len(self._msg_prefix) :])
        removed = self._store.delete_many([key for key, _ in pairs[keep:]])
        if removed:
            self._invalidate_cache()
        logger.info(
            f"truncate_to: session '{self.session_id}' removed {removed} "
            f"message(s), keeping the oldest {keep}"
        )
        return removed

    def _apply_retention(self) -> None:
        if self.max_age_s is not None:
            self.prune_before(_now() - self.max_age_s)
        if self.max_messages is not None:
            self.prune_to_count(self.max_messages)

    # ---- Helpers -----------------------------------------------------------

    def get_session_metadata(self) -> Dict[str, Any]:
        pairs = self._store.all_records(prefix=self._msg_prefix)
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if not pairs:
            return {
                "session_id": self.session_id,
                "message_count": 0,
                "first_message": None,
                "last_message": None,
                "total_usage": usage,
            }
        timestamps = [rec["value"]["ts"] for _, rec in pairs]
        for _, rec in pairs:
            # usage_metadata is persisted by the lossless message payload;
            # legacy rows (no "message" envelope) simply contribute nothing.
            meta = rec["value"].get("message", {}).get("data", {})
            for field, count in (meta.get("usage_metadata") or {}).items():
                if field in usage and isinstance(count, int):
                    usage[field] += count
        return {
            "session_id": self.session_id,
            "message_count": len(pairs),
            "first_message": _iso(min(timestamps)),
            "last_message": _iso(max(timestamps)),
            "total_usage": usage,
        }


# ===========================================================================
# 2. ThreadMemoryManager
# ===========================================================================


class ThreadMemoryManager:
    """
    Manages independent conversation threads, all sharing one KVStore.

    Usage
    -----
    mgr = ThreadMemoryManager(store)
    t1  = mgr.get_thread("user-123-support")
    t1.add_user_message("My order is late.")

    ``max_age_s`` / ``max_messages`` are forwarded to every
    :class:`KVChatMessageHistory` this manager creates (storage-side
    retention, applied after each write).
    """

    def __init__(
        self,
        store: KVStore,
        *,
        max_age_s: float | None = None,
        max_messages: int | None = None,
    ) -> None:
        self._store = store
        self._max_age_s = max_age_s
        self._max_messages = max_messages
        self._threads: Dict[str, KVChatMessageHistory] = {}

    def get_thread(self, thread_id: str) -> KVChatMessageHistory:
        if thread_id not in self._threads:
            self._threads[thread_id] = KVChatMessageHistory(
                thread_id,
                self._store,
                max_age_s=self._max_age_s,
                max_messages=self._max_messages,
            )
        return self._threads[thread_id]

    def fork_thread(
        self,
        src_thread_id: str,
        dst_thread_id: str,
        *,
        upto_messages: int | None = None,
        upto_ts: float | None = None,
    ) -> int:
        """Copy *src*'s messages (oldest first) into the empty thread *dst*.

        The KV-native half of "time travel": rewind a conversation to a
        point — the first ``upto_messages`` messages, and/or those with
        ``ts`` strictly before ``upto_ts`` — and continue it under a new
        thread id, leaving the original untouched. Rows are copied with
        their key suffixes, timestamps, and payloads intact (one batched
        write), so ordering and history survive exactly; only the
        session-id tag is rewritten. Returns the number of messages
        copied.

        Raises ``ValueError`` if *dst* already has messages or equals
        *src*.
        """
        if src_thread_id == dst_thread_id:
            raise ValueError(f"cannot fork thread '{src_thread_id}' onto itself")
        src_prefix = f"msg::{src_thread_id}::"
        dst_prefix = f"msg::{dst_thread_id}::"
        if self._store.has_prefix(dst_prefix):
            raise ValueError(
                f"destination thread '{dst_thread_id}' is not empty"
            )
        records = self._store.all_records(prefix=src_prefix)
        records.sort(key=lambda kv: kv[0][len(src_prefix) :])
        if upto_ts is not None:
            records = [
                (key, rec)
                for key, rec in records
                if rec["value"].get("ts", 0.0) < upto_ts
            ]
        if upto_messages is not None:
            records = records[:upto_messages]
        entries = [
            {
                "key": dst_prefix + key[len(src_prefix) :],
                "value": rec["value"],
                "tags": [
                    dst_thread_id if tag == src_thread_id else tag
                    for tag in rec["tags"]
                ],
                "source": rec["source"],
                "created_at": rec["created_at"],
                "updated_at": rec["updated_at"],
            }
            for key, rec in records
        ]
        self._store.set_many(entries)
        logger.info(
            f"fork_thread: copied {len(entries)} message(s) "
            f"'{src_thread_id}' -> '{dst_thread_id}'"
        )
        return len(entries)

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

    def invalidate_caches(self) -> None:
        """Drop every managed thread's read cache — required after an
        out-of-band store rewrite such as :meth:`KVStore.restore`."""
        for thread in self._threads.values():
            thread._invalidate_cache()

    def thread_summary(self, thread_id: str) -> Dict[str, Any]:
        return self.get_thread(thread_id).get_session_metadata()

    def all_summaries(self) -> List[Dict[str, Any]]:
        return [self.thread_summary(tid) for tid in self.list_threads()]


# ===========================================================================
# 3. KVMemoryStore  (tagged KV + text ingestion + search)
# ===========================================================================


class KVMemoryStore:
    """
    High-level namespaced store for config, summaries, file chunks, etc.,
    backed by KVStore.

    Usage
    -----
    kv = KVMemoryStore(store, namespace="kv")
    kv.set("goal", "RAG pipeline", tags=["project"])
    kv.ingest_text("Long document text …", name="spec", chunk_size=500)
    kv.search("pipeline")

    Parameters
    ----------
    ranker : Any | None
        Optional :class:`memory.relevance.HybridRanker` (duck-typed — this
        module never imports relevance, keeping plain KV usage free of the
        bm25s/faiss dependencies). When set, :meth:`search` ranks entries
        with the BM25 + optional dense + RRF stack instead of the naive
        substring scan.
    """

    def __init__(
        self,
        store: KVStore,
        namespace: str = "kv",
        ranker: Any | None = None,
    ) -> None:
        self._store = store
        self._ns = namespace + "::"
        self._ranker = ranker

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
        return self.items_with_prefix("")

    def items_with_prefix(self, prefix: str) -> List[Dict[str, Any]]:
        """:meth:`all_items` restricted to short keys starting with *prefix*.

        The filter is pushed down to the backend as a full-key prefix
        (namespace + *prefix*), so Delta mode reads only the matching rows
        instead of collecting the whole namespace and filtering client-side.
        """
        return [
            {
                "key": k[len(self._ns) :],
                "value": rec["value"],
                "tags": rec["tags"],
                "source": rec["source"],
                "updated_at": _iso(rec["updated_at"]),
            }
            for k, rec in self._store.all_records(prefix=self._ns + prefix)
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
        """Rank entries against *query*, best first, as (short_key, score).

        With a ``ranker`` injected this is hybrid retrieval (BM25 +
        optional dense, RRF-fused, optional reranker) over each entry's
        key, JSON value, and tags; scores are the 1/rank of the fused
        order. No-signal queries return ``[]``. Without a ranker it falls
        back to the store's substring scan.
        """
        if self._ranker is None:
            raw = self._store.search(query, prefix=self._ns, top_k=top_k)
            return [(k[len(self._ns) :], score) for k, score in raw]
        records = self._store.all_records(prefix=self._ns)
        if not records:
            return []
        short_keys = [k[len(self._ns) :] for k, _ in records]
        docs = [
            f"{short_key}\n{_jdump(rec['value'])}\n{' '.join(rec['tags'])}"
            for short_key, (_, rec) in zip(short_keys, records)
        ]
        candidates = list(range(len(docs)))
        rankings = []
        bm25 = self._ranker.bm25_ranking(docs, candidates, query)
        if bm25 is not None:
            rankings.append(bm25)
        if self._ranker.has_dense:
            dense = self._ranker.dense_ranking(docs, candidates, query)
            if dense is not None:
                rankings.append(dense)
        if not rankings:
            logger.debug(f"search: no scorer has signal for {query!r}")
            return []
        fused = self._ranker.rrf_fuse(rankings)
        if self._ranker.has_reranker:
            fused = self._ranker.rerank(
                fused, docs, query, window_size=max(4 * top_k, top_k)
            )
        return [
            (short_keys[i], 1.0 / (rank + 1))
            for rank, i in enumerate(fused[:top_k])
        ]

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
# 4. MemoryHub  —  unified façade
# ===========================================================================


class MemoryHub:
    """
    Unified façade over all memory layers, all sharing one KVStore.

    Parameters
    ----------
    spark : SparkSession | None
        Active Databricks / local Spark session.  Required only when
        ``table`` is given; in-memory mode never touches Spark.
    table : str | None
        Fully-qualified Delta table name, e.g. ``"main.ml.langchain_memory"``.
        Pass ``None`` (default) for local in-memory operation.
    ranker : Any | None
        Optional ``memory.relevance.HybridRanker`` forwarded to the
        :class:`KVMemoryStore`, upgrading ``mem.kv.search`` from a
        substring scan to hybrid BM25 (+ optional dense) retrieval.
    retention_max_age_s, retention_max_messages : float | int | None
        Storage-side retention forwarded to every chat thread: after each
        write, messages older than ``retention_max_age_s`` seconds are
        pruned, then the oldest beyond ``retention_max_messages`` are.
        Both ``None`` (default) keeps every message forever.

    Usage — Databricks
    ------------------
        mem = MemoryHub(
            spark=spark,
            table="main.langchain.memory",
        )
        thread = mem.get_thread("user-42")
        thread.add_user_message("Hello!")

        # Chat-history threads back a LangGraph model node (see
        # chunker.pipeline): the node reads mem.get_thread(thread_id).messages
        # before each LLM call and persists the new turn via add_messages.

        mem.kv.set("goal", "Build a RAG pipeline", tags=["project"])

    Usage — local (no Spark required)
    ---------------------------------
        mem = MemoryHub()
    """

    def __init__(
        self,
        spark: "SparkSession | None" = None,
        table: Optional[str] = None,
        ranker: Any | None = None,
        retention_max_age_s: float | None = None,
        retention_max_messages: int | None = None,
    ) -> None:
        self._store = KVStore(spark=spark, table=table)
        self.threads = ThreadMemoryManager(
            store=self._store,
            max_age_s=retention_max_age_s,
            max_messages=retention_max_messages,
        )
        self.kv = KVMemoryStore(store=self._store, namespace="kv", ranker=ranker)

    def get_thread(self, thread_id: str) -> KVChatMessageHistory:
        return self.threads.get_thread(thread_id)

    def fork_thread(
        self,
        src_thread_id: str,
        dst_thread_id: str,
        *,
        upto_messages: int | None = None,
        upto_ts: float | None = None,
    ) -> int:
        """Copy *src*'s messages into the empty thread *dst* — see
        :meth:`ThreadMemoryManager.fork_thread`."""
        return self.threads.fork_thread(
            src_thread_id,
            dst_thread_id,
            upto_messages=upto_messages,
            upto_ts=upto_ts,
        )

    def snapshot(self) -> Dict[str, Any]:
        """Dump entire store to a plain dict (JSON-serialisable)."""
        return self._store.snapshot()

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Replace store contents from a snapshot dict."""
        self._store.restore(snapshot)
        # The rewrite happened underneath any threads already handed out;
        # their append-only read caches no longer describe the store.
        self.threads.invalidate_caches()

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
