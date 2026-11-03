"""LangChain chat-history and KV memory. See memory/README.md.

Persists to a family of typed Databricks Delta tables in production (see
memory.schema); runs on a plain Python dict locally, with no Spark (or JVM)
required at all in that mode.

Logger name: ``memory.store``.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import time
import uuid
from bisect import bisect_left, bisect_right, insort
from dataclasses import dataclass
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

from . import _delta, schema
from .chunk_history import ChunkHistory

# PySpark is optional — required only for Delta mode; the in-memory backend
# (table=None) must import and run without it. The type checker imports these
# unconditionally so uses below aren't "possibly unbound"; _PYSPARK_AVAILABLE
# is what actually gates them at runtime.
if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        BooleanType,
        DoubleType,
        LongType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

try:
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        BooleanType,
        DoubleType,
        LongType,
        StringType,
        StructField,
        StructType,
        TimestampType,
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



# The identifier rules live with the table layout; re-exported under the old
# private name for memory.operations and existing callers.
_quoted_table_name = schema.quoted_table_name


@dataclass(frozen=True)
class CDFSyncResult:
    """Outcome of one durable Change Data Feed consumption pass.

    ``baseline`` means the caller must discard all local caches: the consumer
    has just established its initial snapshot/version boundary and CDF cannot
    describe changes written before that boundary.
    """

    baseline: bool = False
    events: tuple[dict[str, Any], ...] = ()
    checkpoint_version: int | None = None



# ---------------------------------------------------------------------------
# Delta staging: Spark schemas and the typed columns derived on write
# ---------------------------------------------------------------------------

# SQL type in memory.schema -> DataType.simpleString() Spark reports back.
_SQL_SIMPLE = {
    "STRING": "string",
    "DOUBLE": "double",
    "BIGINT": "bigint",
    "BOOLEAN": "boolean",
}

_SPARK_SCHEMAS: Dict[str, Any] = {}
_AUDIT_SCHEMA = None


def _spark_schema(spec: "schema.TableSpec"):
    """The staging StructType for *spec* (built lazily; Delta mode only)."""
    cached = _SPARK_SCHEMAS.get(spec.suffix)
    if cached is None:
        types = {
            "STRING": StringType,
            "DOUBLE": DoubleType,
            "BIGINT": LongType,
            "BOOLEAN": BooleanType,
        }
        cached = StructType(
            [
                StructField(name, types[sql_type](), nullable=name not in spec.not_null)
                for name, sql_type in spec.columns
            ]
        )
        _SPARK_SCHEMAS[spec.suffix] = cached
    return cached


def _audit_schema():
    """Schema for the independent, append-only CDF audit table."""
    global _AUDIT_SCHEMA
    if _AUDIT_SCHEMA is None:
        _AUDIT_SCHEMA = StructType(
            [
                StructField("event_id", StringType(), nullable=False),
                StructField("consumer_id", StringType(), nullable=False),
                StructField("source_table", StringType(), nullable=False),
                StructField("kv_key", StringType(), nullable=True),
                StructField("change_type", StringType(), nullable=False),
                StructField("commit_version", LongType(), nullable=False),
                StructField("commit_timestamp", TimestampType(), nullable=True),
                StructField("value", StringType(), nullable=True),
                StructField("tags", StringType(), nullable=True),
                StructField("recorded_at", DoubleType(), nullable=False),
            ]
        )
    return _AUDIT_SCHEMA


def _safe_json(value_json: str) -> Any:
    try:
        return _jload(value_json)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _thread_and_tail(key: str, prefix: str) -> Tuple[Optional[str], Optional[str]]:
    """``prefix{thread}::{tail}`` → (thread, tail), splitting at the last ``::``."""
    rest = key[len(prefix) :]
    if "::" not in rest:
        return None, rest or None
    thread, tail = rest.rsplit("::", 1)
    return thread, tail


def _thread_item(key: str, prefix: str) -> Tuple[Optional[str], Optional[str]]:
    """``prefix{thread}::item::{item}`` → (thread, item)."""
    rest = key[len(prefix) :]
    if "::item::" not in rest:
        return None, None
    thread, item = rest.rsplit("::item::", 1)
    return thread, item


def message_text(payload: Any) -> Optional[str]:
    """Plain text of a stored message payload (lossless or legacy shape)."""
    if not isinstance(payload, dict):
        return None
    message = payload.get("message")
    if isinstance(message, dict):
        data = message.get("data")
        content = data.get("content") if isinstance(data, dict) else None
    else:
        content = payload.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return None


def _derive_message(key: str, value_json: str) -> Tuple[Any, ...]:
    thread, tail = _thread_and_tail(key, "msg::")
    tick = _as_int(tail.split("-", 1)[0]) if tail else None
    payload = _safe_json(value_json)
    role = None
    if isinstance(payload, dict):
        message = payload.get("message")
        if isinstance(message, dict):
            data = message.get("data")
            role = (data.get("role") if isinstance(data, dict) else None) or message.get("type")
        else:
            role = payload.get("role")
    chat = payload.get("chat_id") if isinstance(payload, dict) else None
    return thread, tick, _as_str(role), message_text(payload), _as_str(chat)


def _derive_chat(key: str, value_json: str) -> Tuple[Any, ...]:
    thread, chat = _thread_and_tail(key, "chat::")
    v = _safe_json(value_json)
    v = v if isinstance(v, dict) else {}
    return (
        thread,
        chat,
        _as_str(v.get("label")),
        _as_float(v.get("started_at")),
    )


def _derive_run_item(key: str, value_json: str) -> Tuple[Any, ...]:
    thread, item = _thread_item(key, "kv::run::")
    v = _safe_json(value_json)
    v = v if isinstance(v, dict) else {}
    return (
        thread,
        item,
        _as_str(v.get("status")),
        _as_int(v.get("index")),
        _as_int(v.get("total")),
        _as_int(v.get("attempts")),
        _as_float(v.get("elapsed_s")),
    )


def _derive_validation(key: str, value_json: str) -> Tuple[Any, ...]:
    thread, item = _thread_item(key, "kv::validation::")
    v = _safe_json(value_json)
    v = v if isinstance(v, dict) else {}
    passed = v.get("passed")
    return (
        thread,
        item,
        _as_float(v.get("score")),
        passed if isinstance(passed, bool) else None,
        _as_int(v.get("index")),
    )


def _derive_kv(key: str, value_json: str) -> Tuple[Any, ...]:
    """``(namespace, scope_id)`` — the B_kv clustering columns.

    * ``kv::<ns>::<scope>…`` for a namespace this codebase writes
      (:data:`memory.schema.KV_NAMESPACES`) → ``(ns, scope)``: task, thread
      or candidate id;
    * any other ``kv::<key>`` is a plain KVMemoryStore entry → ``("kv",
      <key>)``, the whole user key (which may itself contain ``::``);
    * a top-level ``<ns>::<scope>…`` key (legacy ``idx::``) → ``(ns, scope)``.
    """
    parts = key.split("::")
    if parts[0] == "kv" and len(parts) > 1:
        if len(parts) >= 3 and parts[1] in schema.KV_NAMESPACES:
            return parts[1], parts[2]
        return "kv", key[len("kv::") :]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None, key


_DERIVE = {
    schema.MESSAGES.suffix: _derive_message,
    schema.CHATS.suffix: _derive_chat,
    schema.RUN_ITEMS.suffix: _derive_run_item,
    schema.VALIDATIONS.suffix: _derive_validation,
    schema.KV.suffix: _derive_kv,
}

_THREAD_PREFIX = {schema.MESSAGES.suffix: "msg::", schema.CHATS.suffix: "chat::"}


def _thread_filter_prefix(spec: "schema.TableSpec", prefix: str) -> Optional[str]:
    """A ``thread_id`` prefix implied by a key prefix, or ``None``.

    The thread id is everything between the namespace and the key's *last*
    ``::``, so a key prefix ``msg::A::…`` only pins the thread to "starts
    with A" (thread ``A`` itself, or ``A::x``); a prefix without a ``::``
    after the namespace pins nothing.
    """
    ns = _THREAD_PREFIX.get(spec.suffix)
    if ns is None or not prefix.startswith(ns):
        return None
    rest = prefix[len(ns) :]
    if "::" not in rest:
        return None
    return rest.rsplit("::", 1)[0]


# Storage backends. Both speak the same raw record shape, a tuple in
# Delta-schema column order (value/tags kept JSON-serialised). Upsert rows with
# created_at=None get each backend's preservation semantics (keep existing on
# update, default to now on insert).
_RawRecord = Tuple[str, str, Optional[str], Optional[float], float, Optional[str]]


# (spark application id, base, full_text_search) -> (provision report,
# search functions available). Filled by the first _DeltaBackend for a base;
# later ones in the same process skip every DDL and startup check.
_KV_PROVISIONED: Dict[Tuple[str, str, bool], Tuple[Any, bool]] = {}


def forget_provisioning(base: Optional[str] = None) -> None:
    """Drop cached provisioning (all bases, or one), so the next store for it
    re-runs DDL, schema validation and startup checks. For tests and for
    DeltaMemoryMaintenance.provision(force=True)."""
    from . import chunk_history

    for cache in (_KV_PROVISIONED, chunk_history._HISTORY_PROVISIONED):
        for key in [k for k in cache if base is None or k[1] == base]:
            del cache[key]


def provision_databricks(
    spark: Any, base: str, specs: Tuple["schema.TableSpec", ...], *, force: bool = False
) -> "schema.ProvisionReport":
    """Predictive optimization + search indexes for *specs*, once.

    Skipped (``reused=True``) when the group's first table already carries
    the provisioned marker for the current schema version; the marker is set
    only after a run with no failures, so an incomplete setup is retried by
    the next process.
    """
    if not force and schema.is_provisioned(spark, base, specs):
        return schema.ProvisionReport(reused=True)
    report = schema.enable_databricks_features(spark, base, specs)
    if not report.failures:
        schema.mark_provisioned(spark, base, specs)
    return report


class _InMemoryBackend:
    """Plain-dict backend for local / CI use.  No pyspark, no JVM.

    Beside the dict, a sorted key list is maintained so the prefix reads
    (``records_after`` / ``keys`` / ``has_prefix`` / ``clear_prefix``)
    bisect straight to the matching range — keys sharing a prefix are
    contiguous in sorted order — instead of scanning every key per call
    (O(store) per read, O(n²) over an n-message run).
    """

    def __init__(self) -> None:
        self._mem: Dict[str, Dict[str, Any]] = {}
        self._sorted_keys: List[str] = []

    def _prefix_range(self, prefix: str) -> Tuple[int, int]:
        """(lo, hi) index range of keys starting with *prefix* in the sorted list."""
        lo = bisect_left(self._sorted_keys, prefix)
        hi = lo
        n = len(self._sorted_keys)
        while hi < n and self._sorted_keys[hi].startswith(prefix):
            hi += 1
        return lo, hi

    def upsert(self, rows: List[_RawRecord], now: float) -> None:
        for key, value_json, tags_json, created_at, updated_at, source in rows:
            existing = self._mem.get(key)
            if existing is None:
                insort(self._sorted_keys, key)
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

    def append(self, rows: List[_RawRecord], now: float) -> None:
        """Rows the caller guarantees are new; a dict needs no fast path."""
        self.upsert(rows, now)

    def thread_ids(self) -> List[str]:
        lo, hi = self._prefix_range("msg::")
        return sorted(
            {k[len("msg::") :].rsplit("::", 1)[0] for k in self._sorted_keys[lo:hi]}
        )

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
        self._sorted_keys = sorted(fresh)

    def get(self, key: str) -> Optional[str]:
        rec = self._mem.get(key)
        return rec["value"] if rec else None

    def records(self, prefix: str = "") -> List[_RawRecord]:
        return self.records_after(prefix, "")

    def records_after(self, prefix: str, after_key: str) -> List[_RawRecord]:
        lo = bisect_right(self._sorted_keys, after_key)
        if prefix:
            lo = max(lo, bisect_left(self._sorted_keys, prefix))
        out: List[_RawRecord] = []
        for i in range(lo, len(self._sorted_keys)):
            k = self._sorted_keys[i]
            if prefix and not k.startswith(prefix):
                break
            r = self._mem[k]
            out.append(
                (k, r["value"], r["tags"], r["created_at"], r["updated_at"], r["source"])
            )
        return out

    def keys(self, prefix: str = "") -> List[str]:
        if not prefix:
            return list(self._sorted_keys)
        lo, hi = self._prefix_range(prefix)
        return self._sorted_keys[lo:hi]

    def has_prefix(self, prefix: str) -> bool:
        i = bisect_left(self._sorted_keys, prefix)
        return i < len(self._sorted_keys) and self._sorted_keys[i].startswith(prefix)

    def delete_many(self, keys: List[str]) -> int:
        removed = 0
        for k in keys:
            if k in self._mem:
                del self._mem[k]
                del self._sorted_keys[bisect_left(self._sorted_keys, k)]
                removed += 1
        return removed

    def clear_prefix(self, prefix: str) -> int:
        lo, hi = self._prefix_range(prefix)
        targets = self._sorted_keys[lo:hi]
        for k in targets:
            del self._mem[k]
        del self._sorted_keys[lo:hi]
        return len(targets)

    def clear(self) -> None:
        self._mem.clear()
        self._sorted_keys.clear()

    def count(self) -> int:
        return len(self._mem)

    @property
    def label(self) -> str:
        return "in-memory"


class _DeltaBackend:
    """Databricks Delta backend (production) over the typed table family.

    One configured *base* name fans out into the tables of
    :mod:`memory.schema`; every key is routed by namespace (``msg::`` →
    ``B_messages``, ``chat::`` → ``B_chats``, ``kv::run::`` → ``B_run_items``,
    ``kv::validation::`` → ``B_validations``, everything else → ``B_kv``).
    Each routed table stores the same six raw-record columns the
    :class:`KVStore` façade speaks, plus typed columns derived from the key
    and value on write (``thread_id``, ``role``, ``content``, ``status``,
    ``score``, …) — which is what liquid clustering, data skipping, SQL
    analysis, and the full-text search indexes work on.

    Write paths
    -----------
    * :meth:`append` — blind ``INSERT`` for rows known to be new (chat
      messages: their keys are collision-free). Appends never conflict with
      each other in Delta and skip MERGE's join entirely.
    * :meth:`upsert` — one ``MERGE`` per table touched.

    A batch that spans tables is one commit per table, written in a fixed
    order (messages before chats), so an index row can never describe
    messages that did not land.
    """

    def __init__(
        self,
        spark: "SparkSession | None",
        table: str,
        *,
        audit_table: str | None = None,
        max_write_retries: int = 3,
        full_text_search: bool = True,
    ) -> None:
        if not _PYSPARK_AVAILABLE:
            raise RuntimeError("pyspark is not installed. Run: pip install pyspark")
        if spark is None:
            raise ValueError(
                f"Delta mode (table={table!r}) requires an active SparkSession"
            )
        if max_write_retries < 0:
            raise ValueError("max_write_retries must be >= 0")
        _quoted_table_name(table)  # validate the base before deriving names
        self._spark = spark
        self._base = table
        self._table_name = table
        self._names = {spec.suffix: spec.name(table) for spec in schema.KV_TABLES}
        self._quoted = {s: _quoted_table_name(n) for s, n in self._names.items()}
        self._audit_table_name = audit_table
        self._audit_table = _quoted_table_name(audit_table) if audit_table else None
        self._max_write_retries = max_write_retries
        self.provision: schema.ProvisionReport | None = None
        self.search_functions = False
        # DDL, schema validation, startup checks and Databricks provisioning
        # run once per Spark application per base — not on every MemoryHub /
        # KVStore construction (see forget_provisioning()).
        key = (*_delta.application_key(spark, table), full_text_search)
        cached = _KV_PROVISIONED.get(key)
        if cached is None:
            self._ensure_tables(full_text_search)
            _KV_PROVISIONED[key] = (self.provision, self.search_functions)
        else:
            self.provision, self.search_functions = cached

    # ---- DDL ---------------------------------------------------------------

    def _ensure_tables(self, full_text_search: bool) -> None:
        schema.ensure_tables(self._spark, self._base, schema.KV_TABLES)
        self._validate_schema()
        self._warn_on_legacy_table()
        self._warn_on_unfinished_restore()
        if full_text_search:
            self.provision = provision_databricks(self._spark, self._base, schema.KV_TABLES)
            self.search_functions = _delta.probe_search_functions(self._spark)
            if self.provision.failures:
                logger.warning(
                    "Delta memory %s: Databricks features unavailable (%d "
                    "statement(s) failed; first: %s). Searches still return "
                    "the same rows, without index acceleration.",
                    self._base,
                    len(self.provision.failures),
                    self.provision.failures[0],
                )

    def _warn_on_legacy_table(self) -> None:
        """Name the migration when the base is still a pre-split table."""
        try:
            exists = self._spark.catalog.tableExists(self._base)
        except Exception:  # pragma: no cover - catalog dependent
            return
        if not exists:
            return
        names = {f.name for f in self._spark.table(self._table_name_q()).schema}
        if not {"kv_key", "value"} <= names:
            return
        migrated = schema.MIGRATED_PROPERTY in schema.table_properties(
            self._spark, self._names["messages"]
        )
        # One property read; the single-row probe only covers a runtime that
        # refused to set the property during migration.
        if migrated or self._spark.table(self._quoted["messages"]).limit(1).count():
            # Already migrated (or in use): the old table is just history.
            logger.info(
                "Delta memory: legacy single table %s is still present but "
                "unused; drop it once the migration is verified.",
                self._base,
            )
        else:
            logger.error(
                "Delta memory: %s is a legacy single-table memory store. The "
                "store now uses %s_* typed tables and does not read it; copy "
                "it once with DeltaMemoryMaintenance.migrate_legacy(%r).",
                self._base,
                self._base,
                self._base,
            )

    def _table_name_q(self) -> str:
        return _quoted_table_name(self._base)

    def _validate_schema(self) -> None:
        for spec in schema.KV_TABLES:
            actual = {
                f.name: f.dataType.simpleString()
                for f in self._spark.table(self._quoted[spec.suffix]).schema
            }
            expected = {
                name: _SQL_SIMPLE[sql_type] for name, sql_type in spec.columns
            }
            invalid = {
                name: (want, actual.get(name))
                for name, want in expected.items()
                if actual.get(name) != want
            }
            if invalid:
                raise RuntimeError(
                    f"Delta table {self._names[spec.suffix]!r} has an "
                    f"incompatible memory schema: {invalid!r}"
                )

    def _cdf_enabled(self, suffix: str = "messages") -> bool:
        return schema.table_property(
            self._spark, self._names[suffix], "delta.enableChangeDataFeed"
        )

    # ---- routing -----------------------------------------------------------

    @staticmethod
    def _route(key: str) -> "schema.TableSpec":
        for prefix, spec in schema.KV_ROUTES:
            if key.startswith(prefix):
                return spec
        return schema.KV_FALLBACK

    @staticmethod
    def _specs_for_prefix(prefix: str) -> List["schema.TableSpec"]:
        """Tables whose key space can contain a key starting with *prefix*."""
        out: List[schema.TableSpec] = []
        for route, spec in schema.KV_ROUTES:
            if prefix.startswith(route):
                # A routed namespace owns every key under it.
                return [spec]
            if route.startswith(prefix):
                out.append(spec)
        # Keys outside every route live in the fallback table.
        out.append(schema.KV_FALLBACK)
        return out

    def _df(self, spec: "schema.TableSpec", prefix: str = "") -> "DataFrame":
        df = self._spark.table(self._quoted[spec.suffix])
        if prefix:
            df = df.filter(F.col("kv_key").startswith(prefix))
            thread = _thread_filter_prefix(spec, prefix)
            if thread is not None:
                # Redundant with the key filter, but on a clustered column:
                # lets Delta skip every file of every other thread.
                df = df.filter(F.col("thread_id").startswith(thread))
        return df

    # ---- staging -----------------------------------------------------------

    def _stage(
        self, spec: "schema.TableSpec", rows: List[_RawRecord], now: float
    ) -> "DataFrame":
        # created_at=None (not supplied) defaults to now; the MERGE's
        # WHEN MATCHED branch never touches created_at, so an existing
        # row's creation time is preserved on update.
        derive = _DERIVE[spec.suffix]
        staged = [
            (
                key,
                value_json,
                tags_json,
                created_at if created_at is not None else now,
                updated_at,
                source,
                *derive(key, value_json),
            )
            for key, value_json, tags_json, created_at, updated_at, source in rows
        ]
        return self._spark.createDataFrame(staged, schema=_spark_schema(spec))

    def _grouped(self, rows: List[_RawRecord]) -> List[Tuple["schema.TableSpec", List[_RawRecord]]]:
        groups: Dict[str, List[_RawRecord]] = {}
        for row in rows:
            groups.setdefault(self._route(row[0]).suffix, []).append(row)
        # Fixed table order: messages land before the chat rows indexing them.
        return [
            (spec, groups[spec.suffix])
            for spec in schema.KV_TABLES
            if spec.suffix in groups
        ]

    def _with_write_retry(self, operation: str, action: Any) -> None:
        """Retry an *idempotent* Delta write on a concurrency conflict."""
        _delta.with_retry(operation, action, max_retries=self._max_write_retries)

    def _with_view(self, df: "DataFrame", stem: str, action: Any) -> None:
        with _delta.temp_view(self._spark, df, stem) as view:
            action(view)

    # ---- writes ------------------------------------------------------------

    def upsert(self, rows: List[_RawRecord], now: float) -> None:
        for spec, group in self._grouped(rows):
            target = self._quoted[spec.suffix]
            derived = [c for c in spec.column_names if c not in schema.KV_CORE_NAMES]
            extra = "".join(f",\n    target.`{c}` = source.`{c}`" for c in derived)

            def merge(view: str, target: str = target, extra: str = extra) -> None:
                self._with_write_retry(
                    "MERGE",
                    lambda: self._spark.sql(f"""
                        MERGE INTO {target} AS target
                        USING {view} AS source
                          ON target.kv_key = source.kv_key
                        WHEN MATCHED THEN UPDATE SET
                            target.value      = source.value,
                            target.tags       = COALESCE(source.tags, target.tags),
                            target.updated_at = source.updated_at,
                            target.source     = COALESCE(source.source, target.source){extra}
                        WHEN NOT MATCHED THEN INSERT *
                    """),
                )

            self._with_view(self._stage(spec, group, now), "upsert", merge)

    def append(self, rows: List[_RawRecord], now: float) -> None:
        """Write rows the caller guarantees are new.

        Message rows go through a blind ``INSERT``; anything else in the batch
        (a chat index row) still MERGEs, since it may already exist.
        """
        for spec, group in self._grouped(rows):
            if spec.suffix != schema.MESSAGES.suffix:
                self.upsert(group, now)
                continue
            self._insert_messages(spec, group, now)

    def _insert_messages(
        self, spec: "schema.TableSpec", group: List[_RawRecord], now: float
    ) -> None:
        """Blind INSERT, made safe to retry by verifying first.

        A blind INSERT is not idempotent: retrying one whose commit landed
        (the client saw an error anyway) would duplicate every row. A Delta
        commit is atomic, so after any failure the batch's keys are looked up:
        all present means the commit landed and nothing is retried; only the
        absent rows are re-sent, and only on a retryable conflict.
        """
        target = self._quoted[spec.suffix]
        cols = ", ".join(f"`{c}`" for c in spec.column_names)
        pending = list(group)
        for attempt in range(self._max_write_retries + 1):
            try:
                self._with_view(
                    self._stage(spec, pending, now),
                    "append",
                    lambda view: self._spark.sql(
                        f"INSERT INTO {target} ({cols}) SELECT {cols} FROM {view}"
                    ),
                )
                return
            except Exception as exc:
                present = self._present_message_keys([r[0] for r in pending])
                pending = [r for r in pending if r[0] not in present]
                if not pending:
                    logger.info(
                        "Delta INSERT reported %s but its commit landed; not retrying",
                        type(exc).__name__,
                    )
                    return
                if not _delta.is_retryable_conflict(exc) or attempt == self._max_write_retries:
                    raise
                delay = 0.05 * (2**attempt)
                logger.warning(
                    "Delta INSERT conflicted; re-sending %d absent row(s) (%s/%s) in %.2fs",
                    len(pending),
                    attempt + 1,
                    self._max_write_retries,
                    delay,
                )
                time.sleep(delay)

    def _present_message_keys(self, keys: List[str]) -> set:
        threads = sorted({t for t in (_thread_and_tail(k, "msg::")[0] for k in keys) if t})
        df = self._spark.table(self._quoted["messages"])
        if threads:
            df = df.filter(F.col("thread_id").isin(threads))  # clustered: skips files
        return {
            r.kv_key for r in df.filter(F.col("kv_key").isin(list(keys))).select("kv_key").collect()
        }

    def replace_all(self, rows: List[_RawRecord], now: float) -> None:
        """Replace every table's contents with *rows*, all or nothing.

        Delta has no multi-table transaction, so the restore is journaled in
        ``B_restore_journal``:

        * ``started`` — every table's version before anything is touched;
        * ``overwritten`` — after each ``INSERT OVERWRITE`` commits, with the
          version it produced (``post_version``);
        * ``completed`` — once every table is overwritten.

        On a failure the tables this run overwrote are rolled back with
        ``RESTORE TABLE … TO VERSION AS OF`` their pre-restore version — but
        only while a table is still at the run's ``post_version``. A table
        someone else wrote to since is *not* restored (RESTORE would erase
        their commits); it is journaled ``rollback_skipped`` and logged for an
        operator. Tables the run never overwrote are never touched. A process
        that dies mid-restore leaves an open run, which the next open logs and
        ``DeltaMemoryMaintenance.recover_restore()`` settles the same way.
        """
        groups = dict((spec.suffix, group) for spec, group in self._grouped(rows))
        restore_id = uuid.uuid4().hex
        pre = {spec.suffix: self._latest_version(spec.suffix) for spec in schema.KV_TABLES}
        self._journal(
            [
                (restore_id, self._names[suffix], version, "started", now, None)
                for suffix, version in pre.items()
            ]
        )
        post: Dict[str, Optional[int]] = {}
        try:
            for spec in schema.KV_TABLES:
                target = self._quoted[spec.suffix]
                cols = ", ".join(f"`{c}`" for c in spec.column_names)
                self._with_view(
                    self._stage(spec, groups.get(spec.suffix, []), now),
                    "replace",
                    lambda view, target=target, cols=cols: self._spark.sql(
                        f"INSERT OVERWRITE {target} ({cols}) SELECT {cols} FROM {view}"
                    ),
                )
                # Recorded in memory first: even if journaling it fails, this
                # process knows the table was overwritten and can roll it back.
                post[spec.suffix] = None
                post[spec.suffix] = self._latest_version(spec.suffix)
                self._journal(
                    [(restore_id, self._names[spec.suffix], pre[spec.suffix],
                      "overwritten", _now(), post[spec.suffix])]
                )
        except Exception:
            logger.error(
                "Delta memory restore %s failed after %d of %d table(s); rolling back",
                restore_id,
                len(post),
                len(schema.KV_TABLES),
            )
            try:
                self.rollback_restore(restore_id, pre, post)
            except Exception:
                # Keep the original error; the run stays open in the journal,
                # so recover_restore() can settle it later.
                logger.error(
                    "Delta memory restore %s: rollback failed; run "
                    "DeltaMemoryMaintenance.recover_restore()",
                    restore_id,
                    exc_info=True,
                )
            raise
        try:
            _delta.with_retry(
                "journal completed",
                lambda: self._journal([(restore_id, None, None, "completed", _now(), None)]),
                retry_on=lambda exc: True,
                max_retries=2,
            )
        except Exception:
            # The data IS restored. Recovery treats a run whose every table
            # is journaled 'overwritten' as complete, never as one to undo.
            logger.warning(
                "Delta memory restore %s succeeded but its completion could not "
                "be journaled; recovery will mark it complete",
                restore_id,
                exc_info=True,
            )

    def _journal(self, rows: List[Tuple[Any, ...]]) -> None:
        self._ensure_journal()
        cols = ", ".join(f"`{c}`" for c in schema.RESTORE_JOURNAL.column_names)
        self._with_view(
            self._spark.createDataFrame(rows, _spark_schema(schema.RESTORE_JOURNAL)),
            "journal",
            lambda view: self._spark.sql(
                f"INSERT INTO {self._journal_table} ({cols}) SELECT {cols} FROM {view}"
            ),
        )

    def _ensure_journal(self) -> None:
        if getattr(self, "_journal_ready", False):
            return
        schema.ensure_tables(self._spark, self._base, (schema.RESTORE_JOURNAL,))
        self._journal_ready = True

    @property
    def _journal_table(self) -> str:
        return _quoted_table_name(schema.RESTORE_JOURNAL.name(self._base))

    def rollback_restore(
        self,
        restore_id: str,
        pre: Dict[str, int],
        post: Dict[str, Optional[int]],
    ) -> Dict[str, List[str]]:
        """Undo one restore run's overwrites, where that is still safe.

        *pre*: suffix → version before the run. *post*: suffix → version the
        run's overwrite produced, for the tables it overwrote (``None`` when
        unknown). Returns ``{"rolled_back": [...], "needs_manual": [...]}``.
        """
        rolled: List[str] = []
        manual: List[str] = []
        for suffix, post_version in post.items():
            current = self._latest_version(suffix)
            if post_version is not None and current == post_version:
                self._spark.sql(
                    f"RESTORE TABLE {self._quoted[suffix]} TO VERSION AS OF {int(pre[suffix])}"
                )
                rolled.append(suffix)
            else:
                manual.append(suffix)
                logger.error(
                    "Delta memory restore %s: not rolling back %s — it is at version "
                    "%s, the restore left it at %s, and RESTORE to %s would erase the "
                    "commits made since. Decide manually.",
                    restore_id,
                    self._names[suffix],
                    current,
                    post_version,
                    pre[suffix],
                )
        self._journal(
            [
                (restore_id, self._names[s], pre[s], "rollback_skipped", _now(), post[s])
                for s in manual
            ]
            + [(restore_id, None, None, "rolled_back", _now(), None)]
        )
        return {"rolled_back": rolled, "needs_manual": manual}

    def unfinished_restores(self) -> Dict[str, Dict[str, Dict[str, Optional[int]]]]:
        """Open restore runs: ``restore_id`` → ``{"pre": {suffix: version},
        "post": {suffix: version}}`` (``post`` holds the overwritten tables).

        Reads only the open runs' rows (a left anti-join against closed
        runs), and nothing at all when the journal was never created.
        """
        journal_name = schema.RESTORE_JOURNAL.name(self._base)
        if not self._spark.catalog.tableExists(journal_name):
            return {}
        journal = self._spark.table(self._journal_table)
        closed = (
            journal.filter(F.col("state").isin("completed", "rolled_back"))
            .select("restore_id")
            .distinct()
        )
        rows = (
            journal.filter(F.col("state").isin("started", "overwritten"))
            .join(closed, on="restore_id", how="left_anti")
            .select("restore_id", "table_name", "state", "pre_version", "post_version")
            .collect()
        )
        by_name = {name: suffix for suffix, name in self._names.items()}
        runs: Dict[str, Dict[str, Dict[str, Optional[int]]]] = {}
        for r in rows:
            suffix = by_name.get(r.table_name)
            if suffix is None:
                continue
            run = runs.setdefault(r.restore_id, {"pre": {}, "post": {}})
            if r.state == "started":
                run["pre"][suffix] = int(r.pre_version)
            else:
                run["post"][suffix] = None if r.post_version is None else int(r.post_version)
        return runs

    def settle_restore(self, restore_id: str, run: Dict[str, Dict[str, Optional[int]]]) -> Dict[str, Any]:
        """Close one open run: complete if every table was overwritten,
        otherwise roll back what it overwrote (see :meth:`rollback_restore`)."""
        if set(run["post"]) >= {spec.suffix for spec in schema.KV_TABLES}:
            self._journal([(restore_id, None, None, "completed", _now(), None)])
            return {"completed": True, "rolled_back": [], "needs_manual": []}
        pre = {k: int(v) for k, v in run["pre"].items() if v is not None}
        outcome = self.rollback_restore(restore_id, pre, run["post"])
        return {"completed": False, **outcome}

    def prune_restore_journal(self, before_ts: float) -> None:
        """Drop journal rows of *closed* runs older than *before_ts*."""
        if not self._spark.catalog.tableExists(schema.RESTORE_JOURNAL.name(self._base)):
            return
        self._with_write_retry(
            "journal prune",
            lambda: self._spark.sql(
                f"DELETE FROM {self._journal_table} WHERE ts < {float(before_ts)!r} "
                f"AND restore_id IN (SELECT restore_id FROM {self._journal_table} "
                "WHERE state IN ('completed', 'rolled_back'))"
            ),
        )

    def _warn_on_unfinished_restore(self) -> None:
        try:
            unfinished = self.unfinished_restores()
        except Exception as exc:  # pragma: no cover - catalog dependent
            logger.debug(f"unfinished_restores: {exc}")
            return
        if unfinished:
            logger.error(
                "Delta memory %s: %d restore(s) were interrupted (%s); the "
                "tables may be partly overwritten. Settle them with "
                "DeltaMemoryMaintenance.recover_restore().",
                self._base,
                len(unfinished),
                ", ".join(sorted(unfinished)),
            )

    def _delete_keys(self, spec: "schema.TableSpec", keys: List[str]) -> None:
        """Delete exactly *keys* from one table, staging them as data.

        Deletes go through MERGE rather than ``DELETE FROM ... WHERE``: Spark
        parameter markers are not substituted inside Delta's DELETE plan, and
        staging keeps key values as *data*, so a key with quotes or LIKE
        wildcards is just a string.
        """
        target = self._quoted[spec.suffix]
        self._with_view(
            self._spark.createDataFrame([(k,) for k in keys], "kv_key STRING"),
            "delete",
            lambda view: self._with_write_retry(
                "DELETE",
                lambda: self._spark.sql(
                    f"MERGE INTO {target} tgt USING {view} src "
                    f"ON tgt.kv_key = src.kv_key WHEN MATCHED THEN DELETE"
                ),
            ),
        )

    def delete_many(self, keys: List[str]) -> int:
        by_table: Dict[str, List[str]] = {}
        for key in set(keys):
            by_table.setdefault(self._route(key).suffix, []).append(key)
        removed = 0
        for spec in schema.KV_TABLES:
            wanted = by_table.get(spec.suffix)
            if not wanted:
                continue
            existing = [
                row.kv_key
                for row in self._spark.table(self._quoted[spec.suffix])
                .filter(F.col("kv_key").isin(wanted))
                .select("kv_key")
                .collect()
            ]
            if existing:
                self._delete_keys(spec, existing)
            removed += len(existing)
        return removed

    def clear_prefix(self, prefix: str) -> int:
        removed = 0
        for spec in self._specs_for_prefix(prefix):
            keys = [row.kv_key for row in self._df(spec, prefix).select("kv_key").collect()]
            if keys:
                self._delete_keys(spec, keys)
            removed += len(keys)
        return removed

    def clear(self) -> None:
        for spec in schema.KV_TABLES:
            target = self._quoted[spec.suffix]
            self._with_write_retry(
                "CLEAR", lambda target=target: self._spark.sql(f"DELETE FROM {target}")
            )

    # ---- reads -------------------------------------------------------------

    def get(self, key: str) -> Optional[str]:
        spec = self._route(key)
        rows = (
            self._df(spec)
            .filter(F.col("kv_key") == key)
            .select("value")
            .limit(1)
            .collect()
        )
        return rows[0].value if rows else None

    def records(self, prefix: str = "") -> List[_RawRecord]:
        return self.records_after(prefix, "")

    def _collect_records(self, df: "DataFrame") -> List[_RawRecord]:
        return [
            (row.kv_key, row.value, row.tags, row.created_at, row.updated_at, row.source)
            for row in df.select(*schema.KV_CORE_NAMES).collect()
        ]

    def records_after(self, prefix: str, after_key: str) -> List[_RawRecord]:
        out: List[_RawRecord] = []
        for spec in self._specs_for_prefix(prefix):
            df = self._df(spec, prefix)
            if after_key:
                df = df.filter(F.col("kv_key") > after_key)
            out.extend(self._collect_records(df))
        out.sort(key=lambda r: r[0])
        return out

    def search_candidates(self, prefix: str, query: str) -> List[_RawRecord]:
        """Rows whose key, value, or tags contain *query*, case-insensitively.

        The superset :meth:`KVStore.search` scores. With ``search_functions``
        the predicate is ``isearch`` (index-accelerated where a full-text
        search index covers the column); otherwise ``lower(col)`` contains —
        identical rows either way.
        """
        out: List[_RawRecord] = []
        q = query.lower()
        for spec in self._specs_for_prefix(prefix):
            df = self._df(spec, prefix)
            if self.search_functions:
                predicate = F.expr(
                    schema.search_sql(("kv_key", "value", "tags"), query)
                )
            else:
                predicate = (
                    F.lower(F.col("kv_key")).contains(q)
                    | F.lower(F.col("value")).contains(q)
                    | F.lower(F.coalesce(F.col("tags"), F.lit(""))).contains(q)
                )
            out.extend(self._collect_records(df.filter(predicate)))
        return out

    def keys(self, prefix: str = "") -> List[str]:
        out: List[str] = []
        for spec in self._specs_for_prefix(prefix):
            out.extend(row.kv_key for row in self._df(spec, prefix).select("kv_key").collect())
        return sorted(out)

    def has_prefix(self, prefix: str) -> bool:
        # Stops at the first matching row instead of collecting every key.
        return any(
            bool(self._df(spec, prefix).limit(1).count())
            for spec in self._specs_for_prefix(prefix)
        )

    def thread_ids(self) -> List[str]:
        """Distinct thread ids with at least one message — one clustered
        column of one table, rather than every message key."""
        return sorted(
            row.thread_id
            for row in self._spark.table(self._quoted["messages"])
            .select("thread_id")
            .distinct()
            .collect()
            if row.thread_id is not None
        )

    def count(self) -> int:
        return sum(self._df(spec).count() for spec in schema.KV_TABLES)

    def rederive(self, batch_rows: int = 5000) -> Dict[str, int]:
        """Recompute every row's derived columns from its key and value.

        For rows written before a derivation rule changed (e.g. B_kv's
        namespace/scope_id). Each batch is a MERGE of the rows' own values
        and timestamps, so nothing but the derived columns changes.
        """
        counts: Dict[str, int] = {}
        for spec in schema.KV_TABLES:
            batch: List[_RawRecord] = []
            seen = 0
            rows = (
                self._spark.table(self._quoted[spec.suffix])
                .select(*schema.KV_CORE_NAMES)
                .toLocalIterator()
            )
            for row in rows:
                batch.append(
                    (row.kv_key, row.value, row.tags, row.created_at,
                     row.updated_at if row.updated_at is not None else _now(), row.source)
                )
                if len(batch) >= batch_rows:
                    self.upsert(batch, _now())
                    seen += len(batch)
                    batch = []
            if batch:
                self.upsert(batch, _now())
                seen += len(batch)
            counts[spec.suffix] = seen
        return counts

    def _thread_messages(self, thread_id: str) -> "DataFrame":
        # Exact thread on the clustered column, plus the key prefix (a thread
        # id cannot be told apart from a longer one by thread_id alone when
        # ids contain '::').
        return (
            self._spark.table(self._quoted["messages"])
            .filter(F.col("thread_id") == thread_id)
            .filter(F.col("kv_key").startswith(f"msg::{thread_id}::"))
        )

    def chat_stats(self, thread_id: str) -> Dict[str, Dict[str, Any]]:
        """One aggregate over the thread's messages, grouped by ``chat_id``."""
        rows = (
            self._thread_messages(thread_id)
            .filter(F.col("chat_id").isNotNull())
            .groupBy("chat_id")
            .agg(
                F.min("kv_key").alias("first_key"),
                F.max("kv_key").alias("last_key"),
                F.count(F.lit(1)).alias("message_count"),
            )
            .collect()
        )
        return {
            r.chat_id: {
                "first_key": r.first_key,
                "last_key": r.last_key,
                "message_count": int(r.message_count),
            }
            for r in rows
        }

    def chat_records(self, thread_id: str, chat_id: str) -> List[_RawRecord]:
        return self._collect_records(
            self._thread_messages(thread_id).filter(F.col("chat_id") == chat_id)
        )

    # ---- CDF ---------------------------------------------------------------

    def _latest_version(self, suffix: str) -> int:
        return _delta.latest_version(self._spark, self._quoted[suffix])

    def _ensure_audit_table(self) -> str:
        if self._audit_table is None:
            raise RuntimeError(
                "CDF synchronization needs audit_table= so checkpoints and "
                "the durable audit trail are outside the source memory tables"
            )
        self._spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self._audit_table} (
                event_id         STRING NOT NULL,
                consumer_id      STRING NOT NULL,
                source_table     STRING NOT NULL,
                kv_key           STRING,
                change_type      STRING NOT NULL,
                commit_version   BIGINT NOT NULL,
                commit_timestamp TIMESTAMP,
                value            STRING,
                tags             STRING,
                recorded_at      DOUBLE NOT NULL
            )
            USING DELTA
            CLUSTER BY (consumer_id, source_table)
            TBLPROPERTIES ('delta.enableRowTracking' = 'true')
        """)
        return self._audit_table

    @staticmethod
    def _audit_event_id(
        consumer_id: str, source_table: str, version: int, key: str, change_type: str
    ) -> str:
        # A source change is global and must be audited exactly once even
        # when many cache consumers read it. Checkpoints are per consumer.
        material = (
            f"{consumer_id}\0{source_table}\0{version}\0{key}\0{change_type}"
            if change_type == "checkpoint"
            else f"{source_table}\0{version}\0{key}\0{change_type}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _audit_upsert(self, rows: list[tuple[Any, ...]]) -> None:
        """Idempotently persist events and the consumer checkpoints together."""
        audit = self._ensure_audit_table()
        self._with_view(
            self._spark.createDataFrame(rows, schema=_audit_schema()),
            "cdf_audit",
            lambda view: self._with_write_retry(
                "CDF audit MERGE",
                lambda: self._spark.sql(f"""
                    MERGE INTO {audit} AS target
                    USING {view} AS source
                      ON target.event_id = source.event_id
                    WHEN MATCHED THEN UPDATE SET
                        target.commit_version = source.commit_version,
                        target.commit_timestamp = source.commit_timestamp,
                        target.recorded_at = source.recorded_at
                    WHEN NOT MATCHED THEN INSERT *
                """),
            ),
        )

    def sync_cdf(self, consumer_id: str) -> CDFSyncResult:
        """Read committed changes since *consumer_id*'s durable checkpoints.

        Only the CDF-enabled tables (messages and chats — the ones local
        thread caches mirror) are read, each against its own checkpoint. The
        first invocation establishes a snapshot boundary for every table
        rather than pretending historical CDF exists; it is reported as a
        baseline so the caller drops all caches.
        """
        if not isinstance(consumer_id, str) or not consumer_id.strip():
            raise ValueError("CDF consumer_id must be a non-empty string")
        audit = self._ensure_audit_table()
        now = _now()
        baseline = False
        events: list[dict[str, Any]] = []
        audit_rows: list[tuple[Any, ...]] = []
        versions: list[int] = []
        cdf_specs = [spec for spec in schema.KV_TABLES if spec.cdf]
        checkpoint_keys = {
            spec.suffix: self._audit_event_id(
                consumer_id, self._names[spec.suffix], -1, "__checkpoint__", "checkpoint"
            )
            for spec in cdf_specs
        }
        # This runs before every LLM call of a CDF-enabled pipeline: read all
        # checkpoints in one query rather than one per table.
        stored = {
            row.event_id: int(row.commit_version)
            for row in self._spark.table(audit)
            .filter(F.col("event_id").isin(list(checkpoint_keys.values())))
            .select("event_id", "commit_version")
            .collect()
        }
        for spec in cdf_specs:
            name = self._names[spec.suffix]
            checkpoint_key = checkpoint_keys[spec.suffix]
            latest = self._latest_version(spec.suffix)
            versions.append(latest)
            checkpoint_row = (
                checkpoint_key, consumer_id, name, None,
                "checkpoint", latest, None, None, None, now,
            )
            if checkpoint_key not in stored:
                baseline = True
                audit_rows.append(checkpoint_row)
                continue
            checkpoint = stored[checkpoint_key]
            if checkpoint >= latest:
                continue
            changes = (
                self._spark.read.format("delta")
                .option("readChangeFeed", "true")
                .option("startingVersion", checkpoint + 1)
                .option("endingVersion", latest)
                .table(self._quoted[spec.suffix])
                .select(
                    "kv_key", "value", "tags",
                    "_change_type", "_commit_version", "_commit_timestamp",
                )
                .collect()
            )
            for row in changes:
                data = row.asDict(recursive=True)
                key = str(data["kv_key"])
                change_type = str(data["_change_type"])
                version = int(data["_commit_version"])
                events.append(
                    {
                        "key": key,
                        "table": name,
                        "change_type": change_type,
                        "commit_version": version,
                        "commit_timestamp": data.get("_commit_timestamp"),
                    }
                )
                audit_rows.append(
                    (
                        self._audit_event_id(consumer_id, name, version, key, change_type),
                        consumer_id, name, key, change_type, version,
                        data.get("_commit_timestamp"), data.get("value"),
                        data.get("tags"), now,
                    )
                )
            audit_rows.append(checkpoint_row)
        if audit_rows:
            self._audit_upsert(audit_rows)
        checkpoint_version = max(versions) if versions else None
        if baseline:
            return CDFSyncResult(baseline=True, checkpoint_version=checkpoint_version)
        return CDFSyncResult(events=tuple(events), checkpoint_version=checkpoint_version)

    def operations(self) -> Dict[str, Any]:
        """Lightweight per-table diagnostics used by the maintenance runbook."""
        tables: Dict[str, Any] = {
            self._names[spec.suffix]: _delta.table_status(self._spark, self._quoted[spec.suffix])
            for spec in schema.KV_TABLES
        }
        return {
            "base_table": self._base,
            "tables": tables,
            "audit_table": self._audit_table_name,
            "search_functions": self.search_functions,
            "search_indexes": list(self.provision.search_indexes) if self.provision else [],
        }

    @property
    def label(self) -> str:
        return f"Delta:{self._base}_*"


# ===========================================================================
# CORE: KVStore
# ===========================================================================


class KVStore:
    """
    Key-value store, backed by one of two interchangeable backends.

    Two modes
    ---------
    - **Delta mode** (production):
      Pass ``spark`` and ``table="catalog.schema.base_name"``. The typed
      tables of :mod:`memory.schema` (``base_name_messages``, ``_chats``,
      ``_run_items``, ``_validations``, ``_kv``) are created if absent and
      every key is routed to one of them by namespace. Messages are
      appended with ``INSERT``, other upserts use ``MERGE INTO``.

    - **In-memory mode** (local / CI):
      Pass ``table=None`` (default).  All state lives in a plain Python
      dict — pyspark is not required, no Spark session is touched.

    All values are JSON-serialised so any Python type (dict, list, str, …)
    can be stored.  Tags are stored as a JSON array string.

    Key layout
    ----------
    Keys follow a namespace::subkey convention; the namespace decides which
    Delta table a key lives in:

        msg::thread-42::00000003   → chat message seq 3 of thread-42
        kv::project_goal           → a KVMemoryStore entry
        idx::thread-42             → next sequence counter for a thread

    Parameters
    ----------
    spark : SparkSession | None
        Active Spark / Databricks session.  Required only when ``table``
        is given; ignored in in-memory mode.
    table : str | None
        Delta base table name for persistence, or None for pure in-memory
        operation.
    full_text_search : bool
        Delta mode on Databricks: predictive optimization, full-text search
        indexes, and ``isearch()`` pushdown for :meth:`search`.
    """

    def __init__(
        self,
        spark: "SparkSession | None" = None,
        table: Optional[str] = None,
        *,
        audit_table: Optional[str] = None,
        max_write_retries: int = 3,
        full_text_search: bool = True,
    ) -> None:
        self._table = table
        if audit_table is not None and table is None:
            raise ValueError("audit_table requires a Delta table")
        self._backend: Any = (
            _DeltaBackend(
                spark,
                table,
                audit_table=audit_table,
                max_write_retries=max_write_retries,
                full_text_search=full_text_search,
            )
            if table
            else _InMemoryBackend()
        )

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
        """Upsert several entries at once — one MERGE per table touched.

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
        self._backend.upsert(self._raw_rows(entries, now), now)

    def append_many(self, entries: List[Dict[str, Any]]) -> None:
        """Write entries whose keys the caller guarantees are new.

        Same entry shape as :meth:`set_many`. Delta mode writes ``msg::`` rows
        with a blind ``INSERT`` (no MERGE join, no conflict with concurrent
        appends); any other key in the batch is still upserted.
        """
        if not entries:
            return
        now = _now()
        self._backend.append(self._raw_rows(entries, now), now)

    @staticmethod
    def _raw_rows(entries: List[Dict[str, Any]], now: float) -> List[_RawRecord]:
        return [
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

    def get(self, key: str, default: Any = None) -> Any:
        raw = self._backend.get(key)
        return _jload(raw) if raw is not None else default

    def delete(self, key: str) -> bool:
        return self._backend.delete_many([key]) == 1

    def delete_many(self, keys: List[str]) -> int:
        """Delete several keys — one staged MERGE … DELETE per table.

        Returns the number of keys that existed and were removed.
        """
        if not keys:
            return 0
        return self._backend.delete_many(keys)

    def keys(self, prefix: str = "") -> List[str]:
        return self._backend.keys(prefix)

    def thread_ids(self) -> List[str]:
        """Distinct thread ids that have at least one stored message."""
        return self._backend.thread_ids()

    def chat_stats(self, thread_id: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """Per-chat first/last key and message count of *thread_id*,
        aggregated by the backend, or ``None`` when it has no pushdown (the
        caller then aggregates the decoded records itself)."""
        stats = getattr(self._backend, "chat_stats", None)
        return stats(thread_id) if stats is not None else None

    def chat_records(self, thread_id: str, chat_id: str) -> Optional[List[Tuple[str, Dict]]]:
        """*thread_id*'s messages in *chat_id*, filtered by the backend, or
        ``None`` when it has no pushdown."""
        records = getattr(self._backend, "chat_records", None)
        if records is None:
            return None
        return [self._decode_record(r) for r in records(thread_id, chat_id)]

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

        Scoring is a case-insensitive substring count. In Delta mode the
        candidate rows are filtered in SQL first — ``isearch()`` on
        Databricks, which full-text search indexes accelerate, or a
        ``lower(...)`` contains scan elsewhere — so only matching rows ever
        leave the cluster.
        """
        q = query.lower()
        hits: List[Tuple[str, float]] = []
        candidates = getattr(self._backend, "search_candidates", None)
        rows = candidates(prefix, query) if candidates is not None else self._backend.records(prefix)
        for key, value_json, tags_json, _, _, _ in rows:
            score = 0.0
            if q in key.lower():
                score += 2.0
            value_lower = value_json.lower()
            if q in value_lower:
                score += value_lower.count(q) * 1.0
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

    # ---- Delta change-feed / operations ----------------------------------

    def sync_cdf(self, consumer_id: str) -> CDFSyncResult:
        """Consume one consumer's CDF tail and return changed memory keys.

        Available only when this store was configured with both a Delta
        ``table`` and an independent ``audit_table``. The latter holds the
        durable checkpoint and the audit trail, so VACUUM of the source table
        cannot make a consumer silently lose its position.
        """
        sync = getattr(self._backend, "sync_cdf", None)
        if sync is None:
            raise RuntimeError("CDF synchronization requires a Delta-backed KVStore")
        return sync(consumer_id)

    def operations(self) -> Dict[str, Any]:
        """Backend diagnostics suitable for a maintenance job or health check."""
        operations = getattr(self._backend, "operations", None)
        return operations() if operations is not None else {"backend": self._backend.label}

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
        → {"message": <langchain message_to_dict>, "ts": ..., "chat_id": ...}
    chat::{session_id}::{chat_id}
        → {"chat_id", "thread_id", "label", "started_at"}
           (static; only when a chat is active — see below)

    Chats
    -----
    A **chat** is one span of a thread: the messages written by one
    consumer, e.g. one ``SasLLMPipeline`` instance (Task → Thread → Chat →
    Message). Set one with :meth:`set_chat` and every later
    :meth:`add_messages` stamps its ``chat_id`` into each message value.
    The chat's first/last key and message count are derived from those
    stamps on read (:meth:`chats`), so they can never drift from the
    messages; the static ``chat::`` row is written once per chat.

    The chat id is deliberately **not** part of the message key. Thread ids
    are recovered from message keys by splitting off the last ``::`` segment
    (:meth:`ThreadMemoryManager.list_threads`), the fork copy is a key-prefix
    range, and the incremental read frontier is a key comparison — a third
    segment would break all three, and every legacy row at once. The id
    rides in the value instead (a derived ``chat_id`` column in Delta).

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
        self._chat_prefix = f"chat::{session_id}::"
        self._last_us = 0
        # Active chat (None = messages are written with no chat record, the
        # pre-chat behaviour). The state dict is read from the store on the
        # first write for a given chat id, so a second process resuming the
        # same chat extends its record instead of restarting it.
        self._chat_id: Optional[str] = None
        self._chat_label: Optional[str] = None
        self._chat_state: Optional[Dict[str, Any]] = None
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
        # Legacy chats (pre-stamp range rows) are backfilled once per instance.
        self._legacy_chats_checked = False
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

    def _refresh(self) -> List[BaseMessage]:
        """Bring the read cache up to date and return it — the *internal*
        list, which callers must not mutate (``messages`` copies it)."""
        if self._msg_cache is None:
            pairs = self._store.all_records(prefix=self._msg_prefix)
            self._msg_cache = []
            self._cache_keys = set()
        else:
            pairs = self._store.records_after(self._msg_prefix, self._cache_frontier)
            # The frontier tick is re-read in full; drop what is cached.
            pairs = [kv for kv in pairs if kv[0] not in self._cache_keys]
            if not pairs:
                return self._msg_cache
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
        return self._msg_cache

    # BaseChatMessageHistory annotates `messages` as a plain attribute, but
    # documents overriding it as a property — which is what this lazy,
    # cache-backed read needs.
    @property
    def messages(self) -> List[BaseMessage]:  # pyright: ignore[reportIncompatibleVariableOverride]
        return list(self._refresh())

    def message_count(self) -> int:
        """Number of stored messages, without copying the message list."""
        return len(self._refresh())

    def _invalidate_cache(self) -> None:
        self._msg_cache = None
        self._cache_keys = set()
        self._cache_frontier = ""

    def add_message(self, message: BaseMessage) -> None:
        self.add_messages([message])

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Persist several messages in ONE store call (a Delta INSERT for the
        messages, plus a MERGE for the chat index row when a chat is set)."""
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
                        # The chat this message belongs to: the chat index
                        # (first/last key, count) is derived from it on
                        # read, so it can never disagree with the messages.
                        **({"chat_id": self._chat_id} if self._chat_id else {}),
                    },
                    "tags": ["message", role, self.session_id],
                }
            )
        if not entries:
            return
        chat_entry = self._chat_entry()
        if chat_entry is not None:
            # Same batch as the messages it indexes. In Delta mode the
            # messages are INSERTed before the chat row is MERGEd, so the
            # index can never describe a write that did not land.
            entries.append(chat_entry)
        # Message keys are freshly minted and collision-free: append, not
        # upsert.
        self._store.append_many(entries)
        self._apply_retention()

    # ---- Chats -------------------------------------------------------------

    def set_chat(self, chat_id: Optional[str], *, label: Optional[str] = None) -> None:
        """Attach later writes to *chat_id* (``None`` detaches).

        Idempotent and cheap: re-stamping the same id keeps the loaded chat
        state, so a consumer may call this every time it takes the thread —
        which is what makes a shared :class:`ThreadMemoryManager` safe when
        two pipelines write to one thread.
        """
        if chat_id != self._chat_id:
            self._chat_state = None
        self._chat_id = chat_id
        self._chat_label = label

    @property
    def chat_id(self) -> Optional[str]:
        """The chat later writes are attributed to, or ``None``."""
        return self._chat_id

    def _chat_entry(self) -> Optional[Dict[str, Any]]:
        """The chat's static row, when it still needs writing — else ``None``.

        A chat row holds only what does not change per message (id, thread,
        label, start time) and is written once per chat — on its first write,
        or when a label arrives later. Everything that does (first/last key,
        message count) is derived from the messages' ``chat_id`` on read, so
        an ordinary append is a single message write with no index row to
        keep in step.
        """
        if self._chat_id is None:
            return None
        if self._chat_state is None:
            stored = self._store.get(self._chat_prefix + self._chat_id)
            if isinstance(stored, dict):
                self._chat_state = stored
                if not (self._chat_label and not stored.get("label")):
                    return None
            else:
                self._chat_state = {
                    "chat_id": self._chat_id,
                    "thread_id": self.session_id,
                    "label": self._chat_label,
                    "started_at": _now(),
                }
                return self._static_chat_row(self._chat_state)
        state = self._chat_state
        if self._chat_label and not state.get("label"):
            state["label"] = self._chat_label
            return self._static_chat_row(state)
        return None

    def _static_chat_row(self, state: Dict[str, Any]) -> Dict[str, Any]:
        static = {k: state.get(k) for k in ("chat_id", "thread_id", "label", "started_at")}
        return {
            "key": self._chat_prefix + str(static["chat_id"]),
            "value": static,
            "tags": ["chat", self.session_id, str(static["chat_id"])],
            "source": "KVChatMessageHistory",
        }

    def _chat_stats(self) -> Dict[str, Dict[str, Any]]:
        """Per-chat ``first_key``/``last_key``/``message_count`` of this
        thread, derived from the messages themselves."""
        derive = getattr(self._store, "chat_stats", None)
        derived = derive(self.session_id) if derive is not None else None
        if derived is not None:
            return derived
        stats: Dict[str, Dict[str, Any]] = {}
        for key, rec in self._store.all_records(prefix=self._msg_prefix):
            value = rec["value"]
            chat = value.get("chat_id") if isinstance(value, dict) else None
            if not chat:
                continue
            entry = stats.setdefault(
                chat, {"first_key": key, "last_key": key, "message_count": 0}
            )
            entry["first_key"] = min(entry["first_key"], key)
            entry["last_key"] = max(entry["last_key"], key)
            entry["message_count"] += 1
        return stats

    def backfill_legacy_chats(self) -> int:
        """Stamp ``chat_id`` into messages of chats written before stamping.

        A chat row that still carries ``first_key``/``last_key`` is from the
        pre-stamp layout: its messages have no ``chat_id``, so derived stats
        would lose them. Each unstamped message inside such a range gets the
        chat's id (one batched write; already-stamped messages are left as
        they are, so overlapping legacy ranges keep the earlier chat), and the
        row is rewritten as a static row. Idempotent; runs once per instance
        before every chat read. Returns the number of messages stamped.
        """
        if self._legacy_chats_checked:
            return 0
        self._legacy_chats_checked = True
        legacy = [
            (key, rec)
            for key, rec in self._store.all_records(prefix=self._chat_prefix)
            if isinstance(rec["value"], dict)
            and rec["value"].get("chat_id")
            and ("first_key" in rec["value"] or "last_key" in rec["value"])
        ]
        if not legacy:
            return 0
        legacy.sort(key=lambda kv: kv[1]["value"].get("started_at") or 0.0)
        messages = self._store.all_records(prefix=self._msg_prefix)
        stamped: Dict[str, Dict[str, Any]] = {}
        for _, rec in legacy:
            state = rec["value"]
            first, last = str(state.get("first_key") or ""), str(state.get("last_key") or "")
            for key, mrec in messages:
                value = mrec["value"]
                if (
                    first <= key <= last
                    and key not in stamped
                    and isinstance(value, dict)
                    and not value.get("chat_id")
                ):
                    stamped[key] = {
                        "key": key,
                        "value": {**value, "chat_id": state["chat_id"]},
                        "created_at": mrec["created_at"],
                        "updated_at": mrec["updated_at"],
                    }
        statics = [self._static_chat_row(rec["value"]) for _, rec in legacy]
        for row, (_, rec) in zip(statics, legacy):
            row["source"] = rec["source"]
        self._store.set_many([*stamped.values(), *statics])
        logger.info(
            f"backfill_legacy_chats: thread '{self.session_id}' stamped "
            f"{len(stamped)} message(s) across {len(legacy)} legacy chat(s)"
        )
        return len(stamped)

    def chats(self) -> List[Dict[str, Any]]:
        """This thread's chats, oldest first.

        Static rows joined with stats derived from the messages; a chat whose
        static row never landed still appears (``label``/``started_at``
        ``None``), because its messages say it exists.
        """
        self.backfill_legacy_chats()
        static = {
            str(rec["value"].get("chat_id")): rec["value"]
            for _, rec in self._store.all_records(prefix=self._chat_prefix)
            if isinstance(rec["value"], dict) and rec["value"].get("chat_id")
        }
        stats = self._chat_stats()
        records = []
        for chat_id in set(static) | set(stats):
            base: Dict[str, Any] = dict(
                static.get(chat_id) or {"chat_id": chat_id, "thread_id": self.session_id}
            )
            base.setdefault("label", None)
            base.setdefault("started_at", None)
            base.update(
                stats.get(chat_id)
                or {"first_key": None, "last_key": None, "message_count": 0}
            )
            records.append(base)
        records.sort(
            key=lambda state: (
                state.get("started_at") if state.get("started_at") is not None else float("inf"),
                state.get("first_key") or "",
            )
        )
        return records

    def messages_in_chat(self, chat_id: str) -> List[BaseMessage]:
        """The messages written under *chat_id*, in order.

        Selected by each message's own ``chat_id``, so messages another chat
        interleaved on the same thread are never included.
        """
        self.backfill_legacy_chats()
        records = getattr(self._store, "chat_records", None)
        pushed = records(self.session_id, chat_id) if records is not None else None
        if pushed is not None:
            pairs = pushed
        else:
            pairs = [
                (key, rec)
                for key, rec in self._store.all_records(prefix=self._msg_prefix)
                if isinstance(rec["value"], dict) and rec["value"].get("chat_id") == chat_id
            ]
        if not pairs:
            logger.warning(
                f"messages_in_chat: thread '{self.session_id}' has no messages "
                f"in chat '{chat_id}'"
            )
            return []
        pairs.sort(key=lambda kv: kv[0][len(self._msg_prefix) :])
        return [self._decode_message(rec["value"]) for _, rec in pairs]

    def clear(self) -> None:
        self._store.clear_prefix(self._msg_prefix)
        self._store.clear_prefix(self._chat_prefix)  # index of what just went
        self._store.delete(self._idx_key)  # legacy counter, if present
        self._chat_state = None
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

    def get_thread(
        self,
        thread_id: str,
        *,
        chat_id: Optional[str] = None,
        chat_label: Optional[str] = None,
    ) -> KVChatMessageHistory:
        """The thread's history, optionally stamped with the active chat.

        *chat_id* is re-stamped on every call rather than fixed at creation:
        thread objects are cached per id, so two consumers sharing a manager
        would otherwise inherit each other's chat.
        """
        if thread_id not in self._threads:
            self._threads[thread_id] = KVChatMessageHistory(
                thread_id,
                self._store,
                max_age_s=self._max_age_s,
                max_messages=self._max_messages,
            )
        thread = self._threads[thread_id]
        if chat_id is not None:
            thread.set_chat(chat_id, label=chat_label)
        return thread

    def chats(self, thread_id: str) -> List[Dict[str, Any]]:
        """*thread_id*'s chat records, oldest first (see
        :meth:`KVChatMessageHistory.chats`)."""
        return self.get_thread(thread_id).chats()

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
            # Also what makes a re-run after a partial fork safe: the rows
            # already copied make the destination non-empty, so the append
            # below can never duplicate them.
            raise ValueError(
                f"destination thread '{dst_thread_id}' is not empty"
            )
        # Copied messages carry their chat_id; stamp legacy chats first.
        self.get_thread(src_thread_id).backfill_legacy_chats()
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
        chat_entries = self._forked_chat_entries(src_thread_id, dst_thread_id, records)
        # The destination was checked empty above: the copies are new rows.
        self._store.append_many([*entries, *chat_entries])
        logger.info(
            f"fork_thread: copied {len(entries)} message(s) and "
            f"{len(chat_entries)} chat record(s) "
            f"'{src_thread_id}' -> '{dst_thread_id}'"
        )
        return len(entries)

    def _forked_chat_entries(
        self, src_thread_id: str, dst_thread_id: str, copied: List[Tuple[str, Dict]]
    ) -> List[Dict[str, Any]]:
        """Static chat rows for the chats the copied slice contains.

        The copied messages keep their ``chat_id``, so the branch's derived
        chat stats are automatically clamped to what was copied; only the
        static row (label, start time) of each chat with at least one copied
        message needs carrying over. A chat that starts past the fork point
        has no copied message and is not carried.
        """
        copied_chats = {
            rec["value"].get("chat_id")
            for _, rec in copied
            if isinstance(rec["value"], dict) and rec["value"].get("chat_id")
        }
        forked: List[Dict[str, Any]] = []
        for _, rec in self._store.all_records(prefix=f"chat::{src_thread_id}::"):
            state = rec["value"]
            if not isinstance(state, dict) or state.get("chat_id") not in copied_chats:
                continue
            chat_id = str(state["chat_id"])
            forked.append(
                {
                    "key": f"chat::{dst_thread_id}::{chat_id}",
                    "value": {
                        k: state.get(k) for k in ("chat_id", "label", "started_at")
                    }
                    | {"thread_id": dst_thread_id},
                    "tags": ["chat", dst_thread_id, chat_id],
                    "source": rec["source"],
                }
            )
        return forked

    def list_threads(self) -> List[str]:
        """Thread ids with at least one stored message.

        Derived from the store's keys — not from the threads this manager
        instance happens to have touched — so with a persistent Delta
        backend the list survives a process restart.
        """
        return self._store.thread_ids()

    def delete_thread(self, thread_id: str) -> None:
        self.get_thread(thread_id).clear()

    def invalidate_caches(self) -> None:
        """Drop every managed thread's read cache — required after an
        out-of-band store rewrite such as :meth:`KVStore.restore`."""
        for thread in self._threads.values():
            thread._invalidate_cache()

    def invalidate_for_keys(self, keys: Sequence[str]) -> None:
        """Invalidate only cached threads affected by CDF change events."""
        affected: set[str] = set()
        for key in keys:
            if key.startswith(("msg::", "chat::")):
                remainder = key.split("::", 1)[1]
                affected.add(remainder.rsplit("::", 1)[0])
        for thread_id in affected:
            thread = self._threads.get(thread_id)
            if thread is not None:
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
        Delta base table name, e.g. ``"main.ml.langchain_memory"``; the
        typed tables ``main.ml.langchain_memory_*`` are derived from it.
        Pass ``None`` (default) for local in-memory operation.
    ranker : Any | None
        Optional ``memory.relevance.HybridRanker`` forwarded to the
        :class:`KVMemoryStore`, upgrading ``mem.kv.search`` from a
        substring scan to hybrid BM25 (+ optional dense) retrieval.
    full_text_search : bool
        Delta mode on Databricks (DBR 18.3+, targeted at DBR 19): enable
        predictive optimization, create the full-text search indexes, and
        push ``kv.search`` / ``chunk_history.search`` down as ``isearch()``.
        Off-Databricks the setup failure is logged once and searches fall
        back to ``LIKE`` scans with identical results.
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
        # pipeline.engine): the node reads mem.get_thread(thread_id).messages
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
        *,
        cdf_consumer_id: str | None = None,
        cdf_audit_table: str | None = None,
        max_write_retries: int = 3,
        full_text_search: bool = True,
    ) -> None:
        if cdf_consumer_id is not None and cdf_audit_table is None:
            raise ValueError(
                "cdf_consumer_id requires cdf_audit_table"
            )
        self._cdf_consumer_id = cdf_consumer_id
        self._store = KVStore(
            spark=spark,
            table=table,
            audit_table=cdf_audit_table,
            max_write_retries=max_write_retries,
            full_text_search=full_text_search,
        )
        # Original-vs-converted chunk history: same base table family, its
        # own append-only tables (memory.chunk_history).
        self.chunk_history = ChunkHistory(
            spark if table else None,
            table,
            full_text_search=full_text_search,
            max_write_retries=max_write_retries,
        )
        self.threads = ThreadMemoryManager(
            store=self._store,
            max_age_s=retention_max_age_s,
            max_messages=retention_max_messages,
        )
        self.kv = KVMemoryStore(store=self._store, namespace="kv", ranker=ranker)

    def get_thread(
        self,
        thread_id: str,
        *,
        chat_id: Optional[str] = None,
        chat_label: Optional[str] = None,
    ) -> KVChatMessageHistory:
        return self.threads.get_thread(
            thread_id, chat_id=chat_id, chat_label=chat_label
        )

    def chats(self, thread_id: str) -> List[Dict[str, Any]]:
        """The chats recorded on *thread_id*, oldest first.

        One chat = one consumer's span of the thread (Task → Thread → Chat →
        Message); ``SasLLMPipeline`` opens one per pipeline instance.
        """
        return self.threads.chats(thread_id)

    def delete_thread(self, thread_id: str) -> None:
        """Drop *thread_id*'s messages and chat index.

        Thread-scoped **notes** (``memory.thread_mem``) and the rolling
        summary are not touched here: both live behind duck-typed stores this
        façade does not own, and both are cleared through the object that
        wrote them (``ThreadMemory.clear`` / ``RollingSummarizer.reset``).
        """
        self.threads.delete_thread(thread_id)

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

    def sync_changes(self) -> CDFSyncResult:
        """Apply other processes' committed Delta changes to local caches.

        Call this at a request boundary in every long-lived process. The first
        call establishes the snapshot boundary and conservatively invalidates
        every cached thread; later calls invalidate only threads with changed
        ``msg::`` or ``chat::`` records.
        """
        if self._cdf_consumer_id is None:
            raise RuntimeError(
                "sync_changes requires cdf_consumer_id and cdf_audit_table"
            )
        result = self._store.sync_cdf(self._cdf_consumer_id)
        if result.baseline:
            self.threads.invalidate_caches()
        else:
            self.threads.invalidate_for_keys(
                [str(event["key"]) for event in result.events]
            )
        return result

    @property
    def cdf_enabled(self) -> bool:
        """Whether this hub has an explicitly configured CDF consumer."""
        return self._cdf_consumer_id is not None

    def prune_thread(self, thread_id: str, before_ts: float) -> int:
        """Storage-side trim: delete *thread_id* messages older than
        *before_ts* (Unix seconds).  Returns the number removed."""
        return self.get_thread(thread_id).prune_before(before_ts)

    def summary(self) -> Dict[str, Any]:
        return {
            "store": self._store.stats(),
            "operations": self._store.operations(),
            "threads": self.threads.list_threads(),
            "kv": self.kv.stats(),
            "chunk_history": self.chunk_history.stats(),
        }
