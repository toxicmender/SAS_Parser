"""Delta helpers shared by memory.store, memory.chunk_history and
memory.operations — one copy of each, so a fix lands everywhere.

No module-level pyspark import: callers pass the SparkSession in.

Logger name: ``memory._delta``.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Tuple

logger = logging.getLogger(__name__)

# Delta optimistic-concurrency failures that are safe to retry once the
# operation is re-run against the new snapshot (docs.delta.io: concurrency
# control). Matched against the exception text, which carries the JVM class
# name and, on newer runtimes, the error class.
_CONFLICT_MARKERS: Tuple[str, ...] = (
    "ConcurrentAppendException",
    "ConcurrentDeleteReadException",
    "ConcurrentDeleteDeleteException",
    "ConcurrentTransactionException",
    "MetadataChangedException",
    "ProtocolChangedException",
    "DELTA_CONCURRENT_APPEND",
    "DELTA_CONCURRENT_DELETE_READ",
    "DELTA_CONCURRENT_DELETE_DELETE",
    "DELTA_CONCURRENT_TRANSACTION",
    "DELTA_CONCURRENT_WRITE",
    "DELTA_METADATA_CHANGED",
    "DELTA_PROTOCOL_CHANGED",
)

# Losing a race to create the same table/view: the object now exists, which is
# what the caller wanted; re-running the IF NOT EXISTS statement settles it.
_CREATE_RACE_MARKERS: Tuple[str, ...] = (
    "TABLE_OR_VIEW_ALREADY_EXISTS",
    "DELTA_TABLE_ALREADY_EXISTS",
    "TableAlreadyExistsException",
    # The loser of a race to create a managed table whose location the
    # winner has just initialised.
    "DELTA_CREATE_TABLE_WITH_NON_EMPTY_LOCATION",
)


def exception_text(exc: BaseException) -> str:
    """Everything that identifies *exc*: its message, its Spark error
    condition, and its repr.

    The condition matters on its own: when the Delta error-class JSON does
    not match the Spark build, ``str(exc)`` comes back *empty* (the message
    template lookup fails) while ``getCondition()`` still names the error.
    """
    parts = [str(exc), type(exc).__name__]
    for getter in ("getCondition", "getErrorClass"):
        method = getattr(exc, getter, None)
        if callable(method):
            try:
                value = method()
            except Exception:  # pragma: no cover - JVM side unavailable
                value = None
            if value:
                parts.append(str(value))
    java = getattr(exc, "java_exception", None)
    if java is not None:
        try:
            parts.append(str(java.getClass().getName()))
        except Exception:  # pragma: no cover
            pass
    return " ".join(parts)


def is_retryable_conflict(exc: BaseException) -> bool:
    """Whether *exc* is a Delta concurrency conflict worth retrying."""
    text = exception_text(exc)
    return any(marker in text for marker in _CONFLICT_MARKERS)


def is_retryable_ddl(exc: BaseException) -> bool:
    """A conflict, or a lost create race, during idempotent DDL."""
    text = exception_text(exc)
    return is_retryable_conflict(exc) or any(m in text for m in _CREATE_RACE_MARKERS)


def with_retry(
    operation: str,
    action: Callable[[], Any],
    *,
    max_retries: int = 3,
    retry_on: Callable[[BaseException], bool] = is_retryable_conflict,
    base_delay: float = 0.05,
) -> Any:
    """Run *action*, retrying with exponential backoff while *retry_on* says so.

    Only for idempotent actions (MERGE, DDL ``IF NOT EXISTS``, DELETE): a blind
    INSERT must verify before retrying (see ``_DeltaBackend.append``).
    """
    for attempt in range(max_retries + 1):
        try:
            return action()
        except Exception as exc:
            if not retry_on(exc) or attempt == max_retries:
                raise
            delay = base_delay * (2**attempt)
            logger.warning(
                "Delta %s conflicted; retrying (%s/%s) in %.2fs",
                operation,
                attempt + 1,
                max_retries,
                delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


@contextmanager
def temp_view(spark: Any, df: Any, stem: str) -> Iterator[str]:
    """Register *df* under a unique temp-view name for one statement.

    Unique per call: a fixed name would let two concurrent writers on one
    SparkSession clobber each other's staged rows.
    """
    view = f"_mem_{stem}_{uuid.uuid4().hex}"
    df.createOrReplaceTempView(view)
    try:
        yield view
    finally:
        spark.catalog.dropTempView(view)


_SEARCH_PROBE: Dict[str, bool] = {}


def _app_id(spark: Any) -> str:
    try:
        return str(spark.sparkContext.applicationId)
    except Exception:  # pragma: no cover - Spark Connect has no sparkContext
        return f"session-{id(spark)}"


def probe_search_functions(spark: Any) -> bool:
    """Whether ``search``/``isearch`` exist on this runtime (DBR 18.3+).

    Probed once per Spark application and cached: the answer cannot change
    while the application runs.
    """
    key = _app_id(spark)
    cached = _SEARCH_PROBE.get(key)
    if cached is not None:
        return cached
    try:
        spark.sql("SELECT isearch('memory', 'mem', mode => 'substring') AS hit").collect()
        available = True
    except Exception:
        logger.info("Delta memory: search()/isearch() unavailable; falling back to LIKE scans")
        available = False
    _SEARCH_PROBE[key] = available
    return available


def application_key(spark: Any, base: str) -> Tuple[str, str]:
    """Cache key for per-process, per-base provisioning state."""
    return _app_id(spark), base


def table_status(spark: Any, quoted_table: str) -> Dict[str, Any]:
    """Latest version/operation and size of one Delta table."""
    latest = spark.sql(f"DESCRIBE HISTORY {quoted_table} LIMIT 1").first()
    detail = spark.sql(f"DESCRIBE DETAIL {quoted_table}").first()
    return {
        "latest_version": int(latest.version) if latest is not None else None,
        "last_operation": getattr(latest, "operation", None),
        "size_in_bytes": getattr(detail, "sizeInBytes", None),
        "num_files": getattr(detail, "numFiles", None),
    }


def latest_version(spark: Any, quoted_table: str) -> int:
    row = spark.sql(f"DESCRIBE HISTORY {quoted_table} LIMIT 1").first()
    return int(row.version) if row is not None else 0
