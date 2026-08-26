"""Native Delta key-value persistence for the v2 memory adapter.

PySpark is loaded only when :class:`DeltaKVStore` is constructed.  Importing
the v2 adapter therefore remains safe in core-only and wheel-smoke runtimes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .delta_operations import quoted_table_name

logger = logging.getLogger(__name__)

_KV_COLUMNS = (
    "kv_key STRING NOT NULL, value STRING NOT NULL, tags STRING, "
    "created_at DOUBLE, updated_at DOUBLE, source STRING"
)
_KV_SCHEMA = (
    "kv_key STRING, value STRING, tags STRING, created_at DOUBLE, "
    "updated_at DOUBLE, source STRING"
)
_AUDIT_COLUMNS = (
    "event_id STRING NOT NULL, consumer_id STRING NOT NULL, "
    "source_table STRING NOT NULL, kv_key STRING, change_type STRING NOT NULL, "
    "commit_version BIGINT NOT NULL, commit_timestamp TIMESTAMP, value STRING, "
    "tags STRING, recorded_at DOUBLE NOT NULL"
)
_AUDIT_SCHEMA = (
    "event_id STRING, consumer_id STRING, source_table STRING, kv_key STRING, "
    "change_type STRING, commit_version BIGINT, commit_timestamp TIMESTAMP, "
    "value STRING, tags STRING, recorded_at DOUBLE"
)


@dataclass(frozen=True)
class CDFSyncResult:
    """Result of consuming one durable Delta Change Data Feed tail."""

    baseline: bool = False
    events: tuple[dict[str, Any], ...] = ()
    checkpoint_version: int | None = None


RawRecord = tuple[str, str, str | None, float | None, float, str | None]


def _encode(value: Any) -> str:
    return json.dumps(value, default=str)


def _decode(value: str) -> Any:
    return json.loads(value)


def _spark_functions() -> Any:
    try:
        from pyspark.sql import functions  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - exercised in a clean subprocess
        raise RuntimeError(
            "Delta memory requires PySpark; install the 'spark' project extra"
        ) from exc
    return functions


class DeltaKVStore:
    """CDF-enabled Delta persistence owned by the v2 memory adapter."""

    def __init__(
        self,
        spark: Any,
        table: str,
        *,
        audit_table: str,
        max_write_retries: int = 3,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if spark is None:
            raise ValueError("Delta memory requires an active SparkSession")
        if max_write_retries < 0:
            raise ValueError("max_write_retries must be >= 0")
        self._functions = _spark_functions()
        self._spark = spark
        self._table_name = table
        self._table = quoted_table_name(table)
        self._audit_table_name = audit_table
        self._audit_table = quoted_table_name(audit_table)
        self._max_write_retries = max_write_retries
        self._clock = clock
        self._sleeper = sleeper
        self._ensure_table()

    def _ensure_table(self) -> None:
        self._spark.sql(
            f"CREATE TABLE IF NOT EXISTS {self._table} ({_KV_COLUMNS}) "
            "USING DELTA TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')"
        )
        if not self._cdf_enabled():
            self._spark.sql(
                f"ALTER TABLE {self._table} SET TBLPROPERTIES "
                "('delta.enableChangeDataFeed' = 'true')"
            )
        fields = {field.name: field for field in self._spark.table(self._table).schema}
        if "source" not in fields:
            self._spark.sql(f"ALTER TABLE {self._table} ADD COLUMNS (source STRING)")
        self._validate_schema()

    def _cdf_enabled(self) -> bool:
        try:
            rows = self._spark.sql(f"SHOW TBLPROPERTIES {self._table}").collect()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            logger.debug("Could not inspect CDF property for %s: %s", self._table, exc)
            return False
        return any(
            row["key"] == "delta.enableChangeDataFeed"
            and str(row["value"]).strip().lower() == "true"
            for row in rows
        )

    def _validate_schema(self) -> None:
        expected = {
            "kv_key": "string",
            "value": "string",
            "tags": "string",
            "created_at": "double",
            "updated_at": "double",
            "source": "string",
        }
        actual = {
            field.name: field.dataType.simpleString()
            for field in self._spark.table(self._table).schema
        }
        invalid = {
            name: (kind, actual.get(name))
            for name, kind in expected.items()
            if actual.get(name) != kind
        }
        if invalid:
            raise RuntimeError(
                f"Delta table {self._table_name!r} has an incompatible memory "
                f"schema: {invalid!r}"
            )

    @staticmethod
    def _is_concurrent_write(exc: Exception) -> bool:
        message = str(exc).upper()
        return (
            "CONCURRENT" in message
            or "CONFLICT" in message
            or ("DELTA_" in message and "COMMIT" in message)
        )

    def _write(self, operation: str, action: Callable[[], None]) -> None:
        for attempt in range(self._max_write_retries + 1):
            try:
                action()
                return
            except Exception as exc:
                if (
                    not self._is_concurrent_write(exc)
                    or attempt == self._max_write_retries
                ):
                    raise
                delay = 0.05 * (2**attempt)
                logger.warning(
                    "Delta %s conflicted; retrying (%s/%s) in %.2fs",
                    operation,
                    attempt + 1,
                    self._max_write_retries,
                    delay,
                )
                self._sleeper(delay)

    def _frame(self, prefix: str = "") -> Any:
        frame = self._spark.table(self._table)
        if prefix:
            frame = frame.filter(self._functions.col("kv_key").startswith(prefix))
        return frame

    def _stage(self, rows: list[RawRecord], now: float) -> Any:
        staged = [
            (
                key,
                value,
                tags,
                created_at if created_at is not None else now,
                updated_at,
                source,
            )
            for key, value, tags, created_at, updated_at, source in rows
        ]
        return self._spark.createDataFrame(staged, schema=_KV_SCHEMA)

    def _upsert(self, rows: list[RawRecord], now: float) -> None:
        view = f"_v2_memory_upsert_{uuid4().hex}"
        self._stage(rows, now).createOrReplaceTempView(view)
        try:
            self._write(
                "MERGE",
                lambda: self._spark.sql(
                    f"""MERGE INTO {self._table} AS target
                    USING {view} AS source ON target.kv_key = source.kv_key
                    WHEN MATCHED THEN UPDATE SET
                        target.value = source.value,
                        target.tags = COALESCE(source.tags, target.tags),
                        target.updated_at = source.updated_at,
                        target.source = COALESCE(source.source, target.source)
                    WHEN NOT MATCHED THEN INSERT *"""
                ),
            )
        finally:
            self._spark.catalog.dropTempView(view)

    def set(
        self,
        key: str,
        value: Any,
        tags: list[str] | None = None,
        source: str | None = None,
    ) -> None:
        now = self._clock()
        self._upsert(
            [(key, _encode(value), None if tags is None else _encode(tags), None, now, source)],
            now,
        )

    def get(self, key: str, default: Any = None) -> Any:
        rows = (
            self._frame()
            .filter(self._functions.col("kv_key") == key)
            .select("value")
            .limit(1)
            .collect()
        )
        return _decode(rows[0].value) if rows else default

    def _delete_keys(self, keys: list[str]) -> None:
        view = f"_v2_memory_delete_{uuid4().hex}"
        self._spark.createDataFrame([(key,) for key in keys], "kv_key STRING").createOrReplaceTempView(view)
        try:
            self._write(
                "DELETE",
                lambda: self._spark.sql(
                    f"MERGE INTO {self._table} target USING {view} source "
                    "ON target.kv_key = source.kv_key WHEN MATCHED THEN DELETE"
                ),
            )
        finally:
            self._spark.catalog.dropTempView(view)

    def delete(self, key: str) -> bool:
        return self.delete_many([key]) == 1

    def delete_many(self, keys: list[str]) -> int:
        unique = list(dict.fromkeys(keys))
        if not unique:
            return 0
        existing = [
            row.kv_key
            for row in self._frame()
            .filter(self._functions.col("kv_key").isin(unique))
            .select("kv_key")
            .collect()
        ]
        if existing:
            self._delete_keys(existing)
        return len(existing)

    def keys(self, prefix: str = "") -> list[str]:
        return [row.kv_key for row in self._frame(prefix).select("kv_key").collect()]

    def all_records(self, prefix: str = "") -> list[tuple[str, dict[str, Any]]]:
        return [
            (
                row.kv_key,
                {
                    "value": _decode(row.value),
                    "tags": _decode(row.tags) if row.tags else [],
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                    "source": row.source,
                },
            )
            for row in self._frame(prefix).collect()
        ]

    def _latest_version(self) -> int:
        row = self._spark.sql(f"DESCRIBE HISTORY {self._table} LIMIT 1").first()
        return int(row.version) if row is not None else 0

    def _ensure_audit_table(self) -> None:
        self._spark.sql(
            f"CREATE TABLE IF NOT EXISTS {self._audit_table} ({_AUDIT_COLUMNS}) "
            "USING DELTA"
        )

    @staticmethod
    def _event_id(
        consumer_id: str,
        source_table: str,
        version: int,
        key: str,
        change_type: str,
    ) -> str:
        material = (
            f"{consumer_id}\0{source_table}\0{version}\0{key}\0{change_type}"
            if change_type == "checkpoint"
            else f"{source_table}\0{version}\0{key}\0{change_type}"
        )
        return hashlib.sha256(material.encode()).hexdigest()

    def _audit_upsert(self, rows: list[tuple[Any, ...]]) -> None:
        self._ensure_audit_table()
        view = f"_v2_memory_cdf_audit_{uuid4().hex}"
        self._spark.createDataFrame(rows, schema=_AUDIT_SCHEMA).createOrReplaceTempView(view)
        try:
            self._write(
                "CDF audit MERGE",
                lambda: self._spark.sql(
                    f"""MERGE INTO {self._audit_table} AS target
                    USING {view} AS source ON target.event_id = source.event_id
                    WHEN MATCHED THEN UPDATE SET
                        target.commit_version = source.commit_version,
                        target.commit_timestamp = source.commit_timestamp,
                        target.recorded_at = source.recorded_at
                    WHEN NOT MATCHED THEN INSERT *"""
                ),
            )
        finally:
            self._spark.catalog.dropTempView(view)

    def sync_cdf(self, consumer_id: str) -> CDFSyncResult:
        if not isinstance(consumer_id, str) or not consumer_id.strip():
            raise ValueError("CDF consumer_id must be a non-empty string")
        self._ensure_audit_table()
        checkpoint_id = self._event_id(
            consumer_id, self._table_name, -1, "__checkpoint__", "checkpoint"
        )
        checkpoint_rows = (
            self._spark.table(self._audit_table)
            .filter(self._functions.col("event_id") == checkpoint_id)
            .select("commit_version")
            .limit(1)
            .collect()
        )
        latest = self._latest_version()
        now = self._clock()
        if not checkpoint_rows:
            self._audit_upsert(
                [
                    (
                        checkpoint_id,
                        consumer_id,
                        self._table_name,
                        None,
                        "checkpoint",
                        latest,
                        None,
                        None,
                        None,
                        now,
                    )
                ]
            )
            return CDFSyncResult(baseline=True, checkpoint_version=latest)

        checkpoint = int(checkpoint_rows[0].commit_version)
        if checkpoint >= latest:
            return CDFSyncResult(checkpoint_version=checkpoint)

        changes = (
            self._spark.read.format("delta")
            .option("readChangeFeed", "true")
            .option("startingVersion", checkpoint + 1)
            .option("endingVersion", latest)
            .table(self._table)
            .collect()
        )
        events: list[dict[str, Any]] = []
        audit_rows: list[tuple[Any, ...]] = []
        for row in changes:
            data = row.asDict(recursive=True)
            key = str(data["kv_key"])
            change_type = str(data["_change_type"])
            version = int(data["_commit_version"])
            events.append(
                {
                    "key": key,
                    "change_type": change_type,
                    "commit_version": version,
                    "commit_timestamp": data.get("_commit_timestamp"),
                }
            )
            audit_rows.append(
                (
                    self._event_id(
                        consumer_id,
                        self._table_name,
                        version,
                        key,
                        change_type,
                    ),
                    consumer_id,
                    self._table_name,
                    key,
                    change_type,
                    version,
                    data.get("_commit_timestamp"),
                    data.get("value"),
                    data.get("tags"),
                    now,
                )
            )
        audit_rows.append(
            (
                checkpoint_id,
                consumer_id,
                self._table_name,
                None,
                "checkpoint",
                latest,
                None,
                None,
                None,
                now,
            )
        )
        self._audit_upsert(audit_rows)
        return CDFSyncResult(events=tuple(events), checkpoint_version=latest)

    def operations(self) -> dict[str, Any]:
        latest = self._spark.sql(f"DESCRIBE HISTORY {self._table} LIMIT 1").first()
        detail = self._spark.sql(f"DESCRIBE DETAIL {self._table}").first()
        return {
            "table": self._table_name,
            "latest_version": int(latest.version) if latest is not None else None,
            "last_operation": getattr(latest, "operation", None),
            "size_in_bytes": getattr(detail, "sizeInBytes", None),
            "num_files": getattr(detail, "numFiles", None),
            "audit_table": self._audit_table_name,
        }


__all__ = ["CDFSyncResult", "DeltaKVStore"]
