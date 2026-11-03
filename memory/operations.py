"""Guarded Delta maintenance commands for the memory store.

Maintenance is intentionally separate from the hot-path backend.  A pipeline
must never compact or vacuum its own state during a request; an operator or
scheduled job owns these explicit, observable actions instead.

The store is a *family* of tables derived from one base name (see
:mod:`memory.schema`); every command here acts on all of them. On Databricks
with predictive optimization enabled (the default when
``memory.full_text_search`` is on), OPTIMIZE and VACUUM already run
automatically — the explicit commands stay for other runtimes and for
operators who need them now.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from . import schema
from .schema import quoted_table_name as _quoted_table_name

logger = logging.getLogger(__name__)

MIN_VACUUM_HOURS = 7 * 24
"""Delta's safe default: one week."""

MAX_VACUUM_HOURS = 4 * 30 * 24
"""Application policy ceiling: four 30-day months."""

MIGRATION_BATCH_ROWS = 5000
"""Rows per MERGE when copying a legacy single-table store."""


@dataclass(frozen=True)
class VacuumPolicy:
    """Retention guardrail for the memory Delta tables and their CDF consumers."""

    retention_hours: int
    max_cdf_outage_hours: int = 0

    def __post_init__(self) -> None:
        if not MIN_VACUUM_HOURS <= self.retention_hours <= MAX_VACUUM_HOURS:
            raise ValueError(
                "retention_hours must be between "
                f"{MIN_VACUUM_HOURS} and {MAX_VACUUM_HOURS}"
            )
        if self.max_cdf_outage_hours < 0:
            raise ValueError("max_cdf_outage_hours must be >= 0")
        if self.retention_hours <= self.max_cdf_outage_hours:
            raise ValueError(
                "retention_hours must exceed max_cdf_outage_hours so every "
                "CDF consumer can resume before VACUUM removes its changes"
            )


class DeltaMemoryMaintenance:
    """Explicitly inspect, compact, vacuum, index, or migrate the memory tables.

    *table* is the memory **base** name (``memory.delta_table``); the commands
    act on every derived table (:data:`memory.schema.ALL_TABLES`).
    """

    def __init__(self, spark: Any, table: str, *, policy: VacuumPolicy) -> None:
        self._spark = spark
        self.table_name = table
        _quoted_table_name(table)  # validate the base
        self.policy = policy

    @property
    def tables(self) -> list[str]:
        """Every derived table name, in :data:`memory.schema.ALL_TABLES` order."""
        return [spec.name(self.table_name) for spec in schema.ALL_TABLES]

    def _existing(self) -> list[schema.TableSpec]:
        return [
            spec
            for spec in schema.ALL_TABLES
            if self._spark.catalog.tableExists(spec.name(self.table_name))
        ]

    def status(self) -> dict[str, Any]:
        """Return the current Delta history/details needed by an operator."""
        tables: dict[str, Any] = {}
        for spec in self._existing():
            quoted = _quoted_table_name(spec.name(self.table_name))
            latest = self._spark.sql(f"DESCRIBE HISTORY {quoted} LIMIT 1").first()
            detail = self._spark.sql(f"DESCRIBE DETAIL {quoted}").first()
            tables[spec.name(self.table_name)] = {
                "latest_version": int(latest.version) if latest is not None else None,
                "last_operation": getattr(latest, "operation", None),
                "size_in_bytes": getattr(detail, "sizeInBytes", None),
                "num_files": getattr(detail, "numFiles", None),
            }
        return {
            "base_table": self.table_name,
            "tables": tables,
            "vacuum_retention_hours": self.policy.retention_hours,
        }

    def optimize(self) -> None:
        """Run Delta compaction (liquid clustering) on every memory table."""
        for spec in self._existing():
            self._spark.sql(f"OPTIMIZE {_quoted_table_name(spec.name(self.table_name))}")

    def vacuum(self, *, dry_run: bool = True) -> dict[str, list[Any]]:
        """Vacuum every table within policy; dry-run by default."""
        out: dict[str, list[Any]] = {}
        for spec in self._existing():
            name = spec.name(self.table_name)
            command = (
                f"VACUUM {_quoted_table_name(name)} RETAIN "
                f"{self.policy.retention_hours} HOURS" + (" DRY RUN" if dry_run else "")
            )
            out[name] = self._spark.sql(command).collect()
        return out

    # ---- full-text search indexes ----------------------------------------

    def search_index_status(self) -> dict[str, Any]:
        """``DESCRIBE INDEX`` for every declared index (``None`` when absent)."""
        out: dict[str, Any] = {}
        for spec in schema.ALL_TABLES:
            for index in spec.search_indexes:
                name = spec.index_name(self.table_name, index.column)
                try:
                    result, _ = schema.run_index_sql(
                        self._spark, spec, self.table_name, index.column, "DESCRIBE INDEX {name}"
                    )
                    out[name] = [r.asDict() for r in result.collect()]
                except Exception as exc:
                    out[name] = None
                    logger.debug(f"search_index_status: {name}: {exc}")
        return out

    def rebuild_search_indexes(self) -> schema.ProvisionReport:
        """Drop and recreate every full-text search index.

        Needed once when the feature leaves Beta: Databricks does not carry
        Beta-era indexes forward to Public Preview.
        """
        for spec in schema.ALL_TABLES:
            for index in spec.search_indexes:
                schema.run_index_sql(
                    self._spark, spec, self.table_name, index.column, "DROP INDEX IF EXISTS {name}"
                )
        return schema.enable_databricks_features(
            self._spark, self.table_name, tuple(self._existing())
        )

    # ---- restore recovery ------------------------------------------------

    def recover_restore(self) -> list[str]:
        """Roll back every interrupted ``KVStore.restore`` run.

        A restore that crashed mid-way leaves some memory tables overwritten
        and others not; each table is put back to the version journaled
        before that run began. Returns the recovered restore ids. Rerun the
        restore afterwards if its snapshot is still wanted.
        """
        from .store import _DeltaBackend

        backend = _DeltaBackend(self._spark, self.table_name, full_text_search=False)
        recovered = []
        for restore_id, versions in backend.unfinished_restores().items():
            backend.rollback_restore(restore_id, versions)
            recovered.append(restore_id)
            logger.info(f"recover_restore: rolled back {restore_id} -> {versions}")
        return recovered

    # ---- chunk history archival ------------------------------------------

    def archive_chunk_history(
        self,
        *,
        before_ts: float | None = None,
        thread_id: str | None = None,
        superseded_only: bool = False,
        keep_current: bool = True,
    ) -> Any:
        """Move stale chunk-history attempts to the archive tables — see
        :meth:`memory.chunk_history.ChunkHistory.archive`. Run :meth:`vacuum`
        afterwards (or let predictive optimization) to reclaim the files the
        deletes leave behind."""
        from .chunk_history import ChunkHistory

        return ChunkHistory(self._spark, self.table_name, full_text_search=False).archive(
            before_ts=before_ts,
            thread_id=thread_id,
            superseded_only=superseded_only,
            keep_current=keep_current,
        )

    # ---- migration ---------------------------------------------------------

    def migrate_legacy(self, legacy_table: str, *, force: bool = False) -> dict[str, int]:
        """Copy a pre-split single-table store into the typed tables, once.

        Every legacy row is routed by its key exactly as live writes are and
        MERGEd on ``kv_key`` (timestamps, tags and provenance preserved), so a
        rerun is idempotent. Refuses to write into a store that already holds
        rows unless ``force=True``. The legacy table itself is left untouched.
        Returns the number of rows copied per target table.
        """
        from .store import _DeltaBackend

        # The legacy table is usually the base name itself: its derived
        # tables carry suffixes, so the two never collide.
        legacy = _quoted_table_name(legacy_table)
        backend = _DeltaBackend(self._spark, self.table_name, full_text_search=False)
        if backend.count() and not force:
            raise RuntimeError(
                f"memory tables under {self.table_name!r} already hold rows; "
                "pass force=True to merge the legacy rows into them"
            )
        copied: dict[str, int] = {}
        batch: list[tuple[Any, ...]] = []

        def flush() -> None:
            if not batch:
                return
            for row in batch:
                suffix = backend._route(row[0]).suffix
                copied[suffix] = copied.get(suffix, 0) + 1
            backend.upsert(list(batch), _now())
            batch.clear()

        from pyspark.sql import functions as F

        df = self._spark.table(legacy)
        present = set(df.columns)
        rows = df.select(
            *[
                F.col(c) if c in present else F.lit(None).alias(c)
                for c in schema.KV_CORE_NAMES
            ]
        ).toLocalIterator()
        for row in rows:
            batch.append(
                (
                    row.kv_key,
                    row.value,
                    row.tags,
                    row.created_at,
                    row.updated_at if row.updated_at is not None else _now(),
                    row.source,
                )
            )
            if len(batch) >= MIGRATION_BATCH_ROWS:
                flush()
        flush()
        logger.info(f"migrate_legacy: {legacy_table} -> {self.table_name}_*: {copied}")
        return copied


def _now() -> float:
    return time.time()
