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

from . import _delta, schema
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
        tables: dict[str, Any] = {
            spec.name(self.table_name): _delta.table_status(
                self._spark, _quoted_table_name(spec.name(self.table_name))
            )
            for spec in self._existing()
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
        """Vacuum every table within policy; dry-run by default.

        A real run also prunes ``B_restore_journal`` rows of settled restores
        older than the retention window, so the journal cannot grow forever.
        """
        out: dict[str, list[Any]] = {}
        if not dry_run:
            self._backend().prune_restore_journal(
                time.time() - self.policy.retention_hours * 3600
            )
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

    def rebuild_search_indexes(self) -> dict[str, schema.ProvisionReport]:
        """Drop and recreate every full-text search index.

        Needed once when the feature leaves Beta: Databricks does not carry
        Beta-era indexes forward to Public Preview.
        """
        for spec in schema.ALL_TABLES:
            for index in spec.search_indexes:
                schema.run_index_sql(
                    self._spark, spec, self.table_name, index.column, "DROP INDEX IF EXISTS {name}"
                )
        return self.provision(force=True)

    def provision(self, *, force: bool = True) -> dict[str, schema.ProvisionReport]:
        """(Re)run the one-time Databricks setup: predictive optimization,
        search indexes and the provisioned marker, for the KV and the
        chunk-history table groups. Clears this process's provisioning cache
        so the next store re-validates too."""
        from .store import forget_provisioning, provision_databricks

        forget_provisioning(self.table_name)
        existing = set(self._existing())
        return {
            group: provision_databricks(
                self._spark,
                self.table_name,
                tuple(spec for spec in specs if spec in existing),
                force=force,
            )
            for group, specs in (("kv", schema.KV_TABLES), ("history", schema.HISTORY_TABLES))
            if specs[0] in existing
        }

    def _backend(self) -> Any:
        from .store import _DeltaBackend

        return _DeltaBackend(self._spark, self.table_name, full_text_search=False)

    # ---- restore recovery ------------------------------------------------

    def recover_restore(self) -> dict[str, dict[str, Any]]:
        """Settle every interrupted ``KVStore.restore`` run.

        Per run:

        * every table journaled ``overwritten`` → the restore did finish
          (only its completion row was lost): marked complete, nothing undone;
        * otherwise only the tables the run overwrote are rolled back, and
          only those still at the version the run left them at; a table
          written to since is listed under ``needs_manual`` instead (RESTORE
          would erase those later commits). Untouched tables are never
          restored.

        Returns ``{restore_id: {"completed", "rolled_back", "needs_manual"}}``.
        Rerun the restore afterwards if its snapshot is still wanted.
        """
        backend = self._backend()
        outcomes: dict[str, dict[str, Any]] = {}
        for restore_id, run in backend.unfinished_restores().items():
            outcomes[restore_id] = backend.settle_restore(restore_id, run)
            logger.info(f"recover_restore: {restore_id} -> {outcomes[restore_id]}")
        return outcomes

    def rederive_columns(self) -> dict[str, int]:
        """Recompute the derived columns (thread_id, namespace, scope_id, …)
        of every KV-routed row from its key and value. Run once after an
        upgrade that changed a derivation rule; values are untouched."""
        return self._backend().rederive()

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
        # The legacy table is usually the base name itself: its derived
        # tables carry suffixes, so the two never collide.
        legacy = _quoted_table_name(legacy_table)
        backend = self._backend()
        if backend.count() and not force:
            raise RuntimeError(
                f"memory tables under {self.table_name!r} already hold rows; "
                "pass force=True to merge the legacy rows into them"
            )
        copied: dict[str, int] = {}
        batch: list[tuple[Any, ...]] = []
        chat_threads: set[str] = set()

        def flush() -> None:
            if not batch:
                return
            for row in batch:
                suffix = backend._route(row[0]).suffix
                copied[suffix] = copied.get(suffix, 0) + 1
                if row[0].startswith("chat::"):
                    chat_threads.add(row[0][len("chat::") :].rsplit("::", 1)[0])
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
        # Legacy messages have no chat_id stamp: stamp them now, from each
        # chat's recorded key range, so chats keep their messages.
        from .store import KVChatMessageHistory, KVStore

        store = KVStore(self._spark, self.table_name, full_text_search=False)
        for thread_id in sorted(chat_threads):
            KVChatMessageHistory(thread_id, store).backfill_legacy_chats()
        schema.set_table_property(
            self._spark,
            schema.MESSAGES.name(self.table_name),
            schema.MIGRATED_PROPERTY,
            legacy_table,
        )
        logger.info(f"migrate_legacy: {legacy_table} -> {self.table_name}_*: {copied}")
        return copied


def _now() -> float:
    return time.time()
