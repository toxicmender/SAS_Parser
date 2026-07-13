"""Spark-backed persistence of validation reports.

One :func:`log_report` call appends one row per (case, metric) to a Spark
target, so run history accumulates queryably over time — trend a metric
across runs with a plain ``GROUP BY``. Two targets, mirroring
``memory.short_mem``'s local/Databricks split:

- **path** (local default): a parquet directory, ``./validation_runs`` unless
  overridden — no server, no service, nothing but pyspark (already a core
  dependency).
- **table** (production): a saved table name, e.g.
  ``catalog.schema.validation_runs`` — Delta on Databricks.

Spark is booted lazily inside :func:`log_report` / :func:`load_runs` only
(never at import, never by the metrics/runner — the repo invariant that
nothing touches Spark unless persistence is actually asked for). Pass an
existing ``spark`` session on Databricks; locally one is created on demand.

Resolution precedence for the target follows the repo-wide rule
(see the ``app_config`` package): explicit argument > config.json
(``validation.table`` / ``validation.path``) > the local parquet default.
A configured/explicit ``table`` always wins over ``path``.

Logger name: ``validation.tracking``.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import app_config

from .models import ValidationReport

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)

DEFAULT_PATH = "validation_runs"

# One row per (run, case, metric); run- and case-level values are repeated on
# each row so any slice of the table is self-describing.
_SCHEMA_DDL = (
    "run_id string, logged_at timestamp, model string, "
    "instructions_fingerprint string, run_score double, "
    "run_passed boolean, case_count int, case_id string, case_score double, "
    "case_passed boolean, item_count int, metric string, score double, "
    "threshold double, passed boolean, skipped boolean, details string"
)


def _ensure_spark(spark: "SparkSession | None") -> "SparkSession":
    if spark is not None:
        return spark
    from pyspark.sql import SparkSession

    logger.info("_ensure_spark: no SparkSession provided, starting local one")
    return (
        SparkSession.builder.master("local[*]")
        .appName("validation_tracking")
        .getOrCreate()
    )


def _resolve_target(table: str | None, path: str | None) -> tuple[str, str]:
    """``("table", name)`` or ``("path", dir)`` after app_config resolution."""
    resolved_table = app_config.resolve(table, "validation", "table", None)
    if resolved_table is not None:
        return "table", resolved_table
    return "path", app_config.resolve(path, "validation", "path", DEFAULT_PATH)


def _report_rows(
    report: ValidationReport, run_id: str, logged_at: datetime
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in report.results:
        for m in result.metrics:
            rows.append(
                {
                    "run_id": run_id,
                    "logged_at": logged_at,
                    "model": report.model,
                    "instructions_fingerprint": report.instructions_fingerprint,
                    "run_score": report.score,
                    "run_passed": report.passed,
                    "case_count": len(report.results),
                    "case_id": result.case_id,
                    "case_score": result.score,
                    "case_passed": result.passed,
                    "item_count": result.item_count,
                    "metric": m.metric,
                    "score": m.score,
                    "threshold": m.threshold,
                    "passed": m.passed,
                    "skipped": m.skipped,
                    "details": m.details,
                }
            )
    return rows


def log_report(
    report: ValidationReport,
    *,
    spark: "SparkSession | None" = None,
    table: str | None = None,
    path: str | None = None,
) -> str:
    """
    Append *report* to the Spark target and return the generated run id.

    Parameters
    ----------
    spark : SparkSession | None
        Existing session (pass the Databricks one in production). ``None``
        (default) starts a local session on first use.
    table : str | None
        Saved-table target, e.g. ``"catalog.schema.validation_runs"``.
        ``None`` reads config.json ``validation.table``; when that is also
        unset, the parquet ``path`` target is used instead.
    path : str | None
        Parquet-directory target. ``None`` reads config.json
        ``validation.path``, falling back to ``./validation_runs``.
    """
    run_id = uuid.uuid4().hex
    logged_at = datetime.now(timezone.utc)
    rows = _report_rows(report, run_id, logged_at)
    if not rows:
        logger.warning(f"log_report: empty report, nothing to log  run_id={run_id}")
        return run_id

    kind, target = _resolve_target(table, path)
    logger.info(
        f"log_report: run_id={run_id}  rows={len(rows)}  {kind}='{target}'"
    )
    session = _ensure_spark(spark)
    df = session.createDataFrame(rows, schema=_SCHEMA_DDL)
    writer = df.write.mode("append")
    if kind == "table":
        writer.saveAsTable(target)
    else:
        writer.parquet(target)
    logger.info(f"log_report: logged run '{run_id}'")
    return run_id


def load_runs(
    *,
    spark: "SparkSession | None" = None,
    table: str | None = None,
    path: str | None = None,
) -> "DataFrame":
    """
    The accumulated run history as a Spark DataFrame (same target resolution
    as :func:`log_report`), e.g.::

        load_runs().groupBy("run_id", "metric").avg("score")
    """
    kind, target = _resolve_target(table, path)
    logger.debug(f"load_runs: {kind}='{target}'")
    session = _ensure_spark(spark)
    if kind == "table":
        return session.read.table(target)
    return session.read.parquet(target)
