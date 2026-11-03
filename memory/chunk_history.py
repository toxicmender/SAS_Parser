"""Original-vs-converted chunk history. See memory/README.md.

Every LLM attempt at a work item is recorded — the accepted one and every
attempt inline validation rejected — together with the ORIGINAL SAS text of
each member chunk and the CONVERTED code the model produced for it. Nothing
here is ever updated or deleted: the history is the reference record of how
each chunk was translated, independent of the chat history (which keeps only
the accepted turn) and of retention pruning.

Storage (Delta mode, see :mod:`memory.schema`):

* ``B_chunk_sources`` — each distinct original chunk, stored once, keyed by
  ``chunk_key = sha256(source_id, text)``; a MERGE that only ever inserts.
* ``B_item_attempts`` — one row per attempt: full response, structured
  document, verdict, ``is_final``. Blind ``INSERT``.
* ``B_chunk_conversions`` — one row per (attempt, member chunk) with the
  chunk's converted code. Blind ``INSERT``.
* ``B_chunk_history`` — the view that puts original and converted chunk,
  response and verdict on one row.

Locally (no ``base`` table) the same rows live in process memory, so the API
and its results are identical without Spark.

Logger name: ``memory.chunk_history``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional, Sequence

from . import schema

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)

HistoryField = Literal["original", "converted", "response"]

#: Search field -> (view column, base table spec whose index covers it).
_FIELDS: Dict[str, str] = {
    "original": "original_text",
    "converted": "converted_text",
    "response": "response",
}

#: Columns of one history row, in view order.
HISTORY_COLUMNS: tuple[str, ...] = (
    "attempt_id", "thread_id", "chat_id", "item_id", "attempt", "is_final",
    "chunk_id", "chunk_key", "member_index", "source_id", "kind", "title",
    "start_line", "end_line", "text_sha256", "original_text", "converted_text",
    "response", "document", "feedback", "target_language", "model",
    "validation_score", "validation_passed", "created_at", "archived_at",
)

_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
# A word is a run of letters (the split tokenizer's \p{L}\p{M}, minus marks).
_WORD = re.compile(r"[^\W\d_]+")


def _sha256(*parts: Optional[str]) -> str:
    material = "\0".join(p or "" for p in parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _metadata_json(metadata: Any) -> Optional[str]:
    dump = getattr(metadata, "model_dump_json", None)
    return str(dump()) if callable(dump) else None


def chunk_key(source_id: Optional[str], text: str) -> str:
    """Stable identity of one original chunk: its file plus its exact text.

    Content-addressed, so re-chunking a file (new chunk ids, same text) or
    re-running it maps onto the row already stored instead of a duplicate.
    """
    return _sha256(source_id or "<inline>", text)


def converted_text_for(
    chunk_id: str,
    *,
    document: Optional[Dict[str, Any]],
    response: str,
    single_member: bool,
) -> Optional[str]:
    """The converted code attributable to *chunk_id*, or ``None``.

    Structured output attributes cells to chunks via ``TranslationCell.chunk_id``;
    those code cells, in order, are the chunk's conversion. A single-chunk
    item needs no attribution — the whole item's code is that chunk's (from
    the document's code cells, else the response's fenced blocks, else the
    response itself). A multi-chunk item whose model attributed nothing to
    this chunk records ``None`` rather than guessing; the item's full
    response is still stored on the attempt row.
    """
    cells = document.get("cells") if isinstance(document, dict) else None
    code_cells = [
        c for c in cells or [] if isinstance(c, dict) and c.get("kind", "code") == "code"
    ]
    attributed = [str(c.get("source", "")) for c in code_cells if c.get("chunk_id") == chunk_id]
    if attributed:
        return "\n\n".join(attributed)
    if not single_member:
        return None
    if code_cells:
        return "\n\n".join(str(c.get("source", "")) for c in code_cells)
    fenced = [block.rstrip("\n") for block in _FENCE.findall(response or "")]
    if fenced:
        return "\n\n".join(fenced)
    return response or None


@dataclass(frozen=True)
class ArchiveResult:
    """What one :meth:`ChunkHistory.archive` call moved out of the active
    tables (``attempt_ids`` lists the archived attempts)."""

    attempts: int = 0
    conversions: int = 0
    attempt_ids: tuple[str, ...] = ()


def _current_ids(attempts: Iterable[Dict[str, Any]]) -> set[str]:
    """Attempt ids of the latest accepted attempt per (thread_id, item_id)."""
    latest: Dict[tuple[Any, Any], tuple[float, int, str]] = {}
    for a in attempts:
        if not a["is_final"]:
            continue
        rank = (a["created_at"] or 0.0, a["attempt"] or 0, a["attempt_id"])
        key = (a["thread_id"], a["item_id"])
        if key not in latest or rank > latest[key]:
            latest[key] = rank
    return {rank[2] for rank in latest.values()}


def _current(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rows of the latest accepted attempt per (thread_id, item_id)."""
    keep = _current_ids(rows)
    return [r for r in rows if r["attempt_id"] in keep]


def _matches(
    texts: Sequence[Optional[str]], pattern: str, *, case_sensitive: bool, mode: str
) -> bool:
    """Python mirror of ``search``/``isearch`` (non-null targets only)."""
    present = [t for t in texts if t is not None]
    if not present:
        return False
    if not case_sensitive:
        pattern = pattern.lower()
        present = [t.lower() for t in present]
    if mode == "substring":
        return any(pattern in t for t in present)
    words = set(_WORD.findall(pattern))
    if not words:
        return False
    tokens: set[str] = set()
    for t in present:
        tokens.update(_WORD.findall(t))
    return words <= tokens


class ChunkHistory:
    """Append-only record of original chunks and their conversions.

    Parameters
    ----------
    spark, base :
        Delta mode when *base* (the memory base table name, as for
        :class:`memory.store.KVStore`) is given; process-local otherwise.
    full_text_search :
        Delta mode only. Turn on predictive optimization and create the
        full-text search indexes (Databricks DBR 18.3+; this project targets
        DBR 19), and use ``search()``/``isearch()`` for :meth:`search`.
    """

    def __init__(
        self,
        spark: "SparkSession | None" = None,
        base: Optional[str] = None,
        *,
        full_text_search: bool = False,
        max_write_retries: int = 3,
    ) -> None:
        self._base = base
        self._max_write_retries = max_write_retries
        self.search_functions = False
        self.provision: schema.ProvisionReport | None = None
        self._sources: Dict[str, Dict[str, Any]] = {}
        self._attempts: List[Dict[str, Any]] = []
        self._conversions: List[Dict[str, Any]] = []
        self._attempts_archive: List[Dict[str, Any]] = []
        self._conversions_archive: List[Dict[str, Any]] = []
        # Rows recorded with buffer=True, written together by flush().
        self._pending_sources: List[Dict[str, Any]] = []
        self._pending_attempts: List[Dict[str, Any]] = []
        self._pending_conversions: List[Dict[str, Any]] = []
        # chunk_keys this instance has already stored: their insert-only
        # MERGE is skipped, so a retry or a re-run writes two INSERTs only.
        self._known_sources: set[str] = set()
        self._spark = spark
        if base is None:
            return
        if spark is None:
            raise ValueError(f"Delta chunk history (base={base!r}) requires a SparkSession")
        tables = schema.HISTORY_TABLES + schema.ARCHIVE_TABLES
        self._names = {s.suffix: s.name(base) for s in tables}
        self._quoted = {k: schema.quoted_table_name(v) for k, v in self._names.items()}
        self._view = schema.quoted_table_name(schema.chunk_history_view_name(base))
        self._archive_view = schema.quoted_table_name(
            schema.chunk_history_view_name(base, archived=True)
        )
        schema.ensure_tables(spark, base, tables)
        spark.sql(schema.chunk_history_view_ddl(base))
        spark.sql(schema.chunk_history_view_ddl(base, archived=True))
        if full_text_search:
            self.provision = schema.enable_databricks_features(
                spark, base, schema.HISTORY_TABLES
            )
            if self.provision.failures:
                logger.warning(
                    "chunk history %s: Databricks features unavailable (%d "
                    "statement(s) failed; first: %s)",
                    base,
                    len(self.provision.failures),
                    self.provision.failures[0],
                )
            self.search_functions = self._probe_search_functions()

    @property
    def is_delta(self) -> bool:
        return self._base is not None

    # ---- writes ------------------------------------------------------------

    def record_attempt(
        self,
        item: Any,
        response: str,
        document: Optional[Dict[str, Any]] = None,
        *,
        thread_id: str,
        attempt: int,
        is_final: bool,
        chat_id: Optional[str] = None,
        validation_score: Optional[float] = None,
        validation_passed: Optional[bool] = None,
        target_language: Optional[str] = None,
        model: Optional[str] = None,
        feedback: Optional[str] = None,
        buffer: bool = False,
    ) -> str:
        """Record one attempt at *item* (a :class:`chunker.models.SasBatch`).

        Writes the attempt row, one conversion row per member chunk, and any
        member chunk's original text not stored yet. *feedback* is the
        corrective note prompted with this attempt — the previous attempt's
        failed verdict — so a retry's history says why it happened.

        With ``buffer=True`` the rows are held until the next unbuffered
        call or :meth:`flush`, so an item's rejected attempts and its final
        one land in one write (three Delta commits per item rather than per
        attempt). Buffered rows are not readable until flushed. Returns the
        attempt id.
        """
        now = time.time()
        attempt_id = uuid.uuid4().hex
        chunks = list(getattr(item, "chunks", []) or [])
        item_id = str(getattr(item, "batch_id"))
        single = len(chunks) == 1
        sources: List[Dict[str, Any]] = []
        conversions: List[Dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            key = chunk_key(chunk.source_id, chunk.text)
            metadata = getattr(chunk, "metadata", None)
            sources.append(
                {
                    "chunk_key": key,
                    "text_sha256": _sha256(chunk.text),
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "kind": getattr(chunk.kind, "value", chunk.kind),
                    "title": chunk.title,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "parent_id": chunk.parent_id,
                    "original_text": chunk.text,
                    "metadata": _metadata_json(metadata),
                    "first_seen_at": now,
                }
            )
            conversions.append(
                {
                    "attempt_id": attempt_id,
                    "thread_id": thread_id,
                    "item_id": item_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_key": key,
                    "member_index": index,
                    "converted_text": converted_text_for(
                        chunk.chunk_id,
                        document=document,
                        response=response,
                        single_member=single,
                    ),
                    "is_final": bool(is_final),
                    "created_at": now,
                }
            )
        attempt_row = {
            "attempt_id": attempt_id,
            "thread_id": thread_id,
            "chat_id": chat_id,
            "item_id": item_id,
            "attempt": int(attempt),
            "is_final": bool(is_final),
            "target_language": target_language,
            "model": model,
            "response": response,
            "document": json.dumps(document, default=str) if document is not None else None,
            "validation_score": validation_score,
            "validation_passed": validation_passed,
            "created_at": now,
            "feedback": feedback or None,
        }
        self._pending_sources.extend(sources)
        self._pending_attempts.append(attempt_row)
        self._pending_conversions.extend(conversions)
        if not buffer:
            self.flush()
        logger.debug(
            f"record_attempt: thread={thread_id} item={item_id} attempt={attempt} "
            f"final={is_final} chunks={len(chunks)}"
        )
        return attempt_id

    def flush(self) -> int:
        """Write every buffered attempt; returns how many were written.

        The buffer is emptied even when the write fails, so a broken store
        cannot make the next item re-send this one's rows.
        """
        if not self._pending_attempts:
            return 0
        sources = list({r["chunk_key"]: r for r in self._pending_sources}.values())
        attempts, conversions = self._pending_attempts, self._pending_conversions
        self._pending_sources, self._pending_attempts, self._pending_conversions = [], [], []
        if self.is_delta:
            self._write_delta(sources, attempts, conversions)
        else:
            for row in sources:
                self._sources.setdefault(row["chunk_key"], row)
            self._attempts.extend(attempts)
            self._conversions.extend(conversions)
        return len(attempts)

    @property
    def pending(self) -> int:
        """Attempts recorded with ``buffer=True`` and not yet flushed."""
        return len(self._pending_attempts)

    # ---- archival ----------------------------------------------------------

    def archive(
        self,
        *,
        before_ts: Optional[float] = None,
        thread_id: Optional[str] = None,
        superseded_only: bool = False,
        keep_current: bool = True,
    ) -> ArchiveResult:
        """Move stale attempts out of the active tables into the archive.

        Selection is per whole attempt (its ``item_attempts`` row and every
        ``chunk_conversions`` row of it):

        * ``before_ts`` — recorded strictly before this Unix time;
        * ``superseded_only`` — only attempts that no longer stand: rejected
          attempts, and accepted ones a later accepted attempt replaced;
        * ``thread_id`` — limit to one thread;
        * ``keep_current`` (default) — the current accepted attempt of each
          (thread, item) is never archived, however old.

        At least one of ``before_ts`` / ``superseded_only`` is required, so a
        bare call cannot empty the active tables. Rows are copied to the
        archive *before* they are deleted from the active tables, and both
        steps are idempotent: an interrupted run leaves at most a duplicate
        (reads show the active copy) and a rerun completes it. Original
        chunks are never archived — both views join the same
        ``chunk_sources`` row. Archived rows stay readable through
        ``include_archived=True``.
        """
        if before_ts is None and not superseded_only:
            raise ValueError("archive needs before_ts and/or superseded_only=True")
        now = time.time()
        if not self.is_delta:
            current = _current_ids(self._attempts)
            selected = {
                a["attempt_id"]
                for a in self._attempts
                if (thread_id is None or a["thread_id"] == thread_id)
                and (before_ts is None or (a["created_at"] or 0.0) < before_ts)
                and not (superseded_only and a["attempt_id"] in current)
                and not (keep_current and a["attempt_id"] in current)
            }
            moved_attempts = [a for a in self._attempts if a["attempt_id"] in selected]
            moved_convs = [c for c in self._conversions if c["attempt_id"] in selected]
            self._attempts_archive.extend({**a, "archived_at": now} for a in moved_attempts)
            self._conversions_archive.extend({**c, "archived_at": now} for c in moved_convs)
            self._attempts = [a for a in self._attempts if a["attempt_id"] not in selected]
            self._conversions = [
                c for c in self._conversions if c["attempt_id"] not in selected
            ]
            result = ArchiveResult(
                attempts=len(moved_attempts),
                conversions=len(moved_convs),
                attempt_ids=tuple(sorted(selected)),
            )
        else:
            result = self._archive_delta(
                before_ts=before_ts,
                thread_id=thread_id,
                superseded_only=superseded_only,
                keep_current=keep_current,
                now=now,
            )
        logger.info(
            f"archive: moved {result.attempts} attempt(s) / {result.conversions} "
            f"conversion row(s) to the archive  thread={thread_id or '*'}  "
            f"before_ts={before_ts}  superseded_only={superseded_only}"
        )
        return result

    def _archive_delta(
        self,
        *,
        before_ts: Optional[float],
        thread_id: Optional[str],
        superseded_only: bool,
        keep_current: bool,
        now: float,
    ) -> ArchiveResult:
        from pyspark.sql import Window
        from pyspark.sql import functions as F

        spark = self._spark_or_raise()
        attempts = spark.table(self._quoted["item_attempts"])
        if thread_id is not None:
            attempts = attempts.filter(F.col("thread_id") == thread_id)
        # "Current" = latest accepted attempt per (thread, item). Computed over
        # the whole item, not the thread_id/before_ts slice, so an old answer
        # that still stands is recognised as current.
        latest = Window.partitionBy("thread_id", "item_id").orderBy(
            F.col("created_at").desc(), F.col("attempt").desc(), F.col("attempt_id").desc()
        )
        current_ids = (
            spark.table(self._quoted["item_attempts"])
            .filter(F.col("is_final"))
            .withColumn("_rn", F.row_number().over(latest))
            .filter(F.col("_rn") == 1)
            .select("attempt_id")
        )
        selected = attempts
        if before_ts is not None:
            selected = selected.filter(F.col("created_at") < F.lit(float(before_ts)))
        if superseded_only or keep_current:
            selected = selected.join(current_ids, on="attempt_id", how="left_anti")
        # Collected: every later statement must act on the same fixed set,
        # whatever is written meanwhile (a new attempt changes "current").
        ids = sorted(r.attempt_id for r in selected.select("attempt_id").collect())
        if not ids:
            return ArchiveResult()

        def ids_view(action: Any) -> None:
            self._with_view(spark.createDataFrame([(i,) for i in ids], "attempt_id STRING"), action)

        conv_count = (
            spark.table(self._quoted["chunk_conversions"])
            .filter(F.col("attempt_id").isin(ids))
            .count()
        )
        stamp = repr(float(now))
        for suffix, keys in (
            ("item_attempts", ("attempt_id",)),
            ("chunk_conversions", ("attempt_id", "member_index")),
        ):
            source = self._quoted[suffix]
            archive = self._quoted[f"{suffix}_archive"]
            spec = schema.ITEM_ATTEMPTS if suffix == "item_attempts" else schema.CHUNK_CONVERSIONS
            cols = ", ".join(f"r.`{c}`" for c in spec.column_names)
            on = " AND ".join(f"t.`{k}` = s.`{k}`" for k in keys)
            ids_view(
                lambda view, source=source, archive=archive, cols=cols, on=on, suffix=suffix: self._retry(
                    f"{suffix} archive copy",
                    lambda: spark.sql(f"""
                        MERGE INTO {archive} AS t
                        USING (
                            SELECT {cols}, CAST({stamp} AS DOUBLE) AS archived_at
                            FROM {source} AS r JOIN {view} AS i ON r.attempt_id = i.attempt_id
                        ) AS s
                        ON {on}
                        WHEN NOT MATCHED THEN INSERT *
                    """),
                )
            )
        # Delete only after both copies landed; conversions first so a
        # half-finished delete never leaves an attempt without its rows.
        for suffix in ("chunk_conversions", "item_attempts"):
            source = self._quoted[suffix]
            ids_view(
                lambda view, source=source, suffix=suffix: self._retry(
                    f"{suffix} archive delete",
                    lambda: spark.sql(
                        f"MERGE INTO {source} AS t USING {view} AS s "
                        f"ON t.attempt_id = s.attempt_id WHEN MATCHED THEN DELETE"
                    ),
                )
            )
        return ArchiveResult(attempts=len(ids), conversions=conv_count, attempt_ids=tuple(ids))

    # ---- reads -------------------------------------------------------------

    def history(
        self,
        *,
        chunk_id: Optional[str] = None,
        chunk_key: Optional[str] = None,
        thread_id: Optional[str] = None,
        item_id: Optional[str] = None,
        final_only: bool = False,
        current_only: bool = False,
        include_archived: bool = False,
    ) -> List[Dict[str, Any]]:
        """History rows (original + converted chunk), oldest first.

        Every filter given must match. ``final_only`` keeps every attempt that
        was accepted when it ran. An item can be accepted more than once: a
        resume with validation retries re-translates an item whose stored
        verdict failed, and its earlier answer keeps ``is_final``.
        ``current_only`` therefore keeps just the most recent accepted attempt
        of each (thread, item) — the answer that stands now.
        ``include_archived`` also reads attempts :meth:`archive` moved out
        (their ``archived_at`` is set; active rows carry ``None``).
        """
        filters = {
            "chunk_id": chunk_id,
            "chunk_key": chunk_key,
            "thread_id": thread_id,
            "item_id": item_id,
        }
        rows = self._query(
            {k: v for k, v in filters.items() if v is not None},
            final_only or current_only,
            include_archived,
        )
        return _current(rows) if current_only else rows

    def for_item(
        self,
        thread_id: str,
        item_id: str,
        *,
        final_only: bool = False,
        current_only: bool = False,
        include_archived: bool = False,
    ) -> List[Dict[str, Any]]:
        """Every attempt's rows for one work item, oldest attempt first."""
        return self.history(
            thread_id=thread_id,
            item_id=item_id,
            final_only=final_only,
            current_only=current_only,
            include_archived=include_archived,
        )

    def search(
        self,
        pattern: str,
        *,
        field: HistoryField = "original",
        case_sensitive: bool = False,
        mode: schema.SearchMode = "substring",
        final_only: bool = False,
        include_archived: bool = False,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """History rows whose *field* contains *pattern*.

        ``field`` is ``"original"`` (the SAS chunk), ``"converted"`` (the
        chunk's converted code) or ``"response"`` (the item's full answer).
        Semantics follow Databricks ``search``/``isearch``: ``mode="word"``
        requires every word of the pattern, in any order. On Databricks the
        predicate is ``isearch``/``search`` itself, so the full-text search
        index on that column skips files that cannot match. Archived rows
        (``include_archived=True``) have no index and are scanned.
        """
        if field not in _FIELDS:
            raise ValueError(f"field must be one of {sorted(_FIELDS)}, got {field!r}")
        if mode not in ("substring", "word"):
            raise ValueError(f"mode must be 'substring' or 'word', got {mode!r}")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("pattern must be a non-empty string")
        if limit < 1:
            raise ValueError("limit must be >= 1")
        column = _FIELDS[field]
        if not self.is_delta:
            rows = [
                r
                for r in self._query({}, final_only, include_archived)
                if _matches([r[column]], pattern, case_sensitive=case_sensitive, mode=mode)
            ]
            return rows[:limit]
        df = self._view_df(final_only, include_archived)
        df = df.filter(self._search_predicate(column, pattern, case_sensitive, mode))
        return self._collect(df.orderBy("created_at", "attempt", "member_index").limit(limit))

    def sources(self, source_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """The distinct original chunks stored, optionally for one file."""
        if not self.is_delta:
            rows = [dict(r) for r in self._sources.values()]
        else:
            df = self._spark_or_raise().table(self._quoted["chunk_sources"])
            if source_id is not None:
                from pyspark.sql import functions as F

                df = df.filter(F.col("source_id") == source_id)
            rows = [r.asDict() for r in df.collect()]
        if source_id is not None:
            rows = [r for r in rows if r["source_id"] == source_id]
        rows.sort(key=lambda r: (r["source_id"] or "", r["start_char"] or 0, r["chunk_id"]))
        return rows

    def stats(self) -> Dict[str, Any]:
        if not self.is_delta:
            return {
                "backend": "in-memory",
                "chunk_sources": len(self._sources),
                "item_attempts": len(self._attempts),
                "chunk_conversions": len(self._conversions),
                "item_attempts_archive": len(self._attempts_archive),
                "chunk_conversions_archive": len(self._conversions_archive),
            }
        spark = self._spark_or_raise()
        return {
            "backend": f"Delta:{self._base}_chunk_*",
            **{
                suffix: spark.table(quoted).count()
                for suffix, quoted in self._quoted.items()
            },
            "search_functions": self.search_functions,
        }

    # ---- in-memory query ---------------------------------------------------

    def _joined(self, archived: bool = False) -> Iterable[Dict[str, Any]]:
        attempt_rows = self._attempts_archive if archived else self._attempts
        conversion_rows = self._conversions_archive if archived else self._conversions
        attempts = {a["attempt_id"]: a for a in attempt_rows}
        for conv in conversion_rows:
            att = attempts.get(conv["attempt_id"])
            src = self._sources.get(conv["chunk_key"])
            if att is None or src is None:
                continue
            yield self._shape(
                {
                    **conv,
                    "chat_id": att["chat_id"],
                    "attempt": att["attempt"],
                    "source_id": src["source_id"],
                    "kind": src["kind"],
                    "title": src["title"],
                    "start_line": src["start_line"],
                    "end_line": src["end_line"],
                    "text_sha256": src["text_sha256"],
                    "original_text": src["original_text"],
                    "response": att["response"],
                    "document": att["document"],
                    "feedback": att.get("feedback"),
                    "target_language": att["target_language"],
                    "model": att["model"],
                    "validation_score": att["validation_score"],
                    "validation_passed": att["validation_passed"],
                }
            )

    def _query(
        self, filters: Dict[str, Any], final_only: bool, include_archived: bool = False
    ) -> List[Dict[str, Any]]:
        if self.is_delta:
            from pyspark.sql import functions as F

            df = self._view_df(final_only, include_archived)
            for column, value in filters.items():
                df = df.filter(F.col(column) == value)
            return self._collect(df.orderBy("created_at", "attempt", "member_index"))
        joined: Iterable[Dict[str, Any]] = list(self._joined())
        if include_archived:
            seen = {(r["attempt_id"], r["member_index"]) for r in joined}
            joined = [
                *joined,
                *(
                    r
                    for r in self._joined(archived=True)
                    if (r["attempt_id"], r["member_index"]) not in seen
                ),
            ]
        rows = [
            r
            for r in joined
            if (not final_only or r["is_final"])
            and all(r[k] == v for k, v in filters.items())
        ]
        rows.sort(key=lambda r: (r["created_at"], r["attempt"], r["member_index"]))
        return rows

    @staticmethod
    def _shape(row: Dict[str, Any]) -> Dict[str, Any]:
        out = {column: row.get(column) for column in HISTORY_COLUMNS}
        doc = out.get("document")
        if isinstance(doc, str):
            try:
                out["document"] = json.loads(doc)
            except ValueError:
                pass
        return out

    # ---- Delta -------------------------------------------------------------

    def _spark_or_raise(self) -> "SparkSession":
        if self._spark is None:  # pragma: no cover - guarded by is_delta
            raise RuntimeError("chunk history has no SparkSession")
        return self._spark

    def _view_df(self, final_only: bool, include_archived: bool = False) -> "DataFrame":
        from pyspark.sql import functions as F

        spark = self._spark_or_raise()
        df = spark.table(self._view)
        if include_archived:
            # An archive run interrupted between its copy and its delete
            # leaves an attempt in both; the active copy wins.
            archived = spark.table(self._archive_view).join(
                df.select("attempt_id", "member_index"),
                on=["attempt_id", "member_index"],
                how="left_anti",
            )
            df = df.unionByName(archived)
        if final_only:
            df = df.filter(F.col("is_final"))
        return df

    def _collect(self, df: "DataFrame") -> List[Dict[str, Any]]:
        return [self._shape(r.asDict()) for r in df.select(*HISTORY_COLUMNS).collect()]

    def _search_predicate(
        self, column: str, pattern: str, case_sensitive: bool, mode: str
    ) -> Any:
        from pyspark.sql import functions as F

        if self.search_functions:
            return F.expr(
                schema.search_sql((column,), pattern, case_sensitive=case_sensitive, mode=mode)  # type: ignore[arg-type]
            )
        col = F.col(column)
        if mode == "substring":
            return (
                col.contains(pattern)
                if case_sensitive
                else F.lower(col).contains(pattern.lower())
            )
        words = _WORD.findall(pattern)
        if not words:
            return F.lit(False)
        flags = "" if case_sensitive else "(?i)"
        predicate = None
        for word in words:
            # Java regex: a whole word, bounded by non-letters.
            regex = rf"{flags}(?<![\p{{L}}\p{{M}}])\Q{word}\E(?![\p{{L}}\p{{M}}])"
            clause = col.rlike(regex)
            predicate = clause if predicate is None else predicate & clause
        return predicate

    def _probe_search_functions(self) -> bool:
        try:
            self._spark_or_raise().sql(
                "SELECT isearch('memory', 'mem', mode => 'substring') AS hit"
            ).collect()
            return True
        except Exception:
            logger.info("chunk history: search()/isearch() unavailable; using LIKE scans")
            return False

    def _retry(self, operation: str, action: Any) -> None:
        for attempt in range(self._max_write_retries + 1):
            try:
                action()
                return
            except Exception as exc:
                text = str(exc).upper()
                transient = "CONCURRENT" in text or "CONFLICT" in text
                if not transient or attempt == self._max_write_retries:
                    raise
                delay = 0.05 * (2**attempt)
                logger.warning(
                    "chunk history %s conflicted; retrying (%s/%s) in %.2fs",
                    operation, attempt + 1, self._max_write_retries, delay,
                )
                time.sleep(delay)

    def _staged(self, spec: schema.TableSpec, rows: List[Dict[str, Any]]) -> "DataFrame":
        ddl = ", ".join(f"`{n}` {t}" for n, t in spec.columns)
        return self._spark_or_raise().createDataFrame(
            [tuple(r.get(n) for n in spec.column_names) for r in rows], ddl
        )

    def _with_view(self, df: "DataFrame", action: Any) -> None:
        spark = self._spark_or_raise()
        view = f"_chunk_hist_{uuid.uuid4().hex}"
        df.createOrReplaceTempView(view)
        try:
            action(view)
        finally:
            spark.catalog.dropTempView(view)

    def _write_delta(
        self,
        sources: List[Dict[str, Any]],
        attempts: List[Dict[str, Any]],
        conversions: List[Dict[str, Any]],
    ) -> None:
        spark = self._spark_or_raise()
        staged = self._staged
        with_view = self._with_view

        # Written sources -> attempt -> conversions: the view inner-joins all
        # three, so a partially written attempt is simply not visible.
        unique = [r for r in sources if r["chunk_key"] not in self._known_sources]
        if unique:
            target = self._quoted["chunk_sources"]
            with_view(
                staged(schema.CHUNK_SOURCES, unique),
                lambda view: self._retry(
                    "chunk_sources MERGE",
                    lambda: spark.sql(
                        f"MERGE INTO {target} AS t USING {view} AS s "
                        f"ON t.chunk_key = s.chunk_key WHEN NOT MATCHED THEN INSERT *"
                    ),
                ),
            )
            self._known_sources.update(r["chunk_key"] for r in unique)
        for suffix, spec, rows in (
            ("item_attempts", schema.ITEM_ATTEMPTS, attempts),
            ("chunk_conversions", schema.CHUNK_CONVERSIONS, conversions),
        ):
            if not rows:
                continue
            target = self._quoted[suffix]
            cols = ", ".join(f"`{c}`" for c in spec.column_names)
            with_view(
                staged(spec, rows),
                lambda view, target=target, cols=cols, suffix=suffix: self._retry(
                    f"{suffix} INSERT",
                    lambda: spark.sql(
                        f"INSERT INTO {target} ({cols}) SELECT {cols} FROM {view}"
                    ),
                ),
            )
