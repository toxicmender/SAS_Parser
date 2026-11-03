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
    "response", "document", "target_language", "model", "validation_score",
    "validation_passed", "created_at",
)

_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
# A word is a run of letters (the split tokenizer's \p{L}\p{M}, minus marks).
_WORD = re.compile(r"[^\W\d_]+")


def _sha256(*parts: Optional[str]) -> str:
    material = "\0".join(p or "" for p in parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _metadata_json(metadata: Any) -> Optional[str]:
    dump = getattr(metadata, "model_dump_json", None)
    return dump() if callable(dump) else None


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
        self._spark = spark
        if base is None:
            return
        if spark is None:
            raise ValueError(f"Delta chunk history (base={base!r}) requires a SparkSession")
        self._names = {s.suffix: s.name(base) for s in schema.HISTORY_TABLES}
        self._quoted = {k: schema.quoted_table_name(v) for k, v in self._names.items()}
        self._view = schema.quoted_table_name(schema.chunk_history_view_name(base))
        schema.ensure_tables(spark, base, schema.HISTORY_TABLES)
        spark.sql(schema.chunk_history_view_ddl(base))
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
    ) -> str:
        """Record one attempt at *item* (a :class:`chunker.models.SasBatch`).

        Writes the attempt row, one conversion row per member chunk, and any
        member chunk's original text not stored yet. Returns the attempt id.
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
        }
        if self.is_delta:
            self._write_delta(sources, attempt_row, conversions)
        else:
            for row in sources:
                self._sources.setdefault(row["chunk_key"], row)
            self._attempts.append(attempt_row)
            self._conversions.extend(conversions)
        logger.debug(
            f"record_attempt: thread={thread_id} item={item_id} attempt={attempt} "
            f"final={is_final} chunks={len(chunks)}"
        )
        return attempt_id

    # ---- reads -------------------------------------------------------------

    def history(
        self,
        *,
        chunk_id: Optional[str] = None,
        chunk_key: Optional[str] = None,
        thread_id: Optional[str] = None,
        item_id: Optional[str] = None,
        final_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """History rows (original + converted chunk), oldest first.

        Every filter given must match; ``final_only`` keeps only accepted
        attempts.
        """
        filters = {
            "chunk_id": chunk_id,
            "chunk_key": chunk_key,
            "thread_id": thread_id,
            "item_id": item_id,
        }
        return self._query({k: v for k, v in filters.items() if v is not None}, final_only)

    def for_item(
        self, thread_id: str, item_id: str, *, final_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Every attempt's rows for one work item, oldest attempt first."""
        return self.history(thread_id=thread_id, item_id=item_id, final_only=final_only)

    def search(
        self,
        pattern: str,
        *,
        field: HistoryField = "original",
        case_sensitive: bool = False,
        mode: schema.SearchMode = "substring",
        final_only: bool = False,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """History rows whose *field* contains *pattern*.

        ``field`` is ``"original"`` (the SAS chunk), ``"converted"`` (the
        chunk's converted code) or ``"response"`` (the item's full answer).
        Semantics follow Databricks ``search``/``isearch``: ``mode="word"``
        requires every word of the pattern, in any order. On Databricks the
        predicate is ``isearch``/``search`` itself, so the full-text search
        index on that column skips files that cannot match.
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
                for r in self._query({}, final_only)
                if _matches([r[column]], pattern, case_sensitive=case_sensitive, mode=mode)
            ]
            return rows[:limit]
        df = self._view_df(final_only)
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

    def _joined(self) -> Iterable[Dict[str, Any]]:
        attempts = {a["attempt_id"]: a for a in self._attempts}
        for conv in self._conversions:
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
                    "target_language": att["target_language"],
                    "model": att["model"],
                    "validation_score": att["validation_score"],
                    "validation_passed": att["validation_passed"],
                }
            )

    def _query(self, filters: Dict[str, Any], final_only: bool) -> List[Dict[str, Any]]:
        if self.is_delta:
            from pyspark.sql import functions as F

            df = self._view_df(final_only)
            for column, value in filters.items():
                df = df.filter(F.col(column) == value)
            return self._collect(df.orderBy("created_at", "attempt", "member_index"))
        rows = [
            r
            for r in self._joined()
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

    def _view_df(self, final_only: bool) -> "DataFrame":
        from pyspark.sql import functions as F

        df = self._spark_or_raise().table(self._view)
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

    def _write_delta(
        self,
        sources: List[Dict[str, Any]],
        attempt_row: Dict[str, Any],
        conversions: List[Dict[str, Any]],
    ) -> None:
        spark = self._spark_or_raise()

        def staged(spec: schema.TableSpec, rows: List[Dict[str, Any]]) -> "DataFrame":
            ddl = ", ".join(f"`{n}` {t}" for n, t in spec.columns)
            return spark.createDataFrame(
                [tuple(r[n] for n in spec.column_names) for r in rows], ddl
            )

        def with_view(df: "DataFrame", action: Any) -> None:
            view = f"_chunk_hist_{uuid.uuid4().hex}"
            df.createOrReplaceTempView(view)
            try:
                action(view)
            finally:
                spark.catalog.dropTempView(view)

        # Written sources -> attempt -> conversions: the view inner-joins all
        # three, so a partially written attempt is simply not visible.
        if sources:
            unique = list({r["chunk_key"]: r for r in sources}.values())
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
        for suffix, spec, rows in (
            ("item_attempts", schema.ITEM_ATTEMPTS, [attempt_row]),
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
