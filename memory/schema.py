"""Delta table layout for the LLM memory store. See memory/README.md.

One configured *base* name ``B`` (``memory.delta_table``) fans out into a
family of purpose-built Delta tables in the same catalog and schema:

========================  ====================================================
``B_messages``            chat messages (``msg::``) — append-only INSERTs
``B_chats``               chat index rows (``chat::``)
``B_run_items``           per-item run facts (``kv::run::``)
``B_validations``         per-item validation verdicts (``kv::validation::``)
``B_kv``                  everything else (policies, thread notes, summaries,
                          plain KV entries)
``B_chunk_sources``       each distinct ORIGINAL SAS chunk, stored once
``B_item_attempts``       every LLM attempt for a work item (full response)
``B_chunk_conversions``   every attempt's CONVERTED code, per member chunk
``B_chunk_history``       VIEW joining the three above: original and
                          converted chunk on one row, every attempt kept
========================  ====================================================

Every table is created with row tracking (a requirement of Databricks
full-text search indexes) and liquid clustering on its lookup keys, and only
the tables a Change Data Feed consumer reads carry CDF. On Databricks
(DBR 18.3+, targeted at DBR 19) :func:`enable_databricks_features` also turns
on predictive optimization — which is what keeps a search index fresh — and
creates the full-text search indexes listed per table. Neither exists in OSS
Delta; there the failure is logged once and searches fall back to a plain
``LIKE`` scan with identical results.

No pyspark import here: this module only builds SQL text, so it is importable
(and unit-testable) without a JVM.

Logger name: ``memory.schema``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


_IDENTIFIER_PART = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def split_table_name(table: str) -> list[str]:
    """Validate a one- to three-part identifier and return its parts."""
    if not isinstance(table, str) or not table:
        raise ValueError("Delta table must be a non-empty identifier")
    parts = table.split(".")
    if not 1 <= len(parts) <= 3 or any(not _IDENTIFIER_PART.fullmatch(p) for p in parts):
        raise ValueError(
            "Delta table must be a one- to three-part identifier containing "
            "only letters, numbers, and underscores"
        )
    return parts


def quoted_table_name(table: str) -> str:
    """Return a SQL-safe one- to three-part table identifier.

    Spark SQL parameter markers cannot represent identifiers.  Memory table
    names are configuration, rather than data, but accepting arbitrary SQL in
    that configuration would still make every DDL/DML call injectable.  Keep
    the production contract deliberately narrow: Unity Catalog identifiers
    are one, two, or three ordinary identifier parts.
    """
    return ".".join(f"`{part}`" for part in split_table_name(table))


def sql_string_literal(value: str) -> str:
    """Render *value* as a single-quoted Spark SQL string literal.

    ``search()`` / ``isearch()`` require their pattern to be a *literal* —
    parameter markers are rejected — so the pattern has to be spelled into the
    statement. Spark SQL string literals honour backslash escapes, so both the
    backslash and the quote are escaped; control characters are escaped too so
    a pattern can never terminate the literal or the statement.
    """
    if not isinstance(value, str):
        raise TypeError("SQL string literal must be a str")
    out = (
        value.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace("\0", "")
    )
    return f"'{out}'"


SearchMode = Literal["substring", "word"]


def search_sql(
    columns: list[str] | tuple[str, ...],
    pattern: str,
    *,
    case_sensitive: bool = False,
    mode: SearchMode = "substring",
) -> str:
    """A ``search``/``isearch`` predicate over *columns* for *pattern*.

    The function is chosen by *case_sensitive*; *mode* ``"word"`` matches when
    every word of the pattern appears in at least one target, in any order.
    Column names come from this module's own table specs, never from callers.
    """
    if mode not in ("substring", "word"):
        raise ValueError(f"mode must be 'substring' or 'word', got {mode!r}")
    for column in columns:
        if not _IDENTIFIER_PART.fullmatch(column):
            raise ValueError(f"invalid column name {column!r}")
    func = "search" if case_sensitive else "isearch"
    targets = ", ".join(f"`{c}`" for c in columns)
    return f"{func}({targets}, {sql_string_literal(pattern)}, mode => '{mode}')"


# ---------------------------------------------------------------------------
# Table specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchIndex:
    """One full-text search index (``CREATE SEARCH INDEX``) on one column.

    ``tokenizer="ngram"`` serves substring lookups (``size`` = ``ngram_size``,
    3..10); ``tokenizer="split"`` serves whole-word lookups (``size`` =
    ``min_token_length``).
    """

    column: str
    tokenizer: Literal["ngram", "split"] = "ngram"
    size: int = 5

    def options(self) -> str:
        if self.tokenizer == "ngram":
            return f"tokenizer = 'ngram', ngram_size = {int(self.size)}"
        return f"tokenizer = 'split', min_token_length = {int(self.size)}"


@dataclass(frozen=True)
class TableSpec:
    """DDL contract for one table of the memory family."""

    suffix: str
    columns: tuple[tuple[str, str], ...]
    cluster_by: tuple[str, ...]
    not_null: tuple[str, ...] = ()
    cdf: bool = False
    search_indexes: tuple[SearchIndex, ...] = ()
    comment: str = ""

    def __post_init__(self) -> None:
        if len(self.search_indexes) > 4:
            raise ValueError("Databricks allows at most four search indexes per table")
        indexed = [i.column for i in self.search_indexes]
        if len(set(indexed)) != len(indexed):
            raise ValueError("each search index must cover a different column")

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.columns)

    def name(self, base: str) -> str:
        return f"{base}_{self.suffix}"

    def ddl(self, base: str) -> str:
        cols = ",\n    ".join(
            f"`{name}` {sql_type}{' NOT NULL' if name in self.not_null else ''}"
            for name, sql_type in self.columns
        )
        props = ["'delta.enableRowTracking' = 'true'"]
        if self.cdf:
            props.append("'delta.enableChangeDataFeed' = 'true'")
        cluster = ", ".join(f"`{c}`" for c in self.cluster_by)
        comment = f"\nCOMMENT {sql_string_literal(self.comment)}" if self.comment else ""
        return (
            f"CREATE TABLE IF NOT EXISTS {quoted_table_name(self.name(base))} (\n"
            f"    {cols}\n)\nUSING DELTA\nCLUSTER BY ({cluster}){comment}\n"
            f"TBLPROPERTIES ({', '.join(props)})"
        )

    def index_name(self, base: str, column: str) -> str:
        """``<catalog>.<schema>.<table>_<column>_fts`` — same schema as the
        table (a Databricks requirement) and never equal to a table name."""
        parts = split_table_name(self.name(base))
        parts[-1] = f"{parts[-1]}_{column}_fts"
        return ".".join(f"`{p}`" for p in parts)

    def index_ddl(self, base: str) -> list[str]:
        table = quoted_table_name(self.name(base))
        return [
            f"CREATE SEARCH INDEX IF NOT EXISTS {self.index_name(base, i.column)} "
            f"ON {table} (`{i.column}`) OPTIONS ({i.options()})"
            for i in self.search_indexes
        ]


# The six columns every KV-routed table shares: the raw-record shape the
# KVStore façade speaks (memory.store._RawRecord). Typed, derived columns
# follow them so reads can always select the core in this order.
KV_CORE: tuple[tuple[str, str], ...] = (
    ("kv_key", "STRING"),
    ("value", "STRING"),
    ("tags", "STRING"),
    ("created_at", "DOUBLE"),
    ("updated_at", "DOUBLE"),
    ("source", "STRING"),
)
KV_CORE_NAMES: tuple[str, ...] = tuple(name for name, _ in KV_CORE)
_KV_NOT_NULL = ("kv_key", "value")


MESSAGES = TableSpec(
    suffix="messages",
    columns=KV_CORE
    + (
        ("thread_id", "STRING"),
        ("msg_tick", "BIGINT"),
        ("role", "STRING"),
        ("content", "STRING"),
    ),
    not_null=_KV_NOT_NULL,
    cluster_by=("thread_id", "msg_tick"),
    cdf=True,
    search_indexes=(
        SearchIndex("content", "ngram", 5),
        SearchIndex("value", "ngram", 5),
    ),
    comment="LLM chat messages; value = lossless LangChain message JSON, "
    "content = its plain text. Append-only.",
)

CHATS = TableSpec(
    suffix="chats",
    columns=KV_CORE
    + (
        ("thread_id", "STRING"),
        ("chat_id", "STRING"),
        ("label", "STRING"),
        ("first_key", "STRING"),
        ("last_key", "STRING"),
        ("message_count", "BIGINT"),
    ),
    not_null=_KV_NOT_NULL,
    cluster_by=("thread_id",),
    cdf=True,
    comment="One row per chat (one pipeline instance's span of a thread).",
)

RUN_ITEMS = TableSpec(
    suffix="run_items",
    columns=KV_CORE
    + (
        ("thread_id", "STRING"),
        ("item_id", "STRING"),
        ("status", "STRING"),
        ("idx", "BIGINT"),
        ("total", "BIGINT"),
        ("attempts", "BIGINT"),
        ("elapsed_s", "DOUBLE"),
    ),
    not_null=_KV_NOT_NULL,
    cluster_by=("thread_id", "item_id"),
    comment="Per-item run outcome facts (resume / audit).",
)

VALIDATIONS = TableSpec(
    suffix="validations",
    columns=KV_CORE
    + (
        ("thread_id", "STRING"),
        ("item_id", "STRING"),
        ("score", "DOUBLE"),
        ("passed", "BOOLEAN"),
        ("idx", "BIGINT"),
    ),
    not_null=_KV_NOT_NULL,
    cluster_by=("thread_id", "item_id"),
    comment="Per-item inline validation verdicts.",
)

KV = TableSpec(
    suffix="kv",
    columns=KV_CORE + (("namespace", "STRING"), ("scope_id", "STRING")),
    not_null=_KV_NOT_NULL,
    cluster_by=("namespace", "scope_id"),
    search_indexes=(SearchIndex("value", "ngram", 5),),
    comment="Low-volume memory records: task policies, thread notes, "
    "rolling summaries and plain KV entries.",
)

CHUNK_SOURCES = TableSpec(
    suffix="chunk_sources",
    columns=(
        ("chunk_key", "STRING"),
        ("text_sha256", "STRING"),
        ("chunk_id", "STRING"),
        ("source_id", "STRING"),
        ("kind", "STRING"),
        ("title", "STRING"),
        ("start_line", "BIGINT"),
        ("end_line", "BIGINT"),
        ("start_char", "BIGINT"),
        ("end_char", "BIGINT"),
        ("parent_id", "STRING"),
        ("original_text", "STRING"),
        ("metadata", "STRING"),
        ("first_seen_at", "DOUBLE"),
    ),
    not_null=("chunk_key", "text_sha256", "chunk_id", "original_text"),
    cluster_by=("source_id", "chunk_id"),
    search_indexes=(SearchIndex("original_text", "ngram", 4),),
    comment="Each distinct ORIGINAL SAS chunk, stored once "
    "(chunk_key = sha256(source_id, text)).",
)

ITEM_ATTEMPTS = TableSpec(
    suffix="item_attempts",
    columns=(
        ("attempt_id", "STRING"),
        ("thread_id", "STRING"),
        ("chat_id", "STRING"),
        ("item_id", "STRING"),
        ("attempt", "BIGINT"),
        ("is_final", "BOOLEAN"),
        ("target_language", "STRING"),
        ("model", "STRING"),
        ("response", "STRING"),
        ("document", "STRING"),
        ("validation_score", "DOUBLE"),
        ("validation_passed", "BOOLEAN"),
        ("created_at", "DOUBLE"),
    ),
    not_null=("attempt_id", "thread_id", "item_id", "attempt", "is_final"),
    cluster_by=("thread_id", "item_id"),
    search_indexes=(SearchIndex("response", "split", 3),),
    comment="Every LLM attempt for a work item, accepted or rejected. "
    "Append-only.",
)

CHUNK_CONVERSIONS = TableSpec(
    suffix="chunk_conversions",
    columns=(
        ("attempt_id", "STRING"),
        ("thread_id", "STRING"),
        ("item_id", "STRING"),
        ("chunk_id", "STRING"),
        ("chunk_key", "STRING"),
        ("member_index", "BIGINT"),
        ("converted_text", "STRING"),
        ("is_final", "BOOLEAN"),
        ("created_at", "DOUBLE"),
    ),
    not_null=("attempt_id", "thread_id", "item_id", "chunk_id", "chunk_key", "is_final"),
    cluster_by=("chunk_key", "thread_id"),
    search_indexes=(SearchIndex("converted_text", "ngram", 4),),
    comment="Every attempt's CONVERTED code for each member chunk. "
    "Append-only.",
)

#: Tables the KVStore façade routes into, most specific prefix first.
KV_ROUTES: tuple[tuple[str, TableSpec], ...] = (
    ("msg::", MESSAGES),
    ("chat::", CHATS),
    ("kv::run::", RUN_ITEMS),
    ("kv::validation::", VALIDATIONS),
)
KV_FALLBACK = KV

KV_TABLES: tuple[TableSpec, ...] = (MESSAGES, CHATS, RUN_ITEMS, VALIDATIONS, KV)
HISTORY_TABLES: tuple[TableSpec, ...] = (CHUNK_SOURCES, ITEM_ATTEMPTS, CHUNK_CONVERSIONS)
ALL_TABLES: tuple[TableSpec, ...] = KV_TABLES + HISTORY_TABLES

CHUNK_HISTORY_VIEW_SUFFIX = "chunk_history"


def chunk_history_view_name(base: str) -> str:
    return f"{base}_{CHUNK_HISTORY_VIEW_SUFFIX}"


def chunk_history_view_ddl(base: str) -> str:
    """The reference view: original chunk, converted chunk and the attempt's
    full response and verdict on one row, one row per (attempt, chunk)."""
    conv = quoted_table_name(CHUNK_CONVERSIONS.name(base))
    att = quoted_table_name(ITEM_ATTEMPTS.name(base))
    src = quoted_table_name(CHUNK_SOURCES.name(base))
    return f"""
        CREATE OR REPLACE VIEW {quoted_table_name(chunk_history_view_name(base))} AS
        SELECT
            c.attempt_id, c.thread_id, a.chat_id, c.item_id, a.attempt,
            c.is_final, c.chunk_id, c.chunk_key, c.member_index,
            s.source_id, s.kind, s.title, s.start_line, s.end_line,
            s.text_sha256, s.original_text, c.converted_text,
            a.response, a.document, a.target_language, a.model,
            a.validation_score, a.validation_passed, c.created_at
        FROM {conv} AS c
        JOIN {att} AS a ON a.attempt_id = c.attempt_id
        JOIN {src} AS s ON s.chunk_key = c.chunk_key
    """


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------


@dataclass
class ProvisionReport:
    """What :func:`enable_databricks_features` managed to turn on."""

    predictive_optimization: list[str] = field(default_factory=list)
    search_indexes: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def full_text_search(self) -> bool:
        """True only when every requested index exists."""
        return bool(self.search_indexes) and not self.failures


def ensure_tables(spark: Any, base: str, specs: tuple[TableSpec, ...]) -> None:
    """``CREATE TABLE IF NOT EXISTS`` every spec; enable CDF where missing."""
    for spec in specs:
        spark.sql(spec.ddl(base))
        if spec.cdf and not table_property(spark, spec.name(base), "delta.enableChangeDataFeed"):
            spark.sql(
                f"ALTER TABLE {quoted_table_name(spec.name(base))} SET TBLPROPERTIES "
                "('delta.enableChangeDataFeed' = 'true')"
            )


def table_property(spark: Any, table: str, key: str) -> bool:
    """Whether boolean table property *key* reads ``true``.

    A table that cannot be interrogated reads ``False`` so the caller still
    attempts the ALTER: the failure mode to avoid is silently *skipping* an
    upgrade, not attempting it once too often.
    """
    try:
        rows = spark.sql(f"SHOW TBLPROPERTIES {quoted_table_name(table)}").collect()
    except Exception as exc:  # pragma: no cover - depends on the backend
        logger.debug(f"table_property: could not read properties of {table}: {exc}")
        return False
    for row in rows:
        if row["key"] == key:
            return str(row["value"]).strip().lower() == "true"
    return False


def enable_databricks_features(
    spark: Any, base: str, specs: tuple[TableSpec, ...]
) -> ProvisionReport:
    """Predictive optimization + full-text search indexes for *specs*.

    Databricks-only (DBR 18.3+; the project targets DBR 19, with the workspace
    preview "SQL: Full-Text Search Index for Unity Catalog managed tables"
    enabled). Each statement is attempted independently and failures are
    collected, not raised: an index only ever accelerates ``search()``, it
    never changes a result, so a store without one is slower, not wrong.
    """
    report = ProvisionReport()
    for spec in specs:
        table = quoted_table_name(spec.name(base))
        try:
            spark.sql(f"ALTER TABLE {table} ENABLE PREDICTIVE OPTIMIZATION")
            report.predictive_optimization.append(spec.name(base))
        except Exception as exc:
            report.failures.append(f"{spec.name(base)}: predictive optimization: {_short(exc)}")
            # Without predictive optimization an index is never refreshed;
            # creating one would only cost storage.
            continue
        for index, ddl in zip(spec.search_indexes, spec.index_ddl(base)):
            try:
                spark.sql(ddl)
                report.search_indexes.append(f"{spec.name(base)}.{index.column}")
            except Exception as exc:
                report.failures.append(
                    f"{spec.name(base)}.{index.column}: search index: {_short(exc)}"
                )
    return report


def _short(exc: Exception) -> str:
    text = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {text[0] if text else ''}"[:300]
