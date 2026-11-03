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
    # Extra TBLPROPERTIES set at CREATE time (e.g. auto compaction for the
    # append-heavy tables, whose small commits would otherwise pile up).
    extra_properties: tuple[tuple[str, str], ...] = ()

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
        props.extend(
            f"{sql_string_literal(k)} = {sql_string_literal(v)}"
            for k, v in self.extra_properties
        )
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

    def index_leaf(self, base: str, column: str) -> str:
        """The index name without catalog/schema, backquoted."""
        return f"`{split_table_name(self.name(base))[-1]}_{column}_fts`"

    def index_ddl(self, base: str) -> list[str]:
        """``CREATE SEARCH INDEX`` per index, with a ``{name}`` placeholder
        for the index identifier (see :func:`run_index_sql`)."""
        table = quoted_table_name(self.name(base))
        return [
            "CREATE SEARCH INDEX IF NOT EXISTS {name} "
            f"ON {table} (`{i.column}`) OPTIONS ({i.options()})"
            for i in self.search_indexes
        ]


# Append-heavy tables get optimized writes and auto compaction: each pipeline
# turn commits a few small files, which these fold together after the write.
# 'true' (not Databricks' 'auto') is the one value OSS Delta also accepts.
APPEND_HEAVY_PROPERTIES: tuple[tuple[str, str], ...] = (
    ("delta.autoOptimize.optimizeWrite", "true"),
    ("delta.autoOptimize.autoCompact", "true"),
)


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
        ("chat_id", "STRING"),
    ),
    not_null=_KV_NOT_NULL,
    cluster_by=("thread_id", "msg_tick"),
    cdf=True,
    extra_properties=APPEND_HEAVY_PROPERTIES,
    # KVStore.search pushes isearch(kv_key, value, tags) down; an index can
    # only skip files when every target column is indexed, so all three are,
    # plus the plain-text content for direct SQL lookups (the 4-index cap).
    search_indexes=(
        SearchIndex("content", "ngram", 5),
        SearchIndex("value", "ngram", 5),
        SearchIndex("kv_key", "ngram", 5),
        SearchIndex("tags", "ngram", 5),
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
        ("started_at", "DOUBLE"),
    ),
    not_null=_KV_NOT_NULL,
    cluster_by=("thread_id",),
    cdf=True,
    comment="Static row per chat (one pipeline instance's span of a "
    "thread); first/last key and message count derive from "
    "messages.chat_id.",
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
    search_indexes=(
        SearchIndex("value", "ngram", 5),
        SearchIndex("kv_key", "ngram", 5),
        SearchIndex("tags", "ngram", 5),
    ),
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
        # The corrective note prompted WITH this attempt (the previous
        # attempt's failed verdict, rendered); NULL on a first attempt.
        # Appended last so the additive migration of an existing table and a
        # fresh CREATE yield the same column order.
        ("feedback", "STRING"),
    ),
    not_null=("attempt_id", "thread_id", "item_id", "attempt", "is_final"),
    cluster_by=("thread_id", "item_id"),
    search_indexes=(SearchIndex("response", "ngram", 4),),
    extra_properties=APPEND_HEAVY_PROPERTIES,
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
    extra_properties=APPEND_HEAVY_PROPERTIES,
    comment="Every attempt's CONVERTED code for each member chunk. "
    "Append-only.",
)

def _archive_of(spec: TableSpec, cluster_by: tuple[str, ...]) -> TableSpec:
    """Cold-storage twin of *spec*: same columns plus ``archived_at``.

    No search indexes and no CDF — archived history is read rarely, and a
    ``LIKE`` scan over it is the accepted cost of keeping the active tables
    (and their indexes) small.
    """
    return TableSpec(
        suffix=f"{spec.suffix}_archive",
        columns=spec.columns + (("archived_at", "DOUBLE"),),
        not_null=spec.not_null + ("archived_at",),
        cluster_by=cluster_by,
        comment=f"Archived rows of {spec.suffix} (ChunkHistory.archive).",
    )


ITEM_ATTEMPTS_ARCHIVE = _archive_of(ITEM_ATTEMPTS, ("thread_id", "item_id"))
CHUNK_CONVERSIONS_ARCHIVE = _archive_of(CHUNK_CONVERSIONS, ("thread_id", "item_id"))

#: Tables the KVStore façade routes into, most specific prefix first.
KV_ROUTES: tuple[tuple[str, TableSpec], ...] = (
    ("msg::", MESSAGES),
    ("chat::", CHATS),
    ("kv::run::", RUN_ITEMS),
    ("kv::validation::", VALIDATIONS),
)
KV_FALLBACK = KV

KV_TABLES: tuple[TableSpec, ...] = (MESSAGES, CHATS, RUN_ITEMS, VALIDATIONS, KV)
RESTORE_JOURNAL = TableSpec(
    suffix="restore_journal",
    columns=(
        ("restore_id", "STRING"),
        ("table_name", "STRING"),
        ("pre_version", "BIGINT"),
        ("state", "STRING"),
        ("ts", "DOUBLE"),
    ),
    not_null=("restore_id", "state", "ts"),
    cluster_by=("restore_id",),
    comment="Append-only log of KVStore.restore runs: each table's version "
    "before the overwrite, so a failed or crashed restore can be rolled back "
    "across every memory table.",
)

HISTORY_TABLES: tuple[TableSpec, ...] = (CHUNK_SOURCES, ITEM_ATTEMPTS, CHUNK_CONVERSIONS)
ARCHIVE_TABLES: tuple[TableSpec, ...] = (ITEM_ATTEMPTS_ARCHIVE, CHUNK_CONVERSIONS_ARCHIVE)
ALL_TABLES: tuple[TableSpec, ...] = (
    KV_TABLES + (RESTORE_JOURNAL,) + HISTORY_TABLES + ARCHIVE_TABLES
)

CHUNK_HISTORY_VIEW_SUFFIX = "chunk_history"


def chunk_history_view_name(base: str, *, archived: bool = False) -> str:
    return f"{base}_{CHUNK_HISTORY_VIEW_SUFFIX}{'_archive' if archived else ''}"


def chunk_history_view_ddl(base: str, *, archived: bool = False) -> str:
    """The reference view: original chunk, converted chunk and the attempt's
    full response and verdict on one row, one row per (attempt, chunk).

    ``archived=True`` builds the same view over the archive tables
    (``B_chunk_history_archive``); original chunks are never archived — one
    row per distinct chunk does not grow with attempts — so both views join
    the same ``B_chunk_sources``.
    """
    conv_spec, att_spec = (
        (CHUNK_CONVERSIONS_ARCHIVE, ITEM_ATTEMPTS_ARCHIVE)
        if archived
        else (CHUNK_CONVERSIONS, ITEM_ATTEMPTS)
    )
    conv = quoted_table_name(conv_spec.name(base))
    att = quoted_table_name(att_spec.name(base))
    src = quoted_table_name(CHUNK_SOURCES.name(base))
    archived_at = "c.archived_at" if archived else "CAST(NULL AS DOUBLE)"
    return f"""
        CREATE OR REPLACE VIEW {quoted_table_name(chunk_history_view_name(base, archived=archived))} AS
        SELECT
            c.attempt_id, c.thread_id, a.chat_id, c.item_id, a.attempt,
            c.is_final, c.chunk_id, c.chunk_key, c.member_index,
            s.source_id, s.kind, s.title, s.start_line, s.end_line,
            s.text_sha256, s.original_text, c.converted_text,
            a.response, a.document, a.feedback, a.target_language, a.model,
            a.validation_score, a.validation_passed, c.created_at,
            {archived_at} AS archived_at
        FROM {conv} AS c
        JOIN {att} AS a ON a.attempt_id = c.attempt_id
        JOIN (
            -- Two workers recording the same new chunk at once can both win
            -- their insert-only MERGE; keep the first so a race can never
            -- duplicate history rows.
            SELECT * FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY chunk_key ORDER BY first_seen_at, chunk_id
                ) AS _rn
                FROM {src}
            ) WHERE _rn = 1
        ) AS s ON s.chunk_key = c.chunk_key
    """


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------


# Errors meaning "this statement does not take a qualified index name" —
# as opposed to a missing preview, a permission, or an unsupported table.
_NAME_FORM_ERRORS = (
    "PARSE_SYNTAX_ERROR",
    "INVALID_IDENTIFIER",
    "REQUIRES_SINGLE_PART_NAMESPACE",
    "IDENTIFIER_TOO_MANY_NAME_PARTS",
    "INVALID_INDEX_NAME",
)


def _is_name_form_error(exc: Exception) -> bool:
    text = str(exc).upper()
    return any(code in text for code in _NAME_FORM_ERRORS)


def run_index_sql(
    spark: Any, spec: "TableSpec", base: str, column: str, template: str
) -> tuple[Any, str]:
    """Run an index statement (*template* has a ``{name}`` placeholder).

    The Databricks docs do not say whether ``CREATE/DESCRIBE/DROP INDEX``
    accept a catalog.schema-qualified index name. The qualified form is tried
    first; if it is rejected *as a name form*, the statement is retried with
    the bare name inside the table's catalog and schema — ``USE`` scoped to
    this call, the session's previous catalog/schema restored afterwards.
    Any other error propagates. Returns ``(result, "qualified" | "bare")``.
    """
    try:
        return spark.sql(template.format(name=spec.index_name(base, column))), "qualified"
    except Exception as exc:
        parts = split_table_name(spec.name(base))
        if len(parts) == 1 or not _is_name_form_error(exc):
            raise
    previous = spark.sql("SELECT current_catalog() AS c, current_schema() AS s").first()
    try:
        if len(parts) == 3:
            spark.sql(f"USE CATALOG `{parts[0]}`")
        spark.sql(f"USE SCHEMA `{parts[-2]}`")
        return spark.sql(template.format(name=spec.index_leaf(base, column))), "bare"
    finally:
        if len(parts) == 3:
            spark.sql(f"USE CATALOG `{previous['c']}`")
        spark.sql(f"USE SCHEMA `{previous['s']}`")


@dataclass
class ProvisionReport:
    """What :func:`enable_databricks_features` managed to turn on."""

    predictive_optimization: list[str] = field(default_factory=list)
    search_indexes: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    #: Which index-name form the runtime accepted ("qualified" / "bare"),
    #: once any index statement succeeded.
    index_name_style: str | None = None

    @property
    def full_text_search(self) -> bool:
        """True only when every requested index exists."""
        return bool(self.search_indexes) and not self.failures


def ensure_tables(spark: Any, base: str, specs: tuple[TableSpec, ...]) -> None:
    """``CREATE TABLE IF NOT EXISTS`` every spec, add any nullable column a
    newer spec declares (additive migration), enable CDF where missing."""
    for spec in specs:
        spark.sql(spec.ddl(base))
        _add_missing_columns(spark, base, spec)
        if spec.cdf and not table_property(spark, spec.name(base), "delta.enableChangeDataFeed"):
            spark.sql(
                f"ALTER TABLE {quoted_table_name(spec.name(base))} SET TBLPROPERTIES "
                "('delta.enableChangeDataFeed' = 'true')"
            )


def _add_missing_columns(spark: Any, base: str, spec: TableSpec) -> None:
    """Append columns *spec* declares but an existing table lacks.

    Only nullable columns can be added safely — existing rows read NULL. A
    missing NOT NULL column means the table is not this store's, so it is an
    error rather than a guess.
    """
    table = spec.name(base)
    present = {f.name for f in spark.table(quoted_table_name(table)).schema}
    missing = [(n, t) for n, t in spec.columns if n not in present]
    if not missing:
        return
    required = [n for n, _ in missing if n in spec.not_null]
    if required:
        raise RuntimeError(
            f"Delta table {table!r} lacks required column(s) {required!r}; "
            "it is not a memory table this version can migrate"
        )
    cols = ", ".join(f"`{n}` {t}" for n, t in missing)
    spark.sql(f"ALTER TABLE {quoted_table_name(table)} ADD COLUMNS ({cols})")
    logger.info(f"ensure_tables: added column(s) {[n for n, _ in missing]} to {table}")


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
                _, style = run_index_sql(spark, spec, base, index.column, ddl)
                report.index_name_style = style
                report.search_indexes.append(f"{spec.name(base)}.{index.column}")
            except Exception as exc:
                report.failures.append(
                    f"{spec.name(base)}.{index.column}: search index: {_short(exc)}"
                )
    return report


def _short(exc: Exception) -> str:
    text = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {text[0] if text else ''}"[:300]
