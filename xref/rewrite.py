"""Rewriting table names in *generated* code, after conversion.

Two languages, two parsers, one rule.

Spark SQL
    ``sqlglot.parse`` under the configured dialect, walk every ``exp.Table``,
    rewrite the ones whose ``db.name`` matches a mapping key, regenerate.

PySpark
    The ``ast`` module over the string literals that name tables:
    ``spark.table("x")``, ``spark.read.table("x")``, ``.saveAsTable("x")``, and
    ``spark.sql("...")`` — the last recursing into the sqlglot path, since the
    SQL inside it is still SQL. Rewriting is done by **source-span
    substitution** rather than ``ast.unparse``, so comments, formatting and
    string quoting survive untouched; only the matched literals change.

The hard rule
-------------
**Unparseable input is left exactly as the model wrote it.** A rewriter that
corrupts generated code is worse than one that no-ops: the no-op leaves a
reviewer with working code and a wrong table name, which is visible and
fixable, while a corrupted file is neither. ``xref.on_parse_failure`` chooses
between a WARNING (default) and raising, and nothing in between.

``sqlglot`` is imported lazily — it is the optional ``sql`` extra — so this
module costs nothing to import and only the ``"post"`` mode needs it
installed.

Logger name: ``xref.rewrite``.
"""

from __future__ import annotations

import ast
import logging

import app_config
from target_language import SPARKSQL, resolve_target_language

logger = logging.getLogger(__name__)

ON_PARSE_FAILURE_WARN = "warn"
ON_PARSE_FAILURE_ERROR = "error"
_ON_PARSE_FAILURE = (ON_PARSE_FAILURE_WARN, ON_PARSE_FAILURE_ERROR)

# Attribute calls whose first string argument names a table.
_TABLE_CALLS = frozenset({"table", "saveAsTable", "insertInto", "createOrReplaceTempView"})
# Attribute calls whose first string argument is SQL to recurse into.
_SQL_CALLS = frozenset({"sql"})


class XrefRewriteError(RuntimeError):
    """Generated code could not be parsed and ``xref.on_parse_failure`` is
    ``"error"``. The output is unchanged either way — this only decides
    whether the run stops."""


def default_dialect() -> str:
    """The run's target language, as a sqlglot dialect.

    Resolved from ``pipeline.output_language`` rather than held as a constant
    here: the syntax checker parses generated SQL under the *same* target's
    :attr:`~target_language.TargetLanguage.sqlglot_dialect`, and a rewriter
    reading it differently would silently return code un-rewritten (the hard
    rule above) on exactly the syntax the checker had just called valid.

    A non-SQL target (PySpark, Spark Scala) has no dialect of its own, but this
    module still reaches sqlglot for the SQL inside ``spark.sql("...")`` — so
    those fall back to the SQL target's dialect rather than to nothing.
    """
    configured = app_config.get_value("pipeline", "output_language")
    if configured:
        try:
            target = resolve_target_language(str(configured))
        except Exception:
            # An unknown name is the pipeline's error to raise, not ours: the
            # rewriter should not be what fails a run over a config typo.
            target = SPARKSQL
    else:
        target = SPARKSQL
    return target.sqlglot_dialect or SPARKSQL.sqlglot_dialect or "databricks"


def dialect() -> str:
    """The sqlglot dialect for the post rewriter.

    ``xref.dialect`` overrides it; unset, it follows the run's target language
    (:func:`default_dialect`), so the rewriter and the syntax checker agree by
    construction.
    """
    return app_config.get_typed_value(
        "xref", "dialect", str, default_dialect()
    )


def on_parse_failure() -> str:
    """What to do with unparseable generated code (``xref.on_parse_failure``).

    An unrecognised value degrades to ``"warn"`` with a WARNING, the same
    degrade-don't-crash rule the rest of the config follows — and the safer
    direction, since ``"warn"`` never fails a run.
    """
    configured = app_config.get_typed_value(
        "xref", "on_parse_failure", str, ON_PARSE_FAILURE_WARN
    )
    if configured not in _ON_PARSE_FAILURE:
        logger.warning(
            f"on_parse_failure: config.json xref.on_parse_failure "
            f"{configured!r} is not one of {'/'.join(_ON_PARSE_FAILURE)}; "
            f"ignoring it ({ON_PARSE_FAILURE_WARN!r} applies)"
        )
        return ON_PARSE_FAILURE_WARN
    return configured


def _unparseable(what: str, exc: Exception, mode: str | None) -> None:
    """Handle a parse failure per *mode*, having changed nothing."""
    resolved = mode if mode in _ON_PARSE_FAILURE else on_parse_failure()
    message = (
        f"could not parse the generated {what} ({exc}); leaving it exactly as "
        f"the model wrote it — no XREF substitution was applied"
    )
    if resolved == ON_PARSE_FAILURE_ERROR:
        raise XrefRewriteError(message) from exc
    logger.warning(f"rewrite: {message}")


def _lookup(name: str, mapping: dict[str, str]) -> str | None:
    """The target for a table name, matched case-insensitively.

    Both the qualified name (``schema.table``) and the bare table name are
    tried, in that order: a mapping is written against the SAS-side
    ``schema.table``, but generated code may name the table alone.
    """
    lowered = name.strip().lower()
    hit = mapping.get(lowered)
    if hit is not None:
        return hit
    _, dot, bare = lowered.rpartition(".")
    if dot and bare:
        return mapping.get(bare)
    return None


def rewrite_sql(
    sql: str, mapping: dict[str, str], *, on_failure: str | None = None
) -> str:
    """
    *sql* with every mapped table reference rewritten to its target.

    Returns *sql* unchanged when it does not parse, when ``sqlglot`` is not
    installed, or when nothing matched.
    """
    if not mapping or not sql.strip():
        return sql
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError:
        logger.warning(
            "rewrite_sql: sqlglot is not installed, so the post-conversion "
            "XREF rewrite cannot run; install the 'sql' extra "
            "('pip install \"sas-parser[sql]\"'). The code is unchanged"
        )
        return sql

    read = dialect()
    try:
        statements = sqlglot.parse(sql, read=read)
    except Exception as exc:  # sqlglot raises several unrelated types
        _unparseable("Spark SQL", exc, on_failure)
        return sql

    changed = False
    for statement in statements:
        if statement is None:
            continue
        for table in statement.find_all(exp.Table):
            qualified = ".".join(
                part.name for part in (table.args.get("db"), table.this) if part
            )
            target = _lookup(qualified, mapping)
            if target is None:
                continue
            parts = target.split(".")
            table.set("this", exp.to_identifier(parts[-1]))
            table.set("db", exp.to_identifier(parts[-2]) if len(parts) > 1 else None)
            table.set(
                "catalog", exp.to_identifier(parts[-3]) if len(parts) > 2 else None
            )
            changed = True
    if not changed:
        return sql
    rewritten = ";\n".join(
        statement.sql(dialect=read) for statement in statements if statement is not None
    )
    if sql.rstrip().endswith(";"):
        rewritten += ";"
    logger.info(f"rewrite_sql: rewrote table references under dialect {read!r}")
    return rewritten


def _string_literal(node: ast.AST) -> tuple[ast.Constant, str] | None:
    """*node* and its text when it is a plain string constant, else ``None``.

    The text comes back alongside the node because ``ast.Constant.value`` is
    typed as the union of every literal type; returning it here is where the
    narrowing happens, once.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node, node.value
    return None


def _call_name(node: ast.Call) -> str | None:
    """The attribute a call was made through (``spark.table`` -> ``table``)."""
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else None


def rewrite_python(
    source: str, mapping: dict[str, str], *, on_failure: str | None = None
) -> str:
    """
    *source* with the table names in its Spark calls rewritten.

    Only string literals in the recognised call positions are touched
    (:data:`_TABLE_CALLS`, :data:`_SQL_CALLS`), and only by replacing their
    source span — everything else in the file, comments included, comes
    through byte-identical. Returns *source* unchanged when it does not parse
    or nothing matched.
    """
    if not mapping or not source.strip():
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        _unparseable("PySpark", exc, on_failure)
        return source

    lines = source.splitlines(keepends=True)
    # Byte offset of the start of each 1-based line, for span arithmetic.
    starts: list[int] = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def offset(lineno: int, col: int) -> int:
        # ast columns are utf-8 byte offsets into the line; the sources here
        # are code, so encode the prefix rather than assume they agree.
        line = lines[lineno - 1]
        return starts[lineno - 1] + len(line.encode("utf-8")[:col].decode("utf-8"))

    # (start, end, replacement text), collected then applied back-to-front so
    # earlier spans keep their offsets.
    edits: list[tuple[int, int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = _call_name(node)
        if name is None:
            continue
        found = _string_literal(node.args[0])
        if found is None:
            continue
        literal, text = found
        if literal.end_lineno is None or literal.end_col_offset is None:
            continue
        if name in _TABLE_CALLS:
            target = _lookup(text, mapping)
            if target is None:
                continue
            new_text = repr(target)
        elif name in _SQL_CALLS:
            rewritten = rewrite_sql(text, mapping, on_failure=on_failure)
            if rewritten == text:
                continue
            # Keep the literal's own quoting style where it is single-line;
            # repr() would mangle a triple-quoted block.
            new_text = _requote(source, literal, rewritten)
            if new_text is None:
                continue
        else:
            continue
        edits.append(
            (
                offset(literal.lineno, literal.col_offset),
                offset(literal.end_lineno, literal.end_col_offset),
                new_text,
            )
        )

    if not edits:
        return source
    out = source
    for start, end, replacement in sorted(edits, reverse=True):
        out = out[:start] + replacement + out[end:]
    logger.info(f"rewrite_python: rewrote {len(edits)} table reference(s)")
    return out


def _requote(source: str, literal: ast.Constant, value: str) -> str | None:
    """*value* re-emitted in the same quoting style the original literal used.

    ``None`` when the original's style cannot be reproduced safely, in which
    case the literal is left alone — the module's rule again: no change beats
    a wrong one.
    """
    if literal.end_lineno is None or literal.end_col_offset is None:
        return None
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    start = starts[literal.lineno - 1] + literal.col_offset
    end = starts[literal.end_lineno - 1] + literal.end_col_offset
    original = source[start:end]
    for quote in ('"""', "'''", '"', "'"):
        if original.startswith(quote) and original.endswith(quote):
            if quote in value or "\\" in value:
                return None  # would need escaping; leave it alone
            return f"{quote}{value}{quote}"
    return None
