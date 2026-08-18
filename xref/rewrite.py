"""Rewriting table names and physical paths in *generated* code, after conversion.

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

Paths, not just tables
----------------------
``xref.pre`` remaps a ``LIBNAME`` / ``INFILE`` / ``%INCLUDE`` path in the SAS
source before conversion. A path can still reach the generated code — no mapping
row covered it when ``pre`` ran, or the model wrote one of its own — so the post
pass rewrites those too: :func:`rewrite_python_paths` over the reader/writer and
``dbutils.fs`` calls, :func:`rewrite_sql_paths` over ``LOCATION`` and friends.
Both resolve through :mod:`xref.mapping`, the same longest-prefix resolver
``pre`` uses, so one path cannot come out of the two halves as two targets.

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
import re

import app_config
from target_language import SPARKSQL

from .mapping import map_path, ordered_keys

logger = logging.getLogger(__name__)

ON_PARSE_FAILURE_WARN = "warn"
ON_PARSE_FAILURE_ERROR = "error"
_ON_PARSE_FAILURE = (ON_PARSE_FAILURE_WARN, ON_PARSE_FAILURE_ERROR)

# Attribute calls whose first string argument names a table.
_TABLE_CALLS = frozenset({"table", "saveAsTable", "insertInto", "createOrReplaceTempView"})
# Attribute calls whose first string argument is SQL to recurse into.
_SQL_CALLS = frozenset({"sql"})

# Attribute calls whose first string argument is a *path* rather than a table:
# the DataFrame reader/writer formats, and the dbutils filesystem utilities.
# ``head`` and ``text`` are also DataFrame methods that take something else —
# the first-argument-is-a-string test and the mapping lookup both have to pass
# before anything is rewritten, so a `df.head(5)` is never touched.
_PATH_CALLS = frozenset(
    {
        # DataFrameReader / DataFrameWriter
        "csv", "parquet", "json", "orc", "text", "load", "save",
        # dbutils.fs
        "ls", "cp", "mv", "rm", "mkdirs", "mount", "head", "put",
    }
)
# Builtins taking a path as their first argument.
_PATH_BUILTINS = frozenset({"open"})
# ``.option("path", "<path>")`` — the one form whose path is the *second*
# argument, so it needs its own branch rather than joining _PATH_CALLS.
_OPTION_CALL = "option"
_OPTION_PATH_KEYS = frozenset({"path"})


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

    The non-SQL target (PySpark) has no dialect of its own, but this
    module still reaches sqlglot for the SQL inside ``spark.sql("...")`` — so
    those fall back to the SQL target's dialect rather than to nothing.
    """
    return SPARKSQL.sqlglot_dialect or "databricks"

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

    read = SPARKSQL.sqlglot_dialect or "databricks"
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

    offset = _offsetter(source)
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
            # The literal's own quoting style, like the path half and the SQL
            # branch below. ``repr`` is the fallback for a value whose style
            # cannot be reproduced (it embeds the quote, or a backslash), not
            # the default: normalising every rewritten "x" to 'x' put quoting
            # churn in the diff of a file whose table name was the only change.
            new_text = _requote(source, literal, target) or repr(target)
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
    logger.info(f"rewrite_python: rewrote {len(edits)} table reference(s)")
    return _apply_edits(source, edits)


def _offsetter(source: str):
    """A ``(lineno, col) -> character offset`` function for *source*.

    ``ast`` reports columns as utf-8 byte offsets into the line, so the prefix
    is encoded and decoded rather than assumed to agree with the character
    index. Built once per rewrite and shared by every span in it.
    """
    lines = source.splitlines(keepends=True)
    starts: list[int] = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def offset(lineno: int, col: int) -> int:
        line = lines[lineno - 1]
        return starts[lineno - 1] + len(line.encode("utf-8")[:col].decode("utf-8"))

    return offset


def _apply_edits(source: str, edits: list[tuple[int, int, str]]) -> str:
    """*source* with each ``(start, end, replacement)`` span substituted.

    Applied back-to-front so an earlier span's offsets are still valid when it
    is reached. Everything outside the listed spans — comments, formatting,
    quoting — comes through byte-identical, which is what makes this module's
    hard rule (leave unparseable input alone) worth anything: a rewrite that
    reformatted would be indistinguishable from a corruption.
    """
    out = source
    for start, end, replacement in sorted(edits, reverse=True):
        out = out[:start] + replacement + out[end:]
    return out


def _path_literal(node: ast.Call) -> ast.Constant | None:
    """The string literal holding a path in *node*, or ``None``.

    Three call shapes, in the order they are cheapest to reject:
    ``dbutils.fs.ls("<path>")`` and the reader/writer formats
    (:data:`_PATH_CALLS`), ``open("<path>")`` (:data:`_PATH_BUILTINS`, a bare
    name rather than an attribute), and ``.option("path", "<path>")`` — the
    only one whose path is not the first argument.
    """
    if not node.args:
        return None
    func = node.func
    if isinstance(func, ast.Name):
        if func.id not in _PATH_BUILTINS:
            return None
        found = _string_literal(node.args[0])
        return found[0] if found else None
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr == _OPTION_CALL:
        key = _string_literal(node.args[0])
        if not key or key[1].strip().lower() not in _OPTION_PATH_KEYS:
            return None
        if len(node.args) < 2:
            return None
        found = _string_literal(node.args[1])
        return found[0] if found else None
    if func.attr not in _PATH_CALLS:
        return None
    found = _string_literal(node.args[0])
    return found[0] if found else None


def rewrite_python_paths(
    source: str, by_path: dict[str, str], *, on_failure: str | None = None
) -> str:
    """
    *source* with the physical paths in its Spark and filesystem calls remapped.

    The path counterpart of :func:`rewrite_python`: same ``ast`` walk, same
    source-span substitution, same hard rule — unparseable input comes back
    exactly as the model wrote it. Only the literals in the recognised path
    positions (:func:`_path_literal`) are considered, and only those a
    ``by_path`` mapping actually addresses are changed, so a string that merely
    looks like a path is left alone.

    Resolution is :func:`xref.mapping.map_path`, shared with the pre pass, so
    one path cannot be rewritten to two different targets by the two halves.
    """
    if not by_path or not source.strip():
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        _unparseable("PySpark", exc, on_failure)
        return source

    keys = ordered_keys(by_path)
    offset = _offsetter(source)
    edits: list[tuple[int, int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        literal = _path_literal(node)
        if literal is None or literal.end_lineno is None:
            continue
        if literal.end_col_offset is None or not isinstance(literal.value, str):
            continue
        target = map_path(literal.value, by_path, keys)
        if target is None:
            continue
        new_text = _requote(source, literal, target)
        if new_text is None:
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
    logger.info(f"rewrite_python_paths: rewrote {len(edits)} path(s)")
    return _apply_edits(source, edits)


# Where a physical path appears in generated Spark SQL. Named groups match
# chunker.paths' convention — ``head`` is everything up to the opening quote,
# ``q`` the quote, ``path`` the value — so the substitution is one helper for
# every entry.
#
# Matched by *position* rather than by parsing, unlike the table rewriter above:
# sqlglot regenerates a whole statement to change a node, which reformats
# everything it touches. That is worth paying to re-emit a table as structured
# identifiers; it is not worth paying to change a string literal.
_SQL_PATH_POSITIONS: tuple[re.Pattern[str], ...] = (
    # CREATE TABLE ... LOCATION '<path>'
    re.compile(r"(?P<head>\blocation\s+)(?P<q>['\"])(?P<path>[^'\"\n]*)(?P=q)", re.I),
    # OPTIONS ('path' = '<path>') — the quoted-key form, matched before the bare
    # one so the key is consumed as part of `head` rather than being mistaken
    # for the start of a bare `path` clause.
    re.compile(
        r"(?P<head>['\"]path['\"]\s*=\s*)(?P<q>['\"])(?P<path>[^'\"\n]*)(?P=q)", re.I
    ),
    # OPTIONS (path '<path>') and PATH = '<path>'. A separator is required, so
    # this cannot latch onto the quoted key above and rewrite the ` = ` between
    # them as though it were the value.
    re.compile(
        r"(?P<head>\bpath\s*(?:=\s*|\s+))(?P<q>['\"])(?P<path>[^'\"\n]*)(?P=q)", re.I
    ),
    # COPY INTO ... FROM '<path>' — a quoted literal in FROM position is a
    # location, not a table name (a table there is an identifier).
    re.compile(r"(?P<head>\bfrom\s+)(?P<q>['\"])(?P<path>[^'\"\n]*)(?P=q)", re.I),
    # read_files('<path>')
    re.compile(
        r"(?P<head>\bread_files\s*\(\s*)(?P<q>['\"])(?P<path>[^'\"\n]*)(?P=q)", re.I
    ),
)


def rewrite_sql_paths(sql: str, by_path: dict[str, str]) -> str:
    """
    *sql* with the paths in its location clauses remapped.

    Needs no parser — see :data:`_SQL_PATH_POSITIONS` — so unlike
    :func:`rewrite_sql` this does not degrade when ``sqlglot`` is absent, and
    SQL it cannot make sense of is simply left untouched rather than
    round-tripped.
    """
    if not by_path or not sql.strip():
        return sql

    keys = ordered_keys(by_path)
    changed = 0

    def _substitute(match: re.Match[str]) -> str:
        nonlocal changed
        target = map_path(match.group("path"), by_path, keys)
        if target is None:
            return match.group(0)
        changed += 1
        quote = match.group("q")
        return f"{match.group('head')}{quote}{target}{quote}"

    out = sql
    for pattern in _SQL_PATH_POSITIONS:
        out = pattern.sub(_substitute, out)
    if changed:
        logger.info(f"rewrite_sql_paths: rewrote {changed} path(s)")
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
