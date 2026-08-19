"""Source-preserving PySpark table and path rewriting for XREF."""

from __future__ import annotations

import ast

from ..mapping import ordered_path_keys, resolve_path
from ..models import ParseFailureMode
from .sql import _lookup_table, _parse_failure, rewrite_sql_tables

_TABLE_CALLS = frozenset(
    {"table", "saveAsTable", "insertInto", "createOrReplaceTempView"}
)
_PATH_CALLS = frozenset(
    {
        "csv",
        "parquet",
        "json",
        "orc",
        "text",
        "load",
        "save",
        "ls",
        "cp",
        "mv",
        "rm",
        "mkdirs",
        "mount",
        "head",
        "put",
    }
)


def _string(node: ast.AST) -> tuple[ast.Constant, str] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node, node.value
    return None


def _call_name(node: ast.Call) -> str | None:
    return node.func.attr if isinstance(node.func, ast.Attribute) else None


def _offsetter(source: str):
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def offset(lineno: int, column: int) -> int:
        line = lines[lineno - 1]
        prefix = line.encode("utf-8")[:column].decode("utf-8")
        return starts[lineno - 1] + len(prefix)

    return offset


def _apply_edits(source: str, edits: list[tuple[int, int, str]]) -> str:
    output = source
    for start, end, replacement in sorted(edits, reverse=True):
        output = output[:start] + replacement + output[end:]
    return output


def _requote(source: str, literal: ast.Constant, value: str) -> str | None:
    if literal.end_lineno is None or literal.end_col_offset is None:
        return None
    offset = _offsetter(source)
    original = source[
        offset(literal.lineno, literal.col_offset) :
        offset(literal.end_lineno, literal.end_col_offset)
    ]
    quote = next((item for item in ('"""', "'''", '"', "'") if original.startswith(item)), None)
    if quote is None or quote in value or "\\" in value:
        return None
    return f"{quote}{value}{quote}"


def rewrite_pyspark_tables(
    source: str,
    mapping: dict[str, str],
    *,
    on_failure: ParseFailureMode = ParseFailureMode.WARN,
) -> str:
    """Rewrite recognized table literals without reformatting Python."""

    if not mapping or not source.strip():
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        _parse_failure("PySpark", exc, on_failure)
        return source

    offset = _offsetter(source)
    edits: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = _call_name(node)
        found = _string(node.args[0])
        if name is None or found is None:
            continue
        literal, text = found
        replacement: str | None = None
        if name in _TABLE_CALLS:
            target = _lookup_table(text, mapping)
            if target is not None:
                replacement = _requote(source, literal, target) or repr(target)
        elif name == "sql":
            rewritten = rewrite_sql_tables(text, mapping, on_failure=on_failure)
            if rewritten != text:
                replacement = _requote(source, literal, rewritten)
        if replacement is None or literal.end_lineno is None:
            continue
        if literal.end_col_offset is None:
            continue
        edits.append(
            (
                offset(literal.lineno, literal.col_offset),
                offset(literal.end_lineno, literal.end_col_offset),
                replacement,
            )
        )
    return _apply_edits(source, edits) if edits else source


def _path_literal(node: ast.Call) -> ast.Constant | None:
    if not node.args:
        return None
    if isinstance(node.func, ast.Name):
        found = _string(node.args[0]) if node.func.id == "open" else None
        return found[0] if found else None
    if not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr == "option":
        key = _string(node.args[0])
        if not key or key[1].strip().casefold() != "path" or len(node.args) < 2:
            return None
        found = _string(node.args[1])
        return found[0] if found else None
    if node.func.attr not in _PATH_CALLS:
        return None
    found = _string(node.args[0])
    return found[0] if found else None


def rewrite_pyspark_paths(
    source: str,
    by_path: dict[str, str],
    *,
    on_failure: ParseFailureMode = ParseFailureMode.WARN,
) -> str:
    """Rewrite recognized path literals without reformatting Python."""

    if not by_path or not source.strip():
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        _parse_failure("PySpark", exc, on_failure)
        return source
    keys = ordered_path_keys(by_path)
    offset = _offsetter(source)
    edits: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        literal = _path_literal(node)
        if literal is None or not isinstance(literal.value, str):
            continue
        if literal.end_lineno is None or literal.end_col_offset is None:
            continue
        target = resolve_path(literal.value, by_path, keys)
        replacement = _requote(source, literal, target) if target is not None else None
        if replacement is None:
            continue
        edits.append(
            (
                offset(literal.lineno, literal.col_offset),
                offset(literal.end_lineno, literal.end_col_offset),
                replacement,
            )
        )
    return _apply_edits(source, edits) if edits else source


__all__ = ["rewrite_pyspark_paths", "rewrite_pyspark_tables"]
