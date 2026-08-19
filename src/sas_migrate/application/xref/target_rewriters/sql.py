"""Databricks SQL table and path rewriting for XREF."""

from __future__ import annotations

import logging
import re

from sas_migrate.core.targets import SPARK_SQL

from ..mapping import ordered_path_keys, resolve_path
from ..models import ParseFailureMode

logger = logging.getLogger(__name__)


class XrefRewriteError(RuntimeError):
    """A generated target could not be parsed under fatal rewrite policy."""


def _parse_failure(what: str, exc: Exception, mode: ParseFailureMode) -> None:
    message = (
        f"could not parse generated {what} ({exc}); leaving it exactly as the "
        "model wrote it — no XREF substitution was applied"
    )
    if mode is ParseFailureMode.ERROR:
        raise XrefRewriteError(message) from exc
    logger.warning(message)


def _lookup_table(name: str, mapping: dict[str, str]) -> str | None:
    lowered = name.strip().casefold()
    if target := mapping.get(lowered):
        return target
    _, dot, bare = lowered.rpartition(".")
    return mapping.get(bare) if dot and bare else None


def rewrite_sql_tables(
    sql: str,
    mapping: dict[str, str],
    *,
    on_failure: ParseFailureMode = ParseFailureMode.WARN,
) -> str:
    """Rewrite structured table nodes using the Databricks SQLGlot dialect."""

    if not mapping or not sql.strip():
        return sql
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError:
        logger.warning("sqlglot is unavailable; leaving generated SQL unchanged")
        return sql

    dialect = SPARK_SQL.sqlglot_dialect
    if dialect != "databricks":
        raise RuntimeError("Spark SQL XREF rewriting requires the Databricks dialect")
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except Exception as exc:  # noqa: BLE001 - sqlglot raises unrelated parser types
        _parse_failure("Spark SQL", exc, on_failure)
        return sql

    changed = False
    for statement in statements:
        if statement is None:
            continue
        for table in statement.find_all(exp.Table):
            qualified = ".".join(
                part.name
                for part in (
                    table.args.get("catalog"),
                    table.args.get("db"),
                    table.this,
                )
                if part
            )
            target = _lookup_table(qualified, mapping)
            if target is None:
                continue
            parts = target.split(".")
            table.set("this", exp.to_identifier(parts[-1]))
            table.set("db", exp.to_identifier(parts[-2]) if len(parts) > 1 else None)
            table.set(
                "catalog",
                exp.to_identifier(parts[-3]) if len(parts) > 2 else None,
            )
            changed = True
    if not changed:
        return sql
    output = ";\n".join(
        statement.sql(dialect=dialect)
        for statement in statements
        if statement is not None
    )
    if sql.rstrip().endswith(";"):
        output += ";"
    return output


_SQL_PATH_POSITIONS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?P<head>\blocation\s+)(?P<q>['\"])(?P<path>[^'\"\n]*)(?P=q)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<head>['\"]path['\"]\s*=\s*)(?P<q>['\"])(?P<path>[^'\"\n]*)(?P=q)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<head>\bpath\s*(?:=\s*|\s+))(?P<q>['\"])(?P<path>[^'\"\n]*)(?P=q)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<head>\bfrom\s+)(?P<q>['\"])(?P<path>[^'\"\n]*)(?P=q)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<head>\bread_files\s*\(\s*)(?P<q>['\"])(?P<path>[^'\"\n]*)(?P=q)",
        re.IGNORECASE,
    ),
)


def rewrite_sql_paths(sql: str, by_path: dict[str, str]) -> str:
    """Rewrite only literals in known Databricks SQL path positions."""

    if not by_path or not sql.strip():
        return sql
    keys = ordered_path_keys(by_path)

    def substitute(match: re.Match[str]) -> str:
        target = resolve_path(match.group("path"), by_path, keys)
        if target is None:
            return match.group(0)
        quote = match.group("q")
        return f"{match.group('head')}{quote}{target}{quote}"

    output = sql
    for pattern in _SQL_PATH_POSITIONS:
        output = pattern.sub(substitute, output)
    return output


__all__ = ["XrefRewriteError", "rewrite_sql_paths", "rewrite_sql_tables"]
