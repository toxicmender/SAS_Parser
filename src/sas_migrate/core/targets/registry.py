"""Pure target registry with no environment or configuration imports."""

from __future__ import annotations

import re

from ..errors import TargetResolutionError
from .models import ResolvedTarget, TargetDefinition, TargetId, TargetSource


def _normalize(value: str) -> str:
    return re.sub(r"[\s_-]+", "", value.casefold())


PYSPARK = TargetDefinition(
    target=TargetId.PYSPARK,
    display_name="PySpark",
    aliases=frozenset({"pyspark", "python", "python3", "py", "sparkpython"}),
    canonical_language="python",
    fence="python",
)
SPARK_SQL = TargetDefinition(
    target=TargetId.SPARK_SQL,
    display_name="Spark SQL",
    aliases=frozenset(
        {"spark_sql", "sparksql", "spark sql", "sql", "databrickssql", "ansi sql"}
    ),
    canonical_language="sql",
    fence="sql",
    sqlglot_dialect="databricks",
)
KNOWN_TARGETS: tuple[TargetDefinition, ...] = (PYSPARK, SPARK_SQL)


def _build_index() -> dict[str, TargetDefinition]:
    index: dict[str, TargetDefinition] = {}
    for definition in KNOWN_TARGETS:
        for value in (definition.target.value, definition.display_name, *definition.aliases):
            key = _normalize(value)
            if key in index and index[key] != definition:
                raise RuntimeError(f"duplicate v2 target alias: {value!r}")
            index[key] = definition
    return index


_BY_ALIAS = _build_index()


def _resolved(
    value: str | TargetId,
    *,
    source: TargetSource,
) -> ResolvedTarget:
    requested = value.value if isinstance(value, TargetId) else value
    definition = _BY_ALIAS.get(_normalize(requested))
    if definition is None:
        supported = ", ".join(target.display_name for target in KNOWN_TARGETS)
        raise TargetResolutionError(
            f"unsupported target {requested!r}; supported targets are {supported}"
        )
    return ResolvedTarget(
        target=definition.target,
        display_name=definition.display_name,
        canonical_language=definition.canonical_language,
        fence=definition.fence,
        source=source,
        requested_value=requested,
    )


def resolve_local_target(
    explicit: str | TargetId | None = None,
    *,
    configured: str | TargetId | None = None,
) -> ResolvedTarget:
    """Resolve explicit, then typed configuration, then Spark SQL."""

    if explicit is not None:
        return _resolved(explicit, source=TargetSource.EXPLICIT)
    if configured is not None:
        return _resolved(configured, source=TargetSource.CONFIG)
    return _resolved(TargetId.SPARK_SQL, source=TargetSource.DEFAULT)


def resolve_sharepoint_target(
    request_value: str | TargetId | None,
    *,
    explicit_fallback: str | TargetId | None = None,
    configured: str | TargetId | None = None,
) -> ResolvedTarget:
    """Resolve request row, command fallback, configuration, then Spark SQL."""

    if request_value is not None:
        return _resolved(request_value, source=TargetSource.REQUEST)
    return resolve_local_target(explicit_fallback, configured=configured)
