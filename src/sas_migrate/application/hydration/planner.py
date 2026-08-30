"""Side-effect-free projection of SAS references into hydration work."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from sas_migrate.application.ports.hydration import HydrationSourceProbe

from .models import (
    HydrationItem,
    HydrationPlan,
    HydrationSettings,
    HydrationSource,
    SourceKind,
    WriteMode,
)
from .naming import TableNameError, render, validate_template
from .partitioning import plan_partitions

if TYPE_CHECKING:
    from sas_migrate.core.sas import SasEngineRef, SasPathRef

REMOTE_KINDS = {"ftp": SourceKind.SFTP, "sftp": SourceKind.SFTP, "azure": SourceKind.BLOB}
SAS_DATA_SUFFIX = ".sas7bdat"
SAS_INDEX_SUFFIX = ".sas7bndx"
UNRESOLVED_TARGET = "<unresolved>"


def _value(value: Any) -> str:
    return str(getattr(value, "value", value)).casefold()


def _basename(path: str) -> str:
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _stem(path: str) -> str:
    tail = _basename(path)
    return tail.rsplit(".", 1)[0] if "." in tail else tail


def _directory(path: str) -> str:
    cleaned = path.replace("\\", "/").rstrip("/")
    return cleaned.rsplit("/", 1)[0] if "/" in cleaned else ""


def _engine_source(ref: SasEngineRef, source_id: str | None) -> HydrationSource:
    options = ref.option_map
    return HydrationSource(
        kind=SourceKind.ORACLE,
        engine=ref.engine,
        locator=options.get("path", "") or options.get("server", ""),
        object_name=options.get("schema", "") or ref.binds,
        libref=ref.binds,
        options=ref.options,
        has_macro_ref=ref.has_macro_ref,
        source_id=source_id,
    )


def _path_source(ref: SasPathRef, source_id: str | None) -> HydrationSource | None:
    if ref.statement in {"include", "ods", "sasautos", "file", "proc_export"}:
        return None
    if ref.path.casefold().endswith(SAS_INDEX_SUFFIX):
        return None

    location = _value(ref.location)
    if ref.engine == "spde":
        kind = SourceKind.SPDE
    elif location == "remote":
        kind = REMOTE_KINDS.get(ref.device or "")
        if kind is None:
            return None
    elif location != "filesystem":
        return None
    elif ref.path.casefold().endswith(SAS_DATA_SUFFIX):
        kind = SourceKind.SAS_DATASET
    else:
        kind = SourceKind.FILE

    if kind is SourceKind.SPDE:
        return HydrationSource(
            kind=kind,
            engine=ref.engine,
            locator=ref.path,
            object_name=ref.binds or _stem(ref.path),
            libref=ref.binds,
            has_macro_ref=ref.has_macro_ref,
            source_id=source_id,
        )
    if kind is SourceKind.FILE and "." not in _basename(ref.path):
        return HydrationSource(
            kind=kind,
            locator=ref.path,
            libref=ref.binds,
            has_macro_ref=ref.has_macro_ref,
            source_id=source_id,
        )
    return HydrationSource(
        kind=kind,
        engine=ref.engine,
        locator=_directory(ref.path),
        object_name=_stem(ref.path),
        source_name=_basename(ref.path),
        libref=ref.binds,
        has_macro_ref=ref.has_macro_ref,
        source_id=source_id,
    )


def _macro_blocker(source: HydrationSource) -> tuple[str, ...]:
    if not source.has_macro_ref:
        return ()
    unresolved = sorted(key for key, value in source.options if "&" in value)
    where = f"option(s) {', '.join(unresolved)}" if unresolved else "the connection"
    return (f"unresolved macro reference in {where}",)


def _library_blocker(source: HydrationSource) -> tuple[str, ...]:
    if source.kind is not SourceKind.FILE or source.object_name:
        return ()
    return (f"'{source.locator}' is a library directory, not a single dataset",)


def _index_note(source: HydrationSource, refs: Iterable[SasPathRef]) -> tuple[str, ...]:
    if source.kind is not SourceKind.SAS_DATASET:
        return ()
    for ref in refs:
        if ref.path.casefold().endswith(SAS_INDEX_SUFFIX) and _stem(ref.path).casefold() == source.object_name.casefold():
            return ("a SAS index sits beside this dataset; its columns are a candidate CLUSTER BY",)
    return ()


def _target_for(source: HydrationSource, settings: HydrationSettings) -> tuple[str, tuple[str, ...]]:
    try:
        return render(
            settings.table_template,
            catalog_name=settings.catalog,
            schema_name=settings.schema_name or source.libref,
            table_name=source.object_name,
            stage=settings.stage,
            date=settings.run_date,
            libref=source.libref,
            source=source.kind.value,
        ), ()
    except TableNameError as exc:
        return UNRESOLVED_TARGET, (f"no target table name: {exc}",)


def build_plan(
    engine_refs: Sequence[SasEngineRef] = (),
    path_refs: Sequence[SasPathRef] = (),
    *,
    settings: HydrationSettings | None = None,
    probe: HydrationSourceProbe | None = None,
    source_id: str | None = None,
) -> HydrationPlan:
    return build_corpus_plan(
        {source_id or "": (engine_refs, path_refs)}, settings=settings, probe=probe
    )


def build_corpus_plan(
    by_source: Mapping[str, tuple[Sequence[SasEngineRef], Sequence[SasPathRef]]],
    *,
    settings: HydrationSettings | None = None,
    probe: HydrationSourceProbe | None = None,
) -> HydrationPlan:
    settings = settings or HydrationSettings()
    validate_template(settings.table_template)
    sources: list[HydrationSource] = []
    path_refs: list[SasPathRef] = []
    for source_id, (engine_refs, refs) in by_source.items():
        sources.extend(_engine_source(ref, source_id or None) for ref in engine_refs)
        sources.extend(source for ref in refs if (source := _path_source(ref, source_id or None)))
        path_refs.extend(refs)

    items: list[HydrationItem] = []
    seen_tables: set[str] = set()
    for source in sources:
        target, target_blockers = _target_for(source, settings)
        library_blockers = _library_blocker(source)
        blockers = list(_macro_blocker(source)) + list(library_blockers)
        if not library_blockers:
            blockers.extend(target_blockers)
        if source.kind is SourceKind.SPDE and not settings.sas_session_configured:
            blockers.append("SPD Engine requires a configured SAS session")

        partitioned = plan_partitions(source, num_partitions=settings.num_partitions, probe=probe)
        notes = _index_note(source, path_refs)
        first = target not in seen_tables
        seen_tables.add(target)
        partitions = partitioned.partitions or (None,)
        for index, partition in enumerate(partitions):
            items.append(
                HydrationItem(
                    source=source,
                    target_table=target,
                    write_mode=WriteMode.OVERWRITE if first and index == 0 else WriteMode.APPEND,
                    strategy=partitioned.strategy,
                    strategy_reason=partitioned.reason,
                    partition=partition,
                    notes=notes,
                    blockers=tuple(blockers),
                )
            )
    return HydrationPlan(run_date=settings.run_date, items=tuple(items))


__all__ = ["UNRESOLVED_TARGET", "build_corpus_plan", "build_plan"]
