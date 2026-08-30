"""Pre-conversion XREF rewriting using the v2 SAS path grammar."""

from __future__ import annotations

import logging
import re
from functools import partial

from sas_migrate.core.sas import PathLocation, SasBatchResult, replace_dataset_names
from sas_migrate.core.sas.paths import PATH_STATEMENTS, PathSpec

from .mapping import ordered_path_keys, resolve_path
from .models import PreRewriteReport, XrefMappings

logger = logging.getLogger(__name__)


def rewrite_datasets(
    result: SasBatchResult, mappings: XrefMappings
) -> SasBatchResult:
    """Apply exact and libref mappings to a v2 batch result."""

    if not mappings.dataset_mapping:
        return result
    return replace_dataset_names(result, mappings.dataset_mapping)


def rewrite_source_text(
    text: str,
    mappings: XrefMappings,
    *,
    source_id: str | None = None,
) -> tuple[str, PreRewriteReport]:
    """Rewrite known filesystem locations and report unresolved macro paths."""

    by_path = mappings.by_path
    if not by_path or not text:
        return text, PreRewriteReport()

    keys = ordered_path_keys(by_path)
    rewritten_paths: dict[str, str] = {}
    unresolved: list[str] = []

    def substitute(match: re.Match[str], *, spec: PathSpec) -> str:
        if spec.location_for(match) is not PathLocation.FILESYSTEM:
            return match.group(0)
        raw = match.group("path")
        if "&" in raw:
            unresolved.append(raw)
            return match.group(0)
        mapped = resolve_path(raw, by_path, keys)
        if mapped is None:
            return match.group(0)
        rewritten_paths[raw] = mapped
        quote = match.group("q")
        return f"{match.group('head')}{quote}{mapped}{quote}"

    output = text
    for spec in PATH_STATEMENTS:
        output = spec.pattern.sub(partial(substitute, spec=spec), output)

    label = source_id or "<inline>"
    if rewritten_paths:
        logger.info("rewrote %d XREF path(s) in %s", len(rewritten_paths), label)
    if unresolved:
        logger.warning(
            "%s has %d unresolved macro path(s); leaving them exactly as written",
            label,
            len(unresolved),
        )
    return output, PreRewriteReport(
        rewritten=rewritten_paths,
        unresolved=tuple(unresolved),
    )


__all__ = ["rewrite_datasets", "rewrite_source_text"]
