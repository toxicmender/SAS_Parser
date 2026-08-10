"""Apply application XREF mappings during conversion. See ``xref/README.md``."""

from __future__ import annotations

# Keep ``apply`` module-qualified to avoid shadowing ``xref.apply``.
from .apply import APPLY_MODES, apply_post, apply_pre
from .pre import PreStats, rewrite_source_text
from .sourcing import XrefMappings, load_databricks_mapping_sharepoint, mappings

__all__ = [
    "APPLY_MODES",
    "PreStats",
    "XrefMappings",
    "apply_post",
    "apply_pre",
    "load_databricks_mapping_sharepoint",
    "mappings",
    "rewrite_source_text",
]
