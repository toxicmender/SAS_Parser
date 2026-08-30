"""XREF mapping, rewriting and application services."""

from .mapping import classify_rows, ordered_path_keys, resolve_path
from .models import (
    BothRewriteResult,
    ParseFailureMode,
    PreRewriteReport,
    XrefApplyMode,
    XrefMappings,
    XrefRow,
)
from .sas_rewriter import rewrite_datasets, rewrite_source_text
from .service import apply, apply_both, apply_post, apply_pre
from .target_rewriters import (
    XrefRewriteError,
    rewrite_pyspark_paths,
    rewrite_pyspark_tables,
    rewrite_sql_paths,
    rewrite_sql_tables,
    rewrite_sql_target,
)

__all__ = [
    "BothRewriteResult",
    "ParseFailureMode",
    "PreRewriteReport",
    "XrefApplyMode",
    "XrefMappings",
    "XrefRewriteError",
    "XrefRow",
    "apply",
    "apply_both",
    "apply_post",
    "apply_pre",
    "classify_rows",
    "ordered_path_keys",
    "resolve_path",
    "rewrite_datasets",
    "rewrite_pyspark_paths",
    "rewrite_pyspark_tables",
    "rewrite_source_text",
    "rewrite_sql_paths",
    "rewrite_sql_tables",
    "rewrite_sql_target",
]
