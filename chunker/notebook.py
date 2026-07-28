"""Deprecated location: moved to ``pipeline.notebook``.

This shim will be removed in a future release.
"""

import warnings

from pipeline.notebook import (
    CROSS_FILE_NOTEBOOK,
    build_notebook,
    code_cell,
    document_to_cells,
    item_cells,
    markdown_cell,
    markdown_to_cells,
    notebook_to_json,
    notebooks_from_outputs,
    write_notebooks,
)

warnings.warn(
    "chunker.notebook is deprecated; import from pipeline.notebook instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CROSS_FILE_NOTEBOOK",
    "build_notebook",
    "code_cell",
    "document_to_cells",
    "item_cells",
    "markdown_cell",
    "markdown_to_cells",
    "notebook_to_json",
    "notebooks_from_outputs",
    "write_notebooks",
]
