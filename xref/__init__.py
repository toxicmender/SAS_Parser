"""Dataset-name cross-reference: SharePoint XREF rows applied to a conversion.

The XREF list maps a SAS dataset name to its Unity Catalog target —
``schema.table`` to ``catalog.schema.table`` — per application. Three modules:

* :mod:`xref.sourcing` — reading and classifying the rows. Source-agnostic in
  signature, because the reference deployment also has a file-based XREF
  (``sftp_config.xref_file_path``) that will want to slot in beside the list.
* :mod:`xref.apply` — *when* the substitution happens: ``"pre"`` (before
  conversion, over the SAS-side metadata), ``"post"`` (after, over the
  generated code), or ``"both"``.
* :mod:`xref.pre` — the other half of ``"pre"``: the physical paths in
  ``LIBNAME`` / ``INFILE`` / ``%INCLUDE``, which are not dataset names and so
  are not something ``replace_dataset_names`` can or should reach.
* :mod:`xref.rewrite` — the post-conversion rewriter, parsing generated Spark
  SQL with ``sqlglot`` and generated PySpark with the ``ast`` module.

Why the split matters
---------------------
``chunker`` stays network-free: it knows how to substitute dataset names
(:func:`chunker.batcher.replace_dataset_names`) and nothing about where a
mapping came from. This package owns SharePoint and hands that function the
shape it already understands, so **chunker/batcher.py is not modified** by any
of this — in the present design or the physical-path one sketched in
:mod:`xref.sourcing`.

Pre vs post
-----------
``"pre"`` is the default and the stronger position: the chunker has already
identified what is a dataset reference, so the substitution is applied to
known dataset names rather than to whatever a regex thinks one looks like.
``"post"`` exists because generated code can name a table the SAS-side
extraction never saw. ``"both"`` runs each and reports the difference, which
is the evidence for choosing between them — see
:func:`xref.apply.apply_both`.

Logger names: ``xref.*``.
"""

from __future__ import annotations

# `apply` (the mode dispatcher) is deliberately NOT re-exported: the name
# would shadow the `xref.apply` submodule on this package. Reach it as
# `xref.apply.apply`, or call apply_pre / apply_post directly.
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
