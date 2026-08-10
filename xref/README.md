# xref

Application-specific SAS-to-Unity-Catalog cross-reference handling. The
package reads XREF mappings from SharePoint or a document-library CSV and
applies them during conversion without giving `chunker` a network dependency.

## Modes

- `pre` (default) rewrites known SAS-side dataset references before
  translation. It is the preferred mode because chunk metadata has already
  identified the references.
- `post` rewrites generated Spark SQL and PySpark names that were not found in
  the SAS-side extraction.
- `both` runs both passes and reports their difference, providing evidence for
  choosing the safer mode for an application.

`pre.py` separately rewrites physical paths in `LIBNAME`, `INFILE`, and
`%INCLUDE`; these are not dataset identifiers. `rewrite.py` uses `sqlglot` for
Spark SQL and Python `ast` spans for PySpark, preserving source layout where
possible.

## Package layout

| File | Responsibility |
|---|---|
| `sourcing.py` | Load and classify XREF rows or CSV mappings. |
| `apply.py` | Resolve and run `pre`, `post`, or `both`. |
| `pre.py` | Rewrite physical-path statements. |
| `rewrite.py` | Rewrite generated SQL/Python dataset references. |

`xref.apply.apply` remains module-qualified so it cannot be confused with the
`xref.apply` module. `apply_pre()` and `apply_post()` are re-exported for the
common direct cases.

Logger names follow `xref.*`.
