# xref

Application-specific SAS-to-Unity-Catalog cross-reference handling. The
package reads XREF mappings from SharePoint or a document-library CSV and
applies them during conversion without giving `chunker` a network dependency.

## Modes

Selected by `--xref-apply`, then `XREF_APPLY`, then `config.json` `xref.apply`,
then the default. `conversion.run` honours all three; `--no-xref` turns the
substitution off entirely and beats every one of them.

- `pre` (default) rewrites known SAS-side references before translation. It is
  the preferred mode because chunk metadata has already identified them.
- `post` rewrites the generated Spark SQL and PySpark, catching what the
  SAS-side extraction never saw — at the cost of parsing model output.
- `both` runs each and reports their difference, which is the evidence for
  choosing the safer mode for an application.

## Two namespaces, both ends

Each mode handles **dataset names and physical paths**, from the two slots
`XrefMappings` sorts rows into:

| | `pre` | `post` |
|---|---|---|
| Dataset names (`exact`, `by_libref`) | `chunker.batcher.replace_dataset_names` over the batch result | `rewrite.py` — `sqlglot` for Spark SQL, Python `ast` spans for PySpark |
| Physical paths (`by_path`) | `pre.py` over the raw source, before chunking | `rewrite.py` — reader/writer and `dbutils.fs` calls, `LOCATION` and friends |

A path reaches the generated code when no mapping row covered it as `pre` swept
the source, or when the model writes one of its own; that is the gap `post`'s
path half closes. Both halves resolve a key through `mapping.py` — exact match,
then longest directory prefix — so one path can never come out of the two ends
as two different targets.

Where a path appears in *SAS* is `chunker/paths.py`'s grammar, imported here.
Where one appears in *generated* code is `rewrite.py`'s, since it is target
syntax rather than source syntax.

Under `pre` and `both` the run also reports SAS paths that survived into the
output unmapped — a location the notebook names that will not exist on the
target.

## Package layout

| File | Responsibility |
|---|---|
| `sourcing.py` | Load and classify XREF rows or CSV mappings. |
| `apply.py` | Resolve and run `pre`, `post`, or `both`. |
| `mapping.py` | Which mapping key wins, and what it rewrites to. Shared by both halves. |
| `pre.py` | Rewrite physical-path statements in SAS source. |
| `rewrite.py` | Rewrite generated SQL/Python — dataset references and paths. |

`xref.apply.apply` remains module-qualified so it cannot be confused with the
`xref.apply` module. `apply_pre()` and `apply_post()` are re-exported for the
common direct cases.

Logger names follow `xref.*`.
