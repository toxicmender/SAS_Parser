# V2 XREF contracts

Status: Phase 7 implemented. The top-level `xref` package remains shipped as a
legacy compatibility surface until the Phase 10 composition cutover.

## Ownership

- `sas_migrate.application.xref.models` owns source-neutral mapping and rewrite
  result contracts.
- `sas_migrate.application.xref.mapping` owns classification and the only
  exact/longest-directory-prefix path resolver.
- `sas_migrate.application.xref.sas_rewriter` owns pre-conversion dataset and
  physical-path substitution. It imports `PATH_STATEMENTS` from
  `sas_migrate.core.sas.paths`; it does not copy SAS grammar.
- `sas_migrate.application.xref.target_rewriters` owns source-preserving
  PySpark and Databricks SQL table/path rewriting.
- `sas_migrate.application.xref.service` dispatches only resolved `pyspark` and
  `spark_sql` targets and contains no configuration or network imports.
- `sas_migrate.application.ports.xref` defines mapping and transport ports.
- `sas_migrate.adapters.xref` owns local/transport CSV and SharePoint-list
  sources. I/O occurs only when an adapter's `load()` method is invoked.

## Compatibility decisions

- Existing exact dataset, libref and path marker behavior is retained.
- Paths resolve by exact match and then the longest directory prefix, with a
  separator boundary.
- `pre`, `post`, and diagnostic `both` behavior is retained through the v2
  service functions.
- Macro-bearing SAS paths are reported and left unchanged.
- A non-fatal target parse failure returns the original bytes unchanged; fatal
  policy raises `XrefRewriteError` without publishing modified code.
- PySpark rewrites only recognized call literals by source span. Comments,
  formatting, unrelated strings and viable quote style are retained.
- Spark SQL uses SQLGlot's `databricks` dialect for both parsing and emission.
  `sparksql` is only a target alias and is never the SQLGlot dialect.
- Spark Scala has no model, alias, branch, rewriter or fallback.

## Executable gates

- `tests/test_v2_xref.py` covers contracts, resolver order, shared grammar,
  unresolved reporting, SQL/Python/path rewriting, byte-identical fallback,
  two-target dispatch, lazy sources, and explicit Databricks dialect capture.
- `tests/test_xref.py` remains the legacy characterization suite and runs next
  to the v2 suite during migration.
- `scripts/check_v2_architecture.py` ensures the application cannot import its
  adapters and adapters cannot coordinate through one another.
- `scripts/check_v2_legacy_inventory.py` keeps the legacy package present but
  records the replacement and closes G-006 without claiming Phase 10 cutover.
- The installed-wheel smoke imports both XREF package families and checks their
  runtime files.

## Remaining cutover work

Phase 9 conversion composition must select the concrete XREF source and pass
the resulting `XrefMappings` into the service. Phase 10 then removes the legacy
package, its config-driven facade, compatibility tests, and wheel references.
That remaining work is tracked by G-010, G-012, and G-021 rather than by a
duplicate XREF gap.
