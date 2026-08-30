# Phase 9 conversion contracts

The Phase 9 conversion slice replaces the orchestration boundary of the
top-level `conversion/` package without importing legacy configuration, Graph,
or pipeline implementations into v2.

## Ownership

- `sas_migrate.application.conversion` owns versioned request, preference,
  command, result, batch, and lifecycle contracts.
- `ConversionWorkflow` filters pending rows, selects the first request-scoped
  model preference, resolves only PySpark or Spark SQL, and processes every row
  independently.
- `sas_migrate.application.ports.conversion` defines request, source, and
  translation ports.
- `sas_migrate.adapters.conversion.local` discovers deterministic `.sas` and
  `.txt` sources from a local directory.
- `sas_migrate.adapters.conversion.sharepoint` projects the existing internal
  list-column names and drive-relative source layout over a narrow injected
  transport. Graph and authentication remain infrastructure concerns.

Spark SQL is resolved through the v2 target registry. The registry definition
continues to require SQLGlot's `databricks` dialect; `sparksql` is never passed
as a SQLGlot dialect.

## Lifecycle and failure behavior

A persisted run writes `In Progress` followed by exactly one terminal state.
Translation, target, source, and terminal-status failures become failed row
outcomes and do not stop later requests. Dry runs neither write request status
nor bypass target/source/translation validation.

The translation port returns already-persisted artifact locators. Concrete
prompt/notebook persistence remains in the existing v2 translation artifact
service; SharePoint publication and the legacy CLI cutover remain Phase 10
work under G-014 and G-013.

## Verification

`tests/test_v2_conversion.py` covers local and SharePoint fake-adapter flows,
model precedence, Databricks SQL dialect selection, Scala rejection, source
ordering, malformed/unreadable row tolerance, dry runs, status-write failures,
empty sources, and per-row isolation. CI enforces at least 90% combined line
and branch coverage for the application, port, and adapter family; the phase
suite measured 98% when this gap was closed.
