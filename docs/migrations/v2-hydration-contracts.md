# V2 hydration contracts and migration boundary

Phase 9 now owns hydration planning and execution contracts under
`sas_migrate.application.hydration`, and owns optional runtime integration under
`sas_migrate.adapters.hydration`. The legacy `data_hydration` package remains
shipped only for the old CLI and configuration composition until Phase 10.

## Owned by v2

- Immutable, schema-versioned settings, source, partition, item, plan, outcome,
  and report contracts.
- Pure projection of v2 SAS engine/path references into hydration work.
- Target rendering, per-item blockers, one run date, corpus-wide overwrite then
  append ordering, SAS index hints, and probe-driven partition selection.
- Failure-isolated execution through driver-registry and sink ports.
- Lazy driver selection and concrete readers for local files and SAS datasets,
  with actionable optional-extra errors.
- Seekable ranged I/O, filesystem SPDE discovery, and managed Delta writes.
- A mandatory optional-driver import matrix and a real containerized PySpark /
  Delta sink contract.

## Deliberate v2 corrections

Static planning never lists an SPDE directory. Component discovery is an
adapter probe, so `probe=None` is now literally side-effect free. A source also
stores `source_name` separately from its logical `object_name`; this preserves
the physical extension (`sales.csv`) while still rendering a safe table name
(`sales`). The legacy planner discarded that extension before its local reader
opened the file.

## Operational composition

`sas-migrate hydrate PLAN.json [--dry-run]` validates a versioned
`HydrationPlan` and emits a versioned `HydrationReport`. Dry-run mode does not
load a source driver or resolve Spark. Live plans compose the concrete local
file or SAS dataset reader with the managed Delta sink, preserve per-item
failure isolation, and return status 1 when any item fails. Invalid contracts,
empty plans, batch sizes, and output paths return status 2.

## Deferred adapters

Credentials, environment/config-file resolution, Azure/Vault/Databricks auth,
SharePoint loop ownership, redaction, and observability belong to G-012. The v2
hydration settings contract contains no password, key, secret scope value, or
credential object. Concrete Oracle, SFTP, ADLS, Blob, SPDE, and SAS-session
drivers remain open under G-011. The optional-driver CI matrix currently proves
that their SDKs install and import; it must exercise real driver construction
before G-011 closes.

## Executable evidence

`tests/test_v2_hydration.py` covers legacy parity, blockers, target names,
partition boundaries, failure isolation, lazy loading, ranged I/O, and the
Delta adapter at 95% combined line/branch coverage. The dedicated
`hydration-drivers` CI job installs the full hydration extra and requires all
eight source-kind imports without skips. The Spark/Delta container additionally
runs `tests/test_v2_hydration_delta.py` against a real managed Delta table.
`tests/test_v2_cli.py` covers dry-run, live composition, unconfigured-driver
failure, invalid operator input, and report output, while the clean-wheel smoke
requires a dependency-free hydration dry-run.
