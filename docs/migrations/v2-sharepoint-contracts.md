# Phase 9 SharePoint Graph contracts

The v2 SharePoint boundary is now owned by
`sas_migrate.adapters.sharepoint` and does not import `app_config`.

`GraphSdkGateway` imports the optional Microsoft Graph/Kiota stack only when a
client is first requested. `SharePointGraphTransport` is the blocking facade
used by conversion and XREF adapters. One `SingleLoopWorker` owns one persistent
event loop on one thread for the lifetime of the transport, keeping Kiota and
HTTP connection pools on the loop where they were created even when calls come
from a Jupyter or Databricks thread with an event loop already running.

The transport preserves the legacy file and list surface: paged directory and
list reads, extension filtering, BOM-tolerant source decoding, JSON reads,
simple uploads, idempotent nested-folder creation, item reads, and partial item
updates. Site-default drive resolution is cached when no explicit drive id is
configured. Errors retain HTTP, Graph code, and request-id diagnostics after
secret redaction.

`SharePointPreflightReport` is a schema-v2 contract. Its service checks config,
optional imports, token metadata and roles, site/default-drive resolution, the
configured base directory, and one sample from every configured list. Checks
run in dependency order, blocked checks are explicit skips, offline mode stops
after config/import validation, and the probe exposes no mutation method.

CI evidence:

- the focused suite holds the package at 96% combined line/branch coverage;
- running-loop and worker self-reentry contracts protect notebook execution;
- the no-skip infrastructure job installs the real `sharepoint` extra and
  constructs the actual SDK client without making a network call;
- architecture and legacy-inventory checks prove there is no `app_config`
  import;
- the installed-wheel smoke requires all three SharePoint implementation files.

Phase 10 now exposes the preflight as:

```text
sas-migrate check sharepoint [--config SETTINGS.json] [--offline]
                             [--output REPORT.json]
```

Offline mode constructs neither an access-token provider nor a Graph client.
Live mode composes the Graph transport with `AZURE_CLIENT_SECRET` for the local
Entra principal, or lazily reads the dedicated tenant/client/secret keys from
`sharepoint.secret_scope` on Databricks. The versioned JSON report is written
to stdout unless `--output` is supplied, and the command exits 1 for a failed
check or 2 for invalid operator input. `convert sharepoint` remains the next
SharePoint CLI cutover; the legacy entry point is still a compatibility
consumer, not an architectural dependency of v2.
