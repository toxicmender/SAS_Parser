# Phase 9 infrastructure contracts

This slice moves the safe half of runtime composition into final v2 owners:

- `sas_migrate.config` owns immutable, versioned Azure, Vault, Databricks,
  SharePoint, and observability settings;
- the settings document rejects credential fields, while environment values
  override non-secret JSON settings;
- `sas_migrate.application.ports` owns credential and access-token boundaries;
- `sas_migrate.adapters.credentials` owns environment, Databricks secret-scope,
  Vault KV, and ordered fallback adapters;
- `sas_migrate.adapters.auth` owns lazy MSAL client-credential token issuance;
- `sas_migrate.observability` owns shared message/traceback redaction and HTTP
  transport log policy;
- `sas_migrate.adapters.sharepoint` owns the lazy Graph SDK gateway, a blocking
  facade backed by one persistent event loop and worker thread, and the
  versioned read-only deployment preflight.

Credential values use Pydantic `SecretStr`, so their representation and JSON
serialization are masked. The adapters still treat logs and diagnostic output
as sensitive: redaction is a safety net, not permission to publish them.

The main test job holds the combined settings/auth/observability/SharePoint implementation
above 90% line-and-branch coverage. A separate no-skip job installs and imports
the real `azure`, `vault`, `databricks`, and `sharepoint` extras, then reruns the
adapter contracts. This closes G-017.

G-012 is closed. File, folder, and list operations now satisfy the conversion
and XREF transport shapes directly, calls made from notebook or Databricks
event-loop threads remain pinned to one private loop, and Graph errors are
normalized through v2 redaction. The preflight runs configuration, optional
dependency, token-role, site/default-drive, base-directory, and configured-list
checks in dependency order and exposes only read operations. The compatibility
runtime still uses `app_config` until the Phase 10 entry-point cutover; it is no
longer the only owner of this infrastructure behavior.
