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
  transport log policy.

Credential values use Pydantic `SecretStr`, so their representation and JSON
serialization are masked. The adapters still treat logs and diagnostic output
as sensitive: redaction is a safety net, not permission to publish them.

The main test job holds the combined settings/auth/observability implementation
above 90% line-and-branch coverage. A separate no-skip job installs and imports
the real `azure`, `vault`, `databricks`, and `sharepoint` extras, then reruns the
adapter contracts. This closes G-017.

G-012 remains open, but is narrower. The active application still composes the
legacy `app_config` Graph transport, its single-event-loop worker, and the
deployment preflight. Those pieces move together so the v2 SharePoint adapter
does not accidentally run Kiota clients on multiple event loops. Phase 10 will
then switch operational entry points after the concrete adapter is green.
