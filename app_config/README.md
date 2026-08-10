# app_config

Shared configuration and integrations for the conversion system. This package
is a dependency leaf: domain packages may depend on it, but it does not import
`chunker`, `pipeline`, `memory`, `llm_client`, or `prompt_builder`.

## Configuration resolution

`load_config()` reads the first usable file in this order:

1. The path in `SAS_PARSER_CONFIG`.
2. `config.json` in the current working directory.
3. The repository-root `config.json`.

The result is cached for the process; call `clear_cache()` after changing the
environment in a long-running process or test. `get_value()` returns raw
values, while `get_typed_value()` rejects bad JSON types with a warning. The
normal precedence is explicit constructor argument, then config value, then
code default. JSON `null`, absent sections, and absent files all mean unset.

`llm_client_value()` centralizes the LLM section's schema validation.
`role_value()` layers `llm_client.roles.<role>` sparse overrides over the base
section. Credentials never belong in `config.json`.

## Service modules

| File | Responsibility |
|---|---|
| `azure.py` | Shared Entra ID identity and token caching. |
| `databricks.py` | Workspace credentials, service-principal resolution, and runtime detection. |
| `sharepoint.py` | Microsoft Graph transport for drives, folders, files, and list items. |
| `vault.py` | HashiCorp Vault authentication and secret retrieval. |
| `spark.py` | Local or configured Spark-session creation. |
| `sharepoint_check.py` | Read-only preflight for the SharePoint deployment. |
| `logging_setup.py` | Console/file logging for the command-line entry points. |

SharePoint is transport-only: conversion request semantics live in
[`conversion`](../conversion/README.md). Delta memory table names must be
fully-qualified `Catalog.Schema.Table` identifiers. `utc_stamp()` provides the
common path-safe UTC format for uploaded and local run artifacts.

Logger names follow `app_config.*`.

## Diagnosing a SharePoint deployment

```bash
python -m app_config.sharepoint_check            # the full preflight
python -m app_config.sharepoint_check --offline  # configuration only, no network
```

Stages run in dependency order — `config`, `imports`, `identity`, `secrets`,
`token`, `site`, `base`, then one per configured list — and a stage that cannot
run because an earlier one failed is reported as skipped rather than silently
passed. Nothing is written: no folder created, no file uploaded, no list item
patched.

Three stages carry most of the diagnostic value:

- **`config`** names the source of every resolved setting (`$SHAREPOINT_SITE_ID`
  vs `config.json sharepoint.site_id` vs unset), and shows the base path both as
  written and as normalised. Which of the three sources won is the thing a stack
  trace never says.
- **`secrets`** actually reads SharePoint's service principal out of the
  Databricks secret scope, which is a *different* authentication from the one it
  produces: reaching the scope needs the cluster runtime's credential or a PAT.
  A configured scope is not a readable one, and the most common cluster failure
  lives here — a `!python …` cell inherits `DATABRICKS_RUNTIME_VERSION` but not
  the notebook's credential, so the SDK walks its whole auth chain and reports
  something about the Azure CLI.
- **`token`** decodes the minted token's `roles` claim, which is the granted
  application permissions. A token mints perfectly well with no permissions at
  all and then every call returns 403, so this is where that is visible.

`run_checks(client=...)` takes a pre-built client, matching the injection point
`SharePointClient` itself offers, so the whole preflight is exercised offline in
[`tests/test_sharepoint_check.py`](../tests/test_sharepoint_check.py).

## Boundary

`app_config` is a dependency leaf: it imports nothing from the other
first-party packages, and its optional service dependencies (`msal`,
`msgraph-sdk`, `hvac`, `pyspark`) are all imported lazily inside the call that
needs them. `import app_config.sharepoint` therefore costs nothing and works
without the `sharepoint` extra installed — the SDK is only reached when a real
Graph client is built.

`logging_setup` holds to the same rule and imports only the standard library,
so an entry point can configure logging before anything else is touched.
