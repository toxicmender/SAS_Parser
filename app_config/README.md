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

SharePoint is transport-only: conversion request semantics live in
[`conversion`](../conversion/README.md). Delta memory table names must be
fully-qualified `Catalog.Schema.Table` identifiers. `utc_stamp()` provides the
common path-safe UTC format for uploaded and local run artifacts.

Logger names follow `app_config.*`.
