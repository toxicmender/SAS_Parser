# Spec: config, vault/azure, llm_client, SharePoint, XREF and complexity refactor

Status: **not started.** This document is the agreed plan; no code has been
changed. Breaking changes are permitted and deprecated code is to be pruned
rather than deprecated in place.

Revision 3 — pruning inventory added (§10).
Revision 2 — re-evaluated against the references. Q1 and Q2 closed (§12); the
SharePoint transport delta measured rather than assumed (C8, §7); the
`powerapps` deletion found to have a consumer (C9, §8.1).

Scope:

1. Evaluation of the current implementation against the reference system.
2. Reference behaviour catalogue — what the production scripts actually do.
3. Target `config.json` schema.
4. Phase 1 — config restructure.
5. Phase 2 — auth blockers (`vault` / `azure`).
6. Phase 3 — `llm_client` gateway conformance.
7. Phase 4 — SharePoint transport.
8. Phase 5 — domain parts: `conversion`, `xref`, `complexity` SharePoint flow.
9. Phase 6 — `complexity` internal split.
10. Pruning and deprecation removal.
11. Decisions, open questions, out of scope.

---

## 1. Evaluation of the current state

The reference system is a set of Databricks-hosted scripts
(`SAS_Conversion_Agent/pyscripts/`): `hashicorp.py`, `databricks_utils.py`,
`llm_client.py`, `sharepoint_utils.py`, plus `config/configurations_dev.json`.
Our implementation is structurally sound and in several respects better —
typed config with graceful degradation, environment-variable precedence,
`Retry-After`-aware retry, token budgeting, prompt-cache breakpoints, TLS
pinning, per-request timeouts, and raising instead of returning `False`. None
of that is to be lost.

The gaps fall into three tiers.

### 1.1 Blockers — a real run fails today

| # | Finding | Location |
|---|---|---|
| B1 | The `azuread` Vault login requests `<client_id>/.default`. The deployment's Vault role is bound to the ARM audience — `sp_scope` is an explicit config key in the reference, set to `https://management.azure.com//.default`. Our `_ARM_DEFAULT_SCOPE` is only a last resort and is unreachable whenever a client id exists, which on the Databricks-SPN path is always. | `app_config/vault.py:298` |
| B2 | `verify` is resolved and handed to `hvac`, but never to MSAL. The reference passes `verify=verify` into `ConfidentialClientApplication`. That knob exists because the network is TLS-intercepted, so the Entra token call fails before Vault is ever contacted. | `app_config/azure.py:372` |
| B3 | The `ai-gateway-version` header is never sent. The reference sets `"ai-gateway-version": gateway_version` (`"v2"`) on every request, in both its ChatOpenAI and its raw-OpenAI path. A repo-wide search returns zero occurrences of `gateway_version` or `ai-gateway-version`; the only mechanism is the generic `url_headers` dict, which ships `null`. | `llm_client/client.py:749` |

### 1.2 Correctness and robustness

| # | Finding | Location |
|---|---|---|
| C1 | The Vault token is minted once per process and never refreshed. The reference re-authenticates on every read. JWT-role tokens are short-lived, so a long pipeline run starts receiving 403s mid-flight. | `app_config/vault.py:482` |
| C2 | `model_provider` is parsed and then discarded. The reference splits `"provider: model"` and uses the *provider* to select the client class, routing `anthropic` through raw `openai.OpenAI` rather than `ChatOpenAI`. We log the prefix away and send everything through `ChatOpenAI`. We also do not lowercase the model id; the reference does. | `llm_client/client.py:590` |
| C3 | `AI_GATEWAY_PATH = "appsvc/ai_gateway"` is a guess. The reference builds `/v1/secret/data/{hashicorp_app_name}/` and appends `credentials.ai_cred_path` (`"ai_gateway"`), so the real path is `{app_name}/ai_gateway`. Our `f"{oidc_role}/ai_gateway"` fallback is correct; only the constant is wrong. `credentials.ai_cred_key` is `"token"`, which is already first in `_AI_GATEWAY_TOKEN_KEYS`. | `app_config/vault.py:100`, `:514` |
| C4 | The `approle` branch does not wrap `hvac` exceptions in `VaultError`, unlike the `azuread` branch, so callers cannot rely on `except VaultError`. | `app_config/vault.py:319` |
| C5 | SharePoint authenticates as a *different* service principal: `saact-hsv-tenantid` / `saact-hsv-appid` / `saact-hsv-secret`, in the same Databricks secret scope. Our `SECRET_KEY_*` constants hardcode `sp-hsv-*`, so the SharePoint principal is currently unreachable. | `app_config/databricks.py:152` |
| C6 | `file_server_base_path` is stored with the document-library prefix (`"Shared Documents/…"`), which the reference strips to obtain a drive-relative base. `_drive_item_id` assumes callers already pass drive-relative paths, so we would look for a folder literally named `Shared Documents`. | `app_config/sharepoint.py:107` |
| C7 | `main()` in the complexity CLI welds source discovery to `args.sas_dir.rglob` and delivery to `args.out_dir`; `_run_evaluation` reaches for `args.out_dir` again. There is no seam for an alternative source or sink. | `complexity/__main__.py:214`, `:251`, `:384` |
| C8 | `SharePointClient`'s surface does not match what the domain layers need, and one primitive is missing outright. See §7 for the full delta. Most importantly there is **no list-item write** of any kind, so `update_request_status` has no transport beneath it. | `app_config/sharepoint.py:486`–`:696` |
| C9 | `app_config/powerapps.py` has a live consumer: `demo_run.py` imports `PowerAppsConfig` / `PowerAppsError` lazily inside one function, and documents `POWERAPPS_LIST_NAME` in its module docstring. Deleting the module requires migrating that function, not just removing the file. | `demo_run.py:881`, `:96` |

### 1.3 Never built

- **Databricks workspace publishing.** The reference has `workspace_directory_exists` (`GET /api/2.0/workspace/list`), `create_workspace_directory` (`POST /api/2.0/workspace/mkdirs`) and `write_content_to_notebook` (`POST /api/2.0/workspace/import`, `format=JUPYTER`, base64, `overwrite=true`). We build nbformat notebooks correctly (`pipeline/notebook.py:319`) but `write_notebooks` only writes `.ipynb` to a local directory (`pipeline/notebook.py:581`). The reference's `ai_validator_config.output_folder` is a `/Workspace/…` path, so validation output currently has nowhere to go.
- **SharePoint list ids.** `SharePointConfig` has no notion of the three (now four) lists the run is driven by.
- **Per-role gateway config.** The reference has `ai_gateway_details` (timeout 6000) and `ai_validator_config` (timeout 12000) as siblings. Our single `llm_client` section cannot express that; callers can only override `model=` (`complexity/__main__.py:370`, `validation/__main__.py:252`, `validation/judge.py`).
- **`adls_config`**, **`sftp_config`**, **`databricks_config.dbfs_root_path`**, **`sas_config.inlining`** — no counterparts.

### 1.4 Confirmed correct

Verified against the reference and requiring no change:

- Trailing-slash `base_url` plus model-as-final-path-segment
  (`llm_client/client.py:746`). The reference's `ai_gateway_url` ends in `/`
  and it does `f"{ai_gateway_url}{model_name}"`.
- `api_key=` plus an `api-key` default header (`llm_client/client.py:750`).
- `cert_file` → the reference's `ai_cert_path` (`gateway.crt`); we additionally
  pin an explicit httpx `SSLContext` for both sync and async.
- Vault auth mount `jwt/azuread/inspirewellness` (`app_config/vault.py:89`).
- KV mount `secret`, KV v2, `resp["data"]["data"]` (`app_config/vault.py:463`).
- `X-Vault-Namespace` via `hvac.Client(namespace=…)`.
- Service-principal secret keys `sp-hsv-appid` / `-secret` / `-tenantid`
  (`app_config/databricks.py:152`) for the *Vault* principal.
- `AZURE_DATABRICKS_SCOPE = "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default"`
  (`app_config/databricks.py:137`) — character-for-character.
- Authority host `https://login.microsoftonline.com`.
- Defaults `max_tokens` 64000, `temperature` 0.0, `max_retries` 4.
- `vault.verify` semantics (bool or CA-bundle path).
- The split of `adb_secret_scope_name` and `hashicorp_app_name` into
  `databricks.secret_scope` and `vault.oidc_role` — the reference's values are
  different strings, so two keys is the right shape.

### 1.5 Already built, and reusable as-is

- **`chunker.batcher.replace_dataset_names` (`chunker/batcher.py:555`)** maps SAS
  dataset names to `catalog.schema.table`. `_split_databricks_mapping`
  (`:389`) classifies by dot count — a dotted key is an exact dataset name
  whose target should be three-part, a bare key is a libref prefix. The XREF
  list supplies `schema.table → catalog.schema.table`, which is exactly the
  exact-key shape. **`batcher.py` requires no changes.** It rewrites chunk
  metadata and `%let` values in source text, canonicalises one-level names to
  `work.`, and collapses duplicates.
- **`_map_ds` (`chunker/batcher.py:428`)** early-returns on quoted physical
  paths and on names containing `&`. The quoted-path guard is the hook for
  future path remapping.
- **`SasSemanticChunker.chunk_text(source, source_id=…)`
  (`chunker/chunker.py:160`)** — `chunk_file` is a read plus a call to it, so
  SharePoint-sourced text needs no temporary files.
- **`WrittenReports.paths` (`complexity/report.py:82`)** enumerates every
  written path, overall report first — the upload set, already assembled.
- **`sqlglot>=25`** is declared under the `sql` optional extra
  (`pyproject.toml:61`).

---

## 2. Reference behaviour catalogue

Recorded so the implementation can be checked without re-reading the source
screenshots.

### 2.1 `hashicorp.py`

```
vault_details      = configManager.get_config_value("vault")
hashicorp_url      = vault_details['key_vault_url']
hashicorp_namespace= vault_details['namespace']
hashicorp_app_name = vault_details['hashicorp_app_name']
verify             = vault_details['verify']
adb_secret_scope_name = vault_details['adb_secret_scope_name']
```

`configure_spn_and_hashicorp(dbutils)`
- `sp_client_id`     = `dbutils.secrets.get(scope=adb_secret_scope_name, key='sp-hsv-appid')`
- `sp_client_secret` = `… key='sp-hsv-secret'`
- `sp_tenant_id`     = `… key='sp-hsv-tenantid'`
- `sp_authority_url` = `https://login.microsoftonline.com/{sp_tenant_id}`
- `sp_scope`         = `["https://management.azure.com//.default"]`
- `hashicorp_login_endpoint` = `{hashicorp_url}/v1/auth/jwt/azuread/inspirewellness/login`
- `hashicorp_secret_url`     = `{hashicorp_url}/v1/secret/data/{hashicorp_app_name}/`

`get_spn_jwt_token(dbutils)` — `msal.ConfidentialClientApplication(client_id,
client_credential, authority, verify=verify)` →
`acquire_token_for_client(scopes=sp_scope)` → `token["access_token"]`.

`get_vault_token(dbutils)` — POST to the login endpoint, headers
`{"Content-type": "application/json", "x-Vault-Namespace": namespace}`, body
`{"role": hashicorp_app_name, "jwt": spn_jwt}`, `verify=verify`. On 200 returns
`resp["auth"]["client_token"]`; **otherwise returns `False`** (we raise instead
— deliberate divergence, kept).

`get_hashicorp_key_value(dbutils, vault_secret_path, hashicorp_key)` — GET
`{hashicorp_secret_url}{vault_secret_path}` with `X-Vault-Namespace` and
`X-Vault-Token`, returns `resp["data"]["data"][hashicorp_key]`.

Note: the role name, the secret-path prefix and `hashicorp_app_name` are one
value. No timeouts are set on any request. No token caching.

### 2.2 `databricks_utils.py`

`get_aad_token()` — POST
`https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token`,
`grant_type=client_credentials`,
`scope=2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default`,
`Content-Type: application/x-www-form-urlencoded`, `raise_for_status()`,
returns `response.json()["access_token"]`. No caching.

`workspace_directory_exists(workspace_url, directory_path)` — `GET
{workspace_url}/api/2.0/workspace/list`, `Authorization: Bearer`,
`json={"path": directory_path}`, returns `status_code == 200`.

`create_workspace_directory(workspace_url, directory_path)` — `POST
/api/2.0/workspace/mkdirs`, `json={"path": …}`, `raise_for_status()`.

`write_content_to_notebook(notebook_path, workspace_url, content,
language="sql")` — builds a single-code-cell notebook, `"source":
content.splitlines(keepends=True)`, `nbformat: 4`, `nbformat_minor: 0`, then
`POST /api/2.0/workspace/import` with `{"path", "format": "JUPYTER",
"language", "content": base64(json), "overwrite": "true"}`.

We emit nbformat 4.5 and drive language from the target-language config;
Databricks' JUPYTER import accepts both, so no change is needed there.

### 2.3 `llm_client.py`

`parse_preferred_llm(preferred_llm)` — `"provider: model"` split on `:`;
provider `.strip()`, model `.strip().lower()`. Sourced from a SharePoint
column, not from config.

`init_llm_model(*, key, gateway_version, model_name, base_url, max_tokens_val,
temperature, timeout_val, max_retries, model_kwargs=None, **kwargs)` —
`ChatOpenAI(model=…, base_url=…, api_key=key, default_headers={"api-key": key,
"ai-gateway-version": gateway_version}, timeout=…, max_retries=…,
temperature=…, max_tokens=…, model_kwargs=…, **kwargs)`.

`init_claude_model(key, gateway_version, model_name, base_url, max_tokens_val,
temperature, timeout_val, max_retries)` — `openai.OpenAI(base_url, api_key,
default_headers={"api-key", "ai-gateway-version"}, timeout, max_retries)`, an
inner `_to_ai_message` mapping LangChain `BaseMessage.type`
(`system`/`ai`→`assistant`/`human`→`user`) to `{"role", "content"}`, an inner
`_call` doing `client.chat.completions.create(model, messages, max_tokens,
temperature)` and returning `resp.choices[0].message.content`, wrapped in
`RunnableLambda(_call)`. Note the create call carries **no** timeout,
max_retries or model_kwargs.

`ai_sas_converter(key, sas_script, script_conversion_style, preferred_llm)` —
reads `ai_gateway_details`, computes `base_url = f"{ai_gateway_url}{model_name}"`,
then `match script_conversion_style.lower(): case "sql" | case "python" | "py"`
selecting prompts.

### 2.4 `sharepoint_utils.py`

Config: `sharepoint.{hostname, site_path, file_server_base_path,
list_id_sas_requests, list_id_sas_conversions, list_id_xref}`;
`vault.adb_secret_scope_name` for the secret scope.

Base-path normalisation:

```python
_DOC_LIB_PREFIX = "Shared Documents/"
_normalised = raw_base.lstrip("/")
sp_file_drive_base_path = (_normalised[len(_DOC_LIB_PREFIX):]
                           if _normalised.startswith(_DOC_LIB_PREFIX)
                           else _normalised)
```

Graph client construction (lazy; credentials fetched on first API call):

```python
SharePointGraphClient(
    config={"hostname": …, "site_path": …,
            "secret_scope": databricks_secret_scope,
            "tenant_id_key": "saact-hsv-tenantid",
            "client_id_key": "saact-hsv-appid",
            "client_secret_key": "saact-hsv-secret"},
    dbutils=dbutils)
```

Folder conventions:

```
{base}/{application_name}/scripts_original
{base}/{application_name}/scripts_converted
{base}/{application_name}/scripts_converted/validation
{base}/{application_name}/scripts_converted/{model}/{timestamp}   ← upload
```

Transport primitives used: `list_files(folder_path, extensions)`,
`download_file_as_text`, `read_json_text`, `list_items(list_id)`,
`get_list_item(list_id, item_id)`, `create_folder`, `upload_file(folder_path,
file_name, content)`, `update_list_item(list_id, item_id, {attr: value})`.

`_get_applicable_file_extensions('sas')` → `['sas', 'txt']`; anything else
raises.

List field mappings (SharePoint internal names, encoded spaces and trailing
spaces are literal and load-bearing):

```python
format_request_item_params(item) = {
    'application_name':     item.get('Application Name'),
    'input_language':       item.get('Source Language'),
    'output_language':      item.get('Destination Language'),
    'macro_file_name':      item.get('Macro File Name x003f '),
    'is_validation_required': item.get('Validation x0020 Documents x0020 '),
}
format_conversion_item_params(item) = {
    'app_request_id': int(item.get('Request_ID')),
    'script_name':    item.get('Script_Name'),
    'preferred_llm':  item.get('Model'),
    'status':         item.get('Status'),
}
_format_xref_item(item) = {
    'bsnId':            item.get('Title'),
    'application_name': item.get('Application'),
    'sourceTable':      item.get('OriginalValue'),
    'destinationTable': item.get('NewValue'),
}
```

`get_xref_mappings(application_name)` filters `list_items(list_id_xref)` on
`item['Application'] == application_name`, then flattens to
`{sourceTable: destinationTable}`, skipping rows where either side is empty.

`upload_converted_script(application_name, file_name, file_type, file_contents,
model, timestamp)` — lowercases `file_type`, applies `set_file_extension`,
converts to a notebook for `pyspark` / `sparksql`, creates
`{converted_base}/{model}/{timestamp}` and uploads, returning the path.

`upload_validation_file(file_path, file_name, file_contents)` — creates
`{file_path}/validation` and uploads into it.

`update_request_status(item_id, new_status)` — writes `Status` on
`list_id_sas_requests`.

### 2.5 `configurations_dev.json`

```json
{
  "vault": {
    "name": "hashicorp_vault_udap3.0",
    "type": "hashicorp_udap3.0",
    "adb_secret_scope_name": "udap_scripts_udappipelines_comp_APPSVC102630_dev",
    "hashicorp_app_name": "udap_scripts_udappipelines_comp_dev.appsvc102630",
    "hasicorp_env": "dev",
    "key_vault_url": "https://qa-itvault.humana.com",
    "namespace": "nsudap",
    "sp_authority_url": "https://login.microsoftonline.com/",
    "sp_scope": "https://management.azure.com//.default",
    "verify": true
  },
  "credentials": { "ai_cred_path": "ai_gateway", "ai_cred_key": "token" },
  "sharepoint": {
    "hostname": "inspirewellness.sharepoint.com",
    "site_path": "sites/ScriptConversionTool",
    "file_server_base_path": "Shared Documents/Script Migration and Conversion Kit/Applications",
    "list_id_sas_requests": "…", "list_id_sas_conversions": "…", "list_id_xref": "…"
  },
  "adls_config": {
    "storage_acct_name": "…", "container_name": "root", "workspace_host": "https://adb-….azuredatabricks.net",
    "adls_root_path": "…", "adls_applications_path": "…", "adls_path_main": "…",
    "archieve_path_main": "…", "external_table_path": "…"
  },
  "sas_config": { "sas_token": "sas_token", "inlining": "N" },
  "ai_gateway_details": {
    "ai_cert_path": "./SAS_Conversion_Agent/gateway.crt", "gateway_version": "v2",
    "model_provider": "anthropic", "model_name": "claude-sonnet-4-5",
    "ai_gateway_url": "https://gateway.aiplatform-npe.humana.com/chat/",
    "max_tokens": 64000, "temperature": 0.0, "timeout": 6000, "max_retries": 4
  },
  "ai_validator_config": {
    "is_validation_required": false, "ai_cert_path": "…", "gateway_version": "v2",
    "model_provider": "anthropic", "model_name": "claude-sonnet-4-5",
    "ai_gateway_url": "…", "output_folder": "/Workspace/Users/…/Validation",
    "max_tokens": 64000, "temperature": 0.0, "timeout": 12000, "max_retries": 4
  },
  "sftp_config": {
    "sftp_source_path": "/basedata/datapmrf/sas_ai_poc",
    "sftp_cred_path": "sftp/isbsas42/cdfhydrt",
    "xref_file_path": "/basedata/datapmrf/sas_ai_poc/XREF/xref.json"
  },
  "databricks_config": { "dbfs_root_path": "/dbfs/tmp/udap/sas_conversion" }
}
```

Note `archieve_path_main` is misspelled in the reference; our key is
`archive_path`, and the mapping is documented rather than propagated.

---

## 3. Target `config.json` schema

Retained conventions: environment variables take precedence over the file;
`get_value` treats `null` as absent; `get_typed_value` degrades a wrong-typed
entry to the hard default with a WARNING rather than raising.

```jsonc
{
  "vault": {
    "address": null,            // was key_vault_url
    "namespace": null,
    "app_name": null,           // NEW: was oidc_role; also the secret-path prefix
    "mount_point": null,        // default "secret"
    "kv_version": null,         // default 2
    "auth_path": null,          // default "jwt/azuread/inspirewellness"
    "azure_scopes": null,       // default ["https://management.azure.com//.default"]
    "timeout": null,
    "verify": null,
    "ai_gateway_path": null,    // default "{app_name}/ai_gateway"
    "ai_gateway_key": null      // default "token"
  },

  "azure": {
    "tenant_id": null, "client_id": null, "authority_host": null,
    "scopes": null, "flow": null, "timeout": null,
    "certificate_path": null, "certificate_thumbprint": null,
    "verify": null,             // NEW: bool or CA-bundle path, passed to MSAL
    "proxies": null             // NEW: optional MSAL proxies mapping
  },

  "databricks": {
    "host": null, "http_path": null, "warehouse_id": null, "cluster_id": null,
    "catalog": null, "schema": null, "timeout": null,
    "azure_tenant_id": null, "azure_client_id": null,
    "azure_workspace_resource_id": null,
    "secret_scope": null,
    "dbfs_root_path": null,     // NEW
    "workspace_output_folder": null  // NEW: /Workspace/... for future publishing
  },

  "llm_client": {
    "model": null, "model_provider": null,   // NEW
    "gateway_version": null,                 // NEW → "ai-gateway-version" header
    "provider_client": null,                 // NEW: "openai_compatible" | "native"
    "base_url": null, "url_headers": null, "cert_file": null,
    "timeout": null, "temperature": null, "max_retries": null,
    "model_kwargs": null, "max_input_tokens": null, "max_output_tokens": null,
    "prompt_caching": null, "requests_per_second": null, "max_bucket_size": null,
    "roles": {                               // NEW: sparse overlays on the above
      "validator":  { "timeout": null, "model": null },
      "complexity": { "timeout": null, "model": null }
    }
  },

  "sharepoint": {
    "site_hostname": null, "site_path": null, "site_id": null, "drive_id": null,
    "scopes": null, "timeout": null,
    "file_server_base_path": null,           // NEW, "Shared Documents/" stripped
    "secret_scope": null,                    // NEW: defaults to databricks.secret_scope
    "tenant_id_key": null,                   // NEW: default "saact-hsv-tenantid"
    "client_id_key": null,                   // NEW: default "saact-hsv-appid"
    "client_secret_key": null,               // NEW: default "saact-hsv-secret"
    "list_id_sas_requests": null,            // NEW
    "list_id_sas_conversions": null,         // NEW
    "list_id_xref": null,                    // NEW
    "list_id_sas_complexity": null           // NEW
  },

  "xref": {                                  // NEW SECTION
    "apply": null,                           // "pre" | "post" | "both"; default "pre"
    "dialect": null,                         // sqlglot dialect, default "databricks"
    "on_parse_failure": null                 // "warn" (default) | "error"
  },

  "adls": {                                  // NEW SECTION — schema only
    "storage_account": null, "container": null, "workspace_host": null,
    "root_path": null, "applications_path": null, "main_path": null,
    "archive_path": null, "external_table_path": null
  },

  "sftp": {                                  // NEW SECTION — schema only
    "source_path": null, "cred_path": null, "xref_file_path": null
  },

  "sas": {                                   // NEW SECTION
    "inlining": null                         // bool; reference uses "N"/"Y"
  }
}
```

Removed: the `powerapps` section in its entirety (§8.1).

Key renames requiring migration: `vault.oidc_role` → `vault.app_name`.

---

## 4. Phase 1 — config restructure

**Goal.** One schema expressing everything the reference deployment needs.

**Changes.**

1. `config.json` — the schema in §3, with `_comment` keys following the
   existing per-section convention.
2. `app_config/__init__.py` — extend the `llm_client_value` type map for
   `gateway_version` (str), `model_provider` (str), `provider_client` (str),
   `roles` (dict). Add a `role_value(role, key, default)` resolver applying
   `roles.<role>.<key>` → `llm_client.<key>` → default.
3. `.env.example` — document every new environment variable:
   `VAULT_APP_NAME`, `AZURE_VERIFY`, `LLM_GATEWAY_VERSION`,
   `SHAREPOINT_FILE_SERVER_BASE_PATH`, `SHAREPOINT_LIST_ID_*`,
   `SHAREPOINT_SECRET_SCOPE`, `XREF_APPLY`.

**Tests.** `tests/test_app_config.py` — new keys present, wrong-typed entries
degrade, role overlay precedence (`role` beats base beats default), unknown
role names fall through to the base section.

**Breaking.** Every renamed key. `vault.oidc_role` no longer read.

**Blocks.** Phases 3 and 5. Phase 2 depends on it only for `app_name`.

---

## 5. Phase 2 — auth blockers

**Goal.** Make a real run possible. B1, B2, C1, C3, C4.

**Changes.**

1. **B1 — ARM scope default.** In `app_config/vault.py::_azure_jwt`, the
   precedence becomes: `VaultConfig.azure_scopes` → the azure module's
   configured scopes → `_ARM_DEFAULT_SCOPE`. The `<client_id>/.default` form is
   **removed**, not demoted: it is not what any deployment wants by default, and
   an operator wanting it can set `vault.azure_scopes` explicitly.
   `_ARM_DEFAULT_SCOPE` keeps its comment explaining the doubled slash.
2. **B2 — `verify` to MSAL.** Add `verify: bool | str = True` and
   `proxies: dict[str, str] | None = None` to `AzureAuthConfig`, resolved in
   `from_env` from `AZURE_VERIFY` / `azure.verify` with the same
   `VAULT_SKIP_VERIFY`-style truthiness handling as `vault._resolve_verify`
   (factor that helper into a shared private function rather than duplicating
   it). Pass both into `ConfidentialClientApplication` and
   `PublicClientApplication`. `AzureAuthConfig.for_principal` must preserve
   them, since the Databricks-SPN path goes through it. When `vault.verify` is
   set and `azure.verify` is not, the vault value is inherited — the reference
   uses one `verify` for both legs.
3. **C1 — Vault token TTL.** Capture `lease_duration` from the `jwt_login` /
   `approle.login` response, store an expiry on `VaultClient` with a 60 s skew
   (mirroring `azure._EXPIRY_SKEW`), and re-authenticate from `_read` when it
   has passed. Token auth (`VAULT_TOKEN`) is exempt — its lifetime is the
   operator's business. Additionally retry once on a 403 from a read, since a
   revoked-early token is indistinguishable from an expired one.
4. **C3 — `app_name`.** `VaultConfig.app_name` replaces `oidc_role` and is used
   both as the `jwt_login` role and as the default secret-path prefix. Delete
   `AI_GATEWAY_PATH`; `get_ai_gateway_secret` resolves
   `vault.ai_gateway_path` → `f"{app_name}/ai_gateway"` and raises `VaultError`
   when neither is available. Remove the direct `os.environ` read at
   `vault.py:514` — resolution goes through `VaultConfig`.
5. **C4 — approle wrap.** Wrap `client.auth.approle.login` in the same
   `except Exception → VaultError` shape as the `azuread` branch.

**Tests.** `tests/test_vault.py`:

- **Invert `test_azuread_scopes_fall_back_to_client_id`** → assert the ARM
  scope is requested when nothing is configured. Rename accordingly.
- `vault.azure_scopes` still wins; the azure module's scopes still win over ARM.
- New: token expiry triggers re-authentication; a 403 read retries once then
  raises; `app_name` drives both role and path; approle failure raises
  `VaultError`.

`tests/test_azure.py`: `verify` and `proxies` reach the MSAL constructor;
`for_principal` preserves them; `vault.verify` is inherited when `azure.verify`
is unset.

**Breaking.** `vault.oidc_role` removed. `VAULT_OIDC_ROLE` → `VAULT_APP_NAME`.
Default JWT audience changes.

**Standalone.** Everything except item 4 lands without Phase 1.

---

## 6. Phase 3 — `llm_client` gateway conformance

**Goal.** B3 and C2.

**Changes.**

1. **B3 — `gateway_version`.** New `LLMClientConfig.gateway_version: str | None`
   defaulting from `app_config.llm_client_value("gateway_version")`. In
   `_build_model`, alongside the existing `api-key` mirroring:
   ```python
   if config.gateway_version:
       headers.setdefault("ai-gateway-version", config.gateway_version)
   ```
   `setdefault`, matching the `api-key` precedent — an explicit `url_headers`
   entry wins. Add it to the INFO log line by name only. The Vault AI-gateway
   secret may also carry it: extend `vault.py` with
   `_AI_GATEWAY_VERSION_KEYS = ("gateway_version", "ai_gateway_version",
   "version")` and an `ai_gateway_version(secret)` accessor, consumed by
   `from_ai_gateway` the way `ai_gateway_base_url` already is.
2. **C2 — provider.** Rename `_gateway_model_name` to `_split_model` returning
   `(provider, model)`; the model is lowercased, matching the reference. The
   provider populates `LLMClientConfig.model_provider` when that field is unset.
   The log line stops claiming the prefix is being "dropped".
3. **C2 — native client path.** `provider_client` selects the construction
   strategy:
   - `"openai_compatible"` (**default**) — today's `ChatOpenAI`.
   - `"native"` — an `openai.OpenAI` client wrapped so it satisfies the same
     minimal surface `LLMClient` needs, mirroring the reference's
     `_to_ai_message` / `_call` / `RunnableLambda`. Built through the existing
     `llm=` injection seam (`client.py:630`) so no new class hierarchy appears.
   Rationale for the default is D1 in §11.
   Documented limitation: on the native path, features attaching at
   construction time (rate limiter, `model_kwargs`, structured output) are
   unavailable; `LLMClient`'s retry and token budget still apply. This must be
   stated in the docstring and logged at WARNING when `provider_client` is
   `"native"` and any of those are configured.
4. **Role overlays.** `LLMClientConfig.for_role(name, **overrides)` resolving
   through `app_config.role_value`. Call sites migrate:
   `complexity/__main__.py:370` → `for_role("complexity", model=args.llm_model)`;
   `validation/__main__.py:252` → `for_role("validator", model=args.judge_model)`;
   `validation/judge.py` docstring example updated.
5. **`pipeline/engine.py`** — add `gateway_version` beside the existing
   per-run `url_headers` parameter (`:413`, `:464`, `:668`) and in the two
   docstring parameter lists (`:142`, `:355`).

**Tests.** `tests/test_llm_client.py` — the header is sent; an explicit
`url_headers` entry wins; `"anthropic:claude-sonnet-4-5"` yields provider
`anthropic` and model `claude-sonnet-4-5` (lowercased); the native path builds
and invokes; role overlays resolve with correct precedence.

**Breaking.** `_gateway_model_name` renamed (private, no external callers).

**Depends on.** Phase 1.

---

## 7. Phase 4 — SharePoint transport

**Goal.** C5, C6, and the primitive set the domain layers need.

**Changes.**

1. **C5 — parameterised secret keys.** In `app_config/databricks.py`, keep
   `SECRET_KEY_CLIENT_ID` / `_CLIENT_SECRET` / `_TENANT_ID` as the *default*
   set and add a `SecretKeySet` dataclass (`tenant_id_key`, `client_id_key`,
   `client_secret_key`). `DatabricksConfig.service_principal(keys=None)` and
   `azure.get_databricks_client(secret_scope=None, keys=None)` accept one. The
   SharePoint config supplies the `saact-hsv-*` set. Because
   `get_client_for_principal` caches on `(tenant_id, client_id)`, two distinct
   principals coexist without interfering.
2. **C6 — base-path normalisation.** `SharePointConfig.file_server_base_path`,
   normalised on resolution: strip leading `/`, then strip a leading
   `"Shared Documents/"` (case-insensitive) if present. A
   `drive_path(*parts)` helper joins the base with relative segments so no
   caller concatenates by hand.
3. **Primitives.** The existing surface differs from what the domain layers
   need, and one primitive does not exist. Verified delta:

   | Needed | Today | Work |
   |---|---|---|
   | `list_files(folder, extensions)` | `list_directory(path)` (`:486`) | add extension filtering; the reference filters `['sas','txt']` |
   | `download_file_as_text(path)` | `read_file(path) -> bytes` (`:519`) | add a text wrapper with explicit encoding |
   | `read_json_text(path)` | — | add, over the text wrapper |
   | `upload_file(folder, name, content)` | `write_file(path, content)` (`:540`) | add a folder+name form; the domain layers compose paths from a base and never concatenate by hand |
   | `create_folder(path)`, idempotent | `create_directory(...)` (`:568`) | confirm idempotence — the reference calls it unconditionally before every upload |
   | `list_items(list_id)` | `read_list_items(...)` (`:616`) | rename or alias |
   | `get_list_item(list_id, item_id)` | — | add |
   | `update_list_item(list_id, item_id, fields)` | **— nothing** | **add; this is the real gap** |

   The list-item write is the one true absence. `conversion.update_request_status`
   has no transport beneath it today. Complexity does not need it (§8.3), so this
   is a Phase 5.1 dependency only.
4. **List-id accessors.** `SharePointConfig.list_id(kind)` over the four
   configured ids, raising `SharePointError` naming the missing config key
   rather than passing `None` to Graph.

**Tests.** `tests/test_sharepoint.py` — base-path stripping (with prefix,
without, leading slash, mixed case); extension filtering; `create_folder`
idempotence; the folder+name upload form composes the same path `write_file`
would take; `get_list_item` and `update_list_item` round-trip against the fake
transport; the `saact-hsv-*` key set reaches the secret read; a missing list id
raises with the config key named.

**Blocks.** Phase 5.

---

## 8. Phase 5 — domain parts

Three domain modules over one transport, each owning one concern. The Graph
transport stays free of domain knowledge; the domain modules stay free of Graph
detail.

### 8.1 `conversion/`

Replaces `app_config/powerapps.py`, which models the same concept — a request
row with a selected model — against a list that does not exist. `powerapps.py`
and `tests/test_powerapps.py` are **deleted**; the `powerapps` config section is
removed. `is_accessible_model` (`app_config/__init__.py`) is retained and reused
for `Preferred_LLM` validation.

**Migration, not just deletion (C9).** `demo_run.py` imports `PowerAppsConfig`
and `PowerAppsError` lazily inside one function (`:881`) and documents
`POWERAPPS_LIST_NAME` in its module docstring (`:96`). That function is
repointed at `conversion.requests`, and the docstring updated to the
`sharepoint.list_id_sas_requests` key. The blast radius is one function; the
lazy import means nothing else in `demo_run.py` is affected. `demo_run.py` is
currently modified in the working tree — rebase before touching it.

Surface:

- `paths.py` — `original_scripts(app)`, `converted_scripts(app)`,
  `validation(app)`, `upload_target(app, model, timestamp)`, all built on
  `SharePointConfig.drive_path`.
- `requests.py` — `format_request_item_params` / `format_conversion_item_params`
  with the §2.4 field mappings in a single module-level table;
  `pending_requests()`; `update_request_status(item_id, status)`.
- `sources.py` — `source_files(app, file_type="sas")` with the `sas`/`txt`
  extension set; `load(path)`.
- `upload.py` — `upload_converted_script(...)` including the notebook
  conversion for `pyspark` / `sparksql` (delegating to `pipeline/notebook.py`,
  not reimplementing it) and `upload_validation_file(...)`.

`preferred_llm` from the `Model` column is parsed by Phase 3's `_split_model`
and fed to `LLMClientConfig` — the same parser, not a second one.

### 8.2 `xref/`

`chunker/batcher.py` is **not modified**.

- `sourcing.py` — `mappings(application_name)`: `list_items(list_id_xref)`,
  filter on `Application`, project via the §2.4 `_format_xref_item` mapping,
  flatten to `{sourceTable: destinationTable}` skipping empty sides.
  Rows are `schema.table → catalog.schema.table`, which `_split_databricks_mapping`
  already classifies as exact keys and already validates as three-part targets.
- `apply.py` — `apply_pre(batch_result, mapping)` delegating to
  `replace_dataset_names`; `apply_post(code, language, mapping)`; and
  `apply(mode, …)` dispatching on `xref.apply`.
- `rewrite.py` — the post-conversion rewriter:
  - **sparksql** — `sqlglot.parse` with dialect `xref.dialect` (default
    `databricks`), walk `exp.Table`, rewrite nodes whose `db.name` matches a
    mapping key, regenerate.
  - **pyspark** — Python `ast` over string literals in `spark.table(...)`,
    `spark.read.table(...)`, `.saveAsTable(...)`, and `spark.sql("…")`; the
    last recurses into the sqlglot path. Rewriting is done by source-span
    substitution rather than `ast.unparse`, so formatting and comments survive.
  - **Unparseable input** — leave the output untouched and log at WARNING
    (`xref.on_parse_failure = "warn"`), or raise (`"error"`). A rewriter that
    corrupts generated code is worse than one that no-ops. This is a hard rule.
  - `sqlglot` is imported lazily; the `sql` extra becomes a documented install
    target for the post mode.
- **`"both"` mode** is the verification path: apply pre, apply post, and report
  names that only one of them reached. Divergence means a dataset name escaped
  the SAS-side metadata extraction, which is precisely the evidence needed to
  choose a permanent mode.

**Physical paths — designed, not built.** `Title` / `bsnId` is confirmed free
to carry a type discriminator, so the design is settled rather than speculative:

- **Row shape.** `Title` optionally carries a type marker. Absent, empty or
  unrecognised → **table mapping**, so every existing row keeps working
  untouched and no backfill is required. A recognised path marker (`path`,
  case-insensitive) routes the row to `by_path`.
- **Container.** `mappings()` returns three slots from the outset — `exact`,
  `by_libref`, `by_path` — with the third populated only when marked rows
  appear. `apply_pre` passes the first two to `replace_dataset_names`
  unchanged, so `chunker/batcher.py` stays untouched in both the present and
  the future design.
- **Enabling it later** then requires only: lifting the quoted-literal guard in
  `_map_ds` (`chunker/batcher.py:437`) behind a `by_path` argument, and
  extending the rewriter to `LIBNAME` / `INFILE` / `%include` targets. No
  config, list-schema or transport change.
- **Validation now.** `sourcing.py` reads and classifies `Title` from day one
  and warns on an unrecognised marker, so a mistyped row is visible before the
  feature exists rather than silently becoming a table mapping.

### 8.3 `complexity/sharepoint.py`

The complexity list is a **request** list with columns `ID`, `Application`,
`Output_Language`, `Preferred_LLM`. It has no result columns, so complexity
output is delivered **only** as uploaded artefacts — there is no per-file
row write-back.

```python
@dataclass
class ComplexityRequest:
    item_id: str        # ID
    application: str    # Application
    output_language: str  # Output_Language  → rules profile / --target
    preferred_llm: str | None  # Preferred_LLM → provider:model, implies --llm-eval
```

- `requests()` — read and project rows from `list_id_sas_complexity`.
- `request(item_id)` — one row.
- `source_files(app)` / `load(path)` — shared with `conversion.sources`.
- `upload_reports(app, label, timestamp, paths)` — create
  `{base}/{app}/complexity/{label}/{timestamp}` and upload each staged file.

`label` is the LLM model id when `--llm-eval` ran, else the resolved rules
target profile — so the folder records what produced the estimate. This mirrors
conversion's `{model}/{timestamp}` while remaining meaningful for an offline run.

**No `Status` field, and none is being added.** This is settled, and it
simplifies the design rather than constraining it:

- **No status write-back.** Complexity never writes to the list. It is a
  read-only request source, so `update_list_item` is not a dependency here
  (unlike conversion — §7 item 3).
- **No pending concept.** Every row is a valid target on every invocation;
  `--sharepoint` processes them all, `--app` and `--item-id` narrow the set.
  The run is explicitly triggered, so "which rows are outstanding" was never a
  question the flow needed to answer — the earlier design imported it from the
  conversion analogy and it does not belong.
- **Idempotence is structural.** Every run lands in a fresh
  `{label}/{timestamp}` folder, so re-running is non-destructive by
  construction and needs no marker.
- **Reporting replaces the column.** A `run-summary.md` is uploaded beside the
  reports: the request row's four field values, resolved target and model,
  files scored, files that failed to chunk, wall time, and the exit status. An
  operator reading SharePoint sees the outcome without a Status field, and
  failures are visible where the artefacts are rather than only in logs.
- **Exit status** remains the machine-readable signal, following the existing
  CLI convention (`complexity/__main__.py:288`).

`pipeline/run_ledger.py` was considered as a processed-row marker and
rejected: it is `MemoryHub`-backed and keyed on pipeline thread ids for
per-item resume, which is a different problem. Should durable run history ever
be wanted here, the repo's existing KV pattern (`memory.store`, Delta-backed on
Databricks) is the place for it — not the SharePoint list.

### 8.4 `complexity/__main__.py` — SharePoint execution flow

**Seam.** Extract from `main()`:

- `_load_sources(args) -> tuple[list[SasChunkResult], str]` — returns the chunk
  results and a label for logging. Local mode: `rglob` + `chunk_file`.
  SharePoint mode: `source_files` + `load` + `chunk_text(content,
  source_id=<drive-relative path>)`.
- `_deliver(args, report, texts, …) -> int` — everything currently at
  `:251`–`:284`, plus the upload step. `_run_evaluation` takes an explicit
  `out_dir` parameter instead of reaching for `args.out_dir`, so staging works
  for `prompts/` and `llm-evaluation.md` too.

The middle — `MultiFileBatcher`, `ComplexityAnalyzer.analyze_items`,
`chunk_texts` — is untouched and shared by both modes.

**Arguments.** `sas_dir` becomes `nargs="?"`, default `None`. New:

| Flag | Meaning |
|---|---|
| `--sharepoint` | Enable SharePoint mode. With no narrowing flag, every row in the complexity list is processed. |
| `--app NAME` | Score one application. Rows are filtered on `Application`. |
| `--item-id ID` | Score exactly one request row, by `ID`. |
| `--sharepoint-out PATH` | Override the upload folder. |
| `--no-upload` | Analyse from SharePoint, deliver locally. The dry run. |

Validation, before any analysis (alongside the existing `--pdf` check at
`:208`): exactly one of `sas_dir` / `--sharepoint` must be given; `--app`,
`--item-id`, `--sharepoint-out` and `--no-upload` require `--sharepoint`. The
`--pdf` guard relaxes — SharePoint mode gives it a destination without
`--out-dir`.

**Row-driven defaults.** `Output_Language` supplies `--target`;
`Preferred_LLM` supplies `--llm-model` and implies `--llm-eval`. Explicit flags
override the row. `--target` resolution order becomes: flag → row →
`config.json complexity.target` → built-in default.

**Delivery.** Stage into a `TemporaryDirectory` using the existing, unmodified
`write_reports` and `render_pdf`, then upload. This is deliberate:
`render_pdf` requires the Markdown and `dependency-graph.png` to be co-located
for image resolution (`complexity/pdf.py:193`), and `WrittenReports.paths`
(`complexity/report.py:82`) already enumerates the upload set. **The report
layer is not made storage-agnostic.** With `--out-dir` *and* `--sharepoint`,
the staging directory is `--out-dir` and the files are kept as well as uploaded.

**Usage.**

```bash
python -m complexity --sharepoint                       # every request row
python -m complexity --sharepoint --app "MyApp" --pdf
python -m complexity --sharepoint --item-id 42
python -m complexity --sharepoint --app "MyApp" --no-upload --out-dir reports/
python -m complexity path/to/sas --out-dir reports/     # unchanged
```

**Multi-row runs.** With more than one row selected, each is scored as its own
corpus and uploaded to its own folder — applications are not merged, since
cross-file resolution across unrelated applications would corrupt every
verdict. One row failing does not abort the rest; the exit status is non-zero
if any row failed, and each row's outcome is in its own `run-summary.md`.

**Module docstring** is updated with the new invocations and the SharePoint
flow, matching the existing style.

**Tests.** New `tests/test_complexity_sharepoint.py` against a fake transport:
row projection; `source_id` becomes the drive-relative path and flows into
report file naming via `source_stems`; the staged tree is uploaded in full
including the PDF and graph image; `--no-upload` uploads nothing; label
selection with and without `--llm-eval`; argument validation rejects both
sources and neither. Existing `tests/test_complexity.py` must pass unchanged —
the local path is not to regress.

---

## 9. Phase 6 — `complexity` internal split

No reference exists for the complexity analysis itself; this is quality-only
and scoped smallest. `rules.py` (1164 lines), `models.py` (951) and
`analyzer.py` (966) are the repo's three largest files, having accreted the
graph, PDF and LLM-eval features. The LLM seam is already clean — the CLI goes
through `LLMClient` — so this is a split by concern with no rewiring. It lands
with Phase 5, which touches the package anyway.

---

## 10. Pruning and deprecation removal

The repo carries deprecation debt from the earlier pipeline-decoupling refactor
that has been waiting for a breaking-change window. This is that window.

### 10.1 Shim modules — delete outright

Four re-export shims left behind when orchestration moved out of `chunker/`,
each raising a `DeprecationWarning` on import and each documenting itself as
"will be removed in a future release":

| Shim | Re-exports |
|---|---|
| `chunker/pipeline.py` | `pipeline.engine.SasLLMPipeline` plus six `pipeline.prompting` helpers |
| `chunker/response_models.py` | `pipeline.response_models` |
| `chunker/pipeline_constants.py` | `pipeline.constants` |
| `chunker/notebook.py` | `pipeline.notebook` |

Plus the lazy `__getattr__` compatibility table in `chunker/__init__.py:53`–`:79`,
which resolves old names to their new modules with a warning.

Procedure, per module: grep the repo — including `tests/`, `demo_run.py` and
`docs/` — for the old import path; repoint every hit at the real module; delete
the shim; run the suite. The `__getattr__` table goes **last**, since the shims'
own imports may route through it.

`pipeline/notebook.py` is the real module and a Phase 5.1 dependency —
conversion's notebook rendering delegates to it. Only the `chunker/` shim goes.

### 10.2 Superseded by this refactor

Listed so nothing is left behind when each phase lands.

| Item | Location | Phase | Replaced by |
|---|---|---|---|
| `AI_GATEWAY_PATH = "appsvc/ai_gateway"` | `vault.py:100` | 2 | `{app_name}/ai_gateway` |
| Direct `os.environ` read for the gateway path | `vault.py:514` | 2 | resolution via `VaultConfig` |
| `<client_id>/.default` scope form | `vault.py:298` | 2 | ARM default (D4) |
| `vault.oidc_role`, `VAULT_OIDC_ROLE` | config, `vault.py` | 1, 2 | `vault.app_name`, `VAULT_APP_NAME` |
| `_gateway_model_name` and its "dropping prefix" log line | `client.py:590` | 3 | `_split_model` |
| `app_config/powerapps.py` | whole module | 5.1 | `conversion.requests` |
| `tests/test_powerapps.py` | whole module | 5.1 | merged into the conversion tests |
| `powerapps` config section | `config.json` | 1 | `sharepoint.list_id_sas_requests` |
| `POWERAPPS_*` environment variables | `.env.example` | 1 | `SHAREPOINT_*` |

### 10.3 Legacy parameter spellings — collapse

`pipeline/engine.py` accepts both the grouped configs (`llm_config=`,
`memory_setup=`) and the individual keyword arguments they replaced, rejecting
the mixture outright because one would otherwise silently lose (`:455`,
`pipeline/setup.py:8`). The grouped form is canonical and the individual
spellings are explicitly "the legacy spelling". Collapsing to grouped-only
removes the rejection branch and a very wide constructor signature.

Two output fields are documented as vestigial: `is_batch` is always `True` and
`kind` always `None`, "both kept for output-shape compatibility and deprecated"
(`:126`). Check every in-repo consumer, then drop both from the output dicts.

Both changes break every caller of `SasLLMPipeline`, `demo_run.py` included, so
they belong in this window — but they are **independent of Phases 1–6** and can
land separately if the schedule is tight.

### 10.4 Explicitly NOT pruned

Named so a later pass does not mistake them for debt:

- **`memory/store.py` legacy row and key handling** (`:680`–`:1064`). This is
  *persisted-data* compatibility, not dead code: legacy `{seq:08d}` counter keys
  and pre-lossless row shapes exist in real Delta tables. Removing it makes
  existing history unreadable. The "legacy" tests in `tests/test_store.py` guard
  exactly this and stay.
- **The LangChain `PendingDeprecationWarning` filter**
  (`pipeline/engine.py:31`–`:40`). A third-party warning, not our deprecation.
- **The httpx `verify=<path>` comment** (`llm_client/client.py:726`). It explains
  why an explicit `SSLContext` is used; the code is current.
- **`build/`, `.prompt_builder_cache/`, `__pycache__/`** — untracked local
  artefacts (confirmed: `git ls-files build/` is empty). Local hygiene, not a
  repository change.

### 10.5 Rule for this refactor

No new deprecation shims. Anything a phase supersedes is deleted **in that
phase**, with its call sites migrated in the same change. A `DeprecationWarning`
bridge is not an acceptable substitute for the migration — leaving nothing
behind is the point of the window.

---

## 11. Decisions

**D1 — `ChatOpenAI` stays the default over the reference's native client.**
The reference routes `anthropic` through raw `openai.OpenAI`. The likely reasons
— the Responses API being inferred, and the SDK's own retry layer — are already
neutralised (`use_responses_api=False`, `max_retries=0`,
`client.py:702`–`710`). Going native would forfeit retry-with-`Retry-After`,
the token budget, prompt-cache breakpoints, structured output and usage
accounting. The native path is therefore built and available via
`provider_client`, but is not the default. **If the gateway rejects the
LangChain payload in practice, this decision reverses to a one-line config
change.**

**D2 — Stage-then-upload rather than a storage abstraction.** §8.4.

**D3 — XREF sourcing is separate from XREF substitution.** `chunker` stays
network-free; `xref/` owns SharePoint. `batcher.py` is untouched.

**D4 — The `<client_id>/.default` scope form is removed, not demoted.** §5.

**D5 — `complexity/sharepoint.py` lives inside `complexity/`,** since it
depends on `CorpusComplexityReport` and nothing depends on it. `conversion/`
and `xref/` are top-level because the pipeline consumes both.

**D6 — Physical-path remapping is designed for but not built.** §8.2. The
`Title` discriminator is confirmed available, so the row shape and container are
settled now; only the guard lift and the rewriter extension remain.

**D7 — Complexity does not write to SharePoint.** It is a read-only consumer of
its request list. There is no `Status` field, none is being added, and the flow
does not need one — §8.3. This keeps `update_list_item` out of complexity's
dependencies entirely, and means Phase 5.3 can land before the transport gains
list-item writes.

**D8 — One row, one corpus.** Multi-row runs never merge applications, because
cross-file dataset/macro resolution across unrelated applications would corrupt
every verdict. §8.4.

---

## 12. Open questions

| # | Question | Status |
|---|---|---|
| Q1 | Completion/pending tracking for complexity runs. | **Closed.** No `Status` field exists and none is being added. Resolved by design in §8.3: no write-back, no pending concept, timestamped folders for idempotence, `run-summary.md` for reporting. No longer blocks anything. |
| Q2 | Is `Title` / `bsnId` free to carry a path-type discriminator? | **Closed — yes.** Design settled in §8.2: unmarked rows are table mappings, so existing rows need no backfill; classification and warning ship from day one, `by_path` activates later. |
| Q3 | Which XREF mode becomes the default once verified? | Open. `"both"` exists to answer it. Blocks nothing; default is `"pre"` until evidence says otherwise. |
| Q4 | Does the gateway accept the LangChain payload for Anthropic models? | Open. Blocks nothing; D1 is reversible by config. |

Neither remaining question blocks any phase.

---

## 13. Out of scope

New integration surface, not refactoring. Config schema lands in Phase 1 so
adopting these later is not a second breaking change.

- **ADLS** — `adls` section only; no client.
- **SFTP** — `sftp` section only; no client. Note the reference's
  `sftp_config.xref_file_path` implies a *file-based* XREF source alongside the
  SharePoint list; `xref/sourcing.py` should keep its public function signature
  source-agnostic so a second backend slots in.
- **Databricks workspace publishing** — `workspace_directory_exists`,
  `create_workspace_directory`, `write_content_to_notebook` (§2.2). We generate
  notebooks but only to local disk (`pipeline/notebook.py:581`). When built, it
  should use the SDK's `w.workspace.mkdirs()` / `import_()` rather than
  hand-rolled `requests`, so token refresh and the `azure_resource_id` headers
  keep working. Until then `ai_validator_config.output_folder` has no
  counterpart and validation output cannot be published.
- **`sas.inlining`** — config key only; no `%include` inlining behaviour change.

---

## 14. Ordering

```
Phase 1 ─┬─ Phase 2   (item 4 only; items 1,2,3,5 are standalone)
         ├─ Phase 3
         └─ Phase 4 ─┬─ Phase 5.1 conversion   (needs list-item write)
                     ├─ Phase 5.2 xref
                     └─ Phase 5.3 complexity ── Phase 6
```

Phase 2 is the only phase that changes whether a real run succeeds, and all of
it but the `app_name` collapse lands without Phase 1. Phase 4 unblocks all
three domain parts.

Because complexity never writes to the list (D7), **Phase 5.3 depends only on
the read half of Phase 4** — extension filtering, the text/JSON read helpers,
`create_folder` and the folder+name upload. The list-item write, the one true
absence in the transport (§7 item 3), is needed by Phase 5.1 alone. The
shortest path to something demonstrable end to end is therefore
Phase 4 (read half) → Phase 5.3.
