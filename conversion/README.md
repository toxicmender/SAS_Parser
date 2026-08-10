# conversion

The conversion-run domain layer over `app_config.sharepoint`. It translates
SharePoint request rows and document-library content into a complete pipeline
run; it does not call Microsoft Graph directly.

## Workflow

1. `requests.py` reads conversion requests and identifies outstanding rows.
2. `sources.py` finds and loads an application's SAS inputs.
3. `run.py` creates the pipeline, translates the application as one corpus,
   and records the outcome on the request row.
4. `upload.py` writes converted notebooks/scripts and validation artifacts.

`run_request()` accepts an optional SharePoint client and pipeline factory, so
the flow can be exercised with fakes without a network connection or LLM.
When no client is supplied, it uses the shared client from `app_config`.

## Package layout

| File | Responsibility |
|---|---|
| `paths.py` | Source, converted-output, and validation folder conventions. |
| `requests.py` | Request/list row models, projections, and status updates. |
| `sources.py` | Source discovery and reading. |
| `upload.py` | Deliverable uploads and notebook rendering. |
| `run.py` | Request selection, orchestration, status handling, and run outcomes. |

The public package intentionally does not export a `requests` function: use
`conversion.requests.requests` when the full row listing is needed, avoiding
an ambiguous package attribute that would shadow the module.

Logger names follow `conversion.*`.
