# SAS Parser

Translate SAS programs into a Databricks target — PySpark or Spark SQL
— with the corpus chunked semantically, batched by its real dependency graph,
and converted through an LLM behind an AI Gateway.

A run is driven from SharePoint by default: request rows say which applications
to convert and with which model, the scripts come out of the document library,
and the notebooks go back into it. A local directory of `.sas` files is the
offline fallback.

## Requirements and installation

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)

Install the core dependencies from the repository root:

```bash
uv sync
```

The core install supports local conversion with an OpenAI-compatible API key.
Add only the integrations your run needs:

| Extra | For |
|---|---|
| `sharepoint` | Microsoft Graph — the request lists and the document library |
| `vault` | HashiCorp Vault — the AI Gateway credential |
| `azure` | Entra ID (MSAL) — the service principal that reaches Vault |
| `databricks` | Workspace and SQL warehouse access |
| `spark` | Delta-backed conversation memory and validation tracking |
| `graph` | The complexity report's dependency-graph PNG |
| `dev` | pytest, ruff, and the test-only dependencies |

For example, a SharePoint run using the default AI Gateway credential chain
needs:

```bash
uv sync --extra sharepoint --extra vault --extra azure
```

`sqlglot`, used for Spark SQL checks and XREF post-rewriting, is included in
the core install; there is no separate `sql` extra.

## Usage

The command has two deliberately separate modes. Supplying a source directory
runs locally; omitting it reads the SharePoint request list. The program
rejects flags that belong to the other mode instead of silently ignoring them.

| Mode | Command shape | Delivers output to |
|---|---|---|
| Local | `sas-parser <sas-directory> [options]` | Standard output and, when requested, local notebooks/reports |
| SharePoint (default) | `sas-parser [options]` | The configured SharePoint document library and request list |

Run `uv run sas-parser --help` for the complete, installed command reference.
`python main.py ...` is equivalent when running from a checkout.

### Local conversion

For a direct local run, set `OPENAI_API_KEY` in your shell or in an untracked
`.env` file. `.env` is loaded automatically from the working directory (or a
parent); an already-exported environment variable wins. Use
[`.env.example`](.env.example) as the annotated list of supported settings,
but never copy its placeholder credentials into a real environment.

From the repository root, convert every `.sas` file under a directory and
write notebooks plus validation reports:

```bash
uv run sas-parser path/to/sas --out-dir out/ --md reports/validation.md --pdf reports/validation.pdf --no-gateway-auth
```

This command recursively discovers `*.sas` files, converts the application as
one corpus so cross-file dependencies can be resolved, prints each generated
translation, and writes one notebook per source file. A batch spanning files
also produces `_cross_file.ipynb`. `--out-dir` is optional: without it,
translations are printed but no notebooks are written.

Useful local options:

```bash
# Use a different filename pattern and emit PySpark instead of the configured target.
uv run sas-parser path/to/sas --pattern "*.SAS" --output-language PySpark --out-dir out/

# Choose the model exposed by the gateway and retain run memory in Delta.
uv run sas-parser path/to/sas --model <gateway-model-id> --delta-table default.sas_parser_memory

# Score output but do not retry failed validation checks.
uv run sas-parser path/to/sas --validation-retries 0

# Disable inline validation. Markdown/PDF validation reports are then not produced.
uv run sas-parser path/to/sas --no-validate
```

The supported targets are `PySpark` and `Spark SQL` (case and spacing are
normalized). `--md` and `--pdf` write reports only when inline validation is
enabled, which is the default. Passing `--delta-table` requires the `spark`
extra and starts a Spark session; otherwise run memory remains in process.

### SharePoint conversion

Configure the SharePoint site, document library, list identifiers, and
credential chain in `config.json` and/or environment variables before running
this mode. See [Configuration](#configuration) below. By default, only
pending request rows are selected.

```bash
# Convert every pending request row.
uv run sas-parser

# Convert one request row, or all pending rows for one application.
uv run sas-parser --request-id 42
uv run sas-parser --app "MyApp"

# Re-run completed rows as well, without writing files or changing row status.
uv run sas-parser --all-rows --no-upload

# Convert without applying the application's XREF mappings.
uv run sas-parser --app "MyApp" --no-xref
```

`--request-id` and `--app` are mutually exclusive. `--no-upload` is a real
conversion dry run: it still reads source files and calls the model, but does
not upload notebooks or update request status.

#### What a SharePoint run does

1. Reads the pending rows of the requests list.
2. Pulls each application's scripts from `{base}/{app}/scripts_original`.
3. Applies the XREF cross-reference — dataset names, and physical paths for
   rows marked as such.
4. Converts the whole application as **one corpus on one thread**, so
   cross-file dataset and macro dependencies resolve.
5. Uploads one notebook per source file to
   `{base}/{app}/scripts_converted/{model}/{timestamp}` and the effective
   prompt for every model call under that run's `prompts/` subdirectory, plus
   validation artefacts under `{base}/{app}/scripts_converted/validation` when
   the row asks for them.
6. Writes the row's `Status`.

One row failing does not stop the others; the exit status is non-zero if any
did.

#### Checking the deployment before converting anything

`--check` is the preflight. It resolves the configuration, mints a Graph token,
and reads the library and lists — writing nothing, converting nothing, and
paying no LLM. Start here whenever a SharePoint run misbehaves.

```bash
uv run sas-parser --check
```

It reports each stage in dependency order, and for every setting it says
**which** source the value came from — the environment variable or the
`config.json` key. That is usually the whole diagnosis: a value that is correct
in `config.json` and stale in the environment looks identical to a correct one
until you know which won. The base path is reported both as written and as
normalised, so the `Shared Documents/` strip is visible rather than surprising.

The token stage decodes the granted application permissions out of the token's
`roles` claim. This is what separates "the app registration was never granted
`Sites.ReadWrite.All`" from "it was granted but admin consent was never
clicked" — the two causes of a Graph 403, which are otherwise indistinguishable
from the outside. A missing role is reported as a warning and the later stages
still run, so you can see exactly which calls it blocks.

The same preflight runs standalone, and takes `--offline` to check the
configuration with no network at all:

```bash
python -m app_config.sharepoint_check --verbose
python -m app_config.sharepoint_check --offline
python -m app_config.sharepoint_check --json
```

#### Capturing a run

```bash
# Everything the console shows, also written to a file (appended).
uv run sas-parser --app "MyApp" --log-file logs/myapp.log

# Plus every individual Graph request, its status, and the SDK's own retries.
uv run sas-parser --app "MyApp" --log-file logs/myapp.log --trace-http
```

`--debug` raises the first-party loggers to DEBUG but deliberately leaves the
HTTP transport libraries at INFO, so the pipeline's own lines stay readable;
`--trace-http` is the opt-in for the wire. Bearer tokens and secret-shaped
values are masked before anything reaches a handler, but a log file is still
sensitive — the redaction covers the shapes these libraries emit, not every
shape possible. All three flags work on `python -m complexity`,
`python -m validation` and `python -m app_config.sharepoint_check` too.

If a run dies, the traceback is captured in the file as well as printed —
including one raised on the Graph client's worker thread — so a log that ends
mid-run says why. Grep `unhandled exception` to find it.

## Other entry points

```bash
python -m complexity path/to/sas --out-dir reports/   # offline complexity + sizing
python -m complexity --sharepoint --app "MyApp"       # from the complexity list
python -m validation --help                           # the offline validation suite
python -m app_config.sharepoint_check                 # read-only SharePoint preflight
python -m app_config.databricks_check                 # read-only cluster preflight
python -m app_config.auth_check                       # credential-chain dry run (offline)
python -m app_config.auth_check --live                # ... with the reads and mints performed
```

On a Databricks cluster, run the conversion from a **cell**, never from
`!python main.py ...` — a child process inherits the runtime marker but not
the notebook's credential, and fails the secret-scope read with a message
about something else entirely:

```python
import main
main.run_in_notebook("--reference-dir '/Workspace/.../reference_docs' --request-id 80")
```

[`databricks/run_conversion.py`](databricks/run_conversion.py) is that plus the
two preflights, as a notebook you can import into the workspace.

## Configuration

Start with [`config.json`](config.json) for non-secret defaults and
[`.env.example`](.env.example) for the environment-variable names and
credential setup. Keep secrets in your shell, an untracked `.env`, Databricks
secret scopes, or Vault — never in `config.json`.

Precedence is the same everywhere: **explicit argument > environment variable >
`config.json` > code default**. A JSON `null` means "unset", so a template
config listing every key changes nothing until edited.

For a local direct-key run, the minimum configuration is `OPENAI_API_KEY`
(plus a gateway base URL when your deployment needs one). For a SharePoint
run, configure the site (`SHAREPOINT_SITE_ID` or hostname/path), document
library, request/conversion list IDs, and an Entra ID credential source. The
example file documents the optional Vault, Databricks, XREF, and Docker
settings as well.

### Credentials

The LLM key resolves in this order:

1. `--vault-secret PATH` — an explicit AppRole read of a named Vault secret.
2. **The AI Gateway chain** (the default whenever Vault is configured): an
   Entra ID service principal from the Databricks secret scope mints a JWT,
   that JWT logs in to Vault, and the secret it unlocks is the gateway token.
   The `ai-gateway-version` header and the gateway's rate-limit pacing come
   from here too.
3. `OPENAI_API_KEY` — the local-development path, when no Vault settings are
   present or `--no-gateway-auth` is passed. The gateway speaks the OpenAI
   protocol for every model it fronts, so that is the variable whatever the
   model.

The chain is walked once per invocation and reused for every row.

## Development

```bash
uv run pytest -q                    # the suite
uvx ruff@0.15.20 check .            # lint, pinned to CI's version
uvx pyright@1.1.406 --outputjson > pyright.json
uv run python scripts/pyright_ratchet.py --input pyright.json
```

The type gate is a **ratchet**: a file already in
`scripts/pyright_baseline.json` may not report more errors than recorded.
Every shipped file currently sits at zero.

There is also a local Docker stack (Vault + Spark/Delta) for exercising the
credential chain and the Delta memory backend without a real deployment — see
[`docker/README.md`](docker/README.md).

## Where things live

[`Architecture.md`](Architecture.md) is the full map — the package layout, the
chunking and batching models, and the load-bearing invariants that look like
implementation details but are contracts. Most packages also carry their own
README.

| Package | Owns |
|---|---|
| `chunker/` | Lexing, semantic chunking, and dependency-graph batching. Network-free. |
| `pipeline/` | The LangGraph engine, prompting, and notebook rendering. |
| `llm_client/` | Chat-model construction, retry, token budgeting, usage. |
| `memory/` | Conversation, policy, and thread memory over a KV store. |
| `prompt_builder/` | Reference-document retrieval and instruction injection. |
| `app_config/` | `config.json` loading, plus Vault / Entra ID / Databricks / SharePoint / Spark. |
| `conversion/` | What the SharePoint rows and folders mean, and the run orchestration. |
| `xref/` | The dataset/path cross-reference: sourcing, and applying it. |
| `complexity/` | Offline complexity scoring, sizing, and reports. |
| `validation/` | The metric suite, inline and offline. |
| `target_language/` | The one place a target's spelling is resolved. |
