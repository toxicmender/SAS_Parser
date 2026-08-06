# SAS Parser

Translate SAS programs into a Spark target — PySpark, Spark SQL, or Spark Scala
— with the corpus chunked semantically, batched by its real dependency graph,
and converted through an LLM behind an AI Gateway.

A run is driven from SharePoint by default: request rows say which applications
to convert and with which model, the scripts come out of the document library,
and the notebooks go back into it. A local directory of `.sas` files is the
offline fallback.

## Install

```bash
uv sync
```

Optional extras, each pulling only what it needs:

| Extra | For |
|---|---|
| `sharepoint` | Microsoft Graph — the request lists and the document library |
| `vault` | HashiCorp Vault — the AI Gateway credential |
| `azure` | Entra ID (MSAL) — the service principal that reaches Vault |
| `databricks` | Workspace and SQL warehouse access |
| `spark` | Delta-backed conversation memory and validation tracking |
| `graph` | The complexity report's dependency-graph PNG |
| `sql` | Real SQL parsing for the Spark SQL syntax check and the XREF post-rewriter |
| `dev` | pytest, ruff, and the test-only dependencies |

A real run wants `uv pip install -e ".[sharepoint,vault,azure]"`.

## Running a conversion

```bash
python main.py                              # every pending request row
python main.py --request-id 42              # one row
python main.py --app "MyApp"                # one application's rows
python main.py --no-upload                  # dry run: convert, write nothing back
python main.py path/to/sas --out-dir out/   # the local fallback
python main.py path/to/sas --md report.md --pdf report.pdf
```

Installed as a console script, so `sas-parser --app MyApp` is the same command.
`python main.py --help` lists every flag.

The mode is explicit either way — a positional path means local, its absence
means SharePoint. Nothing falls back silently: converting the wrong corpus
because a config key was missing is worse than a clear error.

### What a SharePoint run does

1. Reads the pending rows of the requests list.
2. Pulls each application's scripts from `{base}/{app}/scripts_original`.
3. Applies the XREF cross-reference — dataset names, and physical paths for
   rows marked as such.
4. Converts the whole application as **one corpus on one thread**, so
   cross-file dataset and macro dependencies resolve.
5. Uploads one notebook per source file to
   `{base}/{app}/scripts_converted/{model}/{timestamp}`, plus validation
   artefacts under `{base}/{app}/scripts_converted/validation` when the row
   asks for them.
6. Writes the row's `Status`.

One row failing does not stop the others; the exit status is non-zero if any
did.

## Other entry points

```bash
python -m complexity path/to/sas --out-dir reports/   # offline complexity + sizing
python -m complexity --sharepoint --app "MyApp"       # from the complexity list
python -m validation --help                           # the offline validation suite
```

## Configuration

Everything non-secret lives in [`config.json`](config.json), which documents
each section inline. Every key is also readable from an environment variable
that **wins over the file** — see [`.env.example`](.env.example) for the full
list. Secrets are never read from `config.json`.

Precedence is the same everywhere: **explicit argument > environment variable >
`config.json` > code default**. A JSON `null` means "unset", so a template
config listing every key changes nothing until edited.

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
