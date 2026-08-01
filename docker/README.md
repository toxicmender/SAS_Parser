# Docker stack: Vault + Spark + the app

A local stack for running SAS_Parser against the two external systems it
integrates with — HashiCorp Vault for credentials, and Spark for the Delta
paths (`memory.store`'s Delta backend and `validation.tracking`).

| Service        | What it is                                                     | Exposed |
| -------------- | -------------------------------------------------------------- | ------- |
| `vault`        | HashiCorp Vault, **dev mode** (in-memory, auto-unsealed, HTTP)  | `:8200` |
| `vault-init`   | One-shot provisioning: kv-v2, AppRole, seeded secrets           | —       |
| `spark-master` | Standalone Spark master                                         | `:7077`, UI `:8080` |
| `spark-worker` | Standalone Spark worker (scalable)                              | UI via master |
| `app`          | The repo with every extra installed; idles, you `exec` into it  | —       |

Files: [`../docker-compose.yml`](../docker-compose.yml),
[`app.Dockerfile`](app.Dockerfile), [`spark.Dockerfile`](spark.Dockerfile),
[`vault.Dockerfile`](vault.Dockerfile).

Requires Docker Compose ≥ 2.24 (the `env_file: { required: false }` form).

## Quick start

```bash
docker compose up -d --build
```

Then drive the CLI inside the app container:

```bash
docker compose exec app python demo_run.py local sas_scripts --out-dir out --md out/report.md
```

```bash
docker compose exec app python -m complexity sas_scripts --out complexity_report.md
```

```bash
docker compose exec app pytest -q
```

The repo is bind-mounted at `/app` and the project is installed editable, so
host edits are live in the container and everything written under `/app`
(notebooks, reports, `.prompt_builder_cache/`) lands back on the host.

Use `docker compose exec`, not `docker compose run`, for anything that starts a
Spark session against the cluster: `run` containers do not get the service's
network alias, so the executors cannot call the driver back. If you need a
one-off container anyway, `docker compose run --rm --use-aliases app ...`.

### Smoke test

Two commands that exercise both halves of the stack:

```bash
docker compose exec app python -c "from app_config.vault import get_secret; print(get_secret('llm/anthropic', 'api_key'))"
```

```bash
docker compose exec app python -c "import os; from pyspark.sql import SparkSession; s = SparkSession.builder.master(os.environ['SPARK_MASTER_URL']).getOrCreate(); s.sql('CREATE TABLE IF NOT EXISTS smoke (id BIGINT) USING DELTA'); s.sql('INSERT INTO smoke VALUES (7)'); print('rows:', s.table('smoke').count()); s.stop()"
```

The first prints the seeded key (an AppRole login against the dev Vault); the
second writes and reads a Delta table through the cluster's executors.

## What `vault-init` sets up

[`vault/init.sh`](vault/init.sh) is idempotent and re-runs on every `up` — a
dev-mode Vault forgets everything when it restarts.

| Path                        | Read by                                                       |
| --------------------------- | ------------------------------------------------------------- |
| `secret/appsvc/ai_gateway`  | the default credential chain (`app_config.vault.AI_GATEWAY_PATH`) |
| `secret/llm/anthropic`      | `demo_run.py ... --vault-secret llm/anthropic` (explicit AppRole read) |

Both carry `api_key` (and `base_url` on the gateway secret when
`OPENAI_BASE_URL` is set), taken from `OPENAI_API_KEY` in your environment or
`.env`. Without it they hold an obvious placeholder, so a run fails at the
gateway with a 401 rather than looking like a Vault problem.

It also enables the `approle` auth method with role `sas-parser`, scoped by a
policy that can read only those two paths, and **pins** the role's credentials
to `DEV_VAULT_ROLE_ID` / `DEV_VAULT_SECRET_ID` (Vault's custom-role-id and
custom-secret-id endpoints) rather than generating them. Compose then hands the
app the same pair as `VAULT_ROLE_ID` / `VAULT_SECRET_ID`.

Pinning is what makes `docker compose exec` work: `exec` starts a process
beside PID 1 and does not run the image's entrypoint, so credentials delivered
through a file or an entrypoint script would reach the idle `sleep infinity`
process and nothing else. Environment set by compose reaches every container
process, `exec` and `run` alike. Before exiting, `init.sh` performs a real
AppRole login with the pair, so a broken identity fails there rather than in
the middle of a pipeline run.

Check it by hand:

```bash
docker compose exec vault vault kv get secret/appsvc/ai_gateway
```

```bash
docker compose exec app python -c "from app_config.vault import get_secret; print(get_secret('llm/anthropic', 'api_key'))"
```

### Dev mode is dev mode

Three things here exist only because this is a laptop stack, and none should be
copied outwards: Vault runs unsealed with a known root token over plain HTTP;
the app trusts it via `VAULT_SKIP_VERIFY=true`; and the AppRole `secret_id` is a
constant that lives in the compose file. In a real deployment the credentials are
delivered by the platform, or the `azuread` chain is used instead — an Entra ID
service principal mints a JWT, Vault's `jwt` auth method trades it for a token,
and nothing long-lived touches disk. That chain needs a real tenant, so it
cannot be exercised locally; set `VAULT_OIDC_ROLE` and the `AZURE_*`/`ARM_*`
variables to use it (see `app_config/vault.py`).

To point the stack at a **real Vault** instead, drop the `vault` and
`vault-init` services and set `VAULT_ADDR`, `VAULT_NAMESPACE`, and either
`VAULT_TOKEN` or a real `VAULT_ROLE_ID`/`VAULT_SECRET_ID` pair in the `app`
service's environment. Nothing else changes: `app_config.vault` reads the same
standard variables either way.

## Spark, and the Databricks question

**Databricks cannot be run locally** — there is no self-hostable image of it; a
workspace is a hosted service. What this stack gives you instead is the engine
underneath it: Apache Spark with Delta Lake, which is what `memory.store`'s
Delta backend and `validation.tracking` actually talk to. Code that works
against this cluster is the same code that runs on a Databricks cluster; what
you do *not* get locally is Unity Catalog, workspace APIs, or DBFS.

Spark comes from the **pyspark wheel** — no separate distribution — pinned to
the version `uv.lock` pins for the app (`PYSPARK_VERSION`, currently 4.1.2). A
driver and a cluster on different Spark versions fail at handshake, so both
images take the version from the same variable. The pip wheel ships
`bin/spark-class` but not `sbin/start-master.sh`, which is why
[`spark/entrypoint.sh`](spark/entrypoint.sh) launches the daemon classes
directly.

Delta Lake is installed by [`spark/install_delta.sh`](spark/install_delta.sh),
run identically in both images: it pip-installs `delta-spark` (pinning pyspark
so the resolver picks a *compatible* Delta rather than downgrading Spark),
writes `spark-defaults.conf` with the matching maven coordinate plus the Delta
extension and catalog, and pre-resolves the jars into `/opt/ivy` so the first
session neither waits on Maven Central nor needs to reach it. `delta-spark` is
deliberately absent from `pyproject.toml` — it belongs where the Delta backend
actually runs, which is this image.

Run something against the cluster:

```bash
docker compose exec app python -c "from pyspark.sql import SparkSession; s = SparkSession.builder.master('spark://spark-master:7077').getOrCreate(); print(s.range(5).count()); s.stop()"
```

Anything the cluster touches must live on a path **all** the containers share —
in practice `/data/warehouse`, the `spark-warehouse` volume, which is also
`spark.sql.warehouse.dir`. A Delta table written to a driver-local path (`/tmp`,
or a relative path under `/app`) fails on read with `FILE_NOT_EXIST`: the
executors wrote their part files on the worker container's own filesystem.
Managed tables (`USING DELTA` with no `LOCATION`) land in the warehouse and are
fine.

The Delta backend of the KV store, on the cluster:

```bash
docker compose exec app python -c "from pyspark.sql import SparkSession; from memory.store import KVStore; s = SparkSession.builder.master('spark://spark-master:7077').getOrCreate(); kv = KVStore(spark=s, table='default.kv_demo'); kv.set('hello', 'world'); print(kv.get('hello')); s.stop()"
```

More workers:

```bash
docker compose up -d --scale spark-worker=3
```

Warehouse data lives on the `spark-warehouse` volume, mounted at
`/data/warehouse` in every container.

### Talking to a real Databricks workspace

The app image installs the `databricks` extra (`databricks-sdk`,
`databricks-sql-connector`), and `app_config.databricks` reads the standard
variables. Set `DATABRICKS_HOST`, `DATABRICKS_TOKEN` (or the `ARM_*` service
principal), and `DATABRICKS_HTTP_PATH` in `.env` — compose passes all three
through — and the SQL-warehouse paths work from inside the container while the
local cluster stays available for everything else.

`databricks-connect` is *not* installed: it replaces `pyspark` with its own
build, which would fight the locked version and break the local cluster. Adding
it means a separate image.

## Configuration

Compose reads the repo-root `.env` for substitutions, and the `app` service
also loads it as an env-file (optional — no `.env`, no error). Service-level
`environment:` entries win, so a `.env` written for bare-metal runs
(`VAULT_ADDR=https://vault.example:8200`) does not misdirect the container.

| Variable                 | Default | Effect                                        |
| ------------------------ | ------- | --------------------------------------------- |
| `OPENAI_API_KEY`         | —       | seeded into both Vault secrets, and passed to the app |
| `OPENAI_BASE_URL`        | —       | gateway endpoint; adds `base_url` to the gateway secret |
| `VAULT_DEV_ROOT_TOKEN_ID`| `root`  | dev Vault's root token                        |
| `DEV_VAULT_ROLE_ID`      | `1111…` | AppRole role_id, pinned by init and given to the app |
| `DEV_VAULT_SECRET_ID`    | `2222…` | AppRole secret_id, likewise (dev constant, not a secret) |
| `VAULT_PORT`             | `8200`  | host port for Vault                           |
| `VAULT_VERSION`          | `1.18`  | `hashicorp/vault` image tag                   |
| `PYSPARK_VERSION`        | `4.1.2` | Spark version for the cluster image; keep equal to `uv.lock` |
| `PYTHON_VERSION`         | `3.12`  | base image Python                             |
| `SPARK_WORKER_CORES`     | `2`     | per worker                                    |
| `SPARK_WORKER_MEMORY`    | `2g`    | per worker                                    |
| `SPARK_MASTER_UI_PORT`   | `8080`  | host port for the master UI                   |
| `WITH_DELTA`             | `1`     | `0` builds both images without Delta Lake     |
| `DELTA_SPARK_VERSION`    | —       | pin `delta-spark` instead of letting the resolver choose |
| `DATABRICKS_HOST` / `_TOKEN` / `_HTTP_PATH` | — | a real workspace |

## Troubleshooting

**The build fails while installing `delta-spark`.** No published Delta release
supports the locked pyspark yet. Build without it —
`WITH_DELTA=0 docker compose build` — or pin a known-good pair with
`DELTA_SPARK_VERSION` and `PYSPARK_VERSION`. Everything except the Delta
backend still works; `tests/test_backend_contract.py`'s Delta half skips itself.

**A Spark job hangs at `Initial job has not accepted any resources`.** The
driver is unreachable from the executors — almost always a
`docker compose run` container without `--use-aliases`. Use
`docker compose exec app ...`.

**`could not fetch the AI Gateway credential from Vault`.** `vault-init` did
not finish, or Vault restarted and lost its state (dev mode is in-memory).
`docker compose up -d vault-init` re-provisions it; `docker compose logs
vault-init` shows what it wrote.

**Vault is healthy but the app cannot authenticate.** The dev Vault restarted
and forgot the AppRole registration. `docker compose up -d vault-init` re-pins
the same credentials, so nothing on the app side has to change.

**The app image rebuilds from scratch on every code change.** Only if
`pyproject.toml` or `uv.lock` changed — dependencies are installed in their own
layer before the source is copied. Editing repo code needs no rebuild at all,
because `/app` is bind-mounted.
