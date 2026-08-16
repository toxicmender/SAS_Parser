# SAS_Parser application image: the repo + every optional extra, on a JVM so
# the pyspark paths (memory's Delta backend, validation.tracking) work.
#
# Build context is the REPO ROOT:
#   docker build -f docker/app.Dockerfile -t sas-parser-app .
# (docker-compose.yml at the root does exactly that.)

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim-bookworm

# Re-declared: an ARG before FROM is out of scope in the build stage itself.
ARG PYTHON_VERSION

# JDK 17: the version pyspark 4.x runs on (21 also works; 17 is what bookworm
# ships). `procps` is what Spark's launcher scripts shell out to for `ps`.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      openjdk-17-jre-headless \
      ca-certificates \
      curl \
      procps \
 && rm -rf /var/lib/apt/lists/*

# uv drives the install so the image resolves exactly what uv.lock pins —
# the same resolution CI uses.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies first, project second: editing repo code must not re-resolve
# ~600 MB of wheels. Every extra is installed — the point of this image is that
# the optional imports (pyspark, hvac, msal, databricks-sdk, msgraph, sqlglot,
# matplotlib, and data_hydration's drivers via `hydration`) all resolve, exactly
# like the CI `types` job. `hydration` is also what makes
# tests/test_data_hydration_delta.py runnable here — the Delta sink is the one
# module that cannot be tested against a fake.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project \
      --extra dev --extra spark --extra sql --extra graph \
      --extra vault --extra azure --extra databricks --extra sharepoint \
      --extra hydration

COPY . .

# Installs the project itself (editable, as uv does for the workspace root) —
# which is why bind-mounting the repo over /app in compose picks up edits
# without a rebuild.
RUN uv sync --locked \
      --extra dev --extra spark --extra sql --extra graph \
      --extra vault --extra azure --extra databricks --extra sharepoint \
      --extra hydration

# Spark ships inside the pyspark wheel; no separate distribution to install.
# The `test -x` makes a wrong path a build failure rather than a runtime one.
ENV SPARK_HOME=/opt/venv/lib/python${PYTHON_VERSION}/site-packages/pyspark \
    SPARK_CONF_DIR=/opt/spark-conf
RUN test -x "$SPARK_HOME/bin/spark-submit" && mkdir -p "$SPARK_CONF_DIR"

# Delta Lake (memory.store's Delta backend + tests/test_backend_contract.py).
# delta-spark is deliberately not in pyproject.toml — it is installed only
# where the Delta backend actually runs, which is here. Pass
# --build-arg WITH_DELTA=0 to skip it (e.g. when no delta-spark release
# supports the locked pyspark yet, or to build without network access to
# Maven Central).
ARG WITH_DELTA=1
ARG DELTA_SPARK_VERSION=
COPY docker/spark/install_delta.sh docker/spark/warmup.py /opt/docker/
RUN chmod +x /opt/docker/install_delta.sh \
 && WITH_DELTA="${WITH_DELTA}" DELTA_SPARK_VERSION="${DELTA_SPARK_VERSION}" \
    /opt/docker/install_delta.sh

# Warehouse / event data lives on a volume, not in the layer.
RUN mkdir -p /data/warehouse

# No ENTRYPOINT: every credential and Spark setting arrives as environment from
# compose, which `docker compose exec` inherits — an entrypoint script would
# not, since exec bypasses it.
CMD ["python", "demo_run.py", "--help"]
