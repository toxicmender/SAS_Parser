# Reproducible CI runtime for the real PySpark and Delta Lake contracts.
#
# Build context is the repository root. Unlike app.Dockerfile, this image only
# installs the dependencies needed by the test suite: it is deliberately small
# enough to rebuild on a GitHub-hosted runner while still reusing the exact
# Delta installation and build-time compatibility probe used by deployment.

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim-bookworm

ARG PYTHON_VERSION
ARG DELTA_SPARK_VERSION=4.1.0

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      openjdk-17-jre-headless \
      ca-certificates \
      curl \
      procps \
 && rm -rf /var/lib/apt/lists/*

# Pin the installer used by CI rather than allowing the image to change when
# ghcr.io/astral-sh/uv:latest moves. Project dependencies remain locked by
# uv.lock below.
COPY --from=ghcr.io/astral-sh/uv:0.9.21 /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH \
    PYSPARK_PYTHON=/opt/venv/bin/python \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SPARK_LOCAL_IP=127.0.0.1 \
    SPARK_CONF_DIR=/opt/spark-conf

WORKDIR /app

# Keep dependency installation cacheable across source-only changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --extra dev --extra spark

ENV SPARK_HOME=/opt/venv/lib/python${PYTHON_VERSION}/site-packages/pyspark
RUN test -x "$SPARK_HOME/bin/spark-submit" \
 && mkdir -p "$SPARK_CONF_DIR" /data/warehouse

# This shared script installs the pinned Python package, resolves the matching
# Maven jars into the image, and runs real path/catalog/property probes during
# the build. The test container therefore needs no Maven network access.
COPY docker/spark/install_delta.sh docker/spark/warmup.py /opt/docker/
RUN chmod +x /opt/docker/install_delta.sh \
 && DELTA_SPARK_VERSION="${DELTA_SPARK_VERSION}" \
    /opt/docker/install_delta.sh

# Install the editable project without reconciling the environment again:
# another `uv sync` here would correctly remove deployment-only delta-spark,
# because that package intentionally is not part of uv.lock.
COPY . .
RUN uv pip install --python /opt/venv --no-deps --editable .

CMD ["python", "-m", "pytest", "-v", "tests/test_spark_delta_runtime.py", "tests/test_v2_memory_delta.py", "tests/test_backend_contract.py"]
