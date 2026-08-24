# Standalone Spark cluster (master + workers) for the local stand-in of a
# Databricks cluster.
#
# Spark comes from the *pyspark wheel*, pinned to the same version uv.lock
# pins for the app (PYSPARK_VERSION below). That is the point: a driver and a
# cluster on different Spark versions fail at handshake, and deriving both
# from one pinned wheel makes the parity mechanical instead of a comment.
#
# The pip distribution ships bin/spark-class but no sbin/start-master.sh, so
# the daemons are launched directly (and stay in the foreground, which is what
# a container wants anyway) — see docker/spark/entrypoint.sh.

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim-bookworm

ARG PYTHON_VERSION
# Keep in step with uv.lock's pyspark. Compose passes it in from
# PYSPARK_VERSION so both images move together.
ARG PYSPARK_VERSION=4.1.1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      openjdk-17-jre-headless \
      ca-certificates \
      curl \
      procps \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN uv venv /opt/venv \
 && uv pip install --python /opt/venv "pyspark==${PYSPARK_VERSION}"

ENV SPARK_HOME=/opt/venv/lib/python${PYTHON_VERSION}/site-packages/pyspark \
    SPARK_CONF_DIR=/opt/spark-conf
RUN test -x "$SPARK_HOME/bin/spark-class" && mkdir -p "$SPARK_CONF_DIR"

# Same Delta install as the app image — driver and executors must agree on the
# jars, so the two builds run the identical script.
ARG WITH_DELTA=1
ARG DELTA_SPARK_VERSION=
COPY docker/spark/install_delta.sh docker/spark/warmup.py /opt/docker/
RUN chmod +x /opt/docker/install_delta.sh \
 && WITH_DELTA="${WITH_DELTA}" DELTA_SPARK_VERSION="${DELTA_SPARK_VERSION}" \
    PYSPARK_VERSION="${PYSPARK_VERSION}" \
    /opt/docker/install_delta.sh

COPY docker/spark/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

RUN mkdir -p /data/warehouse /opt/spark-work

EXPOSE 7077 8080 8081

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["master"]
