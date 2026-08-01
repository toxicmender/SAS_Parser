#!/usr/bin/env bash
# Launch a standalone Spark daemon in the foreground.
#
#   master  -> org.apache.spark.deploy.master.Master
#   worker  -> org.apache.spark.deploy.worker.Worker
#   <other> -> exec'd as-is (spark-submit, spark-sql, bash, ...)
#
# The pyspark wheel ships bin/spark-class but not sbin/start-master.sh, so the
# daemon classes are invoked directly. That is the better fit for a container
# regardless: no nohup, no pid files, signals reach the JVM, and the logs go to
# stdout where `docker compose logs` can see them.
set -euo pipefail

SPARK_HOME="${SPARK_HOME:?SPARK_HOME is not set}"

# --host is what the master advertises to workers and drivers, so it must be
# the name they resolve it by (the compose service name), not 0.0.0.0.
SPARK_MASTER_HOST="${SPARK_MASTER_HOST:-spark-master}"
SPARK_MASTER_PORT="${SPARK_MASTER_PORT:-7077}"
SPARK_MASTER_WEBUI_PORT="${SPARK_MASTER_WEBUI_PORT:-8080}"
SPARK_MASTER_URL="${SPARK_MASTER_URL:-spark://${SPARK_MASTER_HOST}:${SPARK_MASTER_PORT}}"
SPARK_WORKER_WEBUI_PORT="${SPARK_WORKER_WEBUI_PORT:-8081}"

case "${1:-master}" in
    master)
        exec "$SPARK_HOME/bin/spark-class" org.apache.spark.deploy.master.Master \
            --host "$SPARK_MASTER_HOST" \
            --port "$SPARK_MASTER_PORT" \
            --webui-port "$SPARK_MASTER_WEBUI_PORT"
        ;;
    worker)
        # Cores/memory are passed as flags: SPARK_WORKER_CORES and
        # SPARK_WORKER_MEMORY are read by sbin/start-worker.sh, which this
        # image does not have.
        # `if`, not `[ ... ] && ...`: under `set -e` a false test as the last
        # command of the line would end the script instead of skipping a flag.
        args=("$SPARK_MASTER_URL" --webui-port "$SPARK_WORKER_WEBUI_PORT")
        if [ -n "${SPARK_WORKER_CORES:-}" ]; then args+=(--cores "$SPARK_WORKER_CORES"); fi
        if [ -n "${SPARK_WORKER_MEMORY:-}" ]; then args+=(--memory "$SPARK_WORKER_MEMORY"); fi
        if [ -n "${SPARK_WORKER_DIR:-}" ]; then args+=(--work-dir "$SPARK_WORKER_DIR"); fi
        exec "$SPARK_HOME/bin/spark-class" org.apache.spark.deploy.worker.Worker "${args[@]}"
        ;;
    *)
        exec "$@"
        ;;
esac
