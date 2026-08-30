"""Shared pytest fixtures.

The one thing every test needs: independence from the repo's ``config.json``.

That file used to be an all-null template, so tests that never mentioned it
behaved as though no config existed. It no longer is — it enables the bundled
SparkSQL instruction set and sizes the retrieval budget for it — and a unit
test asserting "no prompt builder means no guidance" would otherwise fail for
a reason that has nothing to do with the code under test.

Isolating to an empty config restores exactly the old baseline while making
the dependency explicit: a test that wants a config value now sets one.

The second fixture here is ``delta_spark``, shared by every suite that needs a
real Delta session (``test_backend_contract.py``, ``test_data_hydration_delta.py``).
It lives here rather than in either of them because there can only be **one**:
``SparkSession.getOrCreate()`` returns the process's existing session, so a
second fixture defining its own would silently hand back the first one — same
JVM, first one's warehouse — and its configuration would appear to apply while
doing nothing.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sys
from importlib import metadata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

import app_config


@pytest.fixture(autouse=True)
def _isolate_repo_config(monkeypatch, tmp_path_factory):
    """Point every test at an empty config, whatever the repo ships."""
    cfg = tmp_path_factory.mktemp("cfg") / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(app_config.ENV_VAR, str(cfg))
    app_config.clear_cache()
    yield
    app_config.clear_cache()


def _delta_broken(what: str) -> str:
    """Failure text naming both versions — the pair is what's actually wrong."""
    try:
        versions = (
            f"delta-spark {metadata.version('delta-spark')} / "
            f"pyspark {metadata.version('pyspark')}"
        )
    except metadata.PackageNotFoundError:  # pragma: no cover - defensive
        versions = "the installed delta-spark / pyspark"
    return (
        f"{versions}: {what}. The two are most likely built against different "
        f"Spark versions — a mismatched delta-spark still serves path-based "
        f"writes but fails every catalog statement. Pin a compatible pair "
        f"(DELTA_SPARK_VERSION / PYSPARK_VERSION) or build with WITH_DELTA=0. "
        f"This is deliberately a failure, not a skip."
    )


@pytest.fixture(scope="session")
def delta_spark(tmp_path_factory):
    """A real local Delta session, or a skip when one genuinely cannot exist.

    Session-scoped and shared: see the module docstring on why there must be
    exactly one of these.

    An *installed but incompatible* Delta is a failure, not a skip — a
    delta-spark built against a different Spark minor serves path-based writes
    perfectly and fails every catalog statement, which is how an entirely broken
    Delta backend once sat behind a green suite reporting "skipped".
    """
    require_delta = os.environ.get("REQUIRE_DELTA_TESTS") == "1"
    if require_delta:
        try:
            __import__("pyspark")
            __import__("delta")
        except ImportError as exc:
            pytest.fail(
                "REQUIRE_DELTA_TESTS=1 but the Spark/Delta runtime is missing: "
                f"{exc}"
            )
    else:
        pytest.importorskip("pyspark")
        pytest.importorskip("delta", reason="delta-spark not installed")
    # delta-spark isn't a project dependency (installed only where Delta is
    # actually exercised); the importorskip above guards this at runtime.
    from delta import (
        configure_spark_with_delta_pip,  # pyright: ignore[reportMissingImports]
    )
    from pyspark.sql import SparkSession

    # Only a genuinely absent JVM is a skip. Anything else — delta imports,
    # java is here, and the session still will not build — is a real defect.
    if shutil.which("java") is None:
        if require_delta:
            pytest.fail("REQUIRE_DELTA_TESTS=1 but no JVM is available on PATH")
        pytest.skip("no JVM on PATH; Spark cannot start")

    warehouse = tmp_path_factory.mktemp("delta-warehouse")
    builder = (
        SparkSession.builder.master("local[1]")
        .appName("sas-parser-tests")
        .config("spark.ui.enabled", "false")
        # These are tiny contract tables. Delta defaults snapshot work to 50
        # partitions, which adds minutes of scheduler overhead on local[1]
        # without exercising a different code path.
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.databricks.delta.snapshotPartitions", "1")
        .config("spark.sql.warehouse.dir", str(warehouse))
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )
    try:
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
    except Exception as exc:
        raise AssertionError(_delta_broken("could not start a Delta session")) from exc

    spark.sparkContext.setLogLevel("WARN")

    # Both suites create tables through the session catalog. Probe that API here
    # so an incompatible delta-spark is named plainly once, instead of surfacing
    # as a wall of identical Py4J traces across every test.
    try:
        spark.sql("CREATE TABLE IF NOT EXISTS _delta_probe (id BIGINT) USING DELTA")
        spark.sql("DROP TABLE IF EXISTS _delta_probe")
    except Exception as exc:
        spark.stop()
        raise AssertionError(_delta_broken("cannot create a Delta table")) from exc

    yield spark
    spark.stop()
