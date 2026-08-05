"""Where the Spark cluster is — the non-secret connection setting.

Submodule of :mod:`app_config`, and the Spark counterpart to
:mod:`app_config.vault`'s connection half: it answers "which master do we
talk to?" and nothing else. There is no secret here, so unlike the Vault
module this one is pure resolution.

Resolution order, matching the rest of the package
(:func:`app_config.resolve`) with the environment given the same precedence
the ``vault`` and ``databricks`` sections give theirs:

    explicit argument  >  ``SPARK_MASTER_URL``  >  ``config.json`` spark.master
                       >  ``local[*]``

``local[*]`` stays the default, so nothing changes for a bare-metal run that
configures nothing. Inside the Docker stack, compose already sets
``SPARK_MASTER_URL`` on the ``app`` service (see ``docker-compose.yml``), so
setting it here is what lets the same code reach the real cluster without a
flag or a code change — the same "change env vars, change nothing else" rule
that points the app at a real Vault.

Why this is not a session factory
---------------------------------
This module returns a *string*. Building the ``SparkSession`` stays at the
call sites, because :mod:`app_config` is the dependency-free leaf every other
package imports (see the package docstring) and must never import pyspark —
Architecture.md invariant 8 requires the in-memory paths to import and run
with no pyspark installed at all.

A note on ``spark-defaults.conf``
---------------------------------
Spark reads ``spark.master`` from ``$SPARK_CONF_DIR/spark-defaults.conf``
already, and the app image writes one. But an explicit
``SparkSession.builder.master(...)`` call overrides that file, so a
hard-coded ``local[*]`` silently wins over any deployment's configuration.
Callers should pass what :func:`master_url` returns rather than a literal.

Logger name: ``app_config.spark``.
"""

from __future__ import annotations

import logging
import os

from . import get_typed_value

logger = logging.getLogger(__name__)

#: Used when neither the environment nor ``config.json`` names a master. A
#: local in-process cluster is the right default for a laptop run and for the
#: test suite; it is the wrong one inside the compose stack, which is exactly
#: why compose sets ``SPARK_MASTER_URL``.
DEFAULT_MASTER = "local[*]"

ENV_VAR = "SPARK_MASTER_URL"


def master_url(explicit: str | None = None) -> str:
    """The Spark master URL to build a session against.

    *explicit* (a ``--master`` flag, a caller's own choice) wins when given;
    otherwise the ``SPARK_MASTER_URL`` environment variable, then
    ``config.json``'s ``spark.master``, then :data:`DEFAULT_MASTER`.

    A blank or whitespace-only environment variable counts as unset — an
    empty value in a ``.env`` file means "I did not configure this", not
    "build a session with no master" (which pyspark would reject).
    """
    if explicit is not None:
        return explicit

    env = os.environ.get(ENV_VAR, "").strip()
    if env:
        return env

    configured = get_typed_value("spark", "master", str, DEFAULT_MASTER)
    return configured.strip() or DEFAULT_MASTER


def describe_master(master: str) -> str:
    """A short phrase naming what *master* points at, for one log line."""
    return "a local in-process cluster" if master.startswith("local") else master
