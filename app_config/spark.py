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

Why the session factory lives here after all
--------------------------------------------
This module used to return only a *string*, leaving every caller to build its
own ``SparkSession`` — on the grounds that :mod:`app_config` is the
dependency-free leaf every other package imports and must never import
pyspark. The rule was right; the conclusion was not. Four call sites
(:meth:`pipeline.setup.MemorySetup._default_hub`,
:func:`validation.tracking._ensure_spark`, ``validation.__main__``, and
:func:`data_hydration.sinks.delta.get_session`) each grew their own copy of
``SparkSession.builder.master(master_url()).getOrCreate()``, and all four were
wrong in the same place: **on Databricks there is already a session, and
asking for one by master is at best ignored and at worst refused.**

:func:`active_or_new_session` is that decision made once. The pyspark import is
*inside* the function, so Architecture.md invariant 8 still holds exactly as
written: ``import app_config.spark`` costs nothing, and the in-memory paths —
which never call this — import and run with no pyspark installed at all.

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
from typing import Any

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


def active_or_new_session(app_name: str, *, master: str | None = None) -> Any:
    """The SparkSession to use: the active one, else a newly built one.

    **Reusing the active session is the whole point.** Inside a Databricks
    notebook or job the runtime has already built one, and it is the only
    session that can reach the workspace's catalogs and credentials. Building
    another by master is wrong there in three separate ways: on classic
    Dedicated compute the ``master`` argument is silently dropped (harmless,
    but every log line then claims a local cluster that is not what ran), and
    on Spark Connect — serverless, and classic *Standard* access mode since
    DBR 14.0 — ``master`` is not supported at all and the call raises.

    Off Databricks nothing is active, so a session is built against
    :func:`master_url` exactly as before: ``local[*]`` on a laptop, whatever
    ``SPARK_MASTER_URL`` says inside the Docker stack.

    Parameters
    ----------
    app_name : str
        ``appName`` for a session this call creates. Ignored when an active
        session is reused — renaming a running application is not possible,
        and pretending otherwise would put a fictional name in the logs.
    master : str | None
        Master URL for a session this call creates, resolved through
        :func:`master_url` when omitted. Also ignored when reusing.

    Returns
    -------
    pyspark.sql.SparkSession

    Raises
    ------
    ImportError
        pyspark is not installed. Only the Delta-backed paths reach this, so
        the message names the extra that provides it.
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(
            f"pyspark is required to build the '{app_name}' Spark session; "
            "install it with 'pip install \"sas-parser[spark]\"'. On a "
            "Databricks cluster it is already provided by the runtime — do "
            "not install the extra there."
        ) from exc

    active = SparkSession.getActiveSession()
    if active is not None:
        logger.info(
            f"active_or_new_session: reusing the active SparkSession for "
            f"'{app_name}' (the runtime's own, on Databricks)"
        )
        return active

    resolved = master_url(master)
    logger.info(
        f"active_or_new_session: no active session, building '{app_name}' "
        f"against {describe_master(resolved)}"
    )
    return SparkSession.builder.master(resolved).appName(app_name).getOrCreate()
