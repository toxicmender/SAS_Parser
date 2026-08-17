"""Non-secret settings for a hydration run.

The two-layer shape the rest of the repo uses: a ``@dataclass`` per domain with a
:meth:`~HydrationConfig.from_env` classmethod reading the environment first and
the ``data_hydration`` section of ``config.json`` second, through
:func:`app_config.get_value` / :func:`app_config.get_typed_value` so a
wrong-typed entry degrades with a WARNING instead of crashing a run.

**No secret is a field here.** Passwords, private keys and storage keys resolve
at connection time through :mod:`data_hydration.secrets`, which reads the
Databricks secret scope, Vault, Entra ID and the environment in that order.
``config.json`` is checked into deployments and must never hold one.

Logger name: ``data_hydration.config``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app_config import get_typed_value, get_value

from .naming import DEFAULT_DATE_FORMAT, DEFAULT_TEMPLATE

logger = logging.getLogger(__name__)

DEFAULT_NUM_PARTITIONS = 8
DEFAULT_FETCH_SIZE = 10_000
DEFAULT_SFTP_PORT = 22
DEFAULT_ON_ERROR = "continue"

_ON_ERROR_VALUES = frozenset({"continue", "stop"})


def _resolve_on_error() -> str:
    """``continue`` or ``stop``; an unrecognised value degrades with a WARNING."""
    configured = get_typed_value("data_hydration", "on_error", str, DEFAULT_ON_ERROR)
    if configured not in _ON_ERROR_VALUES:
        logger.warning(
            f"data_hydration: config.json data_hydration.on_error {configured!r} is "
            f"not one of {'/'.join(sorted(_ON_ERROR_VALUES))}; "
            f"using {DEFAULT_ON_ERROR!r}"
        )
        return DEFAULT_ON_ERROR
    return configured


@dataclass
class HydrationConfig:
    """Everything a run needs that is not a secret.

    Construct it directly to pin values, or call :meth:`from_env` for the
    standard environment-then-``config.json`` resolution.

    Attributes
    ----------
    catalog : str | None
        Unity Catalog catalog every target table lands in.
        ``DATA_HYDRATION_CATALOG`` / ``data_hydration.catalog``. Required unless
        the template omits ``<catalog_name>``.
    schema : str | None
        Target schema. ``DATA_HYDRATION_SCHEMA`` / ``data_hydration.schema``.
        Defaults per source to the SAS libref when unset, which is usually what
        a migration wants.
    table_template : str
        The target-name shape — see :mod:`data_hydration.naming`.
        ``DATA_HYDRATION_TABLE_TEMPLATE`` / ``data_hydration.table_template``,
        default ``<catalog_name>.<schema_name>.<table_name>``.
    stage : str | None
        The load stage (``raw``, ``bronze``, ...) filled into ``<stage>``.
        ``DATA_HYDRATION_STAGE`` / ``data_hydration.stage``. Required only when
        the template names it.
    date_format : str
        ``strftime`` format for ``<date>``. ``DATA_HYDRATION_DATE_FORMAT`` /
        ``data_hydration.date_format``, default ``%Y%m%d``.
    secret_scope : str | None
        Databricks secret scope holding hydration credentials.
        ``DATA_HYDRATION_SECRET_SCOPE`` / ``data_hydration.secret_scope``;
        falls back to ``databricks.secret_scope``, which is where the rest of
        this deployment's principals already live.
    num_partitions : int
        How many slices a partitionable source is divided into when it has no
        native partitioning of its own. ``data_hydration.num_partitions``,
        default 8.
    fetch_size : int
        Rows per round trip for a cursor-based source.
        ``data_hydration.fetch_size``, default 10000.
    staging_dir : str | None
        Where a file source is buffered on its way to a table. Ephemeral —
        nothing here is a deliverable. ``data_hydration.staging_dir``; ``None``
        uses the system temporary directory.
    apply_index_clustering : bool
        Turn a SAS index into ``CLUSTER BY`` on the Delta table. Off by default:
        a SAS index and Delta clustering solve overlapping but different
        problems, so the hint is reported and applied only on request.
        ``data_hydration.apply_index_clustering``.
    on_error : str
        ``continue`` (default) records a failed item and carries on;
        ``stop`` ends the run at the first failure.
        ``data_hydration.on_error``.
    oracle_dsn, oracle_user : str | None
        Connection defaults used when a ``LIBNAME`` does not carry its own.
        ``DATA_HYDRATION_ORACLE_DSN`` / ``DATA_HYDRATION_ORACLE_USER``.
    sftp_host, sftp_port, sftp_username, sftp_key_path : ...
        sFTP connection defaults. The *passphrase* and password are secrets and
        are not here.
    adls_account, adls_filesystem, blob_account, blob_container : str | None
        Azure storage coordinates. Authentication is an Entra ID token minted
        through :mod:`app_config.azure`, so no key is configured here.
    sas_host, sas_port, sas_context : ...
        ``saspy`` session coordinates, needed for SPD Engine reads.
    """

    catalog: str | None = None
    schema: str | None = None
    table_template: str = DEFAULT_TEMPLATE
    stage: str | None = None
    date_format: str = DEFAULT_DATE_FORMAT
    secret_scope: str | None = None
    num_partitions: int = DEFAULT_NUM_PARTITIONS
    fetch_size: int = DEFAULT_FETCH_SIZE
    staging_dir: str | None = None
    apply_index_clustering: bool = False
    on_error: str = DEFAULT_ON_ERROR

    oracle_dsn: str | None = None
    oracle_user: str | None = None

    sftp_host: str | None = None
    sftp_port: int = DEFAULT_SFTP_PORT
    sftp_username: str | None = None
    sftp_key_path: str | None = None

    adls_account: str | None = None
    adls_filesystem: str | None = None
    blob_account: str | None = None
    blob_container: str | None = None

    sas_host: str | None = None
    sas_port: int | None = None
    sas_context: str | None = None

    # Set at construction, not read from anywhere: the single instant every
    # ``<date>`` in this run renders from. See :meth:`run_date`.
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc), repr=False
    )

    @classmethod
    def from_env(cls) -> "HydrationConfig":
        """Resolve every non-secret setting from the environment, then config.json."""
        return cls(
            catalog=(
                os.environ.get("DATA_HYDRATION_CATALOG")
                or get_value("data_hydration", "catalog")
            ),
            schema=(
                os.environ.get("DATA_HYDRATION_SCHEMA")
                or get_value("data_hydration", "schema")
            ),
            table_template=(
                os.environ.get("DATA_HYDRATION_TABLE_TEMPLATE")
                or get_typed_value(
                    "data_hydration", "table_template", str, DEFAULT_TEMPLATE
                )
            ),
            stage=(
                os.environ.get("DATA_HYDRATION_STAGE")
                or get_value("data_hydration", "stage")
            ),
            date_format=(
                os.environ.get("DATA_HYDRATION_DATE_FORMAT")
                or get_typed_value(
                    "data_hydration", "date_format", str, DEFAULT_DATE_FORMAT
                )
            ),
            secret_scope=(
                os.environ.get("DATA_HYDRATION_SECRET_SCOPE")
                or get_value("data_hydration", "secret_scope")
                # One scope holds this deployment's principals already; hydration
                # keys live beside them unless told otherwise.
                or get_value("databricks", "secret_scope")
            ),
            num_partitions=get_typed_value(
                "data_hydration", "num_partitions", int, DEFAULT_NUM_PARTITIONS
            ),
            fetch_size=get_typed_value(
                "data_hydration", "fetch_size", int, DEFAULT_FETCH_SIZE
            ),
            staging_dir=(
                os.environ.get("DATA_HYDRATION_STAGING_DIR")
                or get_value("data_hydration", "staging_dir")
            ),
            apply_index_clustering=bool(
                get_typed_value(
                    "data_hydration", "apply_index_clustering", bool, False
                )
            ),
            on_error=_resolve_on_error(),
            oracle_dsn=(
                os.environ.get("DATA_HYDRATION_ORACLE_DSN")
                or get_value("data_hydration", "oracle_dsn")
            ),
            oracle_user=(
                os.environ.get("DATA_HYDRATION_ORACLE_USER")
                or get_value("data_hydration", "oracle_user")
            ),
            sftp_host=(
                os.environ.get("DATA_HYDRATION_SFTP_HOST")
                or get_value("data_hydration", "sftp_host")
            ),
            sftp_port=get_typed_value(
                "data_hydration", "sftp_port", int, DEFAULT_SFTP_PORT
            ),
            sftp_username=(
                os.environ.get("DATA_HYDRATION_SFTP_USERNAME")
                or get_value("data_hydration", "sftp_username")
            ),
            sftp_key_path=(
                os.environ.get("DATA_HYDRATION_SFTP_KEY_PATH")
                or get_value("data_hydration", "sftp_key_path")
            ),
            adls_account=(
                os.environ.get("DATA_HYDRATION_ADLS_ACCOUNT")
                or get_value("data_hydration", "adls_account")
            ),
            adls_filesystem=(
                os.environ.get("DATA_HYDRATION_ADLS_FILESYSTEM")
                or get_value("data_hydration", "adls_filesystem")
            ),
            blob_account=(
                os.environ.get("DATA_HYDRATION_BLOB_ACCOUNT")
                or get_value("data_hydration", "blob_account")
            ),
            blob_container=(
                os.environ.get("DATA_HYDRATION_BLOB_CONTAINER")
                or get_value("data_hydration", "blob_container")
            ),
            sas_host=(
                os.environ.get("DATA_HYDRATION_SAS_HOST")
                or get_value("data_hydration", "sas_host")
            ),
            sas_port=get_typed_value("data_hydration", "sas_port", int),
            sas_context=(
                os.environ.get("DATA_HYDRATION_SAS_CONTEXT")
                or get_value("data_hydration", "sas_context")
            ),
        )

    @property
    def run_date(self) -> str:
        """:attr:`started_at` formatted with :attr:`date_format`.

        Derived from the instant fixed when this config was built, so every
        table in one run carries the same date however long the run takes.
        (:func:`app_config.utc_stamp` is the fixed-format sibling that names run
        *folders*; this one is configurable because it ends up inside an
        identifier.)
        """
        try:
            return self.started_at.strftime(self.date_format)
        except ValueError:
            logger.warning(
                f"data_hydration: date_format {self.date_format!r} is not a valid "
                f"strftime format; using {DEFAULT_DATE_FORMAT!r}"
            )
            return self.started_at.strftime(DEFAULT_DATE_FORMAT)

    @property
    def has_sas_session(self) -> bool:
        """True when enough is configured to open a ``saspy`` session.

        Read by the planner: an SPD Engine library with no session behind it is
        planned and reported, but marked as needing an operator.
        """
        return bool(self.sas_host)
