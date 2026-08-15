"""Reading a file over sFTP.

``paramiko``, not ``asyncssh``: this package is synchronous throughout, and the
repo has already paid for mixing an event loop into a blocking facade once —
Architecture.md invariant 13 exists because a coroutine-driven client deadlocked
inside the notebook it was deployed in. A synchronous client has nothing to
deadlock.

paramiko's ``SFTPFile`` is already a seekable, buffered file object, so nothing
here needs :mod:`data_hydration.rawio` — that adapter exists for object storage,
which has no file handle to hand out. Wrapping a working file object in another
layer could only lose.

Credentials resolve through :mod:`data_hydration.secrets`: a key file path is
configuration, the passphrase and any password are not.

Logger name: ``data_hydration.sources.sftp``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from ..secrets import resolve_secret
from .base import SourceInfo, frames_from_file

if TYPE_CHECKING:
    from ..config import HydrationConfig
    from ..models import HydrationItem

logger = logging.getLogger(__name__)


def connect(config: "HydrationConfig", host: str) -> Any:
    """An open ``paramiko`` SFTP client for *host*.

    Key authentication is tried when ``sftp_key_path`` is set, password
    otherwise; both credentials come from the chain, never from config.json.
    """
    try:
        import paramiko
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(
            "paramiko is required to read over sFTP; install it with "
            "'pip install \"sas-parser[sftp]\"'"
        ) from exc

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    # RejectPolicy is the default and is what we want: silently trusting an
    # unknown host key is how an sFTP transfer ends up talking to the wrong
    # server. An operator adds the key to known_hosts deliberately.
    client.set_missing_host_key_policy(paramiko.RejectPolicy())

    kwargs: dict[str, Any] = {
        "hostname": host,
        "port": config.sftp_port,
        "username": config.sftp_username,
    }
    if config.sftp_key_path:
        kwargs["key_filename"] = config.sftp_key_path
        passphrase = resolve_secret(
            "sftp_passphrase", scope=config.secret_scope, required=False
        )
        if passphrase:
            kwargs["passphrase"] = passphrase
    else:
        kwargs["password"] = resolve_secret(
            "sftp_password", scope=config.secret_scope
        )

    logger.info(f"connect: sftp {config.sftp_username}@{host}:{config.sftp_port}")
    client.connect(**kwargs)
    return client


class SftpReader:
    """One remote file, streamed as DataFrames."""

    def __init__(self, item: "HydrationItem", config: "HydrationConfig") -> None:
        self._item = item
        self._config = config
        self._client: Any = None
        self._sftp: Any = None
        self._handle: Any = None
        source = item.source
        # The host comes from the FILENAME's own host= option when it carried
        # one, else the configured default.
        self._host = source.option_map.get("host") or config.sftp_host or source.locator
        self._path = f"{source.locator}/{source.object_name}".replace("//", "/")

    def _open(self) -> Any:
        if self._handle is None:
            self._client = connect(self._config, self._host)
            self._sftp = self._client.open_sftp()
            self._handle = self._sftp.open(self._path, "rb")
            # Without this paramiko issues one 32 KiB request per read, which
            # dominates transfer time on anything larger than a few megabytes.
            self._handle.prefetch()
        return self._handle

    def info(self) -> SourceInfo:
        self._open()
        assert self._sftp is not None
        return SourceInfo(size_bytes=self._sftp.stat(self._path).st_size)

    def batches(self) -> Iterator[Any]:
        logger.info(f"SftpReader: {self._host}:{self._path}")
        yield from frames_from_file(
            self._open(), self._path, self._config.fetch_size
        )

    def close(self) -> None:
        for handle in (self._handle, self._sftp, self._client):
            if handle is not None:
                handle.close()
        self._handle = self._sftp = self._client = None
