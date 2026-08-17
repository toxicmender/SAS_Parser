"""Object storage as an ordinary file object.

Azure Blob and ADLS Gen2 hand out bytes through ranged HTTP GETs, not a file
handle. Everything that would consume those bytes — ``pyarrow``, ``csv``,
``pyreadstat``, a decompressor — wants a binary file object it can ``read`` and
``seek``. :class:`RangedRawIO` is the adapter: a :class:`io.RawIOBase` that turns
seeks and reads into range requests, so a blob can be handed to any of them
unchanged.

One implementation, two backends
--------------------------------
The blob and datalake SDKs differ only in how a range is requested, so that is
the only thing :class:`BlobRawIO` and :class:`AdlsRawIO` supply — a callable and
a size. Two full implementations would be two places to get read-ahead, EOF, and
negative-seek handling wrong.

sFTP needs nothing here: paramiko's ``SFTPFile`` is already a seekable file
object. Adding a wrapper for symmetry would add a layer that can only lose.

Read-ahead
----------
A caller doing ``read(4)`` in a loop would otherwise issue one HTTP request per
four bytes. Every miss fetches :attr:`RangedRawIO.block_size` bytes and serves
subsequent reads from that buffer, which turns a scan of a 100 MB blob from
millions of requests into a few hundred.

**Read-only.** Hydration never writes to a source, and a writable subclass would
have to answer questions about block staging and commit ordering that nothing
here asks.

Logger name: ``data_hydration.rawio``.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

#: Bytes fetched per range request when a read misses the buffer. 4 MiB is the
#: usual sweet spot for object storage: large enough that per-request latency
#: stops dominating, small enough that a seek-heavy reader does not pull tens of
#: megabytes it never looks at.
DEFAULT_BLOCK_SIZE = 4 * 1024 * 1024


class RangedRawIO(io.RawIOBase):
    """A seekable, read-only file over any ranged-read callable.

    Parameters
    ----------
    fetch
        ``fetch(offset, length) -> bytes``. May return fewer bytes than asked
        for at the end of the object; may not return more.
    size
        Total length in bytes. Callers that must discover it lazily should pass
        a ``size_fn`` instead.
    size_fn
        Called once, on first need, when *size* is unknown — so constructing the
        object costs no network round trip.
    block_size
        Read-ahead window. See the module docstring.

    Wrap in :class:`io.BufferedReader` for a fully buffered file object; this
    class provides the raw layer only, which is what ``RawIOBase`` means.
    """

    def __init__(
        self,
        fetch: Callable[[int, int], bytes],
        *,
        size: int | None = None,
        size_fn: Callable[[], int] | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        super().__init__()
        self._fetch = fetch
        self._size = size
        self._size_fn = size_fn
        self.block_size = max(1, block_size)
        self._pos = 0
        # The read-ahead buffer and the absolute offset it starts at.
        self._buf = b""
        self._buf_start = 0

    # -- capabilities -----------------------------------------------------

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    # -- geometry ---------------------------------------------------------

    @property
    def size(self) -> int:
        """Total length, fetched on first access when it was not supplied."""
        if self._size is None:
            if self._size_fn is None:
                raise ValueError("neither size nor size_fn was given")
            self._size = int(self._size_fn())
            logger.debug(f"RangedRawIO: resolved size {self._size} bytes")
        return self._size

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Move the cursor. Seeking past the end is allowed, as for a real file."""
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self._pos + offset
        elif whence == io.SEEK_END:
            target = self.size + offset
        else:
            raise ValueError(f"invalid whence {whence!r}")
        if target < 0:
            raise OSError(22, "negative seek position")
        self._pos = target
        return self._pos

    # -- reading ----------------------------------------------------------

    def readinto(self, buffer: Any) -> int:
        """Fill *buffer* from the current position; return the byte count.

        ``0`` means end of object. ``RawIOBase`` builds ``read``, ``readall``
        and the ``BufferedReader`` protocol on top of this one method.
        """
        wanted = len(buffer)
        if wanted == 0:
            return 0
        remaining = self.size - self._pos
        if remaining <= 0:
            return 0
        count = min(wanted, remaining)
        chunk = self._read_bytes(self._pos, count)
        buffer[: len(chunk)] = chunk
        self._pos += len(chunk)
        return len(chunk)

    def _read_bytes(self, offset: int, count: int) -> bytes:
        """*count* bytes at *offset*, from the buffer when it holds them."""
        if self._buffered(offset, count):
            start = offset - self._buf_start
            return self._buf[start : start + count]
        # A read larger than the window is passed straight through: buffering it
        # would mean copying it twice to no purpose.
        if count >= self.block_size:
            self._drop_buffer()
            return self._fetch(offset, count)
        length = min(self.block_size, self.size - offset)
        self._buf = self._fetch(offset, length)
        self._buf_start = offset
        return self._buf[:count]

    def _buffered(self, offset: int, count: int) -> bool:
        """True when the buffer already covers ``[offset, offset + count)``."""
        return (
            bool(self._buf)
            and offset >= self._buf_start
            and offset + count <= self._buf_start + len(self._buf)
        )

    def _drop_buffer(self) -> None:
        self._buf = b""
        self._buf_start = 0

    def close(self) -> None:
        self._drop_buffer()
        super().close()


def _blob_client(account: str, container: str, name: str, credential: Any) -> Any:
    """A ``BlobClient`` for one blob. Imported lazily — the SDK is an extra."""
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(
            "azure-storage-blob is required to read Azure Blob storage; "
            "install it with 'pip install \"sas-parser[adls]\"'"
        ) from exc
    service = BlobServiceClient(
        account_url=f"https://{account}.blob.core.windows.net",
        credential=credential,
    )
    return service.get_blob_client(container=container, blob=name)


def BlobRawIO(  # noqa: N802 - a factory named for the object it produces
    account: str,
    container: str,
    name: str,
    *,
    credential: Any = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> RangedRawIO:
    """A :class:`RangedRawIO` over one Azure Blob.

    *credential* defaults to :class:`data_hydration.secrets.EntraCredential`, so
    the blob is reached with the same Entra ID identity the rest of this
    deployment authenticates with rather than a storage key.
    """
    if credential is None:
        from .secrets import EntraCredential

        credential = EntraCredential()
    client = _blob_client(account, container, name, credential)

    def fetch(offset: int, length: int) -> bytes:
        return client.download_blob(offset=offset, length=length).readall()

    def size_fn() -> int:
        return int(client.get_blob_properties().size)

    logger.info(f"BlobRawIO: {account}/{container}/{name}")
    return RangedRawIO(fetch, size_fn=size_fn, block_size=block_size)


def AdlsRawIO(  # noqa: N802 - a factory named for the object it produces
    account: str,
    filesystem: str,
    path: str,
    *,
    credential: Any = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> RangedRawIO:
    """A :class:`RangedRawIO` over one ADLS Gen2 file.

    The Gen2 endpoint (``dfs.core.windows.net``) rather than the blob one,
    because only it understands the hierarchical namespace a data lake path
    relies on.
    """
    if credential is None:
        from .secrets import EntraCredential

        credential = EntraCredential()
    try:
        from azure.storage.filedatalake import DataLakeServiceClient
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(
            "azure-storage-file-datalake is required to read ADLS Gen2; "
            "install it with 'pip install \"sas-parser[adls]\"'"
        ) from exc
    service = DataLakeServiceClient(
        account_url=f"https://{account}.dfs.core.windows.net",
        credential=credential,
    )
    client = service.get_file_system_client(filesystem).get_file_client(path)

    def fetch(offset: int, length: int) -> bytes:
        return client.download_file(offset=offset, length=length).readall()

    def size_fn() -> int:
        return int(client.get_file_properties().size)

    logger.info(f"AdlsRawIO: {account}/{filesystem}/{path}")
    return RangedRawIO(fetch, size_fn=size_fn, block_size=block_size)


def open_buffered(raw: RangedRawIO, *, buffer_size: int = 0) -> io.BufferedReader:
    """*raw* as a fully buffered binary file object.

    The form callers actually want: ``pyarrow`` and friends take a
    ``BufferedReader`` and will do small reads against it freely.
    """
    return io.BufferedReader(raw, buffer_size or raw.block_size)
