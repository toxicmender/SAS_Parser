"""Seekable read-only file facade over ranged object-store reads."""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

DEFAULT_BLOCK_SIZE = 4 * 1024 * 1024


class RangedRawIO(io.RawIOBase):
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
        self._position = 0
        self._buffer = b""
        self._buffer_start = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    @property
    def size(self) -> int:
        if self._size is None:
            if self._size_fn is None:
                raise ValueError("neither size nor size_fn was given")
            self._size = int(self._size_fn())
        return self._size

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self._position + offset
        elif whence == io.SEEK_END:
            target = self.size + offset
        else:
            raise ValueError(f"invalid whence {whence!r}")
        if target < 0:
            raise OSError(22, "negative seek position")
        self._position = target
        return target

    def readinto(self, buffer: Any) -> int:
        wanted = len(buffer)
        if wanted == 0:
            return 0
        count = min(wanted, max(0, self.size - self._position))
        if count == 0:
            return 0
        chunk = self._read_bytes(self._position, count)
        buffer[: len(chunk)] = chunk
        self._position += len(chunk)
        return len(chunk)

    def _read_bytes(self, offset: int, count: int) -> bytes:
        if self._buffered(offset, count):
            start = offset - self._buffer_start
            return self._buffer[start : start + count]
        if count >= self.block_size:
            self._drop_buffer()
            return self._fetch(offset, count)
        self._buffer = self._fetch(offset, min(self.block_size, self.size - offset))
        self._buffer_start = offset
        return self._buffer[:count]

    def _buffered(self, offset: int, count: int) -> bool:
        return (
            bool(self._buffer)
            and offset >= self._buffer_start
            and offset + count <= self._buffer_start + len(self._buffer)
        )

    def _drop_buffer(self) -> None:
        self._buffer = b""
        self._buffer_start = 0

    def close(self) -> None:
        self._drop_buffer()
        super().close()


def open_buffered(raw: RangedRawIO, *, buffer_size: int = 0) -> io.BufferedReader:
    return io.BufferedReader(raw, buffer_size or raw.block_size)


__all__ = ["DEFAULT_BLOCK_SIZE", "RangedRawIO", "open_buffered"]
