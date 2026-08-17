"""
Tests for data_hydration/rawio.py — object storage as a seekable file object.

``RangedRawIO`` exists so a blob can be handed to pyarrow, csv or pyreadstat
unchanged. That only works if it behaves like a real file in the awkward cases,
so those are what is pinned here: reads that straddle the buffer, seeks past the
end, ``SEEK_END`` with a negative offset, and a final read that returns short.

The second concern is the one that makes it usable rather than merely correct:
**request coalescing**. A caller doing ``read(4)`` in a loop must not produce one
HTTP range request per four bytes. Every test counts the fetches, because the
difference between a working adapter and an unusable one is not in the bytes it
returns.
"""

from __future__ import annotations

import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from data_hydration.rawio import RangedRawIO, open_buffered

PAYLOAD = bytes(range(256)) * 40  # 10240 bytes, every value distinguishable


class _FakeObject:
    """The ranged-read surface an Azure blob client presents, in memory.

    Records every ``(offset, length)`` it is asked for, which is what the
    coalescing assertions read. Clamps at the end of the payload exactly as a
    real ranged GET does — returning short rather than padding.
    """

    def __init__(self, data: bytes = PAYLOAD) -> None:
        self.data = data
        self.calls: list[tuple[int, int]] = []

    def fetch(self, offset: int, length: int) -> bytes:
        self.calls.append((offset, length))
        return self.data[offset : offset + length]

    def size(self) -> int:
        return len(self.data)


def _raw(block_size: int = 1024, **kwargs) -> tuple[RangedRawIO, _FakeObject]:
    obj = _FakeObject(**kwargs)
    return RangedRawIO(obj.fetch, size_fn=obj.size, block_size=block_size), obj


class TestCapabilities:
    def test_it_is_a_readable_seekable_non_writable_file(self):
        raw, _ = _raw()
        assert isinstance(raw, io.RawIOBase)
        assert raw.readable() and raw.seekable()
        assert not raw.writable()

    def test_size_is_not_fetched_until_it_is_needed(self):
        # Constructing the object must not cost a network round trip; the whole
        # point of size_fn over size.
        calls: list[str] = []

        raw = RangedRawIO(
            lambda o, n: b"", size_fn=lambda: (calls.append("size"), 10)[1]
        )
        assert calls == []
        assert raw.size == 10
        assert calls == ["size"]

    def test_size_without_either_source_raises(self):
        with pytest.raises(ValueError):
            RangedRawIO(lambda o, n: b"").size


class TestReading:
    def test_readall_returns_the_whole_object(self):
        raw, obj = _raw()
        assert raw.readall() == obj.data

    def test_a_read_at_an_offset_returns_the_right_bytes(self):
        raw, obj = _raw()
        raw.seek(300)
        assert raw.read(50) == obj.data[300:350]

    def test_reading_past_the_end_returns_what_is_there(self):
        raw, obj = _raw()
        raw.seek(len(obj.data) - 10)
        assert raw.read(100) == obj.data[-10:]

    def test_reading_at_the_end_returns_empty(self):
        raw, obj = _raw()
        raw.seek(len(obj.data))
        assert raw.read(10) == b""

    def test_a_read_straddling_the_buffer_boundary_is_correct(self):
        # The case a naive buffer gets wrong: the request starts inside the
        # cached window and ends outside it.
        raw, obj = _raw(block_size=1024)
        raw.seek(1000)
        assert raw.read(100) == obj.data[1000:1100]

    def test_a_read_larger_than_the_block_size_is_correct(self):
        raw, obj = _raw(block_size=64)
        assert raw.read(5000) == obj.data[:5000]

    def test_readinto_reports_the_count_it_wrote(self):
        raw, obj = _raw()
        buffer = bytearray(32)
        assert raw.readinto(buffer) == 32
        assert bytes(buffer) == obj.data[:32]

    def test_readinto_an_empty_buffer_is_a_no_op(self):
        raw, obj = _raw()
        assert raw.readinto(bytearray(0)) == 0
        assert obj.calls == []


class TestSeeking:
    def test_seek_set_cur_and_end(self):
        raw, obj = _raw()
        assert raw.seek(100) == 100
        assert raw.seek(50, io.SEEK_CUR) == 150
        assert raw.seek(-10, io.SEEK_END) == len(obj.data) - 10
        assert raw.read() == obj.data[-10:]

    def test_tell_tracks_reads(self):
        raw, _ = _raw()
        raw.read(70)
        assert raw.tell() == 70

    def test_a_negative_seek_raises(self):
        raw, _ = _raw()
        with pytest.raises(OSError):
            raw.seek(-1)

    def test_seeking_past_the_end_is_allowed_and_reads_empty(self):
        # Real files permit this; a reader probing for EOF relies on it.
        raw, obj = _raw()
        raw.seek(len(obj.data) + 500)
        assert raw.read(10) == b""

    def test_an_invalid_whence_raises(self):
        raw, _ = _raw()
        with pytest.raises(ValueError):
            raw.seek(0, 99)


class TestCoalescing:
    def test_small_sequential_reads_do_not_issue_one_request_each(self):
        """The property that makes this usable rather than merely correct."""
        raw, obj = _raw(block_size=1024)
        for _ in range(256):  # 256 x 4 bytes = one block's worth
            raw.read(4)
        assert len(obj.calls) == 1, f"expected 1 range request, got {len(obj.calls)}"

    def test_reads_within_one_block_are_served_from_the_buffer(self):
        raw, _obj = _raw(block_size=1024)
        raw.read(10)
        before = len(_obj.calls)
        raw.seek(20)
        raw.read(10)
        raw.seek(500)
        raw.read(10)
        assert len(_obj.calls) == before

    def test_a_large_read_bypasses_the_buffer_rather_than_copying_twice(self):
        raw, obj = _raw(block_size=64)
        raw.read(4096)
        # One request for the data itself, sized to what was asked for.
        assert obj.calls == [(0, 4096)]

    def test_the_buffer_never_reads_past_the_end(self):
        raw, obj = _raw(block_size=4096)
        raw.seek(len(obj.data) - 5)
        raw.read(5)
        assert all(o + n <= len(obj.data) for o, n in obj.calls)


class TestBufferedWrapper:
    def test_open_buffered_gives_a_normal_binary_file_object(self):
        raw, obj = _raw()
        handle = open_buffered(raw)
        assert isinstance(handle, io.BufferedReader)
        assert handle.read() == obj.data

    def test_a_buffered_reader_round_trips_through_seek(self):
        raw, obj = _raw()
        handle = open_buffered(raw)
        handle.seek(1000)
        assert handle.read(20) == obj.data[1000:1020]

    def test_csv_can_be_parsed_straight_off_it(self):
        # The actual use: hand the object to a parser that knows nothing about
        # ranged reads.
        import csv

        payload = b"id,name\n1,alice\n2,bob\n"
        obj = _FakeObject(payload)
        handle = open_buffered(
            RangedRawIO(obj.fetch, size_fn=obj.size, block_size=8)
        )
        rows = list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8")))
        assert rows == [
            {"id": "1", "name": "alice"},
            {"id": "2", "name": "bob"},
        ]


class TestClosing:
    def test_close_releases_the_buffer(self):
        raw, _ = _raw()
        raw.read(10)
        raw.close()
        assert raw.closed
