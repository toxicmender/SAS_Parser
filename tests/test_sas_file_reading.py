"""
Reading SAS source files as an AIX or Linux SAS server writes them.

SAS on UNIX runs with a Latin-1 session encoding unless told otherwise
(ENCODING= ``latin1`` for ``en_US``), so its programs are Latin-1 more often
than not; and a UNIX file system is case-sensitive, so ``*.sas`` does not match
``LOAD.SAS``. These tests pin that the chunker and both CLIs read such a corpus
whole: before, one accented comment raised UnicodeDecodeError and ended the
hydration run, and an upper-case extension dropped the file unannounced.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from chunker import SasSemanticChunker, read_sas_source

#: A comment with an accent and the copyright sign, and `¬=` — SAS's *not
#: equal* — all in Latin-1, none of them valid UTF-8.
LATIN1 = (
    b"/* Caf\xe9 extract \xa9 2019 */\n"
    b"libname stg '/data/stg';\n"
    b"data b; set stg.u; if x \xac= 1; run;\n"
)
UTF8 = "libname raw '/data/raw';\ndata a; set raw.t; run;\n"


def _chunker() -> SasSemanticChunker:
    return SasSemanticChunker(min_words=1, max_words=9_999)


class TestReadingASource:
    def test_a_latin1_file_is_read_as_latin1(self, tmp_path):
        path = tmp_path / "b.sas"
        path.write_bytes(LATIN1)
        text = read_sas_source(path)
        assert "Café" in text and "¬=" in text

    def test_a_utf8_file_keeps_its_characters_and_loses_its_bom(self, tmp_path):
        path = tmp_path / "a.sas"
        path.write_bytes(b"\xef\xbb\xbf/* caf\xc3\xa9 */\n" + UTF8.encode())
        text = read_sas_source(path)
        assert text.startswith("/* café */")

    def test_an_encoding_given_is_used_strictly(self, tmp_path):
        path = tmp_path / "b.sas"
        path.write_bytes(LATIN1)
        assert "Café" in read_sas_source(path, "cp1252")
        with pytest.raises(UnicodeDecodeError):
            read_sas_source(path, "ascii")

    def test_chunk_file_reads_a_latin1_file(self, tmp_path):
        path = tmp_path / "b.sas"
        path.write_bytes(LATIN1)
        result = _chunker().chunk_file(str(path))
        reads = {r.name for c in result.chunks for r in c.metadata.dataset_refs}
        assert "stg.u" in reads

    def test_the_chunker_encoding_is_strict(self, tmp_path):
        path = tmp_path / "b.sas"
        path.write_bytes(LATIN1)
        with pytest.raises(UnicodeDecodeError):
            SasSemanticChunker(encoding="ascii").chunk_file(str(path))

