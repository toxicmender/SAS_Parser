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


@pytest.fixture
def corpus(tmp_path):
    """A UTF-8 file, a Latin-1 one, and one with an upper-case extension."""
    (tmp_path / "a.sas").write_text(UTF8, encoding="utf-8")
    (tmp_path / "b.sas").write_bytes(LATIN1)
    (tmp_path / "LOAD.SAS").write_text(
        "libname up '/data/up';\ndata c; set up.v; run;\n", encoding="utf-8"
    )
    return tmp_path


class TestTheHydrationCli:
    def _dry_run(self, corpus, capsys, *args):
        from data_hydration.__main__ import main

        status = main([str(corpus), "--dry-run", "--catalog", "main", *args])
        return status, capsys.readouterr().out

    def test_every_file_is_planned_whatever_its_encoding_or_extension_case(
        self, corpus, capsys
    ):
        status, out = self._dry_run(corpus, capsys)
        assert status == 0
        for place in ("/data/raw/t", "/data/stg/u", "/data/up/v"):
            assert place in out

    def test_a_file_that_cannot_be_read_costs_the_exit_status_not_the_plan(
        self, corpus, capsys
    ):
        # A strict encoding the Latin-1 file is not valid in: that file is
        # logged and left out, the others are still planned.
        status, out = self._dry_run(corpus, capsys, "--encoding", "utf-8")
        assert status == 1
        assert "/data/raw/t" in out and "/data/up/v" in out
        assert "/data/stg/u" not in out


class TestTheComplexityCli:
    def test_every_file_is_scored_whatever_its_encoding_or_extension_case(
        self, corpus
    ):
        from complexity.__main__ import main

        out = corpus / "report.md"
        assert main([str(corpus), "--out", str(out)]) == 0
        text = out.read_text(encoding="utf-8")
        for name in ("a.sas", "b.sas", "LOAD.SAS"):
            assert name in text
