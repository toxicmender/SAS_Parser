"""
Tests for chunker/paths.py — the single definition of where a physical path
appears in SAS syntax.

Two things are being pinned. First, that each statement form is recognised with
its provenance intact: which statement named the path, and which libref or
fileref it binds. Second, and less obvious, that a quoted argument is *not*
assumed to be a directory — a FILENAME device keyword points the identical
syntax at an FTP server, a mailbox or a shell pipe, and a consumer that maps
paths to storage must be able to tell those apart before it tries.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from chunker.models import PathLocation
from chunker.paths import (
    ENGINE_LIBNAMES,
    classify_location,
    extract_engine_refs,
    extract_paths,
    normalise_path,
)


def _one(source: str):
    """The single ref *source* yields, asserting there is exactly one."""
    refs = extract_paths(source)
    assert len(refs) == 1, f"expected one ref, got {[str(r) for r in refs]}"
    return refs[0]


class TestStatementForms:
    """One case per statement the grammar covers."""

    def test_libname_reports_path_and_libref(self):
        ref = _one("libname dataetl '/sasdata3/dataetl';")
        assert ref.statement == "libname"
        assert ref.path == "/sasdata3/dataetl"
        assert ref.binds == "dataetl"
        assert ref.location is PathLocation.FILESYSTEM

    def test_libname_with_an_explicit_engine(self):
        # The engine sits between the libref and the path; the libref, not the
        # engine, is what the statement binds. The engine is recorded too:
        # `spde` is SAS's own partitioned storage, and a consumer that mistook
        # it for an ordinary directory would try to read files nothing outside
        # SAS can read.
        ref = _one("libname raw spde '/data/spde';")
        assert ref.binds == "raw"
        assert ref.path == "/data/spde"
        assert ref.engine == "spde"

    def test_libname_without_an_engine_records_none(self):
        # The pair to the test above: these two statements were indistinguishable
        # until the engine group was captured, which is the whole point of it.
        assert _one("libname raw '/data/spde';").engine is None

    def test_infile_and_file_are_distinct_statements(self):
        assert _one("infile '/data/in/cust.csv' dlm=',';").statement == "infile"
        assert _one("file '/data/out/rep.txt';").statement == "file"

    def test_infile_naming_a_fileref_yields_nothing(self):
        # An unquoted INFILE names a fileref a FILENAME already declared. The
        # path lives on that FILENAME, and reporting the fileref as a path
        # would put a name that is not a location into the inventory.
        assert extract_paths("infile rawdata;") == []

    def test_include(self):
        ref = _one("%include '/code/macros/common.sas';")
        assert ref.statement == "include"
        assert ref.path == "/code/macros/common.sas"

    def test_proc_import_datafile(self):
        ref = _one('proc import datafile="/in/a.xlsx" out=work.a; run;')
        assert ref.statement == "proc_import"
        assert ref.path == "/in/a.xlsx"

    def test_proc_export_outfile(self):
        ref = _one("proc export data=work.a outfile='/out/a.csv'; run;")
        assert ref.statement == "proc_export"
        assert ref.path == "/out/a.csv"

    def test_ods_file_and_path(self):
        refs = extract_paths("ods html file='/rep/out.html' path='/rep';")
        assert {r.path for r in refs} == {"/rep/out.html", "/rep"}
        assert {r.statement for r in refs} == {"ods"}

    def test_ods_file_does_not_match_inside_outfile(self):
        # 'outfile=' contains 'file=' but has no word boundary before it. If the
        # ODS pattern matched there the same path would be reported twice under
        # two different statements.
        refs = extract_paths("proc export outfile='/out/a.csv'; run;")
        assert [r.statement for r in refs] == ["proc_export"]

    def test_sasautos_reports_the_first_entry_of_a_list(self):
        # Documented limit: the concatenation form needs a scan the
        # head/path substitution shape cannot express. Reporting the first
        # beats reporting none, and this test says so out loud.
        ref = _one("options sasautos=('/mac/a' '/mac/b');")
        assert ref.statement == "sasautos"
        assert ref.path == "/mac/a"


class TestLocationClassification:
    """A quoted argument is not automatically a directory."""

    def test_bare_filename_is_a_filesystem_path(self):
        ref = _one("filename out '/tmp/out.txt';")
        assert ref.location is PathLocation.FILESYSTEM
        assert ref.binds == "out"
        assert ref.device is None

    @pytest.mark.parametrize("device", ["ftp", "sftp", "url", "webdav", "s3"])
    def test_remote_devices(self, device):
        ref = _one(f"filename feed {device} 'rates.dat';")
        assert ref.location is PathLocation.REMOTE
        assert ref.device == device

    def test_email_device(self):
        ref = _one("filename notify email 'ops@example.com';")
        assert ref.location is PathLocation.EMAIL
        assert ref.device == "email"

    def test_pipe_device(self):
        # The "path" is a command line. Nothing downstream should mount it.
        ref = _one("filename lister pipe 'ls -l /data';")
        assert ref.location is PathLocation.PIPE

    def test_unknown_device_is_not_mistaken_for_a_filesystem_path(self):
        # The load-bearing case: SAS gains device types, and a new one must
        # surface as "something else" rather than joining the paths a consumer
        # will try to map to storage.
        ref = _one("filename x someneweng 'whatever';")
        assert ref.location is PathLocation.DEVICE
        assert ref.device == "someneweng"

    def test_disk_device_is_still_the_filesystem(self):
        assert classify_location("disk") is PathLocation.FILESYSTEM

    def test_no_device_is_the_filesystem(self):
        assert classify_location(None) is PathLocation.FILESYSTEM

    def test_case_is_folded(self):
        assert classify_location("FTP") is PathLocation.REMOTE


class TestValues:
    def test_unresolved_macro_reference_is_flagged_not_dropped(self):
        # The value is not knowable without running SAS, so it cannot be mapped
        # as written — which is exactly why it has to be reported.
        ref = _one('libname raw "&root/in";')
        assert ref.has_macro_ref is True
        assert ref.raw == "&root/in"

    def test_raw_is_preserved_while_path_is_normalised(self):
        ref = _one(r"libname win 'D:\Data\ETL';")
        assert ref.raw == r"D:\Data\ETL"
        assert ref.path == "d:/data/etl"

    def test_normalise_path_folds_case_and_separators(self):
        assert normalise_path("  C:\\Temp\\X  ") == "c:/temp/x"

    def test_an_empty_quoted_value_yields_nothing(self):
        assert extract_paths("infile '';") == []

    def test_duplicates_collapse_to_one_ref(self):
        source = "%include '/code/a.sas';\n%include '/code/a.sas';\n"
        assert len(extract_paths(source)) == 1

    def test_records_are_hashable(self):
        # _merge_meta deduplicates these through a set, so this is a contract,
        # not an implementation detail.
        ref = _one("libname a '/x';")
        assert len({ref, ref}) == 1


class TestEngineLibnames:
    """The LIBNAME form that names a connection instead of a place.

    ``libname edwprod oracle path=... user=... pass=...`` has no quoted
    directory, so none of PATH_STATEMENTS can match it — every one of them ends
    in a quoted value. That is the gap this scan fills, and the pinned behaviour
    is that it captures the connection *as written*, macro references included,
    because resolving them is not something the chunker can or should do.
    """

    def _one_engine(self, source: str):
        refs = extract_engine_refs(source)
        assert len(refs) == 1, f"expected one ref, got {[str(r) for r in refs]}"
        return refs[0]

    def test_oracle_libname_captures_engine_libref_and_options(self):
        ref = self._one_engine(
            "libname edwprod oracle path=EDWPRO_READ_ONLY schema=FR_DM_Pro "
            'user="&username." pass="&user_pass." connection=global;'
        )
        assert ref.engine == "oracle"
        assert ref.binds == "edwprod"
        assert ref.option_map == {
            "path": "EDWPRO_READ_ONLY",
            "schema": "FR_DM_Pro",
            "user": "&username.",
            "pass": "&user_pass.",
            "connection": "global",
        }
        assert ref.has_macro_ref is True

    def test_option_values_keep_their_case_but_keys_do_not(self):
        # A schema name is data — `FR_DM_Pro` may be case-sensitive on the
        # remote system — while the option keyword is syntax.
        ref = self._one_engine("libname x oracle SCHEMA=MixedCase;")
        assert ref.option_map == {"schema": "MixedCase"}

    def test_a_statement_spanning_lines_is_one_ref(self):
        # The option tail runs to the semicolon, not the line end: a connection
        # with a dozen options is routinely wrapped.
        ref = self._one_engine(
            "libname edw teradata\n  server=tdprod\n  schema=marts\n  mode=teradata;"
        )
        assert ref.option_map["server"] == "tdprod"
        assert ref.option_map["mode"] == "teradata"

    def test_a_path_libname_is_not_an_engine_libname(self):
        # The two scans partition the LIBNAME space: one needs a quoted value,
        # the other needs a known engine. Neither statement may reach both.
        assert extract_engine_refs("libname flat '/sasdata3/dataetl';") == []
        assert extract_engine_refs("libname raw spde '/data/spde';") == []

    def test_clear_and_list_forms_yield_nothing(self):
        assert extract_engine_refs("libname old clear;") == []
        assert extract_engine_refs("libname _all_ list;") == []

    def test_an_unknown_engine_yields_nothing(self):
        # ENGINE_LIBNAMES is a gate, not a guess: an unrecognised second token
        # is far more likely to be syntax this scan does not model than a
        # database nobody has heard of.
        assert extract_engine_refs("libname x notadatabase foo=bar;") == []

    def test_every_known_engine_is_lowercase(self):
        # The scan lowercases before the membership test, so an upper-case
        # entry in the set could never match.
        assert all(e == e.lower() for e in ENGINE_LIBNAMES)

    def test_engine_matching_is_case_insensitive(self):
        assert self._one_engine("LIBNAME X ORACLE PATH=P;").engine == "oracle"

    def test_quoted_and_parenthesised_values(self):
        ref = self._one_engine(
            "libname x odbc datasrc='My DSN' connection=(a b) schema=s;"
        )
        assert ref.option_map["datasrc"] == "My DSN"
        # Brackets are kept: `connection=(a b)` means something different from
        # `connection=a b`, and nothing downstream could recover the difference.
        assert ref.option_map["connection"] == "(a b)"

    def test_duplicates_collapse_and_records_are_hashable(self):
        source = "libname a oracle path=p;\nlibname a oracle path=p;\n"
        refs = extract_engine_refs(source)
        assert len(refs) == 1
        assert len({refs[0], refs[0]}) == 1

    def test_no_macro_reference_when_there_is_none(self):
        assert self._one_engine("libname x oracle path=p user=svc;").has_macro_ref is False

    def test_empty_text(self):
        assert extract_engine_refs("") == []


class TestScanScope:
    def test_a_path_in_a_comment_is_not_a_path(self):
        # extract_paths is documented as taking the comments-blanked form. This
        # asserts the contract holds for the caller that honours it: a comment
        # arrives already blanked, so nothing inside it can match.
        from chunker.scanner import _sanitise

        source = "/* libname old '/legacy/data'; */\nlibname new '/current';"
        refs = extract_paths(_sanitise(source, blank_strings=False))
        assert [r.path for r in refs] == ["/current"]

    def test_empty_text(self):
        assert extract_paths("") == []
