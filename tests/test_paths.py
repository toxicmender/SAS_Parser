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

    def test_infile_naming_a_fileref_is_a_fileref_reference(self):
        # An unquoted INFILE names a fileref a FILENAME declares. It is recorded
        # as a FILEREF — no location, so nothing that maps paths to storage
        # sees it — until chunker.metadata.resolve_filerefs gives it the path
        # of the FILENAME in force.
        ref = _one("infile rawdata dlm=',' firstobs=2;")
        assert (ref.statement, ref.location, ref.binds, ref.path) == (
            "infile",
            PathLocation.FILEREF,
            "rawdata",
            "rawdata",
        )
        assert _one("file out(report.txt);").raw == "out(report.txt)"
        # SAS's own filerefs belong to no FILENAME.
        assert extract_paths("infile datalines; infile cards4; file print; file log;") == []

    def test_include(self):
        ref = _one("%include '/code/macros/common.sas';")
        assert ref.statement == "include"
        assert ref.path == "/code/macros/common.sas"

    def test_include_of_several_files(self):
        refs = extract_paths("%include '/a.sas' \"/b.sas\" / source2;")
        assert [(r.statement, r.path) for r in refs] == [("include", "/a.sas"), ("include", "/b.sas")]

    def test_include_through_a_fileref(self):
        refs = extract_paths("%include src(one two.sas);\n%include setup lib2 / source2;\n")
        assert [(r.raw, r.binds, r.location) for r in refs] == [
            ("src(one)", "src", PathLocation.FILEREF),
            ("src(two.sas)", "src", PathLocation.FILEREF),
            ("setup", "setup", PathLocation.FILEREF),
            ("lib2", "lib2", PathLocation.FILEREF),
        ]

    def test_an_include_mixing_its_sources_keeps_every_one_in_order(self):
        # "You can specify any number of sources in a %INCLUDE statement, and
        # you can mix the types of included sources" (SAS 9.4 Global
        # Statements). Before, only the quoted files were kept.
        refs = extract_paths(
            "%include '/a.sas' src(b) setup \"/c.sas\";\n"
            "%include mylib(testcode1, testcode2, \"testcode3.txt\") '/d.sas' / source2;\n"
        )
        assert [(r.raw, r.location, r.binds) for r in refs] == [
            ("/a.sas", PathLocation.FILESYSTEM, None),
            ("src(b)", PathLocation.FILEREF, "src"),
            ("setup", PathLocation.FILEREF, "setup"),
            ("/c.sas", PathLocation.FILESYSTEM, None),
            ("mylib(testcode1)", PathLocation.FILEREF, "mylib"),
            ("mylib(testcode2)", PathLocation.FILEREF, "mylib"),
            ('mylib("testcode3.txt")', PathLocation.FILEREF, "mylib"),
            ("/d.sas", PathLocation.FILESYSTEM, None),
        ]

    def test_internal_lines_and_the_keyboard_name_no_file(self):
        refs = extract_paths("%include '/a.sas' 1-3 4:5 *;\n%include 7;\n")
        assert [r.raw for r in refs] == ["/a.sas"]

    def test_a_quoted_include_counts_in_call_execute_text(self):
        # Generated code runs: its quoted file is a dependency. A bare word in
        # a string names no fileref, as before.
        src = "data _null_; call execute('%include \"/gen/a.sas\";'); put '%include src(z);'; run;\n"
        assert [r.raw for r in extract_paths(src)] == ["/gen/a.sas"]

    def test_inc_is_include(self):
        refs = extract_paths("%inc '/code/a.sas';\n%inc src(b);\n%INC setup;\n")
        assert [(r.statement, r.raw) for r in refs] == [
            ("include", "/code/a.sas"),
            ("include", "src(b)"),
            ("include", "setup"),
        ]

    def test_a_fileref_is_named_only_where_a_statement_starts(self):
        # The fileref forms are bare words, so text that only reads like one
        # names nothing: a PUT string, a TITLE, a %PUT message.
        src = (
            "data _null_; put 'the file written to disk'; run;\n"
            "title 'infile errors by day';\n%put file saved;\n"
        )
        assert extract_paths(src) == []
        # THEN and ELSE (and %THEN, %ELSE) start the statement they run.
        src = "if x then infile in1; else infile in2;\n%if &y %then %include src(one);\n"
        assert {r.raw for r in extract_paths(src)} == {"in1", "in2", "src(one)"}

    def test_proc_import_datafile(self):
        ref = _one('proc import datafile="/in/a.xlsx" out=work.a; run;')
        assert ref.statement == "proc_import"
        assert ref.path == "/in/a.xlsx"

    def test_proc_cimport_infile(self):
        # The transport file PROC CIMPORT imports, by path or by fileref.
        ref = _one("proc cimport library=x infile='/t/all.cpt'; run;")
        assert (ref.statement, ref.path) == ("proc_cimport", "/t/all.cpt")
        ref = _one("proc cimport data=y infile=tf; run;")
        assert (ref.statement, ref.location, ref.binds) == (
            "proc_cimport",
            PathLocation.FILEREF,
            "tf",
        )

    def test_proc_python_infile(self):
        # INFILE= 'file-name' | fileref: "a source file or fileref that
        # contains Python statements to run" (SAS 9.4 Procedures Guide, PROC
        # PYTHON Statement). Before, the script a step runs was not recorded.
        ref = _one("proc python terminate infile='/scripts/Prep.py'; run;")
        assert (ref.statement, ref.location, ref.path) == (
            "proc_python",
            PathLocation.FILESYSTEM,
            "/scripts/prep.py",
        )
        ref = _one("proc python infile=py; submit;\nx = 1\nendsubmit; run;")
        assert (ref.statement, ref.location, ref.binds) == (
            "proc_python",
            PathLocation.FILEREF,
            "py",
        )
        # Another PROC's INFILE= is not PROC PYTHON's: PROC LUA's names a
        # script on its LUAPATH, not a file.
        assert extract_paths("proc lua infile='prep'; run;\n") == []

    def test_an_environment_variable_assignment(self):
        # `x setenv` (quoted or not) and `options set=` define a variable that
        # UNIX SAS reads as a libref or fileref: upper-case, a full path.
        for source, binds, path in (
            ("x setenv EDUC /users/mydir/educ;", "educ", "/users/mydir/educ"),
            ("x 'setenv INED /users/myid/educ.dat';", "ined", "/users/myid/educ.dat"),
            ("options set=EDUC '/users/mydir/educ';", "educ", "/users/mydir/educ"),
            ("options nodate set=(EDUC '/u/e');", "educ", "/u/e"),
        ):
            ref = _one(source)
            assert (ref.statement, ref.binds, ref.path) == ("setenv", binds, path)
        # Lower case, or a relative value, is no libref or fileref.
        assert extract_paths("x setenv educ /lower;") == []
        assert extract_paths("options set=TMP 'relative';") == []

    def test_a_cd_command_names_the_current_directory(self):
        # SAS's own cd, as the only command of X, CALL SYSTEM or %SYSEXEC.
        src = (
            "x cd /a;\nx 'cd /b';\nx \"cd &dir\";\n"
            "data _null_; call system('cd /c'); run;\n%sysexec cd /d;\n"
            "x 'cd /e; ls';\nput 'x cd /f';\n"
        )
        refs = extract_paths(src)
        assert [(r.statement, r.raw, r.has_macro_ref) for r in refs] == [
            ("cd", "/a", False),
            ("cd", "/b", False),
            ("cd", "&dir", True),
            ("cd", "/c", False),
            ("cd", "/d", False),
        ]
        # In single quotes SAS reads & as a character.
        assert _one("x 'cd &dir';").has_macro_ref is False

    def test_a_device_type_after_an_infile_or_file_value(self):
        # "device-type ... must immediately follow file-specification" (SAS 9.4
        # DATA Step Statements, INFILE): a command's output, not a file. Before,
        # `infile 'gunzip -c /data/x.gz' pipe;` was a filesystem path.
        refs = extract_paths(
            "infile 'gunzip -c /data/x.gz' pipe;\ninfile 'ps -e' device=PIPE truncover;\n"
            "infile 'https://h/x.csv' url;\nfile 'lpr -Pq' pipe;\n"
            "infile '/a.csv' truncover dsd;\n"
        )
        assert [(r.statement, r.location, r.raw, r.device) for r in refs] == [
            ("infile", PathLocation.PIPE, "gunzip -c /data/x.gz", "pipe"),
            ("infile", PathLocation.PIPE, "ps -e", "pipe"),
            ("infile", PathLocation.REMOTE, "https://h/x.csv", "url"),
            ("infile", PathLocation.FILESYSTEM, "/a.csv", None),
            ("file", PathLocation.PIPE, "lpr -Pq", "pipe"),
        ]

    def test_proc_import_and_export_through_a_fileref(self):
        # DATAFILE= and OUTFILE= take "the complete path and filename or
        # fileref" (SAS 9.4 Procedures Guide). Before, the fileref form named
        # nothing.
        refs = extract_paths(
            "proc import datafile=inref out=x dbms=csv; run;\n"
            "proc export data=x outfile=outref dbms=csv; run;\n"
            "proc import datafile=sales.csv out=y; run;\n"
        )
        assert [(r.statement, r.location, r.binds) for r in refs] == [
            ("proc_import", PathLocation.FILEREF, "inref"),
            ("proc_export", PathLocation.FILEREF, "outref"),
        ]

    def test_proc_export_outfile(self):
        ref = _one("proc export data=work.a outfile='/out/a.csv'; run;")
        assert ref.statement == "proc_export"
        assert ref.path == "/out/a.csv"

    def test_proc_printto_log_and_print(self):
        refs = extract_paths("proc printto log='/logs/job.log' print=\"/out/job.lst\" new; run;")
        assert [(r.statement, r.path) for r in refs] == [
            ("printto", "/logs/job.log"),
            ("printto", "/out/job.lst"),
        ]
        # LOG= and PRINT= mean nothing outside PROC PRINTTO.
        assert extract_paths("%put log='/not/a/path';") == []

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

    def test_an_empty_quoted_value_in_a_list_names_nothing(self):
        # The rest of the list is kept, each value in its place.
        assert [r.raw for r in extract_paths("%include '' '/a.sas';")] == ["/a.sas"]
        refs = extract_paths("libname x ('' '/a');")
        assert [(r.raw, r.concat_index) for r in refs] == [("/a", 1)]

    def test_each_value_s_own_quotes_decide_its_macro_references(self):
        # SAS resolves `&` in double quotes and unquoted text, never in single
        # quotes: each value of a list by its own quote, a cd's by its command's.
        refs = extract_paths("%include '/a.sas' \"&root/b.sas\" '&lit/c.sas';")
        assert [(r.raw, r.has_macro_ref) for r in refs] == [
            ("/a.sas", False),
            ("&root/b.sas", True),
            ("&lit/c.sas", False),
        ]
        assert [r.has_macro_ref for r in extract_paths("x cd &dir;")] == [True]
        assert [r.has_macro_ref for r in extract_paths("x 'cd &dir';")] == [False]

    def test_path_and_file_outside_an_ods_statement_are_no_ods_paths(self):
        # PATH= is an Oracle connect identifier and a CAS action's table too.
        # Before, any `path=` or `file=` was an ODS output: the inventory
        # stored a connect identifier as a file the job writes.
        assert extract_paths(
            "libname o oracle path='EDWPRO' schema=s;\n"
            "proc sql; connect to oracle (path='EDWPRO' user=u); quit;\n"
            "proc cas; table.loadTable / path='x.csv' casOut={name='t'}; quit;\n"
        ) == []

    def test_every_file_an_ods_destination_writes(self):
        # BODY= names "the file that contains the primary output" (alias
        # FILE=), PATH= and GPATH= directories, CONTENTS=, FRAME= and PAGE= the
        # other files (SAS 9.4 ODS User's Guide, ODS HTML Statement). Before,
        # only FILE= and PATH= were read, and FILE= only with no blank before
        # its `=`.
        refs = extract_paths(
            "ods html path='/out' (url=none) body='b.html' contents='c.html' frame='f.html'"
            " page='p.html' gpath='/out/img';\nods pdf file = '/out/r.pdf';\n"
        )
        assert {(r.statement, r.path) for r in refs} == {
            ("ods", p)
            for p in ("/out", "b.html", "c.html", "f.html", "p.html", "/out/img", "/out/r.pdf")
        }

    def test_the_file_aliases_of_import_export_printto_and_cport(self):
        # DATAFILE= has the "Alias: FILE=", OUTFILE= the "Alias: FILE", PRINT=
        # the "Alias: FILE=, NAME=", and CPORT's FILE= names "the transport
        # file to write to" (SAS 9.4 Procedures Guide). Before, each FILE= was
        # an ODS output, so the file PROC IMPORT reads dropped out of the
        # plan, and NAME= named nothing.
        refs = extract_paths(
            "proc import file='/data/in.csv' out=a dbms=csv; run;\n"
            "proc import file=inref out=b dbms=csv; run;\n"
            "proc export data=a file='/out/a.csv' dbms=csv; run;\n"
            "proc printto file='/out/list.lst' new; run;\n"
            "proc printto name='/out/n.lst'; run;\n"
            "proc cport library=work file='/out/tran.cpt'; run;\n"
            "proc cport data=a file=tranfile; run;\n"
        )
        assert {(r.statement, str(r.location), r.raw) for r in refs} == {
            ("proc_import", "filesystem", "/data/in.csv"),
            ("proc_import", "fileref", "inref"),
            ("proc_export", "filesystem", "/out/a.csv"),
            ("printto", "filesystem", "/out/list.lst"),
            ("printto", "filesystem", "/out/n.lst"),
            ("proc_cport", "filesystem", "/out/tran.cpt"),
            ("proc_cport", "fileref", "tranfile"),
        }

    def test_sasautos_reports_the_first_entry_of_a_list(self):
        # Documented limit: the concatenation form needs a scan the
        # head/path substitution shape cannot express. Reporting the first
        # beats reporting none, and this test says so out loud.
        ref = _one("options sasautos=('/mac/a' '/mac/b');")
        assert ref.statement == "sasautos"
        assert ref.path == "/mac/a"


class TestViewLibnames:
    """A LIBNAME clause in the USING list of PROC SQL's CREATE VIEW: its
    libref is the view's alone (SAS 9.4 SQL Procedure User's Guide)."""

    SOURCE = (
        "proc sql;\ncreate view work.v as select * from a.t, b.u, c.w\n"
        "  using libname a '/a', libname b oracle path=x schema=s,\n"
        "        libname c odbc dsn=\"d;q\";\nquit;\nlibname a '/session';\n"
    )

    def test_each_clause_is_a_using_libname(self):
        assert [(r.statement, r.binds, r.path) for r in extract_paths(self.SOURCE)] == [
            ("using_libname", "a", "/a"),
            ("libname", "a", "/session"),
        ]

    def test_a_database_clause_ends_at_the_comma_before_the_next(self):
        refs = extract_engine_refs(self.SOURCE)
        assert [(r.statement, r.binds, r.engine, r.options) for r in refs] == [
            ("using_libname", "b", "oracle", (("path", "x"), ("schema", "s"))),
            ("using_libname", "c", "odbc", (("dsn", "d;q"),)),
        ]
        assert str(refs[0]).startswith("using libname b oracle")

    def test_a_view_assigns_no_session_libref(self):
        from chunker import SasSemanticChunker

        chunks = SasSemanticChunker(min_words=1, max_words=9_999).chunk_text(self.SOURCE).chunks
        assert [c.metadata.defines_librefs for c in chunks] == [[], ["a"]]


class TestConcatenation:
    """One statement naming several places SAS treats as one: each place is a
    reference of its own, with its position in the list (SAS 9.4 Global
    Statements, LIBNAME Statement, Form 4; SAS 9.4 Companion for UNIX
    Environments, Concatenating Filenames)."""

    def test_a_library_of_directories(self):
        refs = extract_paths("libname x ('/a' \"/b\");")
        assert [(r.statement, r.binds, r.location, r.path, r.concat_index) for r in refs] == [
            ("libname", "x", PathLocation.FILESYSTEM, "/a", 0),
            ("libname", "x", PathLocation.FILESYSTEM, "/b", 1),
        ]

    def test_librefs_directories_and_commas(self):
        # The UNIX companion's own example.
        refs = extract_paths("libname profits (income, costs, '/u/2011/capgain');")
        assert [(r.location, r.raw, r.binds, r.concat_index) for r in refs] == [
            (PathLocation.LIBREF, "income", "profits", 0),
            (PathLocation.LIBREF, "costs", "profits", 1),
            (PathLocation.FILESYSTEM, "/u/2011/capgain", "profits", 2),
        ]

    def test_an_engine_applies_to_the_directories_only(self):
        refs = extract_paths("libname x v9('/a' V6);")
        assert [(r.raw, r.path, r.engine) for r in refs] == [("/a", "/a", "v9"), ("V6", "v6", None)]

    def test_a_fileref_of_several_files_or_directories(self):
        refs = extract_paths("filename f ('/q1.sas', \"/q2.sas\");\nfilename g ftp ('a.txt' 'b.txt');\n")
        assert [(r.binds, r.location, r.path, r.concat_index) for r in refs] == [
            ("f", PathLocation.FILESYSTEM, "/q1.sas", 0),
            ("f", PathLocation.FILESYSTEM, "/q2.sas", 1),
            ("g", PathLocation.REMOTE, "a.txt", 0),
            ("g", PathLocation.REMOTE, "b.txt", 1),
        ]

    def test_files_an_infile_lists_in_one_string(self):
        refs = extract_paths("infile '(\"/d/a.csv\" \"/d/b.csv\")' dlm=',';")
        assert [(r.statement, r.path, r.concat_index) for r in refs] == [
            ("infile", "/d/a.csv", 0),
            ("infile", "/d/b.csv", 1),
        ]

    def test_the_quotes_around_a_list_decide_its_macro_references(self):
        # SAS reads a list inside one string by that string's quotes: between
        # single quotes & is a character, between double quotes a reference.
        single = extract_paths("%include '(\"&r/a.sas\")';\ninfile '(\"&r/a.csv\")';\n")
        assert [r.has_macro_ref for r in single] == [False, False]
        double = extract_paths("%include \"('&r/a.sas')\";\ninfile \"('&r/a.csv')\";\n")
        assert [r.has_macro_ref for r in double] == [True, True]

    def test_a_place_named_alone_has_no_index(self):
        assert _one("libname x '/a';").concat_index is None
        assert {r.concat_index for r in extract_paths("%include '/a.sas' '/b.sas';")} == {None}

    def test_a_list_without_its_closing_parenthesis_fails_at_once(self):
        # Possessive: a long word is not tried split every way.
        import time

        started = time.perf_counter()
        assert extract_paths("libname x (" + "a" * 60 + ";\nfilename f ('/a' " + "'/b' " * 3000) == []
        assert time.perf_counter() - started < 1


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

    def test_a_single_quoted_value_holds_no_macro_reference(self):
        # SAS resolves macro references between double quotes only: between
        # single quotes & is a character of the name.
        ref = _one("libname rd '/data/R&D/in';")
        assert (ref.has_macro_ref, ref.raw) == (False, "/data/R&D/in")

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

    def test_records_sort_one_way_whatever_the_set_order(self):
        # Two spellings of one place share its path; every field decides.
        from chunker.models import SasPathRef, _path_ref_sort_key

        spelled = SasPathRef(
            statement="include", location=PathLocation.FILESYSTEM, path="/sas/x.sas",
            raw="&root/x.sas", resolved_path="/SAS/x.sas",
        )
        written = SasPathRef(
            statement="include", location=PathLocation.FILESYSTEM, path="/sas/x.sas",
            raw="/SAS/x.sas",
        )
        assert sorted([spelled, written], key=_path_ref_sort_key) == sorted(
            [written, spelled], key=_path_ref_sort_key
        )

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

    def test_an_unterminated_statement_with_many_quotes_parses_quickly(self):
        # With no ';' to end on, the option tail must not try every way of
        # splitting its quotes: 18 of them took 11 s before, exponential in N.
        import time

        text = "libname x oracle " + " ".join(f"o{i}='v'" for i in range(18))
        started = time.perf_counter()
        assert extract_engine_refs(text) == []
        assert time.perf_counter() - started < 1.0

    def test_a_quoted_value_may_hold_semicolons(self):
        # An ODBC connection string is one value, its semicolons included.
        ref = self._one_engine('libname dw odbc noprompt="DSN=dw;UID=u;" schema=s;')
        assert ref.option_map == {"noprompt": "DSN=dw;UID=u;", "schema": "s"}
        # An unmatched quote is a character, as before.
        assert extract_engine_refs("libname o oracle path=O'Neil;")[0].option_map == {
            "path": "O'Neil"
        }

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

    @pytest.mark.parametrize(
        "engine",
        [
            # Each from its SAS/ACCESS 9.4 LIBNAME page. Before, these named
            # nothing, so a read through the library gave an empty plan.
            "snow", "jdbc", "greenplm", "sapase", "sapiq", "ybrick", "vertica",
            "impala", "hawq", "aster", "mongo", "sforce",
            # The previous names SAS still accepts for SAP ASE and SAP IQ.
            "sybase", "sybaseiq",
        ],
    )
    def test_every_documented_engine_is_a_database(self, engine):
        ref = self._one_engine(f"libname d {engine} server=h user=u;")
        assert (ref.engine, ref.binds) == (engine, "d")

    def test_snowflake_is_not_an_engine_name(self):
        # Snowflake's engine is SNOW; `snowflake` appears on no SAS page.
        assert extract_engine_refs("libname d snowflake server=h;") == []

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

    def test_a_slash_star_inside_a_string_opens_no_comment(self):
        # '/data/in/*.csv' is a UNIX wildcard path. Before, the strings-kept
        # form read its /* as a comment and blanked the rest of the text, so
        # the path and everything after it went unrecorded.
        from chunker.scanner import _sanitise

        source = (
            "filename wild '/data/in/*.csv';\n/* a 'note' */\n"
            "filename sub '/data/*/*';\nlibname out '/data/out';\n"
        )
        cf = _sanitise(source, blank_strings=False)
        assert [r.path for r in extract_paths(cf)] == ["/data/out", "/data/in/*.csv", "/data/*/*"]
        assert "note" not in cf

    def test_a_chunk_keeps_what_follows_such_a_path(self):
        from chunker import SasSemanticChunker

        (chunk,) = SasSemanticChunker(min_words=1, max_words=9_999).chunk_text(
            "data a; infile '/data/in/*.csv'; input x; set raw.t; run;\n"
        ).chunks
        assert [r.path for r in chunk.metadata.external_refs] == ["/data/in/*.csv"]
        assert chunk.metadata.input_datasets == ["raw.t"]

    def test_empty_text(self):
        assert extract_paths("") == []


class TestFilerefResolution:
    """chunker.metadata.resolve_filerefs: a reference made through a fileref
    takes the place of the FILENAME in force where it stands."""

    @staticmethod
    def _refs(source: str) -> list[list[tuple[str, str, str]]]:
        from chunker import SasSemanticChunker

        result = SasSemanticChunker(min_words=1, max_words=9_999).chunk_text(source)
        return [
            [(r.statement, str(r.location), r.path) for r in c.metadata.external_refs]
            for c in result.chunks
        ]

    def test_a_member_of_a_directory_and_a_whole_file(self):
        refs = self._refs(
            "filename src '/code';\n%include src(setup util.sas);\n"
            "filename in url 'https://h/x.csv';\ndata a; infile in; input x; run;\n"
        )
        assert refs[1] == [
            ("include", "filesystem", "/code/setup.sas"),
            ("include", "filesystem", "/code/util.sas"),
        ]
        assert refs[3] == [("infile", "remote", "https://h/x.csv")]

    def test_the_latest_filename_wins_and_clear_ends_it(self):
        refs = self._refs(
            "filename in '/a.csv';\nfilename in '/b.csv';\ndata a; infile in; run;\n"
            "filename in clear;\ndata b; infile in; run;\n"
            "filename in temp;\ndata c; infile in; run;\n"
        )
        assert refs[2] == [("infile", "filesystem", "/b.csv")]
        # Cleared, `in` is a bare word again: the in.dat SAS reads without a
        # fileref. TEMP assigns one, to a scratch file and no place.
        assert refs[4] == [("infile", "fileref", "in.dat")]
        assert refs[6] == [("infile", "fileref", "in")]

    def test_a_fileref_assigned_to_a_device_keeps_it(self):
        from chunker import SasSemanticChunker

        result = SasSemanticChunker(min_words=1, max_words=9_999).chunk_text(
            "filename t temp;\ndata _null_; file t; put 'x'; run;\ndata a; infile t; input v; run;\n"
        )
        assert [
            (r.statement, str(r.location), r.raw, r.device)
            for c in result.chunks
            for r in c.metadata.external_refs
        ] == [("file", "fileref", "t", "temp"), ("infile", "fileref", "t", "temp")]
        # The corpus resolves it again, as it does each file: unchanged.
        from chunker import SasCorpus, resolve_corpus_references

        corpus = SasCorpus(file_results=[result])
        assert resolve_corpus_references(corpus).model_dump() == corpus.model_dump()

    def test_a_fileref_spelled_through_a_macro_variable(self):
        refs = self._refs("%let f = in;\nfilename &f '/data/x.csv';\ndata a; infile in; run;\n")
        assert refs[2] == [("infile", "filesystem", "/data/x.csv")]

    def test_a_concatenated_fileref_reads_each_file_and_searches_each_directory(self):
        # `%include f;` reads both files in turn; `progs(report)` is looked for
        # in /p1, then /p2 (SAS 9.4 Companion for UNIX Environments).
        refs = self._refs(
            "filename f ('/q1.sas' '/q2.sas');\n%include f;\n"
            "filename progs (\"/p1\" '/p2');\n%inc progs(report);\n"
            "data a; infile f; input x; run;\n"
        )
        assert refs[1] == [
            ("include", "filesystem", "/q1.sas"),
            ("include", "filesystem", "/q2.sas"),
        ]
        assert refs[3] == [
            ("include", "filesystem", "/p1/report.sas"),
            ("include", "filesystem", "/p2/report.sas"),
        ]
        assert refs[4] == [("infile", "filesystem", "/q1.sas"), ("infile", "filesystem", "/q2.sas")]

    def test_proc_import_and_export_follow_their_fileref(self):
        refs = self._refs(
            "filename src '/data/in/sales.csv';\nfilename dst '/data/out/r.csv';\n"
            "proc import datafile=src out=work.s dbms=csv replace; run;\n"
            "proc export data=work.s outfile=dst dbms=csv; run;\n"
        )
        assert refs[2] == [("proc_import", "filesystem", "/data/in/sales.csv")]
        assert refs[3] == [("proc_export", "filesystem", "/data/out/r.csv")]

    def test_a_member_is_the_file_sas_on_unix_opens(self):
        # Unquoted and without an extension, a member is lowercased and given
        # the statement's extension (.sas for %INCLUDE, .dat for INFILE and
        # FILE); with an extension, or quoted, it is the file as written (SAS
        # 9.4 Companion for UNIX Environments, Specifying Pathnames). Before,
        # `code(PROGRAM)` opened /code/PROGRAM.sas, `code("TEMP")`
        # /code/TEMP.sas and `prices(cars)` /users/pat/cars.
        from chunker import SasSemanticChunker

        result = SasSemanticChunker(min_words=1, max_words=9_999).chunk_text(
            "filename code '/code';\nfilename prices '/users/pat';\n"
            '%include code(PROGRAM);\n%include code("TEMP");\n%include code(Temp.sas);\n'
            "data a; infile prices(cars); input x; run;\n"
            "data _null_; file prices('Out.txt'); put 'x'; run;\n"
        )
        opened = [
            (r.effective_path, r.path)
            for c in result.chunks
            for r in c.metadata.external_refs
            if r.statement != "filename"
        ]
        assert opened == [
            ("/code/program.sas", "/code/program.sas"),
            ("/code/TEMP", "/code/temp"),
            ("/code/Temp.sas", "/code/temp.sas"),
            ("/users/pat/cars.dat", "/users/pat/cars.dat"),
            ("/users/pat/Out.txt", "/users/pat/out.txt"),
        ]
        assert [f.path for c in result.chunks for f in c.metadata.include_files] == [
            "/code/program.sas",
            "/code/TEMP",
            "/code/Temp.sas",
        ]

    def test_a_name_no_fileref_assigns_is_a_file_in_the_current_directory(self):
        # "If there is no qtr fileref already defined, you can omit the
        # quotation marks and the file extension in the %INCLUDE statement:
        # %include qtr;" — SAS reads qtr.sas from its current directory, and
        # INFILE and FILE assume .dat (SAS 9.4 Companion for UNIX
        # Environments, Specifying Pathnames). It stays FILEREF, since a
        # fileref assigned outside the corpus would win. Before, it named no
        # file at all.
        refs = self._refs(
            "%include QTR;\ndata a; infile qtr; input x; run;\n"
            "filename qtr '/q/qtr.csv';\ndata b; infile qtr; input y; run;\n"
            "%include src(util);\nproc import datafile=imp out=w dbms=csv; run;\n"
        )
        assert refs[0] == [("include", "fileref", "qtr.sas")]
        assert refs[1] == [("infile", "fileref", "qtr.dat")]
        assert refs[3] == [("infile", "filesystem", "/q/qtr.csv")]
        # A member needs a directory; PROC IMPORT names a fileref or a quoted
        # path, never a bare file.
        assert refs[4] == [("include", "fileref", "src(util)")]
        assert refs[5] == [("proc_import", "fileref", "imp")]

    def test_a_fileref_another_file_assigns_wins_over_the_current_directory(self):
        from chunker import SasSemanticChunker
        from chunker.metadata import resolve_filerefs

        chunker = SasSemanticChunker(min_words=1, max_words=9_999)
        setup = chunker.chunk_text("filename qtr '/code/qtr.sas';\n", source_id="a.sas")
        job = chunker.chunk_text("%include qtr;\n", source_id="b.sas")
        assert [f.path for f in job.chunks[0].metadata.include_files] == ["qtr.sas"]
        chunks = [*setup.chunks, *job.chunks]
        resolve_filerefs(chunks)
        assert [(f.path, f.resolved) for f in chunks[1].metadata.include_files] == [
            ("/code/qtr.sas", True)
        ]
        assert chunks[1].metadata.includes == ["/code/qtr.sas"]

    def test_proc_python_follows_its_fileref(self):
        refs = self._refs("filename py '/scripts/prep.py';\nproc python infile=py; run;\n")
        assert refs[1] == [("proc_python", "filesystem", "/scripts/prep.py")]

    def test_a_mixed_include_follows_each_fileref(self):
        refs = self._refs("filename src '/code';\n%include '/a.sas' src(b) src;\n")
        assert refs[1] == [
            ("include", "filesystem", "/a.sas"),
            ("include", "filesystem", "/code/b.sas"),
            ("include", "filesystem", "/code"),
        ]

    def test_following_a_concatenation_again_changes_nothing(self):
        from chunker import SasSemanticChunker
        from chunker.metadata import resolve_filerefs

        result = SasSemanticChunker(min_words=1, max_words=9_999).chunk_text(
            "filename f ('/q1.sas' '/q2.sas');\n%include f;\n"
        )
        chunks = list(result.chunks)
        before = [c.metadata for c in chunks]
        resolve_filerefs(chunks)
        assert [c.metadata for c in chunks] == before
        included = before[1]
        assert [(r.raw, r.concat_index) for r in included.external_refs] == [("f", 0), ("f", 1)]
        assert [f.path for f in included.include_files] == ["/q1.sas", "/q2.sas"]


    def test_a_fileref_form_has_no_value_to_rewrite(self):
        # xref.pre rewrites quoted values in place; `%include src(util);` and
        # `infile in;` name their place through a FILENAME, never here.
        from chunker.paths import PATH_STATEMENTS

        source = "%include src(util);\n%include src('b.sas');\ndata a; infile in; file out; run;\n"
        found = [
            (spec.statement, spec.value_spans(m))
            for spec in PATH_STATEMENTS
            if spec.fileref or spec.sources
            for m in spec.pattern.finditer(source)
        ]
        assert found == [
            ("include", []),
            ("include", []),
            ("infile", []),
            ("file", []),
        ]
