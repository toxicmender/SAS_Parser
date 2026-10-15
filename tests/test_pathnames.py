"""
Tests for chunker/pathnames.py and the path classification built on it.

A place a statement names is one known place, a wildcard SAS reads every match
of, or a place nobody can name before the job runs, and the chunk metadata keeps
the three apart: ``physical_paths``, ``wildcard_paths`` and ``unresolved_paths``.
Three things are pinned: the pathname rules themselves (SAS 9.4 Companion for
UNIX Environments), what each reference is classified as and why, and that the
hydration planner, which keeps its own copy of the rules, gives the same answers.
"""

from __future__ import annotations

import pytest

from chunker import PathLocation, SasBatch, SasPathRef, SasSemanticChunker, UnresolvedReason
from chunker.models import SasChunkMetadata
from chunker.pathnames import has_wildcard, is_absolute, is_relative, server_parts
from data_hydration import planner


def _ref(raw: str, *, statement: str = "infile", **fields) -> SasPathRef:
    fields.setdefault("location", PathLocation.FILESYSTEM)
    return SasPathRef(statement=statement, path=raw.lower(), raw=raw, **fields)


# ── the pathname rules ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/data/in/wild*.dat", True),
        ("/data/in/file?.csv", True),
        ("/data/in/[a-zA-Z]*.dat", True),
        ("/data/in/report[1].csv", True),
        ("*/*", True),
        ("/data/in/cust.csv", False),
        ("/data/in/[unclosed.csv", False),
        ("", False),
    ],
)
def test_a_wildcard_is_a_star_a_question_mark_or_a_set(path, expected):
    assert has_wildcard(path) is expected


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/data/x", True),
        ("\\\\host\\share\\x", True),
        ("D:\\data\\x", True),
        ("D:/data/x", True),
        ("data/x", False),
        ("x.csv", False),
        ("D:data", False),
        ("~/x", False),
        ("", False),
    ],
)
def test_a_full_path_starts_at_the_root_a_drive_or_a_share(path, expected):
    assert is_absolute(path) is expected


HOME = "~ is the home directory of the user SAS runs as"


@pytest.mark.parametrize(
    "path, parts",
    [
        ("~/data/x.csv", [HOME]),
        ("~smith/data", ["~smith is smith's home directory"]),
        ("!SASROOT/misc/x.sas", ["!SASROOT is the directory SAS is installed in"]),
        ("$PROJ/in/x.csv", ["$PROJ is an environment variable of the SAS server"]),
        ("/data/$PROJ/x", ["$PROJ is an environment variable of the SAS server"]),
        ("~/$SUB/x", [HOME, "$SUB is an environment variable of the SAS server"]),
        ("/data/x", []),
        # An administrative share's `$` names no variable.
        ("\\\\host\\c$\\x", []),
    ],
)
def test_the_server_completes_a_home_sasroot_and_environment_variables(path, parts):
    assert server_parts(path) == parts


@pytest.mark.parametrize(
    "path, expected",
    [
        ("in/x.csv", True),
        ("x.csv", True),
        ("../x", True),
        ("./x", True),
        # Taken as written: whether `&` is a macro reference is the statement's
        # to say (between single quotes it is a character).
        ("&root/x.sas", True),
        ("/data/x", False),
        ("~/x", False),
        ("$HOME/x", False),
        ("", False),
    ],
)
def test_a_relative_path_is_read_from_the_current_directory(path, expected):
    assert is_relative(path) is expected


# ── the planner's copy ───────────────────────────────────────────────────────

PATHS = [
    "/data/in/wild*.dat",
    "/data/in/file?.csv",
    "/data/in/[a-zA-Z]*.dat",
    "/data/in/[unclosed.csv",
    "/data/in/cust.csv",
    "\\\\host\\share\\x",
    "\\\\host\\c$\\x",
    "D:\\data\\x",
    "D:data",
    "~/data/x.csv",
    "~smith/data",
    "!SASROOT/misc/x.sas",
    "$PROJ/in/x.csv",
    "/data/$PROJ/x",
    "in/x.csv",
    "../x",
    "&root/x.sas",
    "&root/*.csv",
    "",
]


@pytest.mark.parametrize("path", PATHS)
def test_the_planner_reads_a_path_as_the_chunker_does(path):
    # data_hydration never imports the chunker, so it keeps its own copy of
    # the rules; this holds the two to one answer.
    assert bool(planner._WILDCARD_RE.search(path)) is has_wildcard(path)
    assert planner._absolute(path) is is_absolute(path)
    assert planner._server_parts(path) == server_parts(path)
    # The planner reads a path without its statement, so it cannot tell a
    # macro reference from a `&` written as a character and takes neither for
    # relative; a reference asks has_macro_ref first.
    assert planner._relative(path) is (is_relative(path) and "&" not in path)


# ── what a reference is ──────────────────────────────────────────────────────


class TestClassification:
    @pytest.mark.parametrize(
        "statement", ["filename", "infile", "include", "proc_import", "libname"]
    )
    def test_a_wildcard_a_statement_reads_is_one(self, statement):
        ref = _ref("/data/in/*.csv", statement=statement)
        assert ref.wildcard and ref.unresolved is None

    @pytest.mark.parametrize(
        "statement", ["file", "proc_export", "proc_cport", "ods", "printto"]
    )
    def test_a_statement_that_writes_takes_the_name_as_written(self, statement):
        # "Wildcards are supported for input only. You cannot use wildcards in
        # the FILE statement."
        ref = _ref("/out/report*.txt", statement=statement)
        assert not ref.wildcard and ref.unresolved is None

    def test_only_a_place_on_a_filesystem_is_a_wildcard(self):
        url = _ref(
            "https://host/x.json?y=1",
            statement="filename",
            location=PathLocation.REMOTE,
            device="url",
        )
        assert not url.wildcard and url.unresolved is None

    @pytest.mark.parametrize(
        "raw", ["/data/x.csv", "D:\\data\\x.csv", "\\\\host\\share\\x.csv"]
    )
    def test_a_full_path_is_known(self, raw):
        assert _ref(raw).unresolved is None

    @pytest.mark.parametrize(
        "raw, fields, reason",
        [
            ("&root/x.csv", {"has_macro_ref": True}, UnresolvedReason.MACRO),
            # A macro reference comes first: nothing else can be judged before it.
            ("~/&sub/x.csv", {"has_macro_ref": True}, UnresolvedReason.MACRO),
            (
                "&f",
                {"location": PathLocation.FILEREF, "binds": "&f", "has_macro_ref": True},
                UnresolvedReason.MACRO,
            ),
            (
                "&base/x.json",
                {"location": PathLocation.REMOTE, "device": "url", "has_macro_ref": True},
                UnresolvedReason.MACRO,
            ),
            ("in", {"location": PathLocation.FILEREF, "binds": "in"}, UnresolvedReason.FILEREF),
            ("~/x.csv", {}, UnresolvedReason.SERVER),
            ("$PROJ/x.csv", {}, UnresolvedReason.SERVER),
            ("in/x.csv", {}, UnresolvedReason.RELATIVE),
            # Single-quoted, `&` is a character: a relative path as written.
            ("&root/x.sas", {}, UnresolvedReason.RELATIVE),
        ],
    )
    def test_why_a_place_is_not_known(self, raw, fields, reason):
        assert _ref(raw, **fields).unresolved is reason

    def test_a_fileref_a_filename_assigns_to_a_device_is_assigned(self):
        ref = _ref(
            "t", statement="include", location=PathLocation.FILEREF, binds="t", device="temp"
        )
        assert ref.unresolved is None

    def test_the_place_sas_reads_is_what_is_classified(self):
        # A macro variable a %LET resolved: a full path.
        assert _ref("&root/x.csv", resolved_path="/SAS/x.csv").unresolved is None
        # A member of a directory fileref, followed to a wildcard file name
        # (`infile curdir('*.dat');`).
        member = _ref("curdir('*.dat')", binds="curdir", resolved_path="/data/*.dat")
        assert member.wildcard

    def test_the_classification_is_serialised_and_ignored_on_load(self):
        ref = _ref("~/x.csv")
        dumped = ref.model_dump(mode="json")
        assert (dumped["wildcard"], dumped["unresolved"]) == (False, "server")
        assert SasPathRef.model_validate(dumped) == ref

    def test_a_reference_says_what_it_is(self):
        assert str(_ref("/data/*.csv")) == "infile [filesystem] /data/*.csv (wildcard)"
        assert str(_ref("in/x.csv")) == "infile [filesystem] in/x.csv (unresolved: relative)"


# ── the metadata's views ─────────────────────────────────────────────────────

SOURCE = (
    "libname raw '/data/raw';\n"
    "filename wild '/data/in/wild*.dat';\n"
    "filename tmp temp;\n"
    "filename feed url 'https://host/feed.json';\n"
    "filename note email 'ops@example.com';\n"
    'filename out url "&base/x.json";\n'
    "data a; infile wild; input x; run;\n"
    "data b; infile 'rel/in.csv'; input x; run;\n"
    'data c; infile "&root/c.csv"; input x; run;\n'
    "data d; infile '~/d.csv'; input x; run;\n"
    "data e; infile nowhere; input x; run;\n"
    "%include '/code/macros/*.sas';\n"
    "%include tmp;\n"
    "data _null_; file '/out/report*.txt'; put 'x'; run;\n"
)


def _chunks(src: str = SOURCE):
    chunker = SasSemanticChunker(min_words=1, max_words=9_999)
    return chunker.chunk_text(src, source_id="t.sas").chunks


def _metas(src: str = SOURCE) -> list[SasChunkMetadata]:
    return [c.metadata for c in _chunks(src)]


def _listed(view: str, src: str = SOURCE) -> list[str]:
    return [f"{r.statement} {r.raw}" for m in _metas(src) for r in getattr(m, view)]


class TestMetadataViews:
    def test_each_view_holds_its_own(self):
        assert _listed("physical_paths") == ["libname /data/raw", "file /out/report*.txt"]
        assert _listed("wildcard_paths") == [
            "filename /data/in/wild*.dat",
            "infile wild",
            "include /code/macros/*.sas",
        ]
        assert [
            (r.statement, r.raw, r.unresolved) for m in _metas() for r in m.unresolved_paths
        ] == [
            ("filename", "&base/x.json", UnresolvedReason.MACRO),
            ("infile", "rel/in.csv", UnresolvedReason.RELATIVE),
            ("infile", "&root/c.csv", UnresolvedReason.MACRO),
            ("infile", "~/d.csv", UnresolvedReason.SERVER),
            ("infile", "nowhere", UnresolvedReason.FILEREF),
        ]
        assert _listed("remote_paths") == ["filename https://host/feed.json"]
        assert _listed("email_refs") == ["filename ops@example.com"]

    def test_a_reference_is_in_one_view_at_most(self):
        unlisted = []
        for meta in _metas():
            listed = [
                *meta.physical_paths,
                *meta.wildcard_paths,
                *meta.unresolved_paths,
                *meta.remote_paths,
                *meta.email_refs,
            ]
            assert len(listed) == len(set(listed))
            unlisted += [str(r) for r in meta.external_refs if r not in listed]
        # A fileref a FILENAME assigns to a device names no place of its own.
        assert unlisted == ["include tmp [fileref] tmp"]

    def test_a_wildcard_read_through_a_fileref_is_the_filenames_wildcard(self):
        [ref] = [r for m in _metas() for r in m.wildcard_paths if r.statement == "infile"]
        assert (ref.binds, ref.effective_path) == ("wild", "/data/in/wild*.dat")

    def test_the_views_are_serialised_with_the_metadata(self):
        [meta] = [m for m in _metas() if [r.raw for r in m.unresolved_paths] == ["~/d.csv"]]
        dumped = meta.model_dump(mode="json")
        assert [r["unresolved"] for r in dumped["unresolved_paths"]] == ["server"]
        assert dumped["physical_paths"] == dumped["wildcard_paths"] == []
        assert SasChunkMetadata.model_validate(dumped) == meta

    def test_a_let_that_resolves_the_place_makes_it_known(self):
        src = '%let root = /SASData;\ndata c; infile "&root/c.csv"; input x; run;\n'
        assert _listed("unresolved_paths", src) == []
        assert _listed("physical_paths", src) == ["infile &root/c.csv"]

    def test_a_wildcard_include_is_listed_as_one(self):
        [included] = [f for m in _metas() for f in m.include_files if f.wildcard]
        assert (included.path, included.resolved) == ("/code/macros/*.sas", True)
        assert str(included) == "/code/macros/*.sas [wildcard]"

    def test_an_include_the_server_completes_is_not_resolved(self):
        [included] = [f for m in _metas("%include '~/setup.sas';\n") for f in m.include_files]
        assert (included.path, included.resolved, included.wildcard) == (
            "~/setup.sas",
            False,
            False,
        )

    def test_a_batch_rolls_the_views_up(self):
        batch = SasBatch(batch_id="batch-001", chunks=_chunks())
        assert [r.raw for r in batch.physical_paths] == ["/out/report*.txt", "/data/raw"]
        assert sorted(r.raw for r in batch.wildcard_paths) == [
            "/code/macros/*.sas",
            "/data/in/wild*.dat",
            "wild",
        ]
        assert sorted((r.raw, str(r.unresolved)) for r in batch.unresolved_paths) == [
            ("&base/x.json", "macro"),
            ("&root/c.csv", "macro"),
            ("nowhere", "fileref"),
            ("rel/in.csv", "relative"),
            ("~/d.csv", "server"),
        ]
        assert [r.raw for r in batch.remote_paths] == ["https://host/feed.json"]
