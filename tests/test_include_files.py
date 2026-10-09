"""The %INCLUDE files a chunk pulls in, and paths spelled through macro variables.

``SasChunkMetadata.include_files`` lists each SAS file a ``%INCLUDE`` statement
reads, at the path SAS reads: macro variables expanded and filerefs followed.
Paths of every statement (LIBNAME, FILENAME, INFILE, ...) resolve the same way,
from the values ``%LET`` and its siblings assign — kept as written, since a path
keeps its case and separators where a dataset name does not.
"""

from __future__ import annotations

from chunker import (
    MultiFileBatcher,
    PathLocation,
    SasCorpus,
    SasIncludeFile,
    SasSemanticChunker,
    resolve_corpus_references,
)
from chunker.metadata import resolve_references


def _chunks(src: str, source_id: str = "t.sas"):
    chunker = SasSemanticChunker(min_words=1, max_words=9_999)
    return chunker.chunk_text(src, source_id=source_id).chunks


def _files(src: str) -> list[SasIncludeFile]:
    return [f for c in _chunks(src) for f in c.metadata.include_files]


def _paths(src: str) -> list[str]:
    return [f.path for f in _files(src)]


# ── the field ────────────────────────────────────────────────────────────────


def test_each_included_file_is_listed_as_written():
    [setup] = _files("%include '/Code/Setup.sas';\n")
    assert setup == SasIncludeFile(path="/Code/Setup.sas", raw="/Code/Setup.sas")
    assert setup.resolved
    assert setup.location is PathLocation.FILESYSTEM


def test_a_list_gives_one_file_each_in_order():
    assert _paths("%include '/a.sas' '/B.sas' / source2;\n") == ["/a.sas", "/B.sas"]


def test_a_fileref_member_is_the_file_in_its_directory():
    [util] = _files("filename src '/Code/Lib';\n%include src(util);\n")
    assert (util.path, util.raw, util.fileref) == ("/Code/Lib/util.sas", "src(util)", "src")
    assert util.resolved


def test_a_fileref_no_filename_binds_is_unresolved():
    [util] = _files("%include src(util);\n")
    assert (util.path, util.fileref, util.location) == (
        "src(util)",
        "src",
        PathLocation.FILEREF,
    )
    assert not util.resolved


def test_only_include_statements_are_listed():
    chunks = _chunks("libname out '/data';\n%include '/x.sas';\n")
    assert [f.path for c in chunks for f in c.metadata.include_files] == ["/x.sas"]


def test_the_field_is_serialised_with_the_metadata():
    [chunk] = _chunks("%include '/x.sas';\n")
    assert chunk.metadata.model_dump(mode="json")["include_files"] == [
        {
            "path": "/x.sas",
            "raw": "/x.sas",
            "fileref": None,
            "location": "filesystem",
            "resolved": True,
        }
    ]


# ── macro variables in paths ─────────────────────────────────────────────────


def test_a_let_value_is_pasted_in_as_written():
    src = '%let root = /SAS/Prod;\n%include "&root/setup.sas";\n'
    [setup] = _files(src)
    assert (setup.path, setup.raw, setup.resolved) == (
        "/SAS/Prod/setup.sas",
        "&root/setup.sas",
        True,
    )
    include = _chunks(src)[-1].metadata
    # The comparison key stays normalised; the reference records both.
    assert include.includes == ["/sas/prod/setup.sas"]
    [ref] = include.external_refs
    assert (ref.raw, ref.resolved_path, ref.has_macro_ref) == (
        "&root/setup.sas",
        "/SAS/Prod/setup.sas",
        False,
    )


def test_values_built_from_other_values_resolve():
    src = '%let root = /SAS;\n%let code = &root/code;\n%include "&code/setup.sas";\n'
    assert _paths(src) == ["/SAS/code/setup.sas"]


def test_delimiter_dots_are_read_as_sas_reads_them():
    src = '%let root = /SAS;\n%let mac = Utils;\n%include "&root./lib/&mac..sas";\n'
    assert _paths(src) == ["/SAS/lib/Utils.sas"]


def test_windows_separators_are_kept():
    src = '%let root = C:\\Projects\\Sales;\n%include "&root.\\macros\\load.sas";\n'
    assert _paths(src) == ["C:\\Projects\\Sales\\macros\\load.sas"]
    assert _chunks(src)[-1].metadata.includes == ["c:/projects/sales/macros/load.sas"]


def test_each_include_reads_the_value_in_force_where_it_stands():
    src = (
        "%let root = /A;\n%include \"&root/x.sas\";\n"
        "%let root = /B;\n%include \"&root/y.sas\";\n"
    )
    assert _paths(src) == ["/A/x.sas", "/B/y.sas"]


def test_a_filename_spelled_through_a_variable_leads_to_its_members():
    src = '%let root = /SAS/Prod;\nfilename src "&root/code";\n%include src(util);\n'
    [util] = _files(src)
    assert (util.path, util.fileref, util.resolved) == (
        "/SAS/Prod/code/util.sas",
        "src",
        True,
    )


def test_a_windows_directory_joins_its_member_with_a_backslash():
    src = "filename src 'C:\\Code';\n%include src(util);\n"
    assert _paths(src) == ["C:\\Code\\util.sas"]


def test_a_variable_holding_a_quoted_path_is_the_file_itself():
    src = "%let f = '/sas/x.sas';\n%include &f;\n"
    [x] = _files(src)
    assert (x.path, x.raw, x.fileref, x.location, x.resolved) == (
        "/sas/x.sas",
        "&f",
        None,
        PathLocation.FILESYSTEM,
        True,
    )


def test_a_variable_holding_a_fileref_member_is_followed():
    src = "filename src '/Code/Dir';\n%let f = src(util);\n%include &f;\n"
    [util] = _files(src)
    assert (util.path, util.raw, util.fileref) == ("/Code/Dir/util.sas", "&f", "src")


def test_a_value_known_only_at_run_time_stays_unresolved():
    never_assigned = '%include "&root/x.sas";\n'
    from_the_environment = '%let root = %sysget(HOME);\n%include "&root/x.sas";\n'
    from_a_column = (
        "data _null_; set cfg; call symputx('root', path); run;\n"
        '%include "&root/x.sas";\n'
    )
    for src in (never_assigned, from_the_environment, from_a_column):
        [x] = _files(src)
        assert (x.path, x.resolved) == ("&root/x.sas", False), src


def test_a_single_quoted_path_is_the_file_named():
    # Between single quotes SAS reads & as a character, so no %LET applies.
    src = "%let root = /A;\n%include '&root/x.sas';\n%inc \"&root/y.sas\";\n"
    assert [(f.path, f.resolved) for f in _files(src)] == [
        ("&root/x.sas", True),
        ("/A/y.sas", True),
    ]


def test_a_partly_resolved_path_shows_what_resolved():
    [x] = _files('%let root = /SAS;\n%include "&root/&sub/x.sas";\n')
    assert (x.path, x.resolved) == ("/SAS/&sub/x.sas", False)


def test_a_symputx_literal_assigns_a_path():
    src = "data _null_; call symputx('root', '/Sym/Root'); run;\n%include \"&root/x.sas\";\n"
    assert _paths(src) == ["/Sym/Root/x.sas"]


def test_a_called_macro_sets_a_global_path():
    src = (
        "%macro setp(dir); %global root; %let root = &dir; %mend;\n"
        "%setp(/Arg/Dir);\n"
        '%include "&root/x.sas";\n'
    )
    assert _paths(src) == ["/Arg/Dir/x.sas"]


def test_a_value_local_to_a_macro_body_stays_there():
    src = (
        '%macro m; %let root = /Local; %include "&root/in_body.sas"; %mend;\n'
        "%m;\n"
        '%include "&root/after.sas";\n'
    )
    files = _files(src)
    assert [(f.path, f.resolved) for f in files] == [
        ("/Local/in_body.sas", True),
        ("&root/after.sas", False),
    ]


def test_other_statements_resolve_their_paths_too():
    src = "%let root = /SASData;\nlibname dataetl \"&root/etl\";\n"
    [ref] = _chunks(src)[-1].metadata.physical_paths
    assert (ref.raw, ref.resolved_path, ref.path, ref.has_macro_ref) == (
        "&root/etl",
        "/SASData/etl",
        "/sasdata/etl",
        False,
    )


def test_dataset_names_are_unaffected_by_path_values():
    src = "%let root = /SAS/Prod;\n%let lib = mylib;\ndata &lib..x; set &root; run;\n"
    meta = _chunks(src)[-1].metadata
    assert meta.output_datasets == ["mylib.x"]
    # A path is no dataset name: the reference stays as written.
    assert meta.input_datasets == ["&root"]


# ── across files ─────────────────────────────────────────────────────────────


def _corpus(**files: str) -> SasCorpus:
    chunker = SasSemanticChunker(min_words=1, max_words=9_999)
    return SasCorpus(
        file_results=[
            chunker.chunk_text(text, source_id=f"{name}.sas") for name, text in files.items()
        ]
    )


SETUP = '%let root = /Corp;\nfilename lib "&root/lib";\n'
JOB = (
    '%include "&root/job.sas";\n'
    'filename src "&root/code";\n'
    "%include src(util);\n"
    "%include lib(m1);\n"
)


def test_a_value_from_another_file_resolves_across_the_corpus():
    alone = _corpus(job=JOB).file_results[0].chunks
    assert [f.path for c in alone for f in c.metadata.include_files] == [
        "&root/job.sas",
        "&root/code/util.sas",
        "lib(m1)",
    ]

    corpus = resolve_corpus_references(_corpus(setup=SETUP, job=JOB))
    job = [f for c in corpus.file_results[1].chunks for f in c.metadata.include_files]
    assert [(f.path, f.resolved) for f in job] == [
        ("/Corp/job.sas", True),
        # Followed again once the FILENAME's own path resolved.
        ("/Corp/code/util.sas", True),
        ("/Corp/lib/m1.sas", True),
    ]


def test_the_batcher_resolves_across_the_corpus_too():
    result = MultiFileBatcher().batch(_corpus(setup=SETUP, job=JOB))
    chunks = [*result.singletons, *(c for b in result.batches for c in b.chunks)]
    paths = {f.path for c in chunks for f in c.metadata.include_files}
    assert paths == {"/Corp/job.sas", "/Corp/code/util.sas", "/Corp/lib/m1.sas"}


def test_resolving_twice_changes_nothing():
    chunks = _chunks(
        "%let f = src(util);\nfilename src '/Code';\n%include &f;\n"
        "%let g = '/x.sas';\n%include &g;\n"
        "%let root = /R;\nfilename d \"&root\";\n%include d(a);\n"
    )
    once = [c.metadata for c in chunks]
    resolve_references(chunks)
    assert [c.metadata for c in chunks] == once
