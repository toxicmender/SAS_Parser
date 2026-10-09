"""chunker.statements: what each SAS statement does to the datasets it names.

A region is read one statement at a time, each knowing whether it stands in
open code, a DATA step or a PROC, and inside a ``%MACRO`` or not. These tests
pin that walk, the operand lists it reads, the role each statement gives a
dataset, and how a macro body's references are told apart. The coverage
probes (tests/test_sas_coverage.py) hold the language-level cases.
"""

from __future__ import annotations

import pytest

from chunker import DatasetRole, SasChunkKind, SasChunkMetadata, SasSemanticChunker
from chunker.metadata import resolve_references
from chunker.scanner import _Deadline, _line_starts, _Region, _sanitise
from chunker.statements import DATA, OPEN, PROC, statements_of

R, W, U, M = DatasetRole.READ, DatasetRole.WRITE, DatasetRole.UPDATE, DatasetRole.MENTION
D = DatasetRole.DROP
_CHUNKER = SasSemanticChunker(min_words=1, max_words=100_000)


def _statements(src: str, *, macro_body: bool = False) -> list[tuple[str, str, str, bool]]:
    """``(keyword, context, proc, in_macro)`` for each statement of *src*,
    read as one region."""
    units = _CHUNKER._scan_units(src, _line_starts(src), [], _Deadline(None))
    region = _Region(SasChunkKind.UNKNOWN_STATEMENT_GROUP, units[0].start, units[-1].end, units)
    text = region.code_text
    return [
        (st.keyword, st.context, st.proc, st.in_macro)
        for st in statements_of(
            units, _sanitise(text), _sanitise(text, blank_strings=False), macro_body=macro_body
        )
    ]


def _refs(src: str) -> list[tuple[str, DatasetRole, str]]:
    """``(name, role, via)`` of every reference the top-level chunks of *src* make."""
    return [
        (ref.name, ref.role, ref.via)
        for chunk in _CHUNKER.chunk_text(src).chunks
        if chunk.parent_id is None
        for ref in chunk.metadata.dataset_refs
    ]


def _body(src: str) -> SasChunkMetadata:
    """The metadata of the one %MACRO in *src*."""
    (chunk,) = [
        c
        for c in _CHUNKER.chunk_text(src).chunks
        if c.kind is SasChunkKind.MACRO_DEFINITION and c.parent_id is None
    ]
    return chunk.metadata


# ── the walk ─────────────────────────────────────────────────────────────────


def test_a_step_opens_at_its_header_and_ends_at_run():
    assert _statements("x = 1; data a; set b; run; y = 2;") == [
        ("x", OPEN, "", False),
        ("data", DATA, "", False),
        ("set", DATA, "", False),
        ("run", DATA, "", False),
        ("y", OPEN, "", False),
    ]


def test_a_run_group_proc_ends_only_at_quit():
    contexts = [
        (kw, ctx, proc)
        for kw, ctx, proc, _ in _statements(
            "proc datasets lib=work; modify a; run; delete b; run; quit; z = 1;"
        )
    ]
    assert contexts == [
        ("proc", PROC, "datasets"),
        ("modify", PROC, "datasets"),
        ("run", PROC, "datasets"),
        ("delete", PROC, "datasets"),
        ("run", PROC, "datasets"),
        ("quit", PROC, "datasets"),
        ("z", OPEN, ""),
    ]


def test_a_data_step_after_a_run_group_proc_ends_at_its_own_run():
    # Running in groups is the PROC's rule, and ends with it.
    src = "proc sql; drop table a; quit; data b; set c; run; x = 1;"
    assert [(kw, ctx) for kw, ctx, _, _ in _statements(src)][-4:] == [
        ("data", DATA),
        ("set", DATA),
        ("run", DATA),
        ("x", OPEN),
    ]


def test_proc_ds2_keeps_its_own_data_programs():
    src = "proc ds2; data out; method run(); set in; end; enddata; run; quit;"
    assert {(ctx, proc) for _, ctx, proc, _ in _statements(src)} == {(PROC, "ds2")}


def test_a_data_variable_named_data_opens_nothing():
    assert [ctx for _, ctx, _, _ in _statements("data = 1; proc = 2;")] == [OPEN, OPEN]


def test_the_core_follows_if_then_else_when_otherwise_and_labels():
    src = (
        "data a; if x then output hi; else output lo; select (k); when (1) output one;"
        " otherwise output two; end; next: set b; run;"
    )
    assert [kw for kw, *_ in _statements(src)] == [
        "data", "output", "output", "select", "output", "output", "end", "set", "run",
    ]


def test_a_subsetting_if_has_no_core():
    assert [kw for kw, *_ in _statements("data a; set b; if x > 1; run;")] == [
        "data", "set", "run",
    ]


def test_the_core_follows_macro_if_then_and_else():
    src = "%if &x %then %do; data a; run; %end; %else data b;"
    assert [kw for kw, *_ in _statements(src)] == ["%do", "data", "run", "%end", "data"]


def test_comments_and_in_stream_data_are_no_statements():
    src = "data a;\n  * set lib.old;\n  input x;\ndatalines;\nset lib.z\n;\nrun;\n"
    # The line ending the data is a statement of its own: a null one.
    assert [kw for kw, *_ in _statements(src)] == ["data", "input", "datalines", "", "run"]


def test_a_macro_body_is_tracked_through_nesting():
    src = "%macro m; %macro n; %mend; data a; run; %mend; data b; run;"
    assert [(kw, inside) for kw, _, _, inside in _statements(src)] == [
        ("%macro", True),
        ("%macro", True),
        ("%mend", True),
        ("data", True),
        ("run", True),
        ("%mend", False),
        ("data", False),
        ("run", False),
    ]


def test_a_slice_of_a_macro_body_is_in_it_throughout():
    assert {inside for *_, inside in _statements("set a; %mend;", macro_body=True)} == {True}


# ── DATA step statements ─────────────────────────────────────────────────────


def test_data_statement_names_its_outputs_not_its_options():
    assert _refs("data a(keep=x) lib.b / view=a; set c; run;") == [
        ("work.a", W, "data"),
        ("lib.b", W, "data"),
        ("work.c", R, "set"),
    ]


def test_data_null_names_nothing():
    assert _refs("data _null_; set a; run;") == [("work.a", R, "set")]


def test_set_reads_every_dataset_up_to_its_options():
    src = "data x; set a(where=(y in ('a;b' 'c'))) lib.c end=eof nobs=n; set d key=k / unique; run;"
    assert [n for n, role, _ in _refs(src) if role is R] == ["work.a", "lib.c", "work.d"]


@pytest.mark.parametrize(
    ("operands", "names"),
    [
        ("ds1-ds3", ["work.ds1", "work.ds2", "work.ds3"]),
        ("ds1 - ds3 x", ["work.ds1", "work.ds2", "work.ds3", "work.x"]),
        ("m01-m03", ["work.m01", "work.m02", "work.m03"]),
        ("lib.p8-lib.p10", ["lib.p8", "lib.p9", "lib.p10"]),
        # Too wide to be a range anyone meant: the bounds, not 5,000 names.
        ("d1-d5000", ["work.d1", "work.d5000"]),
    ],
)
def test_a_numbered_range_names_every_dataset_it_spans(operands, names):
    assert [n for n, role, _ in _refs(f"data x; set {operands}; run;") if role is R] == names


def test_a_prefix_list_is_a_pattern():
    (chunk,) = _CHUNKER.chunk_text("data x; set lib.sales_: q:; run;").chunks
    patterns = [(r.name, r.raw) for r in chunk.metadata.dataset_refs if r.pattern]
    assert patterns == [("lib.sales_:", "lib.sales_:"), ("work.q:", "q:")]


def test_quoted_paths_and_name_literals():
    src = "data 'C:\\Tmp\\Out'; set \"/data/in.sas7bdat\" 'my data'n; run;"
    assert _refs(src) == [
        ("'c:/tmp/out'", W, "data"),
        ("'/data/in.sas7bdat'", R, "set"),
        ("'my data'", R, "set"),
    ]


def test_a_macro_call_among_the_operands_names_nothing_it_can_see():
    assert [n for n, *_ in _refs("data x; set %list(lib) work.y; run;")] == ["work.x", "work.y"]


def test_update_reads_both_and_modify_rewrites_its_master():
    # MODIFY changes its master where it stands: the DATA statement names the
    # table being modified, and OUTPUT adds rows to it, so neither creates it.
    src = "data m; update m t; run; data lib.m; modify lib.m t2; run;"
    assert _refs(src) == [
        ("work.m", W, "data"),
        ("work.m", R, "update"),
        ("work.t", R, "update"),
        ("lib.m", U, "modify"),
        ("work.t2", R, "modify"),
    ]
    src = "data lib.m work.log; modify lib.m; output lib.m; output work.log; run;"
    assert _refs(src) == [
        ("work.log", W, "data"),
        ("lib.m", U, "modify"),
        ("work.log", W, "output"),
    ]


def test_output_writes_every_dataset_it_names():
    assert _refs("data a b; set s; if x then output a; else output b; run;")[-2:] == [
        ("work.a", W, "output"),
        ("work.b", W, "output"),
    ]


def test_a_hash_object_reads_its_dataset_and_its_output_method_writes_one():
    src = (
        "data _null_;\n  declare hash h(dataset: 'lib.lk(where=(x>1))');\n"
        "  h.definekey('k'); h.definedone();\n  rc = h.output(dataset: \"work.out\");\n"
        "  declare hash g(dataset: dsname);\nrun;\n"
    )
    assert _refs(src) == [("lib.lk", R, "hash"), ("work.out", W, "hash")]


def test_statements_that_only_look_like_dataset_positions():
    src = "data b; set a; out = data * 2; if data = y then z = 1; put 'set lib.q;'; run;"
    assert _refs(src) == [("work.b", W, "data"), ("work.a", R, "set")]


# ── PROC statements ──────────────────────────────────────────────────────────


def test_proc_data_reads_and_out_writes_on_any_statement():
    src = "proc means data=a noprint; var x; output out=s mean=m; run;"
    assert _refs(src) == [("work.a", R, "data="), ("work.s", W, "out=")]


def test_proc_sort_without_out_replaces_its_data():
    # A new sorted table replaces the old, as `data a; set a;` would: a read
    # and a creation, so a later BY step depends on the sort.
    assert _refs("proc sort data=lib.a; by x; run;") == [
        ("lib.a", R, "data="),
        ("lib.a", W, "data="),
    ]
    assert _refs("proc sort data=a out=b; by x; run;") == [
        ("work.a", R, "data="),
        ("work.b", W, "out="),
    ]


def test_an_assignment_in_a_proc_names_nothing():
    src = "proc phreg data=a;\n  model t*c(0) = x;\n  out = x + 1;\n  data[1] = 2;\nrun;\n"
    assert _refs(src) == [("work.a", R, "data=")]


@pytest.mark.parametrize(
    ("src", "refs"),
    [
        # Each PROC's own options (keywords.PROC_OPTION_ROLES).
        (
            "proc append base=lib.m data=new force; run;",
            [("lib.m", U, "base="), ("work.new", R, "data=")],
        ),
        (
            "proc compare base=a compare=b out=d outstats=s; run;",
            [("work.a", R, "base="), ("work.b", R, "compare="), ("work.d", W, "out="),
             ("work.s", W, "outstats=")],
        ),
        (
            "proc sort data=a out=b dupout=d; by k; run;",
            [("work.a", R, "data="), ("work.b", W, "out="), ("work.d", W, "dupout=")],
        ),
        (
            "proc format cntlin=f library=lib cntlout=g; run;",
            [("work.f", R, "cntlin="), ("work.g", W, "cntlout=")],
        ),
        ("proc corr data=a outp=p noprint; run;", [("work.a", R, "data="), ("work.p", W, "outp=")]),
        ("proc fastclus data=a seed=s mean=m; run;",
         [("work.a", R, "data="), ("work.s", R, "seed="), ("work.m", W, "mean=")]),
        # Everywhere else SEED= is a number, or a macro variable holding one.
        ("proc surveyselect data=a out=b seed=&seed; run;",
         [("work.a", R, "data="), ("work.b", W, "out=")]),
        ("proc score data=a score=c out=s; run;",
         [("work.a", R, "data="), ("work.c", R, "score="), ("work.s", W, "out=")]),
        # The statistics and modelling defaults.
        ("proc reg data=a outest=e; model y = x; output out=p p=yhat; run; quit;",
         [("work.a", R, "data="), ("work.e", W, "outest="), ("work.p", W, "out=")]),
        ("proc logistic inmodel=m; score data=n out=s; run;",
         [("work.m", R, "inmodel="), ("work.n", R, "data="), ("work.s", W, "out=")]),
        ("proc means data=a classdata=c; class g; run;",
         [("work.a", R, "data="), ("work.c", R, "classdata=")]),
        # Values that only look like datasets.
        ("proc http url='https://x' out=resp headerout=h; run;", []),
        ("proc fcmp outlib=work.funcs.pkg; function f(x); return(x); endsub; run;",
         [("work.funcs", W, "outlib=")]),
        ("proc contents data=lib._all_ out=meta; run;",
         [("lib.:", R, "data="), ("work.meta", W, "out=")]),
    ],
)
def test_a_proc_option_names_what_its_proc_says(src, refs):
    assert _refs(src) == refs


def test_proc_options_come_from_the_statements_that_carry_them():
    src = (
        "proc freq data=a;\n  tables g / out=f;\n  label out = 'Output';\n"
        "  where data = x;\n  weight w / data=nope;\nrun;\n"
    )
    # Before a slash: variables and labels. After it, an option the PROC's
    # roles know is read whatever the statement — roles are per PROC.
    assert _refs(src) == [("work.a", R, "data="), ("work.f", W, "out="), ("work.nope", R, "data=")]
    # A dataset's own options are not the statement's.
    assert _refs("proc print data=a(rename=(data=d2 out=o2)); run;") == [("work.a", R, "data=")]


def test_proc_datasets_names_members_of_its_library():
    src = (
        "proc datasets lib=lib nolist;\n  append base=all data=part;\n  delete t1 t2;\n"
        "  change old=new;\n  exchange x=y;\n  modify m;\n    rename a=b;\n  age a1 a2;\n"
        "  contents data=c out=meta;\nquit;\n"
    )
    assert _refs(src) == [
        ("lib.all", U, "base="),
        ("lib.part", R, "data="),
        ("lib.t1", D, "delete"),
        ("lib.t2", D, "delete"),
        ("lib.old", R, "change"),
        ("lib.old", D, "change"),
        ("lib.new", W, "change"),
        ("lib.x", U, "exchange"),
        ("lib.y", U, "exchange"),
        ("lib.m", U, "modify"),
        ("lib.a1", U, "age"),
        ("lib.a2", U, "age"),
        ("lib.c", R, "data="),
        ("work.meta", W, "out="),
    ]


def test_proc_datasets_kill_deletes_every_member():
    assert _refs("proc datasets lib=scratch kill nolist; quit;") == [("scratch.:", D, "kill")]
    assert _refs("proc datasets lib=kill nolist; delete a; quit;") == [("kill.a", D, "delete")]


def test_run_cancel_runs_nothing():
    assert _refs("data a; set b; run cancel;") == []
    assert _refs("proc print data=a; run cancel;") == []
    # A PROC that runs in groups loses the cancelled group only.
    src = "proc datasets lib=work nolist; delete t1; run; delete t2; run cancel; quit;"
    assert _refs(src) == [("work.t1", D, "delete")]


def test_a_macro_library_keeps_its_delimiter_dot():
    assert _refs("proc datasets lib=&lib; delete a; quit;") == [("&lib..a", D, "delete")]


@pytest.mark.parametrize(
    ("src", "refs"),
    [
        (
            "proc copy in=src out=tgt; select a b; run;",
            [("src.a", R, "copy"), ("tgt.a", W, "copy"), ("src.b", R, "copy"), ("tgt.b", W, "copy")],
        ),
        # Without SELECT every member is copied; EXCLUDE names the ones left.
        ("proc copy in=src out=tgt; run;", [("src.:", R, "copy"), ("tgt.:", W, "copy")]),
        ("proc copy in=src out=tgt; exclude z; run;", [("src.:", R, "copy"), ("tgt.:", W, "copy")]),
        (
            "proc copy in=src out=tgt move; select a; run;",
            [("src.a", R, "copy"), ("tgt.a", W, "copy"), ("src.a", D, "copy")],
        ),
        # PROC DATASETS's COPY copies from its own library by default.
        (
            "proc datasets lib=src; copy out=tgt; select a; run; delete z; quit;",
            [("src.a", R, "copy"), ("tgt.a", W, "copy"), ("src.z", D, "delete")],
        ),
        ("proc upload inlib=work outlib=rwork; select a; run;",
         [("work.a", R, "copy"), ("rwork.a", W, "copy")]),
        ("proc upload data=a out=rwork.a; run;", [("work.a", R, "data="), ("rwork.a", W, "out=")]),
    ],
)
def test_a_library_copy_names_its_members(src, refs):
    assert _refs(src) == refs


def test_ods_output_in_a_proc_is_written_by_it():
    src = "proc ttest data=a; class g; var x; ods output TTests=tt Statistics(persist=proc)=st; run;"
    assert _refs(src) == [
        ("work.a", R, "data="),
        ("work.tt", W, "ods_output"),
        ("work.st", W, "ods_output"),
    ]


def test_ods_output_in_open_code_is_written_by_the_next_proc():
    src = (
        "ods output Summary=s1;\nproc means data=a; run;\n"
        "ods output Summary=s2;\nods output close;\nproc means data=a; run;\n"
    )
    chunks = _CHUNKER.chunk_text(src).chunks
    assert [c.metadata.output_datasets for c in chunks] == [[], ["work.s1"], [], [], []]
    # A request closed before any PROC takes it is named, never written.
    assert [(r.name, r.role) for r in chunks[2].metadata.dataset_refs] == [("work.s2", M)]
    before = [c.metadata.dataset_refs for c in chunks]
    resolve_references(chunks)  # the batcher's corpus-level run
    assert [c.metadata.dataset_refs for c in chunks] == before


def test_ods_output_before_a_macro_call_is_written_by_the_proc_it_runs():
    src = (
        "%macro fit(data=);\n  proc reg data=&data; model y = x; run; quit;\n%mend;\n"
        "ods output ParameterEstimates=pe;\n%fit(data=work.train);\n"
        "proc print data=work.other; run;\n"
    )
    *_, call, proc = _CHUNKER.chunk_text(src).chunks
    assert (call.kind, call.metadata.output_datasets) == (SasChunkKind.MACRO_CALL, ["work.pe"])
    assert proc.metadata.output_datasets == []
    # A macro the corpus does not define runs nothing anyone can see.
    src = "ods output Summary=s;\n%elsewhere(x);\nproc means data=a; run;\n"
    *_, call, proc = _CHUNKER.chunk_text(src).chunks
    assert (call.metadata.output_datasets, proc.metadata.output_datasets) == ([], ["work.s"])


def test_ods_output_in_a_macro_body_is_the_bodys():
    meta = _body("%macro m(o);\n  ods output Summary=&o;\n  proc means data=a; run;\n%mend;\n")
    assert meta.body_param_outputs == [{"param": "o", "pos": 0}]


def test_proc_ds2_reads_and_writes_its_tables():
    src = (
        "proc ds2;\n"
        "  thread t / overwrite=yes; method run(); set lib.raw; end; endthread;\n"
        "  data out_ds lib.copy / overwrite=yes;\n"
        "    dcl thread t th;\n"
        "    method run(); set from th; set {select k, v from lib.lk where v > 1}; end;\n"
        "  enddata;\n"
        "run;\nquit;\n"
    )
    assert _refs(src) == [
        ("lib.raw", R, "set"),
        ("work.out_ds", W, "data"),
        ("lib.copy", W, "data"),
        ("lib.lk", R, "from"),
    ]


def test_proc_iml_opens_datasets_by_name():
    src = (
        "proc iml;\n  use lib.a var {x y} where(x > 0); read all var _num_ into m; close lib.a;\n"
        "  edit lib.b; delete all where(x < 0); purge;\n"
        "  create out var {x y}; append from m; close out;\n"
        "  use (dsname);\nquit;\n"
    )
    assert _refs(src) == [("lib.a", R, "use"), ("lib.b", U, "edit"), ("work.out", W, "create")]


def test_proc_sql_clauses():
    src = (
        "proc sql;\n  create table c as select * from a join lib.b on 1;\n"
        "  insert into d select * from e;\nquit;\n"
    )
    assert _refs(src) == [
        ("work.c", W, "create"),
        ("work.a", R, "from"),
        ("lib.b", R, "join"),
        ("work.d", U, "insert"),
        ("work.e", R, "from"),
    ]


def test_open_code_names_no_dataset():
    src = "%put Loading data from staging;\n%let msg = copy from src;\noptions obs=10;\n"
    assert _refs(src) == []


# ── macro bodies ─────────────────────────────────────────────────────────────


def test_a_body_reference_is_a_parameter_a_literal_or_a_macro_variable():
    meta = _body(
        "%macro m(ds, out=);\n  data &out; set &ds lib.x &other &ds._&out; run;\n%mend;\n"
    )
    assert meta.body_param_outputs == [{"param": "out", "pos": -1}]
    assert meta.body_param_inputs == [{"param": "ds", "pos": 0}]
    # A macro variable the call does not supply is fixed for every call; one
    # built from several parameters names no dataset until a call does.
    assert meta.body_literal_inputs == ["lib.x", "&other"]
    assert (meta.input_datasets, meta.output_datasets) == ([], [])


def test_a_name_built_around_one_parameter_keeps_its_spelling():
    # The call site fills it in: lib=prod reads prod.customers, not prod.
    meta = _body("%macro m(lib);\n  data work.out; set &lib..customers; run;\n%mend;\n")
    assert [(r.name, r.param, r.param_pos) for r in meta.dataset_refs if r.param] == [
        ("&lib..customers", "lib", 0)
    ]
    assert meta.body_param_inputs == [{"param": "lib", "pos": 0}]


def test_a_body_step_after_a_run_group_proc_ends_at_its_run():
    # Else the DATA step would run on to %mend and swallow the call after it.
    meta = _body(
        "%macro m;\n  proc sql; create table ids as select id from lib.a; quit;\n"
        "  data ids2; set ids; run;\n  %summarize(data=ids2, out=summary);\n%mend;\n"
    )
    assert meta.body_literal_outputs == ["work.ids", "work.ids2", "work.summary"]


def test_a_quoted_path_built_from_a_parameter_names_no_dataset():
    meta = _body('%macro m(dir);\n  data x; set "&dir/in.sas7bdat" "/fixed/b"; run;\n%mend;\n')
    assert (meta.body_param_inputs, meta.body_literal_inputs) == ([], ["'/fixed/b'"])


@pytest.mark.parametrize(
    ("body", "reads", "writes"),
    [
        # Part of a DATA step, for the caller's to complete.
        ("set &t lib.b;", ["t"], []),
        ("if first.id then output &t;", [], ["t"]),
        # Part of a PROC SQL query.
        ("select * from &t where x = 1;", ["t"], []),
        ("create table &t as select * from lib.src;", [], ["t"]),
        # Part of a PROC: OUTPUT with options is a PROC's.
        ("output out=&t mean=m;", [], ["t"]),
        # A call of another macro: its DATA= and OUT= arguments.
        ("%inner(data=&t, out=lib.res);", ["t"], []),
        # Text, not a dataset position.
        ("%put Reading from &t;", [], []),
        ("x = data;", [], []),
    ],
)
def test_a_body_statement_outside_any_step_is_read_by_its_keyword(body, reads, writes):
    meta = _body(f"%macro m(t);\n  {body}\n%mend;\n")
    assert [e["param"] for e in meta.body_param_inputs] == reads
    assert [e["param"] for e in meta.body_param_outputs] == writes


def test_a_wrapper_macro_names_what_its_calls_read_and_write():
    meta = _body("%macro run_all;\n  %step1(data=lib.a, out=b);\n%mend;\n")
    assert (meta.body_literal_inputs, meta.body_literal_outputs) == (["lib.a"], ["work.b"])


# ── %LET values and the reference views ──────────────────────────────────────


def test_a_let_value_written_like_a_dataset_is_a_mention():
    (chunk,) = _CHUNKER.chunk_text("%let t = edw.accounts;\n").chunks
    meta = chunk.metadata
    assert [(r.name, r.role, r.via) for r in meta.dataset_refs] == [("edw.accounts", M, "%let")]
    assert (meta.input_datasets, meta.output_datasets) == ([], [])
    assert meta.referenced_datasets == ["edw.accounts"]
    assert meta.referenced_librefs == ["edw"]


def test_let_mentions_are_rederived_not_accumulated():
    chunks = _CHUNKER.chunk_text("%let s = prod;\n%let t = &s..orders;\n").chunks
    before = [c.metadata.dataset_refs for c in chunks]
    resolve_references(chunks)  # a second run, as the batcher's corpus pass makes
    assert [c.metadata.dataset_refs for c in chunks] == before
    assert chunks[1].metadata.referenced_datasets == ["prod.orders"]


def test_referenced_views_cover_every_name_and_assigned_libref():
    src = "libname out '/x';\ndata a; set lib.b; run;\n%macro m(ds); set &ds src.c; %mend;\n"
    refs, librefs = set(), set()
    for chunk in _CHUNKER.chunk_text(src).chunks:
        refs.update(chunk.metadata.referenced_datasets)
        librefs.update(chunk.metadata.referenced_librefs)
    assert refs == {"work.a", "lib.b", "&ds", "src.c"}
    assert librefs == {"out", "work", "lib", "src"}


def test_old_metadata_with_referenced_datasets_keeps_them_as_mentions():
    meta = SasChunkMetadata.model_validate(
        {"input_datasets": ["work.a"], "referenced_datasets": ["a", "work.a", "lib.x"]}
    )
    assert meta.input_datasets == ["work.a"]
    assert meta.referenced_datasets == ["a", "lib.x", "work.a"]
    assert {r.name for r in meta.dataset_refs if r.role is M} == {"a", "lib.x"}
