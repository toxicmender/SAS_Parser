"""SAS language coverage of the chunker, as a table of probes.

Each probe is one realistic SAS snippet and what a correct chunker reports for
it: the chunk kinds, the datasets read and written, librefs, external paths and
macros. Probes are grouped by area (``L`` lexical, ``D`` DATA step, ``P``
PROCs, ``S`` PROC SQL, ``G`` global statements, ``M`` macro language, ``A``
SAS/ACCESS and SAS/CONNECT, ``E`` embedded languages). The ``X`` probes test
precision: what must *not* be reported, such as commented-out code, ``%PUT``
text, or variables named like options.

A probe the chunker does not handle yet names its gap: why it fails, and which
phase of ``docs/plans/chunker-sas-coverage.md`` fixes it. It runs as
``xfail(strict=True)``, so the day it starts passing the run fails until its
``gap`` is removed — the table cannot drift from the code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from chunker import SasChunk, SasChunkResult, SasSemanticChunker
from chunker import SasChunkKind as K

# ── why a probe fails today, and the plan phase that fixes it ──────────────────
# Every gap the plan named is closed. A probe added for a construct the chunker
# does not handle yet names its reason here and in its ``gap``.

GAPS: frozenset[str] = frozenset()

# A probe's free-form check: (result, top-level chunks) -> a problem, or None.
Check = Callable[[SasChunkResult, list[SasChunk]], "str | None"]

_UNKNOWN = {K.UNKNOWN_STATEMENT_GROUP, K.UNKNOWN_BLOCK}


@dataclass(frozen=True)
class Probe:
    """One SAS snippet and what a correct chunker reports for it.

    Every expectation is optional. ``inputs``/``outputs`` and the other set
    fields compare exactly with the union over the top-level chunks;
    ``has_in``/``has_out`` only require the names to be present; ``never``
    and ``never_ref`` list names that must not be reported anywhere.
    """

    pid: str
    construct: str
    src: str
    kinds: list[K] | None = None
    inputs: set[str] | None = None
    outputs: set[str] | None = None
    has_in: set[str] = field(default_factory=set)
    has_out: set[str] = field(default_factory=set)
    never: set[str] = field(default_factory=set)  # not an input or output
    never_ref: set[str] = field(default_factory=set)  # nowhere, incl. referenced
    no_unknown: bool = True  # no UNKNOWN_* chunk, no UNRECOGNIZED diagnostic
    closed: bool = True  # no UNCLOSED_* diagnostic
    paths: set[str] = field(default_factory=set)  # substrings of external paths
    defines_librefs: set[str] | None = None
    includes: set[str] | None = None
    defines_macros: set[str] | None = None
    invokes: set[str] | None = None
    produces: set[str] | None = None
    body_in: set[str] | None = None
    body_out: set[str] | None = None
    check: Check | None = None
    gap: str | None = None  # why it fails today; the run expects it to fail


def _problems(p: Probe) -> list[str]:
    """Every way the chunker's answer for *p* differs from the expected one."""
    result = SasSemanticChunker(min_words=1, max_words=100_000, timeout=30).chunk_text(
        p.src, source_id=f"{p.pid}.sas"
    )
    top = [c for c in result.chunks if c.parent_id is None]

    def union(name: str) -> set[str]:
        return set().union(*(getattr(c.metadata, name) for c in top))

    ins, outs = union("input_datasets"), union("output_datasets")
    codes = [d.code for d in result.diagnostics]
    problems: list[str] = []
    kinds = [c.kind for c in top]
    if p.kinds is not None and kinds != p.kinds:
        problems.append(f"kinds {[k.value for k in kinds]} != {[k.value for k in p.kinds]}")
    if p.no_unknown and (
        any(k in _UNKNOWN for k in kinds) or "UNRECOGNIZED_SOURCE_REGION" in codes
    ):
        bad = [c.text.strip()[:50] for c in top if c.kind in _UNKNOWN]
        problems.append(f"unrecognised region(s) {bad}")
    if p.closed and (unclosed := [c for c in codes if c.startswith("UNCLOSED")]):
        problems.append(f"unclosed diagnostics {unclosed}")
    if p.inputs is not None and ins != p.inputs:
        problems.append(f"inputs {sorted(ins)} != {sorted(p.inputs)}")
    if p.outputs is not None and outs != p.outputs:
        problems.append(f"outputs {sorted(outs)} != {sorted(p.outputs)}")
    if missing := p.has_in - ins:
        problems.append(f"inputs lack {sorted(missing)} (got {sorted(ins)})")
    if missing := p.has_out - outs:
        problems.append(f"outputs lack {sorted(missing)} (got {sorted(outs)})")
    if bogus := p.never & (ins | outs):
        problems.append(f"bogus dataset(s) {sorted(bogus)}")
    referenced = (
        ins
        | outs
        | union("referenced_datasets")
        | union("body_literal_inputs")
        | union("body_literal_outputs")
    )
    if bogus := p.never_ref & referenced:
        problems.append(f"bogus reference(s) {sorted(bogus)}")
    paths = {ref.path for c in top for ref in c.metadata.external_refs}
    if missing := {s for s in p.paths if not any(s in path for path in paths)}:
        problems.append(f"paths lack {sorted(missing)} (got {sorted(paths)})")
    for attr, meta_field in (
        ("defines_librefs", "defines_librefs"),
        ("includes", "includes"),
        ("defines_macros", "defines_macros"),
        ("invokes", "invokes_macros"),
        ("produces", "produces_macrovars"),
        ("body_in", "body_literal_inputs"),
        ("body_out", "body_literal_outputs"),
    ):
        want = getattr(p, attr)
        if want is not None and (got := union(meta_field)) != want:
            problems.append(f"{meta_field} {sorted(got)} != {sorted(want)}")
    if p.check is not None and (msg := p.check(result, top)):
        problems.append(msg)
    return problems


def _sql(body: str) -> str:
    return f"proc sql;\n{body}\nquit;\n"


LEXICAL_PROBES = [
    Probe("L01", "block comment between steps", "/* load */\ndata a; set b; run;\n",
          kinds=[K.COMMENT_BLOCK, K.DATA_STEP], inputs={"work.b"}, outputs={"work.a"}),
    Probe("L02", "* comment statement", "* load the data;\ndata a; set b; run;\n",
          kinds=[K.COMMENT_BLOCK, K.DATA_STEP]),
    Probe("L03", "%* macro comment statement", "%* macro-level note;\ndata a; set b; run;\n",
          kinds=[K.COMMENT_BLOCK, K.DATA_STEP]),
    Probe("L04", "semicolon inside a quoted string", "data a; x = 'a;b'; y = \"c;d\"; run;\n",
          kinds=[K.DATA_STEP], outputs={"work.a"}),
    Probe("L05", "doubled quote escape", "data a; x = 'it''s; fine'; run;\n",
          kinds=[K.DATA_STEP], outputs={"work.a"}),
    Probe("L06", "code inside a comment is not code", "/* data x; set y; run; */\n%let a = 1;\n",
          kinds=[K.COMMENT_BLOCK, K.GLOBAL_STATEMENT], inputs=set(), outputs=set()),
    Probe("L07", "name literal dataset ('my data'n)",
          "options validmemname=extend;\ndata 'my data'n; set 'src tab'n; run;\n",
          check=lambda r, top: None
          if any("my data" in o for c in top for o in c.metadata.output_datasets)
          and any("src tab" in i for c in top for i in c.metadata.input_datasets)
          else "name-literal datasets not recorded"),
    Probe("L08", "datalines with keyword-like data",
          "data x;\n  input word $;\ndatalines;\nproc\ndata\nrun\n;\nrun;\n",
          kinds=[K.DATA_STEP], outputs={"work.x"}),
    Probe("L09", "datalines4 with semicolons in the data",
          "data x;\n  input line $40.;\ndatalines4;\na;b;c\nd;e\n;;;;\nrun;\n",
          kinds=[K.DATA_STEP], outputs={"work.x"}),
    Probe("L10", "cards statement", "data x;\n  input a b;\ncards;\n1 2\n3 4\n;\nrun;\n",
          kinds=[K.DATA_STEP], outputs={"work.x"}),
    Probe("L11", "%str(;) inside %let", "%let sep = %str(;);\ndata a; set b; run;\n",
          kinds=[K.GLOBAL_STATEMENT, K.DATA_STEP], inputs={"work.b"}),
    Probe("L12", "upper case, CRLF line ends", "DATA Work.Out;\r\n  SET Lib.In;\r\nRUN;\r\n",
          kinds=[K.DATA_STEP], inputs={"lib.in"}, outputs={"work.out"}),
    Probe("L13", "inline comments inside statements",
          "data /* target */ out; set /* source */ lib.src; run;\n",
          kinds=[K.DATA_STEP], inputs={"lib.src"}, outputs={"work.out"}),
    Probe("L14", "missing RUN between steps (implicit close)",
          "data a; set b;\ndata c; set a; run;\n",
          kinds=[K.DATA_STEP, K.DATA_STEP], inputs={"work.b", "work.a"},
          outputs={"work.a", "work.c"}, closed=False),
    Probe("L15", "RUN CANCEL closes the step", "data a; set b; run cancel;\ndata c; set d; run;\n",
          kinds=[K.DATA_STEP, K.DATA_STEP]),
    Probe("L16", "interactive PROC: run groups until QUIT",
          "proc datasets lib=work nolist;\n  delete a;\nrun;\n  delete b;\nrun;\nquit;\n",
          kinds=[K.PROC_STEP]),
    Probe("L17", "unterminated string is reported, not fatal",
          "data a; x = 'oops; run;\ndata b; set c; run;\n",
          no_unknown=False, closed=False,
          check=lambda r, top: None if r.diagnostics else "no diagnostic for the unterminated string"),
    Probe("L18", "unclosed block comment is reported",
          "data a; set b; run;\n/* never closed\ndata c; run;\n",
          no_unknown=False, closed=False,
          check=lambda r, top: None
          if "UNCLOSED_BLOCK_COMMENT" in {d.code for d in r.diagnostics}
          else "no UNCLOSED_BLOCK_COMMENT"),
]

DATA_STEP_PROBES = [
    Probe("D01", "SET several datasets", "data c; set a lib.b; run;\n",
          inputs={"work.a", "lib.b"}, outputs={"work.c"}),
    Probe("D02", "SET with dataset options",
          "data c; set lib.a(keep=x y where=(x > 1) rename=(y=z)); run;\n",
          inputs={"lib.a"}, outputs={"work.c"}),
    Probe("D03", "SET numbered range list ds1-ds3", "data all; set ds1-ds3; run;\n",
          inputs={"work.ds1", "work.ds2", "work.ds3"}, outputs={"work.all"}),
    Probe("D04", "SET prefix list lib.sales_:", "data all; set lib.sales_:; run;\n",
          check=lambda r, top: None
          if any(i.startswith("lib.sales") for c in top for i in c.metadata.input_datasets)
          else "prefix list not recorded as an input"),
    Probe("D05", "SET ... END= option", "data b; set a end=eof; if eof then put 'done'; run;\n",
          inputs={"work.a"}, outputs={"work.b"}, never={"work.eof"}),
    Probe("D06", "SET ... POINT= NOBS=",
          "data b;\n  do p = 1 to n by 2;\n    set a point=p nobs=n;\n    output;\n  end;\n  stop;\nrun;\n",
          inputs={"work.a"}, outputs={"work.b"}, never={"work.p", "work.n"}),
    Probe("D07", "SET ... INDSNAME=", "data c; set a b indsname=src; from = src; run;\n",
          inputs={"work.a", "work.b"}, outputs={"work.c"}, never={"work.src"}),
    Probe("D08", "MERGE with IN= and BY",
          "data c; merge a(in=ina) lib.b(in=inb); by id; if ina and inb; run;\n",
          inputs={"work.a", "lib.b"}, outputs={"work.c"}),
    Probe("D09", "UPDATE master transaction", "data master; update master trans; by id; run;\n",
          inputs={"work.master", "work.trans"}, outputs={"work.master"}),
    Probe("D10", "MODIFY master transaction",
          "data lib.master; modify lib.master trans; by id; run;\n",
          inputs={"lib.master", "work.trans"}, outputs={"lib.master"}),
    Probe("D11", "several outputs with OUTPUT",
          "data hi lo; set src; if x > 0 then output hi; else output lo; run;\n",
          inputs={"work.src"}, outputs={"work.hi", "work.lo"}),
    Probe("D12", "DATA _NULL_ with CALL SYMPUTX",
          "data _null_; set ctl; call symputx('n', count); run;\n",
          inputs={"work.ctl"}, outputs=set(), produces={"n"}),
    Probe("D13", "DATA step view", "data v / view=v; set a; run;\n",
          inputs={"work.a"}, outputs={"work.v"}),
    Probe("D14", "INFILE + INPUT (external file)",
          "data a; infile '/data/raw.csv' dlm=',' firstobs=2; input x y; run;\n",
          outputs={"work.a"}, inputs=set(), paths={"/data/raw.csv"}),
    Probe("D15", "FILE + PUT (external file)",
          "data _null_; set a; file '/out/report.txt'; put x; run;\n",
          inputs={"work.a"}, outputs=set(), paths={"/out/report.txt"}),
    Probe("D16", "hash object loaded from a dataset",
          "data b;\n  if _n_ = 1 then do;\n    declare hash h(dataset: 'lib.lookup');\n"
          "    h.definekey('k'); h.definedata('v'); h.definedone();\n  end;\n  set a;\n"
          "  rc = h.find();\nrun;\n",
          inputs={"lib.lookup", "work.a"}, outputs={"work.b"}),
    Probe("D17", "hash OUTPUT method writes a dataset",
          "data _null_;\n  declare hash h(dataset: 'a', ordered: 'y');\n"
          "  h.definekey('k'); h.definedata('k', 'v'); h.definedone();\n"
          "  h.output(dataset: 'lib.sorted');\nrun;\n",
          inputs={"work.a"}, outputs={"lib.sorted"}),
    Probe("D18", "IF 0 THEN SET (attributes only)", "data b; if 0 then set lib.tmpl; x = 1; run;\n",
          inputs={"lib.tmpl"}, outputs={"work.b"}),
    Probe("D19", "SASHELP input", "data c; set sashelp.class; run;\n",
          inputs={"sashelp.class"}, outputs={"work.c"}),
    Probe("D20", "physical-path datasets", "data '/tmp/out'; set '/data/in.sas7bdat'; run;\n",
          check=lambda r, top: None
          if top and top[0].metadata.input_datasets and top[0].metadata.output_datasets
          else "physical paths not recorded as I/O"),
    Probe("D21", "SET with KEY= / UNIQUE", "data c; set a; set lib.idx key=id / unique; run;\n",
          inputs={"work.a", "lib.idx"}, outputs={"work.c"}, never={"work.id", "work.unique"}),
    Probe("D22", "SET with OBS=/FIRSTOBS= options", "data c; set b(firstobs=2 obs=10); run;\n",
          inputs={"work.b"}, outputs={"work.c"}),
    Probe("D23", "DATA statement with dataset options",
          "data a(drop=tmp) lib.b(keep=id); set src; run;\n",
          inputs={"work.src"}, outputs={"work.a", "lib.b"}),
    Probe("D24", "DATA step statement catalogue (array/retain/do)",
          "data b; set a; array v{3} x1-x3; retain total 0; do i = 1 to 3; total + v{i}; end; run;\n",
          check=lambda r, top: None
          if {"array", "retain", "do"} <= set(top[0].metadata.data_step_statements)
          else f"data_step_statements={top[0].metadata.data_step_statements}"),
    Probe("D25", "CALL EXECUTE generating code",
          "data _null_; set ctl; call execute(cats('%nrstr(%load)(', name, ');')); run;\n",
          inputs={"work.ctl"}),
    Probe("D26", "DATA step with WHERE and BY", "data c; set lib.a; where x > 1; by id; run;\n",
          inputs={"lib.a"}, outputs={"work.c"}, never={"work.x", "work.id"}),
    Probe("D27", "DATA step OUTPUT with options",
          "data a b; set src; output a(keep=id); output b; run;\n",
          inputs={"work.src"}, outputs={"work.a", "work.b"}),
    Probe("D28", "DATAn: bare DATA statement", "data; set a; run;\n", inputs={"work.a"}),
    Probe("D29", "MERGE with RENAME= options",
          "data c; merge a(in=a1 rename=(x=y)) b; by id; run;\n",
          inputs={"work.a", "work.b"}, outputs={"work.c"}),
    Probe("D30", "WHERE= with a macro date literal",
          "data out; set lib.in(where=(dt >= \"&start\"d)); run;\n",
          inputs={"lib.in"}, outputs={"work.out"}),
]

PROC_PROBES = [
    Probe("P01", "SORT with OUT=", "proc sort data=a out=b nodupkey; by id; run;\n",
          inputs={"work.a"}, outputs={"work.b"}),
    Probe("P02", "SORT in place", "proc sort data=lib.a; by id; run;\n",
          inputs={"lib.a"}, outputs={"lib.a"}),
    Probe("P03", "SORT with DUPOUT=", "proc sort data=a out=b dupout=dups nodupkey; by id; run;\n",
          inputs={"work.a"}, outputs={"work.b", "work.dups"}),
    Probe("P04", "MEANS ... OUTPUT OUT=",
          "proc means data=a noprint; class g; var x; output out=stats mean=m; run;\n",
          inputs={"work.a"}, outputs={"work.stats"}),
    Probe("P05", "SUMMARY with two OUTPUT statements",
          "proc summary data=a nway; class g; var x; output out=s1 mean=; output out=s2 sum=; run;\n",
          inputs={"work.a"}, outputs={"work.s1", "work.s2"}),
    Probe("P06", "FREQ ... TABLES / OUT=", "proc freq data=a; tables g / out=cnt; run;\n",
          inputs={"work.a"}, outputs={"work.cnt"}),
    Probe("P07", "TRANSPOSE", "proc transpose data=a out=t prefix=c; by id; var x; run;\n",
          inputs={"work.a"}, outputs={"work.t"}),
    # BASE= gets DATA='s rows added: read and rewritten in place (UPDATE).
    Probe("P08", "APPEND BASE= DATA=", "proc append base=lib.master data=new force; run;\n",
          inputs={"work.new", "lib.master"}, outputs={"lib.master"}),
    Probe("P09", "COMPARE BASE= COMPARE= OUT=",
          "proc compare base=a compare=b out=diff outnoequal; id k; run;\n",
          inputs={"work.a", "work.b"}, outputs={"work.diff"}),
    Probe("P10", "DATASETS APPEND",
          "proc datasets lib=work nolist;\n  append base=all data=part;\nrun;\nquit;\n",
          inputs={"work.part", "work.all"}, outputs={"work.all"}, kinds=[K.PROC_STEP]),
    Probe("P11", "DATASETS DELETE / CHANGE / MODIFY",
          "proc datasets lib=lib nolist;\n  delete tmp1 tmp2;\n  change old=new;\n  modify new;\n"
          "    rename a=b;\nquit;\n",
          kinds=[K.PROC_STEP], never={"work.tmp1", "work.new"}),
    Probe("P12", "COPY IN= OUT= (librefs, not datasets)",
          "proc copy in=src out=tgt memtype=data; select a b; run;\n",
          never={"work.tgt", "work.src"}),
    Probe("P13", "IMPORT DATAFILE= OUT=",
          "proc import datafile='/d/x.csv' out=lib.x dbms=csv replace; getnames=yes; run;\n",
          outputs={"lib.x"}, paths={"/d/x.csv"}),
    Probe("P14", "EXPORT DATA= OUTFILE=",
          "proc export data=lib.x outfile='/d/x.xlsx' dbms=xlsx replace; sheet='s1'; run;\n",
          inputs={"lib.x"}, paths={"/d/x.xlsx"}),
    Probe("P15", "FORMAT CNTLIN=", "proc format cntlin=fmtds library=lib; run;\n",
          inputs={"work.fmtds"}),
    Probe("P16", "FORMAT CNTLOUT=", "proc format library=lib cntlout=fmtout; run;\n",
          outputs={"work.fmtout"}),
    Probe("P17", "FORMAT VALUE statements",
          "proc format;\n  value agegrp 0-17 = 'child' 18-high = 'adult';\n"
          "  invalue yn 'Y' = 1 'N' = 0;\nrun;\n",
          kinds=[K.PROC_STEP]),
    Probe("P18", "REG OUTEST= and OUTPUT OUT=",
          "proc reg data=a outest=est;\n  model y = x;\n  output out=pred p=yhat;\nrun;\nquit;\n",
          inputs={"work.a"}, outputs={"work.est", "work.pred"}),
    Probe("P19", "LOGISTIC OUTMODEL= / OUTPUT OUT=",
          "proc logistic data=a outmodel=mdl; model y(event='1') = x; output out=scored p=prob; run;\n",
          inputs={"work.a"}, outputs={"work.mdl", "work.scored"}),
    Probe("P20", "LOGISTIC INMODEL= / SCORE DATA= OUT=",
          "proc logistic inmodel=mdl; score data=new out=scored; run;\n",
          inputs={"work.mdl", "work.new"}, outputs={"work.scored"}),
    Probe("P21", "CORR OUTP=", "proc corr data=a outp=corrs noprint; var x y; run;\n",
          inputs={"work.a"}, outputs={"work.corrs"}),
    Probe("P22", "UNIVARIATE OUTPUT OUT=",
          "proc univariate data=a noprint; var x; output out=pct pctlpts=5 95 pctlpre=p; run;\n",
          inputs={"work.a"}, outputs={"work.pct"}),
    Probe("P23", "REPORT OUT=",
          "proc report data=a out=rep nowd; column g x; define g / group; run;\n",
          inputs={"work.a"}, outputs={"work.rep"}),
    Probe("P24", "TABULATE OUT=",
          "proc tabulate data=a out=tab; class g; var x; table g, x*sum; run;\n",
          inputs={"work.a"}, outputs={"work.tab"}),
    Probe("P25", "PRINT with options", "proc print data=lib.a(obs=10) noobs label; run;\n",
          inputs={"lib.a"}, outputs=set()),
    Probe("P26", "SGPLOT", "proc sgplot data=a; scatter x=x y=y; run;\n",
          inputs={"work.a"}, outputs=set()),
    Probe("P27", "CONTENTS DATA=lib._ALL_ OUT=",
          "proc contents data=lib._all_ out=meta noprint; run;\n",
          has_out={"work.meta"}, never={"lib._all_"}),
    Probe("P28", "RANK OUT=", "proc rank data=a out=r groups=10; var x; ranks rx; run;\n",
          inputs={"work.a"}, outputs={"work.r"}),
    Probe("P29", "SURVEYSELECT OUT=",
          "proc surveyselect data=a out=samp method=srs n=100 seed=1; run;\n",
          inputs={"work.a"}, outputs={"work.samp"}),
    Probe("P30", "SCORE DATA= SCORE= OUT=",
          "proc score data=new score=coef out=scored type=parms; var x; run;\n",
          inputs={"work.new", "work.coef"}, outputs={"work.scored"}),
    Probe("P31", "GLM OUTPUT OUT=",
          "proc glm data=a; class g; model y = g; output out=res r=resid; run; quit;\n",
          inputs={"work.a"}, outputs={"work.res"}),
    Probe("P32", "ODS OUTPUT table= creates a dataset",
          "ods output Summary=sumstats;\nproc means data=a; var x; run;\n",
          has_out={"work.sumstats"}),
    Probe("P33", "TTEST inside ODS OUTPUT",
          "proc ttest data=a; class g; var x; ods output TTests=tt; run;\n",
          has_out={"work.tt"}),
    Probe("P34", "IMPORT from a fileref",
          "filename f '/d/x.csv';\nproc import datafile=f out=x dbms=csv replace; run;\n",
          outputs={"work.x"}, paths={"/d/x.csv"}),
    Probe("P35", "PRINTTO LOG=", "proc printto log='/tmp/job.log' new; run;\n",
          paths={"/tmp/job.log"}),
    Probe("P36", "HTTP OUT= is a fileref",
          "filename resp temp;\nproc http url='https://api.example.com/x' method='GET' out=resp; run;\n",
          never={"work.resp"}),
    Probe("P37", "STANDARD OUT=", "proc standard data=a out=z mean=0 std=1; var x; run;\n",
          inputs={"work.a"}, outputs={"work.z"}),
    Probe("P38", "MEANS with CLASSDATA=", "proc means data=a classdata=lvls; class g; run;\n",
          inputs={"work.a", "work.lvls"}),
    Probe("P39", "UPLOAD / DOWNLOAD (SAS/CONNECT)", "proc download data=remote.a out=local; run;\n",
          inputs={"remote.a"}, outputs={"work.local"}),
    Probe("P40", "SORT with WHERE= option on DATA=",
          "proc sort data=lib.a(where=(x > 1)) out=b; by id; run;\n",
          inputs={"lib.a"}, outputs={"work.b"}),
]

SQL_PROBES = [
    Probe("S01", "CREATE TABLE ... INNER JOIN",
          _sql("create table c as select a.*, b.y from a inner join lib.b on a.id = b.id;"),
          inputs={"work.a", "lib.b"}, outputs={"work.c"}),
    Probe("S02", "comma join: FROM a, b",
          _sql("create table c as select * from a as x, lib.b as y where x.id = y.id;"),
          inputs={"work.a", "lib.b"}, outputs={"work.c"}),
    Probe("S03", "three-way comma join",
          _sql("create table c as select * from a, b, lib.c3 where a.k = b.k and b.k = c3.k;"),
          inputs={"work.a", "work.b", "lib.c3"}, outputs={"work.c"}),
    Probe("S04", "subquery in WHERE",
          _sql("create table c as select * from a where id in (select id from lib.b);"),
          inputs={"work.a", "lib.b"}, outputs={"work.c"}),
    Probe("S05", "inline view in FROM",
          _sql("create table c as select * from (select id from lib.b) as s;"),
          inputs={"lib.b"}, outputs={"work.c"}),
    # INSERT adds rows to what the table holds: read and rewritten in place
    # (UPDATE), as APPEND's BASE= is.
    Probe("S06", "INSERT INTO ... SELECT", _sql("insert into lib.t select * from s;"),
          inputs={"work.s", "lib.t"}, outputs={"lib.t"}),
    Probe("S07", "INSERT INTO ... VALUES", _sql("insert into lib.t values (1, 'a');"),
          outputs={"lib.t"}, inputs={"lib.t"}),
    Probe("S08", "DELETE FROM modifies the table", _sql("delete from lib.t where x = 1;"),
          has_out={"lib.t"}),
    Probe("S09", "UPDATE ... SET modifies the table", _sql("update lib.t set x = 1 where y = 2;"),
          has_out={"lib.t"}),
    Probe("S10", "SELECT INTO :macro var",
          "proc sql noprint;\nselect count(*) into :n trimmed from lib.t;\nquit;\n",
          inputs={"lib.t"}, outputs=set(), produces={"n"}),
    Probe("S11", "CREATE VIEW", _sql("create view v as select * from a;"),
          inputs={"work.a"}, outputs={"work.v"}),
    Probe("S12", "UNION of two tables",
          _sql("create table u as select * from a union select * from lib.b;"),
          inputs={"work.a", "lib.b"}, outputs={"work.u"}),
    Probe("S13", "CREATE TABLE ... LIKE", _sql("create table c like lib.t;"),
          inputs={"lib.t"}, outputs={"work.c"}),
    Probe("S14", "DROP TABLE is not a read", _sql("drop table lib.t;"), never={"lib.t"}),
    Probe("S15", "ALTER TABLE modifies the table", _sql("alter table lib.t add z num;"),
          has_out={"lib.t"}),
    Probe("S16", "FROM with dataset options",
          _sql("create table c as select * from lib.a(where=(x > 1));"),
          inputs={"lib.a"}, outputs={"work.c"}),
    Probe("S17", "NATURAL / LEFT JOIN chain",
          _sql("create table c as select * from a natural join b left join lib.d on b.k = d.k;"),
          inputs={"work.a", "work.b", "lib.d"}, outputs={"work.c"}),
    Probe("S18", "OUTER UNION CORR",
          _sql("create table u as select * from a outer union corr select * from b;"),
          inputs={"work.a", "work.b"}, outputs={"work.u"}),
    Probe("S19", "identifier containing 'from'",
          _sql("create table c as select from_code, x from a;"),
          inputs={"work.a"}, outputs={"work.c"}),
    Probe("S20", "several statements in one PROC SQL",
          _sql("create table t1 as select * from a;\ncreate table t2 as select * from t1 where x > 0;\n"
               "drop table t1;"),
          has_in={"work.a"}, has_out={"work.t1", "work.t2"}),
    Probe("S21", "CASE / CALCULATED / GROUP BY / HAVING",
          _sql("create table s as select g, sum(x) as tot, calculated tot / 2 as half,\n"
               "  case when calculated tot > 10 then 'big' else 'small' end as size\n"
               "  from lib.a group by g having calculated tot > 1 order by g;"),
          inputs={"lib.a"}, outputs={"work.s"}),
    Probe("S22", "DESCRIBE TABLE / CREATE INDEX",
          _sql("describe table lib.t;\ncreate index id on lib.t(id);"), kinds=[K.PROC_STEP]),
    Probe("S23", "VALIDATE and RESET statements",
          "proc sql;\nreset noprint;\nvalidate select * from a;\nquit;\n", kinds=[K.PROC_STEP]),
]

GLOBAL_PROBES = [
    Probe("G01", "LIBNAME path", "libname lib '/data/lib';\n",
          kinds=[K.GLOBAL_STATEMENT], defines_librefs={"lib"}, paths={"/data/lib"}),
    Probe("G02", "LIBNAME with engine", "libname x xlsx '/d/a.xlsx';\n",
          kinds=[K.GLOBAL_STATEMENT], defines_librefs={"x"}, paths={"/d/a.xlsx"}),
    Probe("G03", "LIBNAME concatenation", "libname all (lib1 lib2);\n",
          kinds=[K.GLOBAL_STATEMENT], defines_librefs={"all"}),
    Probe("G04", "LIBNAME CLEAR / _ALL_ LIST", "libname lib clear;\nlibname _all_ list;\n",
          kinds=[K.GLOBAL_STATEMENT, K.GLOBAL_STATEMENT]),
    Probe("G05", "FILENAME PIPE / URL / TEMP",
          "filename ls pipe 'ls -l /data';\nfilename web url 'https://example.com/a.csv';\n"
          "filename t temp;\n",
          kinds=[K.GLOBAL_STATEMENT] * 3, paths={"ls -l /data", "https://example.com/a.csv"}),
    Probe("G06", "OPTIONS", "options mprint mlogic symbolgen obs=max;\n", kinds=[K.OPTIONS],
          check=lambda r, top: None if "mprint" in top[0].metadata.options
          else f"options={top[0].metadata.options}"),
    Probe("G07", "TITLE / FOOTNOTE", "title1 'Report';\nfootnote 'Confidential';\n",
          kinds=[K.GLOBAL_STATEMENT, K.GLOBAL_STATEMENT]),
    Probe("G08", "ODS destination around a PROC",
          "ods html file='/out/r.html';\nproc print data=a; run;\nods html close;\n",
          kinds=[K.GLOBAL_STATEMENT, K.PROC_STEP, K.GLOBAL_STATEMENT], paths={"/out/r.html"}),
    Probe("G09", "X command", "x 'mkdir -p /tmp/work';\n", kinds=[K.GLOBAL_STATEMENT]),
    Probe("G10", "SYSTASK", "systask command \"ls\" wait;\n", kinds=[K.GLOBAL_STATEMENT]),
    Probe("G11", "ENDSAS", "data a; set b; run;\nendsas;\n",
          kinds=[K.DATA_STEP, K.GLOBAL_STATEMENT]),
    Probe("G12", "DM command", "dm 'log; clear; output; clear;';\n",
          kinds=[K.GLOBAL_STATEMENT]),
    Probe("G13", "%INCLUDE a path", "%include '/code/setup.sas';\n",
          kinds=[K.INCLUDE], includes={"/code/setup.sas"}),
    Probe("G14", "%INCLUDE fileref(member)", "filename m '/code';\n%include m(setup);\n",
          kinds=[K.GLOBAL_STATEMENT, K.INCLUDE]),
    Probe("G15", "SASFILE / LOCK", "sasfile lib.big load;\nlock lib.big;\n",
          kinds=[K.GLOBAL_STATEMENT, K.GLOBAL_STATEMENT]),
    Probe("G16", "SAS/GRAPH GOPTIONS / AXIS / SYMBOL",
          "goptions reset=all;\naxis1 label=('x');\nsymbol1 v=dot;\n",
          kinds=[K.GLOBAL_STATEMENT] * 3),
    Probe("G17", "MISSING / PAGE statements", "missing a b;\npage;\n",
          kinds=[K.GLOBAL_STATEMENT, K.GLOBAL_STATEMENT]),
    Probe("G18", "standalone RUN / QUIT", "run;\nquit;\n",
          kinds=[K.STEP_BOUNDARY, K.STEP_BOUNDARY]),
    Probe("G19", "ODS GRAPHICS / ODS NORESULTS", "ods graphics on;\nods noresults;\n",
          kinds=[K.GLOBAL_STATEMENT, K.GLOBAL_STATEMENT]),
    Probe("G20", "%INCLUDE several files", "%include '/a.sas' '/b.sas' / source2;\n",
          includes={"/a.sas", "/b.sas"}),
    Probe("G21", "%INCLUDE fileref(members)", "filename src '/code';\n%include src(one two);\n",
          includes={"/code/one.sas", "/code/two.sas"}),
    Probe("G22", "INFILE through a fileref",
          "filename raw '/data/in.csv';\ndata a; infile raw dsd; input x; run;\n",
          outputs={"work.a"},
          check=lambda r, top: None
          if [p.path for p in top[1].metadata.physical_paths] == ["/data/in.csv"]
          else f"infile refs={top[1].metadata.external_refs}"),
    Probe("G23", "%INCLUDE a path built from %LET values",
          "%let root = /Code;\n%let dir = &root/lib;\n%include \"&dir/setup.sas\";\n",
          includes={"/code/lib/setup.sas"},
          check=lambda r, top: None
          if [f.path for f in top[-1].metadata.include_files] == ["/Code/lib/setup.sas"]
          else f"include_files={top[-1].metadata.include_files}"),
    Probe("G24", "%INCLUDE fileref whose FILENAME path is a %LET value",
          "%let root = /Code;\nfilename src \"&root\";\n%include src(one);\n",
          includes={"/code/one.sas"}),
]

MACRO_PROBES = [
    Probe("M01", "%MACRO with positional + keyword params",
          "%macro m(a, b=1);\n  data &a; set &b; run;\n%mend m;\n",
          kinds=[K.MACRO_DEFINITION], defines_macros={"m"},
          check=lambda r, top: None if set(top[0].metadata.macro_param_names) == {"a", "b"}
          else f"macro_param_names={top[0].metadata.macro_param_names}"),
    Probe("M02", "macro call with semicolon", "%macro m; %put hi; %mend;\n%m;\n",
          kinds=[K.MACRO_DEFINITION, K.MACRO_CALL], invokes={"m"}),
    Probe("M03", "macro calls without semicolons",
          "%macro m(x); %put &x; %mend;\n%m(1)\n%m(2)\ndata a; set b; run;\n",
          kinds=[K.MACRO_DEFINITION, K.MACRO_CALL, K.MACRO_CALL, K.DATA_STEP]),
    Probe("M04", "nested %MACRO definitions",
          "%macro outer;\n  %macro inner; %put in; %mend inner;\n  %inner\n%mend outer;\n",
          kinds=[K.MACRO_DEFINITION], defines_macros={"outer", "inner"}),
    Probe("M05", "%LET / %GLOBAL / %LOCAL / %PUT", "%global g1 g2;\n%let g1 = lib.a;\n%put &=g1;\n",
          kinds=[K.GLOBAL_STATEMENT] * 3),
    Probe("M06", "open-code %IF/%THEN/%DO/%END",
          "%if &env = prod %then %do;\n  data a; set b; run;\n%end;\n",
          kinds=[K.MACRO_CONTROL_FLOW, K.DATA_STEP, K.MACRO_CONTROL_FLOW]),
    Probe("M07", "%DO loop generating steps",
          "%macro loop;\n  %do i = 1 %to 3;\n    data out&i; set in&i; run;\n  %end;\n%mend;\n",
          kinds=[K.MACRO_DEFINITION], defines_macros={"loop"}),
    Probe("M08", "%GOTO and labels",
          "%macro m;\n  %goto skip;\n  %put no;\n  %skip: %put yes;\n%mend;\n",
          kinds=[K.MACRO_DEFINITION]),
    Probe("M09", "%ABORT inside a macro",
          "%macro m;\n  %if &syserr > 0 %then %abort cancel;\n%mend;\n",
          kinds=[K.MACRO_DEFINITION],
          check=lambda r, top: None if top[0].metadata.contains_abort else "contains_abort not set"),
    Probe("M10", "%SYSFUNC in a dataset name",
          "data out_%sysfunc(today(), yymmddn8.); set a; run;\n",
          has_in={"work.a"},
          check=lambda r, top: None
          if any("out_" in o for c in top for o in c.metadata.output_datasets)
          else f"generated output name missing: {[c.metadata.output_datasets for c in top]}"),
    Probe("M11", "macro call as a SET operand", "data b; set %dslist(lib); run;\n",
          outputs={"work.b"}, invokes={"dslist"}, never={"work.dslist"}),
    Probe("M12", "%MACRO / PARMBUFF", "%macro m / parmbuff;\n  %put &syspbuff;\n%mend;\n",
          kinds=[K.MACRO_DEFINITION], defines_macros={"m"}),
    Probe("M13", "%MACRO / STORE SOURCE DES=",
          "%macro m(a) / store source des='util';\n  %put &a;\n%mend;\n",
          kinds=[K.MACRO_DEFINITION], defines_macros={"m"}),
    Probe("M14", "%INCLUDE inside a macro", "%macro m;\n  %include '/code/inc.sas';\n%mend;\n",
          kinds=[K.MACRO_DEFINITION], includes={"/code/inc.sas"}),
    Probe("M15", "autocall macro functions", "%let u = %upcase(&x);\n%let t = %trim(%left(&y));\n",
          kinds=[K.GLOBAL_STATEMENT, K.GLOBAL_STATEMENT]),
    Probe("M16", "%STR(;) in macro call arguments",
          "%macro m(sep); %put &sep; %mend;\n%m(sep=%str(;));\ndata a; set b; run;\n",
          kinds=[K.MACRO_DEFINITION, K.MACRO_CALL, K.DATA_STEP]),
    Probe("M17", "%NRSTR(%MEND) inside a macro body",
          "%macro m;\n  %put %nrstr(%mend);\n%mend;\ndata a; set b; run;\n",
          kinds=[K.MACRO_DEFINITION, K.DATA_STEP]),
    Probe("M18", "%DO %WHILE / %UNTIL",
          "%macro m;\n  %let i = 0;\n  %do %while(&i < 3);\n    %let i = %eval(&i + 1);\n"
          "  %end;\n%mend;\n",
          kinds=[K.MACRO_DEFINITION]),
    Probe("M19", "macro body datasets", "%macro m;\n  data lib.out; set lib.in; run;\n%mend;\n",
          body_in={"lib.in"}, body_out={"lib.out"}),
    Probe("M20", "CALL SYMPUT then &var in a later step",
          "data _null_; call symputx('src', 'lib.daily'); run;\ndata copy; set &src; run;\n",
          has_in={"lib.daily"}),
    Probe("M21", "macro call inside a DATA step body", "data b; set a; %derive(x) run;\n",
          kinds=[K.DATA_STEP], inputs={"work.a"}, outputs={"work.b"}),
    Probe("M22", "%SYSFUNC(EXIST()) guard in open code",
          "%if %sysfunc(exist(lib.a)) %then %do;\n  proc print data=lib.a; run;\n%end;\n",
          kinds=[K.MACRO_CONTROL_FLOW, K.PROC_STEP, K.MACRO_CONTROL_FLOW]),
    Probe("M23", "parameter default containing commas",
          "%macro m(list=%str(a,b), n=2);\n  %put &list &n;\n%mend;\n",
          check=lambda r, top: None if set(top[0].metadata.macro_param_names) == {"list", "n"}
          else f"macro_param_names={top[0].metadata.macro_param_names}"),
]

ACCESS_PROBES = [
    Probe("A01", "explicit pass-through (CONNECTION TO)",
          "proc sql;\nconnect to oracle (path=P);\ncreate table nonip as select * from connection to oracle\n"
          "(select * from edw.current_nonip);\ndisconnect from oracle;\nquit;\n",
          outputs={"work.nonip"}, inputs=set(),
          check=lambda r, top: None
          if any(str(t).startswith("oracle:edw.current_nonip") for c in top for t in c.metadata.db_tables)
          else "database table missing"),
    Probe("A02", "EXECUTE (...) BY",
          "proc sql;\nconnect to oracle (path=P);\nexecute (truncate table edw.stage) by oracle;\nquit;\n",
          check=lambda r, top: None
          if any(t.access == "write" for c in top for t in c.metadata.db_tables)
          else "write not recorded"),
    Probe("A03", "database LIBNAME read",
          "libname edw oracle path=P schema=fr_dm;\ndata a; set edw.accounts; run;\n",
          check=lambda r, top: None
          if any(str(t).startswith("oracle:fr_dm.accounts") for c in top for t in c.metadata.db_tables)
          else "LIBNAME database table missing"),
    Probe("A04", "ODBC LIBNAME",
          "libname dw odbc dsn='warehouse' schema=dbo;\nproc print data=dw.orders; run;\n",
          defines_librefs={"dw"}, inputs={"dw.orders"}),
    Probe("A05", "CONNECT USING a libref",
          "libname edw oracle path=P;\nproc sql;\nconnect using edw;\n"
          "select * from connection to edw (select * from s.t);\nquit;\n",
          check=lambda r, top: None
          if any(t.engine == "oracle" for c in top for t in c.metadata.db_tables)
          else "engine not traced"),
    Probe("A06", "RSUBMIT / ENDRSUBMIT block",
          "signon dev;\nrsubmit;\n  data a; set b; run;\nendrsubmit;\nsignoff;\n",
          has_in={"work.b"}, has_out={"work.a"}),
]

EMBEDDED_PROBES = [
    Probe("E01", "PROC DS2 data program",
          "proc ds2;\n  data out_ds / overwrite=yes;\n    method run();\n      set in_ds;\n    end;\n"
          "  enddata;\nrun;\nquit;\n",
          kinds=[K.PROC_STEP], inputs={"work.in_ds"}, outputs={"work.out_ds"}),
    Probe("E02", "PROC FEDSQL", "proc fedsql;\n  create table c as select * from a;\nquit;\n",
          inputs={"work.a"}, outputs={"work.c"}),
    Probe("E03", "PROC IML USE / CREATE",
          "proc iml;\n  use lib.a; read all var _num_ into m; close lib.a;\n"
          "  create out from m; append from m; close out;\nquit;\n",
          kinds=[K.PROC_STEP], inputs={"lib.a"}, outputs={"work.out"}),
    Probe("E04", "PROC PYTHON submit block",
          "proc python;\nsubmit;\ndata = load()\nrun = True\nendsubmit;\nrun;\n",
          kinds=[K.PROC_STEP]),
    Probe("E05", "PROC FCMP function",
          "proc fcmp outlib=work.funcs.pkg;\n  function dbl(x);\n    return(x * 2);\n  endsub;\nrun;\n",
          kinds=[K.PROC_STEP]),
    Probe("E06", "PROC CAS action",
          "proc cas;\n  table.loadTable / path='x.csv' casOut={name='t'};\nquit;\n",
          kinds=[K.PROC_STEP]),
    Probe("E07", "PROC LUA submit block",
          "proc lua;\nsubmit;\nlocal data = sas.load('a')\nendsubmit;\nrun;\n", kinds=[K.PROC_STEP]),
]

PRECISION_PROBES = [
    Probe("X01", "commented-out SET in a DATA step",
          "data a;\n  * was: set lib.old;\n  set lib.new;\nrun;\n",
          inputs={"lib.new"}, outputs={"work.a"}, never_ref={"lib.old"}),
    Probe("X02", "commented-out OUT= in a PROC",
          "proc sort data=a out=b;\n  * out=c was the old target;\n  by id;\nrun;\n",
          outputs={"work.b"}, never_ref={"work.c"}),
    Probe("X03", "%* comment in a macro body",
          "%macro m;\n  %* old: data lib.tmp set lib.x;\n  data b; set c; run;\n%mend;\n",
          body_in={"work.c"}, body_out={"work.b"}),
    Probe("X04", "statement comment in a macro body",
          "%macro m;\n  * data lib.old;\n  data b; set c; run;\n%mend;\n",
          body_in={"work.c"}, body_out={"work.b"}),
    Probe("X05", "commented-out SQL in PROC SQL",
          "proc sql;\n  * create table old as select * from lib.legacy;\n"
          "  create table new as select * from lib.cur;\nquit;\n",
          inputs={"lib.cur"}, outputs={"work.new"}, never_ref={"lib.legacy"}),
    Probe("X06", "%PUT text in open code", "%put Loading data from staging;\n",
          never_ref={"from", "staging", "work.staging"}),
    Probe("X07", "%PUT text in a macro body", "%macro m;\n  %put Reading from stage;\n%mend;\n",
          body_in=set(), never_ref={"work.stage"}),
    Probe("X08", "%LET value containing 'from'", "%let msg = copy from src;\n",
          never_ref={"src", "work.src"}),
    Probe("X09", "DATA step variable named out", "data b; set a; out = x * 2; run;\n",
          inputs={"work.a"}, outputs={"work.b"}, never_ref={"x", "work.x"}),
    Probe("X10", "DATA step variable named data", "data b; set a; if data = y then z = 1; run;\n",
          inputs={"work.a"}, outputs={"work.b"}, never_ref={"y", "work.y"}),
    Probe("X11", "PROC variable named from", "proc print data=a; var from to; run;\n",
          inputs={"work.a"}, never_ref={"to", "work.to"}),
    Probe("X12", "PROC programming statement assigning out",
          "proc phreg data=a;\n  model t*c(0) = x;\n  out = x + 1;\nrun;\n",
          inputs={"work.a"}, outputs=set()),
    Probe("X13", "datalines holding SAS-like text",
          "data x;\n  input line $40.;\ndatalines;\nset lib.z\ndata lib.y\n;\nrun;\n",
          inputs=set(), outputs={"work.x"}, never_ref={"lib.z", "lib.y"}),
    Probe("X14", "Python SUBMIT code that looks like SAS options",
          "proc python;\nsubmit;\nout = df.merge(x)\nendsubmit;\nrun;\n",
          outputs=set(), never_ref={"df.merge"}),
    Probe("X15", "SQL inside a string literal", "data _null_;\n  put 'select * from lib.secret;';\nrun;\n",
          never_ref={"lib.secret"}),
]

PROBES = [
    *LEXICAL_PROBES,
    *DATA_STEP_PROBES,
    *PROC_PROBES,
    *SQL_PROBES,
    *GLOBAL_PROBES,
    *MACRO_PROBES,
    *ACCESS_PROBES,
    *EMBEDDED_PROBES,
    *PRECISION_PROBES,
]


def _param(p: Probe):
    marks =[pytest.mark.xfail(strict=True, reason=p.gap)] if p.gap else []
    return pytest.param(p, id=p.pid, marks=marks)


@pytest.mark.parametrize("probe", [_param(p) for p in PROBES])
def test_probe(probe: Probe) -> None:
    problems = _problems(probe)
    assert not problems, f"{probe.pid} {probe.construct}:\n  " + "\n  ".join(problems)


def test_probe_table_is_well_formed() -> None:
    ids = [p.pid for p in PROBES]
    assert len(ids) == len(set(ids)), "duplicate probe ids"
    assert {p.gap for p in PROBES if p.gap} <= GAPS, "a gap outside the known list"
