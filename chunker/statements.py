"""What each SAS statement does to the datasets it names. See chunker/README.md.

A region is read one statement at a time, never as one stretch of text, so a
dataset is recorded only where SAS reads or writes it: a DATA statement, SET,
MERGE, UPDATE, MODIFY, OUTPUT, a hash object's ``dataset:``, a PROC SQL clause,
a PROC's options — by what each names in that PROC (``keywords.PROC_OPTION_ROLES``)
— PROC DATASETS's and PROC COPY's member statements, and ODS OUTPUT. Comments,
data lines and SUBMIT code are no statements at all, and an assignment, a
``%PUT`` or a ``%LET`` names nothing.

Steps and ``%MACRO`` bodies share the walk. A body may also hold part of a
step for its caller's step to complete (``%macro sets; set a b; %mend;``), so a
body statement outside any step it opens is read as what its keyword makes it.
A body's references are classified afterwards by how their names are spelled:
written out, one of the macro's own parameters (each call site supplies it),
or built from several (no call site names one dataset). Pure: the same
statements always name the same references. No logging.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from itertools import chain

from .keywords import (
    _MACRO_LANGUAGE_WORDS,
    _SAS_RESERVED,
    PROC_OPTION_DEFAULTS,
    PROC_OPTION_ROLES,
    RUN_GROUP_PROCS,
)
from .macro_vars import DS_REF_TOKEN, has_macro_ref
from .models import DatasetRole, SasChunkKind, SasDatasetRef
from .scanner import (
    _PROC_NAME_RE,
    _STEP_END_RE,
    UnitRole,
    _classify_normed,
    _norm,
    _Unit,
    _ws_end,
)
from .sql import Dialect, SqlStatement, _delimited

# ---------------------------------------------------------------------------
# Dataset names
# ---------------------------------------------------------------------------


def _canon_ds(name: str, library: str = "work") -> str:
    """Canonicalise a dataset name for producer/consumer matching.

    A one-level name resolves to the temporary Work library — per the SAS
    Programmer's Guide: Essentials (Ch. 11), ``data mytable;`` "behaves the
    same if you specify work.mytable" — so it is rewritten to
    ``work.<name>``, unifying both spellings in the batcher's exact-string
    dataset namespace. (*library* names another default where SAS has one:
    PROC DATASETS's ``LIB=``.) Everything that is not a plain one-level
    identifier passes through unchanged:

    - two-level ``libref.member`` names;
    - names still holding a macro reference (``&table1``), whose libref is
      not knowable yet: ``&table1`` may well resolve to a two-level name, so
      calling it ``work.&table1`` would assert a library the source never
      named. :func:`~chunker.metadata.resolve_macro_var_refs` canonicalises
      again once the reference has a value, and what never resolves keeps the
      ``&``;
    - special ``_name_`` tokens (``_data_`` / ``_last_``), which are not
      Work members but placeholders the batcher's implicit-dataset pass
      resolves in corpus order;
    - quoted physical-path references (normalised by :func:`_quoted_path`
      to a leading ``'``), which address a file directly, not a library
      member.

    The rewrite is inexact when a USER library is assigned (one-level names
    then resolve to USER, not WORK — guide pp. 236, 252-253); the chunker
    emits a ``USER_LIBRARY_ASSIGNED`` diagnostic in that case rather than
    guessing.
    """
    if (
        "." in name
        or has_macro_ref(name)
        or name.startswith("'")
        or (name.startswith("_") and name.endswith("_"))
    ):
        return name
    return _member(library, name)


def _member(library: str, member: str) -> str:
    """``library.member``. A library spelled through a macro variable keeps
    its delimiter dot: ``&lib`` and ``x`` make ``&lib..x``, which SAS reads as
    the value of ``&lib``, a dot, and ``x``."""
    return f"{_delimited(library)}.{member}"


def _quoted_path(raw: str) -> str:
    """Normalise a quoted physical-path dataset reference to an exact-match
    key: lowercased, backslashes → forward slashes, wrapped in single
    quotes.  The quote wrapper is kept so a path key can never collide with
    an identifier name (``data 'perm';`` addresses a file in the current
    working directory, *not* work.perm) and so :func:`_canon_ds` passes it
    through.  Per-OS path case-sensitivity is deliberately ignored,
    consistent with the module's lowercase-everything policy."""
    inner = raw.strip()[1:-1].strip().lower().replace("\\", "/")
    return f"'{inner}'"


def _ds_name(raw: str) -> str | None:
    """*raw* lowercased, its dataset options dropped; ``None`` for a name SAS
    reserves (``_null_``, ``_all_``, a library's name alone)."""
    name = raw.strip().lower().split("(")[0].strip()
    if not name or name in _SAS_RESERVED:
        return None
    return name


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

# Where a statement stands: open code, a DATA step, or a PROC.
OPEN, DATA, PROC = "open", "data", "proc"


@dataclass(frozen=True, slots=True)
class Statement:
    """One SAS statement, from its core to its end.

    The core is what follows a ``%IF … %THEN`` or ``%ELSE``, a label, and in a
    DATA step an ``IF … THEN``, ``ELSE``, ``WHEN (…)`` or ``OTHERWISE``:
    ``if x then output hi;`` is an OUTPUT statement.

    Attributes
    ----------
    mt, cf
        The statement as sanitised for scanning (:func:`chunker.scanner._sanitise`):
        string contents blanked, and kept.
    keyword
        The core's first word, lowercased: ``set``, ``proc``, ``%let``, ``x``.
    context
        :data:`OPEN`, :data:`DATA` or :data:`PROC`, the statement included: a
        DATA statement stands in its own step.
    proc
        The PROC's name in a PROC, else ``""``.
    in_macro
        Inside a ``%MACRO`` body.
    """

    mt: str
    cf: str
    keyword: str
    context: str
    proc: str
    in_macro: bool


_WORD_RE = re.compile(r"(%\s*)?([A-Za-z_]\w*)")
_MACRO_IF_RE = re.compile(r"%\s*if\b", re.IGNORECASE)
_MACRO_THEN_RE = re.compile(r"%\s*then\b", re.IGNORECASE)
_MACRO_ELSE_RE = re.compile(r"%\s*else\b", re.IGNORECASE)
_MACRO_LABEL_RE = re.compile(r"%[A-Za-z_]\w*\s*:")
_IF_RE = re.compile(r"if\b", re.IGNORECASE)
_THEN_RE = re.compile(r"\bthen\b", re.IGNORECASE)
_ELSE_RE = re.compile(r"else\b", re.IGNORECASE)
_WHEN_RE = re.compile(r"when\s*\(", re.IGNORECASE)
_OTHERWISE_RE = re.compile(r"otherwise\b", re.IGNORECASE)
_LABEL_RE = re.compile(r"[A-Za-z_]\w*\s*:(?![:=])")
_PARENS_RE = re.compile(r"[()]")


def _group_end(mt: str, open_at: int) -> int:
    """Index just past the parenthesis closing the one at *open_at*; the end
    of *mt* when none does. Strings are blanked in *mt*, so every parenthesis
    left is code."""
    depth = 0
    for m in _PARENS_RE.finditer(mt, open_at):
        depth += 1 if m.group() == "(" else -1
        if depth == 0:
            return m.end()
    return len(mt)


def _core_start(mt: str, context: str) -> int:
    """Where the statement in *mt* proper begins (see :class:`Statement`);
    -1 when it has none: a subsetting IF, a ``%IF`` without ``%THEN``."""
    pos = _ws_end(mt, 0)
    while True:
        if mt.startswith("%", pos):
            if m := _MACRO_IF_RE.match(mt, pos):
                then = _MACRO_THEN_RE.search(mt, m.end())
                if then is None:
                    return -1
                pos = _ws_end(mt, then.end())
            elif m := _MACRO_ELSE_RE.match(mt, pos) or _MACRO_LABEL_RE.match(mt, pos):
                pos = _ws_end(mt, m.end())
            else:
                return pos
        elif context != DATA:
            return pos
        elif m := _IF_RE.match(mt, pos):
            then = _THEN_RE.search(mt, m.end())
            if then is None:
                return -1
            pos = _ws_end(mt, then.end())
        elif m := _WHEN_RE.match(mt, pos):
            pos = _ws_end(mt, _group_end(mt, m.end() - 1))
        elif m := (
            _ELSE_RE.match(mt, pos) or _OTHERWISE_RE.match(mt, pos) or _LABEL_RE.match(mt, pos)
        ):
            pos = _ws_end(mt, m.end())
        else:
            return pos


def statements_of(
    units: Sequence[_Unit], mt: str, cf: str, *, macro_body: bool = False
) -> Iterator[Statement]:
    """The statements of the CODE *units*, with where each stands.

    *mt* and *cf* are the sanitised texts of the region the units tile, so a
    unit's statement is the slice at its offset. Steps open and close by the
    scanner's rules (:meth:`~chunker.chunker.SasSemanticChunker._collect_block`):
    a DATA or PROC header opens one, RUN, RUN CANCEL or QUIT ends it, a PROC
    that runs in groups ends only at QUIT, and PROC DS2's own DATA programs
    stay inside it. *macro_body*: every statement is in a ``%MACRO`` body,
    as in a split slice of one that lost its header.
    """
    context, proc, run_groups, depth = OPEN, "", False, 0
    base = units[0].start if units else 0
    for unit in units:
        if unit.role is not UnitRole.CODE:
            continue
        a = unit.start - base
        b = a + len(unit.text)
        in_macro = macro_body or depth > 0
        # A macro body's statement outside any step it opens may run inside
        # its caller's DATA step (see _fragment_refs), so it loses its IF …
        # THEN the same way.
        core = _core_start(mt[a:b], DATA if in_macro and context == OPEN else context)
        if core < 0:
            continue
        s_mt, s_cf = mt[a + core : b], cf[a + core : b]
        word_m = _WORD_RE.match(s_mt)
        if word_m is None:
            keyword = ""
        else:
            keyword = ("%" if word_m.group(1) else "") + word_m.group(2).lower()
        step_end = False
        if keyword == "run" or keyword == "quit":
            if context != OPEN and (ended := _STEP_END_RE.fullmatch(_norm(s_mt))):
                step_end = not (run_groups and ended.group(1))
        elif keyword == "%mend":
            depth = max(0, depth - 1)
            context, run_groups = OPEN, False
        elif keyword == "data" or keyword == "proc" or keyword == "%macro":
            normed = _norm(s_mt)
            kind = _classify_normed(normed)
            if kind is SasChunkKind.MACRO_DEFINITION:
                depth += 1
                context, run_groups = OPEN, False
            elif kind is SasChunkKind.DATA_STEP and not (context == PROC and proc == "ds2"):
                # A DATA step ends at its RUN, whatever PROC ran before it.
                context, proc, run_groups = DATA, "", False
            elif kind is SasChunkKind.PROC_STEP:
                name = _PROC_NAME_RE.match(normed)
                context, proc = PROC, name.group(1) if name else ""
                run_groups = proc in RUN_GROUP_PROCS
        yield Statement(s_mt, s_cf, keyword, context, proc, macro_body or depth > 0)
        if step_end:
            context, proc, run_groups = OPEN, "", False


# ---------------------------------------------------------------------------
# Operands
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(DS_REF_TOKEN)
_MACRO_CALL_HEAD_RE = re.compile(r"%\s*[A-Za-z_]\w*")
_NUMBERED_RE = re.compile(r"^(.*?)(\d+)$")
# A numbered range list never expands past this many datasets: a range is a
# convenience for a handful, and a typo must not mint a million names.
_MAX_RANGE = 1000


@dataclass(frozen=True, slots=True)
class _Operand:
    raw: str  # as written
    name: str  # canonical
    pattern: bool = False


def _token_operand(
    raw: str, *, pattern: bool = False, library: str = "work"
) -> _Operand | None:
    name = _ds_name(raw)
    if name is None:
        return None
    if pattern:
        return _Operand(raw + ":", _canon_ds(name, library) + ":", pattern=True)
    return _Operand(raw, _canon_ds(name, library))


def _numbered_range(first: str, last: str) -> list[str] | None:
    """The names ``first``-``last`` spans (``ds1``-``ds3``: ds1, ds2, ds3),
    when both bounds share a library and a stem and end in numbers. Bounds
    written to one width keep it: ``m01``-``m10`` is m01 … m10."""
    a, b = _NUMBERED_RE.match(first), _NUMBERED_RE.match(last)
    if a is None or b is None or a.group(1).lower() != b.group(1).lower():
        return None
    lo, hi = int(a.group(2)), int(b.group(2))
    if hi < lo or hi - lo >= _MAX_RANGE:
        return None
    width = len(a.group(2)) if len(a.group(2)) == len(b.group(2)) else 0
    return [f"{a.group(1)}{i:0{width}d}" for i in range(lo, hi + 1)]


def _operands(
    mt: str, cf: str, pos: int, *, limit: int = 0, library: str = "work"
) -> list[_Operand]:
    """The datasets a statement lists from *pos*: names with their options,
    quoted paths, numbered ranges (``ds1-ds3``) and prefix lists (``pre:``).

    The list ends at an option (``end=``, ``key=``, ``nobs=``, …), a ``/``,
    or the statement's end; a macro call in it (``set %list(lib);``) names
    nothing it can see. *limit*: at most that many (an option's value).
    *library*: where a one-level name lives.
    """
    found: list[_Operand] = []
    n = len(mt)
    while not limit or len(found) < limit:
        pos = _ws_end(mt, pos)
        if pos >= n:
            break
        c = mt[pos]
        if c == "(":  # the options of the dataset before
            pos = _group_end(mt, pos)
            continue
        if c in "'\"":
            close = mt.find(c, pos + 1)
            if close == -1:
                break
            found.append(_Operand(cf[pos : close + 1], _quoted_path(cf[pos : close + 1])))
            pos = close + 1
            if pos < n and mt[pos] in "nN":  # a name literal: 'my data'n
                pos += 1
            continue
        if c == "%" and (call := _MACRO_CALL_HEAD_RE.match(mt, pos)):
            pos = _ws_end(mt, call.end())
            if pos < n and mt[pos] == "(":
                pos = _group_end(mt, pos)
            continue
        if c == "-" and found:  # a numbered range: ds1-ds3
            last = _TOKEN_RE.match(mt, _ws_end(mt, pos + 1))
            if last is None:
                break
            pos = last.end()
            spanned = _numbered_range(found[-1].raw, cf[last.start() : last.end()])
            for raw in spanned[1:] if spanned else [cf[last.start() : last.end()]]:
                if op := _token_operand(raw, library=library):
                    found.append(op)
            continue
        token = _TOKEN_RE.match(mt, pos)
        if token is None:
            break
        after = _ws_end(mt, token.end())
        if after < n and mt[after] == "=":  # an option: the datasets are done
            break
        pos = token.end()
        prefix = pos < n and mt[pos] == ":"
        if prefix:
            pos += 1
        raw = cf[token.start() : token.end()]
        if op := _token_operand(raw, pattern=prefix, library=library):
            found.append(op)
    return found


# ---------------------------------------------------------------------------
# What each statement names
# ---------------------------------------------------------------------------

READ, WRITE, UPDATE = DatasetRole.READ, DatasetRole.WRITE, DatasetRole.UPDATE
DROP = DatasetRole.DROP

# A hash object's DATASET: argument — the dataset a DECLARE (or _NEW_) loads at
# instantiation, or the one an OUTPUT method writes. The name sits inside a
# quoted literal, so it is read from cf, never mt; an unquoted argument (a
# variable, an expression) is not a name and is skipped.
_HASH_DATASET_ARG_RE = re.compile(
    r"\bdataset\s*:\s*(['\"])\s*([^'\"]+?)\s*\1",
    re.IGNORECASE,
)
_HASH_OUTPUT_RE = re.compile(r"\.\s*output\s*\(", re.IGNORECASE)
_DS_TOKEN_FULL_RE = re.compile(rf"{DS_REF_TOKEN}\Z")

# A statement that assigns — `out = x + 1;` in PROC PHREG, NLMIXED, FCMP or
# IML — names no dataset, whatever its variable is called.
_ASSIGNMENT_RE = re.compile(r"[A-Za-z_]\w*\s*(?:\[[^\]]*\]|\{[^}]*\}|\([^)]*\))?\s*=(?!=)")

# What one statement names: a dataset, what the statement does to it, and the
# statement or option that named it.
_Ref = tuple[_Operand, DatasetRole, str]


def _hash_refs(st: Statement) -> Iterator[_Ref]:
    if "dataset" not in st.cf.lower():
        return
    role = WRITE if _HASH_OUTPUT_RE.search(st.mt) else READ
    for m in _HASH_DATASET_ARG_RE.finditer(st.cf):
        raw = m.group(2).split("(", 1)[0].strip()
        if _DS_TOKEN_FULL_RE.match(raw) and (op := _token_operand(raw)):
            yield op, role, "hash"


def _data_step_refs(st: Statement) -> Iterator[_Ref]:
    after = len(st.keyword)
    if st.keyword in ("data", "set", "merge", "update", "output"):
        role = WRITE if st.keyword in ("data", "output") else READ
        for op in _operands(st.mt, st.cf, after):
            yield op, role, st.keyword
    elif st.keyword == "modify":
        # MODIFY rewrites its master in place; the transaction is read.
        for i, op in enumerate(_operands(st.mt, st.cf, after)):
            yield op, UPDATE if i == 0 else READ, "modify"
    yield from _hash_refs(st)


def _sql_refs(st: Statement, start: int = 0, end: int | None = None) -> Iterator[_Ref]:
    """The tables a SAS SQL statement names — PROC SQL, PROC FEDSQL, or the
    query in ``st.cf[start:end]`` — by the one SQL grammar."""
    for raw, role, via in SqlStatement(st.cf[start:end], Dialect.SAS).refs():
        if raw[0] in "'\"":  # a physical path, or a name literal: 'my data'n
            quoted = raw[:-1] if raw[-1] in "nN" else raw
            yield _Operand(raw, _quoted_path(quoted)), role, via
        elif op := _token_operand(raw):
            yield op, role, via


# PROC DS2. A DATA program writes its tables; SET and MERGE read theirs, a
# `{select …}` query's included. `set from th;` reads a THREAD program's rows,
# not a table.
_SET_FROM_THREAD_RE = re.compile(r"from\b", re.IGNORECASE)


def _ds2_refs(st: Statement) -> Iterator[_Ref]:
    kw, after = st.keyword, len(st.keyword)
    if kw == "data":
        for op in _operands(st.mt, st.cf, after):
            yield op, WRITE, "data"
    elif kw in ("set", "merge"):
        at = _ws_end(st.mt, after)
        if st.mt.startswith("{", at):
            close = st.mt.find("}", at)
            yield from _sql_refs(st, at + 1, close if close >= 0 else None)
        elif not _SET_FROM_THREAD_RE.match(st.mt, at):
            for op in _operands(st.mt, st.cf, after):
                yield op, READ, kw


# PROC IML: USE opens a dataset to read, EDIT to read and change, CREATE a new
# one; its name comes first (`create out from m;`, `use a var {x};`). APPEND,
# READ and CLOSE work on one already open.
_IML_ROLES = {"use": READ, "edit": UPDATE, "create": WRITE}


def _iml_refs(st: Statement) -> Iterator[_Ref]:
    role = _IML_ROLES.get(st.keyword)
    if role is not None:
        for op in _operands(st.mt, st.cf, len(st.keyword), limit=1):
            yield op, role, st.keyword


# ---------------------------------------------------------------------------
# PROC options
# ---------------------------------------------------------------------------

_ROLES = {"read": READ, "write": WRITE, "update": UPDATE}
_NAMED_OPTION_RE = re.compile(r"\b([A-Za-z_]\w*)\s*=(?!=)")
# Statements whose options may stand anywhere in them, as a PROC statement's
# do: `output out=s mean=m;`, `score data=new out=scored;`. Any other
# statement's options follow its `/` (`tables g / out=cnt;`, `model y = x /
# outroc=r;`); before it come variables, labels and conditions, so `label out
# = 'x';` and `where data = 1;` name nothing.
_OPTION_STATEMENTS = frozenset({"proc", "output", "score", "baseline", "forecast"})


@cache
def _option_roles(proc: str) -> Mapping[str, str]:
    """What *proc*'s options name: keywords.PROC_OPTION_DEFAULTS, with the
    PROC's own keywords.PROC_OPTION_ROLES over them."""
    return {**PROC_OPTION_DEFAULTS, **PROC_OPTION_ROLES.get(proc, {})}


def _blank_groups(text: str) -> str:
    """*text* with every parenthesised group blanked, parentheses included: a
    dataset's own options (``data=a(where=(out=1))``) are not the statement's."""
    if "(" not in text:
        return text
    parts: list[str] = []
    depth = last = 0
    for m in _PARENS_RE.finditer(text):
        if m.group() == "(":
            if depth == 0:
                parts.append(text[last : m.start()])
                last = m.start()
            depth += 1
        elif depth:
            depth -= 1
            if depth == 0:
                parts.append(" " * (m.end() - last))
                last = m.end()
    parts.append(" " * (len(text) - last) if depth else text[last:])
    return "".join(parts)


def _option_value(st: Statement, names: Collection[str]) -> str | None:
    """The token the first of the options *names* is set to, lowercased."""
    for m in _NAMED_OPTION_RE.finditer(_blank_groups(st.mt)):
        if m.group(1).lower() in names:
            token = _TOKEN_RE.match(st.mt, _ws_end(st.mt, m.end()))
            return st.cf[token.start() : token.end()].lower() if token else None
    return None


def _option_refs(
    st: Statement,
    roles: Mapping[str, str],
    start: int,
    end: int | None = None,
    *,
    library: str = "work",
) -> Iterator[_Ref]:
    """The datasets the ``name=`` options in ``st.mt[start:end]`` name, by
    *roles* (see keywords.PROC_OPTION_DEFAULTS). ``lib._all_`` is every member
    of lib: the pattern ``lib.:``."""
    stop = len(st.mt) if end is None else end
    for m in _NAMED_OPTION_RE.finditer(_blank_groups(st.mt[start:stop])):
        option = m.group(1).lower()
        role = roles.get(option)
        dataset_role = WRITE if role == "package" else _ROLES.get(role or "")
        if dataset_role is None:
            continue  # no option of this PROC's, or a libref or fileref
        at = _ws_end(st.mt, start + m.end())
        token = _TOKEN_RE.match(st.mt, at)
        written = st.cf[token.start() : token.end()].lower() if token else ""
        if written == "_all_" or written.endswith("._all_"):
            lib = written[:-6] or library
            yield _Operand(written, _member(lib, ":"), pattern=True), dataset_role, f"{option}="
            continue
        for op in _operands(st.mt, st.cf, at, limit=1, library=library):
            if role == "package" and op.name.count(".") >= 2:
                # FCMP's lib.member.package: the package lives in lib.member.
                op = _Operand(op.raw, _canon_ds(op.name.rsplit(".", 1)[0]))
            yield op, dataset_role, f"{option}="


def _statement_option_refs(
    st: Statement, roles: Mapping[str, str], *, library: str = "work"
) -> Iterator[_Ref]:
    """The datasets a PROC's statement names by its options: anywhere in one
    of _OPTION_STATEMENTS, after the ``/`` in any other, never in an
    assignment."""
    if "=" not in st.mt or _ASSIGNMENT_RE.match(st.mt):
        return iter(())
    if st.keyword in _OPTION_STATEMENTS:
        return _option_refs(st, roles, len(st.keyword), library=library)
    slash = _blank_groups(st.mt).find("/")
    if slash < 0:
        return iter(())
    return _option_refs(st, roles, slash + 1, library=library)


# `ods output Summary=s TTests(persist=proc)=t;`: each output object's dataset,
# which the PROC that produces the object writes.
_ODS_OUTPUT_RE = re.compile(r"ods\s+output\b", re.IGNORECASE)


def _ods_output_refs(st: Statement) -> Iterator[_Ref]:
    head = _ODS_OUTPUT_RE.match(st.mt)
    if head is None:
        return
    for m in _NAMED_OPTION_RE.finditer(_blank_groups(st.mt[head.end() :])):
        for op in _operands(st.mt, st.cf, head.end() + m.end(), limit=1):
            yield op, WRITE, "ods_output"


# ---------------------------------------------------------------------------
# PROC steps
# ---------------------------------------------------------------------------

# The libraries PROC COPY copies between, and UPLOAD and DOWNLOAD when they
# copy a library rather than one dataset.
_COPY_LIBRARIES = {
    "copy": ("in", "out"),
    "upload": ("inlib", "outlib"),
    "download": ("inlib", "outlib"),
}
# A bare word option, never an option's value: `kill` in `proc datasets lib=x
# kill;`, not in `lib=kill`.
_KILL_RE = re.compile(r"(?:^|[^=\s])\s+kill\b", re.IGNORECASE)
_MOVE_RE = re.compile(r"(?:^|[^=\s])\s+move\b", re.IGNORECASE)
# CHANGE and EXCHANGE: `old=new` pairs of member names.
_PAIR_RE = re.compile(rf"({DS_REF_TOKEN})\s*=\s*({DS_REF_TOKEN})")
# The PROCs whose statements are SQL.
_SQL_PROCS = frozenset({"sql", "fedsql"})
# PROC DATASETS statements that name members of its library.
_DATASETS_STATEMENTS = frozenset(
    {"append", "change", "exchange", "copy", "delete", "modify", "age", "contents"}
)


class _CopyGroup:
    """Members copied from one library to another: PROC COPY (UPLOAD,
    DOWNLOAD) or PROC DATASETS's COPY statement. SELECT names them; without it
    (or with EXCLUDE, naming the ones left behind) every member goes: the
    patterns ``source.:`` and ``target.:``. MOVE deletes them from the source.
    """

    __slots__ = ("source", "target", "move", "selected")

    def __init__(self, source: str, target: str, move: bool) -> None:
        self.source, self.target, self.move = source, target, move
        self.selected = False

    def _copied(self, member: str, pattern: bool) -> Iterator[_Ref]:
        yield _Operand(member, _member(self.source, member), pattern), READ, "copy"
        yield _Operand(member, _member(self.target, member), pattern), WRITE, "copy"
        if self.move:
            yield _Operand(member, _member(self.source, member), pattern), DROP, "copy"

    def select(self, st: Statement) -> Iterator[_Ref]:
        self.selected = True
        for op in _operands(st.mt, st.cf, len(st.keyword)):
            yield from self._copied(op.raw.lower(), op.pattern)

    def close(self) -> Iterator[_Ref]:
        if not self.selected:
            yield from self._copied(":", True)


class _ProcStep:
    """One PROC step's statements, read in order: each statement's options by
    what they name in this PROC, PROC SORT's in-place rewrite, PROC DATASETS's
    member statements, a library copy's members, and ODS OUTPUT."""

    __slots__ = ("proc", "roles", "library", "copy")

    def __init__(self, proc: str) -> None:
        self.proc = proc
        self.roles = _option_roles(proc)
        self.library = "work"  # where PROC DATASETS's member names live
        self.copy: _CopyGroup | None = None

    def read(self, st: Statement) -> Iterator[_Ref]:
        kw = st.keyword
        # SELECT and EXCLUDE belong to the copy just before them.
        if self.copy is not None and kw not in ("select", "exclude"):
            yield from self.close()
        if kw == "ods":
            yield from _ods_output_refs(st)
        elif kw == "proc":
            yield from self._proc_statement(st)
        elif self.proc in _SQL_PROCS:
            yield from _sql_refs(st)
        elif self.proc == "ds2":
            yield from _ds2_refs(st)
        elif self.proc == "iml":
            yield from _iml_refs(st)
        elif self.copy is not None:
            if kw == "select":
                yield from self.copy.select(st)
        elif self.proc == "datasets" and kw in _DATASETS_STATEMENTS:
            yield from self._datasets_statement(st)
        else:
            yield from _statement_option_refs(st, self.roles)

    def close(self) -> Iterator[_Ref]:
        """What the step names once it ends: a copy without SELECT."""
        if self.copy is not None:
            copy, self.copy = self.copy, None
            yield from copy.close()

    def _proc_statement(self, st: Statement) -> Iterator[_Ref]:
        refs = list(_statement_option_refs(st, self.roles))
        if self.proc == "sort" and all(via != "out=" for _, _, via in refs):
            # Without OUT= the sort replaces DATA= with a sorted copy: it reads
            # the table and creates its new version, as `data x; set x;` does.
            # Not an UPDATE, which supplies nothing: a later BY step reads the
            # sorted table, so it depends on the sort.
            refs += [(op, WRITE, via) for op, role, via in refs if via == "data="]
        yield from refs
        flags = _blank_groups(st.mt)
        if self.proc == "datasets":
            self.library = _option_value(st, ("library", "lib")) or "work"
            if _KILL_RE.search(flags):
                yield _Operand("kill", _member(self.library, ":"), True), DROP, "kill"
        elif self.proc in _COPY_LIBRARIES:
            source_option, target_option = _COPY_LIBRARIES[self.proc]
            source = _option_value(st, (source_option,))
            target = _option_value(st, (target_option,))
            if source and target:
                self.copy = _CopyGroup(source, target, bool(_MOVE_RE.search(flags)))

    def _datasets_statement(self, st: Statement) -> Iterator[_Ref]:
        kw, lib, after = st.keyword, self.library, len(st.keyword)
        if kw == "append":
            yield from _option_refs(st, _option_roles("append"), after, library=lib)
        elif kw == "contents":
            yield from _option_refs(st, {"data": "read"}, after, library=lib)
            yield from _option_refs(st, {"out": "write", "out2": "write"}, after)
        elif kw == "copy":
            source = _option_value(st, ("in",)) or lib
            target = _option_value(st, ("out",))
            if target:
                self.copy = _CopyGroup(source, target, bool(_MOVE_RE.search(st.mt)))
        elif kw == "delete":
            for op in _operands(st.mt, st.cf, after, library=lib):
                yield op, DROP, "delete"
        elif kw in ("modify", "age"):
            limit = 1 if kw == "modify" else 0
            for op in _operands(st.mt, st.cf, after, limit=limit, library=lib):
                yield op, UPDATE, kw
        else:  # change old=new; exchange a=b
            scan = _blank_groups(st.mt)
            slash = scan.find("/")
            for m in _PAIR_RE.finditer(scan, after, slash if slash >= 0 else len(scan)):
                old = _token_operand(st.cf[m.start(1) : m.end(1)], library=lib)
                new = _token_operand(st.cf[m.start(2) : m.end(2)], library=lib)
                if old is None or new is None:
                    continue
                if kw == "change":  # old's rows become new's, and old is gone
                    yield old, READ, kw
                    yield old, DROP, kw
                    yield new, WRITE, kw
                else:
                    yield old, UPDATE, kw
                    yield new, UPDATE, kw


# ---------------------------------------------------------------------------
# Outside a PROC
# ---------------------------------------------------------------------------

# A %MACRO body may hold part of a step, for a call inside one to complete:
# `%macro sets; set a b; %mend;` (a DATA step's), `%macro src; select * from
# &t; %mend;` (PROC SQL's). Its statements outside any step it opens are read
# as what their keyword makes them.
_SQL_KEYWORDS = frozenset({"select", "create", "insert", "delete"})
_DATA_KEYWORDS = frozenset({"set", "merge", "update", "modify", "output"})
# UPDATE as PROC SQL writes it, not the DATA step's: `update t <as a> set c = …`.
_SQL_UPDATE_RE = re.compile(
    rf"update\s+{DS_REF_TOKEN}(?:\s+(?:as\s+)?[A-Za-z_]\w*)?\s+set\b", re.IGNORECASE
)


def _fragment_refs(st: Statement) -> Iterator[_Ref]:
    if st.keyword in _SQL_KEYWORDS or (
        st.keyword == "update" and _SQL_UPDATE_RE.match(st.mt)
    ):
        return _sql_refs(st)
    # OUTPUT with options is a PROC's (`output out=stats mean=m`).
    if st.keyword in _DATA_KEYWORDS and not (st.keyword == "output" and "=" in st.mt):
        return _data_step_refs(st)
    return chain(_statement_option_refs(st, _option_roles("")), _hash_refs(st))


def _call_option_refs(st: Statement) -> Iterator[_Ref]:
    """A macro call's arguments, read as a PROC's options are (by the default
    roles): ``%step1(data=a, out=b)`` reads a and writes b."""
    open_at = st.mt.find("(")
    if open_at < 0:
        return iter(())
    return _option_refs(st, _option_roles(""), open_at + 1, _group_end(st.mt, open_at) - 1)


def _statement_refs(st: Statement) -> Iterator[_Ref]:
    """What a statement outside a PROC step names (a PROC's go through
    :class:`_ProcStep`)."""
    if st.context == DATA:
        return _data_step_refs(st)
    if st.keyword == "ods":
        # A request the next PROC fulfils: in open code it waits on the ODS
        # statement for metadata.resolve_ods_outputs to move; in a macro body
        # the body's PROC writes it.
        return _ods_output_refs(st)
    if not st.in_macro:
        return iter(())
    if not st.keyword.startswith("%"):
        return _fragment_refs(st)
    if st.keyword[1:] not in _MACRO_LANGUAGE_WORDS:
        # A call of another macro: its DATA= and OUT= arguments. The batcher
        # binds the parameters of the macro a job calls, never of the ones it
        # calls in turn, so this is how a wrapper (`%step1(data=a, out=b);`)
        # is seen to read and write.
        return _call_option_refs(st)
    return iter(())


# ---------------------------------------------------------------------------
# Macro bodies
# ---------------------------------------------------------------------------

# What a macro body's dataset reference turns out to be.
_REF_LITERAL = "literal"  # no macro reference; the name is written out
_REF_PARAM = "param"  # exactly this macro's parameter, resolved per call site
_REF_CALL_SITE = "call_site"  # built from parameters, so only a call site knows it
_REF_MACRO_VAR = "macro_var"  # macro variables, none of them this macro's own

_VAR_REF_RE = re.compile(r"&(\w+)\.?")


# A reference that is one macro variable and nothing else: `&ds`, `&ds.`.
_WHOLE_PARAM_RE = re.compile(r"&[A-Za-z_]\w*\.?")


def _classify_ref(raw: str, param_pos: Mapping[str, int]) -> tuple[str, str]:
    """
    Classify a raw dataset reference extracted from a macro body.

    Returns ``(key, kind)``, where *kind* is one of :data:`_REF_LITERAL`,
    :data:`_REF_PARAM`, :data:`_REF_CALL_SITE` or :data:`_REF_MACRO_VAR` and
    *key* is the lowercased parameter name for :data:`_REF_PARAM`, or the
    lowercased reference exactly as written for everything else.

    The distinction that matters is between a name the *call site* supplies
    and one it does not.  ``&lib..&prefix._&suffix.`` is assembled from three
    parameters, so no name exists until a call is made and this module never
    fabricates one from the pieces (:data:`_REF_CALL_SITE`).  ``&reporting_lib``
    inside the same body names a macro variable the corpus assigns, fixed for
    every call, and :func:`~chunker.metadata.resolve_macro_var_refs` can give
    it a value (:data:`_REF_MACRO_VAR`).
    """
    raw = raw.strip()
    if "&" not in raw:
        return raw.lower(), _REF_LITERAL
    refs = [r.lower() for r in _VAR_REF_RE.findall(raw)]
    # A quoted path is never the parameter itself: "&dir/x" is built from it.
    if len(refs) == 1 and refs[0] in param_pos and raw[0] not in "'\"":
        return refs[0], _REF_PARAM
    if any(r in param_pos for r in refs):
        return raw.lower(), _REF_CALL_SITE
    return raw.lower(), _REF_MACRO_VAR


def _body_ref(
    op: _Operand, role: DatasetRole, via: str, param_pos: Mapping[str, int]
) -> SasDatasetRef | None:
    """The macro-body reference *op* makes, by how its name is spelled; None
    for one built from the macro's parameters, which no call names whole."""
    key, kind = _classify_ref(op.raw, param_pos)
    if kind == _REF_CALL_SITE:
        return None
    if kind == _REF_PARAM:
        # The parameter itself (`&ds`), or a name built around it
        # (`&lib..customers`), which keeps its spelling for the call site to
        # fill in: the parameter's argument alone would name another dataset.
        whole = _WHOLE_PARAM_RE.fullmatch(op.raw.strip()) is not None
        return SasDatasetRef(
            f"&{key}" if whole else op.name,
            role,
            raw=op.raw,
            via=via,
            in_macro_body=True,
            param=key,
            param_pos=param_pos[key],
            pattern=op.pattern,
        )
    return SasDatasetRef(
        op.name, role, raw=op.raw, via=via, in_macro_body=True, pattern=op.pattern
    )


# ---------------------------------------------------------------------------
# A region's references
# ---------------------------------------------------------------------------


def dataset_refs(
    units: Sequence[_Unit],
    mt: str,
    cf: str,
    *,
    macro_body: bool = False,
    param_pos: Mapping[str, int] | None = None,
) -> list[SasDatasetRef]:
    """Every dataset reference the region's statements make, in source order.

    *mt* and *cf* are the region's sanitised texts (with SQL pass-through
    blanked: native SQL names no SAS dataset). In a ``%MACRO`` body — every
    statement when *macro_body* — references are classified by spelling
    against the macro's parameters, *param_pos* (name → position, -1 for a
    keyword parameter).
    """
    params = param_pos or {}
    refs: list[SasDatasetRef] = []

    def add(found: Iterable[_Ref], in_macro: bool) -> None:
        for op, role, via in found:
            if in_macro:
                if (ref := _body_ref(op, role, via, params)) is not None:
                    refs.append(ref)
            else:
                refs.append(
                    SasDatasetRef(op.name, role, raw=op.raw, via=via, pattern=op.pattern)
                )

    def close_data_step(start: int) -> None:
        # MODIFY updates its master where it stands: the DATA statement names
        # that table, and OUTPUT adds rows to it, so neither creates it.
        masters = {
            r.name for r in refs[start:] if r.role is UPDATE and r.via == "modify"
        }
        if masters:
            refs[start:] = [
                r
                for r in refs[start:]
                if not (r.role is WRITE and r.name in masters and r.via in ("data", "output"))
            ]

    # A PROC step is read by one _ProcStep, from its PROC statement to the
    # statement that ends it: what one statement names can depend on another's
    # (PROC DATASETS's LIB=, PROC COPY's SELECT).
    step: _ProcStep | None = None
    step_in_macro = False
    group_start = 0  # where the open PROC step's (or run group's) refs begin
    data_start: int | None = None  # where the open DATA step's refs begin
    modifies = False  # and whether it has a MODIFY statement
    for st in statements_of(units, mt, cf, macro_body=macro_body):
        if data_start is not None and (st.context != DATA or st.keyword == "data"):
            if modifies:
                close_data_step(data_start)
            data_start, modifies = None, False
        if step is not None and (st.context != PROC or st.keyword == "proc"):
            add(step.close(), step_in_macro)
            step = None
        # RUN CANCEL: SAS compiles the step (or a run-group PROC's current
        # group) and runs none of it, so it reads and writes nothing.
        cancel = st.keyword == "run" and _norm(st.mt).startswith("run cancel")
        if st.context == PROC:
            if step is None:
                step, step_in_macro = _ProcStep(st.proc), st.in_macro
                group_start = len(refs)
            if cancel:
                del refs[group_start:]
                step = _ProcStep(st.proc)  # and nothing pending at its close
            else:
                add(step.read(st), st.in_macro)
                if st.keyword == "run":
                    group_start = len(refs)
        else:
            if st.context == DATA:
                if data_start is None:
                    data_start = len(refs)
                modifies = modifies or st.keyword == "modify"
                if cancel:
                    del refs[data_start:]
                    continue
            add(_statement_refs(st), st.in_macro)
    if data_start is not None and modifies:
        close_data_step(data_start)
    if step is not None:
        add(step.close(), step_in_macro)
    return refs
