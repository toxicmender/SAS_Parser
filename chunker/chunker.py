"""
chunker.py — dependency-light semantic chunker for Base SAS source files.

Recognised constructs (Base SAS Programming Reference)
-------------------------------------------------------
DATA step, PROC step, %MACRO/%MEND, %INCLUDE, LIBNAME/FILENAME/TITLE/FOOTNOTE,
OPTIONS, ODS, FORMAT/INFORMAT standalone, %LET/%PUT/%GLOBAL/%LOCAL, %macro_call.

All imports are stdlib + pydantic (via local models.py).

Key design rule — block collection
-----------------------------------
FORMAT, LABEL, OPTIONS, LIBNAME, ODS, and other statement keywords that appear
*inside* a DATA or PROC block body are legal SAS statements within that block
and must NOT terminate the block early.  Only a new DATA, PROC, or %MACRO
header, or an explicit RUN/QUIT, closes the current block.

Stuck-parser protection
------------------------
Two independent safeguards keep a pathological input from hanging the parser
forever (see ``SasSemanticChunker(timeout=...)``):

  * A wall-clock **deadline** is checked at statement/region boundaries in the
    scan and grouping loops.  When it is exceeded the parser stops at the next
    boundary, logs where it stopped, emits a ``PARSER_TIMEOUT`` diagnostic and
    returns the *partial* result collected so far — a graceful exit rather than
    an unbounded spin on, say, a multi-megabyte generated file.
  * A background **watchdog** thread logs an escalating trail (WARNING → ERROR)
    naming the phase and elapsed time whenever a parse overruns the timeout.
    This is the only safeguard that fires when the parser is wedged inside a
    single un-interruptible C-level regex call (catastrophic backtracking on
    hostile source): pure-Python code cannot preempt that call, but the
    watchdog still guarantees the stuck phase is named in the logs, so the hang
    is diagnosable even if the process must ultimately be killed.

Logging
-------
Logger: ``chunker.chunker``

  Level    When emitted
  -------  ---------------------------------------------------------------
  DEBUG    Per-unit / per-region decisions (very verbose; off in prod)
  INFO     File-level start/finish, oversized-split decisions, elapsed time
  WARNING  Unclosed blocks, unterminated statements, unrecognised regions;
           watchdog "parse still running / appears stuck" notices
  ERROR    File-not-found (logged then re-raised); parse-deadline exceeded
           (partial result returned); repeated watchdog "appears stuck" notices
"""

from __future__ import annotations

import logging
import re
import threading
import time
from bisect import bisect_right
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import regex

from .models import (
    SasChunk,
    SasChunkKind,
    SasChunkMetadata,
    SasChunkResult,
    SasDiagnostic,
    SasDiagnosticSeverity,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex catalogue (mirrors the Reference Sheet grammar)
# ---------------------------------------------------------------------------

_DATASET_RE = re.compile(
    r"\b(?:data|set|merge|update|modify)\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)",
    re.IGNORECASE,
)
_DATA_OPT_RE = re.compile(
    r"\bdata\s*=\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)",
    re.IGNORECASE,
)
_LIBREF_RE = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")
_MACRO_DEF_RE = re.compile(r"%\s*macro\s+([A-Za-z_]\w*)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Reserved words — SAS Macro Language: Reference, Appendix 1
# (Macro Facility Word Rules / Reserved Words, pp. 495-496)
#
# None of these words can validly be a user-defined macro name.  Any
# "%word" appearing in source text where word is one of these is always a
# macro-language keyword/statement/function, never an invocation of a
# corpus-local macro — so every regex that detects "is this a macro call"
# must exclude all of them, not just the small hand-picked subset that
# earlier testing happened to surface.
#
# A handful of these (CMS, TSO — mainframe operating-environment words;
# EDIT, SAVE, PAUSE, OPEN, CLOSE, CLEAR, ACT, ACTIVATE, DEACT, DEL, DELETE,
# DMIDSPLY, DMISPLIT, COMANDR, METASYM, LIST, LISTM, WINDOW, DISPLAY,
# INPUT, INC, INFILE, FILE, ON — interactive Display-Manager command
# words) are essentially dead in modern batch/Compute-Server SAS, but are
# kept in the set for correctness since excluding them costs nothing and
# a SAS program could legally (if unusually) attempt to invoke one.
# ---------------------------------------------------------------------------
_RESERVED_WORDS = frozenset(
    {
        "abend",
        "abort",
        "act",
        "activate",
        "bquote",
        "by",
        "clear",
        "close",
        "cms",
        "comandr",
        "copy",
        "deact",
        "del",
        "delete",
        "display",
        "dmidsply",
        "dmisplit",
        "do",
        "edit",
        "else",
        "end",
        "eval",
        "file",
        "global",
        "go",
        "goto",
        "if",
        "inc",
        "include",
        "index",
        "infile",
        "input",
        "kcmpres",
        "kindex",
        "kleft",
        "klength",
        "kscan",
        "ksubstr",
        "ktrim",
        "kupcase",
        "length",
        "let",
        "list",
        "listm",
        "local",
        "macro",
        "mend",
        "metasym",
        "nrbquote",
        "nrquote",
        "nrstr",
        "on",
        "open",
        "pause",
        "put",
        "qkcmpres",
        "qkleft",
        "qkscan",
        "qksubstr",
        "qktrim",
        "qkupcase",
        "qscan",
        "qsubstr",
        "qsysfunc",
        "quote",
        "qupcase",
        "resolve",
        "return",
        "run",
        "save",
        "scan",
        "stop",
        "str",
        "substr",
        "superq",
        "symdel",
        "symexist",
        "symglobl",
        "symlocal",
        "syscall",
        "sysevalf",
        "sysexec",
        "sysfunc",
        "sysget",
        "sysrput",
        "then",
        "to",
        "tso",
        "unquote",
        "unstr",
        "until",
        "upcase",
        "while",
        "window",
    }
)

# ---------------------------------------------------------------------------
# Additional macro functions — SAS Macro Language: Reference, Ch. 12
# Table 12.3 ("Macro Functions"), pp. 189-210 — ROADMAP Phase 4 (E10).
#
# These five are genuine macro functions per Ch. 12's own function table,
# but are *not* present in Appendix 1's reserved-word list (verified: all
# other 22 of Table 12.3's 27 function names ARE already covered by
# _RESERVED_WORDS above, purely as a side effect of Appendix 1 happening to
# overlap heavily with the function list — confirmed by exhaustive testing,
# not by assumption). Kept as a separate, clearly-sourced constant rather
# than folded into _RESERVED_WORDS itself, so that constant's identity
# ("Appendix 1, verbatim — 94 words") stays exact and independently
# citable/verifiable, while the *exclusion mechanism* below still covers
# the complete, real macro-function set Ch. 12 documents.
# ---------------------------------------------------------------------------
_ADDITIONAL_MACRO_FUNCTION_WORDS = frozenset(
    {
        "sysmacexec",
        "sysmacexist",
        "sysmexecdepth",
        "sysmexecname",
        "sysprod",
    }
)

# Built once from the union of both reserved-word sources — longest words
# first so the alternation doesn't short-circuit on a shorter word that is
# itself a prefix of a longer one.
_RESERVED_WORDS_PATTERN = "|".join(
    re.escape(w)
    for w in sorted(
        _RESERVED_WORDS | _ADDITIONAL_MACRO_FUNCTION_WORDS,
        key=len,
        reverse=True,
    )
)

# ---------------------------------------------------------------------------
# Standard SAS-provided autocall macros — SAS Macro Language: Reference,
# Ch. 12 Table 12.13 ("Selected Autocall Macros Provided with SAS
# Software") — ROADMAP Phase 5 (F2b).
#
# Unlike the reserved-word sets above, these ARE genuine, callable macro
# names — %left(&var), %trim(&var), etc. are real macro invocations, and
# must still be detected as such by _MACRO_CALL_RE/_MACRO_INVOKE_RE (so
# this set is deliberately NOT folded into _RESERVED_WORDS_PATTERN). The
# distinction this set exists to make is narrower: these ten ship with
# every SAS installation, so a call to one of them will *always* be
# "unresolved" against any user-supplied corpus, even though it's
# perfectly normal, ubiquitous SAS code — not a missing dependency the
# user needs to go find. batcher.py uses this set to exclude these names
# from a batch's `required_macros` (the "you're missing this macro's
# definition" list) while still reporting them separately via
# `SasBatch.standard_autocall_macros`, so the information isn't silently
# dropped — mirrors the existing automatic-macro-variable pattern from
# Phase 1 exactly (tracked separately, never treated as "missing").
#
# Full SASAUTOS directory scanning (F2, resolving *any* externally-defined
# macro by probing `<dir>/<name>.sas` on a search path) and SASMSTORE
# compiled-macro resolution (F3) remain explicitly deferred — see
# MACRO_PARSING_ROADMAP.md Phase 5 for the reasoning.
# ---------------------------------------------------------------------------
_STANDARD_AUTOCALL_MACROS = frozenset(
    {
        "cmpres",
        "qcmpres",
        "left",
        "qleft",
        "trim",
        "qtrim",
        "verify",
        "compstor",
        "datatyp",
        "sysrc",
    }
)

# ---------------------------------------------------------------------------
# SAS DATA-step functions and CALL routines — SAS 9.4 Functions and CALL
# Routines: Reference, Fifth Edition (the "Dictionary of Functions and CALL
# Routines" chapter, and the "Functions and CALL Routines by Category"
# summary table).  Every name below is a documented dictionary entry title
# in that manual, lower-cased and with the ``CALL`` prefix stripped from
# routine names.
#
# Purpose: recognising which built-ins a chunk uses gives an LLM translator
# an at-a-glance inventory of the functions/routines it must map to the
# target language — many of which (INTNX/INTCK date arithmetic, PUT/INPUT
# format application, the PRX* regex family, CALL SYMPUT/EXECUTE, ...) have
# no one-to-one equivalent and need explicit handling.  These are advisory
# metadata only; they never gate chunking or batching decisions.
# ---------------------------------------------------------------------------
_SAS_FUNCTIONS = frozenset(
    {
        'abs', 'addr', 'addrlong', 'airy', 'allcomb', 'allperm', 'anyalnum', 'anyalpha',
        'anycntrl', 'anydigit', 'anyfirst', 'anygraph', 'anylower', 'anyname', 'anyprint',
        'anypunct', 'anyspace', 'anyupper', 'anyxdigit', 'arcos', 'arcosh', 'arsin', 'arsinh',
        'artanh', 'atan', 'atan2', 'attrc', 'attrn', 'band', 'beta', 'betainv', 'blackclprc',
        'blackptprc', 'blkshclprc', 'blkshptprc', 'blshift', 'bnot', 'bor', 'brshift', 'bxor',
        'byte', 'cat', 'catq', 'cats', 'catt', 'catx', 'cdf', 'ceil', 'ceilz', 'cexist', 'char',
        'choosec', 'choosen', 'cinv', 'close', 'cmiss', 'cnonct', 'coalesce', 'coalescec',
        'collate', 'comb', 'compare', 'compbl', 'compfuzz', 'compged', 'complev', 'compound',
        'compress', 'compsrv_oval', 'compsrv_unquote2', 'constant', 'convx', 'convxp', 'cos',
        'cosh', 'cot', 'count', 'countc', 'countw', 'csc', 'css', 'cumipmt', 'cumprinc',
        'curobs', 'cv', 'daccdb', 'daccdbsl', 'daccsl', 'daccsyd', 'dacctab', 'dairy', 'datdif',
        'date', 'datejul', 'datepart', 'datetime', 'day', 'dclose', 'dcreate', 'depdb',
        'depdbsl', 'depsl', 'depsyd', 'deptab', 'dequote', 'deviance', 'dhms', 'dif', 'digamma',
        'dim', 'dinfo', 'distribution', 'divide', 'dnum', 'dopen', 'doptname', 'doptnum',
        'dosubl', 'dread', 'dropnote', 'dsname', 'dsncatlgd', 'dur', 'durp', 'effrate',
        'envlen', 'erf', 'erfc', 'euclid', 'exist', 'exp', 'fact', 'fappend', 'fclose', 'fcol',
        'fcopy', 'fdelete', 'fetch', 'fetchobs', 'fexist', 'fget', 'fileexist', 'fileref',
        'finance', 'find', 'findc', 'findw', 'finfo', 'finv', 'fipname', 'fipnamel', 'fipstate',
        'first', 'floor', 'floorz', 'fmtinfo', 'fnonct', 'fnote', 'fopen', 'foptname',
        'foptnum', 'fpoint', 'fpos', 'fput', 'fread', 'frewind', 'frlen', 'fsep', 'fuzz',
        'fwrite', 'gaminv', 'gamma', 'garkhclprc', 'garkhptprc', 'gcd', 'geodist', 'geomean',
        'geomeanz', 'getvarc', 'getvarn', 'git_branch_chkout', 'git_branch_delete',
        'git_branch_merge', 'git_branch_new', 'git_clone', 'git_commit', 'git_commit_free',
        'git_commit_get', 'git_commit_log', 'git_delete_repo', 'git_diff', 'git_diff_file_idx',
        'git_diff_free', 'git_diff_get', 'git_diff_to_file', 'git_fetch', 'git_index_add',
        'git_index_remove', 'git_init_repo', 'git_pull', 'git_push', 'git_rebase',
        'git_rebase_op', 'git_reset', 'git_reset_file', 'git_stash', 'git_stash_apply',
        'git_stash_drop', 'git_stash_pop', 'git_status', 'git_status_free', 'git_status_get',
        'git_version', 'gitfn_clone', 'gitfn_co_branch', 'gitfn_commit', 'gitfn_commit_get',
        'gitfn_commit_log', 'gitfn_commitfree', 'gitfn_del_branch', 'gitfn_diff',
        'gitfn_diff_free', 'gitfn_diff_get', 'gitfn_diff_idx_f', 'gitfn_idx_add',
        'gitfn_idx_remove', 'gitfn_mrg_branch', 'gitfn_new_branch', 'gitfn_pull', 'gitfn_push',
        'gitfn_reset', 'gitfn_reset_file', 'gitfn_status', 'gitfn_status_get',
        'gitfn_statusfree', 'gitfn_version', 'graycode', 'harmean', 'harmeanz', 'hashing',
        'hashing_file', 'hashing_hmac', 'hashing_hmac_file', 'hashing_hmac_init',
        'hashing_init', 'hashing_part', 'hashing_term', 'hbound', 'hms', 'holiday', 'holidayck',
        'holidaycount', 'holidayname', 'holidaynx', 'holidayny', 'holidaytest', 'hour',
        'htmldecode', 'htmlencode', 'ibessel', 'ifc', 'ifn', 'index', 'indexc', 'indexw',
        'input', 'inputc', 'inputn', 'int', 'intcindex', 'intck', 'intcycle', 'intfit',
        'intfmt', 'intget', 'intindex', 'intnest', 'intnx', 'intrr', 'intseas', 'intshift',
        'inttest', 'intz', 'iorcmsg', 'ipmt', 'iqr', 'irr', 'jbessel', 'juldate', 'juldate7',
        'kurtosis', 'lag', 'largest', 'lbound', 'lcm', 'lcomb', 'left', 'length', 'lengthc',
        'lengthm', 'lengthn', 'lexcomb', 'lexcombi', 'lexperk', 'lexperm', 'lfact', 'lgamma',
        'libname', 'libref', 'log', 'log10', 'log1px', 'log2', 'logbeta', 'logcdf', 'logistic',
        'logpdf', 'logsdf', 'lowcase', 'lperm', 'lpnorm', 'mad', 'margrclprc', 'margrptprc',
        'max', 'md5', 'mdy', 'mean', 'median', 'min', 'minute', 'missing', 'mod', 'module',
        'modulec', 'modulen', 'modz', 'month', 'mopen', 'mort', 'msplint', 'mvalid', 'n',
        'netpv', 'nliteral', 'nmiss', 'nomrate', 'normal', 'notalnum', 'notalpha', 'notcntrl',
        'notdigit', 'note', 'notfirst', 'notgraph', 'notlower', 'notname', 'notprint',
        'notpunct', 'notspace', 'notupper', 'notxdigit', 'npv', 'nvalid', 'nwkdom', 'open',
        'ordinal', 'pctl', 'pdf', 'peek', 'peekc', 'peekclong', 'peeklong', 'perm', 'pmt',
        'point', 'poisson', 'ppmt', 'probbeta', 'probbnml', 'probbnrm', 'probchi', 'probf',
        'probgam', 'probhypr', 'probit', 'probmc', 'probmed', 'probnegb', 'probnorm', 'probt',
        'propcase', 'prxchange', 'prxmatch', 'prxparen', 'prxparse', 'prxposn', 'ptrlongadd',
        'put', 'putc', 'putn', 'pvp', 'qtr', 'quantile', 'quote', 'ranbin', 'rancau', 'rand',
        'ranexp', 'rangam', 'range', 'rank', 'rannor', 'ranpoi', 'rantbl', 'rantri', 'ranuni',
        'repeat', 'resolve', 'reverse', 'rewind', 'right', 'rms', 'round', 'rounde', 'roundz',
        'saving', 'savings', 'scan', 'sdf', 'sec', 'second', 'sha256', 'sha256hex',
        'sha256hmachex', 'sign', 'sin', 'sinh', 'skewness', 'sleep', 'smallest', 'soapweb',
        'soapwebmeta', 'soapwipservice', 'soapwipsrs', 'soapws', 'soapwsmeta', 'sort',
        'soundex', 'spedis', 'sqrt', 'squantile', 'std', 'stderr', 'stfips', 'stname',
        'stnamel', 'strip', 'subpad', 'substr', 'substrn', 'sum', 'sumabs', 'symexist',
        'symget', 'symglobl', 'symlocal', 'sysget', 'sysparm', 'sysprocessid', 'sysprocessname',
        'sysprod', 'system', 'tan', 'tanh', 'time', 'timepart', 'timevalue', 'tinv', 'tnonct',
        'today', 'translate', 'transtrn', 'tranwrd', 'trigamma', 'trim', 'trimn', 'trunc',
        'typeof', 'tzoneid', 'tzonename', 'tzoneoff', 'tzones2u', 'tzoneu2s', 'uniform',
        'upcase', 'urldecode', 'urlencode', 'uss', 'uuidgen', 'var', 'varfmt', 'varinfmt',
        'varlabel', 'varlen', 'varname', 'varnum', 'varray', 'varrayx', 'vartype', 'verify',
        'vformat', 'vformatd', 'vformatdx', 'vformatn', 'vformatnx', 'vformatw', 'vformatwx',
        'vformatx', 'vinarray', 'vinarrayx', 'vinformat', 'vinformatd', 'vinformatdx',
        'vinformatn', 'vinformatnx', 'vinformatw', 'vinformatwx', 'vinformatx', 'vlabel',
        'vlabelx', 'vlength', 'vlengthx', 'vname', 'vnamex', 'vtype', 'vtypex', 'vvalue',
        'vvaluex', 'week', 'weekday', 'whichc', 'whichn', 'year', 'yieldp', 'yrdif', 'yyq',
        'zipcity', 'zipcitydistance', 'zipfips', 'zipname', 'zipnamel', 'zipstate'
    }
)

_SAS_CALL_ROUTINES = frozenset(
    {
        'allcomb', 'allcombi', 'allperm', 'cats', 'catt', 'catx', 'compcost', 'execute',
        'graycode', 'is8601_convert', 'label', 'lexcomb', 'lexcombi', 'lexperk', 'lexperm',
        'logistic', 'missing', 'module', 'poke', 'pokelong', 'prxchange', 'prxdebug', 'prxfree',
        'prxnext', 'prxposn', 'prxsubstr', 'ranbin', 'rancau', 'rancomb', 'ranexp', 'rangam',
        'ranperk', 'ranperm', 'ranpoi', 'rantbl', 'rantri', 'ranuni', 'scan', 'set', 'sleep',
        'softmax', 'sort', 'sortc', 'sortn', 'stdize', 'stream', 'streaminit', 'streamrewind',
        'symput', 'symputx', 'system', 'tanh', 'tso', 'vname', 'vnext'
    }
)

# A function call is ``name(`` (optional whitespace before the paren); a CALL
# routine is ``CALL name`` followed by a word boundary.  Both alternations are
# built longest-name-first so a shorter name that prefixes a longer one can't
# short-circuit the match, mirroring _RESERVED_WORDS_PATTERN's construction.
_SAS_FUNCTIONS_PATTERN = "|".join(
    re.escape(w) for w in sorted(_SAS_FUNCTIONS, key=len, reverse=True)
)
_SAS_CALL_ROUTINES_PATTERN = "|".join(
    re.escape(w) for w in sorted(_SAS_CALL_ROUTINES, key=len, reverse=True)
)
# These two are the only patterns in this module built from a *large* literal
# alternation (~600 function names / ~65 CALL-routine names).  The third-party
# ``regex`` engine compiles such big literal alternations into a far more
# efficient matcher than stdlib ``re`` (measured ~1.75x faster per scan on
# representative chunk text), and this scan runs once per chunk over the whole
# chunk body, so it is a real hot path.  Every *other* pattern here stays on
# stdlib ``re`` — for the small patterns and the reserved-word negative-lookahead
# alternations, ``re`` is as fast or faster, so a blanket swap would be a net loss.
_SAS_FUNCTION_CALL_RE = regex.compile(
    rf"\b({_SAS_FUNCTIONS_PATTERN})\b\s*\(",
    regex.IGNORECASE,
)
_SAS_CALL_ROUTINE_RE = regex.compile(
    rf"\bcall\s+({_SAS_CALL_ROUTINES_PATTERN})\b",
    regex.IGNORECASE,
)

_MACRO_CALL_RE = re.compile(
    rf"%(?!(?:{_RESERVED_WORDS_PATTERN})\b)([A-Za-z_]\w*)",
    re.IGNORECASE,
)
_INCLUDE_RE = re.compile(r"%\s*include\s+([^;]+)", re.IGNORECASE)
_OPTIONS_RE = re.compile(r"\boptions\s+([^;]+)", re.IGNORECASE)
_LABEL_RE = re.compile(r"\blabel\s+([A-Za-z_]\w*)\s*=", re.IGNORECASE)
_PROC_RE = re.compile(r"\bproc\s+([A-Za-z_]\w*)", re.IGNORECASE)
_DATA_RE = re.compile(
    r"\bdata\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Internal parse primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Unit:
    start: int
    end: int
    text: str
    is_comment: bool = False
    terminated: bool = True
    unclosed_comment: bool = False


@dataclass(frozen=True)
class _Region:
    kind: SasChunkKind
    start: int
    end: int
    units: list[_Unit]
    unclosed: bool = False

    @cached_property
    def text(self) -> str:
        # Cached: a region's units are fixed at construction, and `.text` is
        # read several times per region (word count, metadata, chunk build).
        # cached_property writes through the instance __dict__, which a frozen
        # (but unslotted) dataclass still allows.
        return "".join(u.text for u in self.units)


# ---------------------------------------------------------------------------
# Stuck-parser protection — deadline + watchdog
#
# See the module docstring's "Stuck-parser protection" section for the split
# of responsibilities: the deadline gives a *graceful partial exit* from the
# Python-level scan/group loops, while the watchdog guarantees a *log trail*
# for the one case the deadline cannot cover (a hang inside a single
# un-interruptible C-level regex call).
# ---------------------------------------------------------------------------


class _Deadline:
    """Monotonic wall-clock budget for a single parse.

    A ``timeout`` of ``None`` means unbounded — :meth:`expired` then always
    reports ``False`` and costs nothing.  :meth:`expired` reads the monotonic
    clock, so call sites in the hot scan/group loops gate it behind a periodic
    tick counter rather than hitting it on every character/unit.
    """

    __slots__ = ("_deadline",)

    def __init__(self, timeout: float | None) -> None:
        self._deadline = None if timeout is None else time.perf_counter() + timeout

    def expired(self) -> bool:
        return self._deadline is not None and time.perf_counter() >= self._deadline


class _ParseWatchdog:
    """Background timer that logs a trail when a parse appears stuck.

    Used as a context manager around the whole parse.  When ``timeout`` is
    ``None`` it is inert (no thread is started and :meth:`set_phase` is a
    no-op).  Otherwise a daemon thread wakes every ``timeout`` seconds and,
    while the parse is still running, logs which phase it was last in and how
    long it has taken — escalating from WARNING to ERROR after repeated
    strikes.  It never interrupts the parse (Python cannot preempt a C-level
    regex call); its sole job is to make a wedged parse diagnosable from the
    logs.  On a clean/graceful finish it is stopped and stays silent.
    """

    def __init__(self, timeout: float | None, label: str) -> None:
        self._timeout = timeout
        self._label = label
        self._phase = "starting"
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._start = 0.0
        self._thread: threading.Thread | None = None
        if timeout is not None:
            self._thread = threading.Thread(
                target=self._run, name="chunker-watchdog", daemon=True
            )

    def __enter__(self) -> _ParseWatchdog:
        if self._thread is not None:
            self._start = time.perf_counter()
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._done.set()
        if self._thread is not None:
            self._thread.join(timeout=0.1)

    def set_phase(self, phase: str) -> None:
        if self._thread is None:
            return
        with self._lock:
            self._phase = phase

    def _run(self) -> None:
        assert self._timeout is not None
        strikes = 0
        while not self._done.wait(self._timeout):
            strikes += 1
            elapsed = time.perf_counter() - self._start
            with self._lock:
                phase = self._phase
            level = logging.ERROR if strikes >= 3 else logging.WARNING
            logger.log(
                level,
                f"parse watchdog: '{self._label}' still running after "
                f"{elapsed:.1f}s (timeout={self._timeout:.1f}s) — appears stuck "
                f"in phase '{phase}'; graceful exit will occur at the next "
                f"statement boundary if the parser is not wedged in a regex",
            )


def _record_parser_timeout(
    diagnostics: list[SasDiagnostic],
    line_starts: list[int],
    phase: str,
    stop_char: int,
) -> None:
    """Append a single ``PARSER_TIMEOUT`` diagnostic (idempotent across phases).

    Once the deadline expires every subsequent phase sees it as expired and
    bails immediately, so this guards against emitting one diagnostic per
    phase — only the first (where the parser actually got stuck) is recorded.
    """
    if any(d.code == "PARSER_TIMEOUT" for d in diagnostics):
        return
    line = _line_for(stop_char, line_starts)
    logger.error(
        f"parse deadline exceeded during {phase} phase near line {line}; "
        f"returning partial result"
    )
    diagnostics.append(
        SasDiagnostic(
            code="PARSER_TIMEOUT",
            message=(
                f"Parsing exceeded its time budget during the {phase} phase; "
                f"output is partial (stopped near line {line})."
            ),
            severity=SasDiagnosticSeverity.ERROR,
            start_line=line,
            end_line=line,
        )
    )


# ---------------------------------------------------------------------------
# Block-boundary classifier
#
# IMPORTANT: only these three kinds open a new top-level block and therefore
# close whichever block is currently being collected.  Everything else
# (FORMAT, OPTIONS, GLOBAL_STATEMENT, etc.) is a *statement inside* the
# current block and must be collected, not treated as a boundary.
# ---------------------------------------------------------------------------

_BLOCK_OPENERS = frozenset(
    {
        SasChunkKind.DATA_STEP,
        SasChunkKind.PROC_STEP,
        SasChunkKind.MACRO_DEFINITION,
    }
)

# Precompiled statement-classifier patterns.  ``_classify`` runs once per unit
# (tens of thousands of times on a large file), so these are compiled once here
# rather than re-parsed from string literals on every call — which otherwise
# dominates via ``re``'s per-call pattern-cache lookup.  ``_norm`` lower-cases
# its result, so the classifier input is always lowercase and no IGNORECASE flag
# is needed (matching a lowercase literal without the flag is faster still).
_CLS_DATA_RE = re.compile(r"data\b")
_CLS_PROC_RE = re.compile(r"proc\b")
_CLS_MACRO_RE = re.compile(r"%\s*macro\b")
_CLS_INCLUDE_RE = re.compile(r"%\s*include\b")
_CLS_MACROVAR_RE = re.compile(r"%\s*(?:let|put|global|local)\b")
_CLS_CTRLFLOW_RE = re.compile(r"%\s*(?:if|else|do|end|return|goto|abort)\b")
_CLS_MACROCALL_RE = re.compile(r"%[A-Za-z_]\w*\b")
_CLS_STEP_RE = re.compile(r"(?:run|quit)\b")
_CLS_OPTIONS_RE = re.compile(r"options\b")
_CLS_GLOBAL_RE = re.compile(r"(?:libname|filename|title\d*|footnote\d*)\b")
_CLS_ODS_RE = re.compile(r"ods\b")
_CLS_FORMAT_RE = re.compile(r"(?:format|informat)\b")
# %MEND terminator check inside _collect_block; input is _norm'd (lowercase).
_MEND_RE = re.compile(r"%\s*mend\b")


def _classify(stripped: str) -> SasChunkKind | None:
    """
    Map a single SAS statement to its :class:`SasChunkKind`.

    Returns ``None`` for unrecognised statements (they accumulate into
    ``UNKNOWN_STATEMENT_GROUP`` / ``UNKNOWN_BLOCK``).

    This function is called both at the top level (to decide *which* block
    type to open) and inside ``_collect_block`` (only to detect the three
    block-opener kinds that close the current block).  All other statement
    types are transparently collected as block body statements.
    """
    return _classify_normed(_norm(stripped))


def _classify_normed(n: str) -> SasChunkKind | None:
    """Classify an already-``_norm``'d (stripped, lowercased) statement.

    Split out from :func:`_classify` so callers that already hold the
    normalised form — notably ``_collect_block``, which needs it for the
    %MEND / RUN / QUIT terminator checks too — can classify without
    re-normalising the same text.
    """
    if not n:
        return None
    if _CLS_DATA_RE.match(n):
        return SasChunkKind.DATA_STEP
    if _CLS_PROC_RE.match(n):
        return SasChunkKind.PROC_STEP
    if _CLS_MACRO_RE.match(n):
        return SasChunkKind.MACRO_DEFINITION
    if _CLS_INCLUDE_RE.match(n):
        return SasChunkKind.INCLUDE
    if _CLS_MACROVAR_RE.match(n):
        return SasChunkKind.GLOBAL_STATEMENT
    if _CLS_CTRLFLOW_RE.match(n):
        return SasChunkKind.MACRO_CONTROL_FLOW
    if _CLS_MACROCALL_RE.match(n):
        return SasChunkKind.MACRO_CALL
    # A bare RUN;/QUIT; reached here (rather than inside _collect_block) is a
    # standalone step boundary in open code — e.g. a stray RUN; after a
    # global LIBNAME/TITLE statement.  Recognise it so it doesn't fall
    # through to UNKNOWN_STATEMENT_GROUP and raise a spurious
    # UNRECOGNIZED_SOURCE_REGION diagnostic.  RUN CANCEL; is included via the
    # trailing \b (the optional CANCEL keyword follows).  Inside a DATA/PROC
    # block this same statement still terminates the block (handled directly
    # in _collect_block) and never reaches open code.
    if _CLS_STEP_RE.match(n):
        return SasChunkKind.STEP_BOUNDARY
    if _CLS_OPTIONS_RE.match(n):
        return SasChunkKind.OPTIONS
    if _CLS_GLOBAL_RE.match(n):
        return SasChunkKind.GLOBAL_STATEMENT
    if _CLS_ODS_RE.match(n):
        return SasChunkKind.GLOBAL_STATEMENT
    if _CLS_FORMAT_RE.match(n):
        return SasChunkKind.FORMAT_OR_INFORMAT
    return None


# ---------------------------------------------------------------------------
# Public chunker
# ---------------------------------------------------------------------------


class SasSemanticChunker:
    """
    Chunk Base SAS source into source-preserving semantic regions.

    Parameters
    ----------
    min_words : int
        Soft lower bound — chunks smaller than this are never split further.
    max_words : int
        Hard upper bound — regions larger than this are split at statement
        boundaries with a small overlap window.
    timeout : float | None
        Wall-clock budget, in seconds, for a single ``chunk_text`` /
        ``chunk_file`` call.  When the parser overruns it, it stops at the next
        statement/region boundary, logs where it stopped, emits a
        ``PARSER_TIMEOUT`` diagnostic and returns the partial result gathered
        so far (a background watchdog also logs the stuck phase throughout).
        Pass ``None`` to disable both safeguards and parse unbounded.  Defaults
        to 60s, comfortably above any healthy parse.
    """

    def __init__(
        self,
        *,
        min_words: int = 300,
        max_words: int = 700,
        timeout: float | None = 60.0,
    ) -> None:
        self.min_words = min_words
        self.max_words = max_words
        self.timeout = timeout
        logger.debug(
            f"SasSemanticChunker  min_words={min_words}  max_words={max_words}  "
            f"timeout={timeout}"
        )

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def chunk_file(self, path: str) -> SasChunkResult:
        """Read *path* from disk and return a :class:`SasChunkResult`."""
        fp = Path(path)
        logger.info(f"chunk_file: reading '{fp}'")
        if not fp.exists():
            logger.error(f"chunk_file: file not found '{fp}'")
            raise FileNotFoundError(fp)
        source = fp.read_text(encoding="utf-8")
        line_count = source.count("\n") + 1
        logger.debug(f"chunk_file: {len(source)} bytes / {line_count} lines  '{fp}'")
        return self.chunk_text(source, source_id=str(fp))

    def chunk_text(
        self, source: str, *, source_id: str | None = None
    ) -> SasChunkResult:
        """Parse *source* string and return a :class:`SasChunkResult`."""
        label = source_id or "<inline>"
        line_count = source.count("\n") + 1
        logger.info(
            f"chunk_text: start  source='{label}'  chars={len(source)}  lines={line_count}"
        )
        t0 = time.perf_counter()

        line_starts = _line_starts(source)
        diagnostics: list[SasDiagnostic] = []
        deadline = _Deadline(self.timeout)

        with _ParseWatchdog(self.timeout, label) as watchdog:
            watchdog.set_phase("scan")
            units = self._scan_units(source, line_starts, diagnostics, deadline)
            logger.debug(f"chunk_text: scan → {len(units)} units")

            watchdog.set_phase("region grouping")
            regions = self._group_regions(units, line_starts, diagnostics, deadline)
            logger.debug(f"chunk_text: group → {len(regions)} regions")

            watchdog.set_phase("chunk building")
            chunks: list[SasChunk] = []
            for region in regions:
                # Deadline check between regions: a graceful exit here keeps the
                # chunks already built (each region is self-contained) and stops
                # before spending more of the budget on the remaining regions.
                if deadline.expired():
                    _record_parser_timeout(
                        diagnostics, line_starts, "chunk building", region.start
                    )
                    logger.warning(
                        f"chunk_text: deadline exceeded before region at "
                        f"line {_line_for(region.start, line_starts)}; "
                        f"stopping with {len(chunks)} chunk(s) built"
                    )
                    break
                chunks.extend(
                    self._chunks_for_region(
                        source,
                        source_id,
                        region,
                        line_starts,
                        len(chunks),
                        diagnostics,
                    )
                )

        elapsed = time.perf_counter() - t0
        logger.info(
            f"chunk_text: done  source='{label}'  chunks={len(chunks)}  diagnostics={len(diagnostics)}  elapsed={elapsed:.3f}s"
        )
        return SasChunkResult(
            source_id=source_id, chunks=chunks, diagnostics=diagnostics
        )

    # ------------------------------------------------------------------
    # Phase 1 — scan source → _Unit list
    # ------------------------------------------------------------------

    def _scan_units(
        self,
        source: str,
        line_starts: list[int],
        diagnostics: list[SasDiagnostic],
        deadline: _Deadline,
    ) -> list[_Unit]:
        logger.debug(f"_scan_units: {len(source)} chars")
        units: list[_Unit] = []
        stmt_start: int | None = None
        index = 0
        quote: str | None = None
        ticks = 0

        while index < len(source):
            # Deadline check, gated behind a tick counter so perf_counter() is
            # sampled ~once per 8192 iterations, not on every character (this is
            # the module's hottest loop).  On expiry: keep the units gathered so
            # far and return them for grouping — a graceful, partial exit.
            ticks += 1
            if (ticks & 0x1FFF) == 0 and deadline.expired():
                _record_parser_timeout(diagnostics, line_starts, "scan", index)
                logger.warning(
                    f"_scan_units: deadline exceeded at char {index}/{len(source)} "
                    f"(line {_line_for(index, line_starts)}); returning "
                    f"{len(units)} partial unit(s)"
                )
                break

            if stmt_start is None:
                stmt_start = index

            char = source[index]
            nxt = source[index : index + 2]

            # ── inside quoted string ────────────────────────────────────────
            if quote:
                if char == quote:
                    if index + 1 < len(source) and source[index + 1] == quote:
                        index += 2  # doubled-quote escape
                        continue
                    quote = None
                index += 1
                continue

            if char in {"'", '"'}:
                quote = char
                index += 1
                continue

            # ── block comment /* … */ ───────────────────────────────────────
            if nxt == "/*":
                comment_start = index
                comment_end = source.find("*/", index + 2)
                before = source[stmt_start:comment_start]
                if not before.strip():
                    # standalone comment (no code before it on this statement)
                    end = (
                        len(source)
                        if comment_end == -1
                        else _ws_end(source, comment_end + 2)
                    )
                    unit = _Unit(
                        start=stmt_start,
                        end=end,
                        text=source[stmt_start:end],
                        is_comment=True,
                        terminated=(comment_end != -1),
                        unclosed_comment=(comment_end == -1),
                    )
                    units.append(unit)
                    if comment_end == -1:
                        logger.warning(
                            f"_scan_units: unclosed block comment at line {_line_for(stmt_start, line_starts)}"
                        )
                        diagnostics.append(
                            _diag(
                                "UNCLOSED_BLOCK_COMMENT",
                                "Unclosed block comment.",
                                line_starts,
                                unit,
                            )
                        )
                    elif logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"_scan_units: block comment  line={_line_for(stmt_start, line_starts)}"
                        )
                    stmt_start = None
                    index = end
                    continue
                # inline comment — skip past it without ending the statement
                index = len(source) if comment_end == -1 else comment_end + 2
                continue

            # ── statement terminator ────────────────────────────────────────
            if char == ";":
                end = _ws_end(source, index + 1)
                text = source[stmt_start:end]
                is_comment = _is_stmt_comment(text)
                units.append(
                    _Unit(
                        start=stmt_start,
                        end=end,
                        text=text,
                        is_comment=is_comment,
                    )
                )
                if logger.isEnabledFor(logging.DEBUG):
                    text_preview = text[:60].replace("\n", "↵")
                    logger.debug(
                        f"_scan_units: stmt  line={_line_for(stmt_start, line_starts)}  "
                        f"comment={is_comment}  text={text_preview!r}"
                    )
                stmt_start = None
                index = end
                continue

            index += 1

        # trailing unterminated fragment
        if stmt_start is not None and stmt_start < len(source):
            text = source[stmt_start:]
            if text.strip():
                line = _line_for(stmt_start, line_starts)
                logger.warning(f"_scan_units: unterminated statement at line {line}")
                units.append(
                    _Unit(
                        start=stmt_start,
                        end=len(source),
                        text=text,
                        is_comment=_is_stmt_comment(text),
                        terminated=False,
                    )
                )

        non_empty = [u for u in units if u.text]
        logger.debug(
            f"_scan_units: done  total={len(units)}  non_empty={len(non_empty)}"
        )
        return non_empty

    # ------------------------------------------------------------------
    # Phase 2 — _Unit list → _Region list
    # ------------------------------------------------------------------

    def _group_regions(
        self,
        units: list[_Unit],
        line_starts: list[int],
        diagnostics: list[SasDiagnostic],
        deadline: _Deadline,
    ) -> list[_Region]:
        logger.debug(f"_group_regions: {len(units)} units")
        regions: list[_Region] = []
        unknown: list[_Unit] = []
        index = 0
        ticks = 0

        def flush_unknown() -> None:
            if not unknown:
                return
            kind = (
                SasChunkKind.UNKNOWN_BLOCK
                if any(not u.terminated for u in unknown)
                else SasChunkKind.UNKNOWN_STATEMENT_GROUP
            )
            r = _Region(
                kind,
                unknown[0].start,
                unknown[-1].end,
                list(unknown),
                unclosed=(kind == SasChunkKind.UNKNOWN_BLOCK),
            )
            regions.append(r)
            sl = _line_for(r.start, line_starts)
            el = _line_for(max(r.end - 1, r.start), line_starts)
            logger.warning(
                f"_group_regions: unrecognised  kind={kind.value}  lines={sl}-{el}"
            )
            diagnostics.append(
                SasDiagnostic(
                    code="UNRECOGNIZED_SOURCE_REGION",
                    message="Source region did not match a known Base SAS semantic unit.",
                    start_line=sl,
                    end_line=el,
                )
            )
            unknown.clear()

        while index < len(units):
            # Deadline check (tick-gated, as in _scan_units).  On expiry, flush
            # any pending unknown run and return the regions built so far so the
            # chunk-building phase still gets a well-formed, if partial, list.
            ticks += 1
            if (ticks & 0x0FFF) == 0 and deadline.expired():
                _record_parser_timeout(
                    diagnostics, line_starts, "region grouping", units[index].start
                )
                logger.warning(
                    f"_group_regions: deadline exceeded at unit {index}/{len(units)} "
                    f"(line {_line_for(units[index].start, line_starts)}); "
                    f"returning {len(regions)} partial region(s)"
                )
                break

            unit = units[index]
            stripped = unit.text.strip()

            if not stripped:
                unknown.append(unit)
                index += 1
                continue

            if unit.is_comment:
                flush_unknown()
                regions.append(
                    _Region(
                        SasChunkKind.COMMENT_BLOCK,
                        unit.start,
                        unit.end,
                        [unit],
                    )
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"_group_regions: COMMENT_BLOCK  line={_line_for(unit.start, line_starts)}"
                    )
                index += 1
                continue

            cls = _classify(stripped)

            # ── block-opener: collect the whole block ───────────────────────
            if cls in _BLOCK_OPENERS:
                flush_unknown()
                block_units, index, unclosed = self._collect_block(units, index, cls)
                r = _Region(
                    cls,
                    block_units[0].start,
                    block_units[-1].end,
                    block_units,
                    unclosed=unclosed,
                )
                regions.append(r)
                sl = _line_for(r.start, line_starts)
                el = _line_for(max(r.end - 1, r.start), line_starts)
                if unclosed:
                    code = {
                        SasChunkKind.DATA_STEP: "UNCLOSED_DATA_OR_PROC_STEP",
                        SasChunkKind.PROC_STEP: "UNCLOSED_DATA_OR_PROC_STEP",
                        SasChunkKind.MACRO_DEFINITION: "UNCLOSED_MACRO",
                    }[cls]
                    logger.warning(
                        f"_group_regions: unclosed {cls.value}  lines={sl}-{el}  code={code}"
                    )
                    diagnostics.append(
                        SasDiagnostic(
                            code=code,
                            message=f"{cls.value} was not closed before end of file.",
                            start_line=sl,
                            end_line=el,
                        )
                    )
                else:
                    logger.debug(
                        f"_group_regions: {cls.value}  lines={sl}-{el}  units={len(block_units)}"
                    )
                continue

            # ── single-statement kinds ──────────────────────────────────────
            if cls is not None:
                flush_unknown()
                regions.append(
                    _Region(
                        cls,
                        unit.start,
                        unit.end,
                        [unit],
                        unclosed=(not unit.terminated),
                    )
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"_group_regions: {cls.value}  line={_line_for(unit.start, line_starts)}"
                    )
                index += 1
                continue

            unknown.append(unit)
            index += 1

        flush_unknown()
        logger.debug(f"_group_regions: → {len(regions)} regions")
        return regions

    def _collect_block(
        self,
        units: list[_Unit],
        start: int,
        kind: SasChunkKind,
    ) -> tuple[list[_Unit], int, bool]:
        """
        Collect all _Units belonging to a DATA / PROC / %MACRO block.

        A block ends when one of the following is encountered:
        - An explicit ``RUN;`` or ``QUIT;`` statement     (DATA / PROC)
        - An explicit ``%MEND;`` statement                (%MACRO)
        - A new DATA, PROC, or %MACRO header              (implicit close)
        - End of file                                     (unclosed block)

        Critically, FORMAT, LABEL, OPTIONS, LIBNAME, ODS, TITLE, and all
        other statement types are treated as ordinary body statements and
        collected without closing the block.
        """
        logger.debug(f"_collect_block: {kind.value}  start_unit={start}")
        block: list[_Unit] = []
        index = start

        while index < len(units):
            unit = units[index]

            # Comments inside a block are just collected — they don't close it
            if unit.is_comment:
                block.append(unit)
                index += 1
                continue

            # Normalise once and reuse for both classification and the
            # %MEND / RUN / QUIT terminator checks below.
            lowered = _norm(unit.text)
            cls = _classify_normed(lowered)

            # ── only DATA/PROC openers close a DATA/PROC block (implicit).
            # A %MACRO block is closed *only* by %MEND, so DATA and PROC
            # steps that appear inside a macro body are collected, not
            # treated as block boundaries.
            implicit_close = (
                cls in _BLOCK_OPENERS
                and block
                and kind in {SasChunkKind.DATA_STEP, SasChunkKind.PROC_STEP}
            )
            if implicit_close:
                logger.debug(
                    f"_collect_block: implicit close  {kind.value} at unit {index}  next_kind={cls.value}"
                )
                return block, index, True

            block.append(unit)
            index += 1

            # ── explicit terminators ────────────────────────────────────────
            if kind == SasChunkKind.MACRO_DEFINITION and _MEND_RE.match(lowered):
                logger.debug(
                    f"_collect_block: %MEND → closed MACRO_DEFINITION  units={len(block)}"
                )
                return block, index, False

            if kind in {SasChunkKind.DATA_STEP, SasChunkKind.PROC_STEP} and lowered in {
                "run",
                "quit",
            }:
                logger.debug(
                    f"_collect_block: RUN/QUIT → closed {kind.value}  units={len(block)}"
                )
                return block, index, False

        logger.warning(
            f"_collect_block: EOF without closing {kind.value}  units={len(block)}"
        )
        return block, index, True

    # ------------------------------------------------------------------
    # Phase 3 — _Region → SasChunk(s)
    # ------------------------------------------------------------------

    def _chunks_for_region(
        self,
        source: str,
        source_id: str | None,
        region: _Region,
        line_starts: list[int],
        next_index: int,
        diagnostics: list[SasDiagnostic],
    ) -> list[SasChunk]:
        wc = _wc(region.text)
        sl = _line_for(region.start, line_starts)
        el = _line_for(max(region.end - 1, region.start), line_starts)

        if wc <= self.max_words or len(region.units) <= 1:
            logger.debug(
                f"_chunks_for_region: single  kind={region.kind.value}  words={wc}  lines={sl}-{el}"
            )
            return [self._make_chunk(source_id, region, line_starts, next_index)]

        # Oversized — split at statement boundaries with overlap
        logger.info(
            f"_chunks_for_region: oversized {region.kind.value}  words={wc} > max={self.max_words}  lines={sl}-{el}  splitting"
        )
        parent_meta = _metadata_for(region.text, region.kind)
        parent_meta.has_unclosed_block = region.unclosed
        parent_chunk = self._make_chunk(
            source_id,
            region,
            line_starts,
            next_index,
            metadata=parent_meta,
        )
        chunks: list[SasChunk] = [parent_chunk]
        parent_id = parent_chunk.chunk_id
        current: list[_Unit] = []
        # Running word count of the joined `current` text, maintained
        # incrementally so the split loop stays O(units) instead of re-joining
        # and re-splitting the whole accumulated text on every unit.  When two
        # units abut without whitespace (former ends non-space, latter starts
        # non-space) their touching tokens merge into one, so joining adds
        # `_wc(unit.text)` minus one for that merge — mirroring exactly what
        # `_wc("".join(...))` would count.  `cur_tail_nonws` tracks whether the
        # joined text currently ends in a non-space character.
        cur_wc = 0
        cur_tail_nonws = False

        def _wc_and_tail(units_list: list[_Unit]) -> tuple[int, bool]:
            joined = "".join(u.text for u in units_list)
            return _wc(joined), bool(joined) and not joined[-1].isspace()

        for unit in region.units:
            utext = unit.text
            uw = _wc(utext)
            uhead_nonws = bool(utext) and not utext[0].isspace()
            utail_nonws = bool(utext) and not utext[-1].isspace()
            if current:
                merge = 1 if (cur_tail_nonws and uhead_nonws) else 0
                cand_wc = cur_wc + uw - merge
                cand_tail = utail_nonws if utext else cur_tail_nonws
            else:
                cand_wc = uw
                cand_tail = utail_nonws
            if current and cand_wc > self.max_words:
                cr = _Region(
                    region.kind,
                    current[0].start,
                    current[-1].end,
                    list(current),
                    unclosed=region.unclosed,
                )
                child_meta = _merge_meta(
                    parent_meta,
                    _metadata_for(cr.text, region.kind),
                )
                child_meta.has_unclosed_block = region.unclosed
                child = self._make_chunk(
                    source_id,
                    cr,
                    line_starts,
                    next_index + len(chunks),
                    parent_id=parent_id,
                    metadata=child_meta,
                )
                logger.debug(
                    f"_chunks_for_region: child {child.chunk_id}  parent={parent_id}  lines={child.start_line}-{child.end_line}"
                )
                chunks.append(child)
                current = _overlap(current) + [unit]
                cur_wc, cur_tail_nonws = _wc_and_tail(current)
            else:
                current = current + [unit]
                cur_wc = cand_wc
                cur_tail_nonws = cand_tail

        if current:
            cr = _Region(
                region.kind,
                current[0].start,
                current[-1].end,
                list(current),
                unclosed=region.unclosed,
            )
            child_meta = _merge_meta(
                parent_meta,
                _metadata_for(cr.text, region.kind),
            )
            child_meta.has_unclosed_block = region.unclosed
            child = self._make_chunk(
                source_id,
                cr,
                line_starts,
                next_index + len(chunks),
                parent_id=parent_id,
                metadata=child_meta,
            )
            logger.debug(
                f"_chunks_for_region: final child {child.chunk_id}  parent={parent_id}  lines={child.start_line}-{child.end_line}"
            )
            chunks.append(child)

        still_big = [c for c in chunks if _wc(c.text) > self.max_words * 2]
        if still_big:
            logger.warning(
                f"_chunks_for_region: {len(still_big)} chunk(s) remain oversized after split"
            )
            diagnostics.append(
                SasDiagnostic(
                    code="OVERSIZED_ATOMIC_CHUNK",
                    message="A chunk remained oversized because it could not be split safely.",
                    start_line=sl,
                    end_line=el,
                )
            )

        logger.info(
            f"_chunks_for_region: {region.kind.value} → {len(chunks)} chunks (1 parent + {len(chunks) - 1} children)"
        )
        return chunks

    def _make_chunk(
        self,
        source_id: str | None,
        region: _Region,
        line_starts: list[int],
        index: int,
        *,
        parent_id: str | None = None,
        metadata: SasChunkMetadata | None = None,
    ) -> SasChunk:
        meta = metadata or _metadata_for(region.text, region.kind)
        meta.has_unclosed_block = region.unclosed
        return SasChunk(
            chunk_id=f"chunk-{index + 1:04d}",
            source_id=source_id,
            text=region.text,
            kind=region.kind,
            title=_title(region.kind, meta),
            start_line=_line_for(region.start, line_starts),
            end_line=_line_for(max(region.end - 1, region.start), line_starts),
            start_char=region.start,
            end_char=region.end,
            parent_id=parent_id,
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# Pure helper functions  (no side-effects, no logging)
# ---------------------------------------------------------------------------


def _line_starts(source: str) -> list[int]:
    # Walk newline to newline with str.find (a C-level scan) rather than a
    # per-character Python loop; each match yields the start of the next line.
    starts = [0]
    pos = source.find("\n")
    while pos != -1:
        starts.append(pos + 1)
        pos = source.find("\n", pos + 1)
    return starts


def _line_for(char_index: int, line_starts: list[int]) -> int:
    return bisect_right(line_starts, char_index)


def _ws_end(source: str, index: int) -> int:
    while index < len(source) and source[index].isspace():
        index += 1
    return index


def _is_stmt_comment(text: str) -> bool:
    s = text.lstrip()
    return s.startswith("*") and not s.startswith("*/")


_WS_RE = re.compile(r"\s+")


def _norm(text: str) -> str:
    s = text.strip()
    s = _WS_RE.sub(" ", s)
    return s[:-1].strip().lower() if s.endswith(";") else s.lower()


# A block comment (terminated ``/* … */`` or unterminated ``/* …`` to EOF) or a
# quoted string literal (single/double, with the doubled-quote ``''``/``""``
# escape and a possibly-unterminated tail).  The three alternatives start with
# distinct characters, so at any position at most one can match, and ``re``'s
# left-to-right scan reproduces the original hand-written scanner's "first
# delimiter encountered wins" precedence (a quote inside a comment, or ``/*``
# inside a string, is swallowed by whichever region opened first).
_COMMENT_OR_STRING_RE = re.compile(
    r"/\*.*?(?:\*/|\Z)"
    r"|'(?:''|[^'])*(?:'|\Z)"
    r'|"(?:""|[^"])*(?:"|\Z)',
    re.DOTALL,
)
# Comment-only variant used when blank_strings=False: string literals are left
# completely intact and a ``/*`` is a comment even inside apparent quotes,
# matching the original loop's behaviour in that mode.
_COMMENT_ONLY_RE = re.compile(r"/\*.*?(?:\*/|\Z)", re.DOTALL)
# Every non-newline character; used to blank a span to spaces while preserving
# line breaks.  CR is normalised to LF afterwards, exactly as the original loop
# mapped both ``\r`` and ``\n`` to ``\n``.
_NON_NEWLINE_RE = re.compile(r"[^\r\n]")


def _blank_span(s: str) -> str:
    """Map every character of *s* to a space, except newlines: ``\\r`` and
    ``\\n`` both become ``\\n`` so downstream line alignment is preserved."""
    return _NON_NEWLINE_RE.sub(" ", s).replace("\r", "\n")


def _sanitise_repl(m: re.Match[str]) -> str:
    s = m.group(0)
    q = s[0]
    if q == "/":
        # block comment — delimiters included, blanked wholesale
        return _blank_span(s)
    # Quoted string — keep the delimiters, blank only the interior.  The span
    # is terminated (a real closing delimiter is present) iff it holds an even
    # number of the quote char: opener (1) + doubled-quote escapes (2 each) +
    # closer (1).  An odd count means the trailing quote is the second half of
    # a ``''`` escape and the literal actually runs to EOF unterminated — in
    # which case only the opening delimiter is kept.
    if s.count(q) % 2 == 0:
        return q + _blank_span(s[1:-1]) + q
    return q + _blank_span(s[1:])


def _sanitise(text: str, *, blank_strings: bool = True) -> str:
    """
    Blank out block comments, preserving newlines so line numbers stay
    aligned with the original source.

    When ``blank_strings`` is True (the default), quoted string literals
    are also blanked out — their delimiters are kept but their contents
    become spaces, with the doubled-quote escape (``''`` / ``""``) handled
    so it doesn't end the string early.  Pass ``blank_strings=False`` to
    keep quoted text intact, e.g. when extracting a literal filename from
    ``%include 'path.sas'``.

    Implemented as a single compiled-regex substitution rather than a
    character-by-character Python loop: the scanning stays in the regex
    engine and only the matched comment/string spans are rewritten, which is
    substantially faster on real source where most characters are neither.
    """
    if blank_strings:
        return _COMMENT_OR_STRING_RE.sub(_sanitise_repl, text)
    return _COMMENT_ONLY_RE.sub(lambda m: _blank_span(m.group(0)), text)


def _nid(value: str) -> str:
    value = value.strip().strip(";")
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value
    return value.lower()


# Identifies which of %let/%global/%local/%put begins a GLOBAL_STATEMENT
# chunk.  Matched against the *start* of the chunk's sanitised text — by
# construction (see _classify), any chunk already classified as
# GLOBAL_STATEMENT via this same keyword set begins with exactly one of
# these four, so a leading match is always unambiguous.
_MACRO_VAR_OP_RE = re.compile(r"%\s*(let|global|local|put)\b", re.IGNORECASE)

# Leading statement keyword of a GLOBAL_STATEMENT chunk.  Matched against the
# start of the chunk's sanitised text; by construction (see _classify) a
# GLOBAL_STATEMENT always begins with one of these.  ``title``/``footnote``
# capture without their optional occurrence digit (title2 -> title), which
# the caller lower-cases.
_GLOBAL_STMT_KW_RE = re.compile(
    r"%?\s*(let|put|global|local|libname|filename|title|footnote|ods)\b",
    re.IGNORECASE,
)

# ``%LET name`` target — the single macro variable a %LET declares.  The
# optional leading ``&`` covers indirect (double-ampersand-resolved) targets
# such as ``%let &&prefix&i = ...`` where the outer name is still literal.
_LET_TARGET_RE = re.compile(r"%\s*let\s+&*([A-Za-z_]\w*)", re.IGNORECASE)

# ``%GLOBAL``/``%LOCAL`` declaration list — captures everything up to the
# terminating semicolon; the caller splits the list on whitespace/commas.
_GLOBAL_LOCAL_DECL_RE = re.compile(
    r"%\s*(?:global|local)\s+([^;]+?)\s*;",
    re.IGNORECASE,
)

# Identifies which control-flow keyword begins a MACRO_CONTROL_FLOW chunk
# (ROADMAP Phase 3).  Mirrors _MACRO_VAR_OP_RE exactly — matched against
# the start of the chunk's sanitised text, which by construction (see
# _classify) always begins with exactly one of these seven keywords for
# any chunk already classified as MACRO_CONTROL_FLOW.
_CONTROL_FLOW_OP_RE = re.compile(
    r"%\s*(if|else|do|end|return|goto|abort)\b",
    re.IGNORECASE,
)

# Shared, precompiled token/paren helpers reused across the metadata extractors
# below.  Compiling them once here (rather than passing string literals to
# re.sub / re.findall on every call) removes the dominant per-call pattern-cache
# lookup from these hot loops.
_PAREN_RE = re.compile(r"\([^)]*\)")  # a balanced-free "(...)" span to blank out
_DS_TOKEN_RE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?")  # libref.member token
_AMP_TOKEN_RE = re.compile(r"[A-Za-z_&][\w.&]*")  # dataset token that may hold &refs
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")  # bare SAS identifier
_SPLIT_WS_COMMA_RE = re.compile(r"[,\s]+")  # %global/%local list separator
_DATA_HDR_STRIP_RE = re.compile(r"^\s*data\s+", re.IGNORECASE)  # drop DATA keyword
_NUM_SUFFIX_RE = re.compile(r"^([A-Za-z_]+?)(\d+)$")  # split trailing integer

# Any "&name" or "&name." reference, used to scan for automatic macro
# variables.  Deliberately broad (matches every macro-variable reference,
# not just &sys*) so it can be reused for other reference-tracking needs
# without re-deriving the same pattern; the sys-prefix filter is applied
# by the caller via _is_automatic_macro_var.
_VAR_REF_RE = re.compile(r"&(\w+)\.?")


def _is_automatic_macro_var(name: str) -> bool:
    """
    True if *name* (without the leading ``&`` or trailing ``.``) is one of
    SAS's automatic macro variables.

    Per *SAS Macro Language: Reference* (Ch. 12, "Automatic Macro
    Variables"), every automatic macro variable's name begins with the
    reserved ``SYS`` prefix — confirmed across all ~60 of them (SYSDATE,
    SYSLAST, SYSPARM, …).  A simple prefix check is sufficient; no
    enumerated lookup table is needed or maintained.
    """
    return name.lower().startswith("sys")


# ---------------------------------------------------------------------------
# Macro-variable producer/consumer extraction (ROADMAP Phase 2)
#
# Three SAS constructs create a macro variable as a side effect rather than
# via %LET — CALL SYMPUT/SYMPUTX inside a DATA step, and PROC SQL's INTO
# clause.  This section extracts statically-resolvable variable names from
# each, and separately detects the CALL SYMPUT/SYMPUTX local-scope hazard
# documented in SAS Macro Language: Reference, Ch. 5.
# ---------------------------------------------------------------------------


def _split_top_level(s: str, sep: str = ",") -> list[str]:
    """
    Split *s* on *sep* at paren-depth 0, respecting quoted strings.

    Unlike a pure-regex comma splitter, this correctly handles arbitrarily
    nested function calls in the argument list, e.g.
    ``'holdate', trim(left(put(holiday, worddate.)))`` splits into exactly
    two pieces, not four.
    """
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    start = 0
    for i, ch in enumerate(s):
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    parts.append(s[start:])
    return [p.strip() for p in parts]


def _clean_literal(arg: str) -> str | None:
    """
    Return the contents of *arg* if it is a single, clean quoted-string
    literal (no concatenation, no embedded quote of the same type) —
    otherwise ``None``.

    Deliberately conservative: anything that isn't a bare ``'text'`` or
    ``"text"`` (a DATA step variable name, a concatenation expression, a
    function call) returns ``None`` rather than a guessed value, matching
    the "flag as unresolved, do not guess" principle used throughout this
    module for parameterised/dynamic references.
    """
    arg = arg.strip()
    if len(arg) < 2:
        return None
    if arg[0] not in ("'", '"') or arg[-1] != arg[0]:
        return None
    inner = arg[1:-1]
    if arg[0] in inner:
        return None
    return inner


# Matches CALL SYMPUT(...) / CALL SYMPUTX(...) and captures the full
# argument-list text between the parens.
_CALL_SYMPUT_RE = re.compile(
    r"\bcall\s+symput(x?)\s*\(([^;]*?)\)\s*;",
    re.IGNORECASE | re.DOTALL,
)

# Matches CALL EXECUTE(...) and captures its single argument.
_CALL_EXECUTE_RE = re.compile(
    r"\bcall\s+execute\s*\(([^;]*?)\)\s*;",
    re.IGNORECASE | re.DOTALL,
)

# Detects %LOCAL anywhere in a macro body (used by the scope-hazard check;
# a %LOCAL declaration makes the local symbol table non-empty exactly like
# a declared parameter does).
_LOCAL_STMT_RE = re.compile(r"%\s*local\b", re.IGNORECASE)

# A %name pattern inside a CALL EXECUTE string argument's contents is detected
# with the same reserved-word-excluding matcher as every other macro-call site
# (_MACRO_CALL_RE) — the pattern is identical, so it is reused directly rather
# than compiled a second time.


def _extract_symput(
    text: str,
) -> tuple[list[str], bool, list[str]]:
    """
    Scan *text* for CALL SYMPUT/SYMPUTX statements.

    Returns
    -------
    produced : list[str]
        Statically-resolvable macro variable names created (deduplicated,
        order-preserving).
    any_unresolved : bool
        True if at least one CALL SYMPUT/SYMPUTX had a non-literal
        (dynamic) macro-variable-name argument.
    explicit_global_vars : list[str]
        Names (when resolvable) whose CALL SYMPUTX call passed an explicit
        third argument forcing global scope (``'G'`` as the first
        non-blank character) — these are exempt from the scope hazard.
    """
    produced: list[str] = []
    seen: set[str] = set()
    any_unresolved = False
    explicit_global: list[str] = []

    for m in _CALL_SYMPUT_RE.finditer(text):
        is_x = bool(m.group(1))
        args = _split_top_level(m.group(2))
        if not args:
            continue
        name_arg = args[0]
        name = _clean_literal(name_arg)
        if name is None:
            any_unresolved = True
            continue
        name = name.lower()
        if name not in seen:
            seen.add(name)
            produced.append(name)

        if is_x and len(args) >= 3:
            scope_arg = _clean_literal(args[2])
            if scope_arg and scope_arg.strip().lower().startswith("g"):
                explicit_global.append(name)

    return produced, any_unresolved, explicit_global


# Captures a %GOTO statement's label.  Per Ch. 5, a *computed* %GOTO is one
# whose label contains "&" or "%" (e.g. %goto &home;) -- this is one of the
# three documented conditions that forces CALL SYMPUT/SYMPUTX into local
# scope even when the symbol table would otherwise be empty.
_GOTO_LABEL_RE = re.compile(r"%\s*goto\s+([^;]+?)\s*;", re.IGNORECASE)

# Detects a bare %ABORT statement anywhere in a macro body.
_ABORT_STMT_RE = re.compile(r"%\s*abort\b", re.IGNORECASE)


def _macro_contains_computed_goto(text: str) -> bool:
    """True if *text* contains a %GOTO whose label references a macro
    variable or macro function (Ch. 5's "computed %GOTO")."""
    for m in _GOTO_LABEL_RE.finditer(text):
        label = m.group(1)
        if "&" in label or "%" in label:
            return True
    return False


def _macro_has_local_scope(text: str, param_names: list[str]) -> bool:
    """
    True if CALL SYMPUT/SYMPUTX inside the macro body *text* would store
    its variable in the *local* symbol table rather than walking up to the
    nearest non-empty (often global) one.

    Per Ch. 5 "Special Cases of Scope", this happens when either:
    - the local symbol table is non-empty — i.e. the macro has at least
      one declared parameter, or contains an explicit ``%LOCAL`` statement
      anywhere in its body; or
    - the macro contains a *computed* ``%GOTO`` (a label referencing a
      macro variable or function) — one of three documented conditions
      that force local scope *even when the table would otherwise be
      empty* (the other two — CALL SYMPUT used after a PROC SQL step, and
      the rare SYSPBUFF case — are not detected; see ROADMAP Phase 2/3).
    """
    return (
        bool(param_names)
        or bool(_LOCAL_STMT_RE.search(text))
        or _macro_contains_computed_goto(text)
    )


def _extract_call_execute_macros(text: str) -> list[str]:
    """
    Scan *text* for ``CALL EXECUTE(argument)`` statements and return the
    macro name(s) invoked, when statically resolvable.

    Resolvable cases (per Ch. 15's CALL EXECUTE dictionary entry):
    - A clean quoted-string argument containing a literal ``%name`` —
      e.g. ``call execute('%sales');``.
    - A concatenation expression whose *first* piece (before the first
      ``||``) is a clean quoted-string literal containing a complete
      ``%name(`` — e.g. ``call execute('%sales('||month||')');`` resolves
      to ``sales`` even though the full argument list isn't known.

    Unresolvable cases (left alone, per "flag as unresolved, do not
    guess"): an unquoted DATA step variable name, or any expression whose
    first piece isn't itself a clean literal.
    """
    found: list[str] = []
    for m in _CALL_EXECUTE_RE.finditer(text):
        arg = m.group(1).strip()
        first_piece = arg.split("||", 1)[0].strip()
        literal = _clean_literal(first_piece)
        if literal is None:
            continue
        call_m = _MACRO_CALL_RE.search(literal)
        if call_m:
            found.append(call_m.group(1).lower())
    return found


# Captures the INTO clause of a PROC SQL step, up to the next FROM/; .
_SQL_INTO_CLAUSE_RE = re.compile(
    r"\binto\s+(.+?)(?=\bfrom\b|;)",
    re.IGNORECASE | re.DOTALL,
)
# A single ":name" target inside an INTO clause.
_SQL_INTO_VAR_RE = re.compile(r":\s*([A-Za-z_]\w*)")
# THROUGH / THRU are exact synonyms for "-" in a numbered-range INTO target.
_SQL_RANGE_SEP_RE = re.compile(r"-|\bthrough\b|\bthru\b", re.IGNORECASE)


def _enumerate_numbered_range(name1: str, name2: str) -> list[str] | None:
    """
    Expand a ``:var1 - :varN`` / ``THROUGH`` / ``THRU`` numbered-range INTO
    target into the full list of variable names, when both bounds share a
    common alphabetic prefix and end in parseable integers (e.g.
    ``type1``..``type4`` -> ``["type1","type2","type3","type4"]``).

    Returns ``None`` when the bounds don't fit that shape — the two given
    names are still tracked individually by the caller in that case.
    """
    m1 = _NUM_SUFFIX_RE.match(name1)
    m2 = _NUM_SUFFIX_RE.match(name2)
    if not m1 or not m2:
        return None
    prefix1, n1 = m1.group(1), int(m1.group(2))
    prefix2, n2 = m2.group(1), int(m2.group(2))
    if prefix1.lower() != prefix2.lower() or n2 < n1:
        return None
    return [f"{prefix1}{i}" for i in range(n1, n2 + 1)]


def _extract_sql_into_vars(text: str) -> list[str]:
    """
    Scan *text* (a PROC SQL step's source) for every ``INTO`` clause and
    return the macro variable names created, covering all three documented
    forms (Ch. 18):

    - ``into :var1, :var2, ...`` — each name tracked directly.
    - ``into :var1 - :varN`` (or ``THROUGH``/``THRU``) — enumerated when
      both bounds share a prefix and end in integers; otherwise both named
      bounds are tracked individually.
    - ``into :var separated by '...'`` — the single name is tracked.
    """
    produced: list[str] = []
    seen: set[str] = set()

    for clause_m in _SQL_INTO_CLAUSE_RE.finditer(text):
        clause = clause_m.group(1)
        targets = [t.strip() for t in clause.split(",")]
        for target in targets:
            if not target:
                continue
            names = _SQL_INTO_VAR_RE.findall(target)
            if len(names) == 2 and _SQL_RANGE_SEP_RE.search(target):
                expanded = _enumerate_numbered_range(names[0], names[1])
                names = expanded if expanded is not None else names
            for n in names:
                n = n.lower()
                if n not in seen:
                    seen.add(n)
                    produced.append(n)

    return produced


def _metadata_for(text: str, kind: SasChunkKind) -> SasChunkMetadata:
    cf = _sanitise(text, blank_strings=False)
    mt = _sanitise(text)
    # `datasets` is re-sorted later (via `sorted(set(datasets + two_part))`), so
    # collect it unsorted here and skip the redundant intermediate sort.
    datasets = [_nid(m.group(1)) for m in _DATASET_RE.finditer(mt)]
    datasets += [_nid(m.group(1)) for m in _DATA_OPT_RE.finditer(mt)]
    # One _LIBREF_RE pass feeds both the bare-libref set and the two-part
    # (libref.member) set instead of scanning the text twice.
    libref_set: set[str] = set()
    two_part_set: set[str] = set()
    for m in _LIBREF_RE.finditer(mt):
        libref_set.add(_nid(m.group(1)))
        two_part_set.add(_nid(m.group(0)))
    librefs = sorted(libref_set)
    two_part = sorted(two_part_set)
    mdefs = sorted({_nid(m.group(1)) for m in _MACRO_DEF_RE.finditer(mt)})
    mcalls = sorted({_nid(m.group(1)) for m in _MACRO_CALL_RE.finditer(mt)})
    includes = [_nid(m.group(1)).strip("'\"") for m in _INCLUDE_RE.finditer(cf)]
    options = [_nid(p) for m in _OPTIONS_RE.finditer(mt) for p in m.group(1).split()]
    labels = sorted({_nid(m.group(1)) for m in _LABEL_RE.finditer(mt)})
    pm = _PROC_RE.search(mt)
    dm = _DATA_RE.search(mt)
    mm = _MACRO_DEF_RE.search(mt)
    inp, out, defs, invk = _io_for(text, kind, mt)

    # ── macro-variable operation (%let / %global / %local / %put) ──────────
    var_op: str | None = None
    global_stmt_kw: str | None = None
    if kind == SasChunkKind.GLOBAL_STATEMENT:
        op_m = _MACRO_VAR_OP_RE.match(mt.lstrip())
        if op_m:
            var_op = op_m.group(1).lower()
        kw_m = _GLOBAL_STMT_KW_RE.match(mt.lstrip())
        if kw_m:
            global_stmt_kw = kw_m.group(1).lower()

    # ── control-flow operation (ROADMAP Phase 3) ────────────────────────────
    # Which specific keyword (%if/%else/%do/%end/%return/%goto/%abort) this
    # MACRO_CONTROL_FLOW chunk is.  Only ever set for that one kind — these
    # same words appearing *inside* a macro body don't get their own chunk
    # at all (they're part of the enclosing MACRO_DEFINITION's text).
    control_flow_op: str | None = None
    if kind == SasChunkKind.MACRO_CONTROL_FLOW:
        cf_m = _CONTROL_FLOW_OP_RE.match(mt.lstrip())
        if cf_m:
            control_flow_op = cf_m.group(1).lower()

    # ── automatic (system) macro variable references ───────────────────────
    # Scanned on `cf` (quotes preserved, comments stripped) rather than `mt`
    # so references inside double-quoted strings — e.g. title "Report run
    # &sysday" — are still caught.  SAS only resolves macro variables inside
    # double-quoted strings, never single-quoted ones; distinguishing that
    # here would add real complexity for marginal benefit, so this also
    # matches (harmlessly) inside single-quoted text.
    auto_vars = sorted(
        {
            m.group(1).lower()
            for m in _VAR_REF_RE.finditer(cf)
            if _is_automatic_macro_var(m.group(1))
        }
    )

    # ── macro body I/O classification (literal vs parameterised) ───────────
    body_lit_in: list[str] = []
    body_lit_out: list[str] = []
    body_par_in: list[dict] = []
    body_par_out: list[dict] = []
    param_names: list[str] = []
    if kind == SasChunkKind.MACRO_DEFINITION:
        body_lit_in, body_lit_out, body_par_in, body_par_out, param_names = (
            _macro_body_io(text, mt)
        )

    # ── high-severity control-flow visibility (ROADMAP Phase 3) ─────────────
    # %ABORT and a computed %GOTO are macro-definition-only constructs, so
    # they never get their own MACRO_CONTROL_FLOW chunk — they're always
    # part of the enclosing macro's own text.  Surfaced here regardless of
    # how deeply nested inside %if/%do blocks they are.
    has_abort = False
    has_computed_goto = False
    if kind == SasChunkKind.MACRO_DEFINITION:
        has_abort = bool(_ABORT_STMT_RE.search(text))
        has_computed_goto = _macro_contains_computed_goto(text)

    # ── macro-variable producer/consumer edges (ROADMAP Phase 2) ────────────
    produces_macrovars: list[str] = []
    hazard: bool = False
    hazard_vars: list[str] = []

    if kind in {SasChunkKind.DATA_STEP, SasChunkKind.MACRO_DEFINITION}:
        symput_names, _unresolved, explicit_global = _extract_symput(cf)
        produces_macrovars.extend(symput_names)
        invk.extend(_extract_call_execute_macros(cf))

        if kind == SasChunkKind.MACRO_DEFINITION and symput_names:
            has_local = _macro_has_local_scope(text, param_names)
            if has_local:
                at_risk = [n for n in symput_names if n not in explicit_global]
                if at_risk:
                    hazard = True
                    hazard_vars = at_risk

    elif kind == SasChunkKind.PROC_STEP and pm and _nid(pm.group(1)) == "sql":
        produces_macrovars.extend(_extract_sql_into_vars(cf))

    # consumes_macrovars: every "&name" reference in this chunk, excluding
    # automatic variables (tracked separately above) and — for
    # MACRO_DEFINITION chunks — the macro's own declared parameters (those
    # are call-site-resolved, not a corpus-level dependency).
    own_params = set(param_names)
    consumes_macrovars = sorted(
        {
            m.group(1).lower()
            for m in _VAR_REF_RE.finditer(cf)
            if not _is_automatic_macro_var(m.group(1))
            and m.group(1).lower() not in own_params
        }
    )

    # ── macro-language-level declarations and references ────────────────────
    # declared_macro_vars: names introduced by %LET and by %GLOBAL/%LOCAL
    # declaration lists.  Scanned on `cf` so %LET targets inside quoted text
    # aren't a concern (these statements are never quoted).
    declared: list[str] = [m.group(1).lower() for m in _LET_TARGET_RE.finditer(cf)]
    for m in _GLOBAL_LOCAL_DECL_RE.finditer(cf):
        for name in _SPLIT_WS_COMMA_RE.split(m.group(1).strip()):
            name = name.lstrip("&").rstrip(".")
            if _IDENT_RE.fullmatch(name):
                declared.append(name.lower())
    declared_macro_vars = sorted(set(declared))

    # referenced_macro_vars: the complete set of "&name" references (automatic
    # variables included), the unfiltered counterpart to consumes_macrovars.
    referenced_macro_vars = sorted(
        {m.group(1).lower() for m in _VAR_REF_RE.finditer(cf)}
    )

    # ── recognised SAS functions and CALL routines ──────────────────────────
    # Scanned on `mt` (string literals blanked) so a function-like token inside
    # a quoted string isn't mistaken for a real call.
    recognized_functions = sorted(
        {m.group(1).lower() for m in _SAS_FUNCTION_CALL_RE.finditer(mt)}
    )
    recognized_call_routines = sorted(
        {m.group(1).lower() for m in _SAS_CALL_ROUTINE_RE.finditer(mt)}
    )
    # A ``CALL name(...)`` invocation also textually matches the function-call
    # pattern (``name(``); drop those so a routine isn't double-reported as a
    # function of the same name.
    recognized_functions = [
        f for f in recognized_functions if f not in recognized_call_routines
    ]

    return SasChunkMetadata(
        step_name=_nid(dm.group(1)) if dm else None,
        proc_name=_nid(pm.group(1)) if pm else None,
        macro_name=_nid(mm.group(1)) if mm else None,
        labels=labels,
        referenced_librefs=librefs,
        referenced_datasets=sorted(set(datasets + two_part)),
        defined_macros=mdefs,
        called_macros=mcalls,
        includes=includes,
        options=options,
        has_unclosed_block=(kind == SasChunkKind.UNKNOWN_BLOCK),
        macro_var_op=var_op,
        global_statement_keyword=global_stmt_kw,
        referenced_automatic_vars=auto_vars,
        declared_macro_vars=declared_macro_vars,
        referenced_macro_vars=referenced_macro_vars,
        recognized_functions=recognized_functions,
        recognized_call_routines=recognized_call_routines,
        control_flow_op=control_flow_op,
        contains_abort=has_abort,
        contains_computed_goto=has_computed_goto,
        input_datasets=inp,
        output_datasets=out,
        defines_macros=defs,
        invokes_macros=sorted(set(invk)),
        body_literal_inputs=body_lit_in,
        body_literal_outputs=body_lit_out,
        body_param_inputs=body_par_in,
        body_param_outputs=body_par_out,
        macro_param_names=param_names,
        produces_macrovars=sorted(set(produces_macrovars)),
        consumes_macrovars=consumes_macrovars,
        symput_scope_hazard=hazard,
        symput_hazard_vars=sorted(set(hazard_vars)),
    )


def _merge_meta(parent: SasChunkMetadata, child: SasChunkMetadata) -> SasChunkMetadata:
    def ml(a: list[str], b: list[str]) -> list[str]:
        return sorted(set(a + b))

    return SasChunkMetadata(
        step_name=child.step_name or parent.step_name,
        proc_name=child.proc_name or parent.proc_name,
        macro_name=child.macro_name or parent.macro_name,
        labels=ml(parent.labels, child.labels),
        referenced_librefs=ml(parent.referenced_librefs, child.referenced_librefs),
        referenced_datasets=ml(parent.referenced_datasets, child.referenced_datasets),
        defined_macros=ml(parent.defined_macros, child.defined_macros),
        called_macros=ml(parent.called_macros, child.called_macros),
        includes=ml(parent.includes, child.includes),
        options=ml(parent.options, child.options),
        has_unclosed_block=parent.has_unclosed_block or child.has_unclosed_block,
        macro_var_op=child.macro_var_op or parent.macro_var_op,
        global_statement_keyword=child.global_statement_keyword
        or parent.global_statement_keyword,
        referenced_automatic_vars=ml(
            parent.referenced_automatic_vars,
            child.referenced_automatic_vars,
        ),
        declared_macro_vars=ml(parent.declared_macro_vars, child.declared_macro_vars),
        referenced_macro_vars=ml(
            parent.referenced_macro_vars, child.referenced_macro_vars
        ),
        recognized_functions=ml(
            parent.recognized_functions, child.recognized_functions
        ),
        recognized_call_routines=ml(
            parent.recognized_call_routines, child.recognized_call_routines
        ),
        input_datasets=ml(parent.input_datasets, child.input_datasets),
        output_datasets=ml(parent.output_datasets, child.output_datasets),
        defines_macros=ml(parent.defines_macros, child.defines_macros),
        invokes_macros=ml(parent.invokes_macros, child.invokes_macros),
        body_literal_inputs=ml(parent.body_literal_inputs, child.body_literal_inputs),
        body_literal_outputs=ml(
            parent.body_literal_outputs, child.body_literal_outputs
        ),
        body_param_inputs=parent.body_param_inputs or child.body_param_inputs,
        body_param_outputs=parent.body_param_outputs or child.body_param_outputs,
        macro_param_names=parent.macro_param_names or child.macro_param_names,
        produces_macrovars=ml(parent.produces_macrovars, child.produces_macrovars),
        consumes_macrovars=ml(parent.consumes_macrovars, child.consumes_macrovars),
        symput_scope_hazard=parent.symput_scope_hazard or child.symput_scope_hazard,
        symput_hazard_vars=ml(parent.symput_hazard_vars, child.symput_hazard_vars),
        control_flow_op=child.control_flow_op or parent.control_flow_op,
        contains_abort=parent.contains_abort or child.contains_abort,
        contains_computed_goto=parent.contains_computed_goto
        or child.contains_computed_goto,
    )


def _title(kind: SasChunkKind, meta: SasChunkMetadata) -> str | None:
    if kind == SasChunkKind.DATA_STEP and meta.step_name:
        return f"DATA {meta.step_name}"
    if kind == SasChunkKind.PROC_STEP and meta.proc_name:
        return f"PROC {meta.proc_name}"
    if kind == SasChunkKind.MACRO_DEFINITION and meta.macro_name:
        return f"%MACRO {meta.macro_name}"
    if kind == SasChunkKind.MACRO_CONTROL_FLOW and meta.control_flow_op:
        return f"%{meta.control_flow_op.upper()}"
    return kind.value.replace("_", " ").title()


def _wc(text: str) -> int:
    # str.split() with no argument splits on runs of whitespace and drops empty
    # fields, so its length equals the number of ``\S+`` runs — identical to the
    # previous ``len(re.findall(r"\S+", text))`` but without the regex overhead.
    return len(text.split())


def _overlap(units: list[_Unit]) -> list[_Unit]:
    overlap: list[_Unit] = []
    words = 0
    for unit in reversed(units[-3:]):
        uw = _wc(unit.text)
        if words + uw > 100 and overlap:
            break
        overlap.insert(0, unit)
        words += uw
    return overlap


def _diag(
    code: str,
    msg: str,
    line_starts: list[int],
    target: _Unit | _Region,
) -> SasDiagnostic:
    return SasDiagnostic(
        code=code,
        message=msg,
        severity=SasDiagnosticSeverity.WARNING,
        start_line=_line_for(target.start, line_starts),
        end_line=_line_for(max(target.end - 1, target.start), line_starts),
    )


# ---------------------------------------------------------------------------
# Directed I/O extraction  — called from _metadata_for
# ---------------------------------------------------------------------------

_SQL_CREATE_RE = re.compile(
    r"\bcreate\s+(?:table|view)\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)",
    re.IGNORECASE,
)
_SQL_FROM_RE = re.compile(
    r"\bfrom\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)",
    re.IGNORECASE,
)
_SQL_JOIN_RE = re.compile(
    r"\bjoin\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)",
    re.IGNORECASE,
)
_SQL_INTO_RE = re.compile(
    r"\binsert\s+into\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)",
    re.IGNORECASE,
)
_SET_RE = re.compile(
    r"\bset\s+((?:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?(?:\s*\([^)]*\))?\s*)+?)"
    r"(?=;|\bwhere\b|\bby\b|\bobs\b|\bnobs\b)",
    re.IGNORECASE | re.DOTALL,
)
_MERGE_RE = re.compile(
    r"\bmerge\s+((?:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?(?:\s*\([^)]*\))?\s*)+?)"
    r"(?=;|\bby\b)",
    re.IGNORECASE | re.DOTALL,
)
_UPDATE_RE = re.compile(
    r"\bupdate\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)",
    re.IGNORECASE,
)
_MODIFY_RE = re.compile(
    r"\bmodify\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)",
    re.IGNORECASE,
)
_OUTPUT_DS_RE = re.compile(
    r"\boutput\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)",
    re.IGNORECASE,
)
_PROC_OUT_RE = re.compile(
    r"\bout\s*=\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)"
    r"|\boutdata\s*=\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)",
    re.IGNORECASE,
)
# A PROC step's DATA= input is the same "data=<name>" option matched by
# _DATA_OPT_RE (defined near the top of the module); it is reused here rather
# than compiled a second time under a separate name.
_SAS_RESERVED = frozenset(
    {
        "work",
        "_null_",
        "_all_",
        "_numeric_",
        "_character_",
        "sashelp",
        "sasuser",
        "maps",
        "mapssas",
    }
)
_MACRO_INVOKE_RE = re.compile(
    rf"%(?!(?:{_RESERVED_WORDS_PATTERN})\b)([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Macro body dataset classification
#
# A %MACRO body may reference datasets two ways:
#   Literal       — a hard-coded name, e.g. "data work.base;"
#                   Resolvable purely from the macro source text.
#   Parameterised — a macro variable reference, e.g. "data &ds.;"
#                   Only resolvable at the call site where the argument
#                   value is known.
#
# The functions below extract both kinds from a MACRO_DEFINITION chunk's
# source text, used by the batcher to wire up cross-file dependencies that
# pass through macro bodies.
# ---------------------------------------------------------------------------

# A SAS macro variable reference (&name. or &name) is detected with the shared
# _VAR_REF_RE defined above — the pattern is identical, so it is reused here
# rather than compiled a second time.

# DATA statement header inside a macro body (may contain &refs)
_BODY_DATA_HDR_RE = re.compile(
    r"(?<![\w=])data\s+((?:[A-Za-z_&][\w.&]*\s*)+?)(?=;)",
    re.IGNORECASE,
)
_BODY_SET_RE = re.compile(
    r"\bset\s+((?:[A-Za-z_&][\w.&]*(?:\s*\([^)]*\))?\s*)+?)(?=;|\bwhere\b|\bby\b|\bobs\b)",
    re.IGNORECASE | re.DOTALL,
)
_BODY_MERGE_RE = re.compile(
    r"\bmerge\s+((?:[A-Za-z_&][\w.&]*(?:\s*\([^)]*\))?\s*)+?)(?=;|\bby\b)",
    re.IGNORECASE | re.DOTALL,
)
_BODY_UPDATE_RE = re.compile(r"\bupdate\s+([A-Za-z_&][\w.&]*)", re.IGNORECASE)
_BODY_MODIFY_RE = re.compile(r"\bmodify\s+([A-Za-z_&][\w.&]*)", re.IGNORECASE)
_BODY_OUTPUT_RE = re.compile(r"\boutput\s+([A-Za-z_&][\w.&]*)", re.IGNORECASE)
_BODY_PROC_IN_RE = re.compile(r"\bdata\s*=\s*([A-Za-z_&][\w.&]*)", re.IGNORECASE)
_BODY_PROC_OUT_RE = re.compile(
    r"\bout\s*=\s*([A-Za-z_&][\w.&]*)"
    r"|\boutdata\s*=\s*([A-Za-z_&][\w.&]*)",
    re.IGNORECASE,
)
_BODY_SQL_CREATE_RE = re.compile(
    r"\bcreate\s+(?:table|view)\s+([A-Za-z_&][\w.&]*)",
    re.IGNORECASE,
)
_BODY_SQL_FROM_RE = re.compile(r"\bfrom\s+([A-Za-z_&][\w.&]*)", re.IGNORECASE)
_BODY_SQL_JOIN_RE = re.compile(r"\bjoin\s+([A-Za-z_&][\w.&]*)", re.IGNORECASE)
_BODY_SQL_INTO_RE = re.compile(
    r"\binsert\s+into\s+([A-Za-z_&][\w.&]*)",
    re.IGNORECASE,
)

# Splits a macro argument list on commas, respecting nested parens
_ARG_SPLIT_RE = re.compile(r",(?![^(]*\))")

# Extracts macro signature: %macro name(params)
_MACRO_SIG_RE = re.compile(r"%\s*macro\s+\w+\s*\(([^)]*)\)", re.IGNORECASE)


def _classify_ref(
    raw: str,
    param_pos: dict[str, int],
) -> tuple[str, bool]:
    """
    Classify a raw dataset reference extracted from a macro body.

    Returns (resolved_key, is_parameterised).
    ``resolved_key`` is the literal lowercased dataset name if there is no
    ``&`` reference, or the lowercased macro parameter name (without ``&``)
    if the reference is a single, recognised macro variable.
    """
    raw = raw.strip()
    if "&" not in raw:
        return raw.lower(), False
    refs = _VAR_REF_RE.findall(raw)
    if len(refs) == 1 and refs[0].lower() in param_pos:
        return refs[0].lower(), True
    return raw.lower(), True


def _parse_macro_params(sig_text: str) -> list[tuple[str, str | None]]:
    """Parse a comma-separated macro parameter list into (name, default)."""
    params: list[tuple[str, str | None]] = []
    if not sig_text.strip():
        return params
    for part in _ARG_SPLIT_RE.split(sig_text):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            name, default = part.split("=", 1)
            params.append((name.strip().lower(), default.strip()))
        else:
            params.append((part.lower(), None))
    return params


def _macro_body_io(
    macro_text: str,
    mt: str | None = None,
) -> tuple[list[str], list[str], list[dict], list[dict], list[str]]:
    """
    Analyse a %MACRO block's body and classify every dataset reference
    as literal (fixed value) or parameterised (depends on a call argument).

    ``mt`` is the sanitised (comments/strings blanked) form of ``macro_text``;
    callers that already have it — e.g. :func:`_metadata_for` — pass it in to
    avoid re-running the sanitiser over the same body.  When omitted it is
    computed here, so direct callers can still pass just the raw text.

    Returns
    -------
    literal_inputs, literal_outputs : list[str]
    param_inputs, param_outputs     : list[dict]   {"param": name, "pos": idx}
    param_names                     : list[str]    ordered signature names
    """
    if mt is None:
        mt = _sanitise(macro_text)

    sig_m = _MACRO_SIG_RE.search(macro_text)
    params = _parse_macro_params(sig_m.group(1) if sig_m else "")

    param_pos: dict[str, int] = {}
    pos_idx = 0
    for pname, default in params:
        if default is None:
            param_pos[pname] = pos_idx
            pos_idx += 1
        else:
            param_pos[pname] = -1

    param_names = [p[0] for p in params]
    logger.debug(f"_macro_body_io: params={param_names}  param_pos={param_pos}")

    raw_outputs: list[str] = []
    raw_inputs: list[str] = []

    for m in _BODY_DATA_HDR_RE.finditer(mt):
        for tok in _AMP_TOKEN_RE.findall(m.group(1)):
            if tok.lower() not in _SAS_RESERVED:
                raw_outputs.append(tok)

    for m in _BODY_OUTPUT_RE.finditer(mt):
        raw_outputs.append(m.group(1))

    for m in _BODY_PROC_OUT_RE.finditer(mt):
        raw = m.group(1) or m.group(2) or ""
        if raw:
            raw_outputs.append(raw)

    for m in _BODY_SQL_CREATE_RE.finditer(mt):
        raw_outputs.append(m.group(1))
    for m in _BODY_SQL_INTO_RE.finditer(mt):
        raw_outputs.append(m.group(1))

    for m in _BODY_SET_RE.finditer(mt):
        cleaned = _PAREN_RE.sub(" ", m.group(1))
        for tok in _AMP_TOKEN_RE.findall(cleaned):
            raw_inputs.append(tok)

    for m in _BODY_MERGE_RE.finditer(mt):
        cleaned = _PAREN_RE.sub(" ", m.group(1))
        for tok in _AMP_TOKEN_RE.findall(cleaned):
            raw_inputs.append(tok)

    for m in _BODY_UPDATE_RE.finditer(mt):
        raw_inputs.append(m.group(1))
        raw_outputs.append(m.group(1))
    for m in _BODY_MODIFY_RE.finditer(mt):
        raw_inputs.append(m.group(1))
        raw_outputs.append(m.group(1))

    for m in _BODY_PROC_IN_RE.finditer(mt):
        raw_inputs.append(m.group(1))

    for m in _BODY_SQL_FROM_RE.finditer(mt):
        raw_inputs.append(m.group(1))
    for m in _BODY_SQL_JOIN_RE.finditer(mt):
        raw_inputs.append(m.group(1))

    def _classify_list(raws: list[str], role: str) -> tuple[list[str], list[dict]]:
        literals: list[str] = []
        params_out: list[dict] = []
        seen_lit: set[str] = set()
        seen_par: set[str] = set()

        for raw in raws:
            raw = raw.strip()
            if not raw or raw.lower() in _SAS_RESERVED:
                continue
            key, is_param = _classify_ref(raw, param_pos)
            if is_param:
                pname = key
                if pname in param_pos and pname not in seen_par:
                    seen_par.add(pname)
                    params_out.append({"param": pname, "pos": param_pos[pname]})
                    logger.debug(
                        f"_macro_body_io: {role} PARAM  raw={raw!r}  param={pname}  pos={param_pos[pname]}"
                    )
                elif pname not in param_pos:
                    logger.debug(
                        f"_macro_body_io: {role} UNRESOLVABLE param ref {raw!r}"
                    )
            else:
                if key not in seen_lit:
                    seen_lit.add(key)
                    literals.append(key)
                    logger.debug(
                        f"_macro_body_io: {role} LITERAL  raw={raw!r}  name={key}"
                    )

        return literals, params_out

    lit_out, par_out = _classify_list(raw_outputs, "output")
    lit_in, par_in = _classify_list(raw_inputs, "input")

    logger.debug(
        f"_macro_body_io: literal_outputs={lit_out}  literal_inputs={lit_in}  param_outputs={par_out}  param_inputs={par_in}"
    )
    return lit_in, lit_out, par_in, par_out, param_names


def _ds_name(raw: str) -> str | None:
    name = raw.strip().lower().split("(")[0].strip()
    if not name or name in _SAS_RESERVED:
        return None
    return name


def _multi_ds(match_group: str) -> list[str]:
    cleaned = _PAREN_RE.sub(" ", match_group)
    tokens = _DS_TOKEN_RE.findall(cleaned)
    return [n for t in tokens if (n := _ds_name(t))]


def _io_for(
    text: str,
    kind: SasChunkKind,
    mt: str | None = None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Extract directed data-flow edges from a single chunk's source text.

    ``mt`` is the sanitised (comments/strings blanked) form of ``text``;
    callers that already have it pass it in to avoid a redundant sanitise
    pass.  When omitted it is computed here.

    Returns
    -------
    (input_datasets, output_datasets, defines_macros, invokes_macros)
    """
    if mt is None:
        mt = _sanitise(text)

    inputs: list[str] = []
    outputs: list[str] = []
    defines: list[str] = []
    # Every chunk kind may invoke a macro inline (e.g. %clean inside a DATA
    # step body, or a bare top-level call) — this scan is unconditional and
    # identical across all kinds, so it runs once here rather than being
    # repeated in each branch below.
    invokes: list[str] = [m.group(1).lower() for m in _MACRO_INVOKE_RE.finditer(mt)]

    if kind == SasChunkKind.MACRO_DEFINITION:
        for m in _MACRO_DEF_RE.finditer(mt):
            defines.append(m.group(1).lower())

    elif kind == SasChunkKind.MACRO_CALL:
        pass

    elif kind == SasChunkKind.DATA_STEP:
        first_semi = mt.find(";")
        data_header = mt[:first_semi] if first_semi != -1 else mt
        header_body = _DATA_HDR_STRIP_RE.sub("", data_header)
        for tok in _DS_TOKEN_RE.findall(header_body):
            if n := _ds_name(tok):
                outputs.append(n)
        for m in _OUTPUT_DS_RE.finditer(mt):
            if n := _ds_name(m.group(1)):
                outputs.append(n)
        for m in _SET_RE.finditer(mt):
            inputs.extend(_multi_ds(m.group(1)))
        for m in _MERGE_RE.finditer(mt):
            inputs.extend(_multi_ds(m.group(1)))
        for m in _UPDATE_RE.finditer(mt):
            if n := _ds_name(m.group(1)):
                inputs.append(n)
                outputs.append(n)
        for m in _MODIFY_RE.finditer(mt):
            if n := _ds_name(m.group(1)):
                inputs.append(n)
                outputs.append(n)

    elif kind == SasChunkKind.PROC_STEP:
        proc_m = _PROC_RE.search(mt)
        proc_name = proc_m.group(1).lower() if proc_m else ""

        if proc_name == "sql":
            for m in _SQL_CREATE_RE.finditer(mt):
                if n := _ds_name(m.group(1)):
                    outputs.append(n)
            for m in _SQL_INTO_RE.finditer(mt):
                if n := _ds_name(m.group(1)):
                    outputs.append(n)
            for m in _SQL_FROM_RE.finditer(mt):
                if n := _ds_name(m.group(1)):
                    inputs.append(n)
            for m in _SQL_JOIN_RE.finditer(mt):
                if n := _ds_name(m.group(1)):
                    inputs.append(n)
        else:
            for m in _DATA_OPT_RE.finditer(mt):
                if n := _ds_name(m.group(1)):
                    inputs.append(n)
            for m in _PROC_OUT_RE.finditer(mt):
                raw = m.group(1) or m.group(2) or ""
                if n := _ds_name(raw):
                    outputs.append(n)
            if proc_name == "sort" and not _PROC_OUT_RE.search(mt):
                # in-place sort: DATA= is both input and output
                for m in _DATA_OPT_RE.finditer(mt):
                    if (n := _ds_name(m.group(1))) and n not in outputs:
                        outputs.append(n)

    def _dedup(lst: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return _dedup(inputs), _dedup(outputs), _dedup(defines), _dedup(invokes)
