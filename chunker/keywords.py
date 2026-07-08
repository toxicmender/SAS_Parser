"""
keywords.py — SAS keyword catalogues and the patterns compiled from them.

Frozen-set dictionaries transcribed from the SAS documentation — reserved
macro words (Appendix 1), standard autocall macros, the DATA-step
function and CALL-routine dictionaries, and the reserved dataset-name
tokens — plus the regexes built from those alternations: the macro-call
detectors (_MACRO_CALL_RE / _MACRO_INVOKE_RE) and the function /
CALL-routine scanners.  Pure data: no logging, no imports from the rest
of the package.  Every provenance note stays with its constant below.
"""

from __future__ import annotations

import re

import regex


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

# A function call is ``name(`` (optional whitespace before the paren) where
# the name is not glued to a preceding ``%`` (the macro-language counterpart —
# ``%scan(...)``, ``%put (...)`` — or a user macro that happens to share a
# function's name, ``%compress(&ds)``), ``&`` (a macro-variable reference),
# or ``.`` (a hash-object method call ``h.find()`` or an ``&pfx.name``
# concatenation).  A CALL routine is ``CALL name`` followed by a word
# boundary.  Both alternations are built longest-name-first so a shorter name
# that prefixes a longer one can't short-circuit the match, mirroring
# _RESERVED_WORDS_PATTERN's construction.
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
    rf"(?<![%&.\w])({_SAS_FUNCTIONS_PATTERN})\b\s*\(",
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
