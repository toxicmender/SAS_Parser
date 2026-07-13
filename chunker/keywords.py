"""SAS keyword catalogues and the patterns compiled from them. See chunker/README.md.

Pure data: no logging, no imports from the rest of the package.
"""

from __future__ import annotations

import re


# Reserved words — SAS Macro Language: Reference, Appendix 1 (94 words, verbatim).
# None can validly be a user-defined macro name, so every macro-call detector
# must exclude all of them.
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

# Genuine macro functions (SAS Macro Language: Reference, Ch. 12 Table 12.3)
# that are absent from Appendix 1. Kept separate so _RESERVED_WORDS stays
# "Appendix 1 verbatim", while the exclusion mechanism still covers the full set.
_ADDITIONAL_MACRO_FUNCTION_WORDS = frozenset(
    {
        "sysmacexec",
        "sysmacexist",
        "sysmexecdepth",
        "sysmexecname",
        "sysprod",
    }
)

# Longest words first so the alternation doesn't short-circuit on a shorter
# word that is a prefix of a longer one.
_RESERVED_WORDS_PATTERN = "|".join(
    re.escape(w)
    for w in sorted(
        _RESERVED_WORDS | _ADDITIONAL_MACRO_FUNCTION_WORDS,
        key=len,
        reverse=True,
    )
)

# Standard SAS-provided autocall macros (SAS Macro Language: Reference, Ch. 12
# Table 12.13). These ARE genuine callable macro names, so this set is NOT
# folded into _RESERVED_WORDS_PATTERN; batcher.py uses it to exclude them from a
# batch's required_macros while still reporting them via
# SasBatch.standard_autocall_macros.
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


# SAS DATA-step functions and CALL routines — SAS 9.4 Functions and CALL
# Routines: Reference (dictionary entry titles, lower-cased, CALL prefix
# stripped). Advisory metadata only; never gate chunking or batching.
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

# DATA step component objects — SAS Programmer's Guide: Essentials, Ch. 9
# ("DATA Step Component Objects"): hash, hash iterator (HITER), Java, logger,
# and appender objects. Usage is keyed on the declaration syntax
# (``DECLARE``/``DCL`` statement or the ``_NEW_`` operator) because the
# objects' dot-method calls (``h1.definekey(...)``, ``h1.find()``) are member
# access, which the function scan deliberately ignores.
_SAS_COMPONENT_OBJECTS = frozenset(
    {"hash", "hiter", "javaobj", "logger", "appender"}
)
_SAS_COMPONENT_OBJECTS_PATTERN = "|".join(
    re.escape(w) for w in sorted(_SAS_COMPONENT_OBJECTS, key=len, reverse=True)
)
_SAS_COMPONENT_OBJECT_RE = re.compile(
    rf"\b(?:declare|dcl|_new_)\s+({_SAS_COMPONENT_OBJECTS_PATTERN})\b",
    re.IGNORECASE,
)

# A function call is ``name(`` where the name is not glued to a preceding ``%``,
# ``&``, or ``.``; a CALL routine is ``CALL name`` at a word boundary. Both
# scans capture the *generic* identifier token and leave the "is it a known
# SAS function/routine?" test to a frozenset lookup in the consumer
# (metadata.py filters against _SAS_FUNCTIONS / _SAS_CALL_ROUTINES). A literal
# alternation of the ~600 names costs O(names × text) per scan and dominated
# the per-chunk metadata profile; the generic-token + set-filter form is
# match-for-match identical because the alternation's longest-first order and
# trailing ``\b`` already restricted matches to whole identifier tokens.
_SAS_FUNCTION_CALL_RE = re.compile(r"(?<![%&.\w])([A-Za-z_]\w*)\s*\(")
# The routine name is captured in a zero-width lookahead so a rejected token is
# not consumed: in ``call call scan(...)`` the scan resumes right after the
# first ``call`` and still finds the genuine ``call scan``.
_SAS_CALL_ROUTINE_RE = re.compile(
    r"\bcall\s+(?=([A-Za-z_]\w*))",
    re.IGNORECASE,
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
