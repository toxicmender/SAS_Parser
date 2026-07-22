"""The signal catalogue: which SAS constructs imply which complexity tier and
Spark parity. See complexity/README.md.

Pure data: no logging, no imports from the rest of this package beyond the
enums it classifies with. This is the single place to retune the analysis —
:mod:`complexity.analyzer` only looks constructs up here, it never hard-codes a
tier of its own.

Tier assignment follows the project's brief:

- **LOW** — simple SQL and macro variables.
- **MEDIUM** — hashing, MERGE, SFTP, mail, and similar "works, but the
  semantics differ" constructs.
- **HIGH** — arrays, DO loops, and ``%MACRO`` definitions.

Anything not listed here contributes no signal at all, which floors a chunk at
LOW/DIRECT. Silence means "nothing notable found", never "unknown": the
catalogue is deliberately an allowlist of constructs whose translation cost is
understood, so an unrecognised function never inflates a score.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import ComplexityTier, SparkParity


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """The classification attached to one recognised construct.

    Fields mirror :class:`~complexity.models.ComplexitySignal` minus the
    identity the caller supplies (``name``/``evidence``/``source``).
    """

    category: str
    tier: ComplexityTier
    parity: SparkParity
    weight: float
    note: str = ""


# Default per-tier weights. Weight only ranks items *within* a tier — it can
# never change the tier itself, which is presence-based (see analyzer).
WEIGHT_LOW = 1.0
WEIGHT_MEDIUM = 2.5
WEIGHT_HIGH = 5.0


def _low(category: str, parity: SparkParity, note: str = "") -> SignalSpec:
    return SignalSpec(category, ComplexityTier.LOW, parity, WEIGHT_LOW, note)


def _medium(category: str, parity: SparkParity, note: str = "") -> SignalSpec:
    return SignalSpec(category, ComplexityTier.MEDIUM, parity, WEIGHT_MEDIUM, note)


def _high(category: str, parity: SparkParity, note: str = "") -> SignalSpec:
    return SignalSpec(category, ComplexityTier.HIGH, parity, WEIGHT_HIGH, note)


# ---------------------------------------------------------------------------
# PROC steps  — keyed by SasChunkMetadata.proc_name
# ---------------------------------------------------------------------------

PROC_RULES: dict[str, SignalSpec] = {
    # Set-oriented PROCs with a one-for-one Spark equivalent.
    "sql": _low("simple-sql", SparkParity.DIRECT, "PROC SQL maps to spark.sql"),
    "sort": _low("simple-sql", SparkParity.SUPPORTED, "PROC SORT maps to orderBy"),
    "print": _low("reporting", SparkParity.SUPPORTED, "PROC PRINT maps to show"),
    "contents": _low("metadata", SparkParity.SUPPORTED, "schema inspection"),
    "datasets": _low("metadata", SparkParity.SUPPORTED, "catalog maintenance"),
    "append": _low("simple-sql", SparkParity.SUPPORTED, "PROC APPEND maps to union"),
    "means": _low("aggregation", SparkParity.SUPPORTED, "maps to groupBy.agg"),
    "summary": _low("aggregation", SparkParity.SUPPORTED, "maps to groupBy.agg"),
    "freq": _low("aggregation", SparkParity.SUPPORTED, "maps to groupBy.count"),
    # Reshaping and reporting: an equivalent exists but is not mechanical.
    "transpose": _medium(
        "reshape", SparkParity.PARTIAL, "PROC TRANSPOSE needs pivot/stack"
    ),
    "report": _medium("reporting", SparkParity.PARTIAL, "PROC REPORT layout"),
    "tabulate": _medium("reporting", SparkParity.PARTIAL, "PROC TABULATE layout"),
    "format": _medium(
        "format", SparkParity.PARTIAL, "user-defined formats need a lookup"
    ),
    "export": _medium("io", SparkParity.PARTIAL, "external file export"),
    "import": _medium("io", SparkParity.PARTIAL, "external file import"),
    "http": _medium("io-network", SparkParity.PARTIAL, "PROC HTTP call"),
    "soap": _medium("io-network", SparkParity.PARTIAL, "PROC SOAP call"),
    # Procedural / statistical PROCs with no Spark SQL counterpart.
    "fcmp": _high(
        "procedural", SparkParity.MANUAL, "PROC FCMP defines custom functions"
    ),
    "iml": _high("procedural", SparkParity.MANUAL, "PROC IML matrix language"),
    "ds2": _high("procedural", SparkParity.MANUAL, "PROC DS2 program"),
    "lua": _high("procedural", SparkParity.MANUAL, "PROC LUA program"),
}

# ---------------------------------------------------------------------------
# DATA step component objects  — keyed by SasChunkMetadata.component_objects
# ---------------------------------------------------------------------------

COMPONENT_OBJECT_RULES: dict[str, SignalSpec] = {
    "hash": _medium(
        "hashing", SparkParity.PARTIAL, "hash object lookup — becomes a join"
    ),
    "hiter": _medium(
        "hashing", SparkParity.PARTIAL, "hash iterator — becomes an ordered scan"
    ),
    "javaobj": _high("interop", SparkParity.MANUAL, "Java object interop"),
    "logger": _medium("logging", SparkParity.PARTIAL, "SAS logging object"),
    "appender": _medium("logging", SparkParity.PARTIAL, "SAS logging appender"),
}

# ---------------------------------------------------------------------------
# Functions  — keyed by SasChunkMetadata.recognized_functions
#
# Grouped by translation concern rather than listed one-by-one, so the whole
# family shares a spec and new members are picked up by editing one frozenset.
# ---------------------------------------------------------------------------

_HASHING_FUNCTIONS = frozenset(
    {
        "md5",
        "sha256",
        "sha256hex",
        "sha256hmachex",
        "hashing",
        "hashing_file",
        "hashing_hmac",
        "hashing_hmac_file",
        "hashing_hmac_init",
        "hashing_init",
        "hashing_part",
        "hashing_term",
    }
)

# SAS date/datetime *interval* functions. Spark has date arithmetic, but SAS
# interval semantics (SAME/CONTINUOUS alignment, custom intervals, shift
# operators) do not carry over unexamined.
_INTERVAL_FUNCTIONS = frozenset(
    {
        "intnx",
        "intck",
        "intcindex",
        "intcycle",
        "intfit",
        "intfmt",
        "intget",
        "intindex",
        "intnest",
        "intseas",
        "intshift",
        "inttest",
        "holiday",
        "holidayck",
        "holidaycount",
        "holidayname",
        "holidaynx",
        "holidayny",
        "holidaytest",
    }
)

# Row-ordering-dependent functions. LAG looks like Spark's lag() window
# function but is not: SAS Functions and CALL Routines: Reference describes it
# as returning "values from a queue" — "A LAGn function stores a value in a
# queue and returns a value stored previously in that queue. Each occurrence of
# a LAGn function in a program generates its own queue." The queue advances
# only when that call site executes, so a LAG inside a conditional does NOT
# equal lag(col) over an ordered window. SAS's own distributed engine declines
# it ("not supported in a DATA step that runs in CAS"), which is the clearest
# possible signal that inter-row dependency resists distribution.
_ROW_STATE_FUNCTIONS = frozenset({"lag", "dif"})

# Runtime macro resolution — the value is not knowable statically.
_MACRO_RUNTIME_FUNCTIONS = frozenset({"symget", "resolve", "dosubl"})

# Dynamic format/informat application (the format itself is a runtime value).
_DYNAMIC_FORMAT_FUNCTIONS = frozenset({"putn", "putc", "inputn", "inputc"})

# External file I/O via the DATA step's file interface.
_FILE_IO_FUNCTIONS = frozenset(
    {
        "fopen",
        "fclose",
        "fget",
        "fput",
        "fwrite",
        "fread",
        "fappend",
        "fcopy",
        "fdelete",
        "filename",
        "fileexist",
        "fexist",
        "dopen",
        "dclose",
        "dread",
        "dcreate",
        "mopen",
    }
)


def _expand(
    names: frozenset[str], spec: SignalSpec
) -> dict[str, SignalSpec]:
    """Attach *spec* to every name in *names*."""
    return dict.fromkeys(names, spec)


FUNCTION_RULES: dict[str, SignalSpec] = {
    **_expand(
        _HASHING_FUNCTIONS,
        _medium("hashing", SparkParity.PARTIAL, "hashing function"),
    ),
    **_expand(
        _INTERVAL_FUNCTIONS,
        _medium(
            "date-interval",
            SparkParity.PARTIAL,
            "SAS interval semantics differ from Spark date arithmetic",
        ),
    ),
    **_expand(
        _ROW_STATE_FUNCTIONS,
        _high(
            "row-state",
            SparkParity.HARD,
            "per-call-site queue, not 'the previous row' — a conditional LAG "
            "is not lag() over a window",
        ),
    ),
    **_expand(
        _MACRO_RUNTIME_FUNCTIONS,
        _high(
            "macro-runtime",
            SparkParity.MANUAL,
            "resolves macro code at run time",
        ),
    ),
    **_expand(
        _DYNAMIC_FORMAT_FUNCTIONS,
        _medium("format", SparkParity.PARTIAL, "format applied at run time"),
    ),
    **_expand(
        _FILE_IO_FUNCTIONS,
        _medium("io", SparkParity.PARTIAL, "external file I/O"),
    ),
}

# ---------------------------------------------------------------------------
# CALL routines  — keyed by SasChunkMetadata.recognized_call_routines
# ---------------------------------------------------------------------------

CALL_ROUTINE_RULES: dict[str, SignalSpec] = {
    "symput": _medium(
        "macro-var", SparkParity.PARTIAL, "creates a macro variable at run time"
    ),
    "symputx": _medium(
        "macro-var", SparkParity.PARTIAL, "creates a macro variable at run time"
    ),
    "execute": _high(
        "dynamic-code", SparkParity.MANUAL, "CALL EXECUTE generates code at run time"
    ),
    "module": _high("interop", SparkParity.MANUAL, "external routine call"),
    "system": _high("interop", SparkParity.MANUAL, "shells out to the OS"),
    "tso": _high("interop", SparkParity.MANUAL, "shells out to TSO"),
    "sortc": _medium("reshape", SparkParity.PARTIAL, "sorts values across columns"),
    "sortn": _medium("reshape", SparkParity.PARTIAL, "sorts values across columns"),
}

# ---------------------------------------------------------------------------
# Global statements  — keyed by SasChunkMetadata.global_statement_keyword
# ---------------------------------------------------------------------------

GLOBAL_STATEMENT_RULES: dict[str, SignalSpec] = {
    "let": _low("macro-var", SparkParity.DIRECT, "%LET macro variable"),
    "global": _low("macro-var", SparkParity.DIRECT, "%GLOBAL declaration"),
    "local": _low("macro-var", SparkParity.DIRECT, "%LOCAL declaration"),
    "put": _low("logging", SparkParity.SUPPORTED, "%PUT trace"),
    "libname": _low("io", SparkParity.SUPPORTED, "library assignment"),
    "title": _low("reporting", SparkParity.SUPPORTED, "report title"),
    "footnote": _low("reporting", SparkParity.SUPPORTED, "report footnote"),
    "ods": _medium("reporting", SparkParity.PARTIAL, "ODS output destination"),
    # FILENAME is only LOW as a plain path alias; the access-method detectors
    # (SFTP / EMAIL / URL) add their own MEDIUM signal on top.
    "filename": _low("io", SparkParity.SUPPORTED, "file reference"),
}

# ---------------------------------------------------------------------------
# Chunk kinds  — keyed by SasChunkKind.value
# ---------------------------------------------------------------------------

KIND_RULES: dict[str, SignalSpec] = {
    "MACRO_DEFINITION": _high(
        "macro-def",
        SparkParity.MANUAL,
        "%MACRO definition — no Spark equivalent",
    ),
    "MACRO_CALL": _medium(
        "macro-call", SparkParity.PARTIAL, "macro invocation must be resolved"
    ),
    "MACRO_CONTROL_FLOW": _high(
        "macro-control-flow",
        SparkParity.HARD,
        "macro-level control flow generates code conditionally",
    ),
    "INCLUDE": _medium(
        "io", SparkParity.PARTIAL, "%INCLUDE pulls in external source"
    ),
    "UNKNOWN_BLOCK": _medium(
        "unparsed", SparkParity.PARTIAL, "unclosed block — parse is incomplete"
    ),
    "UNKNOWN_STATEMENT_GROUP": _medium(
        "unparsed", SparkParity.PARTIAL, "unrecognised statements"
    ),
}

# ---------------------------------------------------------------------------
# Boolean metadata flags  — (attribute name, spec)
#
# Each entry names a truthy attribute of SasChunkMetadata that, when set,
# contributes its spec. List-valued attributes count as set when non-empty.
# ---------------------------------------------------------------------------

FLAG_RULES: tuple[tuple[str, str, SignalSpec], ...] = (
    (
        "symput_scope_hazard",
        "symput-scope-hazard",
        _high(
            "macro-var",
            SparkParity.MANUAL,
            "CALL SYMPUT scope hazard — the variable may not outlive the step",
        ),
    ),
    (
        "contains_computed_goto",
        "computed-goto",
        _high(
            "macro-control-flow",
            SparkParity.MANUAL,
            "computed %GOTO — control flow is decided at run time",
        ),
    ),
    (
        "contains_abort",
        "abort",
        _medium(
            "macro-control-flow", SparkParity.PARTIAL, "%ABORT terminates the job"
        ),
    ),
    (
        "has_unclosed_block",
        "unclosed-block",
        _medium("unparsed", SparkParity.PARTIAL, "unclosed block"),
    ),
    (
        "defines_macros",
        "defines-macro",
        _high(
            "macro-def", SparkParity.MANUAL, "defines a macro — no Spark equivalent"
        ),
    ),
    (
        "invokes_macros",
        "invokes-macro",
        _medium(
            "macro-call", SparkParity.PARTIAL, "invokes a macro that must be resolved"
        ),
    ),
    (
        "includes",
        "include",
        _medium("io", SparkParity.PARTIAL, "%INCLUDE pulls in external source"),
    ),
    (
        "referenced_macro_vars",
        "macro-var-reference",
        _low("macro-var", SparkParity.DIRECT, "macro variable reference"),
    ),
)

# ---------------------------------------------------------------------------
# Detector-found constructs  — keyed by the detector's construct name
#
# These cover what SasChunkMetadata does not extract: DATA step ARRAY and DO
# statements, the MERGE statement, and FILENAME access methods (SFTP / EMAIL /
# URL). See complexity/detectors.py.
# ---------------------------------------------------------------------------

DETECTOR_RULES: dict[str, SignalSpec] = {
    # NOT a Spark ArrayType column. Essentials Ch. 24 is explicit: "In SAS, an
    # array is not a data structure. An array is just a convenient way of
    # temporarily defining a group of variables." So the plausible-looking
    # mapping to an array column plus explode() is wrong — a SAS array aliases
    # a group of *columns*, and translating it means a wide-to-long
    # restructure or per-column expressions. The note says so explicitly to
    # steer a reader (or an LLM) off the wrong mapping.
    "array": _high(
        "array",
        SparkParity.HARD,
        "ARRAY — aliases a group of columns, not a Spark ArrayType; "
        "needs wide-to-long restructuring, not explode()",
    ),
    "do_loop": _high(
        "do-loop",
        SparkParity.HARD,
        "iterative DO — becomes vectorised columns, explode, or a UDF",
    ),
    "do_while": _high(
        "do-loop", SparkParity.HARD, "DO WHILE — unbounded row-wise iteration"
    ),
    "do_until": _high(
        "do-loop", SparkParity.HARD, "DO UNTIL — unbounded row-wise iteration"
    ),
    # Match-merge: a join, but SAS overlays same-named variables from the
    # last data set read, which a Spark join does not do.
    "merge": _medium(
        "merge",
        SparkParity.PARTIAL,
        "MERGE with BY — a join, but same-named columns overlay differently",
    ),
    # One-to-one merge: no key at all. Essentials Ch. 21 — "There is no key
    # variable on which to base the merge. Instead, rows are merged implicitly
    # by row number." A distributed DataFrame has no inherent row order, so
    # this cannot be expressed as a join; reproducing it means manufacturing an
    # ordering, which is why it rates HIGH rather than MEDIUM.
    "merge_no_by": _high(
        "merge",
        SparkParity.HARD,
        "MERGE without BY — pairs rows by position; no Spark equivalent",
    ),
    "modify": _medium(
        "merge", SparkParity.PARTIAL, "MODIFY updates a dataset in place"
    ),
    "update": _medium(
        "merge", SparkParity.PARTIAL, "UPDATE applies a transaction overlay"
    ),
    "retain": _medium(
        "row-state",
        SparkParity.PARTIAL,
        "RETAIN carries values across rows — needs a window function",
    ),
    "by_group_first_last": _medium(
        "row-state",
        SparkParity.PARTIAL,
        "FIRST./LAST. BY-group flags — need window functions",
    ),
    "filename_sftp": _medium(
        "io-network", SparkParity.PARTIAL, "FILENAME SFTP transfer"
    ),
    "filename_ftp": _medium(
        "io-network", SparkParity.PARTIAL, "FILENAME FTP transfer"
    ),
    "filename_email": _medium(
        "io-network", SparkParity.PARTIAL, "FILENAME EMAIL — sends mail"
    ),
    "filename_url": _medium(
        "io-network", SparkParity.PARTIAL, "FILENAME URL fetch"
    ),
    "filename_socket": _medium(
        "io-network", SparkParity.PARTIAL, "FILENAME SOCKET connection"
    ),
    "filename_pipe": _high(
        "interop", SparkParity.MANUAL, "FILENAME PIPE shells out to the OS"
    ),
    "infile": _medium(
        "io", SparkParity.PARTIAL, "INFILE reads an external raw file"
    ),
    "file_output": _medium(
        "io", SparkParity.PARTIAL, "FILE writes an external raw file"
    ),
    "link_return": _high(
        "procedural", SparkParity.HARD, "LINK/RETURN — procedural subroutine"
    ),
    "data_goto": _high(
        "procedural", SparkParity.HARD, "DATA step GOTO — procedural jump"
    ),
}


# Every catalogue the analyzer consults, for introspection and tests.
ALL_RULES: dict[str, dict[str, SignalSpec]] = {
    "proc": PROC_RULES,
    "component_object": COMPONENT_OBJECT_RULES,
    "function": FUNCTION_RULES,
    "call_routine": CALL_ROUTINE_RULES,
    "global_statement": GLOBAL_STATEMENT_RULES,
    "kind": KIND_RULES,
    "detector": DETECTOR_RULES,
}
