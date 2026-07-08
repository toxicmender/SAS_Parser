"""
metadata.py — per-chunk semantic metadata extraction for the SAS chunker.

Everything that turns a chunk's raw text into SasChunkMetadata:

- _metadata_for / _merge_meta / _title — called by chunker.py per region
  (and per oversized-split child)
- _io_for — directed dataset I/O and macro define/invoke edges
- _macro_body_io — literal vs parameterised macro-body dataset refs
- _extract_symput / _extract_sql_into_vars / _extract_call_execute_macros
  — macro-variable producers and CALL EXECUTE invocations
- the extraction regex catalogue (dataset positions, SQL clauses, quoted
  physical paths, %LET/%GLOBAL/%LOCAL forms, ...)

All scans run on sanitised text from scanner._sanitise; keyword-derived
patterns come from keywords.py.

Logging
-------
Logger: ``chunker.metadata`` — DEBUG only (per-macro body-IO decisions).
"""

from __future__ import annotations

import logging
import re

from .keywords import (
    _MACRO_CALL_RE,
    _MACRO_INVOKE_RE,
    _SAS_CALL_ROUTINE_RE,
    _SAS_FUNCTION_CALL_RE,
    _SAS_RESERVED,
)
from .models import SasChunkKind, SasChunkMetadata
from .scanner import _blank_span, _sanitise

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
# The libref a LIBNAME statement assigns: ``libname <ref> ...``.  Also matches
# the deassignment form ``libname <ref> clear;`` — static extraction is
# positional, not temporal, so a cleared libref is still reported as defined
# (documented on SasChunkMetadata.defines_librefs).  ``libname _all_ clear|list``
# targets every assigned libref rather than naming one; the caller drops it.
_LIBNAME_REF_RE = re.compile(r"\blibname\s+([A-Za-z_]\w*)", re.IGNORECASE)
_MACRO_DEF_RE = re.compile(r"%\s*macro\s+([A-Za-z_]\w*)", re.IGNORECASE)


# INPUT/PUT *statement* grouped-list form — ``input (var-list) (informat-list)``
# / ``put (var-list) (format-list)``.  The keyword directly followed by two
# back-to-back parenthesised groups is never valid function-call syntax (a
# call's argument list is a single group), so the keyword can be blanked ahead
# of the function scan without touching genuine INPUT()/PUT() calls — including
# SQL's ``case when ... then put(x, fmt.)``, whose single group never matches.
_GROUPED_INPUT_PUT_STMT_RE = re.compile(
    r"\b(?:input|put)\b(?=\s*\([^()]*\)\s*\()",
    re.IGNORECASE,
)


def _function_scan_text(mt: str) -> str:
    """Blank the spans of sanitised text *mt* that textually look like
    ``name(`` but are not function calls, so _SAS_FUNCTION_CALL_RE /
    _SAS_CALL_ROUTINE_RE don't misreport them: ``%macro name(...)`` definition
    headers (a macro *named* like a function) and grouped-list INPUT/PUT
    statements (see _GROUPED_INPUT_PUT_STMT_RE)."""
    mt = _MACRO_DEF_RE.sub(lambda m: _blank_span(m.group(0)), mt)
    return _GROUPED_INPUT_PUT_STMT_RE.sub(lambda m: " " * len(m.group(0)), mt)


_INCLUDE_RE = re.compile(r"%\s*include\s+([^;]+)", re.IGNORECASE)
_OPTIONS_RE = re.compile(r"\boptions\s+([^;]+)", re.IGNORECASE)
_LABEL_RE = re.compile(r"\blabel\s+([A-Za-z_]\w*)\s*=", re.IGNORECASE)
_PROC_RE = re.compile(r"\bproc\s+([A-Za-z_]\w*)", re.IGNORECASE)
_DATA_RE = re.compile(
    r"\bdata\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)",
    re.IGNORECASE,
)


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

# Any "&name" or "&name." reference — the single stored scan feeding
# SasChunkMetadata.referenced_macro_vars.  Deliberately broad (matches every
# macro-variable reference, not just &sys*): the automatic-variable and
# consumer views are derived from the stored set by computed fields on the
# model (see models._is_automatic_macro_var).
_VAR_REF_RE = re.compile(r"&(\w+)\.?")


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
    # Dataset names collected from *dataset positions only* — the DATA/SET/
    # MERGE/UPDATE/MODIFY keywords, DATA=/OUT=/OUTDATA= options, and PROC SQL
    # CREATE/FROM/JOIN/INSERT clauses.  An earlier blanket ``word.word`` scan
    # also swept up PROC SQL table aliases (``l.id`` → libref ``l``) and
    # BY-group temporaries (``first.grp`` → libref ``first``); restricting the
    # scan to dataset context removes those false positives from both
    # ``referenced_datasets`` and ``referenced_librefs``.
    datasets = [_nid(m.group(1)) for m in _DATASET_RE.finditer(mt)]
    datasets += [_nid(m.group(1)) for m in _DATA_OPT_RE.finditer(mt)]
    datasets += [_nid(m.group(1)) for m in _SQL_CREATE_RE.finditer(mt)]
    datasets += [_nid(m.group(1)) for m in _SQL_FROM_RE.finditer(mt)]
    datasets += [_nid(m.group(1)) for m in _SQL_JOIN_RE.finditer(mt)]
    datasets += [_nid(m.group(1)) for m in _SQL_INTO_RE.finditer(mt)]
    datasets += [_nid(m.group(1) or m.group(2)) for m in _PROC_OUT_RE.finditer(mt)]
    # The directed I/O extraction parses the *full* dataset lists (a DATA
    # header or SET/MERGE statement may name several datasets; _DATASET_RE
    # above captures only the first), so its canonical names complete
    # ``referenced_datasets``.  One-level names therefore appear here in
    # their canonical ``work.``-qualified spelling.
    inp, out, defs, invk = _io_for(text, kind, mt, cf)
    dataset_set = set(datasets) | set(inp) | set(out)
    # Librefs this chunk assigns (``libname ref ...``); ``_all_`` targets every
    # assigned libref (``libname _all_ clear|list;``) rather than naming one.
    defines_librefs = sorted(
        {_nid(m.group(1)) for m in _LIBNAME_REF_RE.finditer(mt)} - {"_all_"}
    )
    # Referenced librefs: the libref part of every two-level dataset-context
    # name, plus any libref assigned here.  Quoted physical-path references
    # address a file directly and carry no libref.
    librefs = sorted(
        {
            d.split(".", 1)[0]
            for d in dataset_set
            if "." in d and not d.startswith("'")
        }
        | set(defines_librefs)
    )
    includes = [_nid(m.group(1)).strip("'\"") for m in _INCLUDE_RE.finditer(cf)]
    options = [_nid(p) for m in _OPTIONS_RE.finditer(mt) for p in m.group(1).split()]
    labels = sorted({_nid(m.group(1)) for m in _LABEL_RE.finditer(mt)})
    pm = _PROC_RE.search(mt)
    dm = _DATA_RE.search(mt)
    mm = _MACRO_DEF_RE.search(mt)

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

    # ── macro-variable references (single stored scan) ─────────────────────
    # Every "&name" reference in this chunk, automatic (&sys*) variables
    # included.  Scanned on `cf` (quotes preserved, comments stripped) rather
    # than `mt` so references inside double-quoted strings — e.g. title
    # "Report run &sysday" — are still caught.  SAS only resolves macro
    # variables inside double-quoted strings, never single-quoted ones;
    # distinguishing that here would add real complexity for marginal
    # benefit, so this also matches (harmlessly) inside single-quoted text.
    # The automatic-variable subset and the dependency-oriented consumer view
    # (automatics and own-parameters excluded) are computed fields on
    # SasChunkMetadata, derived from this one scan.
    referenced_macro_vars = sorted(
        {m.group(1).lower() for m in _VAR_REF_RE.finditer(cf)}
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

    # ── recognised SAS functions and CALL routines ──────────────────────────
    # Scanned on `mt` (string literals blanked) so a function-like token inside
    # a quoted string isn't mistaken for a real call, with %MACRO definition
    # headers and grouped-list INPUT/PUT statements blanked on top
    # (_function_scan_text) so those non-call ``name(`` shapes don't register.
    ft = _function_scan_text(mt)
    recognized_functions = sorted(
        {m.group(1).lower() for m in _SAS_FUNCTION_CALL_RE.finditer(ft)}
    )
    recognized_call_routines = sorted(
        {m.group(1).lower() for m in _SAS_CALL_ROUTINE_RE.finditer(ft)}
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
        referenced_datasets=sorted(dataset_set),
        defines_librefs=defines_librefs,
        includes=includes,
        options=options,
        has_unclosed_block=(kind == SasChunkKind.UNKNOWN_BLOCK),
        macro_var_op=var_op,
        global_statement_keyword=global_stmt_kw,
        declared_macro_vars=declared_macro_vars,
        referenced_macro_vars=referenced_macro_vars,
        recognized_functions=recognized_functions,
        recognized_call_routines=recognized_call_routines,
        control_flow_op=control_flow_op,
        contains_abort=has_abort,
        contains_computed_goto=has_computed_goto,
        input_datasets=inp,
        output_datasets=out,
        defines_macros=sorted(set(defs)),
        invokes_macros=sorted(set(invk)),
        body_literal_inputs=body_lit_in,
        body_literal_outputs=body_lit_out,
        body_param_inputs=body_par_in,
        body_param_outputs=body_par_out,
        macro_param_names=param_names,
        produces_macrovars=sorted(set(produces_macrovars)),
        symput_scope_hazard=hazard,
        symput_hazard_vars=sorted(set(hazard_vars)),
    )


# Fields where the parent's (whole-region) value is authoritative and the
# child's is only a fallback, instead of the two being unioned.  All three
# derive from the %MACRO signature header, which only the split slice that
# contains it can parse — for every other slice the extractor returns an
# empty list, and unioning positional {"param", "pos"} dicts from partial
# views could double-count or (for the unhashable dicts) not union at all.
_MERGE_PARENT_WINS = frozenset(
    {
        "body_param_inputs",
        "body_param_outputs",
        "macro_param_names",
    }
)


def _merge_meta(parent: SasChunkMetadata, child: SasChunkMetadata) -> SasChunkMetadata:
    """Merge a split child's metadata with its parent region's metadata.

    Driven by ``SasChunkMetadata.model_fields`` so a newly added field is
    merged by its type automatically instead of being silently dropped:

    - ``list[str]``   → sorted union of both sides;
    - ``bool``        → OR (a flag raised anywhere in the region stays raised);
    - ``str | None``  → child's value, falling back to the parent's (the
      child is the more specific view of its own slice);
    - ``_MERGE_PARENT_WINS`` fields → parent's value, falling back to the
      child's (signature-derived fields; see the constant above).

    Any other annotation raises ``TypeError`` at merge time — every test
    that exercises an oversized split trips it — forcing the author of a
    new field shape to pick a rule rather than inherit a wrong default.
    Computed fields derive from their stored inputs and are not merged.
    """
    merged: dict[str, object] = {}
    for name, field in SasChunkMetadata.model_fields.items():
        p = getattr(parent, name)
        c = getattr(child, name)
        if name in _MERGE_PARENT_WINS:
            merged[name] = p or c
        elif field.annotation == list[str]:
            merged[name] = sorted({*p, *c})
        elif field.annotation is bool:
            merged[name] = p or c
        elif field.annotation == (str | None):
            merged[name] = c or p
        else:
            raise TypeError(
                f"SasChunkMetadata.{name}: no merge rule for annotation "
                f"{field.annotation!r} — add a branch to _merge_meta or an "
                f"entry to _MERGE_PARENT_WINS"
            )
    return SasChunkMetadata(**merged)


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
# The inner token quantifiers below are *possessive* (``*+``): the dataset-list
# group ``(?:token (opts)? ws)+?`` is a nested quantifier whose inner ``\w*`` /
# ``\s*`` can re-partition the same text many ways.  With a plain greedy/lazy
# star, a ``set``/``merge`` header that never reaches its terminating ``;`` (or
# BY/WHERE/OBS keyword) — e.g. a source line with a dropped semicolon — makes
# the engine explore those partitions exponentially (catastrophic backtracking,
# which the parse deadline cannot interrupt since it is one un-interruptible
# C-level regex call).  Possessive stars commit each token/whitespace run once
# and never give it back, so a failing match degrades to O(n).  Only the outer
# ``+?`` stays lazy — it must still stop at the first terminator so a trailing
# BY/WHERE/OBS keyword isn't swallowed into the dataset list.
_SET_RE = re.compile(
    r"\bset\s++((?:[A-Za-z_]\w*+(?:\.[A-Za-z_]\w*+)?(?:\s*+\([^)]*\))?\s*+)+?)"
    r"(?=;|\bwhere\b|\bby\b|\bobs\b|\bnobs\b)",
    re.IGNORECASE | re.DOTALL,
)
_MERGE_RE = re.compile(
    r"\bmerge\s++((?:[A-Za-z_]\w*+(?:\.[A-Za-z_]\w*+)?(?:\s*+\([^)]*\))?\s*+)+?)"
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
#
# Quoted physical-path dataset references (SAS Programmer's Guide: Essentials,
# Ch. 11 "Accessing Data without Using a Libref"): a data set may be addressed
# by its quoted path instead of a libref, e.g. ``data 'c:/tmp/perm';`` or
# ``proc print data='c:/tmp/perm';``.  String contents are blanked in ``mt``,
# so these MUST be scanned on the quotes-preserved form (``cf``).  The header
# form uses the same ``(?<![\w=])`` guard as _BODY_DATA_HDR_RE so ``data=``
# options don't match as DATA statements.
_QUOTED_DATA_HDR_RE = re.compile(
    r"(?<![\w=])data\s+((['\"])[^'\";]+\2)", re.IGNORECASE
)
_QUOTED_SET_MERGE_RE = re.compile(
    r"\b(?:set|merge)\s+((['\"])[^'\";]+\2)", re.IGNORECASE
)
_QUOTED_DATA_OPT_RE = re.compile(
    r"\bdata\s*=\s*((['\"])[^'\";]+\2)", re.IGNORECASE
)
_QUOTED_OUT_OPT_RE = re.compile(
    r"\bout\s*=\s*((['\"])[^'\";]+\2)", re.IGNORECASE
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

# DATA statement header inside a macro body (may contain &refs).
# Inner token/whitespace stars are possessive (``*+``) for the same
# catastrophic-backtracking reason documented at _SET_RE above: these run over
# a whole (possibly malformed) macro body, so a ``data``/``set``/``merge`` header
# with no reachable terminator must fail in O(n), not exponentially.  The outer
# ``+?`` stays lazy to preserve the "stop at the first ; / keyword" capture.
_BODY_DATA_HDR_RE = re.compile(
    r"(?<![\w=])data\s++((?:[A-Za-z_&][\w.&]*+(?:\s*+\([^)]*+\))?\s*+)+?)(?=;)",
    re.IGNORECASE,
)
_BODY_SET_RE = re.compile(
    r"\bset\s++((?:[A-Za-z_&][\w.&]*+(?:\s*+\([^)]*\))?\s*+)+?)(?=;|\bwhere\b|\bby\b|\bobs\b)",
    re.IGNORECASE | re.DOTALL,
)
_BODY_MERGE_RE = re.compile(
    r"\bmerge\s++((?:[A-Za-z_&][\w.&]*+(?:\s*+\([^)]*\))?\s*+)+?)(?=;|\bby\b)",
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
        cleaned = _PAREN_RE.sub(" ", m.group(1))
        for tok in _AMP_TOKEN_RE.findall(cleaned):
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
                key = _canon_ds(key)
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


def _canon_ds(name: str) -> str:
    """Canonicalise a dataset name for producer/consumer matching.

    A one-level name resolves to the temporary Work library — per the SAS
    Programmer's Guide: Essentials (Ch. 11), ``data mytable;`` "behaves the
    same if you specify work.mytable" — so it is rewritten to
    ``work.<name>``, unifying both spellings in the batcher's exact-string
    dataset namespace.  Everything that is not a plain one-level identifier
    passes through unchanged:

    - two-level ``libref.member`` names;
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
        or name.startswith("'")
        or (name.startswith("_") and name.endswith("_"))
    ):
        return name
    return f"work.{name}"


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


def _multi_ds(match_group: str) -> list[str]:
    cleaned = _PAREN_RE.sub(" ", match_group)
    tokens = _DS_TOKEN_RE.findall(cleaned)
    return [_canon_ds(n) for t in tokens if (n := _ds_name(t))]


def _io_for(
    text: str,
    kind: SasChunkKind,
    mt: str | None = None,
    cf: str | None = None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Extract directed data-flow edges from a single chunk's source text.

    ``mt`` is the sanitised (comments/strings blanked) form of ``text`` and
    ``cf`` the comments-only-blanked form (string literals intact — needed
    for quoted physical-path dataset references, which live *inside* string
    delimiters); callers that already have them pass them in to avoid
    redundant sanitise passes.  When omitted they are computed here.

    All extracted names are canonicalised via :func:`_canon_ds`, so a
    one-level name and its ``work.``-qualified spelling land in the same
    producer/consumer namespace.

    Returns
    -------
    (input_datasets, output_datasets, defines_macros, invokes_macros)
    """
    if mt is None:
        mt = _sanitise(text)
    if cf is None:
        cf = _sanitise(text, blank_strings=False)

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
        outputs.extend(_multi_ds(header_body))
        for m in _OUTPUT_DS_RE.finditer(mt):
            if n := _ds_name(m.group(1)):
                outputs.append(_canon_ds(n))
        for m in _SET_RE.finditer(mt):
            inputs.extend(_multi_ds(m.group(1)))
        for m in _MERGE_RE.finditer(mt):
            inputs.extend(_multi_ds(m.group(1)))
        for m in _UPDATE_RE.finditer(mt):
            if n := _ds_name(m.group(1)):
                inputs.append(_canon_ds(n))
                outputs.append(_canon_ds(n))
        for m in _MODIFY_RE.finditer(mt):
            if n := _ds_name(m.group(1)):
                inputs.append(_canon_ds(n))
                outputs.append(_canon_ds(n))
        # Quoted physical-path forms: ``data '<path>';`` header (output) and
        # ``set|merge '<path>'`` (input) — scanned on cf, not mt.
        for m in _QUOTED_DATA_HDR_RE.finditer(cf):
            outputs.append(_quoted_path(m.group(1)))
        for m in _QUOTED_SET_MERGE_RE.finditer(cf):
            inputs.append(_quoted_path(m.group(1)))

    elif kind == SasChunkKind.PROC_STEP:
        proc_m = _PROC_RE.search(mt)
        proc_name = proc_m.group(1).lower() if proc_m else ""

        if proc_name == "sql":
            for m in _SQL_CREATE_RE.finditer(mt):
                if n := _ds_name(m.group(1)):
                    outputs.append(_canon_ds(n))
            for m in _SQL_INTO_RE.finditer(mt):
                if n := _ds_name(m.group(1)):
                    outputs.append(_canon_ds(n))
            for m in _SQL_FROM_RE.finditer(mt):
                if n := _ds_name(m.group(1)):
                    inputs.append(_canon_ds(n))
            for m in _SQL_JOIN_RE.finditer(mt):
                if n := _ds_name(m.group(1)):
                    inputs.append(_canon_ds(n))
        else:
            for m in _DATA_OPT_RE.finditer(mt):
                if n := _ds_name(m.group(1)):
                    inputs.append(_canon_ds(n))
            for m in _PROC_OUT_RE.finditer(mt):
                raw = m.group(1) or m.group(2) or ""
                if n := _ds_name(raw):
                    outputs.append(_canon_ds(n))
            # Quoted physical-path options: DATA='<path>' (input) and
            # OUT='<path>' (output) — scanned on cf, not mt.
            for m in _QUOTED_DATA_OPT_RE.finditer(cf):
                inputs.append(_quoted_path(m.group(1)))
            for m in _QUOTED_OUT_OPT_RE.finditer(cf):
                outputs.append(_quoted_path(m.group(1)))
            if proc_name == "sort" and not _PROC_OUT_RE.search(mt):
                # in-place sort: DATA= is both input and output
                for m in _DATA_OPT_RE.finditer(mt):
                    if (n := _ds_name(m.group(1))) and (
                        cn := _canon_ds(n)
                    ) not in outputs:
                        outputs.append(cn)

    def _dedup(lst: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return _dedup(inputs), _dedup(outputs), _dedup(defines), _dedup(invokes)
