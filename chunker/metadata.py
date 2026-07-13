"""Per-chunk semantic metadata extraction for the SAS chunker. See chunker/README.md.

All scans run on sanitised text from scanner._sanitise; keyword-derived patterns
come from keywords.py.

Logger name: ``chunker.metadata``.
"""

from __future__ import annotations

import logging
import re

from .keywords import (
    _MACRO_CALL_RE,
    _MACRO_INVOKE_RE,
    _SAS_CALL_ROUTINE_RE,
    _SAS_CALL_ROUTINES,
    _SAS_COMPONENT_OBJECT_RE,
    _SAS_FUNCTION_CALL_RE,
    _SAS_FUNCTIONS,
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
# The libref a LIBNAME statement assigns (``libname <ref> ...``). Extraction is
# positional, not temporal, so ``libname x clear;`` still reports x. ``_all_``
# targets every assigned libref rather than naming one; the caller drops it.
_LIBNAME_REF_RE = re.compile(r"\blibname\s+([A-Za-z_]\w*)", re.IGNORECASE)
_MACRO_DEF_RE = re.compile(r"%\s*macro\s+([A-Za-z_]\w*)", re.IGNORECASE)


# INPUT/PUT *statement* grouped-list form — the keyword followed by two
# back-to-back parenthesised groups, never valid function-call syntax, so it can
# be blanked ahead of the function scan without touching real INPUT()/PUT() calls.
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


# Which of %let/%global/%local/%put begins a GLOBAL_STATEMENT chunk, matched
# against the start of its sanitised text.
_MACRO_VAR_OP_RE = re.compile(r"%\s*(let|global|local|put)\b", re.IGNORECASE)

# Leading statement keyword of a GLOBAL_STATEMENT chunk. ``title``/``footnote``
# capture without their optional occurrence digit (title2 -> title).
_GLOBAL_STMT_KW_RE = re.compile(
    r"%?\s*(let|put|global|local|libname|filename|title|footnote|ods)\b",
    re.IGNORECASE,
)

# ``%LET name`` target. The optional leading ``&`` covers indirect
# (double-ampersand-resolved) targets whose outer name is still literal.
_LET_TARGET_RE = re.compile(r"%\s*let\s+&*([A-Za-z_]\w*)", re.IGNORECASE)

# ``%GLOBAL``/``%LOCAL`` declaration list up to the terminating semicolon; the
# caller splits on whitespace/commas.
_GLOBAL_LOCAL_DECL_RE = re.compile(
    r"%\s*(?:global|local)\s+([^;]+?)\s*;",
    re.IGNORECASE,
)

# Which control-flow keyword begins a MACRO_CONTROL_FLOW chunk, matched against
# the start of its sanitised text (mirrors _MACRO_VAR_OP_RE).
_CONTROL_FLOW_OP_RE = re.compile(
    r"%\s*(if|else|do|end|return|goto|abort)\b",
    re.IGNORECASE,
)

# Shared precompiled token/paren helpers reused across the extractors below.
_PAREN_RE = re.compile(r"\([^)]*\)")  # a balanced-free "(...)" span to blank out
_DS_TOKEN_RE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?")  # libref.member token
_AMP_TOKEN_RE = re.compile(r"[A-Za-z_&][\w.&]*")  # dataset token that may hold &refs
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")  # bare SAS identifier
_SPLIT_WS_COMMA_RE = re.compile(r"[,\s]+")  # %global/%local list separator
_DATA_HDR_STRIP_RE = re.compile(r"^\s*data\s+", re.IGNORECASE)  # drop DATA keyword
_NUM_SUFFIX_RE = re.compile(r"^([A-Za-z_]+?)(\d+)$")  # split trailing integer

# Any "&name" or "&name." reference — the single stored scan feeding
# SasChunkMetadata.referenced_macro_vars (the automatic-variable and consumer
# views are computed from it).
_VAR_REF_RE = re.compile(r"&(\w+)\.?")


# Macro-variable producer/consumer extraction: CALL SYMPUT/SYMPUTX and PROC SQL
# INTO create a macro variable as a side effect rather than via %LET.


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

# Detects %LOCAL anywhere in a macro body (scope-hazard check: a %LOCAL makes
# the local symbol table non-empty, like a declared parameter).
_LOCAL_STMT_RE = re.compile(r"%\s*local\b", re.IGNORECASE)


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


# Captures a %GOTO statement's label. A *computed* %GOTO (label contains "&" or
# "%") forces CALL SYMPUT/SYMPUTX into local scope (Ch. 5).
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
    # Lowercased copies used only to gate keyword scans: every gated pattern
    # contains its keyword as a contiguous case-insensitive literal, so
    # ``keyword in low`` is a necessary condition for any match and skipping
    # the scan when it fails cannot change the result. The substring test is a
    # single memchr-style pass — far cheaper than a regex scan that starts
    # with \b or a lookbehind (which the engine cannot literal-prefix skip).
    low = mt.lower()
    lowcf = cf.lower()
    # Dataset names collected from dataset positions only (DATA/SET/MERGE/UPDATE/
    # MODIFY keywords, DATA=/OUT=/OUTDATA= options, PROC SQL clauses) so table
    # aliases and BY-group temporaries aren't mistaken for datasets/librefs.
    datasets = [_nid(m.group(1)) for m in _DATASET_RE.finditer(mt)]
    datasets += [_nid(m.group(1)) for m in _DATA_OPT_RE.finditer(mt)]
    if "create" in low:
        datasets += [_nid(m.group(1)) for m in _SQL_CREATE_RE.finditer(mt)]
    if "from" in low:
        datasets += [_nid(m.group(1)) for m in _SQL_FROM_RE.finditer(mt)]
    if "join" in low:
        datasets += [_nid(m.group(1)) for m in _SQL_JOIN_RE.finditer(mt)]
    if "insert" in low:
        datasets += [_nid(m.group(1)) for m in _SQL_INTO_RE.finditer(mt)]
    if "out" in low:
        datasets += [
            _nid(m.group(1) or m.group(2)) for m in _PROC_OUT_RE.finditer(mt)
        ]
    # Directed I/O parses the full dataset lists (_DATASET_RE above captures only
    # the first of a multi-dataset statement), so its canonical names complete
    # referenced_datasets in their work.-qualified spelling.
    inp, out, defs, invk = _io_for(text, kind, mt, cf)
    dataset_set = set(datasets) | set(inp) | set(out)
    # Librefs this chunk assigns; ``_all_`` targets every assigned libref.
    defines_librefs = sorted(
        {_nid(m.group(1)) for m in _LIBNAME_REF_RE.finditer(mt)} - {"_all_"}
        if "libname" in low
        else set()
    )
    # Referenced librefs: the libref part of every two-level name, plus any
    # assigned here. Quoted physical paths carry no libref.
    librefs = sorted(
        {
            d.split(".", 1)[0]
            for d in dataset_set
            if "." in d and not d.startswith("'")
        }
        | set(defines_librefs)
    )
    includes = (
        [_nid(m.group(1)).strip("'\"") for m in _INCLUDE_RE.finditer(cf)]
        if "include" in lowcf
        else []
    )
    options = (
        [_nid(p) for m in _OPTIONS_RE.finditer(mt) for p in m.group(1).split()]
        if "options" in low
        else []
    )
    labels = (
        sorted({_nid(m.group(1)) for m in _LABEL_RE.finditer(mt)})
        if "label" in low
        else []
    )
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

    # ── control-flow operation (only set for MACRO_CONTROL_FLOW chunks) ─────
    control_flow_op: str | None = None
    if kind == SasChunkKind.MACRO_CONTROL_FLOW:
        cf_m = _CONTROL_FLOW_OP_RE.match(mt.lstrip())
        if cf_m:
            control_flow_op = cf_m.group(1).lower()

    # ── macro-variable references (single stored scan) ─────────────────────
    # Scanned on `cf` (quotes preserved) so &refs inside quoted strings are
    # caught. The automatic-variable and consumer views are computed from this.
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
            _macro_body_io(text, mt, cf)
        )

    # ── high-severity control-flow visibility (MACRO_DEFINITION bodies) ─────
    has_abort = False
    has_computed_goto = False
    if kind == SasChunkKind.MACRO_DEFINITION:
        # Scanned on the raw text (comments included), so gate on its lowering.
        lowtext = text.lower()
        has_abort = "abort" in lowtext and bool(_ABORT_STMT_RE.search(text))
        has_computed_goto = "goto" in lowtext and _macro_contains_computed_goto(
            text
        )

    # ── macro-variable producer/consumer edges (ROADMAP Phase 2) ────────────
    produces_macrovars: list[str] = []
    hazard: bool = False
    hazard_vars: list[str] = []

    if kind in {SasChunkKind.DATA_STEP, SasChunkKind.MACRO_DEFINITION}:
        symput_names, _unresolved, explicit_global = (
            _extract_symput(cf) if "symput" in lowcf else ([], False, [])
        )
        produces_macrovars.extend(symput_names)
        if "execute" in lowcf:
            invk.extend(_extract_call_execute_macros(cf))

        if kind == SasChunkKind.MACRO_DEFINITION and symput_names:
            has_local = _macro_has_local_scope(text, param_names)
            if has_local:
                at_risk = [n for n in symput_names if n not in explicit_global]
                if at_risk:
                    hazard = True
                    hazard_vars = at_risk

    elif (
        kind == SasChunkKind.PROC_STEP
        and pm
        and _nid(pm.group(1)) == "sql"
        and "into" in lowcf
    ):
        produces_macrovars.extend(_extract_sql_into_vars(cf))

    # ── macro-language-level declarations (%LET and %GLOBAL/%LOCAL lists) ───
    declared: list[str] = [m.group(1).lower() for m in _LET_TARGET_RE.finditer(cf)]
    for m in _GLOBAL_LOCAL_DECL_RE.finditer(cf):
        for name in _SPLIT_WS_COMMA_RE.split(m.group(1).strip()):
            name = name.lstrip("&").rstrip(".")
            if _IDENT_RE.fullmatch(name):
                declared.append(name.lower())
    declared_macro_vars = sorted(set(declared))

    # ── recognised SAS functions and CALL routines ──────────────────────────
    # Scanned on `mt` (string literals blanked), with %MACRO headers and
    # grouped-list INPUT/PUT blanked on top so non-call ``name(`` don't register.
    ft = _function_scan_text(mt)
    # The patterns capture any identifier token in call position; membership in
    # the keyword catalogues decides what is *recognized* (see keywords.py).
    recognized_functions = sorted(
        {
            name
            for m in _SAS_FUNCTION_CALL_RE.finditer(ft)
            if (name := m.group(1).lower()) in _SAS_FUNCTIONS
        }
    )
    recognized_call_routines = sorted(
        {
            name
            for m in _SAS_CALL_ROUTINE_RE.finditer(ft)
            if (name := m.group(1).lower()) in _SAS_CALL_ROUTINES
        }
    )
    # A ``CALL name(...)`` invocation also textually matches the function-call
    # pattern (``name(``); drop those so a routine isn't double-reported as a
    # function of the same name.
    recognized_functions = [
        f for f in recognized_functions if f not in recognized_call_routines
    ]

    # ── DATA step component objects (hash, hiter, javaobj, logger, appender) ─
    # Keyed on the DECLARE/DCL/_NEW_ declaration; the objects' dot-method calls
    # are member access and invisible to the function scan by design.
    component_objects = sorted(
        {m.group(1).lower() for m in _SAS_COMPONENT_OBJECT_RE.finditer(mt)}
    )

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
        component_objects=component_objects,
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


# Fields where the parent's whole-region value wins over the child's (a fallback)
# rather than being unioned. All derive from the %MACRO signature header, which
# only the split slice containing it can parse.
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
# The inner token/whitespace quantifiers are *possessive* (``*+``) so a
# ``set``/``merge`` header with no reachable terminator fails in O(n) instead of
# backtracking exponentially (which the parse deadline cannot interrupt). Only
# the outer ``+?`` stays lazy, to stop at the first terminator.
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
# Quoted physical-path dataset references (e.g. ``data 'c:/tmp/perm';``). String
# contents are blanked in ``mt``, so these MUST be scanned on the
# quotes-preserved form (``cf``). The header form uses the same ``(?<![\w=])``
# guard as _BODY_DATA_HDR_RE so ``data=`` options don't match as DATA statements.
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
# A hash object constructor's DATASET: argument — the dataset loaded into the
# hash table at instantiation (Programmer's Guide Ch. 21, "Hash Table
# Merging") — is a data input of the step, like SET/MERGE. The name lives
# inside a quoted string literal, so it MUST be scanned on the
# quotes-preserved form (``cf``), never on ``mt``.
_HASH_DATASET_ARG_RE = re.compile(
    r"\bdataset\s*:\s*(['\"])\s*([^'\"]+?)\s*\1",
    re.IGNORECASE,
)


def _hash_dataset_refs(cf: str) -> list[str]:
    """Raw dataset references from hash constructors' ``dataset:`` arguments
    in *cf*, with any parenthesised dataset options stripped. References may
    still hold macro variables (``dataset: "&ds"``) — callers classify or
    skip those; an unquoted argument (a character variable or expression) is
    never matched, per "flag as unresolved, do not guess"."""
    refs: list[str] = []
    for m in _HASH_DATASET_ARG_RE.finditer(cf):
        name = m.group(2).split("(", 1)[0].strip()
        if name:
            refs.append(name)
    return refs


# Macro body dataset classification. A %MACRO body references datasets either
# literally (``data work.base;``, resolvable from source) or parameterised
# (``data &ds.;``, resolvable only at the call site). The functions below
# extract both kinds for the batcher.

# DATA statement header inside a macro body (may contain &refs). Inner stars are
# possessive (``*+``) for the same reason as _SET_RE above; the outer ``+?``
# stays lazy.
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
    cf: str | None = None,
) -> tuple[list[str], list[str], list[dict], list[dict], list[str]]:
    """
    Analyse a %MACRO block's body and classify every dataset reference
    as literal (fixed value) or parameterised (depends on a call argument).

    ``mt`` is the sanitised (comments/strings blanked) form of ``macro_text``
    and ``cf`` the comments-only-blanked form (string literals intact —
    needed for hash constructors' quoted ``dataset:`` arguments); callers
    that already have them — e.g. :func:`_metadata_for` — pass them in to
    avoid re-running the sanitiser over the same body.  When omitted they are
    computed here, so direct callers can still pass just the raw text.

    Returns
    -------
    literal_inputs, literal_outputs : list[str]
    param_inputs, param_outputs     : list[dict]   {"param": name, "pos": idx}
    param_names                     : list[str]    ordered signature names
    """
    if mt is None:
        mt = _sanitise(macro_text)
    if cf is None:
        cf = _sanitise(macro_text, blank_strings=False)

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
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"_macro_body_io: params={param_names}  param_pos={param_pos}")

    raw_outputs: list[str] = []
    raw_inputs: list[str] = []

    # Keyword gates on lowercased copies, as in _metadata_for: each gated
    # pattern contains its keyword as a contiguous case-insensitive literal,
    # so a failed substring test proves the scan would find nothing.
    low = mt.lower()

    if "data" in low:
        for m in _BODY_DATA_HDR_RE.finditer(mt):
            cleaned = _PAREN_RE.sub(" ", m.group(1))
            for tok in _AMP_TOKEN_RE.findall(cleaned):
                if tok.lower() not in _SAS_RESERVED:
                    raw_outputs.append(tok)

    if "output" in low:
        for m in _BODY_OUTPUT_RE.finditer(mt):
            raw_outputs.append(m.group(1))

    if "out" in low:
        for m in _BODY_PROC_OUT_RE.finditer(mt):
            raw = m.group(1) or m.group(2) or ""
            if raw:
                raw_outputs.append(raw)

    if "create" in low:
        for m in _BODY_SQL_CREATE_RE.finditer(mt):
            raw_outputs.append(m.group(1))
    if "insert" in low:
        for m in _BODY_SQL_INTO_RE.finditer(mt):
            raw_outputs.append(m.group(1))

    if "set" in low:
        for m in _BODY_SET_RE.finditer(mt):
            cleaned = _PAREN_RE.sub(" ", m.group(1))
            for tok in _AMP_TOKEN_RE.findall(cleaned):
                raw_inputs.append(tok)

    if "merge" in low:
        for m in _BODY_MERGE_RE.finditer(mt):
            cleaned = _PAREN_RE.sub(" ", m.group(1))
            for tok in _AMP_TOKEN_RE.findall(cleaned):
                raw_inputs.append(tok)

    if "update" in low:
        for m in _BODY_UPDATE_RE.finditer(mt):
            raw_inputs.append(m.group(1))
            raw_outputs.append(m.group(1))
    if "modify" in low:
        for m in _BODY_MODIFY_RE.finditer(mt):
            raw_inputs.append(m.group(1))
            raw_outputs.append(m.group(1))

    if "data" in low:
        for m in _BODY_PROC_IN_RE.finditer(mt):
            raw_inputs.append(m.group(1))

    if "from" in low:
        for m in _BODY_SQL_FROM_RE.finditer(mt):
            raw_inputs.append(m.group(1))
    if "join" in low:
        for m in _BODY_SQL_JOIN_RE.finditer(mt):
            raw_inputs.append(m.group(1))

    # Hash constructors' dataset: arguments — scanned on cf because the name
    # sits inside a quoted literal. May hold &refs (``dataset:"&ds"``), which
    # _classify_ref resolves against the signature like any other reference.
    if "dataset" in cf.lower():
        for raw in _hash_dataset_refs(cf):
            if _AMP_TOKEN_RE.fullmatch(raw):
                raw_inputs.append(raw)

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
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"_macro_body_io: {role} PARAM  raw={raw!r}  param={pname}  pos={param_pos[pname]}"
                        )
                elif pname not in param_pos:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"_macro_body_io: {role} UNRESOLVABLE param ref {raw!r}"
                        )
            else:
                key = _canon_ds(key)
                if key not in seen_lit:
                    seen_lit.add(key)
                    literals.append(key)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"_macro_body_io: {role} LITERAL  raw={raw!r}  name={key}"
                        )

        return literals, params_out

    lit_out, par_out = _classify_list(raw_outputs, "output")
    lit_in, par_in = _classify_list(raw_inputs, "input")

    if logger.isEnabledFor(logging.DEBUG):
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
    # Every chunk kind may invoke a macro inline, so this scan is unconditional.
    invokes: list[str] = [m.group(1).lower() for m in _MACRO_INVOKE_RE.finditer(mt)]

    if kind == SasChunkKind.MACRO_DEFINITION:
        for m in _MACRO_DEF_RE.finditer(mt):
            defines.append(m.group(1).lower())

    elif kind == SasChunkKind.MACRO_CALL:
        pass

    elif kind == SasChunkKind.DATA_STEP:
        # Keyword gates on lowercased copies, as in _metadata_for: each gated
        # pattern contains its keyword as a contiguous case-insensitive
        # literal, so a failed substring test proves the scan would find
        # nothing. Quoted-path patterns additionally require a quote char.
        low = mt.lower()
        lowcf = cf.lower()
        has_quote = "'" in cf or '"' in cf
        first_semi = mt.find(";")
        data_header = mt[:first_semi] if first_semi != -1 else mt
        header_body = _DATA_HDR_STRIP_RE.sub("", data_header)
        outputs.extend(_multi_ds(header_body))
        if "output" in low:
            for m in _OUTPUT_DS_RE.finditer(mt):
                if n := _ds_name(m.group(1)):
                    outputs.append(_canon_ds(n))
        if "set" in low:
            for m in _SET_RE.finditer(mt):
                inputs.extend(_multi_ds(m.group(1)))
        if "merge" in low:
            for m in _MERGE_RE.finditer(mt):
                inputs.extend(_multi_ds(m.group(1)))
        if "update" in low:
            for m in _UPDATE_RE.finditer(mt):
                if n := _ds_name(m.group(1)):
                    inputs.append(_canon_ds(n))
                    outputs.append(_canon_ds(n))
        if "modify" in low:
            for m in _MODIFY_RE.finditer(mt):
                if n := _ds_name(m.group(1)):
                    inputs.append(_canon_ds(n))
                    outputs.append(_canon_ds(n))
        # Quoted physical-path forms: ``data '<path>';`` header (output) and
        # ``set|merge '<path>'`` (input) — scanned on cf, not mt.
        if has_quote and "data" in lowcf:
            for m in _QUOTED_DATA_HDR_RE.finditer(cf):
                outputs.append(_quoted_path(m.group(1)))
        if has_quote and ("set" in lowcf or "merge" in lowcf):
            for m in _QUOTED_SET_MERGE_RE.finditer(cf):
                inputs.append(_quoted_path(m.group(1)))
        # Hash object constructors load their DATASET: argument at
        # instantiation — an input like SET/MERGE. A value holding a macro
        # reference is left unresolved rather than guessed.
        if "dataset" in lowcf:
            for raw in _hash_dataset_refs(cf):
                if "&" in raw:
                    continue
                if (n := _ds_name(raw)) and _DS_TOKEN_RE.fullmatch(n):
                    inputs.append(_canon_ds(n))

    elif kind == SasChunkKind.PROC_STEP:
        low = mt.lower()
        proc_m = _PROC_RE.search(mt)
        proc_name = proc_m.group(1).lower() if proc_m else ""

        if proc_name == "sql":
            if "create" in low:
                for m in _SQL_CREATE_RE.finditer(mt):
                    if n := _ds_name(m.group(1)):
                        outputs.append(_canon_ds(n))
            if "insert" in low:
                for m in _SQL_INTO_RE.finditer(mt):
                    if n := _ds_name(m.group(1)):
                        outputs.append(_canon_ds(n))
            if "from" in low:
                for m in _SQL_FROM_RE.finditer(mt):
                    if n := _ds_name(m.group(1)):
                        inputs.append(_canon_ds(n))
            if "join" in low:
                for m in _SQL_JOIN_RE.finditer(mt):
                    if n := _ds_name(m.group(1)):
                        inputs.append(_canon_ds(n))
        else:
            lowcf = cf.lower()
            has_quote = "'" in cf or '"' in cf
            for m in _DATA_OPT_RE.finditer(mt):
                if n := _ds_name(m.group(1)):
                    inputs.append(_canon_ds(n))
            has_proc_out = "out" in low and _PROC_OUT_RE.search(mt)
            if has_proc_out:
                for m in _PROC_OUT_RE.finditer(mt):
                    raw = m.group(1) or m.group(2) or ""
                    if n := _ds_name(raw):
                        outputs.append(_canon_ds(n))
            # Quoted physical-path options: DATA='<path>' (input) and
            # OUT='<path>' (output) — scanned on cf, not mt.
            if has_quote and "data" in lowcf:
                for m in _QUOTED_DATA_OPT_RE.finditer(cf):
                    inputs.append(_quoted_path(m.group(1)))
            if has_quote and "out" in lowcf:
                for m in _QUOTED_OUT_OPT_RE.finditer(cf):
                    outputs.append(_quoted_path(m.group(1)))
            if proc_name == "sort" and not has_proc_out:
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
