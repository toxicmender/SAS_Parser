"""Per-chunk semantic metadata extraction for the SAS chunker. See chunker/README.md.

All scans run on sanitised text from scanner._sanitise; keyword-derived patterns
come from keywords.py.

Logger name: ``chunker.metadata``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, TypeVar

from .keywords import (
    _MACRO_CALL_RE,
    _MACRO_INVOKE_RE,
    _SAS_CALL_ROUTINE_RE,
    _SAS_CALL_ROUTINES,
    _SAS_COMPONENT_OBJECT_RE,
    _SAS_DATA_STEP_STATEMENT_RE,
    _SAS_DATASET_OPTION_RE,
    _SAS_FUNCTION_CALL_RE,
    _SAS_FUNCTIONS,
    _SAS_RESERVED,
    _SAS_SET_MULTI_RE,
    _SAS_SUBSETTING_IF_RE,
    _SAS_SUM_STATEMENT_RE,
    SAS_GLOBAL_STATEMENT_TOKENS,
)
from .macro_vars import (
    DS_REF_TOKEN,
    DS_REF_TOKEN_POSSESSIVE,
    has_macro_ref,
    is_dataset_shaped,
    let_values,
    resolve_refs,
)
from .models import (
    DbTableAccess,
    DbTableVia,
    SasChunk,
    SasChunkKind,
    SasChunkMetadata,
    SasCorpus,
    SasDbTableRef,
    SasEngineRef,
    SasPathRef,
    _db_table_sort_key,
    _engine_ref_sort_key,
    _path_ref_sort_key,
)
from .passthrough import db_table_ref, mask, scan_pass_through
from .paths import extract_engine_refs, extract_paths
from .scanner import _blank_span, _sanitise

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex catalogue (mirrors the Reference Sheet grammar)
# ---------------------------------------------------------------------------

#: Every dataset and libref position below is scanned with
#: :data:`~chunker.macro_vars.DS_REF_TOKEN` rather than a bare identifier, so a
#: name spelled through a macro variable (``data &table1;``,
#: ``set &lname..&table1;``) is seen instead of being read as the identifier
#: after the ``&`` — which is how ``&table1`` used to be reported as the
#: dataset ``work.table1``. :func:`resolve_macro_var_refs` gives the names their
#: values once the whole file (or corpus) has been walked.
_DATASET_RE = re.compile(
    rf"\b(?:data|set|merge|update|modify)\s+({DS_REF_TOKEN})",
    re.IGNORECASE,
)
_DATA_OPT_RE = re.compile(
    rf"\bdata\s*=\s*({DS_REF_TOKEN})",
    re.IGNORECASE,
)
# The libref a LIBNAME statement assigns (``libname <ref> ...``). Extraction is
# positional, not temporal, so ``libname x clear;`` still reports x. ``_all_``
# targets every assigned libref rather than naming one; the caller drops it.
_LIBNAME_REF_RE = re.compile(rf"\blibname\s+({DS_REF_TOKEN})", re.IGNORECASE)
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


# %INCLUDE paths are not scanned here: chunker.paths owns that grammar and
# ``includes`` is derived from its output, so there is one definition of where
# an include path lives rather than two that can disagree.
_OPTIONS_RE = re.compile(r"\boptions\s+([^;]+)", re.IGNORECASE)
_LABEL_RE = re.compile(r"\blabel\s+([A-Za-z_]\w*)\s*=", re.IGNORECASE)
_PROC_RE = re.compile(r"\bproc\s+([A-Za-z_]\w*)", re.IGNORECASE)
_DATA_RE = re.compile(
    rf"\bdata\s+({DS_REF_TOKEN})",
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
# capture without their optional occurrence digit (title2 -> title). Built from
# the published vocabulary so the tokens an instruction may scope on and the
# tokens this can emit cannot drift apart. Longest-first so no token masks
# another it prefixes.
_GLOBAL_STMT_KW_RE = re.compile(
    r"%?\s*("
    + "|".join(sorted(SAS_GLOBAL_STATEMENT_TOKENS, key=len, reverse=True))
    + r")\b",
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
_AMP_TOKEN_RE = re.compile(DS_REF_TOKEN)  # dataset token that may hold &refs
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
    # ── SQL pass-through: database tables, and the native SQL to mask ───────
    # Native SQL is the database's, not SAS's: every *dataset* scan below runs
    # on mt_ds/cf_ds, where it is blanked, so `from connection to oracle` and
    # `disconnect from oracle` stop reading as datasets and an Oracle owner
    # stops reading as a SAS libref. Everything else keeps the full text —
    # `&ora_user` in the CONNECT options is still a referenced macro variable.
    # "connect" also gates CONNECTION TO; "execute" catches EXECUTE ... BY.
    db_tables: list[SasDbTableRef] = []
    mt_ds, cf_ds = mt, cf
    if "connect" in lowcf or "execute" in lowcf:
        scan = scan_pass_through(cf, mt)
        if scan.spans:
            mt_ds, cf_ds = mask(mt, scan.spans), mask(cf, scan.spans)
        db_tables = [
            t.model_copy(
                update={"sas_targets": tuple(_canon_ds(s) for s in t.sas_targets)}
            )
            if t.sas_targets
            else t
            for t in scan.tables
        ]
    # Dataset names collected from dataset positions only (DATA/SET/MERGE/UPDATE/
    # MODIFY keywords, DATA=/OUT=/OUTDATA= options, PROC SQL clauses) so table
    # aliases and BY-group temporaries aren't mistaken for datasets/librefs.
    datasets = [_nid(m.group(1)) for m in _DATASET_RE.finditer(mt_ds)]
    datasets += [_nid(m.group(1)) for m in _DATA_OPT_RE.finditer(mt_ds)]
    if "create" in low:
        datasets += [_nid(m.group(1)) for m in _SQL_CREATE_RE.finditer(mt_ds)]
    if "from" in low:
        datasets += [_nid(m.group(1)) for m in _SQL_FROM_RE.finditer(mt_ds)]
    if "join" in low:
        datasets += [_nid(m.group(1)) for m in _SQL_JOIN_RE.finditer(mt_ds)]
    if "insert" in low:
        datasets += [_nid(m.group(1)) for m in _SQL_INTO_RE.finditer(mt_ds)]
    if "out" in low:
        datasets += [
            _nid(m.group(1) or m.group(2)) for m in _PROC_OUT_RE.finditer(mt_ds)
        ]
    # Directed I/O parses the full dataset lists (_DATASET_RE above captures only
    # the first of a multi-dataset statement), so its canonical names complete
    # referenced_datasets in their work.-qualified spelling.
    inp, out, defs, invk = _io_for(text, kind, mt_ds, cf_ds, macro_mt=mt)
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
    # Every external reference the chunk names — see chunker/paths.py, which
    # owns the grammar this and xref.pre both read.
    external_refs = extract_paths(cf)
    # The other half of that grammar: LIBNAMEs bound to a database engine, which
    # name a connection rather than a place and so match none of PATH_STATEMENTS.
    engine_refs = extract_engine_refs(cf)
    # ``includes`` is the %INCLUDE slice of that same scan rather than a second
    # definition of where an include path lives. It keeps its list[str] shape
    # and its consumers (complexity.crossfile, the [meta: includes] flag).
    includes = [r.path for r in external_refs if r.statement == "include"]
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
            _macro_body_io(text, mt_ds, cf_ds)
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

    # ── macro-variable values, for the %LET-driven name resolution pass ─────
    # Scanned on `cf` so a quoted value survives to be unquoted; the pass that
    # consumes this is resolve_macro_var_refs, below.
    macro_var_values = let_values(cf) if "let" in lowcf else {}

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

    # ── DATA step statements ────────────────────────────────────────────────
    # Scanned on `mt` (string literals blanked) at statement position, so a
    # variable named `output` or the word `set` inside an expression does not
    # register. Only meaningful for DATA steps; a PROC's own statements are
    # already identified by `proc_name`.
    data_step_statements: set[str] = set()
    if kind == SasChunkKind.DATA_STEP:
        data_step_statements = {
            m.group(1).lower() for m in _SAS_DATA_STEP_STATEMENT_RE.finditer(mt)
        }
        # The sum statement is a retained accumulator without the keyword.
        if _SAS_SUM_STATEMENT_RE.search(mt):
            data_step_statements.add("retain")
        # `SET a b;` concatenates; the ubiquitous single-dataset `SET a;` does
        # not, so only the multi-dataset form earns a token of its own.
        if _SAS_SET_MULTI_RE.search(mt):
            data_step_statements.add("set_multi")
        # A subsetting IF filters rows; an IF/THEN assigns. Different targets.
        if _SAS_SUBSETTING_IF_RE.search(mt):
            data_step_statements.add("subsetting_if")
        if _SAS_DATASET_OPTION_RE.search(mt):
            data_step_statements.add("dataset_option")

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
        macro_var_values=macro_var_values,
        recognized_functions=recognized_functions,
        recognized_call_routines=recognized_call_routines,
        component_objects=component_objects,
        data_step_statements=sorted(data_step_statements),
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
        external_refs=external_refs,
        engine_refs=engine_refs,
        db_tables=db_tables,
    )


# Fields where the parent's whole-region value wins over the child's (a fallback)
# rather than being unioned. Each needs context only the whole region holds: the
# first three derive from the %MACRO signature header, which only the split
# slice containing it can parse; ``db_tables`` from a CONNECT statement that may
# sit in a different slice from the CONNECTION TO using its alias.
_MERGE_PARENT_WINS = frozenset(
    {
        "body_param_inputs",
        "body_param_outputs",
        "macro_param_names",
        "db_tables",
    }
)


def _merge_meta(parent: SasChunkMetadata, child: SasChunkMetadata) -> SasChunkMetadata:
    """Merge a split child's metadata with its parent region's metadata.

    Driven by ``SasChunkMetadata.model_fields`` so a newly added field is
    merged by its type automatically instead of being silently dropped:

    - ``list[str]``   → sorted union of both sides;
    - ``list[SasPathRef]`` → union of both sides, ordered by
      :func:`~chunker.models._path_ref_sort_key` (the records are frozen, so a
      set deduplicates them; the sort is what keeps output reproducible);
    - ``list[SasEngineRef]`` → the same rule under
      :func:`~chunker.models._engine_ref_sort_key`;
    - ``dict[str, str]`` → both sides merged, the child's entry winning on a
      shared key (a %LET the child slice can see is the assignment in force
      there; the parent's whole-region view may already hold a later one);
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
    merged: dict[str, Any] = {}
    for name, field in SasChunkMetadata.model_fields.items():
        p = getattr(parent, name)
        c = getattr(child, name)
        if name in _MERGE_PARENT_WINS:
            merged[name] = p or c
        elif field.annotation == list[str]:
            merged[name] = sorted({*p, *c})
        elif field.annotation == list[SasPathRef]:
            merged[name] = sorted({*p, *c}, key=_path_ref_sort_key)
        elif field.annotation == list[SasEngineRef]:
            merged[name] = sorted({*p, *c}, key=_engine_ref_sort_key)
        elif field.annotation == dict[str, str]:
            merged[name] = {**p, **c}
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
# Macro-variable name resolution — a pass over already-built chunks
# ---------------------------------------------------------------------------

# Dataset lists in the *canonical* namespace: a resolved name is canonicalised
# again (``batch_med`` → ``work.batch_med``) so it lands where the batcher's
# producer/consumer matching looks for it. ``referenced_datasets`` is raw-source
# provenance and is deliberately not in this tuple — it keeps the spelling the
# statement used, exactly as it does for a name written without a macro.
_CANONICAL_DS_FIELDS = (
    "input_datasets",
    "output_datasets",
    "body_literal_inputs",
    "body_literal_outputs",
)

# The two record types that carry a ``binds`` libref/fileref, which _resolve_binds
# rewrites in place of a macro reference without otherwise touching the record.
_BindsRefT = TypeVar("_BindsRefT", SasPathRef, SasEngineRef)


def _resolve_names(
    names: list[str], table: dict[str, str], *, canonical: bool
) -> list[str]:
    """*names* with their ``&`` references expanded, order-preserving.

    Resolution can collapse two spellings onto one name, so the result is
    deduplicated — by insertion order, never sorted, because
    ``output_datasets`` order is load-bearing (invariant 3).
    """
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        resolved = resolve_refs(name, table) if has_macro_ref(name) else name
        if canonical:
            resolved = _canon_ds(resolved)
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def _resolve_binds(refs: list[_BindsRefT], table: dict[str, str]) -> list[_BindsRefT]:
    """*refs* with the libref/fileref each one assigns resolved against *table*.

    ``binds`` is a name, already lowercased rather than kept verbatim, so it
    resolves like any other name — and it has to, or ``libname &l '/data/x';``
    would leave ``defines_librefs`` calling the library ``mylib`` while the
    :class:`~chunker.models.SasPathRef` for the same statement calls it ``&l``.
    ``path`` and ``raw`` are deliberately untouched: ``raw`` is documented as
    the value exactly as written, and a path's own macro references are already
    reported by ``has_macro_ref``.
    """
    return [
        r.model_copy(update={"binds": resolve_refs(r.binds, table)})
        if r.binds and has_macro_ref(r.binds)
        else r
        for r in refs
    ]


def _resolve_db_tables(
    refs: list[SasDbTableRef], table: dict[str, str]
) -> list[SasDbTableRef]:
    """*refs* with macro references in their names and SAS copies resolved.

    A pass-through name is re-parsed from ``raw`` — the text as written, which
    never changes — so a corpus-level run with a larger table resolves what a
    file-level run could not, and repeating a run changes nothing. LIBNAME
    records are left alone: their ``raw`` is the SAS spelling (``edw.accounts``),
    whose first part is a libref, not a schema, and :func:`resolve_db_librefs`
    rebuilds them from already-resolved names anyway.
    """
    out: list[SasDbTableRef] = []
    for ref in refs:
        if ref.via is DbTableVia.LIBNAME or not (
            has_macro_ref(ref.raw)
            or has_macro_ref(ref.connection)
            or any(map(has_macro_ref, ref.sas_targets))
        ):
            out.append(ref)
            continue
        out.append(
            db_table_ref(
                ref.raw,
                name=resolve_refs(ref.raw, table),
                access=ref.access,
                via=ref.via,
                connection=resolve_refs(ref.connection, table),
                engine=ref.engine,
                sas_targets=tuple(
                    _resolve_names(list(ref.sas_targets), table, canonical=True)
                ),
                options=ref.options,
            )
        )
    return sorted(dict.fromkeys(out), key=_db_table_sort_key)


def _resolved_meta(
    meta: SasChunkMetadata, table: dict[str, str], own_values: dict[str, str]
) -> SasChunkMetadata | None:
    """*meta* with every dataset/libref name resolved against *table*, or
    ``None`` when nothing in it changed.

    *own_values* are this chunk's own ``%LET`` assignments, already expanded by
    the caller against the table in force where each one stands.
    """
    updates: dict[str, Any] = {
        field: _resolve_names(getattr(meta, field), table, canonical=True)
        for field in _CANONICAL_DS_FIELDS
    }
    updates["defines_librefs"] = sorted(
        set(_resolve_names(meta.defines_librefs, table, canonical=False))
    )
    updates["external_refs"] = _resolve_binds(meta.external_refs, table)
    updates["engine_refs"] = _resolve_binds(meta.engine_refs, table)
    updates["db_tables"] = _resolve_db_tables(meta.db_tables, table)

    # A %LET whose value is written like a dataset reference names a library on
    # sight — ``%let table_demogr = datacia.member_demographic;`` is how a great
    # deal of production SAS names its tables — so the value joins the chunk's
    # referenced datasets. It is provenance only, never I/O: a %LET reads and
    # writes nothing, the step that uses &table_demogr does.
    let_refs = [v for v in own_values.values() if is_dataset_shaped(v)]
    updates["referenced_datasets"] = sorted(
        {
            *_resolve_names(meta.referenced_datasets, table, canonical=False),
            *updates["input_datasets"],
            *updates["output_datasets"],
            *let_refs,
        }
    )
    # Recomputed from the resolved names by the same recipe _metadata_for uses:
    # the libref half of every two-level name, plus the ones assigned here. An
    # unresolved libref (``&lib_out_spd``) is reported as written — the batch
    # does depend on a library, and saying so beats reporting none.
    updates["referenced_librefs"] = sorted(
        {
            d.split(".", 1)[0]
            for d in updates["referenced_datasets"]
            if "." in d and not d.startswith("'")
        }
        | set(updates["defines_librefs"])
    )
    if meta.step_name and has_macro_ref(meta.step_name):
        updates["step_name"] = resolve_refs(meta.step_name, table)

    changed = {k: v for k, v in updates.items() if v != getattr(meta, k)}
    if not changed:
        return None
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"resolve_macro_var_refs: {changed}")
    return meta.model_copy(update=changed)


class _MacroScope:
    """The ``%LET`` table a source-order walk carries from chunk to chunk.

    The one definition of the scoping rules every resolution pass applies, so
    :func:`resolve_macro_var_refs` and :func:`resolve_db_librefs` cannot
    disagree about what ``&name`` holds at a given chunk.
    """

    def __init__(self) -> None:
        self._table: dict[str, str] = {}

    def enter(self, chunk: SasChunk) -> tuple[dict[str, str], dict[str, str]]:
        """``(scope, own)`` for *chunk*: the table its names resolve against,
        and its own ``%LET`` values resolved where they stand.

        A macro's own parameters are dropped from the scope — the call site
        supplies them. ``%LET`` resolves its value where it stands, so each
        assignment is expanded against the table as it was *before* it, then
        stored resolved: after ``%let x = work.a; %let x = &x.b;`` the variable
        holds work.ab, and a later ``data &x;`` writes work.ab — not a
        reference to itself. Source order inside the chunk is dict insertion
        order, so ``%let a = prod; %let b = &a..orders;`` resolves in one walk.
        """
        meta = chunk.metadata
        shadowed = set(meta.macro_param_names)
        scope = (
            {k: v for k, v in self._table.items() if k not in shadowed}
            if shadowed
            else dict(self._table)
        )
        own: dict[str, str] = {}
        for name, value in meta.macro_var_values.items():
            own[name] = resolve_refs(value, scope) if has_macro_ref(value) else value
            if name not in shadowed:
                scope[name] = own[name]
        return scope, own

    def leave(self, chunk: SasChunk, own: dict[str, str]) -> None:
        """Carry *chunk*'s assignments forward to the chunks after it.

        A ``%MACRO`` body's assignments are conditional on the macro being
        called, so they never join the running table.
        """
        if own and chunk.kind is not SasChunkKind.MACRO_DEFINITION:
            shadowed = set(chunk.metadata.macro_param_names)
            self._table.update({k: v for k, v in own.items() if k not in shadowed})


def resolve_macro_var_refs(chunks: list[SasChunk]) -> None:
    """Give ``&name`` dataset and libref references their values, in place.

    Walks *chunks* once in source order, accumulating the ``%LET`` values each
    one assigns and expanding the references of the ones that follow — so
    ``data &table1; set &lname..&table1;`` reports ``work.batch_med`` and
    ``xwrk.batch_med`` instead of the invented ``work.table1`` and no input at
    all. Chunk ``title`` is refreshed when the DATA step's name resolves.

    Run per file by :meth:`~chunker.chunker.SasSemanticChunker.chunk_text` and
    again over the flattened corpus by
    :class:`~chunker.batcher.MultiFileBatcher`, which is what lets a ``%LET`` in
    one file resolve a name used in another. The second run is cheap and
    idempotent: an already-resolved name holds no ``&`` and is skipped.

    Two deliberate limits, both on the side of not guessing:

    - **Source order, not execution order.** A ``%LET`` is in force for the
      chunks after it, the same nearest-preceding-producer approximation
      :mod:`chunker.batcher` already makes for datasets. A value assigned
      inside a ``%MACRO`` body stays local to that chunk, since whether it ever
      executes depends on a call site.
    - **A macro's own parameters shadow the table.** Inside a
      ``%MACRO m(ds);`` body, ``&ds`` is supplied by the call site, so a
      corpus-level ``%let ds = ...;`` must not be read as its value. Those
      references stay in ``body_param_inputs`` / ``body_param_outputs``, where
      the batcher resolves them per call.
    """
    macros = _MacroScope()
    for idx, chunk in enumerate(chunks):
        scope, own = macros.enter(chunk)
        if scope or own:
            resolved = _resolved_meta(chunk.metadata, scope, own)
            if resolved is not None:
                updates: dict[str, Any] = {"metadata": resolved}
                if resolved.step_name != chunk.metadata.step_name:
                    updates["title"] = _title(chunk.kind, resolved)
                chunks[idx] = chunk.model_copy(update=updates)
        macros.leave(chunk, own)


def _libref_of(name: str) -> str | None:
    """The libref of a two-level SAS name, or ``None`` (one-level, quoted path)."""
    return name.split(".", 1)[0] if "." in name and not name.startswith("'") else None


def _libname_tables(
    meta: SasChunkMetadata, engines: dict[str, tuple[SasEngineRef, str | None]]
) -> list[SasDbTableRef]:
    """The database tables *meta*'s SAS names reach through engine librefs.

    ``set edw.accounts;`` after ``libname edw oracle schema=fr_dm`` reads the
    Oracle table ``fr_dm.accounts``; ``data edw.x;`` writes one. A read's SAS
    copies are the chunk's outputs that are not themselves database tables — for
    a DATA step, exactly what it makes from the read; for a multi-statement
    ``PROC SQL``, possibly more than one statement's worth.
    """
    reads = dict.fromkeys([*meta.input_datasets, *meta.body_literal_inputs])
    writes = dict.fromkeys([*meta.output_datasets, *meta.body_literal_outputs])
    copies = tuple(
        w
        for w in writes
        if _libref_of(w) not in engines and not (w.startswith("_") and w.endswith("_"))
    )
    found: list[SasDbTableRef] = []
    for names, access in ((reads, DbTableAccess.READ), (writes, DbTableAccess.WRITE)):
        for name in names:
            libref = _libref_of(name)
            if libref is None or libref not in engines:
                continue
            ref, db_schema = engines[libref]
            found.append(
                db_table_ref(
                    name,
                    name=name.split(".", 1)[1],
                    default_schema=db_schema,
                    access=access,
                    via=DbTableVia.LIBNAME,
                    connection=libref,
                    engine=ref.engine,
                    sas_targets=copies if access is DbTableAccess.READ else (),
                    options=ref.options,
                )
            )
    return found


def _with_libref_engine(
    ref: SasDbTableRef, engines: dict[str, tuple[SasEngineRef, str | None]]
) -> SasDbTableRef:
    """*ref* with the engine and options of the LIBNAME its connection names.

    For ``CONNECT USING edw`` — which borrows the LIBNAME's connection, so the
    pass-through scan could record neither. Only a record whose engine is still
    unknown is filled; one a ``CONNECT TO`` named is never overridden.
    """
    if ref.engine is not None or ref.connection not in engines:
        return ref
    libname, _ = engines[ref.connection]
    return ref.model_copy(update={"engine": libname.engine, "options": libname.options})


def resolve_db_librefs(chunks: list[SasChunk]) -> None:
    """Register the database tables SAS names reach through engine LIBNAMEs, in place.

    Walks *chunks* in source order keeping the engine LIBNAMEs in force —
    ``libname edw oracle path=EDWPRO schema=fr_dm`` binds ``edw`` to Oracle
    schema ``fr_dm`` — and gives every later SAS name under such a libref a
    :class:`~chunker.models.SasDbTableRef` (``via=LIBNAME``). The SAS name
    itself stays where it is: SAS code does name ``edw.accounts``.

    - Any other binding of the libref ends it: ``libname edw clear;``, or a path
      LIBNAME reusing the name.
    - An engine LIBNAME inside a ``%MACRO`` body *does* bind, unlike a ``%LET``:
      connection macros are how production SAS hides its credentials, and a
      libref names a connection, so binding it cannot give an unrelated name a
      wrong value the way a stray ``%LET`` could.
    - ``schema=`` is resolved against the ``%LET`` table where the LIBNAME
      stands; the options themselves stay as written.
    - ``CONNECT USING`` records get their engine and options here.

    Replaces each chunk's ``via=LIBNAME`` records rather than adding to them, so
    the per-file run in ``chunk_text`` and the corpus-level run in the batcher
    compose: the second only adds what a LIBNAME in another file makes visible.
    Runs after :func:`resolve_macro_var_refs` — see :func:`resolve_references`.
    """
    macros = _MacroScope()
    engines: dict[str, tuple[SasEngineRef, str | None]] = {}
    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata
        scope, own = macros.enter(chunk)
        for libref in meta.defines_librefs:
            engines.pop(libref, None)
        for engine_ref in meta.engine_refs:
            db_schema = engine_ref.option_map.get("schema")
            engines[engine_ref.binds] = (
                engine_ref,
                resolve_refs(db_schema, scope).lower() if db_schema else None,
            )
        tables = [
            _with_libref_engine(t, engines)
            for t in meta.db_tables
            if t.via is not DbTableVia.LIBNAME
        ]
        tables = sorted(
            dict.fromkeys([*tables, *_libname_tables(meta, engines)]),
            key=_db_table_sort_key,
        )
        if tables != meta.db_tables:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"resolve_db_librefs: {chunk.chunk_id} db_tables={[str(t) for t in tables]}"
                )
            chunks[idx] = chunk.model_copy(
                update={"metadata": meta.model_copy(update={"db_tables": tables})}
            )
        macros.leave(chunk, own)


def resolve_references(chunks: list[SasChunk]) -> None:
    """Every cross-chunk name resolution, in the one order that works, in place.

    Macro variables first — a libref, a dataset and a pass-through table can all
    be spelled through one — then database librefs, which read the resolved
    names. :meth:`~chunker.chunker.SasSemanticChunker.chunk_text`,
    :class:`~chunker.batcher.MultiFileBatcher` and
    :func:`resolve_corpus_references` all call this, so the order lives here.
    """
    resolve_macro_var_refs(chunks)
    resolve_db_librefs(chunks)


def resolve_corpus_references(corpus: SasCorpus) -> SasCorpus:
    """*corpus* with names resolved across its files, for callers that do not batch.

    ``chunk_file`` resolves each file alone, so a ``%LET`` or a database
    LIBNAME in ``setup.sas`` cannot reach a name in ``job.sas`` until something
    walks the corpus. :class:`~chunker.batcher.MultiFileBatcher` does; a caller
    that only wants metadata — the hydration planner, say — calls this. Chunk
    ids are unchanged, and the input is not mutated.
    """
    flat = [c for r in corpus.file_results for c in r.chunks]
    resolve_references(flat)
    results = []
    start = 0
    for result in corpus.file_results:
        end = start + len(result.chunks)
        results.append(result.model_copy(update={"chunks": flat[start:end]}))
        start = end
    return SasCorpus(file_results=results)


# ---------------------------------------------------------------------------
# Directed I/O extraction  — called from _metadata_for
# ---------------------------------------------------------------------------

_SQL_CREATE_RE = re.compile(
    rf"\bcreate\s+(?:table|view)\s+({DS_REF_TOKEN})",
    re.IGNORECASE,
)
_SQL_FROM_RE = re.compile(
    rf"\bfrom\s+({DS_REF_TOKEN})",
    re.IGNORECASE,
)
_SQL_JOIN_RE = re.compile(
    rf"\bjoin\s+({DS_REF_TOKEN})",
    re.IGNORECASE,
)
_SQL_INTO_RE = re.compile(
    rf"\binsert\s+into\s+({DS_REF_TOKEN})",
    re.IGNORECASE,
)
# The inner token/whitespace quantifiers are *possessive* (``*+``) so a
# ``set``/``merge`` header with no reachable terminator fails in O(n) instead of
# backtracking exponentially (which the parse deadline cannot interrupt). Only
# the outer ``+?`` stays lazy, to stop at the first terminator.
_SET_RE = re.compile(
    rf"\bset\s++((?:{DS_REF_TOKEN_POSSESSIVE}(?:\s*+\([^)]*\))?\s*+)+?)"
    r"(?=;|\bwhere\b|\bby\b|\bobs\b|\bnobs\b)",
    re.IGNORECASE | re.DOTALL,
)
_MERGE_RE = re.compile(
    rf"\bmerge\s++((?:{DS_REF_TOKEN_POSSESSIVE}(?:\s*+\([^)]*\))?\s*+)+?)"
    r"(?=;|\bby\b)",
    re.IGNORECASE | re.DOTALL,
)
_UPDATE_RE = re.compile(
    rf"\bupdate\s+({DS_REF_TOKEN})",
    re.IGNORECASE,
)
_MODIFY_RE = re.compile(
    rf"\bmodify\s+({DS_REF_TOKEN})",
    re.IGNORECASE,
)
_OUTPUT_DS_RE = re.compile(
    rf"\boutput\s+({DS_REF_TOKEN})",
    re.IGNORECASE,
)
_PROC_OUT_RE = re.compile(
    rf"\bout\s*=\s*({DS_REF_TOKEN})"
    rf"|\boutdata\s*=\s*({DS_REF_TOKEN})",
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
    rf"(?<![\w=])data\s++((?:{DS_REF_TOKEN_POSSESSIVE}(?:\s*+\([^)]*+\))?\s*+)+?)(?=;)",
    re.IGNORECASE,
)
_BODY_SET_RE = re.compile(
    rf"\bset\s++((?:{DS_REF_TOKEN_POSSESSIVE}(?:\s*+\([^)]*\))?\s*+)+?)"
    r"(?=;|\bwhere\b|\bby\b|\bobs\b)",
    re.IGNORECASE | re.DOTALL,
)
_BODY_MERGE_RE = re.compile(
    rf"\bmerge\s++((?:{DS_REF_TOKEN_POSSESSIVE}(?:\s*+\([^)]*\))?\s*+)+?)(?=;|\bby\b)",
    re.IGNORECASE | re.DOTALL,
)
_BODY_UPDATE_RE = re.compile(rf"\bupdate\s+({DS_REF_TOKEN})", re.IGNORECASE)
_BODY_MODIFY_RE = re.compile(rf"\bmodify\s+({DS_REF_TOKEN})", re.IGNORECASE)
_BODY_OUTPUT_RE = re.compile(rf"\boutput\s+({DS_REF_TOKEN})", re.IGNORECASE)
_BODY_PROC_IN_RE = re.compile(rf"\bdata\s*=\s*({DS_REF_TOKEN})", re.IGNORECASE)
_BODY_PROC_OUT_RE = re.compile(
    rf"\bout\s*=\s*({DS_REF_TOKEN})"
    rf"|\boutdata\s*=\s*({DS_REF_TOKEN})",
    re.IGNORECASE,
)
_BODY_SQL_CREATE_RE = re.compile(
    rf"\bcreate\s+(?:table|view)\s+({DS_REF_TOKEN})",
    re.IGNORECASE,
)
_BODY_SQL_FROM_RE = re.compile(rf"\bfrom\s+({DS_REF_TOKEN})", re.IGNORECASE)
_BODY_SQL_JOIN_RE = re.compile(rf"\bjoin\s+({DS_REF_TOKEN})", re.IGNORECASE)
_BODY_SQL_INTO_RE = re.compile(
    rf"\binsert\s+into\s+({DS_REF_TOKEN})",
    re.IGNORECASE,
)

# Splits a macro argument list on commas, respecting nested parens
_ARG_SPLIT_RE = re.compile(r",(?![^(]*\))")

# Extracts macro signature: %macro name(params)
_MACRO_SIG_RE = re.compile(r"%\s*macro\s+\w+\s*\(([^)]*)\)", re.IGNORECASE)


# What a macro body's dataset reference turns out to be — the vocabulary
# _classify_ref answers in and _classify_list dispatches on.
_REF_LITERAL = "literal"  # no macro reference; the name is written out
_REF_PARAM = "param"  # exactly this macro's parameter, resolved per call site
_REF_CALL_SITE = "call_site"  # built from parameters, so only a call site knows it
_REF_MACRO_VAR = "macro_var"  # macro variables, none of them this macro's own


def _classify_ref(
    raw: str,
    param_pos: dict[str, int],
) -> tuple[str, str]:
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
    every call, and :func:`resolve_macro_var_refs` can give it a value
    (:data:`_REF_MACRO_VAR`).
    """
    raw = raw.strip()
    if "&" not in raw:
        return raw.lower(), _REF_LITERAL
    refs = [r.lower() for r in _VAR_REF_RE.findall(raw)]
    if len(refs) == 1 and refs[0] in param_pos:
        return refs[0], _REF_PARAM
    if any(r in param_pos for r in refs):
        return raw.lower(), _REF_CALL_SITE
    return raw.lower(), _REF_MACRO_VAR


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
            key, kind = _classify_ref(raw, param_pos)
            if kind == _REF_PARAM:
                if key not in seen_par:
                    seen_par.add(key)
                    params_out.append({"param": key, "pos": param_pos[key]})
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"_macro_body_io: {role} PARAM  raw={raw!r}  param={key}  pos={param_pos[key]}"
                        )
            elif kind == _REF_CALL_SITE:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"_macro_body_io: {role} CALL-SITE ref {raw!r} — no name "
                        f"until the call's arguments are known"
                    )
            else:
                # _REF_LITERAL, and _REF_MACRO_VAR alongside it: a macro
                # variable the call site does not supply has one value for
                # every call, so it belongs with the literals — written as it
                # stands until resolve_macro_var_refs finds the %LET that gives
                # it a value, rather than dropped as it used to be.
                key = _canon_ds(key)
                if key not in seen_lit:
                    seen_lit.add(key)
                    literals.append(key)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"_macro_body_io: {role} {kind.upper()}  raw={raw!r}  name={key}"
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
    - names still holding a macro reference (``&table1``), whose libref is
      not knowable yet: ``&table1`` may well resolve to a two-level name, so
      calling it ``work.&table1`` would assert a library the source never
      named. :func:`resolve_macro_var_refs` canonicalises again once the
      reference has a value, and what never resolves keeps the ``&``;
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
    tokens = _AMP_TOKEN_RE.findall(cleaned)
    return [_canon_ds(n) for t in tokens if (n := _ds_name(t))]


def _io_for(
    text: str,
    kind: SasChunkKind,
    mt: str | None = None,
    cf: str | None = None,
    macro_mt: str | None = None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Extract directed data-flow edges from a single chunk's source text.

    ``mt`` is the sanitised (comments/strings blanked) form of ``text`` and
    ``cf`` the comments-only-blanked form (string literals intact — needed
    for quoted physical-path dataset references, which live *inside* string
    delimiters); callers that already have them pass them in to avoid
    redundant sanitise passes.  When omitted they are computed here.

    ``macro_mt`` is the text the macro-name scans read, when it differs from
    ``mt``: :func:`_metadata_for` passes a *masked* ``mt`` (SQL pass-through
    blanked) for the dataset scans, but a ``%macro`` call inside native SQL
    still runs — SAS resolves it before sending the text — so the invocation
    scan reads the unmasked form.  Defaults to ``mt``.

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
    if macro_mt is None:
        macro_mt = mt

    inputs: list[str] = []
    outputs: list[str] = []
    defines: list[str] = []
    # Every chunk kind may invoke a macro inline, so this scan is unconditional.
    invokes: list[str] = [
        m.group(1).lower() for m in _MACRO_INVOKE_RE.finditer(macro_mt)
    ]

    if kind == SasChunkKind.MACRO_DEFINITION:
        for m in _MACRO_DEF_RE.finditer(macro_mt):
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
        # reference is recorded as written, like every other dataset position:
        # resolve_macro_var_refs gives it a value if the corpus assigns one,
        # and never guesses one if it does not.
        if "dataset" in lowcf:
            for raw in _hash_dataset_refs(cf):
                if (n := _ds_name(raw)) and _AMP_TOKEN_RE.fullmatch(n):
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
