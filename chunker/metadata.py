"""Per-chunk semantic metadata extraction for the SAS chunker. See chunker/README.md.

All scans run on sanitised text from scanner._sanitise; keyword-derived patterns
come from keywords.py.

Logger name: ``chunker.metadata``.
"""

from __future__ import annotations

import logging
import re
from collections import ChainMap
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, TypeVar

from .keywords import (
    _MACRO_CALL_RE,
    _MACRO_INVOKE_RE,
    _NUMBERED_GLOBAL_STATEMENTS,
    _SAS_CALL_ROUTINE_RE,
    _SAS_CALL_ROUTINES,
    _SAS_COMPONENT_OBJECT_RE,
    _SAS_DATA_STEP_STATEMENT_RE,
    _SAS_DATASET_OPTION_RE,
    _SAS_FUNCTION_CALL_RE,
    _SAS_FUNCTIONS,
    _SAS_SET_MULTI_RE,
    _SAS_SUBSETTING_IF_RE,
    _SAS_SUM_STATEMENT_RE,
    SAS_GLOBAL_STATEMENT_TOKENS,
)
from .macro_vars import (
    DS_REF_TOKEN,
    _parse_call_args,
    call_spans,
    has_macro_ref,
    is_dataset_shaped,
    let_assignments,
    let_values,
    macro_signature,
    name_value,
    resolve_refs,
)
from .models import (
    DatasetRole,
    DbTableAccess,
    DbTableVia,
    PathLocation,
    SasChunk,
    SasChunkKind,
    SasChunkMetadata,
    SasCorpus,
    SasDatasetRef,
    SasDbTableRef,
    SasEngineRef,
    SasPathRef,
    _db_table_sort_key,
    _engine_ref_sort_key,
    _libref_of,
    _path_ref_sort_key,
)
from .passthrough import db_table_ref, mask, scan_pass_through
from .paths import ENGINE_LIBNAMES, extract_engine_refs, extract_paths, normalise_path
from .scanner import _blank_span, _Region, _sanitise
from .statements import _canon_ds
from .statements import dataset_refs as statement_dataset_refs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex catalogue (mirrors the Reference Sheet grammar)
# ---------------------------------------------------------------------------

#: Every libref position below is scanned with
#: :data:`~chunker.macro_vars.DS_REF_TOKEN` rather than a bare identifier, so a
#: name spelled through a macro variable (``libname &lib ...``, ``data &table1;``)
#: is seen instead of being read as the identifier after the ``&``.
#: :func:`resolve_macro_var_refs` gives the names their values once the whole
#: file (or corpus) has been walked. Datasets are read statement by statement,
#: in :mod:`chunker.statements`.
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

# Leading statement keyword of a GLOBAL_STATEMENT chunk. A numbered statement
# captures without its occurrence number (title2 -> title, group 1); the rest
# whole (group 2). Built from the published vocabulary so the tokens an
# instruction may scope on and the tokens this can emit cannot drift apart.
# Longest-first so no token masks another it prefixes.
_GLOBAL_STMT_KW_RE = re.compile(
    r"%?\s*(?:("
    + "|".join(sorted(_NUMBERED_GLOBAL_STATEMENTS, key=len, reverse=True))
    + r")\d*|("
    + "|".join(
        sorted(
            SAS_GLOBAL_STATEMENT_TOKENS - _NUMBERED_GLOBAL_STATEMENTS,
            key=len,
            reverse=True,
        )
    )
    + r"))\b",
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

# Shared precompiled token helpers reused across the extractors below.
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")  # bare SAS identifier
_SPLIT_WS_COMMA_RE = re.compile(r"[,\s]+")  # %global/%local list separator
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


_DATASET_KINDS = frozenset(
    {SasChunkKind.DATA_STEP, SasChunkKind.PROC_STEP, SasChunkKind.MACRO_DEFINITION}
)
# An ODS OUTPUT statement, and the ones that end its requests.
_ODS_OUTPUT_RE = re.compile(r"ods\s+output\b", re.IGNORECASE)
_ODS_OUTPUT_END_RE = re.compile(
    r"ods\s+(?:output\s+(?:close|clear)|_all_\s+close)\b", re.IGNORECASE
)


def _metadata_for(region: _Region) -> SasChunkMetadata:
    """The metadata of *region*'s code: its comments, in-stream data and
    SUBMIT code are blanked first (:attr:`~chunker.scanner._Region.code_text`),
    and its datasets are read statement by statement (:mod:`chunker.statements`).
    """
    text, kind = region.code_text, region.kind
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
    # ── a %MACRO's parameters: position, or -1 for a keyword parameter ──────
    params = macro_signature(text) if kind == SasChunkKind.MACRO_DEFINITION else []
    param_names = [name for name, _ in params]
    param_pos: dict[str, int] = {}
    positional = 0
    for name, default in params:
        if default is None:
            param_pos[name] = positional
            positional += 1
        else:
            param_pos[name] = -1

    # ── datasets, statement by statement ────────────────────────────────────
    # Only a step or a %MACRO holds a statement that reads or writes one —
    # and an ODS OUTPUT statement, whose datasets resolve_ods_outputs hands to
    # the PROC that writes them. Every statement of a %MACRO region stands in
    # its body, a split slice without the %MACRO statement included.
    refs = (
        statement_dataset_refs(
            region.units,
            mt_ds,
            cf_ds,
            macro_body=kind == SasChunkKind.MACRO_DEFINITION,
            param_pos=param_pos,
        )
        if kind in _DATASET_KINDS
        or (kind == SasChunkKind.GLOBAL_STATEMENT and _ODS_OUTPUT_RE.match(mt.lstrip()))
        else []
    )

    # ── macros: defined here, and invoked ───────────────────────────────────
    # Invocations are read on the unmasked text: a %macro call inside native
    # SQL still runs, SAS resolving it before the text is sent.
    invk = [m.group(1).lower() for m in _MACRO_INVOKE_RE.finditer(mt)]
    defs = (
        [m.group(1).lower() for m in _MACRO_DEF_RE.finditer(mt)]
        if kind == SasChunkKind.MACRO_DEFINITION
        else []
    )
    # Librefs this chunk assigns; ``_all_`` targets every assigned libref.
    defines_librefs = sorted(
        {_nid(m.group(1)) for m in _LIBNAME_REF_RE.finditer(mt)} - {"_all_"}
        if "libname" in low
        else set()
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
            global_stmt_kw = (kw_m.group(1) or kw_m.group(2)).lower()

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

    # A pass-through table named by this macro's own parameters is a template:
    # each call's reading is recorded at the call (resolve_macro_var_refs).
    if param_names and db_tables:
        params = set(param_names)
        db_tables = [
            t.model_copy(update={"parameterised": True})
            if params & {r.lower() for r in _VAR_REF_RE.findall(t.raw)}
            else t
            for t in db_tables
        ]

    return SasChunkMetadata(
        step_name=_nid(dm.group(1)) if dm else None,
        proc_name=_nid(pm.group(1)) if pm else None,
        macro_name=_nid(mm.group(1)) if mm else None,
        labels=labels,
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
        dataset_refs=tuple(dict.fromkeys(refs)),
        defines_macros=sorted(set(defs)),
        invokes_macros=sorted(set(invk)),
        macro_param_names=param_names,
        produces_macrovars=sorted(set(produces_macrovars)),
        symput_scope_hazard=hazard,
        symput_hazard_vars=sorted(set(hazard_vars)),
        external_refs=external_refs,
        engine_refs=engine_refs,
        db_tables=db_tables,
    )


# Fields where the parent's whole-region value wins over the child's (a fallback)
# rather than being unioned. Each needs context only the whole region holds:
# ``macro_param_names`` derives from the %MACRO signature header, which only the
# split slice containing it can parse; ``db_tables`` from a CONNECT statement
# that may sit in a different slice from the CONNECTION TO using its alias.
# (Parameter dataset references need no entry: a slice without the header has
# no parameters, so it adds none to the parent's.)
_MERGE_PARENT_WINS = frozenset(
    {
        "macro_param_names",
        "db_tables",
    }
)


def _merge_meta(parent: SasChunkMetadata, child: SasChunkMetadata) -> SasChunkMetadata:
    """Merge a split child's metadata with its parent region's metadata.

    Driven by ``SasChunkMetadata.model_fields`` so a newly added field is
    merged by its type automatically instead of being silently dropped:

    - ``list[str]``   → sorted union of both sides;
    - ``tuple[SasDatasetRef, ...]`` → union in source order, the parent's
      first: the child is a slice of the parent's region, so this keeps the
      region's own order, which ``output_datasets`` relies on (``_LAST_``);
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
        elif field.annotation == tuple[SasDatasetRef, ...]:
            merged[name] = tuple(dict.fromkeys([*p, *c]))
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

# The two record types that carry a ``binds`` libref/fileref, which _resolve_binds
# rewrites in place of a macro reference without otherwise touching the record.
_BindsRefT = TypeVar("_BindsRefT", SasPathRef, SasEngineRef)


def _resolve_names(
    names: list[str], table: Mapping[str, str], *, canonical: bool
) -> list[str]:
    """*names* with their ``&`` references expanded, order-preserving.

    Resolution can collapse two spellings onto one name, so the result is
    deduplicated — by insertion order, never sorted, so a list kept in source
    order (a pass-through read's ``sas_targets``) stays in it.
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


def _resolve_binds(refs: list[_BindsRefT], table: Mapping[str, str]) -> list[_BindsRefT]:
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


def _resolve_name(name: str, table: Mapping[str, str]) -> str:
    """A connection name or engine with its macro references expanded."""
    return resolve_refs(name, table).lower() if has_macro_ref(name) else name


def _resolve_db_table(ref: SasDbTableRef, table: Mapping[str, str]) -> SasDbTableRef:
    """*ref* re-derived with every macro reference it holds expanded against *table*.

    The name is re-parsed from ``raw`` — the text as written, which never
    changes — so resolving again with a larger table (the corpus-level run, or
    a macro call's arguments) completes what an earlier run could not, and
    resolving twice changes nothing. A connection name resolves too, and when
    the resolved alias is itself an engine (``connection to &db`` with
    ``%let db = oracle;``) the engine is taken from it, as an unaliased
    ``CONNECT TO`` would have said.
    """
    connection = _resolve_name(ref.connection, table)
    engine = _resolve_name(ref.engine, table) if ref.engine else None
    if (engine is None or has_macro_ref(engine)) and connection in ENGINE_LIBNAMES:
        engine = connection
    return db_table_ref(
        ref.raw,
        name=resolve_refs(ref.raw, table),
        access=ref.access,
        via=ref.via,
        connection=connection,
        engine=engine,
        sas_targets=tuple(_resolve_names(list(ref.sas_targets), table, canonical=True)),
        options=ref.options,
        macro=ref.macro,
        parameterised=ref.parameterised,
    )


def _resolve_db_tables(
    refs: list[SasDbTableRef], table: Mapping[str, str]
) -> list[SasDbTableRef]:
    """*refs* with the macro references in their names resolved — see
    :func:`_resolve_db_table`.

    Two kinds are left alone. LIBNAME records: their ``raw`` is the SAS spelling
    (``edw.accounts``), whose first part is a libref, not a schema, and
    :func:`resolve_db_librefs` rebuilds them from already-resolved names. And
    records attributed to a macro call: those were resolved with the call's
    arguments, which this chunk's own table does not hold.
    """
    out: list[SasDbTableRef] = []
    for ref in refs:
        held = (ref.raw, ref.connection, ref.engine or "", *ref.sas_targets)
        if (
            ref.via is DbTableVia.LIBNAME
            or ref.macro is not None
            or not any(map(has_macro_ref, held))
        ):
            out.append(ref)
        else:
            out.append(_resolve_db_table(ref, table))
    return sorted(dict.fromkeys(out), key=_db_table_sort_key)


def _resolved_meta(
    meta: SasChunkMetadata, table: Mapping[str, str], own_values: dict[str, str]
) -> SasChunkMetadata | None:
    """*meta* with every dataset/libref name resolved against *table*, or
    ``None`` when nothing in it changed.

    *own_values* are this chunk's own ``%LET`` assignments, already expanded by
    the caller against the table in force where each one stands.

    A dataset reference's resolved name is canonicalised again (``batch_med``
    → ``work.batch_med``) so it lands where the batcher's producer/consumer
    matching looks for it; ``referenced_datasets`` and ``referenced_librefs``
    follow, being views of the references.

    A %LET whose value is written like a dataset reference names a library on
    sight — ``%let table_demogr = datacia.member_demographic;`` is how a great
    deal of production SAS names its tables — so the value becomes a MENTION:
    provenance only, never I/O, since a %LET reads and writes nothing (the step
    that uses &table_demogr does). Those are re-derived from *own_values* on
    every run rather than resolved again, so a later run's values replace an
    earlier run's.
    """

    def resolved_name(ref: SasDatasetRef) -> str:
        name = resolve_refs(ref.name, table) if has_macro_ref(ref.name) else ref.name
        return _canon_ds(name)

    refs = [
        ref
        for ref in meta.map_dataset_names(resolved_name).dataset_refs
        if ref.via != "%let"
    ]
    refs += (
        SasDatasetRef(value, DatasetRole.MENTION, raw=value, via="%let")
        for value in own_values.values()
        if is_dataset_shaped(value)
    )
    updates: dict[str, Any] = {"dataset_refs": tuple(dict.fromkeys(refs))}
    updates["defines_librefs"] = sorted(
        set(_resolve_names(meta.defines_librefs, table, canonical=False))
    )
    updates["external_refs"] = _resolve_binds(meta.external_refs, table)
    updates["engine_refs"] = _resolve_binds(meta.engine_refs, table)
    updates["db_tables"] = _resolve_db_tables(meta.db_tables, table)
    if meta.step_name and has_macro_ref(meta.step_name):
        updates["step_name"] = resolve_refs(meta.step_name, table)

    changed = {k: v for k, v in updates.items() if v != getattr(meta, k)}
    if not changed:
        return None
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"resolve_macro_var_refs: {changed}")
    return meta.model_copy(update=changed)


# Macro control flow: an assignment under one of these may or may not run.
_MACRO_CONTROL_RE = re.compile(r"%\s*(?:if|do|goto)\b", re.IGNORECASE)
# %GLOBAL / %LOCAL declaration lists, told apart (unlike _GLOBAL_LOCAL_DECL_RE).
_SCOPE_DECL_RE = re.compile(r"%\s*(global|local)\s+([^;]+?)\s*;", re.IGNORECASE)


def _assign(table: dict[str, str], name: str, value: str) -> None:
    """Give *name* its *value* in *table* — or forget it when the value is
    ``""``, which is how every source here says "unknown from now on"."""
    if value:
        table[name] = value
    else:
        table.pop(name, None)


def _symput_values(cf: str) -> dict[str, str]:
    """The static values a step's ``CALL SYMPUT``/``SYMPUTX`` calls assign.

    ``call symputx('sch', 'EDW_EXPORT');`` — a literal name and a literal,
    name-shaped value — is knowable without running SAS. A value from a data
    column, or two calls giving one name different values (an ``IF``/``ELSE``
    choosing between them), is not: those map to ``""``. A single call is taken
    as executed, which is the residual approximation.
    """
    seen: dict[str, set[str]] = {}
    for m in _CALL_SYMPUT_RE.finditer(cf):
        args = _split_top_level(m.group(2))
        name = _clean_literal(args[0]) if args else None
        if name is None or len(args) < 2:
            continue
        literal = _clean_literal(args[1])
        value = name_value(literal) if literal is not None else None
        seen.setdefault(name.strip().lower(), set()).add(value or "")
    return {name: values.pop() if len(values) == 1 else "" for name, values in seen.items()}


def _run_time_values(chunk: SasChunk) -> dict[str, str]:
    """What the macro variables *chunk* creates at run time hold once it has run.

    Every variable it produces — ``CALL SYMPUT``/``SYMPUTX``, ``PROC SQL INTO`` —
    gets its static value when :func:`_symput_values` knows one, and is
    otherwise *unknown*, which matters as much: an earlier ``%let sch = old;``
    must stop answering for ``&sch`` once ``select s into :sch`` has replaced it.
    Applied after the chunk, never to it — SAS resolves a step's own ``&refs``
    when the step is compiled, before any of its CALLs run.
    """
    names = chunk.metadata.produces_macrovars
    if not names:
        return {}
    static = (
        _symput_values(_sanitise(chunk.text, blank_strings=False))
        if "symput" in chunk.text.lower()
        else {}
    )
    return {name: static.get(name, "") for name in names}


@dataclass(frozen=True)
class _MacroDef:
    """What a call needs from a ``%MACRO`` defined earlier in the walk."""

    name: str
    positional: tuple[str, ...]
    defaults: tuple[tuple[str, str], ...]
    params: frozenset[str]
    global_names: frozenset[str]
    local_names: frozenset[str]
    assignments: tuple[tuple[str, str], ...]
    produces: frozenset[str]
    conditional: bool
    db_tables: tuple[SasDbTableRef, ...]

    @classmethod
    def of(cls, chunk: SasChunk) -> "_MacroDef":
        meta = chunk.metadata
        cf = _sanitise(chunk.text, blank_strings=False)
        mt = _sanitise(chunk.text)
        params = macro_signature(cf)
        declared: dict[str, set[str]] = {"global": set(), "local": set()}
        for m in _SCOPE_DECL_RE.finditer(mt):
            for name in _SPLIT_WS_COMMA_RE.split(m.group(2).strip()):
                name = name.lstrip("&").rstrip(".")
                if _IDENT_RE.fullmatch(name):
                    declared[m.group(1).lower()].add(name.lower())
        return cls(
            name=meta.macro_name or "",
            positional=tuple(n for n, d in params if d is None),
            defaults=tuple((n, d) for n, d in params if d is not None),
            params=frozenset(n for n, _ in params),
            global_names=frozenset(declared["global"]),
            local_names=frozenset(declared["local"]),
            assignments=tuple(let_assignments(cf)),
            produces=frozenset(meta.produces_macrovars),
            conditional=bool(_MACRO_CONTROL_RE.search(mt)),
            db_tables=tuple(meta.db_tables),
        )

    def global_effects(
        self, globals_now: dict[str, str], binding: ChainMap[str, str]
    ) -> dict[str, str]:
        """What one call leaves in the *global* table, by SAS's scoping rule.

        A ``%LET`` in the body updates the global variable when the name is
        declared ``%GLOBAL`` there, or already exists globally and is neither a
        parameter nor ``%LOCAL``; anything else lands in the macro's own local
        table and vanishes with the call. Values resolve in body order against
        the call's *binding*. A body with ``%IF``/``%DO``/``%GOTO`` may or may
        not run an assignment, so every global it could touch becomes unknown
        rather than guessed — including one holding a value before the call.
        So does a global the body may overwrite with ``CALL SYMPUT``/``INTO``.
        """
        assigned: dict[str, str] = {}
        local = ChainMap(assigned, *binding.maps)
        effects: dict[str, str] = {}
        for name, raw in self.assignments:
            resolved = resolve_refs(raw, local) if has_macro_ref(raw) else raw
            value = name_value(resolved) or ""
            assigned[name] = value  # "" hides an outer value: unknown
            if name in self.params or name in self.local_names:
                continue
            if name in self.global_names or name in globals_now or name in effects:
                unknown = self.conditional or has_macro_ref(value)
                effects[name] = "" if unknown else value
        for name in self.produces:
            if name in globals_now and name not in self.params | self.local_names:
                effects[name] = ""
        return effects


class _MacroScope:
    """The macro-variable table a source-order walk carries from chunk to chunk.

    The one definition of the scoping rules every resolution pass applies, so
    :func:`resolve_macro_var_refs` and :func:`resolve_db_librefs` cannot
    disagree about what ``&name`` holds at a given chunk. Its sources: ``%LET``
    in open code; ``CALL SYMPUT``/``SYMPUTX`` with static values (and the
    run-time producers that make a variable unknown); and the globals a called
    ``%MACRO`` sets. The macros defined so far are kept too, for
    :meth:`_calls` to bind a call's arguments.
    """

    def __init__(self) -> None:
        self._table: dict[str, str] = {}
        self._macros: dict[str, _MacroDef] = {}

    def enter(self, chunk: SasChunk) -> tuple[Mapping[str, str], dict[str, str]]:
        """``(scope, own)`` for *chunk*: the table its names resolve against,
        and its own ``%LET`` values resolved where they stand.

        A macro's own parameters are dropped from the scope — the call site
        supplies them. ``%LET`` resolves its value where it stands, so each
        assignment is expanded against the table as it was *before* it, then
        stored resolved: after ``%let x = work.a; %let x = &x.b;`` the variable
        holds work.ab, and a later ``data &x;`` writes work.ab — not a
        reference to itself. Source order inside the chunk is dict insertion
        order, so ``%let a = prod; %let b = &a..orders;`` resolves in one walk.
        A value that is not a name (``""``) makes the variable unknown.
        """
        meta = chunk.metadata
        # Layered, never copied: copying the table for every chunk made a file
        # of thousands of %LETs quadratic. Parameters are hidden by an empty
        # value, which resolve_refs reads as unknown.
        hidden = dict.fromkeys(meta.macro_param_names, "")
        overlay: dict[str, str] = {}
        scope: Mapping[str, str] = ChainMap(overlay, hidden, self._table)
        own: dict[str, str] = {}
        for name, value in meta.macro_var_values.items():
            own[name] = resolve_refs(value, scope) if has_macro_ref(value) else value
            if name not in hidden:
                overlay[name] = own[name]
        return scope, own

    def _binding(
        self, macro: _MacroDef, positional: list[str], keyword: dict[str, str]
    ) -> ChainMap[str, str]:
        """The table *macro*'s body resolves against for one call.

        Defaults, then positional arguments by position, then keyword arguments
        by name; each value resolved against the caller's globals. Every
        parameter shadows a global of the same name, bound or not: an
        unbound ``&tbl`` stays ``&tbl`` rather than borrowing a stranger's value.
        """
        supplied = dict(macro.defaults)
        supplied.update(zip(macro.positional, positional))
        supplied.update((k, v) for k, v in keyword.items() if k in macro.params)
        bound: dict[str, str] = {}
        for name, raw in supplied.items():
            resolved = resolve_refs(raw, self._table) if has_macro_ref(raw) else raw
            if value := name_value(resolved):
                bound[name] = value
        return ChainMap(bound, dict.fromkeys(macro.params, ""), self._table)

    def _calls(
        self, chunk: SasChunk
    ) -> list[tuple[_MacroDef, list[str], dict[str, str]]]:
        """The calls a MACRO_CALL chunk makes of macros defined earlier, in
        source order, each with its positional and keyword arguments.

        One, as a rule: the scanner ends a statement after a call that needs
        no semicolon (see :func:`~chunker.scanner._split_after_calls`). Every
        call the text opens with is still read, so binding never depends on
        where a chunk happens to end.
        """
        if chunk.kind is not SasChunkKind.MACRO_CALL:
            return []
        return [
            (macro, *_parse_call_args(chunk.text[span.start : span.end]))
            for span in call_spans(_sanitise(chunk.text))
            if (macro := self._macros.get(span.name)) is not None
        ]

    def leave(self, chunk: SasChunk, own: dict[str, str]) -> list[SasDbTableRef]:
        """Carry what *chunk* leaves behind to the chunks after it, and return
        the database tables its macro calls read.

        Defining a ``%MACRO`` runs nothing: its assignments wait for a call, and
        the definition is kept for :meth:`_calls`. A MACRO_CALL runs its calls
        first, in order: each binds its own arguments against the globals the
        calls before it left, reads its body's tables (see
        :func:`_call_site_tables`) and leaves its global effects. Then, as for
        any other chunk: its own ``%LET`` values, what its run-time producers
        leave (see :func:`_run_time_values`), and — for a chunk that called
        nothing — the global effects of the macros it invokes inline, with
        their defaults alone.
        """
        meta = chunk.metadata
        if chunk.kind is SasChunkKind.MACRO_DEFINITION:
            # Only the region's whole text holds the signature: a split child
            # would register a body cut in half under the same name.
            if chunk.parent_id is None and meta.macro_name:
                self._macros[meta.macro_name] = _MacroDef.of(chunk)
            return []
        tables: list[SasDbTableRef] = []
        calls = self._calls(chunk)
        for macro, positional, keyword in calls:
            binding = self._binding(macro, positional, keyword)
            tables += _call_site_tables(macro, binding)
            for name, value in macro.global_effects(self._table, binding).items():
                _assign(self._table, name, value)
        for name, value in own.items():
            _assign(self._table, name, value)
        for name, value in _run_time_values(chunk).items():
            _assign(self._table, name, value)
        if not calls:
            for name in meta.invokes_macros:
                if (macro := self._macros.get(name)) is not None:
                    binding = self._binding(macro, [], {})
                    for var, value in macro.global_effects(self._table, binding).items():
                        _assign(self._table, var, value)
        return tables


def _call_site_tables(
    macro: _MacroDef, binding: Mapping[str, str]
) -> list[SasDbTableRef]:
    """*macro*'s pass-through tables as one call of it reads them.

    ``%pull(tbl=current_nonip, out=nonip)`` runs a body that says
    ``create table &out as select * from connection to oracle (select * from
    &schema..&tbl)``; resolved with the call's binding that is
    ``edw_export.current_nonip → work.nonip``, recorded on the call's chunk and
    marked with the macro's name. LIBNAME records are SAS names, which the
    batcher already resolves per call for datasets.
    """
    return [
        _resolve_db_table(ref, binding).model_copy(
            update={"macro": macro.name, "parameterised": False}
        )
        for ref in macro.db_tables
        if ref.via is not DbTableVia.LIBNAME and ref.macro is None
    ]


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
        meta = chunk.metadata
        updates: dict[str, Any] = {}
        resolved = _resolved_meta(meta, scope, own) if scope or own else None
        if resolved is not None:
            updates["metadata"] = resolved
            if resolved.step_name != meta.step_name:
                updates["title"] = _title(chunk.kind, resolved)
        if updates:
            chunks[idx] = chunk = chunk.model_copy(update=updates)
        # Each call of a macro defined earlier reads the tables its body names,
        # resolved with that call's arguments. Recomputed on every run (not
        # added to), so the corpus-level run replaces the file-level answer.
        called = macros.leave(chunk, own)
        current = chunk.metadata
        tables = sorted(
            {*(t for t in current.db_tables if t.macro is None), *called},
            key=_db_table_sort_key,
        )
        if tables != current.db_tables:
            chunks[idx] = chunk.model_copy(
                update={"metadata": current.model_copy(update={"db_tables": tables})}
            )


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
    if not engines:
        return []  # no database LIBNAME in force: no name can reach a table
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


def _with_dataset_refs(
    chunk: SasChunk, refs: tuple[SasDatasetRef, ...]
) -> SasChunk:
    meta = chunk.metadata.add_dataset_refs(refs)
    return chunk if meta is chunk.metadata else chunk.model_copy(update={"metadata": meta})


def resolve_ods_outputs(chunks: list[SasChunk]) -> None:
    """Give each ODS OUTPUT request in open code to the PROC that fulfils it, in place.

    ``ods output Summary=sumstats;`` asks the next procedure for its Summary
    table: the PROC MEANS after it writes work.sumstats, not the ODS statement.
    The statement's requests — WRITE references with ``via="ods_output"`` —
    move to the next PROC_STEP of the same file: its chunk and the chunks it
    was split into. ``ods output close|clear`` or ``ods _all_ close`` before
    any PROC cancels them; each stays on its statement as a MENTION, named but
    never written. A request nothing takes or cancels stays as it is.

    Moving is idempotent: a request once moved or cancelled is no longer a
    WRITE on its statement, so the corpus-level run finds none to move. An ODS
    OUTPUT inside a PROC, or in a ``%MACRO`` body, is that PROC's or that
    body's from the start.
    """

    def requested(ref: SasDatasetRef) -> bool:
        return ref.via == "ods_output" and ref.role is DatasetRole.WRITE

    pending: list[int] = []  # ODS statements whose requests wait for a PROC
    source: str | None = None
    claimed_by: str | None = None
    taken: tuple[SasDatasetRef, ...] = ()
    for idx, chunk in enumerate(chunks):
        if idx == 0 or chunk.source_id != source:
            source, pending, claimed_by = chunk.source_id, [], None
        if claimed_by is not None and chunk.parent_id == claimed_by:
            chunks[idx] = _with_dataset_refs(chunk, taken)  # a split slice
            continue
        claimed_by = None
        if (
            chunk.kind is SasChunkKind.GLOBAL_STATEMENT
            and chunk.metadata.global_statement_keyword == "ods"
        ):
            if _ODS_OUTPUT_END_RE.match(_sanitise(chunk.text).lstrip()):
                for i in pending:
                    meta = chunks[i].metadata
                    refs = tuple(
                        replace(ref, role=DatasetRole.MENTION) if requested(ref) else ref
                        for ref in meta.dataset_refs
                    )
                    chunks[i] = chunks[i].model_copy(
                        update={"metadata": meta.model_copy(update={"dataset_refs": refs})}
                    )
                pending = []
            elif any(map(requested, chunk.metadata.dataset_refs)):
                pending.append(idx)
        elif (
            chunk.kind is SasChunkKind.PROC_STEP
            and chunk.parent_id is None
            and pending
        ):
            taken = tuple(
                ref
                for i in pending
                for ref in chunks[i].metadata.dataset_refs
                if requested(ref)
            )
            for i in pending:
                ods = chunks[i]
                kept = ods.metadata.map_dataset_names(
                    lambda ref: None if requested(ref) else ref.name
                )
                chunks[i] = ods.model_copy(update={"metadata": kept})
            chunks[idx] = _with_dataset_refs(chunk, taken)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"resolve_ods_outputs: {chunk.chunk_id} writes "
                    f"{[ref.name for ref in taken]}"
                )
            claimed_by, pending = chunk.chunk_id, []


# A FILENAME statement's fileref: `filename in '/x';` binds it, and one with no
# quoted place (`filename in clear;`, `filename in temp;`) ends what it held.
_FILENAME_STMT_RE = re.compile(rf"\bfilename\s+({DS_REF_TOKEN})", re.IGNORECASE)
# `src(one)`: a member of the directory a fileref names.
_FILEREF_MEMBER_RE = re.compile(r"(?P<fref>[^(]+)\((?P<member>.*)\)\Z", re.DOTALL)


def _through_fileref(ref: SasPathRef, filename: SasPathRef) -> SasPathRef:
    """*ref*, made through a fileref, at the place the FILENAME *filename*
    gave it: the file itself, or a member of the directory — ``%include
    src(one)`` reads ``<dir>/one.sas``, SAS adding the extension."""
    path = filename.path
    if (m := _FILEREF_MEMBER_RE.match(ref.raw)) is not None:
        member = m.group("member").strip().strip("'\"")
        if ref.statement == "include" and "." not in member:
            member += ".sas"
        path = f"{path.rstrip('/')}/{member}"
    path = normalise_path(path)
    return ref.model_copy(
        update={
            "location": filename.location,
            "path": path,
            "device": filename.device,
            "has_macro_ref": "&" in path,
        }
    )


def resolve_filerefs(chunks: list[SasChunk]) -> None:
    """Give each reference made through a fileref the place its FILENAME names, in place.

    ``filename src '/code';`` then ``%include src(setup);`` includes
    ``/code/setup.sas``; ``filename in '/data/x.csv';`` then ``infile in;``
    reads ``/data/x.csv``. Walks *chunks* in source order keeping the filerefs
    in force — the last FILENAME of each wins, and one with no quoted place
    (``clear``, ``temp``) ends it — and gives each later FILEREF reference
    (:attr:`~chunker.models.PathLocation.FILEREF`) the FILENAME's location
    and path. One no FILENAME before it binds stays FILEREF. ``includes`` is
    derived again from the result.

    A FILENAME inside a ``%MACRO`` body binds too, as a LIBNAME does there. A
    resolved reference is FILEREF no longer, so running the pass again — the
    corpus-level run after the per-file one — changes nothing it resolved.
    """
    bound: dict[str, SasPathRef] = {}
    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata
        refs = meta.external_refs
        named = {r.binds: r for r in refs if r.statement == "filename" and r.binds}
        if named or meta.global_statement_keyword == "filename":
            bound.update(named)
            for m in _FILENAME_STMT_RE.finditer(_sanitise(chunk.text)):
                fref = m.group(1).lower()
                if fref == "_all_":
                    bound.clear()
                elif fref not in named:
                    bound.pop(fref, None)
        if not bound or all(r.location is not PathLocation.FILEREF for r in refs):
            continue
        resolved = [
            _through_fileref(r, bound[r.binds])
            if r.location is PathLocation.FILEREF and r.binds in bound
            else r
            for r in refs
        ]
        if resolved == refs:
            continue
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"resolve_filerefs: {chunk.chunk_id} "
                f"{[(r.raw, r.path) for r in resolved if r not in refs]}"
            )
        chunks[idx] = chunk.model_copy(
            update={
                "metadata": meta.model_copy(
                    update={
                        "external_refs": resolved,
                        "includes": [r.path for r in resolved if r.statement == "include"],
                    }
                )
            }
        )


def resolve_references(chunks: list[SasChunk]) -> None:
    """Every cross-chunk name resolution, in the one order that works, in place.

    Macro variables first — a libref, a dataset, a fileref and a pass-through
    table can all be spelled through one — then filerefs, ODS OUTPUT
    requests, moved to their PROCs with their names resolved, and database
    librefs, which read the resolved names where they end up.
    :meth:`~chunker.chunker.SasSemanticChunker.chunk_text`,
    :class:`~chunker.batcher.MultiFileBatcher` and
    :func:`resolve_corpus_references` all call this, so the order lives here.
    """
    resolve_macro_var_refs(chunks)
    resolve_filerefs(chunks)
    resolve_ods_outputs(chunks)
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
