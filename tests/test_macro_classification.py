"""
test_macro_classification.py — consolidated tests for every chunk-kind and
metadata-level *classification* decision the chunker makes about macro
constructs.  Combines MACRO_PARSING_ROADMAP.md Phase 1 and Phase 3, since
both are fundamentally the same kind of work: deciding what a given macro
statement *is*, without attempting to evaluate or interpret it.

Phase 1 — reserved-word exclusion and two new metadata fields:

1. Reserved-word exclusion — _MACRO_CALL_RE and _MACRO_INVOKE_RE must
   exclude the complete ~94-word reserved-word set, not the small
   hand-maintained partial list that predated this phase.  Confirmed bug:
   control-flow keywords (%do, %if, %then, %else, %end, %return, %abort,
   %goto, ...) and macro functions (%scan, %substr, %bquote, ...) were
   leaking into invokes_macros/called_macros as if they were real
   corpus-local macro invocations.

2. macro_var_op metadata field — distinguishes %let/%global/%local/%put
   from each other and from the other statements that share the
   GLOBAL_STATEMENT chunk kind (libname/filename/title/footnote/ods),
   without changing any chunk's kind (several existing tests already pin
   %let to GLOBAL_STATEMENT).

3. referenced_automatic_vars metadata field — every automatic macro
   variable SAS provides begins with the reserved SYS prefix; this field
   is populated by a simple prefix check, no enumerated lookup table.

Phase 3 — a new SasChunkKind for open-code control flow, plus two
high-severity visibility flags:

4. SasChunkKind.MACRO_CONTROL_FLOW — %if/%else/%do/%end/%return/%goto/
   %abort appearing as a standalone statement *outside* any macro
   definition (legal for %if/%then/%else specifically, per Ch. 12; the
   others are macro-definition-only and would represent malformed source
   if seen at this level, but are still recognised defensively).  The
   same constructs appearing *inside* a macro body remain part of that
   single MACRO_DEFINITION chunk, exactly as before — this is pure
   classification ("containment"), not interpretation of what an %if
   branch decides.

5. contains_abort / contains_computed_goto metadata fields (on
   MACRO_DEFINITION chunks) — %ABORT is high-severity enough (it stops
   not just the macro but the current DATA step/session/job) to surface
   regardless of nesting depth; a computed %GOTO is one of three Ch. 5
   conditions that force CALL SYMPUT/SYMPUTX into local scope, completing
   the item explicitly deferred in Phase 2's roadmap entry.

Phase 4 — macro function exclusion completeness ("recognize, don't
evaluate"):

6. _ADDITIONAL_MACRO_FUNCTION_WORDS — Ch. 12 Table 12.3 lists 27 macro
   functions; 22 of them happen to already be covered by Appendix 1's
   94-word reserved-word set (a side effect Phase 1 didn't specifically
   verify against the function table). The remaining 5 (%sysmacexec,
   %sysmacexist, %sysmexecdepth, %sysmexecname, %sysprod) are genuine
   macro functions per Ch. 12 but are absent from Appendix 1 — a real
   discrepancy between two sections of SAS's own manual. Kept as a
   separate constant rather than folded into _RESERVED_WORDS, so that
   constant's identity ("Appendix 1, verbatim — 94 words") stays exact.

Phase 5 — out-of-static-analysis-scope items ("flag, don't attempt"):

7. _STANDARD_AUTOCALL_MACROS — ten well-known SAS-provided autocall
   macros (Ch. 12 Table 12.13) that ship with every SAS install. Unlike
   the reserved-word sets above, these ARE genuine, callable macro names
   — still detected as invocations — but a call to one is never a
   "missing dependency" the way a genuinely-undefined corpus macro is, so
   batcher.py excludes them from SasBatch.required_macros while still
   reporting them via the new SasBatch.standard_autocall_macros field.
   Full SASAUTOS directory scanning (F2) and SASMSTORE compiled-macro
   resolution (F3) remain explicitly deferred per the roadmap's own
   recommended build order; G4-G6's conservative non-fabrication behavior
   (never guessing a %scan/%substr-built or multi-parameter-concatenated
   dataset name) was re-verified, not re-implemented.

Run:  python -m pytest tests/test_macro_classification.py -v
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chunker import SasChunkBatcher, SasChunkKind, SasSemanticChunker
from chunker.chunker import (
    _ADDITIONAL_MACRO_FUNCTION_WORDS,
    _MACRO_CALL_RE,
    _MACRO_INVOKE_RE,
    _RESERVED_WORDS,
    _STANDARD_AUTOCALL_MACROS,
    _is_automatic_macro_var,
    _macro_contains_computed_goto,
    _macro_has_local_scope,
)

_C = SasSemanticChunker(min_words=1, max_words=9_999)


# ---------------------------------------------------------------------------
# 1. Reserved-word exclusion
# ---------------------------------------------------------------------------


class TestReservedWordExclusion(unittest.TestCase):
    def test_reserved_word_set_size(self):
        """
        Pin the exact count extracted from SAS Macro Language: Reference,
        Appendix 1 — guards against silent shrinkage/duplication if the
        set is ever edited.
        """
        self.assertEqual(len(_RESERVED_WORDS), 94)

    def test_known_reserved_words_present(self):
        """Spot-check a representative sample from each reserved-word category."""
        control_flow = {
            "do",
            "if",
            "then",
            "else",
            "end",
            "to",
            "by",
            "while",
            "until",
            "goto",
            "go",
            "return",
        }
        functions = {
            "scan",
            "substr",
            "qscan",
            "qsubstr",
            "length",
            "index",
            "bquote",
            "nrbquote",
            "quote",
            "nrquote",
            "superq",
            "unquote",
            "str",
            "nrstr",
        }
        statements = {
            "macro",
            "mend",
            "let",
            "global",
            "local",
            "put",
            "include",
            "abort",
        }
        introspection = {"symexist", "symglobl", "symlocal", "symdel"}
        system = {
            "syscall",
            "sysexec",
            "sysfunc",
            "sysget",
            "sysrput",
            "sysevalf",
            "eval",
            "resolve",
        }

        for word in control_flow | functions | statements | introspection | system:
            self.assertIn(word, _RESERVED_WORDS, f"{word!r} missing from reserved set")

    def test_upcase_is_reserved_but_not_a_macro_function(self):
        """
        UPCASE is reserved (can't be a macro name) but there is no
        standalone %upcase(...) macro function -- only %sysfunc(upcase(...)).
        The reserved-word set still must exclude %upcase from invocation
        tracking, since *attempting* %upcase(...) is reserved-word misuse,
        not a real macro call.
        """
        self.assertIn("upcase", _RESERVED_WORDS)

    def test_control_flow_not_in_invokes_macros(self):
        """
        %do/%if/%then/%else/%end/%return/%abort/%goto must never appear in
        invokes_macros, even when they appear as bare top-level statements
        (the confirmed bug from the roadmap).
        """
        cases = [
            "%do i=1 %to 5; %end;",
            "%if &x=1 %then %put yes;",
            "%return;",
            "%abort;",
            "%goto skip;",
        ]
        for src in cases:
            cr = _C.chunk_text(src)
            all_invokes = {m for c in cr.chunks for m in c.metadata.invokes_macros}
            all_called = {m for c in cr.chunks for m in c.metadata.called_macros}
            self.assertEqual(all_invokes, set(), f"leak in invokes_macros for {src!r}")
            self.assertEqual(all_called, set(), f"leak in called_macros for {src!r}")

    def test_macro_functions_not_in_invokes_macros(self):
        """%scan/%substr/%bquote/%eval/%sysfunc etc. must never leak either."""
        cases = [
            "%let x = %scan(&list, 1);",
            "%let y = %substr(&str, 1, 3);",
            "%let z = %bquote(&val);",
            "%let w = %eval(1+2);",
            "%let v = %sysfunc(today(), date9.);",
        ]
        for src in cases:
            cr = _C.chunk_text(src)
            all_invokes = {m for c in cr.chunks for m in c.metadata.invokes_macros}
            self.assertEqual(all_invokes, set(), f"leak in invokes_macros for {src!r}")

    def test_real_user_macro_still_detected(self):
        """Widening the exclusion list must not swallow real macro names."""
        cr = _C.chunk_text("%clean(work.orders);")
        self.assertIn("clean", cr.chunks[0].metadata.invokes_macros)
        self.assertIn("clean", cr.chunks[0].metadata.called_macros)

    def test_macro_name_resembling_reserved_substring_still_detected(self):
        """
        A real macro whose name merely *contains* a reserved word as a
        substring (not as the whole word) must still be detected -- the
        exclusion is word-bounded (\\b), not a substring match.
        """
        cr = _C.chunk_text("%doit(work.a);")
        self.assertIn("doit", cr.chunks[0].metadata.invokes_macros)

    def test_call_re_and_invoke_re_consistent(self):
        """
        _MACRO_CALL_RE (legacy called_macros) and _MACRO_INVOKE_RE (modern
        invokes_macros) must now agree on every reserved word, fixing the
        documented inconsistency between the two regexes.
        """
        sample = " ".join(f"%{w}" for w in sorted(_RESERVED_WORDS))
        call_matches = {m.group(1).lower() for m in _MACRO_CALL_RE.finditer(sample)}
        invoke_matches = {m.group(1).lower() for m in _MACRO_INVOKE_RE.finditer(sample)}
        self.assertEqual(call_matches, set())
        self.assertEqual(invoke_matches, set())

    def test_end_to_end_macro_with_control_flow_batches_correctly(self):
        """
        A realistic macro using %if/%do internally must not have those
        keywords pollute its invokes_macros, and the macro definition
        must still correctly batch with its own call site (the
        macro_invocation edge), unaffected by the control-flow noise
        inside the macro body.

        Note: work.enriched's *producer* DATA step is read (not written)
        inside the macro body via a literal PROC data= reference -- linking
        a macro body's literal *inputs* back to their producer is a
        separate capability (the existing Fix A only wires up literal
        *outputs*) and is not part of this phase, so this test does not
        assert on that cross-chunk relationship.
        """
        src = (
            "%macro reportit(request);\n"
            "%if %upcase(&request)=STAT %then %do;\n"
            "  proc means data=work.enriched; run;\n"
            "%end;\n"
            "%else %do;\n"
            "  proc print data=work.enriched; run;\n"
            "%end;\n"
            "%mend reportit;\n"
            "data work.enriched; set mylib.raw; run;\n"
            "%reportit(stat);\n"
        )
        cr = _C.chunk_text(src)
        macro_def = next(
            c for c in cr.chunks if c.kind == SasChunkKind.MACRO_DEFINITION
        )
        self.assertNotIn("if", macro_def.metadata.invokes_macros)
        self.assertNotIn("do", macro_def.metadata.invokes_macros)
        self.assertNotIn("then", macro_def.metadata.invokes_macros)
        self.assertNotIn("else", macro_def.metadata.invokes_macros)
        self.assertNotIn("end", macro_def.metadata.invokes_macros)
        self.assertNotIn("upcase", macro_def.metadata.invokes_macros)
        # The macro's literal body input IS correctly captured by the
        # existing (separate) body-IO mechanism, even though nothing yet
        # links it back to its producer:
        self.assertIn("work.enriched", macro_def.metadata.body_literal_inputs)

        br = SasChunkBatcher().batch(cr)
        # The macro definition and its own call site batch together via
        # the macro_invocation edge, regardless of the control-flow noise.
        def_and_call_batch = next(
            b for b in br.batches if "reportit" in b.defined_macros
        )
        self.assertEqual(len(def_and_call_batch.chunks), 2)


# ---------------------------------------------------------------------------
# 2. macro_var_op classification
# ---------------------------------------------------------------------------


class TestMacroVarOp(unittest.TestCase):
    def test_let_classified(self):
        cr = _C.chunk_text("%let cutoff = 01JAN2020;")
        self.assertEqual(cr.chunks[0].kind, SasChunkKind.GLOBAL_STATEMENT)
        self.assertEqual(cr.chunks[0].metadata.macro_var_op, "let")

    def test_global_classified(self):
        cr = _C.chunk_text("%global region_filter;")
        self.assertEqual(cr.chunks[0].metadata.macro_var_op, "global")

    def test_local_classified(self):
        cr = _C.chunk_text("%local i;")
        self.assertEqual(cr.chunks[0].metadata.macro_var_op, "local")

    def test_put_classified(self):
        cr = _C.chunk_text("%put NOTE: starting;")
        self.assertEqual(cr.chunks[0].metadata.macro_var_op, "put")

    def test_libname_has_no_var_op(self):
        """The other GLOBAL_STATEMENT members must remain None."""
        cr = _C.chunk_text("libname mylib '/data';")
        self.assertEqual(cr.chunks[0].kind, SasChunkKind.GLOBAL_STATEMENT)
        self.assertIsNone(cr.chunks[0].metadata.macro_var_op)

    def test_title_footnote_filename_ods_have_no_var_op(self):
        src = (
            "filename out '/tmp/a';\n"
            "title 'Demo';\n"
            "footnote 'Annual refresh';\n"
            "ods pdf file='/tmp/r.pdf';\n"
        )
        cr = _C.chunk_text(src)
        for c in cr.chunks:
            self.assertIsNone(c.metadata.macro_var_op)

    def test_non_global_statement_kinds_have_no_var_op(self):
        """DATA/PROC/MACRO_DEFINITION/MACRO_CALL chunks are always None."""
        src = "data work.a; run;\nproc print data=work.a; run;\n%macro m; %mend;\n%m;\n"
        cr = _C.chunk_text(src)
        for c in cr.chunks:
            self.assertIsNone(c.metadata.macro_var_op)

    def test_kind_unchanged_for_all_four_statements(self):
        """
        Critical backward-compatibility check: introducing macro_var_op
        must NOT change any chunk's kind away from GLOBAL_STATEMENT.
        """
        src = "%let a=1;\n%global b;\n%local c;\n%put d;\n"
        cr = _C.chunk_text(src)
        for c in cr.chunks:
            self.assertEqual(c.kind, SasChunkKind.GLOBAL_STATEMENT)

    def test_oversized_split_preserves_var_op(self):
        """A %let chunk is single-statement and never splits, but verify
        _merge_meta's macro_var_op precedence rule directly."""
        from chunker.chunker import _merge_meta
        from chunker.models import SasChunkMetadata

        parent = SasChunkMetadata(macro_var_op="let")
        child = SasChunkMetadata(macro_var_op=None)
        merged = _merge_meta(parent, child)
        self.assertEqual(merged.macro_var_op, "let")


# ---------------------------------------------------------------------------
# 3. referenced_automatic_vars (SYS-prefix detection)
# ---------------------------------------------------------------------------


class TestAutomaticMacroVariables(unittest.TestCase):
    def test_is_automatic_macro_var_predicate(self):
        self.assertTrue(_is_automatic_macro_var("sysdate"))
        self.assertTrue(_is_automatic_macro_var("SYSDATE9"))
        self.assertTrue(_is_automatic_macro_var("syslast"))
        self.assertTrue(_is_automatic_macro_var("sysparm"))
        self.assertFalse(_is_automatic_macro_var("cutoff_date"))
        self.assertFalse(_is_automatic_macro_var("region_filter"))
        # The check is a simple prefix match (per the SAS reservation rule
        # "do not prefix macro variable names with SYS"), so any name that
        # happens to start with "sys" -- even one that isn't a real
        # automatic variable -- is documented, expected behaviour, not a
        # false positive to guard against.
        self.assertTrue(_is_automatic_macro_var("system_x"))

    def test_sysdate_detected_in_title_string(self):
        """Automatic vars inside double-quoted strings must be caught."""
        cr = _C.chunk_text('title "Report run &sysday, &sysdate9";')
        self.assertEqual(
            set(cr.chunks[0].metadata.referenced_automatic_vars),
            {"sysday", "sysdate9"},
        )

    def test_sysjobid_detected_inside_data_step(self):
        src = 'data work.out;\n  set work.in;\n  run_id = "&sysjobid";\nrun;\n'
        cr = _C.chunk_text(src)
        self.assertIn("sysjobid", cr.chunks[0].metadata.referenced_automatic_vars)

    def test_no_automatic_vars_when_none_present(self):
        cr = _C.chunk_text("%let normal_var = 5;")
        self.assertEqual(cr.chunks[0].metadata.referenced_automatic_vars, [])

    def test_user_macro_param_not_misdetected_as_automatic(self):
        """A normal &param. reference must never be flagged as automatic."""
        cr = _C.chunk_text("%macro clean(ds);\n  data &ds.; set &ds.; run;\n%mend;\n")
        self.assertEqual(cr.chunks[0].metadata.referenced_automatic_vars, [])

    def test_automatic_var_alongside_macro_param_in_same_chunk(self):
        """Both classifications must coexist correctly in one chunk."""
        src = (
            "%macro tag(ds);\n"
            "  data &ds.;\n"
            "    set &ds.;\n"
            '    load_date = "&sysdate9";\n'
            "  run;\n"
            "%mend;\n"
        )
        cr = _C.chunk_text(src)
        meta = cr.chunks[0].metadata
        self.assertEqual(meta.macro_param_names, ["ds"])
        self.assertEqual(meta.body_param_inputs, [{"param": "ds", "pos": 0}])
        self.assertEqual(meta.referenced_automatic_vars, ["sysdate9"])

    def test_multiple_distinct_automatic_vars_deduplicated_and_sorted(self):
        src = "%put &sysdate &sysday &sysdate;"
        cr = _C.chunk_text(src)
        self.assertEqual(
            cr.chunks[0].metadata.referenced_automatic_vars,
            ["sysdate", "sysday"],
        )

    def test_does_not_affect_batching(self):
        """Automatic-var detection is informational only -- it must not
        create any spurious *dataset_flow*/*macro_invocation* dependency
        edges in the batcher.  Context absorption (a separate, pre-existing
        feature that pulls a leading GLOBAL_STATEMENT into the following
        batch) is disabled here so it doesn't get confused with this."""
        src = (
            'title "Run on &sysdate9";\n'
            "data work.a; set mylib.raw; run;\n"
            "proc print data=work.a; run;\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher(include_options_chunks=False).batch(cr)
        # work.a <-> proc print batch together; the title stays unrelated
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.batches[0].chunks), 2)
        self.assertEqual(len(br.singletons), 1)
        self.assertEqual(br.singletons[0].kind, SasChunkKind.GLOBAL_STATEMENT)


# ---------------------------------------------------------------------------
# 4. MACRO_CONTROL_FLOW chunk kind (Phase 3)
# ---------------------------------------------------------------------------


class TestMacroControlFlowKind(unittest.TestCase):
    """
    %if/%then/%else is legal in open code (outside any macro definition)
    since a relatively recent SAS release (Ch. 12, Tables 12.1 and 12.2
    both list it).  %do/%end/%return/%goto/%abort are macro-definition-only
    per Table 12.2, so seeing them at the top level would represent
    malformed source -- but the classifier still recognises them
    defensively rather than letting them fall through to MACRO_CALL.
    """

    def test_open_code_if_classified(self):
        cr = _C.chunk_text("%if &sysscp=WIN %then %put windows;\n")
        self.assertEqual(cr.chunks[0].kind, SasChunkKind.MACRO_CONTROL_FLOW)
        self.assertEqual(cr.chunks[0].metadata.control_flow_op, "if")

    def test_open_code_else_classified(self):
        """The block form's %else %do;...%end; is its own standalone
        statement/unit when %if appeared in open code (each %; boundary
        starts a fresh unit -- %else is not a continuation of %if)."""
        src = "%if &sysscp=WIN %then %put windows;\n%else %put not windows;\n"
        cr = _C.chunk_text(src)
        kinds = [c.kind for c in cr.chunks]
        ops = [c.metadata.control_flow_op for c in cr.chunks]
        self.assertEqual(kinds, [SasChunkKind.MACRO_CONTROL_FLOW] * 2)
        self.assertEqual(ops, ["if", "else"])

    def test_open_code_return_classified(self):
        cr = _C.chunk_text("%return;\n")
        self.assertEqual(cr.chunks[0].kind, SasChunkKind.MACRO_CONTROL_FLOW)
        self.assertEqual(cr.chunks[0].metadata.control_flow_op, "return")

    def test_open_code_abort_classified(self):
        cr = _C.chunk_text("%abort;\n")
        self.assertEqual(cr.chunks[0].kind, SasChunkKind.MACRO_CONTROL_FLOW)
        self.assertEqual(cr.chunks[0].metadata.control_flow_op, "abort")

    def test_open_code_goto_classified(self):
        cr = _C.chunk_text("%goto skip;\n")
        self.assertEqual(cr.chunks[0].kind, SasChunkKind.MACRO_CONTROL_FLOW)
        self.assertEqual(cr.chunks[0].metadata.control_flow_op, "goto")

    def test_open_code_do_classified(self):
        cr = _C.chunk_text("%do i=1 %to 5;\n")
        self.assertEqual(cr.chunks[0].kind, SasChunkKind.MACRO_CONTROL_FLOW)
        self.assertEqual(cr.chunks[0].metadata.control_flow_op, "do")

    def test_open_code_end_classified(self):
        cr = _C.chunk_text("%end;\n")
        self.assertEqual(cr.chunks[0].kind, SasChunkKind.MACRO_CONTROL_FLOW)
        self.assertEqual(cr.chunks[0].metadata.control_flow_op, "end")

    def test_other_kinds_have_no_control_flow_op(self):
        src = (
            "data work.a; run;\n"
            "proc print data=work.a; run;\n"
            "%macro m; %mend;\n"
            "%m;\n"
            "%let x=1;\n"
        )
        cr = _C.chunk_text(src)
        for c in cr.chunks:
            self.assertIsNone(c.metadata.control_flow_op)

    def test_title_shows_control_flow_op(self):
        cr = _C.chunk_text("%abort;\n")
        self.assertEqual(cr.chunks[0].title, "%ABORT")

    def test_invokes_macros_clean_for_open_code_control_flow(self):
        """The Phase 1 reserved-word fix and the Phase 3 kind are
        independent layers -- confirm both hold simultaneously."""
        cr = _C.chunk_text("%if &sysscp=WIN %then %put windows;\n")
        self.assertEqual(cr.chunks[0].metadata.invokes_macros, [])

    def test_containment_inside_data_step_unaffected(self):
        """%if appearing inside an already-open DATA step body must not
        fragment the block or change its kind -- containment, not a
        separate top-level chunk."""
        src = (
            "data work.out;\n"
            "  set work.in;\n"
            "  %if &debug=1 %then %put DEBUG mode;\n"
            "  x = 1;\n"
            "run;\n"
        )
        cr = _C.chunk_text(src)
        self.assertEqual(len(cr.chunks), 1)
        self.assertEqual(cr.chunks[0].kind, SasChunkKind.DATA_STEP)

    def test_containment_inside_macro_definition_unaffected(self):
        """The full %if/%then/%do/%end/%else/%do/%end block inside a macro
        body stays part of that single MACRO_DEFINITION chunk."""
        src = (
            "%macro reportit(request);\n"
            "%if %upcase(&request)=STAT %then %do;\n"
            "  proc means data=work.enriched; run;\n"
            "%end;\n"
            "%else %do;\n"
            "  proc print data=work.enriched; run;\n"
            "%end;\n"
            "%mend reportit;\n"
        )
        cr = _C.chunk_text(src)
        self.assertEqual(len(cr.chunks), 1)
        self.assertEqual(cr.chunks[0].kind, SasChunkKind.MACRO_DEFINITION)


# ---------------------------------------------------------------------------
# 5. contains_abort / contains_computed_goto (Phase 3)
# ---------------------------------------------------------------------------


class TestAbortAndComputedGotoVisibility(unittest.TestCase):
    def test_abort_detected_inside_macro_body(self):
        src = (
            "%macro guard(x);\n"
            "  %if &x= %then %do;\n"
            "    %put ERROR: x is required;\n"
            "    %abort;\n"
            "  %end;\n"
            "%mend;\n"
        )
        cr = _C.chunk_text(src)
        self.assertTrue(cr.chunks[0].metadata.contains_abort)

    def test_no_abort_when_absent(self):
        cr = _C.chunk_text("%macro clean(ds);\n  data &ds.; run;\n%mend;\n")
        self.assertFalse(cr.chunks[0].metadata.contains_abort)

    def test_abort_does_not_leak_into_invokes_macros(self):
        src = "%macro guard;\n  %abort;\n%mend;\n"
        cr = _C.chunk_text(src)
        self.assertNotIn("abort", cr.chunks[0].metadata.invokes_macros)

    def test_computed_goto_detected(self):
        src = "%macro env7(param1);\n  %goto &param1;\n  data _null_; run;\n%mend;\n"
        cr = _C.chunk_text(src)
        self.assertTrue(cr.chunks[0].metadata.contains_computed_goto)

    def test_non_computed_goto_not_flagged(self):
        """A %goto to a plain literal label is NOT computed -- per Ch. 5,
        only a label containing & or % counts."""
        src = "%macro env8;\n  %goto skip;\n  %put unreachable;\n%mend;\n"
        cr = _C.chunk_text(src)
        self.assertFalse(cr.chunks[0].metadata.contains_computed_goto)

    def test_macro_contains_computed_goto_helper_direct(self):
        self.assertTrue(_macro_contains_computed_goto("%goto &home;"))
        self.assertTrue(_macro_contains_computed_goto("%goto %scan(&list,1);"))
        self.assertFalse(_macro_contains_computed_goto("%goto skip;"))
        self.assertFalse(_macro_contains_computed_goto("data _null_; run;"))


# ---------------------------------------------------------------------------
# 6. Computed %GOTO completes the Phase 2 deferred scope-hazard condition
# ---------------------------------------------------------------------------


class TestComputedGotoExtendsScopeHazard(unittest.TestCase):
    """
    Per Ch. 5, a computed %GOTO is one of three conditions that force
    CALL SYMPUT/SYMPUTX into local scope *even when the symbol table would
    otherwise be empty* -- the other two (parameters, %local) were
    implemented in Phase 2; this phase closes the %goto gap that Phase 2's
    roadmap entry explicitly deferred.
    """

    def test_macro_has_local_scope_true_for_computed_goto_alone(self):
        """No parameters, no %local -- only a computed %goto -- must still
        report a non-empty effective scope."""
        text = "%goto &home; data _null_; run;"
        self.assertTrue(_macro_has_local_scope(text, []))

    def test_macro_has_local_scope_false_without_any_trigger(self):
        """Sanity check: the three existing Phase 2 cases are unaffected
        by this extension."""
        self.assertFalse(_macro_has_local_scope("data _null_; run;", []))

    def test_hazard_triggered_by_computed_goto_alongside_a_parameter(self):
        """env7 has both a parameter AND a computed %goto -- either alone
        would trigger the hazard; this just confirms they coexist without
        conflict, and that contains_computed_goto is reported alongside
        the hazard fields."""
        src = (
            "%macro env7(param1);\n"
            "  %goto &param1;\n"
            "  %skip: %put past skip;\n"
            "  data _null_;\n"
            "    call symput('myvar7', 1);\n"
            "  run;\n"
            "%mend;\n"
        )
        cr = _C.chunk_text(src)
        m = cr.chunks[0].metadata
        self.assertTrue(m.symput_scope_hazard)
        self.assertTrue(m.contains_computed_goto)

    def test_hazard_triggered_by_computed_goto_with_truly_no_params(self):
        """The precise claim: zero parameters, no %local, only a computed
        %goto -- and the hazard still fires."""
        src = (
            "%macro env9;\n"
            "  %goto &target;\n"
            "  data _null_;\n"
            "    call symput('myvar9', 1);\n"
            "  run;\n"
            "%mend;\n"
        )
        cr = _C.chunk_text(src)
        m = cr.chunks[0].metadata
        self.assertEqual(m.macro_param_names, [])
        self.assertTrue(m.contains_computed_goto)
        self.assertTrue(m.symput_scope_hazard)
        self.assertEqual(m.symput_hazard_vars, ["myvar9"])


# ---------------------------------------------------------------------------
# 7. Macro function exclusion completeness (Phase 4)
# ---------------------------------------------------------------------------


class TestMacroFunctionExclusionCompleteness(unittest.TestCase):
    """
    Per Ch. 12 Table 12.3, there are exactly 27 macro functions. 22 of them
    were already excluded as a side effect of Phase 1's complete
    Appendix-1 reserved-word set; this phase closes the remaining 5 that
    Appendix 1 doesn't list (a genuine discrepancy between two sections of
    the same manual, not a documentation error on this project's part).
    """

    ALL_27_MACRO_FUNCTIONS = [
        "bquote",
        "nrbquote",
        "eval",
        "index",
        "length",
        "quote",
        "nrquote",
        "scan",
        "qscan",
        "str",
        "nrstr",
        "substr",
        "qsubstr",
        "superq",
        "symexist",
        "symglobl",
        "symlocal",
        "sysevalf",
        "sysfunc",
        "qsysfunc",
        "sysget",
        "sysmacexec",
        "sysmacexist",
        "sysmexecdepth",
        "sysmexecname",
        "sysprod",
        "unquote",
    ]

    PREVIOUSLY_MISSING = {
        "sysmacexec",
        "sysmacexist",
        "sysmexecdepth",
        "sysmexecname",
        "sysprod",
    }

    def test_table_12_3_has_exactly_27_functions(self):
        self.assertEqual(len(self.ALL_27_MACRO_FUNCTIONS), 27)

    def test_additional_words_constant_matches_documented_gap(self):
        self.assertEqual(_ADDITIONAL_MACRO_FUNCTION_WORDS, self.PREVIOUSLY_MISSING)

    def test_reserved_words_size_unchanged_by_phase4(self):
        """
        _RESERVED_WORDS must stay exactly the 94-word Appendix-1 set --
        the 5 additional functions are deliberately a separate constant,
        not folded into it, per the docstring's stated rationale.
        """
        self.assertEqual(len(_RESERVED_WORDS), 94)

    def test_22_functions_already_covered_by_appendix_1(self):
        already_covered = set(self.ALL_27_MACRO_FUNCTIONS) - self.PREVIOUSLY_MISSING
        self.assertEqual(len(already_covered), 22)
        for fn in already_covered:
            self.assertIn(
                fn, _RESERVED_WORDS, f"{fn!r} unexpectedly missing from _RESERVED_WORDS"
            )

    def test_all_27_functions_excluded_by_both_regexes(self):
        for fn in self.ALL_27_MACRO_FUNCTIONS:
            text = f"%{fn}(x);"
            call_matches = [m.group(1).lower() for m in _MACRO_CALL_RE.finditer(text)]
            invoke_matches = [
                m.group(1).lower() for m in _MACRO_INVOKE_RE.finditer(text)
            ]
            self.assertEqual(call_matches, [], f"{fn!r} leaked through _MACRO_CALL_RE")
            self.assertEqual(
                invoke_matches, [], f"{fn!r} leaked through _MACRO_INVOKE_RE"
            )

    def test_previously_missing_five_now_excluded_end_to_end(self):
        for fn in self.PREVIOUSLY_MISSING:
            cr = _C.chunk_text(f"%let x = %{fn}(test);")
            self.assertEqual(cr.chunks[0].metadata.invokes_macros, [])

    def test_real_macro_names_sharing_substrings_still_detected(self):
        """Word-boundary precision must hold for the newly-added words too."""
        cases = ["sysprodlist", "sysmacexecutive", "do_sysmexecdepth"]
        for name in cases:
            cr = _C.chunk_text(f"%{name}(work.a);")
            self.assertIn(name, cr.chunks[0].metadata.invokes_macros)

    def test_realistic_macro_using_two_previously_missing_functions(self):
        """End-to-end: a macro using %sysmacexist and %sysprod for
        conditional logic must report only its real dependency
        ('clean') in invokes_macros, with zero leakage."""
        src = (
            "%macro safe_run(ds);\n"
            "  %if %sysmacexist(clean) %then %do;\n"
            "    %clean(&ds.);\n"
            "  %end;\n"
            "  %if %sysprod(ets) %then %do;\n"
            "    proc timeseries data=&ds.; run;\n"
            "  %end;\n"
            "%mend;\n"
        )
        cr = _C.chunk_text(src)
        self.assertEqual(cr.chunks[0].metadata.invokes_macros, ["clean"])

    def test_batching_unaffected_by_previously_missing_functions(self):
        """The exclusion fix must not change batching outcomes for a
        macro that legitimately invokes a real dependency alongside the
        newly-excluded introspection functions."""
        src = (
            "%macro safe_run(ds);\n"
            "  %if %sysmacexist(clean) %then %do;\n"
            "    %clean(&ds.);\n"
            "  %end;\n"
            "%mend;\n"
            "%macro clean(ds); data &ds.; set &ds.; run; %mend;\n"
            "%safe_run(work.orders);\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        self.assertEqual(len(br.batches), 1)
        self.assertIn("clean", br.batches[0].defined_macros)
        self.assertIn("safe_run", br.batches[0].defined_macros)


# ---------------------------------------------------------------------------
# 8. Standard autocall macro allowlist (Phase 5, F2b)
# ---------------------------------------------------------------------------


class TestStandardAutocallMacroAllowlist(unittest.TestCase):
    """
    Ten well-known SAS-provided autocall macros (Ch. 12 Table 12.13) ship
    with every SAS installation. A call to one is never a "missing
    dependency" the way a genuinely-undefined corpus macro is, so these
    are excluded from SasBatch.required_macros while still being reported
    separately via SasBatch.standard_autocall_macros -- mirroring the
    Phase 1 automatic-macro-variable pattern exactly (tracked, never
    silently dropped, never treated as a problem).
    """

    EXPECTED_TEN = {
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

    def test_allowlist_has_exactly_ten_names(self):
        self.assertEqual(_STANDARD_AUTOCALL_MACROS, self.EXPECTED_TEN)

    def test_none_of_the_ten_are_reserved_words(self):
        """
        These are genuine, callable macro names -- unlike Phase 1/4's
        exclusion sets, they must NOT overlap with _RESERVED_WORDS or
        _ADDITIONAL_MACRO_FUNCTION_WORDS, since calls to them must still
        be detected as real invocations, not silently excluded entirely.
        """
        self.assertEqual(_STANDARD_AUTOCALL_MACROS & _RESERVED_WORDS, set())
        self.assertEqual(
            _STANDARD_AUTOCALL_MACROS & _ADDITIONAL_MACRO_FUNCTION_WORDS, set()
        )

    def test_standard_autocall_call_still_detected_as_invocation(self):
        """Unlike a reserved word, %trim(...) IS a real macro call and
        must still appear in invokes_macros at the chunk level."""
        cr = _C.chunk_text("%let region = %trim(&raw_region);")
        self.assertIn("trim", cr.chunks[0].metadata.invokes_macros)

    def test_excluded_from_required_macros_when_batched(self):
        src = (
            "%macro normalize(ds);\n"
            "  data &ds.;\n"
            "    set &ds.;\n"
            "    clean_name = %trim(name);\n"
            "  run;\n"
            "%mend;\n"
            "%normalize(work.orders);\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        batch = br.batches[0]
        self.assertEqual(batch.required_macros, [])
        self.assertIn("trim", batch.standard_autocall_macros)

    def test_genuinely_missing_macro_still_flagged(self):
        """Negative control: required_macros must still work normally for
        a real, undefined corpus macro."""
        src = (
            "data work.report;\n"
            "  set mylib.raw;\n"
            "  %custom_transform(work.report);\n"
            "run;\n"
            "proc print data=work.report; run;\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        batch = br.batches[0]
        self.assertIn("custom_transform", batch.required_macros)
        self.assertEqual(batch.standard_autocall_macros, [])

    def test_user_defined_macro_shadows_standard_name(self):
        """If a corpus defines its OWN macro named 'left', that local
        definition takes precedence -- it's a real, resolved dependency,
        not the standard autocall macro, and must not appear in
        standard_autocall_macros."""
        src = "%macro left(x); data &x.; set &x.; run; %mend;\n%left(work.orders);\n"
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        batch = br.batches[0]
        self.assertIn("left", batch.defined_macros)
        self.assertEqual(batch.required_macros, [])
        self.assertEqual(batch.standard_autocall_macros, [])

    def test_multiple_standard_autocall_macros_in_one_batch(self):
        src = (
            "%macro tidy(ds);\n"
            "  data &ds.;\n"
            "    set &ds.;\n"
            "    a = %left(x);\n"
            "  run;\n"
            "%mend;\n"
            "%macro tidy2(ds);\n"
            "  data &ds.;\n"
            "    set &ds.;\n"
            "    b = %verify(y);\n"
            "  run;\n"
            "%mend;\n"
            "%tidy(work.a);\n"
            "%tidy2(work.b);\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        all_standard = {m for b in br.batches for m in b.standard_autocall_macros}
        self.assertEqual(all_standard, {"left", "verify"})

    def test_case_insensitive_detection(self):
        cr = _C.chunk_text("%let x = %TRIM(&y);")
        self.assertIn("trim", cr.chunks[0].metadata.invokes_macros)

    def test_json_serialisable(self):
        import json

        src = (
            "%macro normalize(ds); data &ds.; set &ds.; x=%left(y); run; %mend;\n"
            "%normalize(work.a);\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        json.dumps(br.model_dump())


# ---------------------------------------------------------------------------
# 9. Phase 5 deferred-scope verification (F2, F3, G4-G6)
# ---------------------------------------------------------------------------


class TestPhase5DeferredScopeRemainsCorrect(unittest.TestCase):
    """
    Confirms the explicitly-deferred Phase 5 items behave exactly as
    documented: G4-G6's conservative non-fabrication holds under fresh
    scenarios (re-verification, not re-implementation), and full SASAUTOS
    directory scanning (F2) / SASMSTORE resolution (F3) remain undone --
    this class exists so any future accidental implementation drift (or
    accidental regression of the conservative behavior) is caught.
    """

    def test_multi_param_concatenation_never_fabricated(self):
        src = (
            "%macro snapshot(lib, prefix, suffix);\n"
            "  data &lib..&prefix._&suffix.; set &lib..&prefix.; run;\n"
            "%mend;\n"
            "%snapshot(work, daily, archive);\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        batch = next(b for b in br.batches if "snapshot" in b.defined_macros)
        self.assertEqual(batch.output_datasets, [])

    def test_scan_built_dataset_name_never_fabricated(self):
        src = (
            "%macro extract(spec);\n"
            "  data %scan(&spec., 1, .); set %scan(&spec., 2, .); run;\n"
            "%mend;\n"
            "%extract(work.orders);\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        batch = next(b for b in br.batches if "extract" in b.defined_macros)
        self.assertEqual(batch.output_datasets, [])

    def test_concatenation_does_not_falsely_link_to_unrelated_chunk(self):
        """A separately-defined, similarly-named dataset must not be
        wrongly pulled into a batch via a guessed concatenation match."""
        src = (
            "%macro snapshot(lib, prefix, suffix);\n"
            "  data &lib..&prefix._&suffix.; set &lib..&prefix.; run;\n"
            "%mend;\n"
            "%snapshot(work, daily, archive);\n"
            "data work.daily_archive_unrelated; set work.other; run;\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        snap_batch = next(b for b in br.batches if "snapshot" in b.defined_macros)
        snap_chunk_ids = set(snap_batch.chunk_ids)
        unrelated_chunk = next(
            c
            for c in cr.chunks
            if "work.daily_archive_unrelated" in c.metadata.output_datasets
        )
        self.assertNotIn(unrelated_chunk.chunk_id, snap_chunk_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
