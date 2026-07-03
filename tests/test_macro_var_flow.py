"""
test_phase2_macrovar_flow.py — tests for MACRO_PARSING_ROADMAP.md Phase 2.

Covers four constructs, all grounded in SAS Macro Language: Reference
(Chapters 5, 12, 15, 18):

1. CALL SYMPUT / CALL SYMPUTX (C5) — DATA-step side effects that create a
   macro variable, tracked via SasChunkMetadata.produces_macrovars.
2. CALL SYMPUT/SYMPUTX local-scope hazard (C5c) — flags the documented
   Ch. 5 pitfall where a macro with a non-empty local symbol table
   silently scopes the created variable locally instead of globally.
3. CALL EXECUTE (C5b) — dynamic macro invocation from a DATA step,
   trackable only when the argument is statically resolvable.
4. PROC SQL INTO (C6) — all three documented syntax forms.
5. End-to-end macro_var_flow batching in batcher.py, including the new
   produces_macrovar index and SasBatch.produced_macrovars/required_macrovars.

Run:  python -m pytest tests/test_macro_variable_flow.py -v
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from chunker import (
    SasChunkBatcher,
    SasChunkKind,
    SasCorpus,
    SasSemanticChunker,
)
from chunker.batcher import MultiFileBatcher
from chunker.chunker import (
    _clean_literal,
    _enumerate_numbered_range,
    _extract_call_execute_macros,
    _extract_sql_into_vars,
    _extract_symput,
    _macro_has_local_scope,
    _split_top_level,
)

_C = SasSemanticChunker(min_words=1, max_words=9_999)


# ---------------------------------------------------------------------------
# 1. Low-level parser unit tests
# ---------------------------------------------------------------------------


class TestSplitTopLevel(unittest.TestCase):
    """Paren/quote-aware argument splitter, tested against the exact
    examples used in the Reference manual's CALL SYMPUT dictionary entry."""

    def test_simple_two_args(self):
        self.assertEqual(_split_top_level("'new','testing'"), ["'new'", "'testing'"])

    def test_nested_function_calls_not_split(self):
        self.assertEqual(
            _split_top_level("'holdate',trim(left(put(holiday,worddate.)))"),
            ["'holdate'", "trim(left(put(holiday,worddate.)))"],
        )

    def test_concatenation_expression_not_split(self):
        self.assertEqual(
            _split_top_level("'POS'||left(_n_), position"),
            ["'POS'||left(_n_)", "position"],
        )

    def test_three_args_with_quoted_scope(self):
        self.assertEqual(
            _split_top_level("'unit', 1000, 'G'"),
            ["'unit'", "1000", "'G'"],
        )


class TestCleanLiteral(unittest.TestCase):
    def test_single_quoted(self):
        self.assertEqual(_clean_literal("'new'"), "new")

    def test_double_quoted(self):
        self.assertEqual(_clean_literal('"new"'), "new")

    def test_concatenation_not_literal(self):
        self.assertIsNone(_clean_literal("'POS'||left(_n_)"))

    def test_bare_variable_not_literal(self):
        self.assertIsNone(_clean_literal("position"))

    def test_literal_with_space(self):
        self.assertEqual(_clean_literal("'a token'"), "a token")


class TestEnumerateNumberedRange(unittest.TestCase):
    def test_simple_range(self):
        self.assertEqual(
            _enumerate_numbered_range("type1", "type4"),
            ["type1", "type2", "type3", "type4"],
        )

    def test_single_element_range(self):
        self.assertEqual(_enumerate_numbered_range("type1", "type1"), ["type1"])

    def test_non_numbered_returns_none(self):
        self.assertIsNone(_enumerate_numbered_range("foo", "bar"))

    def test_mismatched_prefix_returns_none(self):
        self.assertIsNone(_enumerate_numbered_range("type1", "kind4"))

    def test_descending_range_returns_none(self):
        self.assertIsNone(_enumerate_numbered_range("type4", "type1"))


# ---------------------------------------------------------------------------
# 2. CALL SYMPUT / CALL SYMPUTX extraction (C5)
# ---------------------------------------------------------------------------


class TestCallSymputExtraction(unittest.TestCase):
    def test_literal_name_extracted(self):
        produced, unresolved, _ = _extract_symput(
            "call symput('holdate', trim(left(put(holiday,worddate.))));"
        )
        self.assertEqual(produced, ["holdate"])
        self.assertFalse(unresolved)

    def test_dynamic_name_flagged_unresolved(self):
        produced, unresolved, _ = _extract_symput("call symput(position, player);")
        self.assertEqual(produced, [])
        self.assertTrue(unresolved)

    def test_symputx_with_explicit_global_scope(self):
        produced, _, explicit_global = _extract_symput(
            "call symputx('unit', 1000, 'G');"
        )
        self.assertEqual(produced, ["unit"])
        self.assertEqual(explicit_global, ["unit"])

    def test_symputx_with_local_scope_not_in_explicit_global(self):
        produced, _, explicit_global = _extract_symput(
            "call symputx('unit', 1000, 'L');"
        )
        self.assertEqual(produced, ["unit"])
        self.assertEqual(explicit_global, [])

    def test_symput_no_third_arg_not_in_explicit_global(self):
        """Plain CALL SYMPUT has no scope-override argument at all."""
        produced, _, explicit_global = _extract_symput("call symput('myvar1', x);")
        self.assertEqual(produced, ["myvar1"])
        self.assertEqual(explicit_global, [])

    def test_multiple_calls_in_one_chunk(self):
        text = "call symput('a', 1);\ncall symput('b', 2);\n"
        produced, _, _ = _extract_symput(text)
        self.assertEqual(set(produced), {"a", "b"})

    def test_metadata_field_end_to_end_top_level(self):
        cr = _C.chunk_text(
            "data c;\n"
            "  input holiday mmddyy.;\n"
            "  call symput('holdate', trim(left(put(holiday,worddate.))));\n"
            "  datalines;\n070497\n;\nrun;\n"
        )
        self.assertEqual(cr.chunks[0].metadata.produces_macrovars, ["holdate"])

    def test_metadata_field_inside_macro_definition(self):
        """CALL SYMPUT inside a macro body attributes the produced
        variable to the enclosing MACRO_DEFINITION chunk (there is no
        separate nested chunk for the embedded DATA step)."""
        cr = _C.chunk_text(
            "%macro setit;\ndata _null_;\n  call symput('x', 1);\nrun;\n%mend;\n"
        )
        self.assertEqual(cr.chunks[0].kind, SasChunkKind.MACRO_DEFINITION)
        self.assertEqual(cr.chunks[0].metadata.produces_macrovars, ["x"])


# ---------------------------------------------------------------------------
# 3. CALL SYMPUT/SYMPUTX local-scope hazard (C5c)
# ---------------------------------------------------------------------------


class TestSymputScopeHazard(unittest.TestCase):
    def test_manual_example_exact_reproduction(self):
        """The literal example from Ch. 5 of the Reference manual."""
        cr = _C.chunk_text(
            "%macro env1(param1);\n"
            "data _null_;\n"
            "  x = 'a token';\n"
            "  call symput('myvar1',x);\n"
            "run;\n"
            "%mend env1;\n"
        )
        m = cr.chunks[0].metadata
        self.assertTrue(m.symput_scope_hazard)
        self.assertEqual(m.symput_hazard_vars, ["myvar1"])

    def test_no_hazard_without_local_scope(self):
        """A macro with no parameters and no %local has an empty local
        symbol table, so CALL SYMPUT walks up to the global table safely."""
        cr = _C.chunk_text(
            "%macro env3;\n"
            "data _null_;\n"
            "  call symput('myvar3', 1);\n"
            "run;\n"
            "%mend env3;\n"
        )
        self.assertFalse(cr.chunks[0].metadata.symput_scope_hazard)

    def test_hazard_triggered_by_local_statement_without_parameters(self):
        """An explicit %LOCAL also makes the symbol table non-empty, even
        with zero declared parameters."""
        cr = _C.chunk_text(
            "%macro env4;\n"
            "%local i;\n"
            "data _null_;\n"
            "  call symput('myvar4', 1);\n"
            "run;\n"
            "%mend env4;\n"
        )
        self.assertTrue(cr.chunks[0].metadata.symput_scope_hazard)

    def test_explicit_global_override_suppresses_hazard(self):
        """CALL SYMPUTX's third argument forcing 'G' means the author
        deliberately chose global scope -- not a hazard."""
        cr = _C.chunk_text(
            "%macro env2(param1);\n"
            "data _null_;\n"
            "  call symputx('myvar2', 99, 'G');\n"
            "run;\n"
            "%mend env2;\n"
        )
        self.assertFalse(cr.chunks[0].metadata.symput_scope_hazard)

    def test_explicit_local_override_does_not_suppress_hazard(self):
        """'L' is still effectively local -- the hazard (mismatch with the
        intended global use) still applies since the symbol table is
        already non-empty for other reasons."""
        cr = _C.chunk_text(
            "%macro env5(param1);\n"
            "data _null_;\n"
            "  call symputx('myvar5', 99, 'L');\n"
            "run;\n"
            "%mend env5;\n"
        )
        self.assertTrue(cr.chunks[0].metadata.symput_scope_hazard)

    def test_top_level_data_step_never_hazardous(self):
        """No enclosing macro at all -- there is no local-scope risk."""
        cr = _C.chunk_text("data _null_;\n  call symput('x', 1);\nrun;\n")
        self.assertFalse(cr.chunks[0].metadata.symput_scope_hazard)

    def test_macro_has_local_scope_helper_with_params(self):
        self.assertTrue(_macro_has_local_scope("data _null_; run;", ["ds"]))

    def test_macro_has_local_scope_helper_with_local_stmt(self):
        self.assertTrue(_macro_has_local_scope("%local i; data _null_; run;", []))

    def test_macro_has_local_scope_helper_neither(self):
        self.assertFalse(_macro_has_local_scope("data _null_; run;", []))

    def test_dynamic_name_does_not_populate_hazard_vars(self):
        """A hazard from a dynamically-named CALL SYMPUT is not flagged at
        all (the name is unresolved, so we cannot know what it is) --
        confirms the hazard logic only operates on resolvable names."""
        cr = _C.chunk_text(
            "%macro env6(param1);\n"
            "data _null_;\n"
            "  call symput(some_var, x);\n"
            "run;\n"
            "%mend env6;\n"
        )
        self.assertFalse(cr.chunks[0].metadata.symput_scope_hazard)
        self.assertEqual(cr.chunks[0].metadata.symput_hazard_vars, [])


# ---------------------------------------------------------------------------
# 4. CALL EXECUTE extraction (C5b)
# ---------------------------------------------------------------------------


class TestCallExecuteExtraction(unittest.TestCase):
    def test_clean_literal_invocation(self):
        found = _extract_call_execute_macros("call execute('%sales');")
        self.assertEqual(found, ["sales"])

    def test_concatenation_with_literal_prefix(self):
        found = _extract_call_execute_macros("call execute('%sales('||month||')');")
        self.assertEqual(found, ["sales"])

    def test_unquoted_variable_unresolved(self):
        found = _extract_call_execute_macros("call execute(findobs);")
        self.assertEqual(found, [])

    def test_double_quoted_literal(self):
        found = _extract_call_execute_macros('call execute("%report");')
        self.assertEqual(found, ["report"])

    def test_metadata_end_to_end(self):
        cr = _C.chunk_text("data work.a;\n  call execute('%sales');\nrun;\n")
        self.assertIn("sales", cr.chunks[0].metadata.invokes_macros)

    def test_call_execute_inside_macro_definition(self):
        cr = _C.chunk_text(
            "%macro overdue;\n"
            "data Work.Billed;\n"
            "  call execute('%report');\n"
            "run;\n"
            "%mend overdue;\n"
        )
        self.assertEqual(cr.chunks[0].kind, SasChunkKind.MACRO_DEFINITION)
        self.assertIn("report", cr.chunks[0].metadata.invokes_macros)

    def test_resolved_call_execute_creates_real_batching_edge(self):
        """A resolved CALL EXECUTE target must batch with its own
        definition, exactly like a normal %name(...) invocation."""
        src = (
            "%macro report; proc print data=work.a; run; %mend;\n"
            "data work.a;\n"
            "  call execute('%report');\n"
            "run;\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        self.assertEqual(len(br.batches), 1)
        self.assertIn("report", br.batches[0].defined_macros)


# ---------------------------------------------------------------------------
# 5. PROC SQL INTO extraction (C6) -- all three syntax forms
# ---------------------------------------------------------------------------


class TestProcSqlInto(unittest.TestCase):
    def test_form1_single_row_named_vars(self):
        found = _extract_sql_into_vars(
            "select style, sqfeet into :type, :size from sasuser.houses;"
        )
        self.assertEqual(set(found), {"type", "size"})

    def test_form2_numbered_range_with_dash(self):
        found = _extract_sql_into_vars(
            "select style into :type1 - :type4 notrim from sasuser.houses;"
        )
        self.assertEqual(found, ["type1", "type2", "type3", "type4"])

    def test_form2_numbered_range_with_through(self):
        found = _extract_sql_into_vars(
            "select style into :type1 through :type3 from sasuser.houses;"
        )
        self.assertEqual(found, ["type1", "type2", "type3"])

    def test_form2_numbered_range_with_thru(self):
        found = _extract_sql_into_vars(
            "select style into :type1 thru :type2 from sasuser.houses;"
        )
        self.assertEqual(found, ["type1", "type2"])

    def test_form3_separated_by(self):
        found = _extract_sql_into_vars(
            "select distinct style into :types separated by ', ' from sasuser.houses;"
        )
        self.assertEqual(found, ["types"])

    def test_metadata_only_populated_for_proc_sql(self):
        """A non-SQL PROC step must never populate produces_macrovars even
        if it coincidentally contains a colon-prefixed token."""
        cr = _C.chunk_text("proc print data=work.a; run;")
        self.assertEqual(cr.chunks[0].metadata.produces_macrovars, [])

    def test_metadata_end_to_end_proc_sql(self):
        cr = _C.chunk_text(
            "proc sql;\n  select max(order_dt) into :max_dt from work.orders;\nquit;\n"
        )
        self.assertEqual(cr.chunks[0].kind, SasChunkKind.PROC_STEP)
        self.assertEqual(cr.chunks[0].metadata.produces_macrovars, ["max_dt"])

    def test_multiple_into_clauses_in_one_proc_sql_step(self):
        cr = _C.chunk_text(
            "proc sql;\n"
            "  select max(dt) into :max_dt from work.a;\n"
            "  select min(dt) into :min_dt from work.a;\n"
            "quit;\n"
        )
        self.assertEqual(
            set(cr.chunks[0].metadata.produces_macrovars),
            {"max_dt", "min_dt"},
        )


# ---------------------------------------------------------------------------
# 6. consumes_macrovars field
# ---------------------------------------------------------------------------


class TestConsumesMacrovars(unittest.TestCase):
    def test_basic_reference_tracked(self):
        cr = _C.chunk_text('data work.a;\n  set work.b;\n  x = "&cutoff_date";\nrun;\n')
        self.assertIn("cutoff_date", cr.chunks[0].metadata.consumes_macrovars)

    def test_automatic_vars_excluded(self):
        cr = _C.chunk_text('title "Report run &sysdate9";')
        self.assertEqual(cr.chunks[0].metadata.consumes_macrovars, [])

    def test_own_macro_parameter_excluded(self):
        """A macro's own parameter reference is not a corpus-level
        macro-variable dependency -- it's resolved at the call site."""
        cr = _C.chunk_text("%macro clean(ds);\n  data &ds.; set &ds.; run;\n%mend;\n")
        self.assertEqual(cr.chunks[0].metadata.consumes_macrovars, [])

    def test_outer_scope_reference_inside_macro_is_tracked(self):
        """A reference to a variable that is NOT one of the macro's own
        parameters (e.g. set by an enclosing/global context) IS tracked."""
        cr = _C.chunk_text(
            "%macro report(ds);\n"
            '  proc print data=&ds.; title "&report_title"; run;\n'
            "%mend;\n"
        )
        self.assertIn("report_title", cr.chunks[0].metadata.consumes_macrovars)
        self.assertNotIn("ds", cr.chunks[0].metadata.consumes_macrovars)


# ---------------------------------------------------------------------------
# 7. End-to-end macro_var_flow batching
# ---------------------------------------------------------------------------


class TestMacroVarFlowBatching(unittest.TestCase):
    def test_symput_producer_links_to_consumer(self):
        src = (
            "data _null_;\n"
            "  call symput('cutoff_date', '01JAN2020');\n"
            "run;\n"
            "data work.recent;\n"
            "  set mylib.raw;\n"
            '  where order_dt >= "&cutoff_date"d;\n'
            "run;\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.batches[0].chunks), 2)
        self.assertIn("cutoff_date", br.batches[0].produced_macrovars)
        self.assertIn("macro_var_flow", br.batches[0].reason)

    def test_sql_into_producer_links_to_consumer(self):
        src = (
            "proc sql;\n"
            "  select max(order_dt) into :max_dt from work.orders;\n"
            "quit;\n"
            "data work.filtered;\n"
            "  set work.orders;\n"
            '  where order_dt = "&max_dt"d;\n'
            "run;\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        self.assertEqual(len(br.batches), 1)
        self.assertIn("max_dt", br.batches[0].produced_macrovars)

    def test_unresolved_consumer_external_to_batch(self):
        """A macro variable referenced but never produced anywhere in the
        corpus is reported as required_macrovars, not silently dropped."""
        src = (
            "data work.recent;\n"
            "  set mylib.raw;\n"
            '  where order_dt >= "&external_cutoff"d;\n'
            "run;\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        self.assertEqual(len(br.batches), 0)
        self.assertEqual(len(br.singletons), 1)
        self.assertIn("external_cutoff", br.singletons[0].metadata.consumes_macrovars)

    def test_cross_file_macro_var_flow(self):
        cr1 = _C.chunk_text(
            "proc sql;\n  select max(order_dt) into :max_dt from work.orders;\nquit;\n",
            source_id="setup.sas",
        )
        cr2 = _C.chunk_text(
            'data work.filtered;\n  set work.orders;\n  where order_dt = "&max_dt"d;\nrun;\n',
            source_id="filter.sas",
        )
        corpus = SasCorpus(file_results=[cr1, cr2])
        br = MultiFileBatcher().batch(corpus)
        self.assertEqual(len(br.batches), 1)
        self.assertTrue(br.batches[0].is_cross_file)
        self.assertIn("max_dt", br.batches[0].produced_macrovars)
        self.assertEqual(
            set(br.batches[0].source_files),
            {"setup.sas", "filter.sas"},
        )

    def test_self_reference_does_not_create_self_loop(self):
        """A chunk that both produces and (trivially) references the same
        macro variable name must not create a degenerate self-edge."""
        src = "data _null_;\n  call symput('total', 1);\n  put \"&total\";\nrun;\n"
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        # Single chunk, no batch needed -- self-reference is naturally a no-op
        self.assertEqual(len(br.batches), 0)
        self.assertEqual(len(br.singletons), 1)

    def test_does_not_interfere_with_dataset_flow_or_macro_invocation(self):
        """A complex program mixing dataset_flow, macro_invocation, and
        macro_var_flow edges must batch everything into one group."""
        src = (
            "%macro clean(ds); data &ds.; set &ds.; run; %mend;\n"
            "proc sql;\n  select max(dt) into :max_dt from mylib.raw;\nquit;\n"
            "data work.orders;\n"
            "  set mylib.raw;\n"
            '  where dt <= "&max_dt"d;\n'
            "run;\n"
            "%clean(work.orders);\n"
            "proc print data=work.orders; run;\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.batches[0].chunks), 5)
        self.assertIn("max_dt", br.batches[0].produced_macrovars)


# ---------------------------------------------------------------------------
# 8. JSON serialisability of all new fields
# ---------------------------------------------------------------------------


class TestPhase2FieldsSerialisable(unittest.TestCase):
    def test_chunk_result_serialises(self):
        import json

        src = (
            "%macro env1(param1);\n"
            "data _null_;\n  call symput('myvar1',x);\nrun;\n"
            "%mend env1;\n"
            "proc sql;\n  select max(dt) into :max_dt from work.a;\nquit;\n"
        )
        cr = _C.chunk_text(src)
        json.dumps(cr.model_dump())

    def test_batch_result_serialises(self):
        import json

        src = (
            "data _null_;\n  call symput('cutoff_date', '01JAN2020');\nrun;\n"
            'data work.recent;\n  set mylib.raw;\n  where dt >= "&cutoff_date"d;\nrun;\n'
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        json.dumps(br.model_dump())


if __name__ == "__main__":
    unittest.main(verbosity=2)
