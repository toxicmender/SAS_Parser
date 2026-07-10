"""
test_function_detection.py — tests for the recognized_functions /
recognized_call_routines metadata fields (_SAS_FUNCTION_CALL_RE,
_SAS_CALL_ROUTINE_RE and the _function_scan_text pre-blanking pass).

The function scan is advisory metadata for an LLM translator (an inventory
of DATA-step built-ins a chunk uses), so its failure mode is polluting that
inventory rather than mis-chunking — but the pollution was real.  Confirmed
false-positive classes, each pinned here:

1. Macro-language names reported as DATA-step functions — ``%scan(...)``,
   ``%index(...)``, ``%length(...)``, ``%put (text);``.  The old pattern's
   leading ``\\b`` treated the ``%`` as an ordinary word boundary.

2. User macros *named* like functions — ``%compress(&ds)`` was reported both
   as macro call ``compress`` (correct) and function ``compress`` (wrong);
   ``%macro compress(x);`` definition headers likewise.

3. INPUT/PUT *statements* in grouped-list form — ``input (v1 v2) ($8. 2.);``
   / ``put (a b) (=);`` reported the INPUT()/PUT() functions.  Two
   back-to-back parenthesised groups directly after the keyword is never
   valid function-call syntax, which is what _GROUPED_INPUT_PUT_STMT_RE keys
   on — so SQL's single-group ``case when ... then put(x, fmt.)`` still
   detects.

4. Member access — hash-object methods (``h.find()``, ``h.first()``) and
   ``&pfx.name(`` macro-variable concatenation.

Fix: ``(?<![%&.\\w])`` lookbehind on the function pattern (classes 1, 2
call-site, 4) plus _function_scan_text blanking %MACRO definition headers
and grouped-list INPUT/PUT keywords (classes 2 definition-site, 3).

Run:  python -m pytest tests/test_function_detection.py -v
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chunker import SasSemanticChunker

_C = SasSemanticChunker(min_words=1, max_words=9_999)


def _functions(code: str) -> set[str]:
    """Union of recognized_functions across every chunk of *code*."""
    return {
        f for c in _C.chunk_text(code).chunks for f in c.metadata.recognized_functions
    }


def _call_routines(code: str) -> set[str]:
    """Union of recognized_call_routines across every chunk of *code*."""
    return {
        r
        for c in _C.chunk_text(code).chunks
        for r in c.metadata.recognized_call_routines
    }


def _component_objects(code: str) -> set[str]:
    """Union of component_objects across every chunk of *code*."""
    return {
        o for c in _C.chunk_text(code).chunks for o in c.metadata.component_objects
    }


# ---------------------------------------------------------------------------
# False positives — none of these are DATA-step function calls
# ---------------------------------------------------------------------------


class TestFunctionFalsePositives(unittest.TestCase):
    def test_libname_engine_statement_reports_nothing(self):
        """
        Real-world LIBNAME statement with engine options, quoted values and
        macro-variable credentials — no parenthesis anywhere, so nothing may
        register (regression guard for the originally reported statement).
        """
        code = (
            "libname edwprod oracle path=EDWPRO_READ_ONLY schema=FR_DM_Pro "
            'connection="global"\n'
            'connection_group = "EDWPROD_READ_ONLY" user="&username." '
            'pass="&user_pass.";\n'
            "run;\n"
        )
        self.assertEqual(_functions(code), set())
        self.assertEqual(_call_routines(code), set())

    def test_macro_functions_not_reported(self):
        """%scan/%index/%length are macro-language, not DATA-step calls."""
        code = (
            "%let word = %scan(&list, 1); "
            "%let pos = %index(&s, x); "
            "%let n = %length(&s);"
        )
        self.assertEqual(_functions(code), set())

    def test_percent_put_with_parenthesised_text(self):
        self.assertEqual(_functions("%put (checkpoint reached);"), set())

    def test_macro_invocation_named_like_function(self):
        """
        A user macro sharing a function's name stays a macro call only —
        it must not be double-reported as the function.
        """
        code = "%compress(&ds);"
        self.assertEqual(_functions(code), set())
        invoked = {
            m for c in _C.chunk_text(code).chunks for m in c.metadata.invokes_macros
        }
        self.assertIn("compress", invoked)

    def test_macro_definition_header_named_like_function(self):
        self.assertEqual(_functions("%macro compress(x); %mend compress;"), set())

    def test_input_statement_grouped_list(self):
        code = "data a; infile f; input (v1 v2 v3) ($8. 2. 2.); run;"
        self.assertEqual(_functions(code), set())

    def test_put_statement_grouped_list(self):
        code = "data _null_; set a; put (v1 v2) (=); run;"
        self.assertEqual(_functions(code), set())

    def test_put_statement_after_then_else(self):
        code = "data a; if x then put (a b) (=); else put (c d) (=); run;"
        self.assertEqual(_functions(code), set())

    def test_hash_object_methods(self):
        code = "data a; declare hash h(); rc = h.find(); rc = h.first(); run;"
        self.assertEqual(_functions(code), set())

    def test_macro_var_dot_concatenation(self):
        self.assertEqual(_functions("%let x = &pfx.scan(1);"), set())

    def test_function_name_inside_string_literal(self):
        self.assertEqual(_functions('title "run min(x) daily";'), set())


# ---------------------------------------------------------------------------
# Genuine calls — every one of these must still be detected
# ---------------------------------------------------------------------------


class TestFunctionTruePositives(unittest.TestCase):
    def test_input_function(self):
        self.assertEqual(_functions("data a; x = input(s, 8.); run;"), {"input"})

    def test_put_function_inside_call_routine(self):
        code = "data _null_; call symputx('m', put(x, 8.)); run;"
        self.assertEqual(_functions(code), {"put"})
        self.assertEqual(_call_routines(code), {"symputx"})

    def test_put_function_in_if_condition(self):
        code = "data a; if put(x, fmt.) = 'A' then y = 1; run;"
        self.assertEqual(_functions(code), {"put"})

    def test_sysfunc_wrapped_functions(self):
        """The inner name of %sysfunc(fn(...)) is a real function usage."""
        code = "%if %sysfunc(exist(work.t)) %then %do; %put yes; %end;"
        self.assertEqual(_functions(code), {"exist"})
        code = "%let v = %sysfunc(inputn(&d, date9.));"
        self.assertEqual(_functions(code), {"inputn"})

    def test_proc_sql_summary_functions(self):
        code = (
            "proc sql; create table t as "
            "select min(x) as mn, count(*) as n from b; quit;"
        )
        self.assertEqual(_functions(code), {"count", "min"})

    def test_sql_case_when_then_put(self):
        """Single-group PUT() after CASE ... THEN is a call, not a statement."""
        code = (
            "proc sql; create table t as "
            "select case when x = 1 then put(y, 8.) else 'z' end as lbl "
            "from b; quit;"
        )
        self.assertEqual(_functions(code), {"put"})

    def test_multiple_functions_in_expression(self):
        code = 'data a; d = intnx("month", today(), -1); s = strip(nm); run;'
        self.assertEqual(_functions(code), {"intnx", "strip", "today"})

    def test_hashing_function_call(self):
        """A genuine HASHING() call (crypto checksum) is a function usage."""
        code = "data a; set b; digest = hashing('sha256', payload); run;"
        self.assertEqual(_functions(code), {"hashing"})


# ---------------------------------------------------------------------------
# DATA step component objects (hash, hiter, ...) — declaration-keyed detection
# ---------------------------------------------------------------------------


class TestComponentObjectDetection(unittest.TestCase):
    def test_declare_hash_with_dataset_argument(self):
        """Hash object declared with a quoted dataset: argument (the shape
        originally reported missing) registers as a component object, and its
        dot-method calls still register nothing as functions."""
        code = (
            "data a;\n"
            "declare\n"
            '  hash h1(dataset:"monthly_notes(rename=(cf_total_auth_cnt = '
            'cf_monthly_notes lag_month_notes = lag_month_max))");\n'
            "  h1.definekey('admit_type', 'reg_cf', 'lag_month_max');\n"
            "  h1.definedata('cf_monthly_notes');\n"
            "  h1.definedone();\n"
            "run;\n"
        )
        self.assertEqual(_component_objects(code), {"hash"})
        self.assertEqual(_functions(code), set())
        self.assertEqual(_call_routines(code), set())

    def test_dcl_abbreviation(self):
        code = "data a; dcl hash h(); h.definedone(); run;"
        self.assertEqual(_component_objects(code), {"hash"})

    def test_new_operator(self):
        code = "data a; declare hash h; h = _new_ hash(); run;"
        self.assertEqual(_component_objects(code), {"hash"})

    def test_hash_iterator(self):
        code = "data a; declare hiter hi('h1'); rc = hi.first(); run;"
        self.assertEqual(_component_objects(code), {"hiter"})

    def test_variable_named_hash_not_reported(self):
        """`hash` as an ordinary variable name is not a declaration."""
        code = "data a; set b; hash = md5(s); run;"
        self.assertEqual(_component_objects(code), set())


if __name__ == "__main__":
    unittest.main()
