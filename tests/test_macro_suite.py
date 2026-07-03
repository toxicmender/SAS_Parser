"""
test_macro_suite.py — the canonical index of the macro testing suite.

This file is the single entry point for "every test related to SAS macro
parsing" across the project.  It does two things, deliberately kept
separate so neither interferes with the other:

  1. Documents, exhaustively, where every macro-related test actually
     lives (see MACRO TEST LANDSCAPE below) — including the tests that
     remain embedded in non-macro-specific files for sound architectural
     reasons, not by oversight.

  2. Provides `build_macro_suite()`, which dynamically assembles every
     TestCase from the four *dedicated* macro test modules into one
     `unittest.TestSuite` at runtime, so the complete dedicated-macro
     surface can be executed in a single process via:

         python sas_chunker/tests/test_macro_suite.py

     This does NOT import or re-expose those TestCase classes as
     module-level names here.  Pytest's default collection (no
     [tool.pytest.ini_options] section exists in this project's
     pyproject.toml, so discovery uses stock settings) finds TestCase
     subclasses by scanning each collected module's *own* namespace — if
     this file re-exported the same classes by import, a plain
     `pytest sas_chunker/tests/` run would discover and execute every one
     of them *twice* (once from its defining module, once from here).
     Building the suite dynamically via `unittest.TestLoader` at runtime
     sidesteps that entirely: pytest never sees a second class definition
     in this module, only a manifest test and a `__main__` block.


MACRO TEST LANDSCAPE
=====================

Dedicated macro test files (fully consolidated; covered by
`build_macro_suite()` below)
------------------------------------------------------------------------
test_macro_classification.py   (Phase 1 + Phase 3 — "what is this
                                 statement, structurally")
  TestReservedWordExclusion          - the ~94-word Appendix-1 reserved
                                        set excluded from invokes_macros/
                                        called_macros
  TestMacroVarOp                     - %let/%global/%local/%put
                                        distinguished within GLOBAL_STATEMENT
  TestAutomaticMacroVariables        - &sys* prefix detection
  TestMacroControlFlowKind           - SasChunkKind.MACRO_CONTROL_FLOW for
                                        open-code %if/%else/%do/%end/
                                        %return/%goto/%abort
  TestAbortAndComputedGotoVisibility - contains_abort / contains_computed_goto
                                        on MACRO_DEFINITION chunks
  TestComputedGotoExtendsScopeHazard - closes the Phase 2 deferred item
  TestMacroFunctionExclusionCompleteness - Phase 4: closes the 5-function
                                        gap between Appendix 1's reserved
                                        words and Ch. 12 Table 12.3's full
                                        27-function macro-function list
  TestStandardAutocallMacroAllowlist - Phase 5 (F2b): the 10-name
                                        SAS-provided autocall allowlist,
                                        excluded from required_macros but
                                        reported via standard_autocall_macros
  TestPhase5DeferredScopeRemainsCorrect - re-verifies G4-G6's conservative
                                        non-fabrication behaviour; confirms
                                        F2 (directory scanning) and F3
                                        (SASMSTORE) remain undone, by design

test_macro_body_resolution.py  (mixed literal/parameterised macro-body
                                 dataset I/O, plus nested invocation)
  TestMacroBodyIOExtraction          - _macro_body_io() unit tests
  TestParseCallArgs                  - _parse_call_args() unit tests
  TestLiteralMacroBodyBatching        - Fix A: literal body outputs
  TestParameterisedMacroBodyBatching  - Fix B: call-site parameter resolution
  TestNestedMacroInvocation           - macro invoked inside another
                                         macro's own definition body
  TestUnresolvableReferencesStayConservative - compound refs (&lib..raw)
                                         never silently mis-resolved
  TestNewFieldsSerialisable           - JSON round-trip of body-IO fields

test_macro_variable_flow.py    (Phase 2 — CALL SYMPUT/SYMPUTX, CALL
                                 EXECUTE, PROC SQL INTO, macro_var_flow)
  TestSplitTopLevel / TestCleanLiteral / TestEnumerateNumberedRange
                                      - low-level parser unit tests
  TestCallSymputExtraction           - producer-side extraction
  TestSymputScopeHazard              - the Ch. 5 local-scope pitfall
  TestCallExecuteExtraction          - dynamic macro invocation
  TestProcSqlInto                    - all three INTO syntax forms
  TestConsumesMacrovars              - consumer-side &name scanning
  TestMacroVarFlowBatching            - end-to-end batcher integration
  TestPhase2FieldsSerialisable        - JSON round-trip

Cross-cutting macro tests that remain in their non-macro-specific files
(by design, not oversight)
------------------------------------------------------------------------
These tests exercise macro behaviour *alongside* dataset-flow, multi-file,
or general batching behaviour in the same scenario — splitting them out
would sever exactly the interaction they're meant to demonstrate (e.g. a
macro invocation chained with a dataset producer/consumer is the point of
the test, not an incidental detail).  They are intentionally left where
their narrative context lives:

  test_chunker.py
    TestSasSemanticChunker.test_macro_definition_and_call
    TestSasSemanticChunker.test_unclosed_macro_diagnostic
      -- basic chunker-level recognition, alongside the file's other
         construct-recognition tests (DATA/PROC/comments/etc).

  test_batcher.py
    TestMacroInvocation (whole class)
    TestBatchReasons.test_macro_invocation_reason_contains_macro_name
    TestBatchReasons.test_mixed_reason_contains_both_edge_types
    TestComplexPrograms.test_macro_library_pattern
      -- macro_invocation/macro_arg_dataset edges demonstrated alongside
         dataset_flow edges in the same batching scenarios.

  test_multifile.py
    TestCrossFileMacroInvocation (whole class)
    TestCrossFileMacroArgDataset (whole class)
    TestTransitiveCrossFile.test_macro_chain_plus_dataset_chain
    TestBatchIOFields.test_cross_file_required_macros
    TestBatchIOFields.test_cross_file_defined_macros
    TestComplexMultiFile.test_macro_lib_etl_report_three_files
      -- macro dependency resolution specifically in its cross-file form,
         alongside the file's other cross-file dataset-flow scenarios.

Run everything macro-related, including the cross-cutting tests above:
    pytest sas_chunker/tests/ -k "macro or Macro"

Run only the four fully-dedicated files via pytest (each independently
collected and run exactly once, no double-counting):
    pytest sas_chunker/tests/test_macro_classification.py \\
           sas_chunker/tests/test_macro_body_resolution.py \\
           sas_chunker/tests/test_macro_variable_flow.py -v

Run the dedicated suite as one process via this file's manifest:
    python sas_chunker/tests/test_macro_suite.py
"""

from __future__ import annotations

import pathlib
import sys
import unittest

_THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parents[1]))  # makes `sas_chunker` importable
sys.path.insert(0, str(_THIS_DIR))  # makes sibling test modules importable
# (sas_chunker/tests/ has no __init__.py,
# so these are plain top-level imports,
# not a sas_chunker.tests sub-package)

# Imported only as modules (never their TestCase classes), specifically to
# avoid the pytest double-collection problem described above.
import test_macro_body_resolution as _body_resolution
import test_macro_classification as _classification
import test_macro_variable_flow as _variable_flow

_DEDICATED_MACRO_MODULES = (
    _classification,
    _body_resolution,
    _variable_flow,
)


def build_macro_suite() -> unittest.TestSuite:
    """
    Assemble every TestCase from the dedicated macro test modules into one
    runnable suite, built dynamically via TestLoader so this module itself
    never holds a second copy of any TestCase class.
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for module in _DEDICATED_MACRO_MODULES:
        suite.addTests(loader.loadTestsFromModule(module))
    return suite


def count_macro_tests() -> int:
    """Total number of individual test methods across the dedicated suite."""
    return build_macro_suite().countTestCases()


# ---------------------------------------------------------------------------
# Manifest integrity check — a tripwire, not a re-run of the dedicated
# suite itself.  Catches accidental breakage (a module fails to import, a
# test class is silently deleted) without duplicating any other file's
# actual test execution.
# ---------------------------------------------------------------------------


class TestMacroSuiteManifestIntegrity(unittest.TestCase):
    def test_all_dedicated_modules_importable(self):
        for module in _DEDICATED_MACRO_MODULES:
            self.assertTrue(hasattr(module, "__file__"), f"{module} failed to import")

    def test_suite_assembles_without_error(self):
        suite = build_macro_suite()
        self.assertGreater(suite.countTestCases(), 0)

    def test_minimum_expected_test_class_count_per_module(self):
        """
        A structural tripwire: if a future refactor accidentally deletes a
        test class from one of the dedicated modules, this fails loudly
        rather than silently shrinking coverage.  Thresholds are set just
        below the actual counts at time of writing (6 / 7 / 10 classes).
        """
        loader = unittest.TestLoader()
        minimums = {
            _classification: 8,
            _body_resolution: 6,
            _variable_flow: 9,
        }
        for module, minimum in minimums.items():
            suite = loader.loadTestsFromModule(module)
            class_count = len({test.__class__ for test in _iter_test_cases(suite)})
            self.assertGreaterEqual(
                class_count,
                minimum,
                f"{module.__name__} has only {class_count} test classes, "
                f"expected at least {minimum} -- a test class may have "
                f"been accidentally removed",
            )

    def test_total_dedicated_macro_test_count_is_substantial(self):
        """A coarse sanity floor on the overall dedicated-suite size."""
        self.assertGreaterEqual(count_macro_tests(), 155)


def _iter_test_cases(suite: unittest.TestSuite):
    """Flatten a (possibly nested) TestSuite into individual TestCase instances."""
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_test_cases(item)
        else:
            yield item


if __name__ == "__main__":
    print(
        f"Dedicated macro test suite: {count_macro_tests()} test(s) "
        f"across {len(_DEDICATED_MACRO_MODULES)} module(s)\n"
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(build_macro_suite())
    sys.exit(0 if result.wasSuccessful() else 1)
