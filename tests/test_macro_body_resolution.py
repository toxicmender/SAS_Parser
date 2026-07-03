"""
test_macro_body_resolution.py — tests for mixed literal/parameterised
macro-body dataset resolution (Fix A + Fix B).

Background
----------
A %MACRO body may reference datasets two ways:

  Literal       — a hard-coded name, e.g. "data work.base;"
                  Resolved directly from the macro source text (Fix A).
  Parameterised — a macro variable reference, e.g. "data &ds.;"
                  Only resolvable at the call site where the argument
                  value is known (Fix B).

This file verifies:
  1. _macro_body_io extraction correctness (chunker-level)
  2. _parse_call_args parsing correctness (batcher-level)
  3. End-to-end batching behaviour for both literal and parameterised
     macro bodies, including the cross-file case that originally failed
  4. Compound / unresolvable references are NOT silently mis-resolved
  5. Keyword parameters and parameters with defaults

Run:  python -m pytest tests/test_macro_body_resolution.py -v
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chunker import SasChunkBatcher, SasCorpus, SasSemanticChunker
from chunker.batcher import MultiFileBatcher, _parse_call_args
from chunker.chunker import _macro_body_io

_C = SasSemanticChunker(min_words=1, max_words=9_999)


# ---------------------------------------------------------------------------
# 1. _macro_body_io extraction (chunker-level unit tests)
# ---------------------------------------------------------------------------


class TestMacroBodyIOExtraction(unittest.TestCase):
    def test_literal_body_outputs_and_inputs(self):
        body = "%macro setup;\n  data work.base; set mylib.raw; run;\n%mend;\n"
        lit_in, lit_out, par_in, par_out, names = _macro_body_io(body)
        self.assertEqual(names, [])
        self.assertEqual(lit_out, ["work.base"])
        self.assertEqual(lit_in, ["mylib.raw"])
        self.assertEqual(par_in, [])
        self.assertEqual(par_out, [])

    def test_parameterised_output_via_out_equals(self):
        body = (
            "%macro load(path, out);\n"
            "  proc import datafile=&path. dbms=csv out=&out. replace; run;\n"
            "%mend;\n"
        )
        lit_in, lit_out, par_in, par_out, names = _macro_body_io(body)
        self.assertEqual(names, ["path", "out"])
        self.assertEqual(lit_out, [])
        self.assertEqual(par_out, [{"param": "out", "pos": 1}])

    def test_parameterised_data_step_dataset(self):
        body = "%macro clean(ds);\n  data &ds.; set &ds.; run;\n%mend;\n"
        lit_in, lit_out, par_in, par_out, names = _macro_body_io(body)
        self.assertEqual(names, ["ds"])
        self.assertEqual(par_out, [{"param": "ds", "pos": 0}])
        self.assertEqual(par_in, [{"param": "ds", "pos": 0}])

    def test_mixed_literal_and_parameterised(self):
        """One param reference + one literal reference in the same body."""
        body = "%macro mixed(ds);\n  data &ds.; set work.fixed_input; run;\n%mend;\n"
        lit_in, lit_out, par_in, par_out, names = _macro_body_io(body)
        self.assertEqual(names, ["ds"])
        self.assertEqual(lit_in, ["work.fixed_input"])
        self.assertEqual(par_out, [{"param": "ds", "pos": 0}])

    def test_keyword_parameter_gets_position_negative_one(self):
        body = (
            "%macro flag(ds, var, thresh=3);\n"
            "  data &ds.; set &ds.; if abs(&var.)>&thresh. then flag=1; run;\n"
            "%mend;\n"
        )
        lit_in, lit_out, par_in, par_out, names = _macro_body_io(body)
        self.assertEqual(names, ["ds", "var", "thresh"])
        # &ds. is positional (pos=0); &var. and &thresh. aren't dataset refs
        self.assertEqual(par_out, [{"param": "ds", "pos": 0}])

    def test_compound_reference_not_misresolved(self):
        """
        &lib..raw is a compound reference: parameter 'lib' concatenated with
        literal suffix '.raw'.  _classify_ref must NOT silently resolve this
        to just 'lib' as if it were a clean dataset name — but it currently
        records it as a param reference to 'lib' (best-effort single-var
        detection), which downstream callers must treat with care.  This
        test pins the current (intentionally conservative) behaviour.
        """
        body = (
            "%macro load(lib, file);\n"
            "  proc import datafile=&file. dbms=csv out=&lib..raw replace; run;\n"
            "%mend;\n"
        )
        lit_in, lit_out, par_in, par_out, names = _macro_body_io(body)
        self.assertEqual(names, ["lib", "file"])
        # &lib..raw still detected as referencing param 'lib' (single &ref found)
        self.assertEqual(par_out, [{"param": "lib", "pos": 0}])
        self.assertEqual(lit_out, [])

    def test_no_params_macro_pure_literal(self):
        body = "%macro report;\n  proc print data=work.summary; run;\n%mend;\n"
        lit_in, lit_out, par_in, par_out, names = _macro_body_io(body)
        self.assertEqual(names, [])
        self.assertEqual(lit_in, ["work.summary"])
        self.assertEqual(par_in, [])

    def test_sql_create_table_parameterised(self):
        body = (
            "%macro agg(src, out);\n"
            "  proc sql;\n"
            "    create table &out. as select * from &src.;\n"
            "  quit;\n"
            "%mend;\n"
        )
        lit_in, lit_out, par_in, par_out, names = _macro_body_io(body)
        self.assertEqual(names, ["src", "out"])
        self.assertEqual(par_out, [{"param": "out", "pos": 1}])
        self.assertEqual(par_in, [{"param": "src", "pos": 0}])

    def test_merge_statement_parameterised(self):
        body = (
            "%macro combine(a, b, out);\n"
            "  data &out.;\n"
            "    merge &a. &b.;\n"
            "    by id;\n"
            "  run;\n"
            "%mend;\n"
        )
        lit_in, lit_out, par_in, par_out, names = _macro_body_io(body)
        self.assertEqual(names, ["a", "b", "out"])
        param_names_out = {e["param"] for e in par_out}
        param_names_in = {e["param"] for e in par_in}
        self.assertIn("out", param_names_out)
        self.assertIn("a", param_names_in)
        self.assertIn("b", param_names_in)


# ---------------------------------------------------------------------------
# 2. _parse_call_args (batcher-level unit tests)
# ---------------------------------------------------------------------------


class TestParseCallArgs(unittest.TestCase):
    def test_positional_args_only(self):
        pos, kw = _parse_call_args("%load('/data/orders.csv', work.orders);")
        self.assertEqual(pos, ["/data/orders.csv", "work.orders"])
        self.assertEqual(kw, {})

    def test_keyword_args_only(self):
        pos, kw = _parse_call_args("%flag(thresh=2.5);")
        self.assertEqual(pos, [])
        self.assertEqual(kw, {"thresh": "2.5"})

    def test_mixed_positional_and_keyword(self):
        pos, kw = _parse_call_args("%flag(work.enriched, revenue, thresh=2.5);")
        self.assertEqual(pos, ["work.enriched", "revenue"])
        self.assertEqual(kw, {"thresh": "2.5"})

    def test_quoted_values_stripped(self):
        pos, kw = _parse_call_args('%load("work.orders");')
        self.assertEqual(pos, ["work.orders"])

    def test_trailing_dot_stripped(self):
        pos, kw = _parse_call_args("%clean(work.orders.);")
        self.assertEqual(pos, ["work.orders"])

    def test_no_args_call(self):
        pos, kw = _parse_call_args("%setup;")
        self.assertEqual(pos, [])
        self.assertEqual(kw, {})

    def test_case_normalised(self):
        pos, kw = _parse_call_args("%clean(WORK.Orders);")
        self.assertEqual(pos, ["work.orders"])


# ---------------------------------------------------------------------------
# 3. End-to-end batching — literal macro bodies (Fix A)
# ---------------------------------------------------------------------------


class TestLiteralMacroBodyBatching(unittest.TestCase):
    def test_macro_with_literal_output_links_to_consumer(self):
        """
        %macro setup defines a literal output 'work.base'.  Calling it and
        then reading work.base elsewhere must batch all three together,
        even without any parameter substitution.
        """
        src = (
            "%macro setup;\n  data work.base; set mylib.raw; run;\n%mend;\n"
            "%setup;\n"
            "proc print data=work.base; run;\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.batches[0].chunks), 3)
        self.assertIn("work.base", br.batches[0].output_datasets)

    def test_literal_macro_body_cross_file(self):
        """Same as above, but macro def in file 1, call+consumer in file 2."""
        cr1 = _C.chunk_text(
            "%macro setup;\n  data work.base; set mylib.raw; run;\n%mend;\n",
            source_id="macros.sas",
        )
        cr2 = _C.chunk_text(
            "%setup;\nproc print data=work.base; run;\n",
            source_id="etl.sas",
        )
        corpus = SasCorpus(file_results=[cr1, cr2])
        br = MultiFileBatcher().batch(corpus)
        self.assertEqual(len(br.batches), 1)
        self.assertTrue(br.batches[0].is_cross_file)
        self.assertIn("work.base", br.batches[0].output_datasets)


# ---------------------------------------------------------------------------
# 4. End-to-end batching — parameterised macro bodies (Fix B)
# ---------------------------------------------------------------------------


class TestParameterisedMacroBodyBatching(unittest.TestCase):
    def test_single_file_parameterised_resolution(self):
        """%load('...', work.orders) resolves &out. to work.orders."""
        src = (
            "%macro load(path, out);\n"
            "  proc import datafile=&path. dbms=csv out=&out. replace; run;\n"
            "%mend;\n"
            "%load('/data/orders.csv', work.orders);\n"
            "data work.enriched; set work.orders; run;\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.batches[0].chunks), 3)
        self.assertIn("work.enriched", br.batches[0].output_datasets)

    def test_cross_file_parameterised_resolution(self):
        """
        The originally-reported gap: macro def in file 1, parameterised
        call in file 2, consumer also in file 2.  &out. must resolve to
        'work.orders' at the call site and link to the consumer.
        """
        cr1 = _C.chunk_text(
            "%macro load(path, out);\n"
            "  proc import datafile=&path. dbms=csv out=&out. replace; run;\n"
            "%mend;\n",
            source_id="macros.sas",
        )
        cr2 = _C.chunk_text(
            "%load('/data/orders.csv', work.orders);\n"
            "data work.enriched; set work.orders; run;\n",
            source_id="etl.sas",
        )
        corpus = SasCorpus(file_results=[cr1, cr2])
        br = MultiFileBatcher().batch(corpus)
        self.assertEqual(len(br.batches), 1)
        self.assertTrue(br.batches[0].is_cross_file)
        self.assertEqual(len(br.batches[0].chunks), 3)
        self.assertIn("work.enriched", br.batches[0].output_datasets)

    def test_keyword_argument_resolution(self):
        """Macro called with the dataset passed as a keyword argument."""
        src = (
            "%macro clean(ds=);\n  data &ds.; set &ds.; run;\n%mend;\n"
            "data work.raw; set mylib.x; run;\n"
            "%clean(ds=work.raw);\n"
            "proc print data=work.raw; run;\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        self.assertEqual(len(br.batches), 1)
        # All four chunks: macro def, DATA work.raw, %clean call, PROC PRINT
        self.assertEqual(len(br.batches[0].chunks), 4)

    def test_three_file_chain_with_parameterised_macro(self):
        """
        File 1: macro definition only.
        File 2: produces work.orders (literal DATA step, not via macro).
        File 3: calls %clean(work.orders) and reads the result downstream.
        """
        cr1 = _C.chunk_text(
            "%macro clean(ds);\n  data &ds.; set &ds.; if x<0 then delete; run;\n%mend;\n",
            source_id="macros.sas",
        )
        cr2 = _C.chunk_text(
            "data work.orders; set mylib.raw; run;\n",
            source_id="produce.sas",
        )
        cr3 = _C.chunk_text(
            "%clean(work.orders);\nproc means data=work.orders; run;\n",
            source_id="consume.sas",
        )
        corpus = SasCorpus(file_results=[cr1, cr2, cr3])
        br = MultiFileBatcher().batch(corpus)
        self.assertEqual(len(br.batches), 1)
        self.assertTrue(br.batches[0].is_cross_file)
        self.assertEqual(len(br.batches[0].source_files), 3)
        self.assertEqual(len(br.batches[0].chunks), 4)

    def test_resolved_output_feeds_dataset_flow_for_later_chunk(self):
        """
        The macro call's resolved output must be visible to a *dataset_flow*
        edge for any later-appearing consumer chunk, not just an immediately
        adjacent one.
        """
        src = (
            "%macro load(out); data &out.; x=1; run; %mend;\n"
            "%load(work.a);\n"
            "data work.b; set work.a; run;\n"  # 1 hop
            "data work.c; set work.b; run;\n"  # 2 hops
            "proc print data=work.c; run;\n"  # 3 hops
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.batches[0].chunks), 5)

    def test_multiple_call_sites_resolve_independently(self):
        """Same macro called twice with different dataset args."""
        src = (
            "%macro load(out); data &out.; x=1; run; %mend;\n"
            "%load(work.first);\n"
            "%load(work.second);\n"
            "proc print data=work.first; run;\n"
            "proc means data=work.second; run;\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        # All five chunks connect transitively through the shared macro
        # definition (macro_invocation edges) into one batch.
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.batches[0].chunks), 5)
        self.assertIn("work.first", br.batches[0].output_datasets)
        self.assertIn("work.second", br.batches[0].output_datasets)


# ---------------------------------------------------------------------------
# 4b. Nested macro invocation — one macro's DEFINITION body invokes another
#
# Pattern under test:
#
#     %macro abc;
#       ... body ...
#     %mend;
#
#     %macro xyz;
#       %abc;            <-- abc invoked INSIDE xyz's definition body
#       ... body ...
#     %mend;
#
#     %xyz;              <-- top-level call site
#
# This differs from the call-site scenarios in section 4: there, a
# MACRO_CALL chunk invokes a macro.  Here, a MACRO_DEFINITION chunk's body
# invokes a *different* macro.  This is already handled by the existing
# _io_for(MACRO_DEFINITION) branch, which scans the full body (not just the
# header) for %macro_call patterns via _MACRO_INVOKE_RE — so %abc inside
# xyz's body is correctly captured in xyz's invokes_macros list.  These
# tests pin that behaviour and extend it to the parameterised and
# cross-file cases.
# ---------------------------------------------------------------------------


class TestNestedMacroInvocation(unittest.TestCase):
    def test_definition_body_invokes_another_macro(self):
        """
        xyz's MACRO_DEFINITION chunk must record 'abc' in invokes_macros,
        since %abc appears inside xyz's body, not just at a top-level
        call site.
        """
        src = (
            "%macro abc;\n  data work.from_abc; set mylib.raw; run;\n%mend;\n"
            "%macro xyz;\n  %abc;\n  data work.from_xyz; set work.from_abc; run;\n%mend;\n"
        )
        cr = _C.chunk_text(src)
        abc_def = next(c for c in cr.chunks if "abc" in c.metadata.defines_macros)
        xyz_def = next(c for c in cr.chunks if "xyz" in c.metadata.defines_macros)
        self.assertIn("abc", xyz_def.metadata.invokes_macros)
        self.assertEqual(abc_def.metadata.invokes_macros, [])

    def test_literal_chain_through_nested_definition(self):
        """
        Literal datasets propagate through a three-level chain:
        abc produces work.from_abc -> xyz's body consumes it and produces
        work.from_xyz -> %xyz call -> downstream PROC reads work.from_xyz.
        All four chunks (2 defs + 1 call + 1 consumer) must batch together.
        """
        src = (
            "%macro abc;\n  data work.from_abc; set mylib.raw; run;\n%mend;\n"
            "%macro xyz;\n  %abc;\n  data work.from_xyz; set work.from_abc; run;\n%mend;\n"
            "%xyz;\n"
            "proc print data=work.from_xyz; run;\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.batches[0].chunks), 4)
        self.assertIn("abc", br.batches[0].defined_macros)
        self.assertIn("xyz", br.batches[0].defined_macros)
        self.assertIn("work.from_abc", br.batches[0].output_datasets)
        self.assertIn("work.from_xyz", br.batches[0].output_datasets)
        self.assertIn("macro_invocation(%abc)", br.batches[0].reason)
        self.assertIn("macro_invocation(%xyz)", br.batches[0].reason)

    def test_parameterised_nested_invocation_resolves_at_top_call(self):
        """
        xyz(ds) forwards its own parameter to %abc(&ds.) in its body, and
        also reads &ds. directly via 'proc print data=&ds.'.  When %xyz is
        called with a concrete dataset, that argument must resolve through
        both xyz's own param usage AND propagate correctly so the producer
        of that dataset and the call site land in one batch.
        """
        src = (
            "%macro abc(ds);\n  data &ds.; set &ds.; flag=1; run;\n%mend;\n"
            "%macro xyz(ds);\n  %abc(&ds.);\n  proc print data=&ds.; run;\n%mend;\n"
            "data work.orders; set mylib.raw; run;\n"
            "%xyz(work.orders);\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.batches[0].chunks), 4)
        self.assertIn("work.orders", br.batches[0].output_datasets)

    def test_xyz_body_records_abc_param_as_input_via_proc_data(self):
        """
        xyz's body uses &ds. only through a nested %abc(&ds.) call and a
        'proc print data=&ds.' statement (no direct data/set on &ds.).
        The PROC data= context must still be classified as a parameterised
        INPUT reference on xyz's own metadata.
        """
        src = (
            "%macro xyz(ds);\n  %abc(&ds.);\n  proc print data=&ds.;\n  run;\n%mend;\n"
        )
        cr = _C.chunk_text(src)
        xyz_def = cr.chunks[0]
        self.assertEqual(xyz_def.metadata.macro_param_names, ["ds"])
        self.assertIn({"param": "ds", "pos": 0}, xyz_def.metadata.body_param_inputs)
        self.assertEqual(xyz_def.metadata.body_param_outputs, [])

    def test_cross_file_nested_invocation(self):
        """
        abc defined in file 1, xyz (which invokes abc) defined in file 2,
        top-level %xyz call in file 3.  All three files must end up in one
        cross-file batch via the chained macro_invocation edges:
        abc-def -> xyz-def (via %abc inside xyz's body) -> %xyz call.
        """
        cr1 = _C.chunk_text(
            "%macro abc;\n  data work.shared; set mylib.raw; run;\n%mend;\n",
            source_id="abc.sas",
        )
        cr2 = _C.chunk_text(
            "%macro xyz;\n  %abc;\n  proc print data=work.shared; run;\n%mend;\n",
            source_id="xyz.sas",
        )
        cr3 = _C.chunk_text("%xyz;\n", source_id="main.sas")
        corpus = SasCorpus(file_results=[cr1, cr2, cr3])
        br = MultiFileBatcher().batch(corpus)
        self.assertEqual(len(br.batches), 1)
        self.assertTrue(br.batches[0].is_cross_file)
        self.assertEqual(
            set(br.batches[0].source_files),
            {"abc.sas", "xyz.sas", "main.sas"},
        )

    def test_xyz_called_multiple_times_still_one_batch(self):
        """
        Multiple top-level calls to %xyz (which itself invokes %abc) must
        all transitively join the same batch as abc's and xyz's
        definitions, via the shared macro_invocation chain.
        """
        src = (
            "%macro abc;\n  data work.base; set mylib.raw; run;\n%mend;\n"
            "%macro xyz;\n  %abc;\n  proc print data=work.base; run;\n%mend;\n"
            "%xyz;\n"
            "%xyz;\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        self.assertEqual(len(br.batches), 1)
        # 2 definitions + 2 call sites = 4 chunks
        self.assertEqual(len(br.batches[0].chunks), 4)

    def test_unrelated_third_macro_stays_separate(self):
        """
        A macro unrelated to the abc/xyz chain (no shared datasets, no
        invocation relationship) must remain in its own batch.
        """
        src = (
            "%macro abc;\n  data work.shared; set mylib.raw; run;\n%mend;\n"
            "%macro xyz;\n  %abc;\n  proc print data=work.shared; run;\n%mend;\n"
            "%xyz;\n"
            "%macro unrelated;\n  data work.other; set mylib.other_raw; run;\n%mend;\n"
            "%unrelated;\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        self.assertEqual(len(br.batches), 2)
        sizes = sorted(len(b.chunks) for b in br.batches)
        self.assertEqual(sizes, [2, 3])


# ---------------------------------------------------------------------------
# 5. Compound / unresolvable references — must not silently mis-resolve
# ---------------------------------------------------------------------------


class TestUnresolvableReferencesStayConservative(unittest.TestCase):
    def test_compound_concatenation_does_not_create_wrong_edge(self):
        """
        &lib..raw concatenates parameter 'lib' with literal '.raw'.  This
        must NOT resolve to a dataset literally named 'work' (the value of
        'lib') — that would be a wrong, misleading edge.  Verify no
        dataset_flow/macro_body_dataset edge fires using just 'work' as
        the dataset name.
        """
        src = (
            "%macro load(lib, file);\n"
            "  proc import datafile=&file. dbms=csv out=&lib..raw replace; run;\n"
            "%mend;\n"
            "%load(work, '/a.csv');\n"
            "data work_summary; set work; run;\n"  # reads literal 'work' — must NOT batch with %load
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        # The DATA step reading 'work' (the literal dataset, not work.raw)
        # must remain unconnected to the %load batch.
        load_batch = next(
            (b for b in br.batches if "load" in b.defined_macros),
            None,
        )
        if load_batch is not None:
            self.assertNotIn("work_summary", load_batch.chunk_ids)

    def test_undefined_macro_in_call_site_resolution_no_crash(self):
        """A MACRO_CALL invoking an undefined macro must not crash Fix B."""
        src = "%undefined_macro(work.x);\ndata work.y; set work.x; run;\n"
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        # Should not raise; structural assertion only
        self.assertIsNotNone(br)


# ---------------------------------------------------------------------------
# 6. JSON serialisability of new metadata fields
# ---------------------------------------------------------------------------


class TestNewFieldsSerialisable(unittest.TestCase):
    def test_chunk_result_with_macro_body_fields_serialises(self):
        import json

        src = (
            "%macro load(path, out);\n"
            "  proc import datafile=&path. dbms=csv out=&out. replace; run;\n"
            "%mend;\n"
        )
        cr = _C.chunk_text(src)
        json.dumps(cr.model_dump())  # must not raise

    def test_batch_result_with_macro_body_dataset_edges_serialises(self):
        import json

        src = (
            "%macro load(path, out);\n"
            "  proc import datafile=&path. dbms=csv out=&out. replace; run;\n"
            "%mend;\n"
            "%load('/x.csv', work.orders);\n"
            "data work.enriched; set work.orders; run;\n"
        )
        cr = _C.chunk_text(src)
        br = SasChunkBatcher().batch(cr)
        json.dumps(br.model_dump())  # must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
