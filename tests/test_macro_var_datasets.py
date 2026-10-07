"""
test_macro_var_datasets.py — libraries and datasets named by macro variables.

Production SAS routinely spells a library or a table through a macro variable
rather than writing it out::

    %let lname  = xwrk;
    %let table1 = &suf;
    data &table1;
      set &lname..&table1;
    run;

Covers the two halves of recognising those names:

- **resolution** — ``chunker.macro_vars`` builds the ``%LET`` symbol table and
  expands ``&name`` / ``&name.`` references, including chains and the
  double-dot form where one dot ends the reference and the other separates
  libref from member;
- **reporting** — ``chunker.metadata.resolve_macro_var_refs`` feeds the results
  back into ``referenced_datasets`` / ``referenced_librefs`` / the directed I/O
  lists, canonicalising what resolved and keeping what did not **exactly as
  written**, so a dependency that only exists at run time is reported rather
  than dropped or guessed at.

Run:  python -m pytest tests/test_macro_var_datasets.py -v
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chunker import SasChunkBatcher, SasCorpus, SasSemanticChunker
from chunker.batcher import MultiFileBatcher
from chunker.macro_vars import (
    is_dataset_shaped,
    let_values,
    resolve_refs,
    strip_quotes,
)
from chunker.metadata import _canon_ds
from chunker.models import SasChunkKind

# ── helpers ────────────────────────────────────────────────────────────────


def _chunk(source: str, source_id: str = "test.sas"):
    return SasSemanticChunker(min_words=1, max_words=9_999).chunk_text(
        source, source_id=source_id
    )


def _chunk_and_batch(source: str, **kwargs):
    result = _chunk(source)
    return result, SasChunkBatcher(**kwargs).batch(result)


def _kind(result, kind: SasChunkKind):
    """Every chunk of *kind*, in source order."""
    return [c for c in result.chunks if c.kind == kind]


# The example from the feature request, verbatim.
MANUAL_EXAMPLE = """%LET LNAME = XWRK;
%LET SUF = BATCH_MED;

%LET TABLE_DEMOGR = DATACIA.MEMBER_DEMOGRAPHIC;

%LET TABLE_REG_EXCL_spd = &LIB_OUT_spd..CIA_HSO_EXCL;

%LET TABLE1 = &SUF;


data &TABLE1;
  set &TABLE_REG_EXCL_spd;
run;

data datacia.dummy;
  set &LNAME..&TABLE1;
run;
"""


# ── 1. macro_vars: the pure grammar ────────────────────────────────────────


class TestLetValues(unittest.TestCase):
    def test_name_shaped_values_kept_lowercased(self):
        self.assertEqual(
            let_values("%LET LNAME = XWRK;\n%let Tbl = Lib.Member;"),
            {"lname": "xwrk", "tbl": "lib.member"},
        )

    def test_quoted_value_unquoted(self):
        self.assertEqual(let_values("%let ds = 'lib.member';"), {"ds": "lib.member"})

    def test_non_name_values_are_recorded_as_unknown(self):
        # Any string at all can be a %LET value; only one that could be a name,
        # or part of one, is kept. The rest map to "" — unknown from here on —
        # rather than vanishing, so an earlier name-shaped value of the same
        # variable cannot keep answering for it.
        src = (
            "%let where = age > 30 and sex = 'M';\n"
            "%let path = /sasdata3/dataetl/in.csv;\n"
            "%let lib = prod;\n"
        )
        self.assertEqual(let_values(src), {"where": "", "path": "", "lib": "prod"})

    def test_numeric_value_kept_as_a_name_fragment(self):
        # Not a name on its own, but half of one: &&ds&i, &lib..sales&yr.
        self.assertEqual(
            let_values("%let i = 1;\n%let yr = 2024;"), {"i": "1", "yr": "2024"}
        )

    def test_computed_target_has_no_key_to_store_under(self):
        self.assertEqual(let_values("%let &&outer&i = work.x;"), {})

    def test_last_assignment_in_the_text_wins(self):
        self.assertEqual(
            let_values("%let t = work.first;\n%let t = work.second;"),
            {"t": "work.second"},
        )

    def test_unterminated_let_still_assigns(self):
        # An oversized split can cut a chunk before the semicolon.
        self.assertEqual(let_values("%let ds = work.orders"), {"ds": "work.orders"})


class TestStripQuotes(unittest.TestCase):
    def test_matching_quotes_removed(self):
        self.assertEqual(strip_quotes("  'lib.member' "), "lib.member")
        self.assertEqual(strip_quotes('"lib.member"'), "lib.member")

    def test_unmatched_and_bare_left_alone(self):
        self.assertEqual(strip_quotes("'lib.member"), "'lib.member")
        self.assertEqual(strip_quotes("lib.member"), "lib.member")


class TestResolveRefs(unittest.TestCase):
    def test_single_reference(self):
        self.assertEqual(resolve_refs("&ds", {"ds": "work.orders"}), "work.orders")

    def test_trailing_delimiter_dot_is_consumed(self):
        self.assertEqual(resolve_refs("&ds.", {"ds": "work.orders"}), "work.orders")

    def test_double_dot_separates_libref_from_member(self):
        """&lname..&table1 — one dot ends the reference, one is the separator."""
        self.assertEqual(
            resolve_refs("&lname..&table1", {"lname": "xwrk", "table1": "batch_med"}),
            "xwrk.batch_med",
        )

    def test_chain_resolves_through_intermediate_variables(self):
        self.assertEqual(
            resolve_refs("&table1", {"table1": "&suf", "suf": "batch_med"}),
            "batch_med",
        )

    def test_unknown_reference_kept_byte_for_byte(self):
        raw = "&lib_out_spd..cia_hso_excl"
        self.assertEqual(resolve_refs(raw, {"other": "x"}), raw)

    def test_partial_resolution_keeps_the_unresolved_half_verbatim(self):
        self.assertEqual(
            resolve_refs("&lib..&tbl", {"tbl": "orders"}), "&lib..orders"
        )

    def test_self_reference_terminates(self):
        # %let x = &x.y; is a legitimate append idiom; expanding a name twice
        # would grow the string without end.
        self.assertEqual(resolve_refs("&x.y", {"x": "&x.y"}), "&x.yy")

    def test_mutual_reference_terminates(self):
        self.assertEqual(resolve_refs("&a", {"a": "&b", "b": "&a"}), "&a")

    def test_indirect_reference_is_rescanned(self):
        # &&a&i → &a1 → work.one, the "loop over a numbered list" idiom.
        self.assertEqual(
            resolve_refs("&&a&i", {"a1": "work.one", "i": "1"}), "work.one"
        )

    def test_indirect_reference_the_corpus_cannot_resolve_stays_verbatim(self):
        # A half-rescanned &&name is neither the source nor a name.
        self.assertEqual(resolve_refs("&&a&i", {"i": "1"}), "&&a&i")
        self.assertEqual(resolve_refs("&&a&i", {"a1": "work.one"}), "&&a&i")

    def test_empty_table_and_plain_text_short_circuit(self):
        self.assertEqual(resolve_refs("&ds", {}), "&ds")
        self.assertEqual(resolve_refs("work.orders", {"ds": "x"}), "work.orders")


class TestIsDatasetShaped(unittest.TestCase):
    def test_two_level_names(self):
        self.assertTrue(is_dataset_shaped("datacia.member_demographic"))
        self.assertTrue(is_dataset_shaped("lib.&member"))
        self.assertTrue(is_dataset_shaped("&lib_out_spd..cia_hso_excl"))

    def test_one_level_values_are_just_strings(self):
        self.assertFalse(is_dataset_shaped("batch_med"))
        self.assertFalse(is_dataset_shaped("xwrk"))
        self.assertFalse(is_dataset_shaped("&suf"))

    def test_trailing_dot_is_not_a_member(self):
        self.assertFalse(is_dataset_shaped("mydate."))


# ── 2. The example from the feature request ────────────────────────────────


class TestManualExample(unittest.TestCase):
    """End-to-end on the source the request was filed with."""

    def setUp(self):
        self.result = _chunk(MANUAL_EXAMPLE)
        self.steps = _kind(self.result, SasChunkKind.DATA_STEP)

    def test_chained_reference_resolves_to_its_final_value(self):
        # data &TABLE1;  →  &TABLE1 → &SUF → BATCH_MED → work.batch_med
        first = self.steps[0]
        self.assertEqual(first.metadata.output_datasets, ["work.batch_med"])
        self.assertEqual(first.title, "DATA batch_med")

    def test_unresolved_reference_is_reported_as_written(self):
        # set &TABLE_REG_EXCL_spd;  →  &LIB_OUT_spd..CIA_HSO_EXCL, whose libref
        # no %LET in the corpus supplies.
        first = self.steps[0]
        self.assertEqual(
            first.metadata.input_datasets, ["&lib_out_spd..cia_hso_excl"]
        )
        self.assertIn(
            "&lib_out_spd..cia_hso_excl", first.metadata.referenced_datasets
        )
        self.assertEqual(
            first.metadata.unresolved_dataset_refs, ["&lib_out_spd..cia_hso_excl"]
        )

    def test_unresolved_libref_is_reported_as_written(self):
        self.assertIn("&lib_out_spd", self.steps[0].metadata.referenced_librefs)

    def test_two_references_build_one_two_level_name(self):
        # set &LNAME..&TABLE1;  →  xwrk.batch_med
        second = self.steps[1]
        self.assertEqual(second.metadata.input_datasets, ["xwrk.batch_med"])
        self.assertEqual(second.metadata.referenced_librefs, ["datacia", "xwrk"])

    def test_let_value_that_names_a_table_is_a_referenced_dataset(self):
        # %LET TABLE_DEMOGR = DATACIA.MEMBER_DEMOGRAPHIC; names a library on
        # sight, even though nothing in this program reads it.
        let_chunk = next(
            c
            for c in self.result.chunks
            if "TABLE_DEMOGR" in c.text and c.kind == SasChunkKind.GLOBAL_STATEMENT
        )
        self.assertEqual(
            let_chunk.metadata.referenced_datasets, ["datacia.member_demographic"]
        )
        self.assertEqual(let_chunk.metadata.referenced_librefs, ["datacia"])

    def test_a_let_is_provenance_never_io(self):
        """A %LET reads and writes nothing — the step that uses it does."""
        for chunk in _kind(self.result, SasChunkKind.GLOBAL_STATEMENT):
            self.assertEqual(chunk.metadata.input_datasets, [])
            self.assertEqual(chunk.metadata.output_datasets, [])

    def test_no_dataset_is_invented_from_a_reference(self):
        """The old scan read `data &TABLE1;` as the identifier `table1`."""
        for chunk in self.result.chunks:
            self.assertNotIn("work.table1", chunk.metadata.referenced_datasets)


# ── 3. Every dataset position, not just DATA/SET ───────────────────────────


class TestDatasetPositions(unittest.TestCase):
    def test_proc_data_and_out_options(self):
        src = (
            "%let lib = prod;\n"
            "%let tbl = orders;\n"
            "proc sort data=&lib..&tbl out=&lib..sorted; by id; run;\n"
        )
        chunk = _kind(_chunk(src), SasChunkKind.PROC_STEP)[0]
        self.assertEqual(chunk.metadata.input_datasets, ["prod.orders"])
        self.assertEqual(chunk.metadata.output_datasets, ["prod.sorted"])
        self.assertEqual(chunk.metadata.referenced_librefs, ["prod"])

    def test_sql_create_from_and_join(self):
        src = (
            "%let lib = prod;\n"
            "proc sql;\n"
            "  create table &lib..agg as\n"
            "    select * from &lib..sorted join work.dim on 1=1;\n"
            "quit;\n"
        )
        chunk = _kind(_chunk(src), SasChunkKind.PROC_STEP)[0]
        self.assertEqual(chunk.metadata.output_datasets, ["prod.agg"])
        self.assertEqual(
            sorted(chunk.metadata.input_datasets), ["prod.sorted", "work.dim"]
        )

    def test_merge_with_mixed_literal_and_reference(self):
        src = (
            "%let b = work.right;\n"
            "data work.both;\n"
            "  merge work.left &b;\n"
            "  by id;\n"
            "run;\n"
        )
        chunk = _kind(_chunk(src), SasChunkKind.DATA_STEP)[0]
        self.assertEqual(chunk.metadata.input_datasets, ["work.left", "work.right"])

    def test_multi_dataset_data_header(self):
        src = "%let a = work.one;\n%let b = two;\ndata &a &b; run;\n"
        chunk = _kind(_chunk(src), SasChunkKind.DATA_STEP)[0]
        self.assertEqual(chunk.metadata.output_datasets, ["work.one", "work.two"])

    def test_reference_with_dataset_options(self):
        src = "%let src = work.big;\ndata work.small; set &src(keep=x y); run;\n"
        chunk = _kind(_chunk(src), SasChunkKind.DATA_STEP)[0]
        self.assertEqual(chunk.metadata.input_datasets, ["work.big"])
        for token in ("keep", "x", "y"):
            self.assertNotIn(token, chunk.metadata.input_datasets)

    def test_libname_whose_libref_is_a_macro_variable(self):
        src = "%let l = mylib;\nlibname &l '/data/x';\ndata &l..out; set &l..in; run;\n"
        result = _chunk(src)
        libname = next(c for c in result.chunks if "libname" in c.text.lower())
        self.assertEqual(libname.metadata.defines_librefs, ["mylib"])
        # The statement's directory is recorded too — while the libref had to
        # be a bare identifier, this matched nothing and the path was lost.
        self.assertEqual([r.path for r in libname.metadata.external_refs], ["/data/x"])
        self.assertEqual(libname.metadata.external_refs[0].binds, "mylib")

    def test_engine_libname_whose_libref_is_a_macro_variable(self):
        src = "%let l = edw;\nlibname &l oracle path=EDWPRO schema=fr_dm;\n"
        chunk = next(c for c in _chunk(src).chunks if "oracle" in c.text)
        self.assertEqual([r.binds for r in chunk.metadata.engine_refs], ["edw"])

    def test_user_library_assigned_through_a_macro_variable(self):
        """One-level names stop resolving to WORK just the same when the
        libref arrives through a macro variable."""
        src = "%let u = user;\nlibname &u '/u/perm';\ndata q1; x=1; run;\n"
        codes = [d.code for d in _chunk(src).diagnostics]
        self.assertIn("USER_LIBRARY_ASSIGNED", codes)


# ── 4. Canonicalisation of resolved and unresolved names ───────────────────


class TestCanonicalisation(unittest.TestCase):
    def test_unresolved_name_is_never_work_qualified(self):
        """&suf may well resolve to a two-level name — asserting work.&suf
        would claim a library the source never named."""
        self.assertEqual(_canon_ds("&suf"), "&suf")
        self.assertEqual(_canon_ds("&lib..member"), "&lib..member")

    def test_resolved_one_level_name_is_work_qualified(self):
        src = "%let t = staging;\ndata &t; x=1; run;\nproc print data=work.staging; run;\n"
        _, br = _chunk_and_batch(src)
        # Both spellings land in one namespace, so the step and the PROC batch.
        self.assertEqual(len(br.batches), 1)
        self.assertIn("work.staging", br.batches[0].output_datasets)

    def test_resolved_name_reaches_the_databricks_mapping(self):
        src = "%let lib = sales;\ndata &lib..orders; x=1; run;\n"
        _, br = _chunk_and_batch(
            src, databricks_mapping={"sales": "prod.sales_schema"}
        )
        self.assertIn(
            "prod.sales_schema.orders", br.all_ordered_items[0].output_datasets
        )


# ── 5. Scope: macro parameters, macro bodies, source order ─────────────────


class TestScope(unittest.TestCase):
    def test_macro_parameter_shadows_a_corpus_let(self):
        """&ds inside %macro m(ds) is supplied by the call site, so a
        corpus-level `%let ds = ...` must not be read as its value."""
        src = (
            "%let ds = work.global_one;\n"
            "%macro m(ds);\n"
            "  data work.out; set &ds; run;\n"
            "%mend;\n"
        )
        macro = _kind(_chunk(src), SasChunkKind.MACRO_DEFINITION)[0]
        self.assertEqual(macro.metadata.body_param_inputs, [{"param": "ds", "pos": 0}])
        self.assertNotIn("work.global_one", macro.metadata.body_literal_inputs)

    def test_non_parameter_reference_in_a_body_resolves(self):
        src = (
            "%let ref_lib = reporting;\n"
            "%macro m(ds);\n"
            "  data work.out; set &ref_lib..lookup; run;\n"
            "%mend;\n"
        )
        macro = _kind(_chunk(src), SasChunkKind.MACRO_DEFINITION)[0]
        self.assertIn("reporting.lookup", macro.metadata.body_literal_inputs)

    def test_unresolved_body_reference_is_kept_not_dropped(self):
        src = "%macro m;\n  data work.out; set &mystery_ds; run;\n%mend;\n"
        macro = _kind(_chunk(src), SasChunkKind.MACRO_DEFINITION)[0]
        self.assertEqual(macro.metadata.body_literal_inputs, ["&mystery_ds"])
        self.assertIn("&mystery_ds", macro.metadata.unresolved_dataset_refs)

    def test_a_name_built_from_parameters_is_never_fabricated(self):
        """`&lib..&prefix._&suffix.` has no name until the call is made."""
        src = (
            "%macro snapshot(lib, prefix, suffix);\n"
            "  data &lib..&prefix._&suffix.; set &lib..&prefix.; run;\n"
            "%mend;\n"
        )
        macro = _kind(_chunk(src), SasChunkKind.MACRO_DEFINITION)[0]
        self.assertEqual(macro.metadata.body_literal_outputs, [])

    def test_a_let_inside_a_macro_body_stays_local_to_it(self):
        """Whether it executes depends on a call site, so it must not give a
        value to a reference in a later, unrelated step."""
        src = (
            "%macro setup;\n"
            "  %let target = work.inside;\n"
            "%mend;\n"
            "data &target; run;\n"
        )
        step = _kind(_chunk(src), SasChunkKind.DATA_STEP)[0]
        self.assertEqual(step.metadata.output_datasets, ["&target"])

    def test_a_reference_before_its_let_is_not_resolved(self):
        src = "data &later; run;\n%let later = work.after;\n"
        step = _kind(_chunk(src), SasChunkKind.DATA_STEP)[0]
        self.assertEqual(step.metadata.output_datasets, ["&later"])

    def test_reassignment_applies_from_where_it_stands(self):
        src = (
            "%let t = work.first;\n"
            "data &t; run;\n"
            "%let t = work.second;\n"
            "data &t; run;\n"
        )
        steps = _kind(_chunk(src), SasChunkKind.DATA_STEP)
        self.assertEqual(steps[0].metadata.output_datasets, ["work.first"])
        self.assertEqual(steps[1].metadata.output_datasets, ["work.second"])

    def test_let_value_resolves_where_it_stands(self):
        """`%let x = &x.b;` appends to x's current value, as SAS does."""
        src = "%let x = work.a;\n%let x = &x.b;\ndata &x; run;\n"
        step = _kind(_chunk(src), SasChunkKind.DATA_STEP)[0]
        self.assertEqual(step.metadata.output_datasets, ["work.ab"])

    def test_several_lets_in_one_chunk_resolve_in_order(self):
        src = "%let lib = prod;\n%let tbl = &lib..orders;\ndata work.c; set &tbl; run;\n"
        step = _kind(_chunk(src), SasChunkKind.DATA_STEP)[0]
        self.assertEqual(step.metadata.input_datasets, ["prod.orders"])

    def test_indirect_reference_end_to_end(self):
        src = (
            "%let ds1 = prod.january;\n"
            "%let i = 1;\n"
            "data work.out; set &&ds&i; run;\n"
        )
        step = _kind(_chunk(src), SasChunkKind.DATA_STEP)[0]
        self.assertEqual(step.metadata.input_datasets, ["prod.january"])

    def test_unresolvable_indirect_reference_end_to_end(self):
        src = "%let i = 1;\ndata work.out; set &&ds&i; run;\n"
        step = _kind(_chunk(src), SasChunkKind.DATA_STEP)[0]
        self.assertEqual(step.metadata.input_datasets, ["&&ds&i"])


# ── 6. Batching on resolved names ──────────────────────────────────────────


class TestBatching(unittest.TestCase):
    def test_producer_and_consumer_named_by_the_same_variable_batch(self):
        src = (
            "%let stage = work.staged;\n"
            "data &stage; x=1; run;\n"
            "proc means data=&stage; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(br.batches[0].output_datasets, ["work.staged"])

    def test_resolved_libref_is_reported_as_a_batch_requirement(self):
        src = "%let lib = extlib;\ndata work.out; set &lib..source; run;\n"
        _, br = _chunk_and_batch(src)
        item = br.all_ordered_items[0]
        self.assertIn("extlib", item.required_librefs)

    def test_unresolved_libref_is_reported_as_a_batch_requirement(self):
        """The batch does depend on a library — saying `&lib_out` beats
        reporting none at all."""
        src = "data work.out; set &lib_out..source; run;\n"
        _, br = _chunk_and_batch(src)
        item = br.all_ordered_items[0]
        self.assertEqual(item.metadata.referenced_librefs, ["&lib_out", "work"])

    def test_batch_surfaces_its_unresolved_references(self):
        src = (
            "%let stage = work.staged;\n"
            "data &stage; set &unknown_src; run;\n"
            "proc means data=&stage; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(br.batches[0].unresolved_dataset_refs, ["&unknown_src"])

    def test_a_let_in_one_file_resolves_a_name_in_another(self):
        setup = _chunk("%let lib_out = prodlib;\n", source_id="setup.sas")
        job = _chunk(
            "data &lib_out..result; set work.src; run;\n", source_id="job.sas"
        )
        br = MultiFileBatcher().batch(SasCorpus(file_results=[setup, job]))
        item = br.all_ordered_items[0]
        self.assertIn("prodlib.result", item.output_datasets)
        self.assertIn("prodlib", item.required_librefs)

    def test_cross_file_dataset_flow_through_a_shared_variable(self):
        setup = _chunk("%let shared = work.bridge;\n", source_id="setup.sas")
        writer = _chunk("data &shared; x=1; run;\n", source_id="writer.sas")
        reader = _chunk("proc print data=&shared; run;\n", source_id="reader.sas")
        br = MultiFileBatcher().batch(
            SasCorpus(file_results=[setup, writer, reader])
        )
        cross = br.cross_file_batches
        self.assertEqual(len(cross), 1)
        self.assertEqual(
            cross[0].source_files, ["setup.sas", "writer.sas", "reader.sas"]
        )


# ── 7. Metadata plumbing ───────────────────────────────────────────────────


class TestMetadataPlumbing(unittest.TestCase):
    def test_macro_var_values_merge_child_over_parent(self):
        from chunker.metadata import _merge_meta
        from chunker.models import SasChunkMetadata

        merged = _merge_meta(
            SasChunkMetadata(macro_var_values={"a": "work.one", "b": "work.two"}),
            SasChunkMetadata(macro_var_values={"b": "work.three"}),
        )
        self.assertEqual(
            merged.macro_var_values, {"a": "work.one", "b": "work.three"}
        )

    def test_unresolved_refs_view_spans_every_dataset_field(self):
        from chunker.models import SasChunkMetadata

        meta = SasChunkMetadata(
            referenced_datasets=["&a", "work.plain"],
            input_datasets=["&b"],
            output_datasets=["work.plain"],
            body_literal_inputs=["&c"],
        )
        self.assertEqual(meta.unresolved_dataset_refs, ["&a", "&b", "&c"])

    def test_result_stays_json_serialisable(self):
        import json

        _, br = _chunk_and_batch(MANUAL_EXAMPLE)
        json.dumps(br.model_dump())

    def test_oversized_split_children_inherit_the_regions_values(self):
        """A %LET and the step using it can land in different split children;
        _merge_meta gives every child the whole region's values."""
        body = "".join(f"  x{i} = {i};\n" for i in range(400))
        src = f"%macro big;\n  %let out = work.wide;\n  data &out;\n{body}  run;\n%mend;\n"
        result = SasSemanticChunker(min_words=1, max_words=200).chunk_text(src)
        children = [c for c in result.chunks if c.parent_id]
        self.assertTrue(children)
        for child in children:
            self.assertEqual(child.metadata.macro_var_values, {"out": "work.wide"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
