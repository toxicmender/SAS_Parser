"""
test_batcher.py — unit tests for SasChunkBatcher.

All tests work at the SAS source level: feed source through the chunker
(which populates input/output metadata), then through the batcher, then
assert on the resulting batch/singleton structure.

Run:  python -m pytest tests/test_batcher.py -v
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chunker import SasSemanticChunker
from chunker.batcher import SasChunkBatcher
from chunker.models import SasBatch, SasChunkKind

# ── helpers ────────────────────────────────────────────────────────────────


def _chunk_and_batch(source: str, **batcher_kwargs) -> tuple:
    """Return (SasChunkResult, SasBatchResult) for the given SAS source."""
    chunker = SasSemanticChunker(min_words=1, max_words=9_999)
    result = chunker.chunk_text(source, source_id="test.sas")
    batcher = SasChunkBatcher(**batcher_kwargs)
    batch_result = batcher.batch(result)
    return result, batch_result


def _batch_ids(batch_result) -> list[str]:
    return [b.batch_id for b in batch_result.batches]


def _singleton_ids(batch_result) -> list[str]:
    return [c.chunk_id for c in batch_result.singletons]


def _ordered_chunk_ids(batch_result) -> list[str]:
    """Chunk IDs in original source order across all batches and singletons."""
    ids = []
    for item in batch_result.all_ordered_items:
        if isinstance(item, SasBatch):
            ids.extend(c.chunk_id for c in item.chunks)
        else:
            ids.append(item.chunk_id)
    return ids


# ── 1. No dependencies ─────────────────────────────────────────────────────


class TestNoDependencies(unittest.TestCase):
    def test_independent_steps_all_singletons(self):
        """Steps that share no datasets → every chunk is a singleton."""
        src = "data work.a; x=1; run;\ndata work.b; y=2; run;\ndata work.c; z=3; run;\n"
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 0)
        self.assertEqual(len(br.singletons), 3)

    def test_empty_source(self):
        _, br = _chunk_and_batch("")
        self.assertEqual(len(br.batches), 0)
        self.assertEqual(len(br.singletons), 0)

    def test_single_chunk(self):
        _, br = _chunk_and_batch("data work.a; x=1; run;")
        self.assertEqual(len(br.singletons), 1)
        self.assertEqual(len(br.batches), 0)


# ── 2. Dataset-flow edges ──────────────────────────────────────────────────


class TestDatasetFlow(unittest.TestCase):
    def test_data_step_feeds_proc(self):
        """DATA writes work.clean → PROC reads work.clean → one batch."""
        src = (
            "data work.clean;\n set mylib.raw;\n run;\n"
            "proc means data=work.clean; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.singletons), 0)
        batch = br.batches[0]
        self.assertEqual(len(batch.chunks), 2)
        self.assertIn("work.clean", batch.output_datasets)
        self.assertIn("work.clean", batch.reason)

    def test_proc_sort_feeds_merge(self):
        """PROC SORT (in-place) + MERGE DATA step → one batch."""
        src = (
            "proc sort data=work.orders; by id; run;\n"
            "proc sort data=work.customers; by id; run;\n"
            "data work.combined;\n"
            "  merge work.orders (in=a) work.customers (in=b);\n"
            "  by id;\n"
            "run;\n"
        )
        _, br = _chunk_and_batch(src)
        # All three depend on each other → one batch
        self.assertEqual(len(br.batches), 1)
        batch = br.batches[0]
        self.assertEqual(len(batch.chunks), 3)
        self.assertIn("work.combined", batch.output_datasets)

    def test_hash_dataset_argument_feeds_consumer(self):
        """DATA writes work.lu → a hash constructor loads dataset:'lu' in a
        later step → the dataset: argument is a consume edge → one batch."""
        src = (
            "data work.lu;\n set mylib.raw;\n run;\n"
            "data work.flagged;\n"
            "  set work.claims;\n"
            "  declare hash h1(dataset:\"lu(rename=(cnt = monthly_cnt))\");\n"
            "  h1.definekey('id');\n"
            "  h1.definedone();\n"
            "run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.batches[0].chunks), 2)

    def test_chain_a_b_c_one_batch(self):
        """A→B→C chain (transitive) collapses to one batch."""
        src = (
            "data work.b; set work.a; run;\n"
            "data work.c; set work.b; run;\n"
            "data work.d; set work.c; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.batches[0].chunks), 3)

    def test_fork_one_producer_two_consumers(self):
        """A produces ds; B and C both consume ds → all in one batch."""
        src = (
            "data work.shared; set mylib.raw; x=1; run;\n"
            "proc print data=work.shared; run;\n"
            "proc means data=work.shared; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.batches[0].chunks), 3)

    def test_two_independent_pipelines(self):
        """Two separate A→B chains stay in separate batches."""
        src = (
            "data work.p1; set mylib.src1; run;\n"
            "proc print data=work.p1; run;\n"
            "data work.p2; set mylib.src2; run;\n"
            "proc means data=work.p2; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 2)
        self.assertEqual(len(br.singletons), 0)

    def test_proc_sql_create_and_from(self):
        """PROC SQL CREATE TABLE + earlier DATA step → one batch."""
        src = (
            "data work.orders_clean;\n  set mylib.orders;\nrun;\n"
            "proc sql;\n"
            "  create table work.summary as\n"
            "    select region, sum(revenue) as total\n"
            "    from work.orders_clean\n"
            "    group by region;\n"
            "quit;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        batch = br.batches[0]
        self.assertIn("work.summary", batch.output_datasets)
        self.assertIn("work.orders_clean", batch.output_datasets)

    def test_proc_sql_join_two_inputs(self):
        """PROC SQL joining two tables that each have a prior DATA step."""
        src = (
            "data work.left; set mylib.a; run;\n"
            "data work.right; set mylib.b; run;\n"
            "proc sql;\n"
            "  create table work.joined as\n"
            "    select l.*, r.extra\n"
            "    from work.left as l\n"
            "    join work.right as r on l.id = r.id;\n"
            "quit;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        batch = br.batches[0]
        self.assertEqual(len(batch.chunks), 3)
        self.assertIn("work.joined", batch.output_datasets)

    def test_external_dataset_not_batched(self):
        """A dataset from an external library has no producer → no batch edge."""
        src = (
            "data work.out; set mylib.external; run;\n"
            "proc print data=work.standalone; run;\n"
        )
        _, br = _chunk_and_batch(src)
        # work.out does not feed work.standalone → still two singletons
        self.assertEqual(len(br.batches), 0)
        self.assertEqual(len(br.singletons), 2)

    def test_inplace_sort_batch_with_consumer(self):
        """PROC SORT (no OUT=) writes the same dataset it reads → consumer batched."""
        src = "proc sort data=work.ds; by name; run;\nproc print data=work.ds; run;\n"
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        batch = br.batches[0]
        self.assertEqual(len(batch.chunks), 2)

    def test_external_inputs_reported_correctly(self):
        """External (pre-existing) datasets appear in batch.input_datasets."""
        src = (
            "data work.clean;\n  set mylib.raw;\nrun;\n"
            "proc means data=work.clean; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        batch = br.batches[0]
        # mylib.raw is read by the DATA step but not produced inside the batch
        self.assertIn("mylib.raw", batch.input_datasets)
        # work.clean is produced inside → NOT an external input
        self.assertNotIn("work.clean", batch.input_datasets)

    def test_batch_output_datasets(self):
        """All datasets produced inside a batch are listed in output_datasets."""
        src = "data work.a; set mylib.raw; run;\ndata work.b; set work.a; run;\n"
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        outs = set(br.batches[0].output_datasets)
        self.assertIn("work.a", outs)
        self.assertIn("work.b", outs)

    def test_merge_inputs_all_captured(self):
        """Both merge inputs are tracked; sorted ones that precede merge form a batch."""
        src = (
            "proc sort data=work.left; by id; run;\n"
            "proc sort data=work.right; by id; run;\n"
            "data work.merged;\n"
            "  merge work.left(in=a) work.right(in=b);\n"
            "  by id;\n"
            "run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        batch = br.batches[0]
        self.assertEqual(len(batch.chunks), 3)


# ── 3. Macro-invocation edges ──────────────────────────────────────────────


class TestMacroInvocation(unittest.TestCase):
    def test_macro_def_and_call_batched(self):
        """%MACRO def + %call → one batch."""
        src = (
            "%macro clean(ds);\n"
            "  data &ds.;\n  set &ds.;\n  if x<0 then delete;\n  run;\n"
            "%mend;\n"
            "%clean(work.orders);\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        batch = br.batches[0]
        self.assertIn("clean", batch.defined_macros)
        self.assertEqual(len(batch.chunks), 2)

    def test_macro_def_two_callsites_globals_batch(self):
        """One macro definition + two independent call sites: the definition
        is consumed by two separate components, so it is promoted to the
        global-context batch (emitted first) instead of fusing the two
        call sites into one mega-batch."""
        src = (
            "%macro report(ds);\n"
            "  proc print data=&ds.; run;\n"
            "%mend;\n"
            "%report(work.a);\n"
            "%report(work.b);\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        batch = br.batches[0]
        self.assertTrue(batch.is_global_context)
        self.assertEqual(batch.batch_id, "batch-001")
        self.assertIn("report", batch.defined_macros)
        self.assertEqual(len(batch.chunks), 1)
        # The two call sites stay independent singletons
        self.assertEqual(len(br.singletons), 2)
        for solo in br.singletons:
            self.assertIn("report", solo.metadata.invokes_macros)

    def test_macro_with_dataset_flow(self):
        """Macro def + DATA step that calls it + downstream PROC → one batch."""
        src = (
            "%macro flag(ds, var);\n"
            "  data &ds.; set &ds.;\n"
            "  if &var. < 0 then flag=1; else flag=0;\n"
            "  run;\n"
            "%mend;\n"
            "data work.out; set mylib.raw; run;\n"
            "%flag(work.out, revenue);\n"
            "proc freq data=work.out; tables flag; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        batch = br.batches[0]
        self.assertEqual(len(batch.chunks), 4)

    def test_unresolved_macro_call_is_singleton(self):
        """
        A macro call with no file-local definition AND no argument matching a
        produced dataset stays a singleton.  Using a non-dataset argument
        to ensure no macro_arg_dataset edge fires.
        """
        src = "data work.a; x=1; run;\n%external_macro(some_param);\n"
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 0)
        self.assertEqual(len(br.singletons), 2)

    def test_external_macro_reported_in_required_macros(self):
        """Macro called inside a batch but defined outside → required_macros."""
        src = (
            "%macro helper; data work.x; run; %mend;\n"
            "data work.y; set work.x; run;\n"
            "%helper;\n"  # creates work.x which feeds work.y
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        # helper is defined inside the batch — not required externally
        self.assertNotIn("helper", br.batches[0].required_macros)
        self.assertIn("helper", br.batches[0].defined_macros)

    def test_two_independent_macros_separate_batches(self):
        """Two macro def+call pairs with no shared data stay in separate batches."""
        src = (
            "%macro m1; data work.x; run; %mend;\n"
            "%m1;\n"
            "%macro m2; data work.y; run; %mend;\n"
            "%m2;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 2)

    def test_macro_invoked_inside_data_step(self):
        """%macro call inside a DATA step body is tracked."""
        src = (
            "%macro add_flag; flag=1; %mend;\n"
            "data work.out; set mylib.raw; %add_flag; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.batches[0].chunks), 2)


# ── 4. Transitive closure ──────────────────────────────────────────────────


class TestTransitiveClosure(unittest.TestCase):
    def test_long_chain_single_batch(self):
        """A → B → C → D → E (five-step dataset chain) → one batch."""
        src = (
            "data work.b; set work.a; run;\n"
            "data work.c; set work.b; run;\n"
            "data work.d; set work.c; run;\n"
            "data work.e; set work.d; run;\n"
            "proc print data=work.e; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.batches[0].chunks), 5)

    def test_diamond_dependency(self):
        """Diamond: A produces ds; B and C both read ds; D reads B's and C's outputs."""
        src = (
            "data work.shared; set mylib.raw; run;\n"  # A
            "data work.left;   set work.shared; run;\n"  # B
            "data work.right;  set work.shared; run;\n"  # C
            "data work.joined;\n"  # D
            "  merge work.left work.right;\n"
            "  by id;\n"
            "run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.batches[0].chunks), 4)

    def test_two_disjoint_chains_stay_separate(self):
        src = (
            "data work.x1; set mylib.a; run;\n"
            "data work.x2; set work.x1; run;\n"
            "data work.y1; set mylib.b; run;\n"
            "data work.y2; set work.y1; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 2)
        sizes = sorted(len(b.chunks) for b in br.batches)
        self.assertEqual(sizes, [2, 2])


# ── 5. Source order preservation ──────────────────────────────────────────


class TestSourceOrderPreservation(unittest.TestCase):
    def test_batch_members_in_source_order(self):
        """Chunks inside a batch appear in the same order as in the source."""
        src = (
            "data work.clean;\n set mylib.raw;\nrun;\n"
            "proc sort data=work.clean; by id; run;\n"
            "proc print data=work.clean; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        # Ordering is verified by start_line — IDs carry a file prefix after
        # the multi-file re-stamping pass, so compare by position instead.
        chunk_lines = [c.start_line for c in br.batches[0].chunks]
        self.assertEqual(chunk_lines, sorted(chunk_lines))

    def test_all_ordered_items_ascending_line_numbers(self):
        """all_ordered_items preserves overall source ordering."""
        src = (
            "libname mylib '/data';\n"
            "data work.a; set mylib.raw; run;\n"
            "data work.b; set work.a; run;\n"
            "proc print data=work.standalone; run;\n"
        )
        _, br = _chunk_and_batch(src, include_options_chunks=False)
        items = br.all_ordered_items
        lines = []
        for item in items:
            if isinstance(item, SasBatch):
                lines.append(item.start_line)
            else:
                lines.append(item.start_line)
        self.assertEqual(lines, sorted(lines))

    def test_interleaved_batches_maintain_order(self):
        """
        Two separate batches (non-adjacent chunks) interleaved with a singleton.

        Source layout:
          line 1: data work.a               (batch-1 producer)
          line 2: data work.lone            (singleton)
          line 3: data work.b <- work.a     (batch-1 consumer; non-adjacent)
          line 4: data work.c               (batch-2 producer)
          line 5: proc print <- work.c      (batch-2 consumer)

        The invariant is item-level ordering of all_ordered_items, not
        individual chunk ordering (a non-adjacent batch by definition
        spans a gap in the source).
        """
        src = (
            "data work.a; set mylib.src; run;\n"
            "data work.lone; run;\n"
            "data work.b; set work.a; run;\n"
            "data work.c; set mylib.other; run;\n"
            "proc print data=work.c; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 2)
        self.assertEqual(len(br.singletons), 1)
        # Item-level start_lines must be non-decreasing
        items = br.all_ordered_items
        item_lines = [
            item.start_line if isinstance(item, SasBatch) else item.start_line
            for item in items
        ]
        self.assertEqual(item_lines, sorted(item_lines))
        # Chunks inside each batch are in source order
        for batch in br.batches:
            chunk_lines = [c.start_line for c in batch.chunks]
            self.assertEqual(chunk_lines, sorted(chunk_lines))


# ── 6. Context absorption ─────────────────────────────────────────────────


class TestContextAbsorption(unittest.TestCase):
    def test_options_chunk_absorbed_into_next_batch(self):
        """OPTIONS chunk before a dataset-flow batch is pulled in when enabled."""
        src = (
            "options mprint;\n"
            "data work.clean; set mylib.raw; run;\n"
            "proc print data=work.clean; run;\n"
        )
        _, br = _chunk_and_batch(src, include_options_chunks=True)
        # All three should now be in one batch
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.batches[0].chunks), 3)

    def test_options_chunk_singleton_when_disabled(self):
        """OPTIONS chunk stays a singleton when include_options_chunks=False."""
        src = (
            "options mprint;\n"
            "data work.clean; set mylib.raw; run;\n"
            "proc print data=work.clean; run;\n"
        )
        _, br = _chunk_and_batch(src, include_options_chunks=False)
        # options chunk is a singleton; the data+proc form their own batch
        singleton_kinds = [c.kind for c in br.singletons]
        self.assertIn(SasChunkKind.OPTIONS, singleton_kinds)

    def test_comment_chunk_absorbed_when_enabled(self):
        """COMMENT_BLOCK pulled into adjacent batch when include_comment_chunks=True."""
        src = (
            "/* clean the data */\n"
            "data work.clean; set mylib.raw; run;\n"
            "proc means data=work.clean; run;\n"
        )
        _, br = _chunk_and_batch(src, include_comment_chunks=True)
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.batches[0].chunks), 3)

    def test_comment_chunk_singleton_when_disabled(self):
        src = "/* standalone comment */\ndata work.a; set mylib.b; run;\n"
        _, br = _chunk_and_batch(src, include_comment_chunks=False)
        singleton_kinds = [c.kind for c in br.singletons]
        self.assertIn(SasChunkKind.COMMENT_BLOCK, singleton_kinds)


# ── 7. Batch reason strings ────────────────────────────────────────────────


class TestBatchReasons(unittest.TestCase):
    def test_dataset_flow_reason_contains_dataset_name(self):
        src = (
            "data work.enriched; set mylib.raw; run;\n"
            "proc freq data=work.enriched; tables region; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        reason = br.batches[0].reason
        self.assertIn("dataset_flow", reason)
        self.assertIn("work.enriched", reason)

    def test_macro_invocation_reason_contains_macro_name(self):
        src = "%macro do_it; data work.x; run; %mend;\n%do_it;\n"
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        reason = br.batches[0].reason
        self.assertIn("macro_invocation", reason)
        self.assertIn("%do_it", reason)

    def test_mixed_reason_contains_both_edge_types(self):
        src = (
            "%macro clean(ds); data &ds.; set &ds.; run; %mend;\n"
            "data work.out; set mylib.raw; run;\n"
            "%clean(work.out);\n"
            "proc print data=work.out; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        reason = br.batches[0].reason
        self.assertIn("macro_invocation", reason)
        self.assertIn("dataset_flow", reason)


# ── 8. Result model properties ────────────────────────────────────────────


class TestResultModel(unittest.TestCase):
    def test_all_ordered_items_covers_all_chunks(self):
        """all_ordered_items contains every original chunk exactly once.

        Chunk IDs in batched output carry a file-rank prefix (e.g. 'f1-chunk-0001')
        from the multi-file ID re-stamping pass.  We verify coverage by count
        rather than exact ID match.
        """
        src = (
            "libname mylib '/data';\n"
            "data work.a; set mylib.raw; run;\n"
            "data work.b; set work.a; run;\n"
            "proc print data=work.standalone; run;\n"
        )
        cr, br = _chunk_and_batch(src)
        ordered_ids = _ordered_chunk_ids(br)
        # Same count as original chunks
        self.assertEqual(len(ordered_ids), len(cr.chunks))
        # Each appears exactly once
        self.assertEqual(len(set(ordered_ids)), len(cr.chunks))

    def test_batch_chunk_ids_property(self):
        """chunk_ids returns one ID per member chunk, globally unique."""
        src = "data work.a; set mylib.raw; run;\nproc print data=work.a; run;\n"
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        ids = br.batches[0].chunk_ids
        self.assertEqual(len(ids), 2)
        # IDs carry a file-rank prefix after multi-file re-stamping
        self.assertTrue(all("chunk-" in cid for cid in ids))

    def test_batch_start_end_line(self):
        src = "data work.a; set mylib.raw; run;\nproc print data=work.a; run;\n"
        _, br = _chunk_and_batch(src)
        batch = br.batches[0]
        self.assertLessEqual(batch.start_line, batch.end_line)
        self.assertEqual(batch.start_line, batch.chunks[0].start_line)
        self.assertEqual(batch.end_line, batch.chunks[-1].end_line)

    def test_result_json_serialisable(self):
        import json

        src = "data work.a; set mylib.raw; run;\nproc print data=work.a; run;\n"
        _, br = _chunk_and_batch(src)
        json.dumps(br.model_dump())  # must not raise


# ── 9. Complex real-world-like programs ───────────────────────────────────


class TestComplexPrograms(unittest.TestCase):
    def test_etl_pipeline_full_flow(self):
        """
        Realistic ETL:
          1. LIBNAME (singleton / absorbed)
          2. PROC IMPORT
          3. DATA step clean
          4. PROC SORT x2
          5. DATA MERGE
          6. PROC MEANS with OUT=
          7. PROC FREQ
          8. PROC EXPORT
        """
        src = (
            "libname mylib '/data/sales';\n"
            "proc import datafile='/raw/orders.csv' dbms=csv out=mylib.orders replace;\n"
            "  guessingrows=200;\n"
            "run;\n"
            "data work.orders_clean;\n"
            "  set mylib.orders;\n"
            "  where order_dt >= '01JAN2020'd;\n"
            "run;\n"
            "proc sort data=work.orders_clean; by customer_id; run;\n"
            "proc sort data=mylib.customers;   by customer_id; run;\n"
            "data work.enriched;\n"
            "  merge work.orders_clean(in=a) mylib.customers(in=b);\n"
            "  by customer_id;\n"
            "  if a and b;\n"
            "run;\n"
            "proc means data=work.enriched noprint;\n"
            "  class region;\n"
            "  var revenue;\n"
            "  output out=work.summary sum=total_revenue;\n"
            "run;\n"
            "proc freq data=work.enriched;\n"
            "  tables region;\n"
            "run;\n"
            "proc export data=work.summary outfile='/output/summary.csv' dbms=csv replace;\n"
            "run;\n"
        )
        _, br = _chunk_and_batch(src, include_options_chunks=True)

        # There should be exactly one large batch (all connected via dataset flow)
        # plus the LIBNAME singleton (absorbed if include_options_chunks=True)
        total_batched = sum(len(b.chunks) for b in br.batches)
        total_singles = len(br.singletons)
        total = total_batched + total_singles
        # All 8 substantive steps must be accounted for
        self.assertGreaterEqual(total, 8)

        # Confirm work.enriched bridges the merge and the two proc steps
        all_batch_ds = {ds for b in br.batches for ds in b.output_datasets}
        self.assertIn("work.enriched", all_batch_ds)
        self.assertIn("work.summary", all_batch_ds)

    def test_macro_library_pattern(self):
        """
        Two macro definitions + two call sites + a downstream PROC.

        Static analysis result (with parameterised macro-body resolution):
          batch-1: %macro load + %load(work, '/a.csv')
                   — &lib..raw is a *compound* reference (parameter 'lib'
                     concatenated with literal suffix '.raw'); this is
                     intentionally left unresolved rather than guessing,
                     so %load does not link to anything downstream.
          batch-2: %macro clean + %clean(work.raw) + proc print data=work.raw
                   — &ds. inside %clean is a *single* parameter reference,
                     fully resolved to 'work.raw' at the call site, which
                     correctly links the call to the downstream PROC PRINT.

        This demonstrates the mixed literal/parameterised resolution
        strategy: simple single-variable parameter references resolve
        correctly across files/call-sites, while compound concatenated
        references (var + literal suffix, e.g. &lib..raw) are left
        unresolved rather than silently producing a wrong dataset name.
        """
        src = (
            "%macro load(lib, file);\n"
            "  proc import datafile=&file. dbms=csv out=&lib..raw replace;\n"
            "  run;\n"
            "%mend;\n"
            "%macro clean(ds);\n"
            "  data &ds.; set &ds.; if x<0 then delete; run;\n"
            "%mend;\n"
            "%load(work, '/a.csv');\n"
            "%clean(work.raw);\n"
            "proc print data=work.raw; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 2)
        self.assertEqual(len(br.singletons), 0)
        all_def_macros = {mac for b in br.batches for mac in b.defined_macros}
        self.assertIn("load", all_def_macros)
        self.assertIn("clean", all_def_macros)
        total = sum(len(b.chunks) for b in br.batches) + len(br.singletons)
        self.assertEqual(total, 5)
        # The %clean batch must now include the downstream PROC PRINT,
        # since &ds. resolves to 'work.raw' at the call site.
        clean_batch = next(b for b in br.batches if "clean" in b.defined_macros)
        self.assertEqual(len(clean_batch.chunks), 3)
        self.assertIn("work.raw", clean_batch.reason)

    def test_proc_sql_subquery_chain(self):
        """PROC SQL that references a table built by a prior DATA step."""
        src = (
            "data work.base;\n  set mylib.transactions;\nrun;\n"
            "proc sql;\n"
            "  create table work.agg as\n"
            "    select account_id, sum(amount) as total\n"
            "    from work.base\n"
            "    group by account_id;\n"
            "quit;\n"
            "proc sql;\n"
            "  create table work.final as\n"
            "    select a.*, b.name\n"
            "    from work.agg as a\n"
            "    join mylib.accounts as b on a.account_id = b.id;\n"
            "quit;\n"
        )
        _, br = _chunk_and_batch(src)
        # All three chunks are connected: base → agg → final
        self.assertEqual(len(br.batches), 1)
        batch = br.batches[0]
        self.assertEqual(len(batch.chunks), 3)
        self.assertIn("work.final", batch.output_datasets)


# ── 10. Order-aware dataset flow (nearest preceding producer) ─────────────


class TestOrderAwareDatasetFlow(unittest.TestCase):
    def test_scratch_name_reuse_stays_separate(self):
        """Two independent jobs both using work.tmp must NOT fuse: each
        consumer links only to the nearest preceding producer."""
        src = (
            "data work.tmp; set mylib.a; run;\n"
            "proc print data=work.tmp; run;\n"
            "data work.tmp; set mylib.b; run;\n"
            "proc means data=work.tmp; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 2)
        sizes = sorted(len(b.chunks) for b in br.batches)
        self.assertEqual(sizes, [2, 2])
        # Each batch reads a different external source
        ext_inputs = sorted(ds for b in br.batches for ds in b.input_datasets)
        self.assertEqual(ext_inputs, ["mylib.a", "mylib.b"])

    def test_consumer_before_producer_gets_no_edge(self):
        """A read that precedes the only producer of that name cannot be
        fed by it under sequential SAS execution → no edge."""
        src = (
            "proc print data=work.later; run;\n"
            "data work.later; set mylib.raw; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 0)
        self.assertEqual(len(br.singletons), 2)


# ── 11. Global-context batch (tiered weak-edge resolution) ────────────────


class TestGlobalContextBatch(unittest.TestCase):
    def test_shared_let_promoted_to_globals(self):
        """A %LET consumed by two independent pipelines becomes the first
        batch (is_global_context) instead of fusing the pipelines."""
        src = (
            "%let cutoff = 20240101;\n"
            "data work.a; set mylib.x; where dt >= &cutoff; run;\n"
            "proc print data=work.a; run;\n"
            "data work.b; set mylib.y; where dt >= &cutoff; run;\n"
            "proc means data=work.b; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 3)
        globals_batch = br.batches[0]
        self.assertTrue(globals_batch.is_global_context)
        self.assertEqual(globals_batch.batch_id, "batch-001")
        self.assertIn("cutoff", globals_batch.produced_macrovars)
        # The two pipelines stay separate and report the external need
        for b in br.batches[1:]:
            self.assertFalse(b.is_global_context)
            self.assertEqual(len(b.chunks), 2)
            self.assertIn("cutoff", b.required_macrovars)

    def test_single_consumer_macro_still_absorbed(self):
        """A macro with exactly one call site batches with it — no globals
        batch is created."""
        src = (
            "%macro clean(ds); data &ds.; set &ds.; run; %mend;\n"
            "data work.raw; set mylib.x; run;\n"
            "%clean(work.raw);\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        self.assertFalse(br.batches[0].is_global_context)
        self.assertIn("clean", br.batches[0].defined_macros)

    def test_no_globals_batch_when_no_shared_producers(self):
        """Ordinary corpora get no global-context batch and numbering
        starts at batch-001 as before."""
        src = (
            "data work.a; set mylib.raw; run;\n"
            "proc print data=work.a; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        self.assertFalse(br.batches[0].is_global_context)
        self.assertEqual(br.batches[0].batch_id, "batch-001")


# ── 12. Call-argument parsing with nested parens ──────────────────────────


class TestNestedParenCallArgs(unittest.TestCase):
    def test_str_quoted_arg_does_not_truncate_list(self):
        """%str(...) inside the argument list must not stop parsing at its
        inner ')': the dataset argument after it still resolves."""
        from chunker.batcher import _parse_call_args

        pos, kw = _parse_call_args("%clean(%str(a,b), out=work.x);")
        self.assertEqual(pos, ["%str(a,b)"])
        self.assertEqual(kw, {"out": "work.x"})

    def test_comma_after_nested_parens_still_splits(self):
        from chunker.batcher import _parse_call_args

        pos, kw = _parse_call_args("%m(a, f(x,y), b);")
        self.assertEqual(pos, ["a", "f(x,y)", "b"])
        self.assertEqual(kw, {})

    def test_positional_value_with_equals_inside_parens_not_keyword(self):
        from chunker.batcher import _parse_call_args

        pos, kw = _parse_call_args("%m(f(x=1), ds=work.a);")
        self.assertEqual(pos, ["f(x=1)"])
        self.assertEqual(kw, {"ds": "work.a"})

    def test_macro_name_matching_dataset_creates_no_arg_edge(self):
        """A macro NAMED like a produced dataset must not create a
        macro_arg_dataset edge — only argument values are scanned."""
        src = (
            "data work.base; set mylib.raw; run;\n"
            "%base(some_param);\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 0)
        self.assertEqual(len(br.singletons), 2)

    def test_keyword_name_matching_dataset_creates_no_arg_edge(self):
        """A keyword parameter NAME that collides with a produced dataset
        must not create an edge; only its value is scanned."""
        src = (
            "data work.title; set mylib.raw; run;\n"
            "%report(title=nothing_relevant);\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 0)
        self.assertEqual(len(br.singletons), 2)


# ── 13. Single-file chunk IDs are not re-stamped ──────────────────────────


class TestSingleFileChunkIds(unittest.TestCase):
    def test_ids_match_input_chunk_result(self):
        src = "data work.a; set mylib.raw; run;\nproc print data=work.a; run;\n"
        cr, br = _chunk_and_batch(src)
        original_ids = [c.chunk_id for c in cr.chunks]
        batched_ids = _ordered_chunk_ids(br)
        self.assertEqual(sorted(batched_ids), sorted(original_ids))
        self.assertTrue(all(not cid.startswith("f1-") for cid in batched_ids))


# ── 14. Databricks dataset-name mapping ───────────────────────────────────


class TestDatabricksMapping(unittest.TestCase):
    SRC = (
        "data work.clean;\n set mylib.raw;\n run;\n"
        "proc means data=work.clean; run;\n"
    )
    MAPPING = {
        "work": "dev.staging",
        "mylib": "prod.sales",
    }

    def test_libref_mapping_rewrites_batch_datasets(self):
        """libref → catalog.schema rewrites batch inputs/outputs to
        three-part Databricks names."""
        _, br = _chunk_and_batch(self.SRC, databricks_mapping=self.MAPPING)
        batch = br.batches[0]
        self.assertEqual(batch.output_datasets, ["dev.staging.clean"])
        self.assertEqual(batch.input_datasets, ["prod.sales.raw"])

    def test_chunk_metadata_rewritten_in_batches_and_singletons(self):
        src = self.SRC + "data work.other; set otherlib.src; run;\n"
        _, br = _chunk_and_batch(src, databricks_mapping=self.MAPPING)
        data_chunk = br.batches[0].chunks[0]
        self.assertEqual(data_chunk.metadata.output_datasets, ["dev.staging.clean"])
        self.assertEqual(data_chunk.metadata.input_datasets, ["prod.sales.raw"])
        (solo,) = br.singletons
        # Unmapped libref passes through; mapped work. member is rewritten.
        self.assertEqual(solo.metadata.output_datasets, ["dev.staging.other"])
        self.assertEqual(solo.metadata.input_datasets, ["otherlib.src"])

    def test_exact_override_beats_libref_mapping(self):
        mapping = {**self.MAPPING, "work.clean": "gold.curated.cleaned_v2"}
        _, br = _chunk_and_batch(self.SRC, databricks_mapping=mapping)
        batch = br.batches[0]
        self.assertEqual(batch.output_datasets, ["gold.curated.cleaned_v2"])

    def test_grouping_identical_with_and_without_mapping(self):
        """The mapping is a post-pass: batch/singleton structure, chunk IDs,
        reason strings, and required_librefs are byte-identical."""
        _, plain = _chunk_and_batch(self.SRC)
        _, mapped = _chunk_and_batch(self.SRC, databricks_mapping=self.MAPPING)
        self.assertEqual(_batch_ids(plain), _batch_ids(mapped))
        self.assertEqual(_singleton_ids(plain), _singleton_ids(mapped))
        self.assertEqual(_ordered_chunk_ids(plain), _ordered_chunk_ids(mapped))
        self.assertEqual(
            [b.reason for b in plain.batches], [b.reason for b in mapped.batches]
        )
        self.assertEqual(
            [b.required_librefs for b in plain.batches],
            [b.required_librefs for b in mapped.batches],
        )

    def test_no_mapping_is_noop(self):
        _, plain = _chunk_and_batch(self.SRC)
        _, empty = _chunk_and_batch(self.SRC, databricks_mapping={})
        self.assertEqual(plain.model_dump(), empty.model_dump())

    def test_one_level_names_map_via_work(self):
        """Bare one-level names were canonicalised to work.<name>, so the
        'work' libref entry covers them."""
        src = "data clean; set mylib.raw; run;\nproc print data=clean; run;\n"
        _, br = _chunk_and_batch(src, databricks_mapping=self.MAPPING)
        batch = br.batches[0]
        self.assertEqual(batch.output_datasets, ["dev.staging.clean"])

    def test_input_chunk_result_not_mutated(self):
        cr, _ = _chunk_and_batch(self.SRC, databricks_mapping=self.MAPPING)
        all_names = [
            ds
            for c in cr.chunks
            for ds in (*c.metadata.input_datasets, *c.metadata.output_datasets)
        ]
        self.assertTrue(all(ds.count(".") < 2 for ds in all_names))

    def test_standalone_function_on_existing_result(self):
        from chunker import replace_dataset_names

        _, plain = _chunk_and_batch(self.SRC)
        mapped = replace_dataset_names(plain, self.MAPPING)
        self.assertEqual(mapped.batches[0].output_datasets, ["dev.staging.clean"])
        # original untouched
        self.assertEqual(plain.batches[0].output_datasets, ["work.clean"])

    # -- %let / %global / %local values holding a dataset reference --------

    def _all_texts(self, br) -> str:
        texts = [c.text for b in br.batches for c in b.chunks]
        texts += [c.text for c in br.singletons]
        return "\n".join(texts)

    def test_let_value_dataset_reference_rewritten(self):
        src = "%let src = mylib.orders;\ndata work.a; set &src; run;\n"
        _, br = _chunk_and_batch(src, databricks_mapping=self.MAPPING)
        joined = self._all_texts(br)
        self.assertIn("%let src = prod.sales.orders;", joined)
        self.assertNotIn("mylib.orders", joined)

    def test_global_then_let_pattern_rewritten(self):
        src = (
            "%global src;\n"
            "%let src = mylib.orders;\n"
            "data work.a; set &src; run;\n"
        )
        _, br = _chunk_and_batch(src, databricks_mapping=self.MAPPING)
        joined = self._all_texts(br)
        self.assertIn("%global src;", joined)  # declaration untouched
        self.assertIn("%let src = prod.sales.orders;", joined)

    def test_quoted_let_value_rewritten_preserving_quotes(self):
        src = "%let src = 'mylib.orders';\ndata work.a; set mylib.orders; run;\n"
        _, br = _chunk_and_batch(src, databricks_mapping=self.MAPPING)
        self.assertIn("%let src = 'prod.sales.orders';", self._all_texts(br))

    def test_non_dataset_let_values_untouched(self):
        """Only a two-level libref.member value is a dataset reference; free
        text, numbers, and bare identifiers pass through even with 'work'
        mapped."""
        src = (
            "%let cutoff = 20240101;\n"
            "%let ttl = hello world;\n"
            "%let name = orders;\n"
            "data work.a; x=&cutoff; run;\n"
        )
        _, br = _chunk_and_batch(src, databricks_mapping=self.MAPPING)
        joined = self._all_texts(br)
        self.assertIn("%let cutoff = 20240101;", joined)
        self.assertIn("%let ttl = hello world;", joined)
        self.assertIn("%let name = orders;", joined)

    def test_let_rewrite_only_with_mapped_libref(self):
        src = "%let src = otherlib.orders;\ndata work.a; set &src; run;\n"
        _, br = _chunk_and_batch(src, databricks_mapping=self.MAPPING)
        self.assertIn("%let src = otherlib.orders;", self._all_texts(br))

    # -- SET-statement rename and PROC OUTPUT OUT= scenarios ----------------

    def test_set_statement_rename_maps_both_names(self):
        """data new; set old; — the dataset's new name (and the old) reach
        the mapper through the extracted metadata."""
        src = (
            "data work.orig; x=1; run;\n"
            "data work.renamed; set work.orig; run;\n"
        )
        _, br = _chunk_and_batch(src, databricks_mapping=self.MAPPING)
        batch = br.batches[0]
        self.assertEqual(
            batch.output_datasets, ["dev.staging.orig", "dev.staging.renamed"]
        )
        rename_chunk = batch.chunks[1]
        self.assertEqual(rename_chunk.metadata.input_datasets, ["dev.staging.orig"])
        self.assertEqual(
            rename_chunk.metadata.output_datasets, ["dev.staging.renamed"]
        )

    def test_proc_output_statement_dataset_mapped(self):
        """PROC ... OUTPUT OUT=work.stats — the OUT= name reaches the mapper
        through the extracted metadata."""
        src = (
            "data work.clean; set mylib.raw; run;\n"
            "proc means data=work.clean; var x; output out=work.stats mean=avg; run;\n"
        )
        _, br = _chunk_and_batch(src, databricks_mapping=self.MAPPING)
        batch = br.batches[0]
        self.assertIn("dev.staging.stats", batch.output_datasets)
        proc_chunk = batch.chunks[1]
        self.assertIn("dev.staging.stats", proc_chunk.metadata.output_datasets)
        self.assertIn("dev.staging.clean", proc_chunk.metadata.input_datasets)


# ── 15. Databricks mapping CSV parser ─────────────────────────────────────


class TestDatabricksMappingCsv(unittest.TestCase):
    def _parse(self, text: str) -> dict:
        from chunker import parse_databricks_mapping_csv

        return parse_databricks_mapping_csv(text)

    def test_basic_rows_with_header(self):
        csv_text = (
            "sas_name,databricks_name\n"
            "work,dev.staging\n"
            "mylib.orders,prod.sales.orders_v2\n"
        )
        self.assertEqual(
            self._parse(csv_text),
            {"work": "dev.staging", "mylib.orders": "prod.sales.orders_v2"},
        )

    def test_comments_blanks_and_extra_columns_skipped(self):
        csv_text = (
            "# comment row\n"
            "\n"
            "WORK,dev.staging,ignored extra\n"
        )
        self.assertEqual(self._parse(csv_text), {"work": "dev.staging"})

    def test_malformed_rows_skipped(self):
        csv_text = "work\n,dev.staging\nmylib,prod.sales\n"
        self.assertEqual(self._parse(csv_text), {"mylib": "prod.sales"})

    def test_duplicate_key_last_wins(self):
        csv_text = "work,dev.staging\nwork,qa.scratch\n"
        self.assertEqual(self._parse(csv_text), {"work": "qa.scratch"})

    def test_parsed_mapping_round_trips_through_batcher(self):
        csv_text = "sas,databricks\nwork,dev.staging\nmylib,prod.sales\n"
        mapping = self._parse(csv_text)
        src = (
            "data work.clean;\n set mylib.raw;\n run;\n"
            "proc means data=work.clean; run;\n"
        )
        _, br = _chunk_and_batch(src, databricks_mapping=mapping)
        self.assertEqual(br.batches[0].output_datasets, ["dev.staging.clean"])
        self.assertEqual(br.batches[0].input_datasets, ["prod.sales.raw"])


class TestCoalesceIntoBatches(unittest.TestCase):
    """coalesce_into_batches turns the ordered items into SasBatch-only."""

    def _items(self, source: str):
        _, br = _chunk_and_batch(source)
        return br.all_ordered_items

    def test_every_output_is_a_batch(self):
        from chunker.batcher import coalesce_into_batches

        # Three independent DATA steps -> three singletons.
        src = "data work.a; x=1; run;\ndata work.b; y=2; run;\ndata work.c; z=3; run;\n"
        out = coalesce_into_batches(self._items(src), max_chunks=8)
        self.assertTrue(all(isinstance(b, SasBatch) for b in out))
        # All three consecutive singletons merged into one batch.
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].batch_id, "merged-001")
        self.assertEqual(len(out[0].chunks), 3)

    def test_max_chunks_caps_group_size(self):
        from chunker.batcher import coalesce_into_batches

        src = "".join(f"data work.t{i}; v=1; run;\n" for i in range(5))
        out = coalesce_into_batches(self._items(src), max_chunks=2)
        # 5 singletons, cap 2 -> [2, 2, 1].
        self.assertEqual([len(b.chunks) for b in out], [2, 2, 1])
        self.assertEqual([b.batch_id for b in out], ["merged-001", "merged-002", "merged-003"])

    def test_real_batches_pass_through_and_split_singleton_runs(self):
        from chunker.batcher import coalesce_into_batches

        # A dependency pair (a->print a) sits between independent singletons.
        src = (
            "data work.x; p=1; run;\n"           # singleton
            "data work.dep; q=1; run;\n"          # producer
            "proc print data=work.dep; run;\n"    # consumer -> batch with producer
            "data work.y; r=1; run;\n"            # singleton
        )
        out = coalesce_into_batches(self._items(src), max_chunks=8)
        kinds = ["real" if b.batch_id.startswith("batch") else "merged" for b in out]
        # merged(x)  batch(dep+print)  merged(y) — singletons never span the batch.
        self.assertEqual(kinds, ["merged", "real", "merged"])
        self.assertEqual(len(out[1].chunks), 2)

    def test_merged_batch_aggregates_external_io(self):
        from chunker.batcher import coalesce_into_batches

        # Two independent steps; work.seed produced outside the run is external.
        src = (
            "data work.one; set other.seed; a=1; run;\n"
            "data work.two; b=2; run;\n"
        )
        out = coalesce_into_batches(self._items(src), max_chunks=8)
        self.assertEqual(len(out), 1)
        batch = out[0]
        self.assertIn("other.seed", batch.input_datasets)
        self.assertIn("work.one", batch.output_datasets)
        self.assertIn("work.two", batch.output_datasets)
        self.assertIn("other", batch.required_librefs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
