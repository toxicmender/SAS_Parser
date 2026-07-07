"""
test_multifile.py — unit tests for MultiFileBatcher (cross-file dependency batching).

All tests use in-memory SAS strings so no disk I/O is required.
The helper ``_corpus(*sources)`` chunks each source independently and
assembles a SasCorpus in the order provided.

Run:  python -m pytest tests/test_multifile.py -v

Test sections
-------------
1.  Corpus construction
2.  No cross-file dependencies (each file independent)
3.  Cross-file dataset-flow edges
4.  Cross-file macro-invocation edges
5.  Cross-file macro-argument-dataset edges
6.  Transitive cross-file closure
7.  Source ordering inside batches
8.  Singleton / batch I/O fields
9.  Context absorption (same-file only)
10. SasMultiBatchResult model properties
11. Complex realistic multi-file programs
12. MultiFileBatcher.from_files factory (mocked paths)
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chunker import (
    SasBatch,
    SasCorpus,
    SasMultiBatchResult,
    SasSemanticChunker,
)
from chunker.batcher import MultiFileBatcher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHUNKER = SasSemanticChunker(min_words=1, max_words=9_999)


def _corpus(*sources: str, ids: list[str] | None = None) -> SasCorpus:
    """Chunk each source string and return a SasCorpus in order."""
    results = []
    for i, src in enumerate(sources):
        sid = (ids[i] if ids else None) or f"file_{i + 1}.sas"
        results.append(_CHUNKER.chunk_text(src, source_id=sid))
    return SasCorpus(file_results=results)


def _batch(
    *sources: str,
    ids: list[str] | None = None,
    include_options: bool = True,
    include_comments: bool = False,
) -> SasMultiBatchResult:
    corp = _corpus(*sources, ids=ids)
    return MultiFileBatcher(
        include_options_chunks=include_options,
        include_comment_chunks=include_comments,
    ).batch(corp)


def _all_chunk_ids(result: SasMultiBatchResult) -> list[str]:
    ids = []
    for item in result.all_ordered_items:
        if isinstance(item, SasBatch):
            ids.extend(c.chunk_id for c in item.chunks)
        else:
            ids.append(item.chunk_id)
    return ids


# ---------------------------------------------------------------------------
# 1. Corpus construction
# ---------------------------------------------------------------------------


class TestCorpusConstruction(unittest.TestCase):
    def test_corpus_source_ids(self):
        corp = _corpus(
            "data work.a; run;", "data work.b; run;", ids=["alpha.sas", "beta.sas"]
        )
        self.assertEqual(corp.source_ids, ["alpha.sas", "beta.sas"])

    def test_corpus_all_chunks_flat(self):
        corp = _corpus(
            "data work.a; run;\ndata work.b; run;",
            "data work.c; run;",
        )
        self.assertEqual(len(corp.all_chunks), 3)

    def test_corpus_all_diagnostics(self):
        corp = _corpus(
            "data work.a;\n x=1;\n",  # unclosed
            "data work.b; run;",
        )
        self.assertGreater(len(corp.all_diagnostics), 0)

    def test_empty_corpus(self):
        br = MultiFileBatcher().batch(SasCorpus())
        self.assertEqual(len(br.batches), 0)
        self.assertEqual(len(br.singletons), 0)

    def test_single_file_corpus_matches_single_batcher(self):
        from chunker import SasChunkBatcher

        src = "data work.a; set mylib.raw; run;\nproc print data=work.a; run;\n"
        cr = _CHUNKER.chunk_text(src, source_id="f.sas")
        corp = SasCorpus(file_results=[cr])

        multi_br = MultiFileBatcher().batch(corp)
        single_br = SasChunkBatcher().batch(cr)

        self.assertEqual(len(multi_br.batches), len(single_br.batches))
        self.assertEqual(len(multi_br.singletons), len(single_br.singletons))


# ---------------------------------------------------------------------------
# 2. No cross-file dependencies — all chunks independent
# ---------------------------------------------------------------------------


class TestNoCrossFileDependencies(unittest.TestCase):
    def test_two_independent_files_all_singletons(self):
        br = _batch(
            "data work.a; x=1; run;",
            "data work.b; y=2; run;",
        )
        self.assertEqual(len(br.batches), 0)
        self.assertEqual(len(br.singletons), 2)

    def test_three_files_no_shared_datasets(self):
        br = _batch(
            "data work.p; run;\ndata work.q; run;",
            "data work.r; run;",
            "proc print data=work.standalone; run;",
        )
        self.assertEqual(len(br.batches), 0)
        self.assertEqual(len(br.singletons), 4)

    def test_intra_file_batches_preserved(self):
        """Each file has its own internal batch; no cross-file edges."""
        br = _batch(
            "data work.a; set mylib.x; run;\nproc print data=work.a; run;",
            "data work.b; set mylib.y; run;\nproc means data=work.b; run;",
        )
        self.assertEqual(len(br.batches), 2)
        for b in br.batches:
            self.assertFalse(b.is_cross_file)
            self.assertEqual(len(b.source_files), 1)


# ---------------------------------------------------------------------------
# 3. Cross-file dataset-flow edges
# ---------------------------------------------------------------------------


class TestCrossFileDatasetFlow(unittest.TestCase):
    def test_file_a_produces_file_b_consumes(self):
        """Simple A→B cross-file flow: one cross-file batch."""
        br = _batch(
            "data work.base; set mylib.raw; run;",  # file 1 produces work.base
            "proc print data=work.base; run;",  # file 2 consumes work.base
        )
        self.assertEqual(len(br.batches), 1)
        batch = br.batches[0]
        self.assertTrue(batch.is_cross_file)
        self.assertEqual(len(batch.source_files), 2)
        self.assertIn("work.base", batch.output_datasets)

    def test_multiple_consumers_across_files(self):
        """File 1 produces work.ds; files 2 and 3 both consume it → one batch."""
        br = _batch(
            "data work.ds; set mylib.src; run;",
            "proc print data=work.ds; run;",
            "proc means data=work.ds; run;",
        )
        self.assertEqual(len(br.batches), 1)
        self.assertTrue(br.batches[0].is_cross_file)
        self.assertEqual(len(br.batches[0].chunks), 3)

    def test_file_b_feeds_file_c(self):
        """File 2 produces and file 3 consumes — file 1 is independent."""
        br = _batch(
            "data work.standalone; x=1; run;",  # independent
            "data work.mid; set mylib.raw; run;",  # produces work.mid
            "proc means data=work.mid; run;",  # consumes work.mid
        )
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.singletons), 1)
        self.assertTrue(br.batches[0].is_cross_file)
        self.assertIn("file_2.sas", br.batches[0].source_files)
        self.assertIn("file_3.sas", br.batches[0].source_files)

    def test_cross_file_merge(self):
        """File 1 sorts left, file 2 sorts right, file 3 merges them."""
        br = _batch(
            "proc sort data=work.left; by id; run;",
            "proc sort data=work.right; by id; run;",
            "data work.merged;\n"
            "  merge work.left(in=a) work.right(in=b);\n"
            "  by id;\n"
            "run;",
        )
        self.assertEqual(len(br.batches), 1)
        batch = br.batches[0]
        self.assertTrue(batch.is_cross_file)
        self.assertEqual(len(batch.chunks), 3)
        self.assertIn("work.merged", batch.output_datasets)

    def test_proc_sql_cross_file(self):
        """PROC SQL in file 2 reads table produced by file 1."""
        br = _batch(
            "data work.orders; set mylib.raw; run;",
            "proc sql;\n"
            "  create table work.summary as\n"
            "    select region, sum(revenue) as total\n"
            "    from work.orders\n"
            "    group by region;\n"
            "quit;",
        )
        self.assertEqual(len(br.batches), 1)
        self.assertTrue(br.batches[0].is_cross_file)
        self.assertIn("work.summary", br.batches[0].output_datasets)

    def test_inplace_sort_cross_file_consumer(self):
        """File 1 sorts work.ds in-place; file 2 reads work.ds → cross-file batch."""
        br = _batch(
            "proc sort data=work.ds; by name; run;",
            "proc print data=work.ds; run;",
        )
        self.assertEqual(len(br.batches), 1)
        self.assertTrue(br.batches[0].is_cross_file)

    def test_two_independent_cross_file_pipelines(self):
        """Two pairs of files each with their own flow: two separate batches."""
        br = _batch(
            "data work.p1; set mylib.a; run;",
            "proc print data=work.p1; run;",
            "data work.p2; set mylib.b; run;",
            "proc means data=work.p2; run;",
        )
        self.assertEqual(len(br.batches), 2)
        self.assertTrue(all(b.is_cross_file for b in br.batches))

    def test_external_input_not_batched(self):
        """Dataset from external library has no producer → no cross-file edge."""
        br = _batch(
            "data work.a; set mylib.external; run;",
            "proc print data=work.standalone; run;",
        )
        self.assertEqual(len(br.batches), 0)
        self.assertEqual(len(br.singletons), 2)

    def test_cross_file_external_inputs_reported(self):
        """External inputs (not produced by any file) appear in batch.input_datasets."""
        br = _batch(
            "data work.clean; set mylib.raw; run;",
            "proc means data=work.clean; run;",
        )
        self.assertEqual(len(br.batches), 1)
        self.assertIn("mylib.raw", br.batches[0].input_datasets)
        self.assertNotIn("work.clean", br.batches[0].input_datasets)


# ---------------------------------------------------------------------------
# 4. Cross-file macro-invocation edges
# ---------------------------------------------------------------------------


class TestCrossFileMacroInvocation(unittest.TestCase):
    def test_macro_def_in_file_a_call_in_file_b(self):
        """Classic macro library pattern: definitions in file 1, calls in file 2."""
        br = _batch(
            "%macro clean(ds);\n"
            "  data &ds.; set &ds.; if x<0 then delete; run;\n"
            "%mend;",
            "%clean(work.orders);",
        )
        self.assertEqual(len(br.batches), 1)
        batch = br.batches[0]
        self.assertTrue(batch.is_cross_file)
        self.assertIn("clean", batch.defined_macros)
        self.assertEqual(len(batch.chunks), 2)

    def test_macro_def_multiple_files_calls(self):
        """Macro defined in file 1; called in files 2 and 3.  The definition
        feeds two independent components, so it becomes the global-context
        batch and the two call sites stay separate singletons."""
        br = _batch(
            "%macro report(ds);\n  proc print data=&ds.; run;\n%mend;",
            "%report(work.a);",
            "%report(work.b);",
        )
        self.assertEqual(len(br.batches), 1)
        batch = br.batches[0]
        self.assertTrue(batch.is_global_context)
        self.assertIn("report", batch.defined_macros)
        self.assertEqual(batch.source_files, ["file_1.sas"])
        self.assertEqual(len(br.singletons), 2)
        singleton_sources = {c.source_id for c in br.singletons}
        self.assertEqual(singleton_sources, {"file_2.sas", "file_3.sas"})

    def test_two_macro_libs_separate_batches(self):
        """Two unrelated macro def+call pairs → two separate cross-file batches."""
        br = _batch(
            "%macro m1; data work.x; run; %mend;",
            "%macro m2; data work.y; run; %mend;",
            "%m1;",
            "%m2;",
        )
        self.assertEqual(len(br.batches), 2)
        self.assertTrue(all(b.is_cross_file for b in br.batches))

    def test_unresolved_cross_file_macro_call_stays_singleton(self):
        """Macro called in file 2 has no definition anywhere → singleton."""
        br = _batch(
            "data work.a; x=1; run;",
            "%unknown_macro(some_param);",
        )
        self.assertEqual(len(br.batches), 0)
        self.assertEqual(len(br.singletons), 2)

    def test_macro_def_in_middle_file(self):
        """Macro defined in file 2 (not first); file 3 calls it."""
        br = _batch(
            "data work.setup; run;",  # file 1 — unrelated
            "%macro transform(ds);\n  data &ds.; run;\n%mend;",  # file 2 — defines
            "%transform(work.out);",  # file 3 — calls
        )
        # file 2 and file 3 must be in one batch; file 1 is a singleton
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.singletons), 1)
        self.assertTrue(br.batches[0].is_cross_file)

    def test_macro_redefinition_warning_cross_file(self):
        """Same macro name defined in two files — last definition wins; warning logged."""

        with self.assertLogs("chunker.batcher", level="WARNING") as cm:
            br = _batch(
                "%macro helper; data work.v1; run; %mend;",
                "%macro helper; data work.v2; run; %mend;",  # redefines
                "%helper;",
            )
        self.assertTrue(any("redefined" in msg for msg in cm.output))


# ---------------------------------------------------------------------------
# 5. Cross-file macro-argument-dataset edges
# ---------------------------------------------------------------------------


class TestCrossFileMacroArgDataset(unittest.TestCase):
    def test_macro_call_passes_cross_file_dataset(self):
        """
        File 1 defines %flag and produces work.out.
        File 2 calls %flag(work.out).

        Three edges must fire:
          macro_invocation  : file 1 %macro flag  → file 2 %flag call
          macro_arg_dataset : file 1 DATA work.out → file 2 %flag call
        All three chunks end up in one cross-file batch.
        """
        br = _batch(
            "%macro flag(ds);\n"
            "  data &ds.; set &ds.; flag=1; run;\n"
            "%mend;\n"
            "data work.out; set mylib.raw; run;",
            "%flag(work.out);",
        )
        self.assertEqual(len(br.batches), 1)
        batch = br.batches[0]
        self.assertTrue(batch.is_cross_file)
        self.assertIn("flag", batch.defined_macros)
        self.assertEqual(len(batch.chunks), 3)

    def test_macro_arg_links_file2_producer_to_file3_call(self):
        """File 2 produces work.ds; file 3 calls macro passing work.ds."""
        br = _batch(
            "%macro process(ds); proc print data=&ds.; run; %mend;",  # file 1
            "data work.ds; set mylib.raw; run;",  # file 2
            "%process(work.ds);",  # file 3
        )
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.batches[0].chunks), 3)
        self.assertTrue(br.batches[0].is_cross_file)


# ---------------------------------------------------------------------------
# 6. Transitive cross-file closure
# ---------------------------------------------------------------------------


class TestTransitiveCrossFile(unittest.TestCase):
    def test_three_file_chain(self):
        """File1 → work.a → File2 → work.b → File3: all in one batch."""
        br = _batch(
            "data work.a; set mylib.raw; run;",
            "data work.b; set work.a; run;",
            "proc print data=work.b; run;",
        )
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.batches[0].chunks), 3)
        self.assertTrue(br.batches[0].is_cross_file)

    def test_four_file_chain(self):
        """File1→File2→File3→File4 chain via datasets → single batch."""
        br = _batch(
            "data work.step1; set mylib.src; run;",
            "data work.step2; set work.step1; run;",
            "data work.step3; set work.step2; run;",
            "proc means data=work.step3; run;",
        )
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.batches[0].chunks), 4)

    def test_diamond_cross_file(self):
        """
        File1 produces work.shared → File2 and File3 each read it →
        File4 merges their outputs: one batch spanning all four files.
        """
        br = _batch(
            "data work.shared; set mylib.raw; run;",  # file 1
            "data work.left;   set work.shared; run;",  # file 2
            "data work.right;  set work.shared; run;",  # file 3
            "data work.joined;\n"  # file 4
            "  merge work.left work.right;\n"
            "  by id;\n"
            "run;",
        )
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.batches[0].chunks), 4)

    def test_macro_chain_plus_dataset_chain(self):
        """Macro def in file 1, dataset producer in file 2, call+consumer in file 3."""
        br = _batch(
            "%macro enrich(ds);\n"
            "  data &ds.; set &ds.; score=revenue*1.1; run;\n"
            "%mend;",
            "data work.orders; set mylib.raw; run;",
            "%enrich(work.orders);\nproc means data=work.orders; run;",
        )
        self.assertEqual(len(br.batches), 1)
        self.assertTrue(br.batches[0].is_cross_file)

    def test_two_independent_chains_stay_separate(self):
        br = _batch(
            "data work.x1; set mylib.a; run;",
            "proc print data=work.x1; run;",
            "data work.y1; set mylib.b; run;",
            "proc means data=work.y1; run;",
        )
        self.assertEqual(len(br.batches), 2)
        sizes = sorted(len(b.chunks) for b in br.batches)
        self.assertEqual(sizes, [2, 2])


# ---------------------------------------------------------------------------
# 7. Source ordering inside cross-file batches
# ---------------------------------------------------------------------------


class TestBatchOrdering(unittest.TestCase):
    def test_producers_before_consumers_in_batch(self):
        """Producer chunk (file 1) must appear before consumer (file 2) in batch."""
        br = _batch(
            "data work.base; set mylib.raw; run;",
            "proc print data=work.base; run;",
        )
        batch = br.batches[0]
        sources = [c.source_id for c in batch.chunks]
        self.assertEqual(sources[0], "file_1.sas")
        self.assertEqual(sources[1], "file_2.sas")

    def test_all_ordered_items_file_then_line(self):
        """all_ordered_items respects (file_rank, start_line) globally."""
        br = _batch(
            "data work.a; set mylib.raw; run;\ndata work.lone; run;",  # file 1
            "proc print data=work.a; run;",  # file 2
        )
        items = br.all_ordered_items
        # Extract (source_file, start_line) for each item's first chunk
        positions = []
        source_ids = br.source_ids
        for item in items:
            first = item.chunks[0] if isinstance(item, SasBatch) else item
            fid = first.source_id or ""
            rank = source_ids.index(fid) if fid in source_ids else 99
            positions.append((rank, first.start_line))
        self.assertEqual(positions, sorted(positions))

    def test_intra_batch_chunks_in_source_order(self):
        """Within any batch, chunks appear in (file_rank, start_line) order."""
        br = _batch(
            "data work.a; set mylib.raw; run;\n"
            "data work.b; set work.a; run;",  # file 1 — two steps
            "proc print data=work.b; run;",  # file 2
        )
        self.assertEqual(len(br.batches), 1)
        lines = [(c.source_id, c.start_line) for c in br.batches[0].chunks]
        self.assertEqual(lines, sorted(lines))

    def test_cross_file_batch_source_files_order(self):
        """source_files lists files in the order they appear in the batch."""
        br = _batch(
            "data work.mid; set mylib.raw; run;",
            "data work.out; set work.mid; run;",
            "proc print data=work.out; run;",
        )
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(
            br.batches[0].source_files, ["file_1.sas", "file_2.sas", "file_3.sas"]
        )


# ---------------------------------------------------------------------------
# 8. Batch I/O fields
# ---------------------------------------------------------------------------


class TestBatchIOFields(unittest.TestCase):
    def test_cross_file_external_inputs(self):
        br = _batch(
            "data work.clean; set mylib.raw; run;",
            "proc means data=work.clean; run;",
        )
        batch = br.batches[0]
        self.assertIn("mylib.raw", batch.input_datasets)
        self.assertNotIn("work.clean", batch.input_datasets)

    def test_cross_file_output_datasets(self):
        br = _batch(
            "data work.a; set mylib.x; run;",
            "data work.b; set work.a; run;",
        )
        outs = set(br.batches[0].output_datasets)
        self.assertIn("work.a", outs)
        self.assertIn("work.b", outs)

    def test_cross_file_required_macros(self):
        """
        File 2 contains a DATA step that reads work.a (produced in file 1)
        and also invokes %undefined_macro (not defined anywhere).

        Because the DATA step is cross-file-linked to file 1, it ends up
        in the batch, and %undefined_macro appears in required_macros.
        """
        br = _batch(
            "data work.a; set mylib.raw; run;",
            "data work.report; set work.a; %undefined_macro; run;",
        )
        self.assertEqual(len(br.batches), 1)
        self.assertIn("undefined_macro", br.batches[0].required_macros)
        self.assertTrue(br.batches[0].is_cross_file)

    def test_cross_file_defined_macros(self):
        """
        %macro helper is defined in file 1 and called in file 2.
        The macro_invocation edge places both in one cross-file batch
        and defined_macros must list 'helper'.
        """
        br = _batch(
            "%macro helper; data work.x; run; %mend;",
            "%helper;",
        )
        self.assertEqual(len(br.batches), 1)
        self.assertIn("helper", br.batches[0].defined_macros)
        self.assertTrue(br.batches[0].is_cross_file)

    def test_reason_contains_cross_file_tag(self):
        br = _batch(
            "data work.base; set mylib.raw; run;",
            "proc print data=work.base; run;",
        )
        self.assertIn("cross-file", br.batches[0].reason)
        self.assertIn("work.base", br.batches[0].reason)


# ---------------------------------------------------------------------------
# 9. Context absorption — same-file only
# ---------------------------------------------------------------------------


class TestContextAbsorptionMultiFile(unittest.TestCase):
    def test_options_absorbed_same_file(self):
        """OPTIONS chunk in file 1 absorbed into data step in file 1."""
        br = _batch(
            "options mprint;\ndata work.a; set mylib.raw; run;\n"
            "proc print data=work.a; run;",
            "data work.b; run;",
            include_options=True,
        )
        # file 1's three chunks form one batch; file 2 is a singleton
        file1_batch = next(b for b in br.batches if "file_1.sas" in b.source_files)
        kinds = [c.kind.value for c in file1_batch.chunks]
        self.assertIn("OPTIONS", kinds)

    def test_options_not_absorbed_across_files(self):
        """OPTIONS at end of file 1 must NOT be absorbed into chunks in file 2."""
        br = _batch(
            "options mprint;",  # file 1 — only chunk
            "data work.a; set mylib.raw; run;",  # file 2
            include_options=True,
        )
        # No dependency between them → options stays a singleton in file 1
        singleton_sources = [c.source_id for c in br.singletons]
        self.assertIn("file_1.sas", singleton_sources)

    def test_comment_absorbed_same_file_when_enabled(self):
        br = _batch(
            "/* load data */\ndata work.a; set mylib.raw; run;\n"
            "proc print data=work.a; run;",
            include_comments=True,
        )
        # All three (comment + data + proc) should be in one batch
        self.assertEqual(len(br.batches), 1)
        kinds = [c.kind.value for c in br.batches[0].chunks]
        self.assertIn("COMMENT_BLOCK", kinds)


# ---------------------------------------------------------------------------
# 10. SasMultiBatchResult model properties
# ---------------------------------------------------------------------------


class TestMultiBatchResultModel(unittest.TestCase):
    def test_source_ids_preserved(self):
        br = _batch(
            "data work.a; run;",
            "data work.b; run;",
            ids=["alpha.sas", "beta.sas"],
        )
        self.assertEqual(br.source_ids, ["alpha.sas", "beta.sas"])

    def test_cross_file_batches_property(self):
        br = _batch(
            "data work.a; set mylib.raw; run;",
            "proc print data=work.a; run;",
        )
        self.assertEqual(len(br.cross_file_batches), 1)

    def test_cross_file_batches_empty_when_no_cross_file(self):
        br = _batch(
            "data work.a; run;",
            "data work.b; run;",
        )
        self.assertEqual(len(br.cross_file_batches), 0)

    def test_all_ordered_items_covers_all_chunks(self):
        """Every original chunk appears exactly once in all_ordered_items.

        Chunk IDs in batched output carry a file-rank prefix (e.g. 'f1-chunk-0001')
        from the multi-file ID re-stamping pass.  We verify coverage by count,
        uniqueness, and that every batched ID embeds 'chunk-'.
        """
        corp = _corpus(
            "data work.a; set mylib.raw; run;\ndata work.lone; run;",
            "proc print data=work.a; run;",
        )
        br = MultiFileBatcher().batch(corp)
        ordered_ids = _all_chunk_ids(br)
        original_count = len(corp.all_chunks)
        # Same count as original chunks
        self.assertEqual(len(ordered_ids), original_count)
        # Each appears exactly once (globally unique IDs)
        self.assertEqual(len(set(ordered_ids)), original_count)
        # All IDs carry the "chunk-" substring
        self.assertTrue(all("chunk-" in cid for cid in ordered_ids))

    def test_result_json_serialisable(self):
        import json

        br = _batch(
            "data work.a; set mylib.raw; run;",
            "proc print data=work.a; run;",
        )
        json.dumps(br.model_dump())  # must not raise

    def test_is_cross_file_flag_on_batch(self):
        br = _batch(
            "data work.a; set mylib.raw; run;",
            "proc print data=work.a; run;",
        )
        self.assertTrue(br.batches[0].is_cross_file)

    def test_single_file_batch_not_cross_file(self):
        br = _batch(
            "data work.a; set mylib.raw; run;\nproc print data=work.a; run;",
        )
        self.assertFalse(br.batches[0].is_cross_file)

    def test_batch_start_end_line(self):
        br = _batch(
            "data work.a; set mylib.raw; run;",
            "proc print data=work.a; run;",
        )
        batch = br.batches[0]
        self.assertLessEqual(batch.start_line, batch.end_line)


# ---------------------------------------------------------------------------
# 11. Complex realistic multi-file programs
# ---------------------------------------------------------------------------


class TestComplexMultiFile(unittest.TestCase):
    def test_macro_lib_etl_report_three_files(self):
        """
        Classic three-file SAS project layout:
          macros.sas  — shared utility macros
          etl.sas     — data loading + transformation (calls macros)
          reports.sas — reporting (reads ETL outputs)
        """
        macros = (
            "%macro load_csv(path, out);\n"
            "  proc import datafile=&path. dbms=csv out=&out. replace;\n"
            "  run;\n"
            "%mend;\n"
            "%macro clean(ds);\n"
            "  data &ds.; set &ds.; if missing(id) then delete; run;\n"
            "%mend;"
        )
        etl = (
            "%load_csv('/data/orders.csv', work.orders_raw);\n"
            "data work.orders_clean;\n"
            "  set work.orders_raw;\n"
            "  where order_dt >= '01JAN2020'd;\n"
            "run;\n"
            "%clean(work.orders_clean);\n"
            "proc sort data=work.orders_clean; by customer_id; run;\n"
            "proc sort data=mylib.customers;   by customer_id; run;\n"
            "data work.enriched;\n"
            "  merge work.orders_clean(in=a) mylib.customers(in=b);\n"
            "  by customer_id;\n"
            "  if a and b;\n"
            "run;"
        )
        reports = (
            "proc means data=work.enriched noprint;\n"
            "  class region;\n"
            "  var revenue;\n"
            "  output out=work.summary sum=total_rev;\n"
            "run;\n"
            "proc print data=work.enriched; run;\n"
            "proc export data=work.summary outfile='/out/summary.csv' dbms=csv replace;\n"
            "run;"
        )
        br = _batch(macros, etl, reports, ids=["macros.sas", "etl.sas", "reports.sas"])

        # Everything should be in one big cross-file batch
        self.assertGreaterEqual(len(br.batches), 1)
        cross_file = br.cross_file_batches
        self.assertGreater(len(cross_file), 0)

        all_source_files = {f for b in cross_file for f in b.source_files}
        self.assertIn("macros.sas", all_source_files)
        self.assertIn("etl.sas", all_source_files)
        self.assertIn("reports.sas", all_source_files)

        all_outputs = {ds for b in br.batches for ds in b.output_datasets}
        self.assertIn("work.enriched", all_outputs)
        self.assertIn("work.summary", all_outputs)

    def test_proc_sql_join_cross_three_files(self):
        """PROC SQL in file 3 joins tables produced by files 1 and 2."""
        br = _batch(
            "data work.accounts; set mylib.acct_raw; run;",
            "data work.transactions; set mylib.txn_raw; run;",
            "proc sql;\n"
            "  create table work.final as\n"
            "    select a.name, sum(t.amount) as total\n"
            "    from work.accounts as a\n"
            "    join work.transactions as t on a.id = t.acct_id\n"
            "    group by a.name;\n"
            "quit;",
        )
        self.assertEqual(len(br.batches), 1)
        batch = br.batches[0]
        self.assertEqual(len(batch.chunks), 3)
        self.assertTrue(batch.is_cross_file)
        self.assertIn("work.final", batch.output_datasets)

    def test_accumulator_pattern_cross_file(self):
        """File 1 produces sorted data; file 2 does by-group accumulation."""
        br = _batch(
            "data work.sales; set mylib.raw; run;\n"
            "proc sort data=work.sales; by region; run;",
            "data work.region_totals;\n"
            "  set work.sales;\n"
            "  by region;\n"
            "  if first.region then total=0;\n"
            "  total + revenue;\n"
            "  if last.region then output;\n"
            "run;",
        )
        self.assertEqual(len(br.batches), 1)
        self.assertTrue(br.batches[0].is_cross_file)
        self.assertIn("work.region_totals", br.batches[0].output_datasets)


# ---------------------------------------------------------------------------
# 12. MultiFileBatcher.from_files factory
# ---------------------------------------------------------------------------


class TestFromFilesFactory(unittest.TestCase):
    def test_from_files_returns_corpus_and_result(self):
        """from_files reads real disk files and returns (corpus, result)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = pathlib.Path(tmpdir) / "producer.sas"
            f2 = pathlib.Path(tmpdir) / "consumer.sas"
            f1.write_text("data work.base; set mylib.raw; run;\n")
            f2.write_text("proc print data=work.base; run;\n")

            corpus, result = MultiFileBatcher.from_files([str(f1), str(f2)])

        self.assertIsInstance(corpus, SasCorpus)
        self.assertIsInstance(result, SasMultiBatchResult)
        self.assertEqual(len(result.batches), 1)
        self.assertTrue(result.batches[0].is_cross_file)

    def test_from_files_preserves_file_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = pathlib.Path(tmpdir) / "a.sas"
            f2 = pathlib.Path(tmpdir) / "b.sas"
            f3 = pathlib.Path(tmpdir) / "c.sas"
            f1.write_text("data work.x; run;\n")
            f2.write_text("data work.y; set work.x; run;\n")
            f3.write_text("proc print data=work.y; run;\n")

            corpus, result = MultiFileBatcher.from_files(
                [str(f1), str(f2), str(f3)],
            )

        self.assertEqual(result.source_ids, [str(f1), str(f2), str(f3)])
        self.assertEqual(len(result.batches), 1)

    def test_from_files_with_chunker_kwargs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = pathlib.Path(tmpdir) / "big.sas"
            stmts = "\n".join(f"x{i}={i};" for i in range(200))
            f1.write_text(f"data work.big;\n{stmts}\nrun;\n")

            corpus, result = MultiFileBatcher.from_files(
                [str(f1)],
                chunker_kwargs={"max_words": 50},
            )

        # With max_words=50 a 200-statement DATA step should be split
        all_chunks = corpus.all_chunks
        self.assertGreater(len(all_chunks), 1)

    def test_from_files_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            MultiFileBatcher.from_files(["/nonexistent/path.sas"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
