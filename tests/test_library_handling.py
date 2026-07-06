"""
test_library_handling.py — SAS library semantics in chunking and batching.

Covers the library-handling behaviors documented in the SAS Programmer's
Guide: Essentials (Ch. 3 "Words and Names", Ch. 11 "SAS Libraries"):

- one-level dataset names canonicalise to the WORK library, so mixed
  one-level / work.-qualified references to the same dataset batch together;
- LIBNAME statements populate ``defines_librefs``, and batches report the
  librefs they use but do not assign via ``SasBatch.required_librefs``;
- a USER library assignment (``options user=...`` / ``libname user ...``)
  emits the ``USER_LIBRARY_ASSIGNED`` diagnostic;
- implicit _LAST_ flow (PROC with no DATA=, bare ``set;``, literal
  ``_last_``) and the _DATA_ / DATAn naming convention resolve to concrete
  dataset names in the batcher;
- advisory fields: SQL table aliases and BY-group FIRST./LAST. temporaries
  are not misreported as librefs/datasets, and quoted physical-path
  references are tracked as I/O.

Run:  python -m pytest tests/test_library_handling.py -v
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chunker import SasSemanticChunker
from chunker.batcher import SasChunkBatcher

# ── helpers ────────────────────────────────────────────────────────────────


def _chunk_and_batch(source: str, **batcher_kwargs) -> tuple:
    """Return (SasChunkResult, SasBatchResult) for the given SAS source."""
    chunker = SasSemanticChunker(min_words=1, max_words=9_999)
    result = chunker.chunk_text(source, source_id="test.sas")
    batcher = SasChunkBatcher(**batcher_kwargs)
    batch_result = batcher.batch(result)
    return result, batch_result


# ── 1. One-level ↔ two-level WORK equivalence ──────────────────────────────


class TestOneLevelCanonicalization(unittest.TestCase):
    def test_one_level_producer_two_level_consumer(self):
        """data mytable; ≡ work.mytable (guide p. 251) → one batch."""
        src = "data mytable; x=1; run;\nproc means data=work.mytable; run;\n"
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.singletons), 0)
        self.assertIn("work.mytable", br.batches[0].output_datasets)

    def test_two_level_producer_one_level_consumer(self):
        src = "data work.mytable; x=1; run;\nproc means data=mytable; run;\n"
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.singletons), 0)

    def test_one_level_names_reported_canonically(self):
        """Chunk metadata reports one-level names as work.<name>."""
        cr, _ = _chunk_and_batch("data a; set b; run;\n")
        meta = cr.chunks[0].metadata
        self.assertEqual(meta.output_datasets, ["work.a"])
        self.assertEqual(meta.input_datasets, ["work.b"])

    def test_null_still_produces_nothing(self):
        cr, _ = _chunk_and_batch("data _null_; set work.a; run;\n")
        self.assertEqual(cr.chunks[0].metadata.output_datasets, [])

    def test_bare_work_token_still_dropped(self):
        """`set work;` names the reserved bare libref, not a dataset."""
        cr, _ = _chunk_and_batch("data work_summary; set work; run;\n")
        self.assertEqual(cr.chunks[0].metadata.input_datasets, [])

    def test_macro_call_one_level_arg_links_to_two_level_producer(self):
        """Parameterised macro output resolves through canonicalisation."""
        src = (
            "%macro clean(ds); data &ds.; x=1; run; %mend;\n"
            "%clean(orders);\n"
            "proc print data=work.orders; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        self.assertIn("work.orders", br.batches[0].output_datasets)

    def test_macro_body_literal_one_level_canonicalised(self):
        src = "%macro setup; data base; set mylib.raw; run; %mend;\n"
        cr, _ = _chunk_and_batch(src)
        meta = cr.chunks[0].metadata
        self.assertIn("work.base", meta.body_literal_outputs)


# ── 2. LIBNAME tracking and required_librefs ───────────────────────────────


class TestRequiredLibrefs(unittest.TestCase):
    def test_libname_populates_defines_librefs(self):
        cr, _ = _chunk_and_batch("libname mylib 'C:/data';\n")
        self.assertEqual(cr.chunks[0].metadata.defines_librefs, ["mylib"])

    def test_libname_all_clear_defines_nothing(self):
        cr, _ = _chunk_and_batch("libname _all_ clear;\n")
        self.assertEqual(cr.chunks[0].metadata.defines_librefs, [])

    def test_batch_without_libname_reports_required_libref(self):
        src = (
            "data work.b; set mylib.other; run;\n"
            "proc print data=work.b; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(br.batches[0].required_librefs, ["mylib"])

    def test_batch_containing_libname_reports_no_requirement(self):
        src = (
            "libname mylib 'C:/data';\n"
            "data work.a; set mylib.raw; run;\n"
            "proc means data=work.a; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(br.batches[0].required_librefs, [])

    def test_default_libraries_never_required(self):
        src = (
            "data work.a; set sashelp.class; run;\n"
            "proc means data=work.a; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(br.batches[0].required_librefs, [])


# ── 3. USER library diagnostic ─────────────────────────────────────────────


class TestUserLibraryDiagnostic(unittest.TestCase):
    def _diag_codes(self, src: str) -> list[str]:
        cr, _ = _chunk_and_batch(src)
        return [d.code for d in cr.diagnostics]

    def test_options_user_emits_diagnostic(self):
        codes = self._diag_codes("options user=sales;\ndata q1; x=1; run;\n")
        self.assertIn("USER_LIBRARY_ASSIGNED", codes)

    def test_libname_user_emits_diagnostic(self):
        codes = self._diag_codes("libname user '/u/perm';\ndata q1; x=1; run;\n")
        self.assertIn("USER_LIBRARY_ASSIGNED", codes)

    def test_diagnostic_emitted_once(self):
        codes = self._diag_codes(
            "options user=sales;\nlibname user '/u/perm';\ndata q1; x=1; run;\n"
        )
        self.assertEqual(codes.count("USER_LIBRARY_ASSIGNED"), 1)

    def test_no_diagnostic_without_user_assignment(self):
        codes = self._diag_codes(
            "options nodate;\nlibname mylib '/data';\ndata q1; x=1; run;\n"
        )
        self.assertNotIn("USER_LIBRARY_ASSIGNED", codes)


# ── 4. Implicit _LAST_ / _DATA_ resolution ─────────────────────────────────


class TestImplicitLastResolution(unittest.TestCase):
    def test_proc_without_data_reads_last_created(self):
        src = "data work.a; x=1; run;\nproc print; run;\n"
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        self.assertEqual(len(br.singletons), 0)

    def test_non_whitelisted_proc_gets_no_implicit_input(self):
        src = "data work.a; x=1; run;\nproc format; value f 1='x'; run;\n"
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 0)
        self.assertEqual(len(br.singletons), 2)

    def test_proc_without_data_at_file_start_is_singleton(self):
        _, br = _chunk_and_batch("proc print; run;\ndata work.a; x=1; run;\n")
        self.assertEqual(len(br.batches), 0)

    def test_bare_set_reads_last_created(self):
        src = "data work.a; x=1; run;\ndata work.b; set; y=2; run;\n"
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)

    def test_data_data_uses_datan_convention(self):
        src = "data _data_; x=1; run;\ndata work.c; set _last_; run;\n"
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        self.assertIn("work.data1", br.batches[0].output_datasets)

    def test_datan_counter_increments(self):
        src = (
            "data _data_; x=1; run;\n"
            "data _data_; y=2; run;\n"
            "proc print data=work.data2; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        outs = set()
        for b in br.batches:
            outs.update(b.output_datasets)
        self.assertIn("work.data2", outs)


# ── 5. Advisory fields: librefs, aliases, quoted paths ─────────────────────


class TestAdvisoryFields(unittest.TestCase):
    def test_sql_aliases_not_reported_as_librefs(self):
        src = (
            "proc sql;\n"
            "  create table work.j as\n"
            "  select l.id, r.amt from work.left as l\n"
            "  join work.right as r on l.id = r.id;\n"
            "quit;\n"
        )
        cr, _ = _chunk_and_batch(src)
        meta = cr.chunks[0].metadata
        self.assertEqual(meta.referenced_librefs, ["work"])
        self.assertNotIn("l.id", meta.referenced_datasets)

    def test_by_group_first_not_reported_as_libref(self):
        src = "data work.f; set work.s; by grp; if first.grp then n=1; run;\n"
        cr, _ = _chunk_and_batch(src)
        meta = cr.chunks[0].metadata
        self.assertEqual(meta.referenced_librefs, ["work"])
        self.assertNotIn("first.grp", meta.referenced_datasets)

    def test_multi_dataset_headers_fully_reported(self):
        """DATA headers and MERGE lists name several datasets; all appear."""
        src = (
            "data work.cheap work.expensive;\n"
            "  merge work.left work.right;\n"
            "  by id;\n"
            "run;\n"
        )
        cr, _ = _chunk_and_batch(src)
        meta = cr.chunks[0].metadata
        for ds in ("work.cheap", "work.expensive", "work.left", "work.right"):
            self.assertIn(ds, meta.referenced_datasets)

    def test_quoted_path_producer_consumer_link(self):
        src = (
            "data 'c:/tmp/perm'; x=1; run;\n"
            "proc print data='c:/tmp/perm'; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)
        self.assertIn("'c:/tmp/perm'", br.batches[0].output_datasets)

    def test_quoted_path_normalises_quotes_case_and_slashes(self):
        src = (
            'data "C:\\Tmp\\Perm"; x=1; run;\n'
            "proc print data='c:/tmp/perm'; run;\n"
        )
        _, br = _chunk_and_batch(src)
        self.assertEqual(len(br.batches), 1)

    def test_quoted_path_carries_no_libref(self):
        src = "data 'c:/tmp/perm'; x=1; run;\n"
        cr, br = _chunk_and_batch(src)
        self.assertEqual(cr.chunks[0].metadata.referenced_librefs, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
