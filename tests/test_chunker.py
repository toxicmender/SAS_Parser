"""
test_chunker.py — unit tests for SasSemanticChunker (zero LLM, zero disk I/O).

Run:  python -m pytest tests/test_chunker.py -v
"""

import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chunker import SasChunkKind, SasSemanticChunker


class TestSasSemanticChunker(unittest.TestCase):
    # ── DATA step ─────────────────────────────────────────────────────────────

    def test_data_step_basic_offsets(self):
        source = (
            "/* head */\ndata work.out;\n set lib.in;\n x='proc sort; run;';\nrun;\n"
        )
        result = SasSemanticChunker().chunk_text(source, source_id="example.sas")

        self.assertEqual(result.chunks[0].kind, SasChunkKind.COMMENT_BLOCK)
        data_chunk = result.chunks[1]
        self.assertEqual(data_chunk.kind, SasChunkKind.DATA_STEP)
        # char offsets must reconstruct original text
        self.assertEqual(
            source[data_chunk.start_char : data_chunk.end_char], data_chunk.text
        )
        self.assertEqual(data_chunk.start_line, 2)
        self.assertIn("lib.in", data_chunk.metadata.referenced_datasets)
        self.assertIn("lib", data_chunk.metadata.referenced_librefs)

    def test_data_step_multiple_output_tables(self):
        source = "data work.cheap work.expensive;\n set work.shopping;\n if price > 100 then output work.expensive;\n else output work.cheap;\nrun;\n"
        result = SasSemanticChunker().chunk_text(source)
        self.assertEqual(len(result.chunks), 1)
        chunk = result.chunks[0]
        self.assertEqual(chunk.kind, SasChunkKind.DATA_STEP)
        self.assertIn("work.cheap", chunk.metadata.referenced_datasets)
        self.assertIn("work.expensive", chunk.metadata.referenced_datasets)

    def test_data_step_keep_drop_by_groups(self):
        source = (
            "data work.expensive (keep = price item_name);\n"
            " set work.shopping (drop = city state);\n"
            " by category;\n"
            " if first.category then total = 0;\n"
            " total + price;\n"
            "run;\n"
        )
        result = SasSemanticChunker().chunk_text(source)
        chunk = result.chunks[0]
        self.assertEqual(chunk.kind, SasChunkKind.DATA_STEP)
        self.assertIn("work.shopping", chunk.metadata.referenced_datasets)

    # ── PROC step ────────────────────────────────────────────────────────────

    def test_proc_step_quit_terminator(self):
        source = "proc sql;\ncreate table work.out as select * from work.in;\nquit;\n"
        result = SasSemanticChunker().chunk_text(source)
        self.assertEqual(len(result.chunks), 1)
        chunk = result.chunks[0]
        self.assertEqual(chunk.kind, SasChunkKind.PROC_STEP)
        self.assertEqual(chunk.metadata.proc_name, "sql")
        self.assertIn("work.in", chunk.metadata.referenced_datasets)

    def test_proc_sort_nodupkey(self):
        source = (
            "proc sort data=work.class out=work.sorted nodupkey;\n by name;\nrun;\n"
        )
        result = SasSemanticChunker().chunk_text(source)
        chunk = result.chunks[0]
        self.assertEqual(chunk.kind, SasChunkKind.PROC_STEP)
        self.assertEqual(chunk.metadata.proc_name, "sort")

    def test_proc_means_class(self):
        source = "proc means data=mylib.sales;\n class region;\n var revenue;\nrun;\n"
        result = SasSemanticChunker().chunk_text(source)
        self.assertEqual(result.chunks[0].kind, SasChunkKind.PROC_STEP)
        self.assertIn("mylib", result.chunks[0].metadata.referenced_librefs)

    def test_proc_import(self):
        source = 'proc import datafile="/tmp/file.csv" dbms=csv out=work.raw replace;\n guessingrows=100;\nrun;\n'
        result = SasSemanticChunker().chunk_text(source)
        self.assertEqual(result.chunks[0].kind, SasChunkKind.PROC_STEP)
        self.assertEqual(result.chunks[0].metadata.proc_name, "import")

    # ── MACRO ────────────────────────────────────────────────────────────────

    def test_macro_definition_and_call(self):
        source = "%macro build(ds);\ndata &ds.;\nrun;\n%mend;\n%build(work.out);\n%let x=1;\n"
        result = SasSemanticChunker().chunk_text(source)

        self.assertEqual(result.chunks[0].kind, SasChunkKind.MACRO_DEFINITION)
        self.assertIn("build", result.chunks[0].metadata.defined_macros)
        self.assertEqual(result.chunks[1].kind, SasChunkKind.MACRO_CALL)
        self.assertIn("build", result.chunks[1].metadata.called_macros)
        self.assertEqual(result.chunks[2].kind, SasChunkKind.GLOBAL_STATEMENT)

    def test_unclosed_macro_diagnostic(self):
        result = SasSemanticChunker().chunk_text("%macro broken;\ndata a;\nrun;\n")
        codes = {d.code for d in result.diagnostics}
        self.assertIn("UNCLOSED_MACRO", codes)

    # ── GLOBAL statements ────────────────────────────────────────────────────

    def test_global_statements_libname_options_title(self):
        source = (
            "%include 'common.sas';\n"
            "libname raw '/data/raw';\n"
            "filename out '/tmp/a';\n"
            "options mprint symbolgen validvarname=v7;\n"
            "title 'Demo';\n"
            "footnote 'Annual refresh';\n"
        )
        result = SasSemanticChunker().chunk_text(source)
        kinds = [c.kind for c in result.chunks]
        self.assertEqual(kinds[0], SasChunkKind.INCLUDE)
        self.assertEqual(kinds[1], SasChunkKind.GLOBAL_STATEMENT)  # libname
        self.assertEqual(kinds[2], SasChunkKind.GLOBAL_STATEMENT)  # filename
        self.assertEqual(kinds[3], SasChunkKind.OPTIONS)
        self.assertEqual(kinds[4], SasChunkKind.GLOBAL_STATEMENT)  # title
        self.assertEqual(kinds[5], SasChunkKind.GLOBAL_STATEMENT)  # footnote
        self.assertEqual(result.chunks[0].metadata.includes, ["common.sas"])

    def test_libname_clear(self):
        source = "libname myxl xlsx '/data/class.xlsx';\ndata work.tmp;\n set myxl.sheet1;\nrun;\nlibname myxl clear;\n"
        result = SasSemanticChunker().chunk_text(source)
        kinds = [c.kind for c in result.chunks]
        self.assertIn(SasChunkKind.GLOBAL_STATEMENT, kinds)
        self.assertIn(SasChunkKind.DATA_STEP, kinds)

    # ── comment / string edge cases ──────────────────────────────────────────

    def test_semicolons_inside_strings_and_comments_not_split(self):
        source = (
            "/* proc print; run; */\ndata a;\nx=\"run; quit;\";\ny='data z;';\nrun;\n"
        )
        result = SasSemanticChunker().chunk_text(source)
        self.assertEqual(result.chunks[0].kind, SasChunkKind.COMMENT_BLOCK)
        self.assertEqual(result.chunks[1].kind, SasChunkKind.DATA_STEP)
        self.assertIn('x="run; quit;";', result.chunks[1].text)

    def test_star_comment_inside_data_step(self):
        source = "data work.a;\n* this is a comment;\n x = 1;\nrun;\n"
        result = SasSemanticChunker().chunk_text(source)
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].kind, SasChunkKind.DATA_STEP)

    def test_unclosed_block_comment_diagnostic(self):
        result = SasSemanticChunker().chunk_text("/* never closed\ndata a;\nrun;\n")
        codes = {d.code for d in result.diagnostics}
        self.assertIn("UNCLOSED_BLOCK_COMMENT", codes)

    # ── merge / combining data ────────────────────────────────────────────────

    def test_merge_datasets_detected(self):
        source = (
            "data work.combined;\n"
            " merge work.left (in=a) work.right (in=b);\n"
            " by id;\n"
            " if a and b;\n"
            "run;\n"
        )
        result = SasSemanticChunker().chunk_text(source)
        meta = result.chunks[0].metadata
        self.assertIn("work.left", meta.referenced_datasets)
        self.assertIn("work.right", meta.referenced_datasets)

    # ── format / label ───────────────────────────────────────────────────────

    def test_proc_format(self):
        source = "proc format;\n value yesno 1='Yes' 0='No';\nrun;\n"
        result = SasSemanticChunker().chunk_text(source)
        self.assertEqual(result.chunks[0].kind, SasChunkKind.PROC_STEP)
        self.assertEqual(result.chunks[0].metadata.proc_name, "format")

    def test_format_statement_outside_step(self):
        source = "format dob mmddyy10. salary dollar13.2;\n"
        result = SasSemanticChunker().chunk_text(source)
        self.assertEqual(result.chunks[0].kind, SasChunkKind.FORMAT_OR_INFORMAT)

    def test_label_extracted(self):
        source = "data work.out;\n set work.in;\n label height = 'Height (cm)';\nrun;\n"
        result = SasSemanticChunker().chunk_text(source)
        self.assertIn("height", result.chunks[0].metadata.labels)

    # ── ODS / PROC EXPORT ────────────────────────────────────────────────────

    def test_proc_export(self):
        source = (
            'proc export data=work.out outfile="/tmp/out.csv" dbms=csv replace;\nrun;\n'
        )
        result = SasSemanticChunker().chunk_text(source)
        self.assertEqual(result.chunks[0].kind, SasChunkKind.PROC_STEP)
        self.assertEqual(result.chunks[0].metadata.proc_name, "export")

    # ── oversized splitting ───────────────────────────────────────────────────

    def test_oversized_block_splits_with_parent_id(self):
        stmts = "\n".join(f"x{i} = {i};" for i in range(900))
        source = f"data work.big;\n{stmts}\nrun;\n"
        result = SasSemanticChunker(max_words=120).chunk_text(source)
        self.assertGreater(len(result.chunks), 1)
        self.assertTrue(all(c.kind == SasChunkKind.DATA_STEP for c in result.chunks))
        # The first chunk IS the parent (no parent_id); all children reference it
        parent = result.chunks[0]
        children = result.chunks[1:]
        self.assertIsNone(parent.parent_id)
        self.assertTrue(all(c.parent_id == parent.chunk_id for c in children))

    # ── diagnostics ──────────────────────────────────────────────────────────

    def test_unclosed_data_step_diagnostic(self):
        result = SasSemanticChunker().chunk_text("data a;\nx=1;\n")
        self.assertTrue(result.chunks[0].metadata.has_unclosed_block)
        codes = {d.code for d in result.diagnostics}
        self.assertIn("UNCLOSED_DATA_OR_PROC_STEP", codes)

    def test_unrecognised_source_fallthrough(self):
        source = "some vendor syntax here;\nmore odd syntax;\ndata a;\nrun;\n"
        result = SasSemanticChunker().chunk_text(source)
        self.assertEqual(result.chunks[0].kind, SasChunkKind.UNKNOWN_STATEMENT_GROUP)
        self.assertEqual(result.chunks[1].kind, SasChunkKind.DATA_STEP)
        codes = {d.code for d in result.diagnostics}
        self.assertIn("UNRECOGNIZED_SOURCE_REGION", codes)

    def test_unterminated_unknown_is_unknown_block(self):
        result = SasSemanticChunker().chunk_text("vendor thing without semicolon")
        self.assertEqual(result.chunks[0].kind, SasChunkKind.UNKNOWN_BLOCK)
        self.assertTrue(result.chunks[0].metadata.has_unclosed_block)

    # ── serialisation ────────────────────────────────────────────────────────

    def test_result_is_json_serialisable(self):
        result = SasSemanticChunker().chunk_text("data a; run;", source_id="inline")
        json.dumps(result.model_dump())  # must not raise

    def test_chunk_ids_are_sequential(self):
        source = "data a; run;\nproc print data=a; run;\n%let x=1;\n"
        result = SasSemanticChunker().chunk_text(source)
        ids = [c.chunk_id for c in result.chunks]
        expected = [f"chunk-{i:04d}" for i in range(1, len(ids) + 1)]
        self.assertEqual(ids, expected)

    # ── date / numeric SAS-specific constructs ───────────────────────────────

    def test_where_date_constant(self):
        """WHERE with SAS date literal should not confuse the chunker."""
        source = (
            "data work.recent;\n"
            " set mylib.events;\n"
            " where event_dt >= '01JAN2020'd;\n"
            "run;\n"
        )
        result = SasSemanticChunker().chunk_text(source)
        self.assertEqual(result.chunks[0].kind, SasChunkKind.DATA_STEP)

    def test_do_loop_retained_in_single_chunk(self):
        source = (
            "data work.compound;\n"
            " set work.principal;\n"
            " do year = 1 to 10;\n"
            "  balance = balance * 1.05;\n"
            "  output;\n"
            " end;\n"
            "run;\n"
        )
        result = SasSemanticChunker().chunk_text(source)
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].kind, SasChunkKind.DATA_STEP)


if __name__ == "__main__":
    unittest.main(verbosity=2)
