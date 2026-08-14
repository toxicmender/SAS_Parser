"""
test_notebook.py — unit tests for pipeline.notebook and pipeline.response_models
(zero LLM, zero network).

The notebooks are written by hand against the nbformat v4.5 spec, so the
central test here validates what we emit against the *real* published schema
via the ``nbformat`` package (a dev-extra dependency). Everything else covers
the two ways cells are produced — from a structured TranslationDocument and
from parsed Markdown — and the per-source-file grouping.

Run:  python -m pytest tests/test_notebook.py -v
"""

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pipeline.notebook import (
    CROSS_FILE_NOTEBOOK,
    build_notebook,
    document_to_cells,
    item_cells,
    markdown_to_cells,
    notebook_to_json,
    notebooks_from_outputs,
    write_notebooks,
)
from pipeline.response_models import (
    MappingEntry,
    RiskNote,
    TranslationCell,
    TranslationDocument,
)

# A realistic four-section response, as the unstructured prompt asks for.
MARKDOWN_RESPONSE = """\
## Analysis

The DATA step reads `raw.customers` and filters to active rows. The original
source is:

```sas
data work.customers;
  set raw.customers;
run;
```

## Mapping

- **DATA step** → `DataFrame.filter`

## Translation

First load the source table.

```python
df = spark.table("raw.customers")
```

Then filter it.

```python
active = df.filter(df.status == "A")
```

## Risks

- ⚠️ **P0** — A missing WHERE clause changes the row count silently.
"""


def _document() -> TranslationDocument:
    return TranslationDocument(
        analysis="Reads WORK.A and filters it.",
        mapping=[
            MappingEntry(
                sas_construct="DATA step",
                equivalent="DataFrame.filter",
                difference="DAG, not PDV",
            )
        ],
        cells=[
            TranslationCell(
                kind="code",
                language="python",
                source='df = spark.table("a")',
                comment="Load the source table",
            ),
            TranslationCell(kind="markdown", source="Now filter it."),
            TranslationCell(
                kind="code", language="python", source='df.filter("status = 1")'
            ),
        ],
        risks=[RiskNote(severity="P0", note="Row count may differ.")],
    )


def _output(item_id="item-1", sources=("a.sas",), response="", document=None, **extra):
    out = {
        "item_id": item_id,
        "is_batch": len(sources) > 1,
        "chunk_ids": ["c1"],
        "source_files": list(sources),
        "kind": "DATA_STEP",
        "response": response or MARKDOWN_RESPONSE,
        "document": document,
        "validation": None,
    }
    out.update(extra)
    return out


class TestSchemaConformance(unittest.TestCase):
    """What we emit must satisfy the real nbformat v4.5 schema."""

    def _validate(self, notebook):
        import nbformat

        # read() applies the schema and raises on a violation; going through
        # JSON first proves the emitted *file* is what validates, not just the
        # in-memory dict.
        parsed = nbformat.reads(notebook_to_json(notebook), as_version=4)
        nbformat.validate(parsed)
        return parsed

    def test_structured_notebook_validates(self):
        notebook = build_notebook(
            document_to_cells(_document()), output_language="PySpark"
        )
        self._validate(notebook)

    def test_markdown_notebook_validates(self):
        notebook = build_notebook(
            markdown_to_cells(MARKDOWN_RESPONSE), output_language="SparkSQL"
        )
        self._validate(notebook)

    def test_empty_notebook_validates(self):
        # A run that produced nothing must still write a legal notebook.
        self._validate(build_notebook([], output_language="PySpark"))

    def test_cell_ids_are_unique_and_v45_legal(self):
        notebook = build_notebook(
            document_to_cells(_document()), output_language="PySpark"
        )
        ids = [c["id"] for c in notebook["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for cell_id in ids:
            self.assertRegex(cell_id, r"^[a-zA-Z0-9-_]+$")
            self.assertLessEqual(len(cell_id), 64)

    def test_notebook_is_reproducible(self):
        # Positional ids, not random ones: the same outputs must serialise
        # identically so re-runs diff cleanly.
        first = notebook_to_json(build_notebook(document_to_cells(_document())))
        second = notebook_to_json(build_notebook(document_to_cells(_document())))
        self.assertEqual(first, second)


class TestKernelSelection(unittest.TestCase):
    def test_pyspark_gets_the_python_kernel(self):
        nb = build_notebook([], output_language="PySpark")
        self.assertEqual(nb["metadata"]["kernelspec"]["name"], "python3")
        self.assertEqual(nb["metadata"]["language_info"]["name"], "python")

    def test_sparksql_gets_the_sql_kernel(self):
        nb = build_notebook([], output_language="Spark SQL")
        self.assertEqual(nb["metadata"]["language_info"]["name"], "sql")

    def test_unknown_language_is_rejected(self):
        from target_language import UnknownTargetLanguage

        with self.assertRaises(UnknownTargetLanguage):
            build_notebook([], output_language="Cobol")

    def test_output_language_is_recorded(self):
        nb = build_notebook([], output_language="PySpark")
        self.assertEqual(nb["metadata"]["sas_parser"]["output_language"], "PySpark")

    def test_recorded_output_language_is_canonical_not_the_callers_spelling(self):
        for spelling in ("SparkSQL", "spark sql", "spark_sql"):
            nb = build_notebook([], output_language=spelling)
            self.assertEqual(
                nb["metadata"]["sas_parser"]["output_language"], "Spark SQL"
            )

    def test_untagged_code_cells_take_the_run_target_not_python(self):
        doc = TranslationDocument(
            analysis="x",
            cells=[TranslationCell(kind="code", source="SELECT 1")],
        )
        cells = document_to_cells(doc, output_language="SparkSQL")
        code = [c for c in cells if c["cell_type"] == "code"]
        self.assertEqual(code[0]["metadata"]["language"], "sql")

    def test_untagged_markdown_fences_take_the_run_target(self):
        cells = markdown_to_cells(
            "## Translation\n```\nSELECT 1\n```\n", output_language="SparkSQL"
        )
        code = [c for c in cells if c["cell_type"] == "code"]
        self.assertEqual(code[0]["metadata"]["language"], "sql")

    def test_off_target_code_still_ships_tagged_as_what_it_is(self):
        # The deliverable is already paid for: an off-target block becomes a
        # cell labelled with its real language rather than being dropped.
        # language_compliance is what fails the item.
        cells = markdown_to_cells(
            "## Translation\n```python\ndf = 1\n```\n", output_language="SparkSQL"
        )
        code = [c for c in cells if c["cell_type"] == "code"]
        self.assertEqual(len(code), 1)
        self.assertEqual(code[0]["metadata"]["language"], "python")

    def test_code_cells_carry_their_own_language(self):
        cells = document_to_cells(_document())
        code = [c for c in cells if c["cell_type"] == "code"]
        self.assertTrue(code)
        self.assertEqual(code[0]["metadata"]["language"], "python")


class TestDocumentToCells(unittest.TestCase):
    def test_every_translation_cell_becomes_a_cell_in_order(self):
        cells = document_to_cells(_document())
        kinds = [c["cell_type"] for c in cells]
        # analysis, mapping, comment, code, markdown, code, risks
        self.assertEqual(kinds.count("code"), 2)
        code = [c["source"] for c in cells if c["cell_type"] == "code"]
        self.assertEqual(code, ['df = spark.table("a")', 'df.filter("status = 1")'])

    def test_code_is_not_fenced_inside_the_cell(self):
        cells = document_to_cells(_document())
        for cell in cells:
            if cell["cell_type"] == "code":
                self.assertNotIn("```", cell["source"])

    def test_comment_becomes_a_heading_cell_before_its_code(self):
        cells = document_to_cells(_document())
        sources = [c["source"] for c in cells]
        heading = next(s for s in sources if "Load the source table" in s)
        self.assertLess(
            sources.index(heading), sources.index('df = spark.table("a")')
        )

    def test_empty_document_produces_no_cells(self):
        self.assertEqual(document_to_cells(TranslationDocument(analysis="")), [])


class TestToMarkdown(unittest.TestCase):
    """The rendered Markdown is what memory and the validation metrics see."""

    def test_renders_the_four_sections(self):
        md = _document().to_markdown()
        for heading in ("## Analysis", "## Mapping", "## Translation", "## Risks"):
            self.assertIn(heading, md)

    def test_code_cells_are_fenced_so_the_syntax_metric_finds_them(self):
        from target_language import PYSPARK
        from validation.metrics import _target_blocks

        blocks = [
            b.strip() for b in _target_blocks(_document().to_markdown(), PYSPARK)
        ]
        self.assertEqual(blocks, ['df = spark.table("a")', 'df.filter("status = 1")'])

    def test_untagged_language_defaults_to_python(self):
        doc = TranslationDocument(
            analysis="x", cells=[TranslationCell(kind="code", source="print(1)")]
        )
        self.assertIn("```python", doc.to_markdown())

    def test_untagged_language_can_be_rendered_for_another_target(self):
        # What the pipeline does: pass the run's fence so a Spark SQL
        # translation is not rendered as ```python and then failed for it.
        doc = TranslationDocument(
            analysis="x", cells=[TranslationCell(kind="code", source="SELECT 1")]
        )
        self.assertIn("```sql", doc.to_markdown("sql"))

    def test_p0_risks_are_marked(self):
        self.assertIn("⚠️", _document().to_markdown())

    def test_round_trips_back_through_the_markdown_parser(self):
        cells = markdown_to_cells(_document().to_markdown())
        code = [c["source"] for c in cells if c["cell_type"] == "code"]
        self.assertEqual(code, ['df = spark.table("a")', 'df.filter("status = 1")'])


class TestMarkdownToCells(unittest.TestCase):
    def test_only_translation_fences_become_code(self):
        cells = markdown_to_cells(MARKDOWN_RESPONSE)
        code = [c["source"] for c in cells if c["cell_type"] == "code"]
        self.assertEqual(
            code,
            [
                'df = spark.table("raw.customers")',
                'active = df.filter(df.status == "A")',
            ],
        )

    def test_the_echoed_sas_source_stays_prose(self):
        # A ```sas block in Analysis is the input, not the translation; it must
        # never become a runnable cell.
        cells = markdown_to_cells(MARKDOWN_RESPONSE)
        for cell in cells:
            if cell["cell_type"] == "code":
                self.assertNotIn("set raw.customers", cell["source"])
        analysis = next(c for c in cells if "Analysis" in c["source"])
        self.assertIn("```sas", analysis["source"])

    def test_prose_between_fences_is_preserved(self):
        cells = markdown_to_cells(MARKDOWN_RESPONSE)
        joined = "\n".join(c["source"] for c in cells)
        self.assertIn("First load the source table.", joined)
        self.assertIn("Then filter it.", joined)

    def test_response_without_headings_is_all_translation(self):
        response = "Here you go:\n\n```python\nx = 1\n```\n"
        cells = markdown_to_cells(response)
        code = [c["source"] for c in cells if c["cell_type"] == "code"]
        self.assertEqual(code, ["x = 1"])

    def test_response_with_headings_but_no_translation_section(self):
        # The model deviated; the code must still be reachable rather than
        # silently demoted to prose.
        response = "## Analysis\n\nSome notes.\n\n```python\nx = 1\n```\n"
        code = [
            c["source"] for c in markdown_to_cells(response) if c["cell_type"] == "code"
        ]
        self.assertEqual(code, ["x = 1"])

    def test_heading_inside_a_fence_is_not_a_section(self):
        response = (
            "## Translation\n\n```python\n# ## Mapping is a comment here\n"
            "x = 1\n```\n"
        )
        cells = markdown_to_cells(response)
        code = [c["source"] for c in cells if c["cell_type"] == "code"]
        self.assertEqual(len(code), 1)
        self.assertIn("## Mapping is a comment here", code[0])

    def test_empty_response_produces_no_cells(self):
        self.assertEqual(markdown_to_cells(""), [])


class TestItemCells(unittest.TestCase):
    def test_structured_document_wins_over_the_response_text(self):
        doc = _document()
        cells = item_cells(
            _output(response="## Translation\n\n```python\nWRONG\n```\n",
                    document=doc.model_dump())
        )
        code = [c["source"] for c in cells if c["cell_type"] == "code"]
        self.assertNotIn("WRONG", code)
        self.assertIn('df = spark.table("a")', code)

    def test_falls_back_to_the_response_without_a_document(self):
        code = [
            c["source"] for c in item_cells(_output()) if c["cell_type"] == "code"
        ]
        self.assertEqual(len(code), 2)

    def test_unreadable_document_falls_back_rather_than_raising(self):
        cells = item_cells(_output(document={"cells": "not a list"}))
        code = [c["source"] for c in cells if c["cell_type"] == "code"]
        self.assertEqual(len(code), 2)

    def test_header_cell_carries_the_item_identity_and_verdict(self):
        cells = item_cells(
            _output(
                validation={
                    "passed": False,
                    "score": 0.42,
                    "metrics": [
                        {"metric": "target_syntax", "passed": False, "skipped": False},
                        {"metric": "coverage", "passed": True, "skipped": False},
                    ],
                }
            )
        )
        header = cells[0]["source"]
        self.assertIn("item-1", header)
        self.assertIn("a.sas", header)
        self.assertIn("FAIL", header)
        self.assertIn("0.42", header)
        self.assertIn("target_syntax", header)
        self.assertNotIn("coverage", header)


class TestGrouping(unittest.TestCase):
    def test_one_notebook_per_source_file(self):
        notebooks = notebooks_from_outputs(
            [
                _output("i1", ("dir/a.sas",)),
                _output("i2", ("dir/b.sas",)),
                _output("i3", ("dir/a.sas",)),
            ]
        )
        self.assertEqual(sorted(notebooks), ["a", "b"])
        # Both of a.sas's items landed in the one notebook, in order.
        sources = [c["source"] for c in notebooks["a"]["cells"]]
        self.assertLess(
            next(i for i, s in enumerate(sources) if "i1" in s),
            next(i for i, s in enumerate(sources) if "i3" in s),
        )

    def test_cross_file_batch_goes_to_the_shared_notebook_once(self):
        notebooks = notebooks_from_outputs([_output("i1", ("a.sas", "b.sas"))])
        self.assertIn(CROSS_FILE_NOTEBOOK, notebooks)
        body = "\n".join(
            c["source"] for c in notebooks[CROSS_FILE_NOTEBOOK]["cells"]
        )
        self.assertIn("spark.table", body)
        # The per-file notebooks only point at it.
        for name in ("a", "b"):
            pointer = "\n".join(c["source"] for c in notebooks[name]["cells"])
            self.assertIn(f"{CROSS_FILE_NOTEBOOK}.ipynb", pointer)
            self.assertNotIn("spark.table", pointer)

    # ---- Per-source split of tagged cross-file documents (Phase 5) -------

    @staticmethod
    def _tagged_document():
        return TranslationDocument(
            analysis="Two steps across two files.",
            mapping=[MappingEntry(sas_construct="DATA step", equivalent="DataFrame")],
            cells=[
                TranslationCell(
                    kind="code",
                    language="python",
                    source='df_a = spark.table("a")',
                    chunk_id="a-chunk-1",
                ),
                TranslationCell(kind="markdown", source="Shared note."),
                TranslationCell(
                    kind="code",
                    language="python",
                    source='df_b = df_a.filter("x = 1")',
                    chunk_id="b-chunk-1",
                ),
            ],
            risks=[RiskNote(severity="P1", note="Check the filter.")],
        )

    def _tagged_output(self, document=None, chunk_sources=None):
        return _output(
            "i1",
            ("a.sas", "b.sas"),
            document=(document or self._tagged_document()).model_dump(),
            chunk_sources=(
                {"a-chunk-1": "a.sas", "b-chunk-1": "b.sas"}
                if chunk_sources is None
                else chunk_sources
            ),
        )

    def test_tagged_cross_file_document_splits_per_source(self):
        notebooks = notebooks_from_outputs([self._tagged_output()])
        # Every source gets its own complete notebook; no shared one at all.
        self.assertNotIn(CROSS_FILE_NOTEBOOK, notebooks)
        a = "\n".join(c["source"] for c in notebooks["a"]["cells"])
        b = "\n".join(c["source"] for c in notebooks["b"]["cells"])
        # Code cells route by chunk_id...
        self.assertIn("df_a = spark.table", a)
        self.assertNotIn("df_b", a)
        self.assertIn("df_b = df_a.filter", b)
        self.assertNotIn("spark.table", b)
        # ...while the header, Analysis/Mapping, untagged prose, and Risks are
        # duplicated so each notebook stands alone.
        for body in (a, b):
            self.assertIn("## i1", body)
            self.assertIn("Split: translation cells are routed", body)
            self.assertIn("Two steps across two files.", body)
            self.assertIn("Shared note.", body)
            self.assertIn("Check the filter.", body)

    def test_untagged_code_cell_falls_back_whole_item(self):
        doc = self._tagged_document()
        doc.cells[2].chunk_id = None  # one code cell loses its tag
        notebooks = notebooks_from_outputs([self._tagged_output(document=doc)])
        # All-or-nothing: the WHOLE item keeps the shared-notebook treatment.
        self.assertIn(CROSS_FILE_NOTEBOOK, notebooks)
        shared = "\n".join(
            c["source"] for c in notebooks[CROSS_FILE_NOTEBOOK]["cells"]
        )
        self.assertIn("df_a = spark.table", shared)
        self.assertIn("df_b = df_a.filter", shared)
        for name in ("a", "b"):
            pointer = "\n".join(c["source"] for c in notebooks[name]["cells"])
            self.assertIn(f"{CROSS_FILE_NOTEBOOK}.ipynb", pointer)
            self.assertNotIn("df_a", pointer)

    def test_tag_resolving_outside_the_item_falls_back(self):
        notebooks = notebooks_from_outputs(
            [
                self._tagged_output(
                    chunk_sources={"a-chunk-1": "a.sas", "b-chunk-1": "elsewhere.sas"}
                )
            ]
        )
        self.assertIn(CROSS_FILE_NOTEBOOK, notebooks)

    def test_missing_chunk_sources_falls_back(self):
        out = self._tagged_output()
        del out["chunk_sources"]  # an output produced before Phase 5
        notebooks = notebooks_from_outputs([out])
        self.assertIn(CROSS_FILE_NOTEBOOK, notebooks)

    def test_markdown_fallback_path_keeps_the_shared_notebook(self):
        # No structured document at all: the pre-split behavior, untouched.
        notebooks = notebooks_from_outputs(
            [_output("i1", ("a.sas", "b.sas"), chunk_sources={"c1": "a.sas"})]
        )
        self.assertIn(CROSS_FILE_NOTEBOOK, notebooks)

    def test_split_notebooks_validate_against_the_nbformat_schema(self):
        import nbformat

        for notebook in notebooks_from_outputs([self._tagged_output()]).values():
            parsed = nbformat.reads(notebook_to_json(notebook), as_version=4)
            nbformat.validate(parsed)

    def test_same_basename_in_different_directories_does_not_collide(self):
        notebooks = notebooks_from_outputs(
            [_output("i1", ("x/load.sas",)), _output("i2", ("y/load.sas",))]
        )
        self.assertEqual(sorted(notebooks), ["load", "load_2"])

    def test_item_without_a_source_file_lands_in_the_shared_notebook(self):
        notebooks = notebooks_from_outputs([_output("i1", ())])
        self.assertEqual(list(notebooks), [CROSS_FILE_NOTEBOOK])

    def test_unsafe_source_id_is_sanitised(self):
        notebooks = notebooks_from_outputs([_output("i1", ("a b/we:ird?.sas",))])
        name = next(iter(notebooks))
        self.assertRegex(name, r"^[A-Za-z0-9._-]+$")


class TestWriteNotebooks(unittest.TestCase):
    def test_writes_one_file_per_notebook(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = write_notebooks(
                [_output("i1", ("a.sas",)), _output("i2", ("b.sas",))],
                tmp,
                output_language="PySpark",
            )
            self.assertEqual(
                sorted(p.name for p in written), ["a.ipynb", "b.ipynb"]
            )
            for path in written:
                notebook = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(notebook["nbformat_minor"], 5)

    def test_creates_a_missing_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = pathlib.Path(tmp) / "deep" / "nested"
            write_notebooks([_output()], dest)
            self.assertTrue((dest / "a.ipynb").is_file())


class TestMixedLanguageNotebook(unittest.TestCase):
    """A Spark SQL run whose items fell back to PySpark still has to run.

    The kernel is the constraint: Python can host SQL through the %sql magic,
    a SQL kernel cannot host Python at all.
    """

    def _cells(self, *specs):
        """One output per (language, source) pair, as the pipeline emits them."""
        return [
            _output(
                item_id=f"item-{i}",
                document={
                    "analysis": "",
                    "cells": [{"kind": "code", "language": lang, "source": src}],
                },
                target_language="PySpark" if lang == "python" else "Spark SQL",
                fallback_reasons=(
                    ["kind:MACRO_DEFINITION rates MANUAL ... and HARD ..."]
                    if lang == "python"
                    else []
                ),
            )
            for i, (lang, src) in enumerate(specs)
        ]

    def _notebook(self, *specs):
        return notebooks_from_outputs(
            self._cells(*specs), output_language="Spark SQL"
        )["a"]

    def _code_cells(self, notebook):
        return [c for c in notebook["cells"] if c["cell_type"] == "code"]

    def test_a_mixed_notebook_is_hosted_by_the_python_kernel(self):
        notebook = self._notebook(("sql", "SELECT 1"), ("python", "x = 1"))
        self.assertEqual(notebook["metadata"]["kernelspec"]["name"], "python3")
        self.assertEqual(notebook["metadata"]["language_info"]["name"], "python")

    def test_sql_cells_in_a_mixed_notebook_carry_the_magic(self):
        notebook = self._notebook(("sql", "SELECT 1"), ("python", "x = 1"))
        sql_cell, py_cell = self._code_cells(notebook)
        self.assertEqual(sql_cell["source"], "%sql\nSELECT 1")
        # The Python cell is untouched — the magic is only how SQL is hosted.
        self.assertEqual(py_cell["source"], "x = 1")
        # Per-cell language metadata still describes what each cell really is.
        self.assertEqual(sql_cell["metadata"]["language"], "sql")
        self.assertEqual(py_cell["metadata"]["language"], "python")

    def test_an_all_sql_notebook_is_unchanged(self):
        """The common path must not move: no magic, no kernel switch."""
        notebook = self._notebook(("sql", "SELECT 1"), ("sql", "SELECT 2"))
        self.assertEqual(notebook["metadata"]["kernelspec"]["name"], "sql")
        self.assertEqual(
            [c["source"] for c in self._code_cells(notebook)],
            ["SELECT 1", "SELECT 2"],
        )

    def test_an_all_python_notebook_is_unchanged(self):
        notebook = self._notebook(("python", "x = 1"))
        self.assertEqual(notebook["metadata"]["kernelspec"]["name"], "python3")
        self.assertEqual([c["source"] for c in self._code_cells(notebook)], ["x = 1"])

    def test_the_fallback_reason_reaches_the_header_cell(self):
        notebook = self._notebook(("sql", "SELECT 1"), ("python", "x = 1"))
        markdown = "\n".join(
            c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown"
        )
        self.assertIn("**Translated to PySpark**", markdown)
        self.assertIn("MACRO_DEFINITION", markdown)

    def test_a_mixed_notebook_still_validates(self):
        import nbformat

        notebook = self._notebook(("sql", "SELECT 1"), ("python", "x = 1"))
        nbformat.validate(nbformat.reads(notebook_to_json(notebook), as_version=4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
