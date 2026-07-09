# prompt_builder

Reads reference PDFs (SAS language manuals, target-platform guides) and turns
them into retrieval-ready instruction chunks, so the pipeline can inject
guidance relevant to *each* batch/singleton it sends the LLM — targeted per
item instead of one bloated static system prompt.

The package imports nothing from `chunker` or `llm_client`; it reuses
`memory.relevance.HybridRanker` for retrieval. `chunker.pipeline` remains the
sole integration point.

> **Status:** built incrementally. **Phase 2** (this commit) ships the data
> models and the PDF reader. Chunking, indexing/selection, and pipeline wiring
> land in later phases.

## Package layout

```
models.py       InstructionDiagnostic, ConstructKey, DocSection,
                InstructionDoc, InstructionChunk (+ DocRole / ExtractionStrategy)
pdf_reader.py   PdfReader — PDF -> list[DocSection], two strategies
doc_chunker.py  (Phase 3) DocSection -> word-budgeted InstructionChunk
selector.py     (Phase 5) construct lookup + HybridRanker retrieval
builder.py      (Phase 6) PromptBuilder facade: read -> chunk -> index -> build
```

## PdfReader

```python
from prompt_builder.pdf_reader import PdfReader
from prompt_builder.models import DocRole

reader = PdfReader()

# SAS manual: segment on the PDF's own table of contents
summary, sections = reader.read(
    "reference_docs/SAS_Functions_and_Call_Routines.pdf",
    doc_id="functions",
    section_level=4,          # None auto-picks the most populated TOC level
)

# Target guide with no usable TOC: segment on font-size heading heuristics
summary, sections = reader.read(
    "reference_docs/Apache-Spark-The-Definitive-Guide-Excerpts-R1.pdf",
    doc_id="spark",
    role=DocRole.TARGET_GUIDE,
    strategy="auto",          # TOC when present, else font
)
```

`read` returns an `InstructionDoc` summary (page count, chosen strategy,
diagnostics) and a list of `DocSection`s. Each section carries a breadcrumb
`section_path` (`"Dictionary of Functions > INTNX Function"`), its page span,
the cleaned body `text`, and — for parseable SAS titles — a `construct_key`.

### Two strategies

- **TOC** (`ExtractionStrategy.TOC`): segments on `doc.get_toc()` at a chosen
  depth. Section text is sliced between a heading's position in the page text
  and the next heading's, so two sections sharing a page split correctly.
  Front/back matter (Contents, About This Book, Syntax Conventions, Index, …)
  is dropped by title. Ideal for the SAS manuals, whose deep TOCs give one leaf
  entry per function/statement/PROC.
- **Font** (`ExtractionStrategy.FONT`): for documents with no usable TOC. The
  modal span size is the body size; lines at least `min_body_ratio`× larger are
  headings, tiered into levels by distinct size. Falls back to one section per
  page (`ExtractionStrategy.PAGE`) when no heading tier is found.

### Construct keys

SAS reference section titles are parsed into `ConstructKey(kind, name)` lookup
keys so the selector (Phase 5) can match a pipeline item's constructs to the
exact reference section — `"INTNX Function"` → `function:intnx`, `"%LET
Statement"` → `macro_statement:let`, `"The SQL Procedure"` → `proc:sql`,
`"CALL SYMPUT Routine"` → `call_routine:symput`.

### Text cleanup

Applied to every section body (and heading): NFKC folding, straightened
curly quotes/dashes, dropped replacement characters, de-hyphenation across line
breaks, collapsed blank-line runs, and removal of running headers/footers and
bare page numbers that repeat across pages.

### Graceful degradation

Like the SAS chunker, the reader never raises on a malformed document; it emits
`InstructionDiagnostic`s and returns what it recovered:

| Code | When emitted |
|------|--------------|
| `NO_TOC` | TOC strategy requested but the PDF has none (falls back to font) |
| `NO_HEADINGS_DETECTED` | Font strategy found no heading tier (page fallback) |
| `NO_TEXT_LAYER` | Over half the pages have no extractable text (scanned?) |
| `EMPTY_DOCUMENT` | The PDF has no pages |

No OCR and no table-structure extraction in v1 — every bundled reference PDF has
a clean text layer.

### Logging

Logger name: `prompt_builder.pdf_reader`

| Level | When emitted |
|-------|--------------|
| INFO | Per-`read` entry/exit (doc id, section count, strategy, diagnostics) |
| DEBUG | Per-document TOC segmentation summary (boundaries → sections) |
