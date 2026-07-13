# prompt_builder

Reads reference PDFs (SAS language manuals, target-platform guides) and turns
them into retrieval-ready instruction chunks, so the pipeline can inject
guidance relevant to *each* batch/singleton it sends the LLM — targeted per
item instead of one bloated static system prompt.

The package imports nothing from `chunker` or `llm_client`; it reuses
`memory.relevance.HybridRanker` for retrieval. `chunker.pipeline` remains the
sole integration point.

> **Status:** complete. The pipeline injects per-item reference guidance when a
> `PromptBuilder` is passed to `SasLLMPipeline(prompt_builder=...)`.

## Package layout

```
models.py            InstructionDiagnostic, ConstructKey, DocSection,
                     InstructionDoc, InstructionChunk (+ DocRole / ExtractionStrategy)
pdf_reader.py        PdfReader — PDF -> list[DocSection], two strategies
doc_chunker.py       InstructionChunker — DocSection -> word-budgeted InstructionChunk
catalog.py           DocumentSpec + default_catalog + CorpusLoader (on-disk cache)
selector.py          InstructionSelector — construct lookup + HybridRanker retrieval
builder.py           PromptBuilder facade: read -> chunk -> index -> build(query)
user_instructions.py UserInstructionSet — operator rules (str) -> scoped chunks
```

## Quick start

```python
from chunker.pipeline import SasLLMPipeline
from prompt_builder import PromptBuilder

# Load + chunk + index the reference corpus once (cached on disk after run 1).
builder = PromptBuilder.from_reference_dir("reference_docs")

pipeline = SasLLMPipeline(model="claude-haiku-4-5-20251001", prompt_builder=builder)
pipeline.run_file("etl.sas")   # each item's prompt now carries relevant guidance
```

Every batch/singleton the pipeline sends the LLM gains a `## Relevant migration
guidance` block: the reference sections for that item's exact constructs, plus
topically retrieved target-platform chunks. The guidance is **ephemeral** — it
is prompted but never written to the conversation history (it is re-derivable,
would bloat the store, and would skew relevance-based history selection).

### The `reference_docs/` directory

Drop your reference PDFs into `reference_docs/` at the repo root. The directory
is **gitignored** — these are user-provided, copyrighted SAS/O'Reilly manuals
that must never be committed. `default_catalog` recognises the bundled filenames
(the SAS language manuals, the Base ref sheet, and the Spark excerpt) and reads
only the ones actually present, so a partial set works. First load extracts and
caches to `.prompt_builder_cache/` (also gitignored); later loads are ~50×
faster. To index a document not in the default set, build a `DocumentSpec` for
it and pass it via `PromptBuilder.from_specs`.

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

## InstructionChunker

Turns reader sections into retrieval-ready `InstructionChunk`s under a word
budget:

```python
from prompt_builder.doc_chunker import InstructionChunker

chunks = InstructionChunker(min_words=120, max_words=900, overlap_words=60).chunk(
    sections, role=summary.role
)
```

- **Merge.** Consecutive sections under the *same parent heading* whose combined
  text is below `min_words` merge into one chunk (SAS function dictionaries have
  the odd one-line entry). The merged chunk collapses to the shared parent
  breadcrumb and aggregates every member's construct key; a section that already
  meets `min_words` stands alone.
- **Split.** A chunk over `max_words` splits into overlapping windows at
  paragraph boundaries (a single over-long paragraph is hard-split on word
  count). Unlike the SAS chunker there is no parent/child pair — the LLM only
  ever sees the retrieved window, so plain windows suffice.
- **Breadcrumb prefix.** Every chunk's stored text is prefixed with its section
  breadcrumb, so heading terms ("MERGE", "INTNX") weigh on retrieval even when
  the prose below never repeats them. The word budget governs the section
  *body*; the small breadcrumb prefix sits on top of it (the hard token cap is
  `llm_client`'s job at prompt time).

Logger name: `prompt_builder.doc_chunker` (INFO on section→chunk counts and each
oversized split).

## Catalog and extraction cache

`CorpusLoader` reads and chunks a list of `DocumentSpec`s into instruction
chunks, memoised on disk:

```python
from prompt_builder.catalog import default_catalog, CorpusLoader

specs = default_catalog("reference_docs")   # only the files actually present
chunks = CorpusLoader().load(specs)          # cold: reads PDFs; warm: from cache
```

- **`DocumentSpec`** says how to read one document — `path`, `doc_id`, `role`,
  `strategy` (`"auto"`/`"toc"`/`"font"`), `section_level`, and `pinned_sections`
  (used in Phase 6). `default_catalog` ships specs for the bundled
  `reference_docs/` set with per-document TOC depths pinned from each manual's
  structure, and returns only the specs whose file is present (the directory is
  user-provided and untracked).
- **Extraction cache.** Reading + chunking the ~7,400-page corpus costs tens of
  seconds per document and never changes between runs, so each document's chunks
  are cached as JSON under `.prompt_builder_cache/` (gitignored). The cache key
  is the file's SHA-256 plus everything else that affects output — a fingerprint
  of the extractor source itself (`pdf_reader.py` + `doc_chunker.py`, so editing
  the code re-extracts automatically; `EXTRACTOR_VERSION` remains only as a
  manual escape hatch), the spec, and the reader/chunker parameters. A stat
  fast-path trusts the cached SHA when the file's size and mtime are unchanged,
  so a warm load never rehashes a multi-MB PDF. A hit skips PyMuPDF entirely
  (measured ~50× faster than a cold read). Pass `use_cache=False` to bypass.
- **Freshness API.** `check_freshness(spec)` returns
  `fresh | stale | uncached | missing` without extracting;
  `freshness_report(specs)` maps every `doc_id` to its status; and
  `prune_stale(specs)` deletes stale entries, entries whose source PDF is gone,
  and orphaned entries no spec refers to (fresh entries are kept).
- **Unknown PDFs.** `default_catalog(dir, include_unknown=True)` (also exposed
  via `PromptBuilder.from_reference_dir`) gives every unrecognised `*.pdf` a
  generic auto-strategy spec with a slugged `doc_id`, so dropping a new manual
  into the directory is enough to index it.
- **LangChain interop.** `loader.load_documents(specs)` returns the corpus as
  `langchain_core.documents.Document`s (`InstructionChunk.to_document()` /
  `from_document()` round-trip losslessly; construct keys flatten to
  `"kind:name"` strings), for feeding a LangChain vector store / retriever /
  index instead of the built-in selector.

## InstructionSelector

Retrieves the chunks most relevant to one pipeline item, in two stages:

```python
from prompt_builder.selector import InstructionSelector

sel = InstructionSelector(chunks, pinned_sections=["Output Format"])
picks = sel.select(
    query="advance a date to the next month interval",
    constructs=[ConstructKey(kind="function", name="intnx")],
    max_words=1500,
    top_k=6,
)
```

1. **Construct lookup (deterministic).** The item's constructs map straight to
   the reference section documenting each — an exact hit no ranker can beat.
   Hazard-linked constructs (SYMPUT/SYMGET, %GOTO, %ABORT, CALL EXECUTE,
   %SYSFUNC) are fetched first and never stop-listed; a stop-list drops trivial
   ubiquitous functions (PUT, INPUT, SUM, …) so they don't flood the budget.
2. **Hybrid ranking (topical).** `HybridRanker` (BM25 always, dense optional)
   over the whole chunk corpus surfaces guidance no title lookup can find —
   target-platform sections keyed off the free-text query.

Results fill `max_words` in priority order — **pinned → hazard constructs →
other constructs → topical** (at most `top_k` topical chunks) — dropping whole
chunks at the tail, never truncating. Nothing relevant yields an empty list, so
the caller emits no guidance block (irrelevant reference pages are worse than
none). The metadata→`ConstructKey`/query mapping lives in the pipeline (Phase 6)
to keep `prompt_builder` free of any `chunker` import.

### Dense retrieval and the embedding cache

Pass `embeddings=` (a LangChain `Embeddings` or provider string) to add the
dense stage; `DiskCachedEmbeddings` then memoises document vectors to an `.npz`
keyed by content SHA-1 (`embedding_cache_path=`), so embedding the 6–9k-chunk
corpus — the one genuinely expensive step — happens once across runs. It sits
under `HybridRanker`'s in-process cache, so a warm disk cache means no model
call at all. Queries, which vary every call, are never cached.

## PromptBuilder

The facade over the whole package. Load + chunk + index the corpus once
(`PromptBuilder(chunks)`, `PromptBuilder.from_specs(specs)`, or
`PromptBuilder.from_reference_dir(dir)`), then `build(query, constructs)`
returns a Markdown block or `None`:

```
## Relevant migration guidance

### [functions · … > INTNX Function · pp. 1109-1118]
INTNX Function  Increments a date, time, or datetime value …

### [spark_guide · … > DataFrames and SQL · p. 15]
…
```

Keep `max_instruction_words` ≥ the chunker's `max_words` (default 1500 ≥ 900)
so any single reference section always fits — the budget then limits only the
*number* of chunks, dropping whole chunks at the tail, never a lone construct
hit; `from_specs` logs a WARNING when the budget is misconfigured below the
window size. The pipeline builds the `(query, constructs)` for each item from
its SAS metadata (`chunker.pipeline._query_for_item` / `_constructs_for_item`)
— that mapping lives in the pipeline, not here, so this package imports no
`chunker`.

`top_k` and `max_instruction_words` default from `config.json`
(`prompt_builder.*`), as do the chunkers' word budgets — see the `app_config`
package: explicit argument > config.json > hard default.

## User instructions

Operators supply project rules as a plain string (or file) of markdown-ish
sections; each `## heading` opens one instruction and an optional directive
sets its scope:

```markdown
Always target Delta Lake tables, never pandas.        <- preamble: always-on

## Output format                                       <- always-on
One fenced PySpark block per SAS step, then a risk table.

## [when: proc:sql, component_object:hash] Lookup rules  <- construct-scoped
Prefer broadcast joins when the lookup side is small.

## [topic] Partitioning guidance                       <- retrieved by ranking
Wide fact tables are partitioned by load_date.
```

Wire them in at any level: `PromptBuilder(chunks, user_instructions=...)`,
`builder.with_user_instructions(...)` (rebuilds over the same reference
corpus), or `SasLLMPipeline(user_instructions=...)` — which, with no
`prompt_builder`, constructs a corpus-less builder so rules work without any
reference PDFs. When no explicit set is passed, the pipeline auto-loads the
standing file named by `config.json` `user_instructions.path` (missing file =
WARNING and continue).

Selection priority per item: **user always → user construct-matched →
reference pinned → hazard constructs → other constructs → user `[topic]` →
reference topical** — the topical ranking is partitioned so every relevant
user `[topic]` chunk precedes any reference hit, and `top_k` caps the tier as
a whole. Operator rules have first claim on the budget; a rule that doesn't
fit logs a WARNING naming it. `user_instructions.max_words` (config or the
`user_max_words` argument) additionally caps the user block inside the
overall budget. Selected rules render in a `## Project instructions` block
above the reference guidance, with the operator's own headings and no page
citations.

Parsing never raises: unknown directives, malformed construct keys, and
empty-bodied sections emit `InstructionDiagnostic`s and degrade toward
*over*-inclusion (always-on) — an operator rule silently vanishing is the
failure mode this module refuses. Like all guidance, rules are **ephemeral**:
prompted, never persisted to history.

Each set carries a 16-hex content `fingerprint`, exposed as
`SasLLMPipeline.instructions_fingerprint` and recorded into the validation
run history (`instructions_fingerprint` column) — eval runs under different
instructions are never compared as equals.

### Logging — pdf_reader

Logger name: `prompt_builder.pdf_reader`

| Level | When emitted |
|-------|--------------|
| INFO | Per-`read` entry/exit (doc id, section count, strategy, diagnostics) |
| DEBUG | Per-document TOC segmentation summary (boundaries → sections) |
