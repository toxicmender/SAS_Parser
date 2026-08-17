# prompt_builder

Reads reference PDFs (SAS language manuals, target-platform guides) and turns
them into retrieval-ready instruction chunks, so the pipeline can inject
guidance relevant to *each* batch/singleton it sends the LLM — targeted per
item instead of one bloated static system prompt.

The package imports nothing from `chunker` or `llm_client`; it reuses
`memory.relevance.HybridRanker` for retrieval and the leaf `token_budget` for
counting. `pipeline.engine` remains the sole integration point.

Budgets here are **tokens**, the currency the prompt is priced in. Words were
a stand-in and a poor one: over the bundled corpus the tokens-per-word ratio
runs 1.01 to 4.41 (median 1.44), and for the SQL-heavy instruction set it is
1.72 — so a word budget believed it had ~40% more room than it did, and
dropped chunks later and less predictably than its numbers suggested. Counting
lives in `token_budget` rather than `llm_client.tokens` because importing the
latter executes `llm_client/__init__.py` and drags in langchain (7.5s, 1,642
modules) — see `token_budget/README.md`.

> **Status:** complete. The pipeline injects per-item reference guidance when a
> `PromptBuilder` is passed to `SasLLMPipeline(prompt_builder=...)`.

## Package layout

```
models.py            InstructionDiagnostic, ConstructKey, DocSection,
                     InstructionDoc, InstructionChunk, SelectedInstruction
                     (+ DocRole / ExtractionStrategy / SelectionTier)
pdf_reader.py        PdfReader — PDF -> list[DocSection], two strategies
doc_chunker.py       InstructionChunker — DocSection -> token-budgeted InstructionChunk
catalog.py           DocumentSpec + default_catalog + CorpusLoader (on-disk cache)
selector.py          InstructionSelector — construct lookup + HybridRanker retrieval
builder.py           PromptBuilder facade: read -> chunk -> index -> build(query)
user_instructions.py UserInstructionSet — operator rules (str) -> scoped chunks
```

## Quick start

```python
from llm_client import LLMClientConfig
from pipeline import SasLLMPipeline
from prompt_builder import PromptBuilder

# Load + chunk + index the reference corpus once (cached on disk after run 1).
builder = PromptBuilder.from_reference_dir("reference_docs")

pipeline = SasLLMPipeline(
    llm_config=LLMClientConfig(model="claude-sonnet-4-5"), prompt_builder=builder
)
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

The bundled catalog covers the local SAS programmer's guide, DATA-step
statements, functions and CALL routines, formats and informats, global
statements, macro language, procedures, component objects, FedSQL, metadata
interfaces, and the Base reference sheet. It also covers the Spark excerpt and
selects only migration-relevant TOC sections from the much larger Azure
Databricks reference.

The coverage taxonomy is checked against SAS's programming-documentation
indexes: [language elements by name, product, and category](https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.5/allprodsle/titlepage.htm),
[procedures by name and product](https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.5/allprodsproc/procedures.htm),
and [CAS actions and action sets by name and product](https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.5/allprodsactions/actionsByName.htm).
These indexes define what must be recognised; the local PDFs supply the text
retrieved into an item's runtime prompt.

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

Turns reader sections into retrieval-ready `InstructionChunk`s under a token
budget, and records each chunk's `token_count` so the selector never has to
re-tokenise the corpus (counting it costs ~3.9s against ~0.18s for words, and
rides the on-disk cache):

```python
from prompt_builder.doc_chunker import InstructionChunker

chunks = InstructionChunker(min_tokens=175, max_tokens=1300, overlap_tokens=90).chunk(
    sections, role=summary.role
)
```

- **Merge.** Consecutive sections under the *same parent heading* whose combined
  text is below `min_tokens` merge into one chunk (SAS function dictionaries have
  the odd one-line entry). The merged chunk collapses to the shared parent
  breadcrumb and aggregates every member's construct key; a section that already
  meets `min_tokens` stands alone.
- **Split.** A chunk over `max_tokens` splits into overlapping windows at
  paragraph boundaries (a single over-long paragraph is hard-split on word
  boundaries, stepped by its own token density). Unlike the SAS chunker there is no parent/child pair — the LLM only
  ever sees the retrieved window, so plain windows suffice.
- **Breadcrumb prefix.** Every chunk's stored text is prefixed with its section
  breadcrumb, so heading terms ("MERGE", "INTNX") weigh on retrieval even when
  the prose below never repeats them. The token budget governs the section
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
    max_tokens=8000,
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

Results fill `max_tokens` in priority order — **pinned → hazard constructs →
other constructs → topical** (at most `top_k` topical chunks) — dropping whole
chunks at the tail, never truncating. Nothing relevant yields an empty list, so
the caller emits no guidance block (irrelevant reference pages are worse than
none). The metadata→`ConstructKey`/query mapping lives in the pipeline (Phase 6)
to keep `prompt_builder` free of any `chunker` import.

`select_detailed(...)` returns the same picks with provenance — each
`SelectedInstruction` carries the `SelectionTier` that claimed it
(`user_always | user_when | pinned | hazard | construct | user_topic |
topical`) and, for construct-lookup tiers, the matched `ConstructKey` — so
formatting can treat picks differently by tier (the builder's focus hints are
built on this). `select()` is `select_detailed()` minus the provenance.

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
## Focus hints

- ⚠️ Hazards to address explicitly: CALL SYMPUT routine — run-time
  macro-variable write; scope/timing differs from %LET
- Constructs to map: INTNX function, PROC SQL
- Related reference topics: DataFrames and SQL

## Reasoning directives

Before writing the translation, in your Analysis:
- Trace step by step when each macro variable is written versus read …

## Relevant migration guidance

### [functions · … > INTNX Function · pp. 1109-1118 · construct: intnx]
INTNX Function  Increments a date, time, or datetime value …

### [spark_guide · … > DataFrames and SQL · p. 15 · topical]
…
```

Each reference chunk's header ends with its **selection reason** —
`construct: <name>` / `hazard: <name>` for deterministic lookup hits,
`pinned`, or `topical` — so the model can weigh an authoritative match for
the item's exact construct above a merely related retrieved section.

### Retrieval and rendering, separately

`build()` is `select()` + `build_from_picks()`, and both halves are public:

```python
picks = builder.select(query, constructs, output_language="PySpark")
block = builder.build_from_picks(picks, constructs)   # == builder.build(...)
```

`select()` returns the `SelectedInstruction`s **in priority order** — the
ranking `InstructionSelector.select_detailed` produced. The pipeline uses the
pair to keep both artefacts from one retrieval: the Markdown block goes into
the prompt, and the chunk texts ride out on each output dict as that item's
`retrieval_context`, which is what `validation`'s RAG metrics
(`faithfulness`, `contextual_precision`, `contextual_relevancy`) score. Note
that `build_from_picks` takes the **item's** constructs, not the selection's:
the hazard hints and reasoning directives are keyed on them so they survive
when no reference section matched.

### Member attribution (`attribution=`)

An item is a *batch*, and a batch spanning a macro, a PROC SQL, a DATA step and
a PROC SORT gets one guidance block covering all four. Passing `attribution` — a
`{ConstructKey: [member id, ...]}` map — labels each section with the member
whose constructs brought it in:

```
### MERGE and BY-group joins  [chunk-0002]
### PROC SORT  [chunk-0003]
### Output format
```

so the model is not left to re-derive from the member bodies what the batch
summary already knows. Reference sections carry the id inside their existing
citation run (`### [functions · … · construct: intnx · chunk-0002]`).

Only picks with a matched construct key are labelled. An always-on rule, a
pinned or topical section, and a `[kind:]`/`[meta:]`-gated note have no key and
are batch-wide by nature, so they stay unlabelled — that is the correct reading,
not a missing label. `None` (the default) renders exactly the unlabelled block,
which is what a single-member batch should pass, since there every label would
name the same chunk.

The ids are opaque strings: `pipeline.prompting._attribution_for_item` builds the
map by running the same construct-key derivation over each member, which is what
keeps this package free of any `chunker` import.

### Focus hints (directional stimulus)

The `## Focus hints` block is a compact per-item stimulus — the item's
hazards (with a one-line caution each), matched constructs (as readable
labels: `PROC SQL`, `CALL SYMPUT routine`, `hash object`), and the leaf
titles of topically retrieved sections — rendered as explicit keywords the
response should address, above the reference guidance. Hazards are listed
from the *item's* constructs even when no reference section matched;
construct and topic lines come from the selection, so they are already
stop-list-filtered and budget-bounded. The block is skipped when there is
nothing to hint at (e.g. a pinned-only selection), and `focus_hints=False`
(or config.json `prompt_builder.focus_hints`) disables it entirely. Hint
lines are small and sit outside the token budget, like breadcrumb prefixes.

### Reasoning directives (conditional chain-of-thought)

The `## Reasoning directives` block carries one imperative step-by-step
reasoning instruction per hazard construct the item uses (SYMPUT/SYMPUTX/
SYMGET, CALL EXECUTE, %GOTO, %ABORT, %SYSFUNC) — e.g. SYMPUT → *"Trace step
by step when each macro variable is written versus read …"*. Chain-of-thought
costs tokens, so it is triggered per item, only where the known silent-error
failure modes live; hazard-free items get no block. Like the hazard hint
line, directives key off the *item's* constructs, so they survive even when
no reference section matched. Disable with `reasoning_directives=False` or
config.json `prompt_builder.reasoning_directives`. The directives complement
the pipeline's system prompt, which scaffolds every response as
**Analysis → Mapping → Translation → Risks** and scopes conciseness to the
final sections so reasoning isn't suppressed.

Keep `max_instruction_tokens` ≥ the chunker's `max_tokens` (default 8000 ≥ 1300)
so any single reference section always fits — the budget then limits only the
*number* of chunks, dropping whole chunks at the tail, never a lone construct
hit; `from_specs` logs a WARNING when the budget is misconfigured below the
window size. The pipeline builds the `(query, constructs)` for each item from
its SAS metadata (`pipeline.prompting._query_for_item` / `_constructs_for_item`)
— that mapping lives in the pipeline, not here, so this package imports no
`chunker`.

`top_k` and `max_instruction_tokens` default from `config.json`
(`prompt_builder.*`), as do the chunker's token budgets — see the `app_config`
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

## [category: date_time] Date family rules            <- function-family-scoped
One section covering every date/time function.

## [topic] Partitioning guidance                       <- retrieved by ranking
Wide fact tables are partitioned by load_date.

## [example: proc:sql] SQL join                        <- few-shot example
One worked SAS -> PySpark pair demonstrating the response shape.

## [when: proc:sql] [lang: sparksql] SQL joins          <- language-scoped
Translate PROC SQL directly to Spark SQL.               (stacked groups)

## [kind: DATA_STEP] [meta: symput_hazard] SYMPUT       <- kind/metadata-scoped
Trace the write/read ordering before translating.
```

### Statement scoping (`[when: statement:...]`)

A DATA step's *statements* are construct keys too, so guidance can name the
problem it solves rather than the chunk kind it lives in:
`## [when: statement:merge] ...` fires on steps that merge, where
`## [kind: DATA_STEP] ...` fires on every step there is. The vocabulary is
`chunker.keywords.SAS_DATA_STEP_STATEMENT_TOKENS` — the statement keywords
(`merge`, `by`, `retain`, `array`, `output`, `set`, `update`, `modify`,
`where`, `infile`, `do`) plus four the scan derives rather than reads:
`retain` for a sum statement (`x + expr;`), `subsetting_if` for an `if <expr>;`
that drops rows, `set_multi` for a concatenating `SET a b;`, and
`dataset_option` for `keep=` and friends.

This matters more than it looks. Measured on a DATA step using dates, hashing
and sequential `IF`s, `[kind: DATA_STEP]` scoping delivered 520 words (22% of
the block) of MERGE, BY-group, RETAIN and ARRAY guidance the step could not
use; on statement scoping that is 0.

### Family scoping (`[category:]`)

`## [category: date_time, hashing_security] Rule` fires when the item uses
*any* function in one of those SAS families, so one section covers a whole
category without enumerating its members. It is **sugar for
`[when: category:date_time]`**: `pipeline.prompting._constructs_for_item`
derives a `category` construct key per recognised function from
`chunker.keywords.SAS_FUNCTION_CATEGORIES`, so the rule rides the ordinary
construct-matched tier with no extra machinery — it stacks with `[kind:]` /
`[meta:]` / `[lang:]` like any other primary scope, and shows up in focus
hints as *"date time functions"*.

Category keys are emitted **after** the specific `function:`/`call_routine:`
keys, so a rule for the exact function is offered before the family rule.
They never match the reference corpus: PDF sections are titled per function
(`INTNX Function` → `function:intnx`), so no reference chunk carries a
`category` key. The axis exists for operator rules only.

`SAS_FUNCTION_CATEGORIES` is a curated subset of the taxonomy in *SAS
Functions and CALL Routines by Category* — only families that carry
translation guidance are mapped, because an unmapped name simply contributes
no category key. Widening a category there widens every rule scoped on it.

### Modifier clauses: language, chunk kind, metadata (`[lang:]`/`[kind:]`/`[meta:]`)

A heading may carry several leading bracket groups, combined as **AND** across
clauses. Beyond the primary scope (`when`/`topic`/`example`, else always),
three *modifier* clauses stack on to restrict a section further — each passes
only when the item matches it, and a section that omits a clause is agnostic
on that axis:

- **`[lang: sparksql, pyspark]`** — the run's `output_language` must be one of
  the listed targets. Matching is case/space/underscore-insensitive
  (`normalize_language` folds `"SparkSQL"`, `"Spark SQL"`, `"spark_sql"` to one
  key — re-exported from `target_language`, so a directive token and a
  resolved target can never disagree about what a name folds to). Applied at
  **selection time**: the pipeline passes its resolved
  `SasLLMPipeline(output_language=...)` into `build()`. With
  `output_language=None` this axis is off (language-scoped sections are all
  kept), so a builder used without a language over-includes rather than
  silently dropping rules.
- **`[kind: DATA_STEP, PROC_STEP]`** — the item must use one of the listed
  `SasChunkKind` values (`normalize_kind` folds `data step`/`data-step` to
  `DATA_STEP`).
- **`[meta: symput_hazard, unclosed_block]`** — the item's metadata must raise
  one of the listed predicate flags. The vocabulary (owned by the pipeline):
  `symput_hazard`, `abort`, `computed_goto`, `component_object`,
  `unclosed_block`, `includes`, `defines_macros`, `invokes_macros`,
  `produces_macrovars`, `automatic_vars`.

So `## [when: proc:sql] [kind: PROC_STEP] [meta: symput_hazard] [lang: sparksql]
Rule` fires only for a SparkSQL run whose item is a PROC step that uses PROC SQL
*and* carries a SYMPUT hazard. The item's kinds and flags reach the selector as
plain strings — the pipeline computes them in `_kinds_for_item` /
`_meta_flags_for_item` (unioned over a batch's member chunks) — so
`prompt_builder` still imports nothing from `chunker`. Unlike `[lang:]`, the
`[kind:]`/`[meta:]` axes have no "off" state: a caller that passes no kinds/flags
simply never fires a kind/meta-scoped rule (like `[when:]` with no constructs).

### Loading a directory of files (`from_dir`)

`UserInstructionSet.from_dir(dir)` merges every `*.md` under *dir* (sorted
path order, deterministic fingerprint). A file's **first path component**,
when nested, names the target language: sections in `<dir>/sparksql/*.md`
are scoped `[lang: sparksql]` unless they set their own `[lang: ...]`; files
directly under *dir* or under a `_`-prefixed subdirectory (e.g. `_common/`)
are language-agnostic. So one directory holds guidance for several targets
side by side, and the active-language filter picks the right slice per run.
Point config.json `user_instructions.dir` at such a directory (it takes
precedence over `user_instructions.path`); a missing directory warns and
continues, like a missing file.

```
instructions/
  sparksql/          -> every section scoped [lang: sparksql]
    overview.md        always-on: output shape, ANSI mode, null semantics
    constructs.md      PROC SQL, MERGE, BY-groups, MEANS/SUMMARY, TRANSPOSE
    functions.md       string / numeric / conditional scalar functions
    datetime.md        epoch, INTNX/INTCK, date parts, date literals
    hashing.md         MD5 / SHA256 / HASHING family
    regex.md           PRX family
    dataset_ops.md     SORT / APPEND / SET-union / dataset options
    datastep.md        RETAIN, ARRAY, OUTPUT, sequential-IF consolidation
    formats.md         PROC FORMAT and PUT with a user-defined format
    lookup.md          hash-object lookups (the join they really are)
    librefs.md         LIBNAME / FILENAME -> catalogs, schemas, volumes
    ingest.md          PROC IMPORT/EXPORT, INFILE/INPUT -> COPY INTO
    proc_freq.md       PROC FREQ
    proc_univariate.md PROC UNIVARIATE
    proc_rank.md       PROC RANK / PROC STANDARD
    proc_compare.md    PROC COMPARE as a reconciliation query
    catalog_ops.md     DATASETS / DELETE / COPY / CONTENTS -> DDL and catalog
    reporting.md       PRINT / REPORT / TABULATE (OUT= only), FEDSQL
    orchestration.md   [topic] SAS step sequence as a job DAG
    macros.md          macro variables, SYMPUT, macro decomposition
    examples.md        worked SAS -> Spark SQL pairs
  _common/           -> language-agnostic (the leading _ opts out of scoping)
  <root>.md          -> language-agnostic
```

Add a target by creating a sibling directory (`pyspark/`, `snowflake/`, …).
The run's `output_language` selects the matching slice at build time, so
several targets coexist without leaking into each other.

**Documentation files are skipped**, not ingested: a `README*.md` at any
depth, and any file whose name starts with `_`. An instructions directory
almost always carries a README describing itself, and prose *about* the rules
is not a rule — ingested, it parses as always-on sections and is prompted with
every item, for every target.

The package ships a starter set at `prompt_builder/instructions/sparksql/`,
checked against the [Databricks SQL language
reference](https://learn.microsoft.com/azure/databricks/sql/language-manual/)
and Spark 4.1.2 — load it with `from_dir("prompt_builder/instructions")` and
run the pipeline with `output_language="SparkSQL"`. Review and adapt it to your
project's conventions before relying on it.

> ⚠️ **The shipped slice targets Databricks**, not open-source Spark. It uses
> `QUALIFY`, SQL scripting (`FOR`/`WHILE`/`SIGNAL`), `EXECUTE IMMEDIATE`,
> `UNPIVOT` and three-level `catalog.schema.table` names, none of which
> open-source Spark has — a non-Databricks deployment needs those rules
> rewritten. The directory is named `sparksql/` because that is the target key
> `databrickssql` folds to (see `target_language`); the two are one target
> today, so the slice must commit to one of them.

The shipped slice holds only guidance that is true of the *languages* —
nothing about how your shop wants its SQL to look. House rules (parameterising
hardcoded filters, leaving physical layout alone, emitting reconciliation
queries) are illustrated in
[`docs/instruction-examples/site-policy.md`](../docs/instruction-examples/site-policy.md),
deliberately outside this tree so pointing `dir` at the bundled set never
drags one organisation's conventions along. Copy it into your own directory
and edit it.

`[example: <keys>]` sections hold **few-shot worked pairs** — curated
SAS → target translations demonstrating the full desired response shape
(reasoning, fenced code, ⚠️ risk markers). They are selected by construct
match like `[when:]` rules (a bare `[example]` is unconditional) but render
in their own `## Worked examples` block placed *last*, adjacent to the item
they demonstrate for; 5–10 canonical examples (PROC SQL, MERGE, SYMPUT, a
macro loop) anchor output format better than instruction prose.

Wire them in at any level: `PromptBuilder(chunks, user_instructions=...)`,
`builder.with_user_instructions(...)` (rebuilds over the same reference
corpus), or `SasLLMPipeline(user_instructions=...)` — which, with no
`prompt_builder`, constructs a corpus-less builder so rules work without any
reference PDFs. When no explicit set is passed, the pipeline auto-loads the
standing file named by `config.json` `user_instructions.path` (missing file =
WARNING and continue).

Selection priority per item: **user always → user construct-matched →
user kind/meta-gated → user examples → reference pinned → hazard constructs →
other constructs → user `[topic]` → reference topical** — the topical ranking is partitioned so
every relevant user `[topic]` chunk precedes any reference hit, and `top_k`
caps the tier as a whole. Operator rules and examples have first claim on
the budget; one that doesn't fit logs a WARNING naming it.
The three user-rule tiers run **most specific first**: unconditional rules,
then rules matched on the item's exact constructs (`[when:]`/`[category:]`),
then rules merely gated on its chunk kind or metadata (`[kind:]`/`[meta:]`
with no primary scope). A `[kind: DATA_STEP]` note is broader than a rule
naming the function the item actually calls, so it yields to it — otherwise a
handful of broad notes fill the budget and the precise guidance, the reason
the item retrieved anything at all, is what gets dropped.

`user_instructions.max_tokens` (config or the `user_max_tokens` argument)
additionally caps the user chunks (rules and examples together) inside the
overall budget.

**Budget the two together, and size them for your items.** The bundled
SparkSQL set is ~17.9k tokens, so what fits matters. Measured over a 70-line
reference program that batches into two items:

| `max_instruction_tokens` / `max_tokens` | Tokens drawn | Matched rules dropped |
|---|---|---|
| 4000 / 2800 | 5,419 | 14 |
| 8000 / 5600 | 8,250 | 7 |
| 10000 / 7000 | 9,679 | 4 |
| **14000 / 9800** *(shipped)* | 12,368 | **0** |

config.json ships the last row — the measured point where every rule an item's
constructs match actually arrives. Past it the items still draw 12,368 tokens,
so more budget buys nothing.

**It is a ceiling, not a cost floor.** Every section is scoped to a construct,
statement, or function family the item actually uses, so a simple DATA step
draws far less however high the budget goes — raising it buys nothing for
simple items and completeness for complex ones.

⚠️ The pipeline reserves this figure as packing headroom
(`_resolve_packing_budget`), so raising it *lowers* how much SAS source shares
a call when `llm_client.max_input_tokens` is set. That is the intended trade —
guidance and items compete for one request — but it is why the number is read
from the builder rather than duplicated as a constant.

Leaving `max_tokens` null is not the cheaper alternative: uncapped operator
rules simply take the whole budget and starve reference retrieval. Selected rules render in a `## Project instructions` block
above the reference guidance, with the operator's own headings and no page
citations; selected examples render in a `## Worked examples` block placed
last.

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
