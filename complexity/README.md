# complexity

Translation-complexity analysis for SAS chunks, batches, and files. Reads what
the chunker already produced and answers three questions:

1. **How complex is this data step?** — `LOW` / `MEDIUM` / `HIGH`
   (`ComplexityTier`). A property of the **SAS source**, so it does not change
   with the output language.
2. **How well does it map onto the target language?** — `DIRECT` → `MANUAL`
   (`TranslationParity`). A property of the **SAS/target pair**, so it *does*
   change: the same `%MACRO` rates `MANUAL` against Spark SQL and `HARD`
   against PySpark.
3. **How much work is this file?** — `Small` / `Medium` / `Large` /
   `Extra Large` (`TShirtSize`), per **source file**, with story points. Unlike
   the first two this is **volume-aware**, and it accounts for what the file
   borrows from and lends to the rest of the corpus. See
   [T-shirt sizing](#t-shirt-sizing) and [Cross-file references](#cross-file-references).

The catalogue that assigns those ratings is **JSON data**, not code — see
[Retargeting](#retargeting-to-another-output-language). The package is
**standalone**: nothing in the pipeline imports it, so scoring a corpus never
changes what the LLM is asked to translate. It adds no dependencies beyond
`chunker` and `app_config`.

## Quick start

```python
from chunker import SasSemanticChunker, SasChunkBatcher
from complexity import ComplexityAnalyzer

result = SasSemanticChunker().chunk_file("job.sas")
report = ComplexityAnalyzer().analyze_batch_result(SasChunkBatcher().batch(result))

print(report.overall_tier)        # HIGH
print(report.overall_difficulty)  # MANUAL
print(report.tier_counts)         # {'LOW': 2, 'MEDIUM': 5, 'HIGH': 2}
print(report.overall_size)        # EXTRA_LARGE
print(report.total_points)        # 13.0  — the backlog estimate
print(report.to_markdown())       # summary + sizes + hardest-units table

for item in report.hardest(5):
    print(item.tier, item.translation_difficulty, item.rationale)

for f in report.files:
    print(f.source_id, f.size.label, f.points)   # points are 2 / 3 / 5 / 8

for f in report.files_needing_breakdown:      # the Extra Large ones
    print(f.source_id, "→ split at", f.suggested_split)
```

The delivery format is two levels of Markdown — one corpus report plus one per
source script, each printing the SAS behind every verdict — see
[Two reports, not one](#two-reports-not-one). An optional LLM pass adds what the
rules cannot see; see [The LLM evaluation](#the-llm-evaluation).

## CLI

The report's delivery format is Markdown:

```bash
python -m complexity path/to/sas_dir --out complexity-report.md
python -m complexity path/to/sas_dir --out-dir reports/
python -m complexity path/to/sas_dir --target pyspark --top 25
python -m complexity path/to/sas_dir --out-dir reports/ --pdf
```

Without `--out` or `--out-dir` the Markdown goes to stdout. The CLI chunks every
matching file, batches the whole corpus with `MultiFileBatcher` — so cross-file
dataset/macro edges resolve into shared batches — and scores the *batched*
units, the same work items the pipeline would translate, so an estimate lines
up with the run it is estimating. `--no-cross-file` scores every file as if it
were alone; `--size-anchor` recalibrates the T-shirt scale (lowering it rates
every file larger); `--min-story-points` / `--max-story-points` report the
points on your own scale instead of the Fibonacci 2-8; `--no-graph-image` skips
drawing [the dependency graph](#the-image); `--pdf` renders [the corpus report
as a PDF](#the-pdf) as well. Nothing here calls an LLM or touches the network
unless you ask for [the LLM evaluation](#the-llm-evaluation).

## Two reports, not one

`--out-dir` writes the whole deliverable:

```
reports/
  complexity-report.md      the corpus: sizes, backlog total, hardest units
  complexity-report.pdf     the same, as a PDF (--pdf only, see below)
  dependency-graph.png      the corpus dependency graph (optional, see below)
  files/
    load.md                 one report per source SAS script
    report.md
```

The corpus report is `CorpusComplexityReport.to_markdown()` with an index of the
individual ones appended. Each **individual** report answers the question the
corpus one cannot — *what is in this file, and why did it score that?* — so it
carries the file's dimensions, its [datasets](#the-datasets-section), its
cross-file coupling, its drivers table, and then every chunk with its verdict
**and its SAS source**:

````markdown
### `f1-chunk-0001` — MACRO_DEFINITION (lines 1–9)

- Tier **HIGH** · parity **MANUAL** · score 21.00
- HIGH tier driven by kind:MACRO_DEFINITION, array, do_loop; …

Signals:

- array [HIGH/HARD]: ARRAY array x — aliases a group of columns, not a Spark
  ArrayType; needs wide-to-long restructuring, not explode()

```sas
%macro load(lib=raw);
  data work.a;
    array x{3} v1-v3;
```
````

Printing the source is the point: a verdict a reader has to go and look up in
another file is a verdict they will not check. `--no-source-text` drops it and
`--max-chunk-lines N` caps each snippet.

The text is **passed in**, not read off the verdict — a `ChunkComplexity`
carries no `text` field, because a verdict model that embedded its own source
would double the size of every serialised report and duplicate what the chunker
already holds. `chunk_texts()` builds the lookup:

```python
from complexity import ComplexityAnalyzer, chunk_texts, write_reports

batch_result = MultiFileBatcher().batch(corpus)
report = ComplexityAnalyzer().analyze_items(batch_result.all_ordered_items)
write_reports(report, "reports/", texts=chunk_texts(batch_result.all_ordered_items))
```

Build the lookup from **the same items you scored**. `MultiFileBatcher` re-ids
every chunk per file (`f1-chunk-0001`), so one built from the unbatched corpus
would match nothing. It is keyed `(source_id, chunk_id)` rather than on the id
alone for the same reason `crossfile.py` is: files are chunked independently, so
two files' first chunks share an id.

### Files are named, not pathed

A `source_id` is a path, usually absolute — the CLI hands the chunker
`str(path)` for whatever directory you pointed it at. That is the right
identity and the wrong label: twenty rows of
`D:\corp\migration\sas\etl\load_customers.sas` are a column of shared prefix
with the informative part pushed off the edge. So every rendered report — the
corpus tables, the dependency edges and waves, the graph image, the per-file
titles, the cross-file evidence, the LLM prompts — prints the **name**, while
the model keeps the path.

`naming.display_names()` decides what the name is, and it is not simply the
basename: two `load.sas` scripts in different directories are two different
files with two different scores, and printing both as `load.sas` would make the
report a puzzle. Each id is shortened to its last segment, then widened one
parent at a time **only if it would otherwise be ambiguous**:

```
/corp/sas/etl/load.sas     ->  etl/load.sas
/corp/sas/adhoc/load.sas   ->  adhoc/load.sas
/corp/sas/report.sas       ->  report.sas      (nothing collides, so nothing widens)
```

One collision does not lengthen every other label. The full path is still
printed exactly once, in the `- Path:` bullet of the file's own report — the
reader looking at a single file's verdict is the one who may need to go open
it. Nothing keys off a name: every lookup and every model field still uses the
`source_id`. This is the display counterpart of `source_stems()`, which solves
the same collision for output *filenames* (`files/load.md`, `files/load_2.md`).

## The LLM evaluation

Everything above is rule-driven, which is both its strength and its ceiling. It
counts constructs, not intent, so it cannot see business rules packed into one
arithmetic-heavy DATA step (which scores `0.0`), a `DO` loop that is really a
join, hard-coded environment assumptions, or a step whose output nothing
consumes.

`--llm-eval` asks a model for exactly that, one call per file:

```bash
python -m complexity path/to/sas_dir --out-dir reports/ --llm-eval
python -m complexity path/to/sas_dir --out-dir reports/ --llm-eval --eval-top 5
python -m complexity path/to/sas_dir --out-dir reports/ --prompt-only
```

The prompt hands the model the file's measured verdict, its drivers, its
coupling, and its full SAS source, and asks where the static numbers are wrong
or incomplete: a purpose, a size verdict (`agree` / `larger` / `smaller`) with a
justification, P0/P1/P2 findings the construct rules missed, manual steps, split
points, and open questions. The static analysis is the ground the prompt stands
on, not something the model is asked to reproduce — *"Do not re-derive the
numbers below and do not restate the drivers table. Argue with it."*

The result lands in `reports/llm-evaluation.md`. `--eval-top N` evaluates only
the N largest files, since every file is a paid call. `--prompt-only` writes the
prompts to `reports/prompts/` and calls nothing, so the prompt can be read and
tuned for free.

```python
from complexity import build_evaluation_prompt, evaluate_report

print(build_evaluation_prompt(report.files[0], texts=texts))   # offline
evaluation = evaluate_report(llm, report, texts=texts)         # one call per file
print(evaluation.to_markdown())
```

`llm` is anything with a LangChain-style `invoke(input) -> message` — an
`llm_client.LLMClient` (which additionally gets a typed answer, since
`invoke_structured` is used when offered), a raw chat model, or a fake in tests.
So this package still depends on nothing but `chunker` and `app_config`, and on
`pipeline` not at all. A reply that will not parse is kept as prose with the
reason, and the other files still get their evaluations: one bad reply must not
cost a corpus its run.

Score against a different output language by naming its profile:

```python
ComplexityAnalyzer(target="pyspark").analyze_result(result)
```

Every verdict carries its own evidence — the `signals` list names each construct
found, where it came from (`metadata` or `detector`), the source snippet that
triggered it, and the catalogue's standing guidance on why it is rated that way.

## Retargeting to another output language

The construct catalogue lives in JSON under `complexity/profiles/`, one file
per target. Nothing about a target is hardcoded, so adding one is a data
change:

| Profile | Target | Notes |
| --- | --- | --- |
| `sparksql.json` | Spark SQL | The default (`rules.DEFAULT_TARGET`). |
| `pyspark.json` | PySpark | `extends` the Spark SQL profile and restates only what differs. |

With no `target` argument and no `complexity.target` in config.json, the
profile follows the **run's** output language (config.json
`pipeline.output_language`, via `target_language`), so a PySpark run rates its
files against PySpark without a second knob to keep in sync. Set
`complexity.target` to score against a different target than the pipeline
translates into.

A profile may inherit with `"extends": "<name>"`; the child's entries deep-merge
over the parent's per construct, so a derived target states **only its
differences**. `pyspark.json` is ~15 entries against Spark SQL's ~130, because
almost everything is identical — what moves is the handful of constructs that
become easier once a full Python host language surrounds the DataFrame API:

```json
{
  "target": "pyspark",
  "display_name": "PySpark",
  "extends": "sparksql",
  "constructs": {
    "kind": {
      "MACRO_DEFINITION": {
        "category": "macro-def", "tier": "HIGH", "parity": "HARD",
        "note": "%MACRO -> a parameterised Python function; a real mapping exists here, unlike in pure SQL"
      }
    }
  }
}
```

Note what does **not** move: the *tier* stays `HIGH`. Tiers describe the SAS
side and are target-independent; only parity is a statement about the pair. A
test asserts this across every bundled profile.

### Writing a profile

Each entry is `{category, tier, parity, note}`, grouped by construct namespace
— `proc`, `component_object`, `function`, `call_routine`, `global_statement`,
`kind` (a `SasChunkKind` value), and `detector` (a name from `detectors.py`).
Boolean metadata flags live in a `flags` list, each adding an `attr` naming the
`SasChunkMetadata` attribute to test. Optional `weights` set the per-tier score
weights.

For a whole family that shares a rating, `construct_groups` avoids repetition:

```json
"construct_groups": [
  {"kind": "function", "names": ["md5", "sha256"],
   "category": "hashing", "tier": "MEDIUM", "parity": "SUPPORTED"}
]
```

Profiles are **validated on load** and a bad one raises `RuleSetError` naming
the offending key rather than being silently skipped — a profile that does not
parse means the operator asked for a classification scheme that cannot be
applied, which should stop the analysis instead of quietly scoring everything
against a different one.

Point at a profile this package does not ship with either
`ComplexityAnalyzer(rules_path="my_rules.json")` or `complexity.rules_path` in
`config.json`.

## The two aggregation rules

- **Tier is presence-based.** A unit's tier is the *highest* tier among its
  signals, so one `ARRAY` in an otherwise trivial step still reads `HIGH`. This
  follows the brief literally ("High for arrays, do loops, `%macro`
  definitions"); a weighted-threshold scheme would let a lone array average
  away to `MEDIUM`.
- **Difficulty is worst-case.** A unit's `translation_difficulty` is the
  *least* translatable parity among its signals, for the same reason.

`score` exists only to rank units *within* a tier and never feeds back into the
tier. It sums each **distinct** construct's weight once, so a step using five
different hard constructs outranks one that mentions the same construct five
times — repetition is verbosity, variety is work. A repeated construct is
collapsed into a single signal whose evidence is annotated `(×N)`.

A batch's tier and difficulty are the worst any member reaches; its **score is
the sum** of its members', because ten simple steps genuinely are more work
than one.

## T-shirt sizing

A tier and a parity together still cannot answer "how big is this file?". Both
are **presence-based** — one `ARRAY` makes a file HIGH however short it is —
and neither counts anything. A 2000-line file of plain DATA steps raises no
signal at all, scores `0.0`, and would read as trivial. It is not.

So each **source file** also gets a `TShirtSize`: `Small`, `Medium`, `Large`,
`Extra Large`. Files rather than batches, because a batch may span several
files (`SasBatch.source_files`) while every chunk belongs to exactly one, so a
file rollup built from chunks is unambiguous however the corpus was batched.

Four sizes, not the conventional six: XS and XXL are dropped because a size
nobody can tell from its neighbour costs more than it explains.

### The three dimensions

A size is not one number. The method requires stating up front what a size
*means* and then holding to it, so this one is declared as three terms,
reported separately on `FileComplexity`:

| Dimension | Fields | Measures | Fed by |
| --- | --- | --- | --- |
| **Effort** | `effort_raw` / `effort_norm` | how much there is | chunks, lines, contained DATA/PROC steps, dataset I/O, macro parameters |
| **Complexity** | `complexity_raw` / `complexity_norm` | how hard it is | each distinct signal's tier weight **plus** its parity weight |
| **Uncertainty** | `uncertainty_raw` / `uncertainty_norm` | what we could not pin down | unresolved cross-file refs, unclosed blocks, `UNKNOWN_*` chunks, parser diagnostics |

Keeping them apart is the point: a `Large` file that is large on *uncertainty*
needs someone to go and find the missing pieces, while one large on *effort*
just needs more hands. A single blended number hides that distinction.

**Comment blocks are excluded from all three.** A `COMMENT_BLOCK` chunk raises
no signal in any profile — there is nothing in it to translate — yet it would
still count as chunk and line volume, which would make a file's size partly a
measure of how well it was documented. The exclusion is at the *analysis*
layer, not the rendering one (`analyzer.EXCLUDED_KINDS`): comment chunks never
become a `ChunkComplexity` at all, so the numbers change and not just the
prose, and a well-commented file scores exactly what its bare equivalent does.
`FileComplexity.comment_chunk_count` reports how many were dropped, and each
individual report says so, so a `chunk_count` can always be reconciled against
the chunker's own — a gap with nothing explaining it reads as a bug.

Including **parity** in the complexity term is what makes a size
target-dependent. Unlike a tier, the same file is genuinely less work against
PySpark than against Spark SQL, because a Python host language absorbs macros
and loops that pure SQL cannot express. That is why the sizing model lives in
the profile.

### Rescaling: log, then min-max

The three dimensions are counted in incomparable units — effort runs to the
hundreds on line and step counts, while uncertainty rarely passes ten — so
adding them up would weight them by magnitude rather than by intent. Each is
therefore **log-transformed and min-max rescaled** onto `0…1` before they are
combined:

```
norm = clamp01( (log1p(raw) − log1p(min)) / (log1p(max) − log1p(min)) )
```

`log1p` rather than `log` because every dimension legitimately reaches zero —
a file with nothing unresolved has no uncertainty at all.

The **log** says that returns diminish: the 200th step of a file tells you far
less than the 20th, and equal *ratios* rather than equal *counts* are what move
a size. The **min-max** says where the scale spends its resolution: below `min`
a dimension has stopped discriminating (a file with less effort than a third of
an anchor is small however you count it), above `max` it has saturated.

Both bounds are declared per dimension in the profile, as **multiples of the
anchor**, which keeps the anchor the single knob that moves everything:

```json
"bounds": {
  "effort":      {"min": 0.35, "max": 2.40},
  "complexity":  {"min": 0.35, "max": 1.40},
  "uncertainty": {"min": 0.00, "max": 0.50}
},
"dimension_weights": {"effort": 0.88, "complexity": 0.50, "uncertainty": 0.20}
```

Rescaling makes the mix explicit where a raw sum left it to chance, so the
weights have to be stated too. They blend the three normalised dimensions into
one `0…1` number (`FileComplexity.blend`) — a weighted **sum, clamped at 1**,
not a mean. The weights therefore say how far each dimension reaches *on its
own*, which is why they add up to more than 1:

- a mean would cap a file that is nothing but volume at the effort weight, so
  no amount of bulk could ever ask to be broken down. An enormous file needs
  splitting however plain its contents are, so effort's reach (`0.88`) clears
  the `Extra Large` boundary (`0.85`) by itself;
- a mean would also dock every file that has nothing unresolved in it, since
  the unused uncertainty share would be lost. Summing means uncertainty only
  ever *adds*: having no unknowns costs a file nothing.

Effort reaches furthest because it is most of what a size is answering;
uncertainty reaches least, because it asks for an investigation rather than for
translation hands — and it is reported separately precisely so that a small
reach never means "ignored".

`FileComplexity` carries each dimension twice — `effort_raw` and `effort_norm`
— because they answer different questions. The raw number is the measurement;
the normalised one is how much of the scale that measurement actually claimed.
A dimension at `1.00` has saturated, and more of the same will not move the
size again: 200 plain DATA steps and 400 of them score identically, because
past the top of the window a longer file of the same trivial steps is not
telling you anything new.

### The anchor

Sizing is **relative estimation against a fixed reference**, which is the
method's own prescription — you size a story by comparing it to a reference
story the team already knows, not by consulting a table of absolute cutoffs.

The reference lives in the profile:

```json
"sizes": {
  "anchor": {
    "raw": 87.5,
    "dimensions": {"effort": 50.0, "complexity": 37.5, "uncertainty": 0.0},
    "describes": "The reference MEDIUM file: ~200 lines and 16 chunks, comprising a %MACRO wrapping two DATA steps, nine further DATA steps, a match-merge, a PROC SORT, a PROC SUMMARY, and two LIBNAME assignments it uses throughout."
  }
}
```

Every window above is a multiple of `anchor.raw`, so **lowering the anchor makes
every file rate larger.** It is deliberately the one knob exposed in
`config.json` (`complexity.size_anchor`), because it moves every verdict
coherently, where editing a single band would just skew one rung.

`87.5` is that reference file's **measured** raw score, not a guess: the file
described above was written out and run through the analyzer. `dimensions` is
the same measurement split three ways, and it must sum to `raw` — a profile
where it does not is rejected on load. It is stated because the dimensions are
normalised *separately*, so a file's composition changes its size and not just
its total: 87.5 spent entirely on effort does not land where 50 + 37.5 does.

Each profile anchors to its *own* measurement of the same reference (PySpark
reads `81.5`), which keeps that file Medium against every target — as the
definition of Medium requires — so what stays target-dependent is the *relative*
construct mix. A macro-heavy file rates larger against Spark SQL than against
PySpark; a file of plain DATA steps rates identically against both.

The anchor is *fixed data*, not recomputed per run. A corpus-relative
(percentile) scheme would re-rate the same file differently depending on which
files it happened to be analysed alongside, and would be undefined for a
single-file run — useless for a tool whose whole job is planning a migration
subset by subset.

The reference file is synthetic — there is no SAS corpus in this repo to
calibrate against — so treat the anchor as a defensible starting point, and
argue with `anchor.describes` rather than with the individual thresholds. If
you have a real file your team would call a textbook Medium, measure it and put
both numbers in the profile:

```python
f = ComplexityAnalyzer().analyze_result(result).files[0]
f.raw_total                                        # -> anchor.raw
f.effort_raw, f.complexity_raw, f.uncertainty_raw  # -> anchor.dimensions
```

A test asserts the shipped anchors still measure what they claim, in both
profiles, so a drifted calibration fails rather than silently re-rating a
corpus.

### Scale and bands

Points follow the Fibonacci progression the method uses, because estimation
confidence is geometric — the gap between Small and Medium really is smaller
than the gap between Large and Extra Large:

| Size | Points | Band (upper bound) |
| --- | ---: | ---: |
| `Small` | 2 | ≤ 2.5 |
| `Medium` | 3 | ≤ 4.0 |
| `Large` | 5 | ≤ 6.5 |
| `Extra Large` | 8 | above 6.5 |

Bands sit at the geometric midpoints of the rungs. The blend reaches those
bands by the same move as the dimensions themselves — a min-max rescale,
applied **in log space**, between the ends of the scale:

```
position = min_story_points × (max_story_points / min_story_points) ^ blend
```

so a blend of `0` is exactly `Small`'s 2, a blend of `1` exactly `Extra
Large`'s 8, and the reference file lands on `3.01` by calibration. Log space
for the same reason the rungs are Fibonacci: equal steps of evidence should be
equal *ratios*, not equal differences.

That `position` is what the **banding** reads. It is not what a file
**reports**. `FileComplexity.points` is its size's rung — always 2, 3, 5, or 8
— and the un-snapped position is kept alongside as `continuous_points`.

Reporting the rung rather than the position is the point of a Fibonacci scale,
not a rounding convenience. The gaps widen with the number precisely so that
nobody is asked to defend an 8 against a 9; a file reported at `6.77` has
quietly reintroduced the precision the method exists to refuse, and invites a
reader to compare two files that no team could tell apart. It also removes a
contradiction the continuous number could not avoid: a short `%MACRO` floors at
`Medium` (see [Floors](#floors)) while its position stays at `2.00`, so the old
scheme labelled a file `Medium` and printed `Small`'s number next to it. Points
now follow the **final** size, floors included.

`continuous_points` keeps what the rung gives up. A rung cannot rank two files
*inside* one size, and an anchor is calibrated against the position rather than
the rung it lands on, so both are reported — every table here orders on
`(points, continuous_points)`, since most files tie on the first.

Points still **sum**: `report.total_points` is a backlog estimate for the whole
migration, which a purely qualitative label could never give you — read as "n
files at 2-8 points each", not as an absolute quantity of days. Summing deck
entries is exactly how a sprint backlog is totalled.

#### Estimating on your own scale

Not every team estimates on 2-8. Both ends are configurable — in the profile,
in `config.json`, or per analyzer:

```json
"sizes": {"story_points": {"min": 1, "max": 13}}
```

```python
ComplexityAnalyzer(min_story_points=1, max_story_points=13)
```

```bash
python -m complexity path/to/sas_dir --min-story-points 1 --max-story-points 13
```

This **re-denominates the numbers and moves nothing else**. The bands are read
as fractions of the scale's log span (`SizeModel.band_blends`), not as absolute
point values, so the same file gets the same `blend` and the same `TShirtSize`
whichever scale is reporting it — the reference file lands mid-`Medium` on 1-13
exactly as it does on 2-8. `min` must be greater than zero: points are rescaled
geometrically, so the smallest size cannot be worth nothing.

The re-denominated rungs stay on the deck. Both **ends are kept exactly** as
asked for, and the two interior rungs are re-derived at their geometric
positions and snapped back to the nearest Fibonacci number — measured in log
space, so `4` snaps to `5` rather than `3`. On 1-13 the rungs are therefore
`1 / 2 / 5 / 13`, not `1 / 2.11 / 5.42 / 13`:

```python
ComplexityAnalyzer(min_story_points=1, max_story_points=13)
# Small 1, Medium 2, Large 5, Extra Large 13
```

Where a range is too narrow to hold four distinct deck entries the un-snapped
value stands for that rung, with a DEBUG line saying so. Monotonicity wins over
Fibonacci there: the progression is worth having, but not at the price of two
sizes reporting the same number.

### Extra Large means *split this*

The top rung is an instruction, not just a magnitude — its published meaning is
work that must be broken down before it can be estimated at all. So
`TShirtSize.EXTRA_LARGE.needs_breakdown` is True (and no other size's is), and
`FileComplexity.suggested_split` names the batch ids inside that file as cut
points — batches are already dependency-respecting translation units, which
makes them the natural seams. `to_markdown()` renders these under **Files
needing breakdown**.

### Floors

Some kinds have a size floor regardless of what the numbers say
(`sizes.min_size_by_kind`):

```json
"min_size_by_kind": {
  "GLOBAL_STATEMENT": "SMALL",
  "OPTIONS": "SMALL",
  "MACRO_DEFINITION": "MEDIUM"
}
```

A file of `%LET`/`LIBNAME`/`OPTIONS` configuration floors at `Small` — a no-op
at the shipped anchor, stated explicitly so the intent is legible in the data
rather than being an absence. A file defining a `%MACRO` floors at `Medium`,
and that one **does** bind: a short macro scores well under the Medium band on
volume, and "we found a macro definition" is worth more than its line count
suggests. `FileComplexity.floored_by` names the kind responsible, and only when
the floor actually changed the answer.

### Worked examples

Measured, not estimated — these are real outputs at the shipped anchor, each
dimension shown as `raw (normalised)`:

`Position` is the continuous `continuous_points`, which the banding read;
`Points` is the reported estimate, always a deck entry.

| File | Effort | Cplx | Uncert | Blend | Position | Size | Points |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Config only: 3 `%LET` + 1 `LIBNAME` | 2.0 (0.00) | 2.5 (0.00) | 0.0 (0.00) | 0.00 | 2.00 | Small | 2 |
| Thin macro wrapper, 1 step, 2 params | 2.5 (0.00) | 21.0 (0.00) | 0.0 (0.00) | 0.00 | 2.00 | Medium *(floored)* | 3 |
| 12 plain DATA steps | 42.1 (0.16) | 0.0 (0.00) | 0.0 (0.00) | 0.14 | 2.44 | Small | 2 |
| **The reference file** (see the anchor) | 50.0 (0.25) | 37.5 (0.14) | 0.0 (0.00) | 0.29 | 3.01 | **Medium** | **3** |
| Macro-heavy: array, DO, 12 merges | 57.3 (0.32) | 49.0 (0.34) | 0.0 (0.00) | 0.45 | 3.74 | Medium | 3 |
| 50 plain DATA steps | 175.5 (0.91) | 0.0 (0.00) | 0.0 (0.00) | 0.80 | 6.04 | Large | 5 |
| 80 plain DATA steps | 280.8 (1.00) | 0.0 (0.00) | 0.0 (0.00) | 0.88 | 6.77 | Extra Large | 8 |
| Bulk **and** hard: 45 merge steps behind a macro of arrays, DO forms, `LAG`, `CALL EXECUTE` | 212.9 (1.00) | 175.5 (1.00) | 0.0 (0.00) | 1.00 | 8.00 | Extra Large | 8 |

Four things worth reading off that table. The reference file lands mid-`Medium`
and reports `3`, which is what "anchored at Medium" means. The 50- and 80-step
files rate `Large` and `Extra Large` on **volume alone** — their complexity term
is `0.0`, because nothing in them is individually notable — which is precisely
the case a tier cannot express, and the reason sizing exists. The last row
reaches the ceiling from both directions at once, which is what a file nobody
should start without splitting looks like. And row two is why points follow the
size rather than the position: its position is `2.00`, but the `%MACRO` floor
makes it a `Medium`, so it is estimated at `3`.

### References

The scale, the anchored-relative method, the effort/complexity/uncertainty
split, and the "Extra Large means break it down" rule are taken from:

- [ActiveCollab — T-shirt sizing](https://activecollab.com/blog/project-management/t-shirt-sizing)
- [Asana — T-shirt sizing](https://asana.com/resources/t-shirt-sizing)
- [StarAgile — T-shirt sizing in agile](https://staragile.com/blog/t-shirt-sizing-in-agile)

## Cross-file references

A SAS script is rarely self-contained, and what it borrows is exactly what
makes it hard: a file you cannot translate without three others in front of you
is more work than its own text suggests. `crossfile.py` resolves every outward
reference against the rest of the corpus and raises ordinary
`ComplexitySignal`s (`source="cross_file"`) on the chunk that carries the
reference — so they flow through the same max-tier / worst-parity / summed-weight
aggregation as everything else, and batch scores become cross-file-aware for
free.

References are resolved from metadata the chunker already extracts. No new
source scanning:

| Reference | Producer field | Consumer field |
| --- | --- | --- |
| macro | `defines_macros` | `invokes_macros` |
| dataset | `output_datasets`, `body_literal_outputs` | `input_datasets` |
| macro variable | `produces_macrovars`, `declared_macro_vars` | `consumes_macrovars` |
| libref | `defines_librefs` | libref prefix of dataset I/O |

Each reference lands in one of three states: **internal** (same file — no
signal at all), **import/export** (satisfied by another file in the corpus), or
**unresolved** (satisfied by nothing in scope).

`%INCLUDE` is deliberately **not** among them. The chunker already surfaces it
as both a chunk kind and a metadata flag, and the catalogue rates both
MEDIUM/PARTIAL — precisely what a cross-file entry would assign — so adding a
third signal would inflate the score without adding information.

Two exclusion sets are reused from the chunker rather than re-listed, so they
cannot drift: `_STANDARD_AUTOCALL_MACROS` (`%left`, `%trim`, … ship with SAS,
so calling one is never a missing dependency) and `_DEFAULT_LIBREFS` (`work`,
`sashelp`, … are always assigned).

### Unresolved requires having actually looked

The state that matters most is the one you can get wrong. `macro_unresolved`
rates HIGH/MANUAL — you cannot translate a macro whose body you do not have —
but that verdict is only defensible if there were *other files to search*.

So the macro case has four states, not three:

| Where the `%MACRO` body is | Construct | Rating |
| --- | --- | --- |
| same file | — | no signal |
| another file in the corpus | `macro_import` | MEDIUM / PARTIAL |
| **nowhere, corpus has ≥ 2 files** | `macro_unresolved` | **HIGH / MANUAL** |
| nowhere, corpus has 1 file | `macro_external` | MEDIUM / PARTIAL |

With a single file in scope, absence proves nothing — the reference is merely
*external*, and the note says so. HIGH/MANUAL is reserved for the case where
the corpus was searched and came up empty.

`CrossFileProfile.corpus_files` records how many files were in scope, so a
consumer can always tell which regime a verdict came from.

### The full catalogue

Under a profile's `cross_file` namespace:

| Construct | Spark SQL rating | Reasoning |
| --- | --- | --- |
| `macro_import` | MEDIUM / PARTIAL | needs another file's output in scope first |
| `macro_unresolved` | HIGH / MANUAL | body unavailable; no faithful translation exists |
| `macro_external` | MEDIUM / PARTIAL | a gap, but not proof of absence |
| `macro_export` | LOW / DIRECT | being depended on is effort, not difficulty |
| `dataset_import` | MEDIUM / PARTIAL | inter-file execution ordering |
| `dataset_export` | LOW / DIRECT | as `macro_export` |
| `dataset_unresolved` | LOW / SUPPORTED | almost always a pre-existing source table |
| `macrovar_import` | MEDIUM / PARTIAL | a value crosses a file boundary at run time |
| `macrovar_export` | LOW / DIRECT | as `macro_export` |
| `libref_import` | LOW / SUPPORTED | maps onto catalog/schema configuration |
| `libref_unresolved` | MEDIUM / PARTIAL | physical location of the data is unknown |

The `_export` directions are rated LOW/DIRECT on purpose: being depended on is
a *scheduling* cost, not a translation one, so it nudges a file's size without
ever inflating its tier or parity.

Only `macro_unresolved` and `libref_unresolved` feed the **uncertainty**
dimension. `dataset_unresolved` explicitly does not — reading a table no
analysed file writes is the normal case for the first job in a pipeline, not a
hazard.

PySpark restates `macro_import` and `macro_external` as SUPPORTED, since a
macro in another file becomes a Python function imported from another module.
Tiers are unchanged, as everywhere else.

`ComplexityAnalyzer(use_cross_file=False)` — or `complexity.use_cross_file` in
`config.json` — scores every file as if it were the only one. `analyze_chunk(chunk)`
called *without* an index raises no cross-file signals either: a lone chunk has
no corpus to resolve against, and inventing one would be a guess.

### The datasets section

Each individual report states the file's **data interface** before it states
its coupling — the second only means something once you know the first:

```markdown
## Datasets

- Inputs (read here, written elsewhere): edw.raw
- Outputs (written here): work.stg, mart.out
- Intermediates (written and read here): work.stg
```

The three-way split is the useful part. **Inputs** must already exist when this
file runs; **outputs** are what downstream files are waiting on; and
**intermediates** — written *and* read inside this file — are its own business,
so nobody has to provide them. A dataset the file writes is therefore never
reported as an input, which is the same rule `crossfile.py` applies when
deciding whether a read is a cross-file import. The two sections cannot
contradict each other by construction.

Every chunk that touches a dataset prints its own `Reads:` / `Writes:` line, so
a reader who doubts a rollup can find the chunk that put each name in it.

## The dependency graph

`crossfile.py` answers "what does *this* file depend on?" one file at a time.
That is the right shape for a file report and the wrong shape for planning: a
migration is ordered work, and the order is a property of the whole corpus.
`graph.py` folds every resolved reference into file-to-file edges, and the
corpus report gains a section reading the structure off them:

```markdown
## Dependency graph

| Upstream (migrate first) | Downstream | Via |
| --- | --- | --- |
| load.sas | transform.sas | dataset raw.customers |
| transform.sas | report_a.sas | dataset mart.agg |
| transform.sas | report_b.sas | dataset mart.agg |

### Migration order

- **Wave 1**: load.sas
- **Wave 2**: transform.sas
- **Wave 3**: report_a.sas, report_b.sas
```

Imports and exports are one dependency seen from both ends — an import in B
naming A and an export in A naming B both mean "A before B" — so they fold into
a single edge carrying every name that caused it. Edges come from all four
reference kinds, labelled, because they do not carry the same weight: a shared
dataset is an ordering problem, while a shared macro means two files cannot be
translated independently without agreeing on the macro first.

**A "DAG" is an aspiration, not a guarantee.** Two SAS jobs can each read a
dataset the other writes; such a corpus is unusual but not invalid, and
emitting a topological order for one would be a confident lie. So
`DependencyGraph.cycles` is reported explicitly, `is_acyclic` says which case
you are looking at, and files caught in a cycle are parked in a trailing
*Unordered* layer rather than given a wave the graph does not support.

### The image

With `--out-dir`, the graph is also drawn to `reports/dependency-graph.png` and
linked from the corpus report — waves left to right, arrows coloured by what
causes the dependency:

| Colour | Edge |
| --- | --- |
| Red | macro |
| Yellow | macro variable |
| Violet | dataset |
| Green | other (libref) |

An edge can carry several kinds at once — one file may share both a macro and a
dataset with the next — so it takes the colour of the hardest one present, in
the order of that table: the macro coupling has to be agreed before either file
can be translated, while the dataset is only an ordering constraint that the
left-to-right layout already shows. The legend lists only the kinds actually
drawn. Colour is therefore the dependency kind and never anything else, so a
cycle is shown as a **dashed** arrow and a dashed node border instead.

The image needs matplotlib, which is the **optional `graph` extra**:

```bash
uv pip install -e ".[graph]"
```

Without it `render_png` returns `None` and logs a line; the run still produces
every other artefact. That is deliberate: the Markdown edge table is the
primary form, not a fallback. It renders in every viewer, in a terminal, and on
the stdout path where there is nowhere to put a file — so the edges are never
only in the picture. `--no-graph-image` skips the drawing outright, and a
corpus of more than `MAX_GRAPH_NODES` (60) files skips it too, since past that
the picture is a hairball and the table is strictly more legible.

## The PDF

The Markdown is the primary artefact and stays it — it diffs, it renders in
every viewer, and it is what everything else here produces. The PDF is for the
other audience: the estimate goes to someone who does not have the repository
checked out, and "here is a link to a `.md` file" is not an answer for them.

```bash
python -m complexity path/to/sas_dir --out-dir reports/ --pdf
python -m complexity path/to/sas_dir --out estimate.md --pdf
```

`--pdf` converts the **overall** report, never replacing it: with `--out-dir`
the PDF lands at `reports/complexity-report.pdf`, and with `--out` it takes
that file's name with a `.pdf` suffix. It needs one of the two — stdout has
nowhere to put a file, and asking for a PDF without a destination exits 1 before
the corpus is even scored. Rendering from the written report (rather than from
the Markdown in memory) is what lets `dependency-graph.png` resolve, so the
graph lands in the document.

No new dependency and no external converter: `markdown-it-py` parses and
PyMuPDF's `Story` lays the HTML out, and both are already core to this project.
Two details `complexity/pdf.py` handles for the layout engine:

- **Code blocks are folded first.** `Story` clips a long `<pre>` line at the
  frame edge instead of wrapping it, and the individual reports print whole SAS
  statements — routinely wider than a page. Lines are soft-wrapped to
  `CODE_WIDTH` columns, keeping their indentation, before becoming HTML.
- **Images resolve through an archive.** The report links its graph relative to
  itself, so the Markdown's own directory is handed to `Story` as a
  `pymupdf.Archive`.

Unlike the graph image — a supplement that degrades to `None` when matplotlib is
absent — a PDF only ever exists because someone asked for one, so a failure
`raise`s `PdfRenderError` rather than being logged and swallowed. The CLI
reports it and exits 1; the Markdown is already on disk either way.

## Tiers

Tiers are target-independent. The parity column below is the **Spark SQL**
profile's; `pyspark.json` rates several of the HIGH constructs more favourably.

| Tier | Constructs | Typical Spark SQL parity |
| --- | --- | --- |
| **LOW** | simple `PROC SQL`, macro variables (`%LET`, `&var`, `%GLOBAL`/`%LOCAL`), plain DATA steps, `PROC SORT`/`MEANS`/`FREQ`/`PRINT` | `DIRECT` / `SUPPORTED` |
| **MEDIUM** | hash objects and hashing functions (`MD5`, `SHA256`), **match-merge** (`MERGE` *with* `BY`), `UPDATE`/`MODIFY`, `RETAIN` and `FIRST.`/`LAST.`, SFTP/FTP/email/URL `FILENAME` methods, `PROC HTTP`, `PROC TRANSPOSE`, `CALL SYMPUT`, date-interval functions (`INTNX`, `INTCK`) | `PARTIAL` |
| **HIGH** | `ARRAY`, `DO` loops (iterative, `DO WHILE`, `DO UNTIL`), `%MACRO` definitions, macro control flow, computed `%GOTO`, `CALL EXECUTE`, `SYMGET`/`RESOLVE`/`DOSUBL`, `LAG`/`DIF`, **one-to-one merge** (`MERGE` *without* `BY`), `PROC FCMP`/`IML`/`DS2`, `FILENAME PIPE` | `HARD` / `MANUAL` |

## Where the ratings come from

The tier and parity assignments are grounded in documentation, not intuition:
the bundled `reference_docs/` corpus for the SAS side, and the published Spark
function reference for the target side. Each load-bearing finding is quoted at
its rule in the profile's `note`.

The Spark-side ratings are checked against the [Spark SQL
built-in function reference](https://spark.apache.org/docs/latest/api/sql/index.html).
That is what moved the SAS hashing *functions* to `SUPPORTED`: Spark SQL ships
`md5`, `sha1`, `sha2`, `crc32`, and `xxhash64`, so `MD5(x)` is a mechanical
rewrite rather than a semantic mismatch. The hash *object* is a lookup table,
not a function, and stays `PARTIAL`.

**A SAS `ARRAY` is not a Spark array.** *SAS Programmer's Guide: Essentials*,
Ch. 24: "In SAS, an array is not a data structure. An array is just a
convenient way of temporarily defining a group of variables." The
plausible-looking mapping — array column plus `explode()` — is therefore
**wrong**. A SAS array aliases a group of *columns*, so translating it means a
wide-to-long restructure or per-column expressions. The rule's evidence string
says this outright, to steer a reader (or an LLM reading the output) off the
wrong mapping.

**`MERGE` is two different constructs**, and the presence of a `BY` statement
is the documented discriminator (Essentials, Ch. 21): match-merging "requires
the MERGE statement together with the BY statement", while one-to-one merging
"requires the MERGE statement without the BY statement. There is no key
variable on which to base the merge. Instead, rows are merged implicitly by row
number." A match-merge is a join with different overlay rules (`MEDIUM`). A
BY-less merge has no key at all — it pairs rows positionally, which a
distributed DataFrame has no inherent ordering to reproduce, so it rates
`HIGH`/`HARD`. The detector splits them.

**`LAG` is not `lag()`.** *SAS Functions and CALL Routines: Reference*
describes it as returning "values from a queue": "A LAGn function stores a
value in a queue and returns a value stored previously in that queue. Each
occurrence of a LAGn function in a program generates its own queue." The queue
advances only when that call site executes, so a `LAG` inside a conditional is
**not** `lag(col)` over an ordered window. SAS's own distributed engine declines
it outright — "not supported in a DATA step that runs in CAS" — which is the
clearest available evidence that inter-row dependency resists distribution.
`HIGH`/`HARD` stands.

One caveat on the corpus: the bundled Spark document is an *excerpt* (127
pages) that explicitly defers "aggregations, window functions, and joins" to
chapters it does not include. Absence of a function there is therefore not
evidence that Spark lacks it, and no rating below was lowered on that basis.

## Spark parity scale

| Rating | Meaning | Example |
| --- | --- | --- |
| `DIRECT` | literal equivalent exists | `PROC SQL` select → `spark.sql` |
| `SUPPORTED` | idiomatic equivalent, mechanical rewrite | `PROC SORT` → `orderBy` |
| `PARTIAL` | equivalent exists, semantics differ enough to need care | SAS `MERGE` is not a plain join |
| `HARD` | no direct equivalent; needs a different paradigm | row-wise `DO` loop → vectorised columns / `explode` / UDF |
| `MANUAL` | outside the translation target; a human must decide | `%MACRO` definition |

## Layout

```
models.py      ComplexityTier, TranslationParity, TShirtSize (ordered scales)
               + max_tier / worst_parity / max_size helpers; ComplexitySignal;
               ChunkComplexity, BatchComplexity, FileComplexity,
               CrossFileProfile, DependencyEdge / DependencyGraph (layers,
               cycles), CorpusComplexityReport (with to_markdown()).
rules.py       RuleSet + SizeModel + the JSON profile loader (inheritance,
               validation, caching). Holds no ratings of its own.
profiles/      The catalogues themselves, one JSON file per target language.
               THE place to retune the analysis.
detectors.py   Regex scans for what SasChunkMetadata does not extract:
               ARRAY, DO loops, MERGE/UPDATE/MODIFY, RETAIN, FIRST./LAST.,
               FILENAME access methods, INFILE/FILE, LINK, DATA step GOTO.
crossfile.py   CrossFileIndex — resolves macro/dataset/macro-var/libref
               references across the corpus into import / export / unresolved.
graph.py       build_graph — folds those references into the corpus dependency
               graph; renders it as a Markdown edge table, and as a PNG when
               the optional `graph` extra (matplotlib) is installed.
analyzer.py    ComplexityAnalyzer — aggregation and sizing only; owns no tier
               of its own. EXCLUDED_KINDS drops COMMENT_BLOCK before scoring.
report.py      Markdown rendering: the corpus report plus one report per source
               SAS script, each printing the source behind every verdict.
               Rendering only — scores nothing, calls nothing.
llm_eval.py    The optional second opinion: the evaluation prompt, the shape
               asked back, and the invocation. Duck-typed on the client, so the
               package gains no LLM dependency.
```

## Where signals come from

Most constructs are already extracted by the chunker, and are read straight off
`SasChunkMetadata`: `proc_name`, `recognized_functions`,
`recognized_call_routines`, `component_objects`, `global_statement_keyword`,
plus the boolean hazards (`symput_scope_hazard`, `contains_computed_goto`,
`contains_abort`, `defines_macros`, …) and the chunk's `kind`.

What the chunker does **not** extract is the DATA step's own imperative
vocabulary — `ARRAY`, `DO`, `MERGE`, `RETAIN`, `FIRST.`/`LAST.` — and the
`FILENAME` access methods. Those are exactly the constructs the brief turns on,
so `detectors.py` scans for them directly. All scans run on text sanitised by
`chunker.scanner._sanitise`, so a construct named inside a comment or a quoted
string can never fire a signal.

The macro language's `%DO` is deliberately **not** a DATA step `DO` loop: it is
compile-time code generation, already classified through the
`MACRO_CONTROL_FLOW` chunk kind. The detectors' negative lookbehinds keep the
two apart, and tests assert it.

## The catalogue is an allowlist

A construct with no entry in `rules.py` contributes **no signal at all**, which
floors a chunk at `LOW`/`DIRECT`. Silence means "nothing notable found", never
"unknown" — an unrecognised function must not inflate a score. This is why
`ComplexityAnalyzer(...)` on a step full of ordinary arithmetic returns an
empty `signals` list and a score of `0.0`.

The one exception is logged loudly: if a *detector* fires for a construct with
no `DETECTOR_RULES` entry, that is a wiring bug rather than a property of the
SAS source, so the signal is dropped with a `WARNING` instead of being given an
invented classification. A test asserts every detector name has an entry.

## Configuration

`config.json`, section `complexity`:

```json
"complexity": {
  "target": null,
  "rules_path": null,
  "weight_low": null,
  "weight_medium": null,
  "weight_high": null,
  "size_anchor": null,
  "min_story_points": null,
  "max_story_points": null,
  "use_cross_file": null
}
```

`target` names a bundled profile; `rules_path` points at a profile file of your
own and wins over `target`. Precedence throughout is the repo standard:
explicit constructor argument > `config.json` > the profile's own value >
built-in default. Weights only rank units within a tier — they can never change
a tier. To retune **which construct means what**, edit a profile JSON, not the
config.

`size_anchor` overrides the profile's reference-Medium raw score; lowering it
makes every file rate larger. `min_story_points` / `max_story_points`
re-denominate the reported points (see
[Estimating on your own scale](#estimating-on-your-own-scale)) and cannot move
a size. The bands, the min-max `bounds`, and the per-dimension weights stay in
the profile, because they are calibrated relative to the anchor and only make
sense alongside it — the bounds are literally stated as multiples of it, so
overriding the anchor moves every window with it.

`ComplexityAnalyzer(use_detectors=False)` restricts the analysis to what the
chunker's own metadata reports, dropping the supplementary scans.
`use_cross_file=False` scores every file as if it were the only one.

## Entry points

| Method | Input |
| --- | --- |
| `analyze_chunk` | one `SasChunk` |
| `analyze_batch` | one `SasBatch` (aggregates its members) |
| `analyze_items` | any mix of batches and chunks — takes `SasBatchResult.all_ordered_items` directly, so you can score exactly the units the pipeline translates |
| `analyze_result` | a single-file `SasChunkResult` |
| `analyze_batch_result` | a `SasBatchResult` (batches + singletons) |
| `analyze_corpus` | a multi-file `SasCorpus`, unbatched |

All return a `CorpusComplexityReport` except the first two.

One asymmetry worth knowing: the **uncertainty** dimension folds in parser
diagnostics, and `SasBatchResult` does not carry any. So `analyze_result` and
`analyze_corpus` supply them automatically, while `analyze_batch_result` cannot
and every file it produces is marked `uncertainty_complete=False` rather than
quietly reporting a lower uncertainty than the file really has. Pass
`diagnostics=` to `analyze_items` directly if you need the full picture from a
batched run.

## Tests

`tests/test_complexity.py` — 258 tests, no LLM (the evaluation is exercised
through fakes) and no disk I/O apart from the tests that are about disk: the
rule-set loader's, and the report writer's and CLI's. Covers each tier
against the constructs the brief names, the max-tier/worst-parity aggregation
rules, detector precision (comments, string literals, `%DO` vs `DO`,
`if…then do;` blocks, MERGE with vs without BY), batch aggregation, report
rendering, the config and `use_detectors` switches, retargeting between
profiles, profile inheritance, and rule-set validation failures.

The log + min-max rescale is covered in its own right: clamping at both ends of
each window, monotonicity, the two halves of the log property (equal increments
buy less the further up you are; equal *ratios* are equal steps), the points
scale spanning exactly `Small`…`Extra Large` and rescaling geometrically,
weights reading as reaches rather than shares (volume alone reaching `Extra
Large`, uncertainty only ever adding), volume saturating at the top of its
window, windows moving with the anchor, and the profile validation for
`bounds`, `dimension_weights`, and an `anchor.dimensions` split that does not
sum to `anchor.raw`. A story-point range set in the profile, in `config.json`,
or per analyzer is asserted to re-denominate `points` while leaving every
`blend` and every `TShirtSize` untouched.

The Fibonacci reporting has its own set: every file's `points` is a deck entry
and equals its size's rung (over a corpus spanning at least three sizes, so the
assertion is not vacuous); `continuous_points` still separates two files that
tie on the rung; the floored `%MACRO` case reports `Medium`'s 3 while its
position stays below 2.5, which is the contradiction the snap removes; the
nearest-entry search is geometric (`4 → 5`, not `3`) and idempotent on the deck;
re-denominating to 1-13 keeps both ends and snaps the interior to `1/2/5/13`;
and a range too narrow for four entries falls back to un-snapped rungs rather
than emitting two sizes with the same number.

The sizing and cross-file additions bring: Fibonacci banding and monotonicity,
anchor rescaling in both directions, the chunk-kind floors (including a forced
case where the `MACRO_DEFINITION` floor actually binds), `Extra Large` implying
`needs_breakdown` and offering cut points, each of the three dimensions moving
a size independently, volume alone driving a `Large` on a file that raises no
signal at all, the four macro-resolution states (and that `macro_unresolved`
requires a multi-file corpus), autocall macros and default librefs never being
flagged, librefs assigned in another chunk of the same file, chunk-id collision
across independently chunked files, a batch spanning two files still yielding
two file rollups, and catalogue coverage in both directions — every construct
`crossfile.py` can emit has a profile entry, and every entry is reachable.

The reporting and evaluation additions bring: the text lookup keyed by source
*and* chunk id (so two files sharing a chunk id do not collide, and a lookup
built from batched items carries the re-ided keys), every mentioned chunk's SAS
appearing in its file's report, `--no-source-text` and `--max-chunk-lines`,
a missing snippet rendering a placeholder rather than vanishing, two scripts
sharing a basename getting distinct report files, the corpus report gaining an
index without otherwise changing, the prompt carrying verdict and source and
being built with nothing called, and the three replies an evaluation has to
survive — a structured one, JSON recovered from prose, and an unusable one kept
as prose — plus a client that raises failing only its own file.

The comment exclusion, datasets, and dependency graph bring: that the chunker
really does emit a `COMMENT_BLOCK` for the fixture (without which the rest
proves nothing), that none reaches a verdict by either the singleton or the
batched path, that a commented file scores *identically* to its bare
equivalent, and that a comment-only file still gets a rollup instead of
vanishing; the input/output/intermediate split, that a dataset written then
read stays out of the inputs, case-insensitive dataset identity, and that the
rollup cannot contradict the cross-file coupling; and for the graph, that
**both** readers of one dataset get an edge (the case a single-peer record
loses), that import and export fold into one edge, the wave levelling, macro
edges and their labels, a two-file cycle being reported rather than silently
ordered, and `render_png` writing real PNG bytes when matplotlib is present
while returning `None` — not raising — when it is absent, when there are no
edges, or when the node cap is exceeded.

Naming and the PDF bring: that a unique basename is the whole name, that a
collision widens only the files that collide (and widens as far as it must),
that both separators split, and that a corpus of absolute ids renders reports
with no directory prefix anywhere — tables, edges, waves, cross-file evidence,
and prompts — while the models still hold the paths and the file's own report
prints its path exactly once; and for the PDF, real `%PDF-` bytes beside an
untouched Markdown, text surviving the conversion, code folded rather than
clipped, SAS angle brackets escaped rather than opening an element, the graph
image landing in the document, a missing source raising `PdfRenderError`, and
`--pdf` without `--out`/`--out-dir` exiting 1 having written nothing.

```
python -m pytest tests/test_complexity.py -v
```
