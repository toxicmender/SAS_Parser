# complexity

Translation-complexity analysis for SAS chunks and batches. Reads what the
chunker already produced and answers two questions per unit:

1. **How complex is this data step?** — `LOW` / `MEDIUM` / `HIGH`
   (`ComplexityTier`).
2. **How well does it map onto Spark?** — `DIRECT` → `MANUAL`
   (`SparkParity`), the SAS → Spark SQL feature-parity and code-translation
   difficulty rating.

The package is **standalone**: nothing in the pipeline imports it, so scoring a
corpus never changes what the LLM is asked to translate. It adds no
dependencies beyond `chunker` and `app_config`.

## Quick start

```python
from chunker import SasSemanticChunker, SasChunkBatcher
from complexity import ComplexityAnalyzer

result = SasSemanticChunker().chunk_file("job.sas")
report = ComplexityAnalyzer().analyze_batch_result(SasChunkBatcher().batch(result))

print(report.overall_tier)        # HIGH
print(report.overall_difficulty)  # MANUAL
print(report.tier_counts)         # {'LOW': 2, 'MEDIUM': 5, 'HIGH': 2}
print(report.to_markdown())       # summary + hardest-units table

for item in report.hardest(5):
    print(item.tier, item.translation_difficulty, item.rationale)
```

Every verdict carries its own evidence — the `signals` list names each
construct found, where it came from (`metadata` or `detector`), and the source
snippet that triggered it.

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

## Tiers

| Tier | Constructs | Typical Spark parity |
| --- | --- | --- |
| **LOW** | simple `PROC SQL`, macro variables (`%LET`, `&var`, `%GLOBAL`/`%LOCAL`), plain DATA steps, `PROC SORT`/`MEANS`/`FREQ`/`PRINT` | `DIRECT` / `SUPPORTED` |
| **MEDIUM** | hash objects and hashing functions (`MD5`, `SHA256`), **match-merge** (`MERGE` *with* `BY`), `UPDATE`/`MODIFY`, `RETAIN` and `FIRST.`/`LAST.`, SFTP/FTP/email/URL `FILENAME` methods, `PROC HTTP`, `PROC TRANSPOSE`, `CALL SYMPUT`, date-interval functions (`INTNX`, `INTCK`) | `PARTIAL` |
| **HIGH** | `ARRAY`, `DO` loops (iterative, `DO WHILE`, `DO UNTIL`), `%MACRO` definitions, macro control flow, computed `%GOTO`, `CALL EXECUTE`, `SYMGET`/`RESOLVE`/`DOSUBL`, `LAG`/`DIF`, **one-to-one merge** (`MERGE` *without* `BY`), `PROC FCMP`/`IML`/`DS2`, `FILENAME PIPE` | `HARD` / `MANUAL` |

## Where the ratings come from

The tier and parity assignments are grounded in the bundled `reference_docs/`
corpus, not in intuition. Three findings shaped the catalogue, and each is
quoted in a comment at its rule:

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
models.py      ComplexityTier, SparkParity (ordered scales) + max_tier /
               worst_parity helpers; ComplexitySignal; ChunkComplexity,
               BatchComplexity, CorpusComplexityReport (with to_markdown()).
rules.py       The signal catalogue: construct -> (category, tier, parity,
               weight). Pure data. THE place to retune the analysis.
detectors.py   Regex scans for what SasChunkMetadata does not extract:
               ARRAY, DO loops, MERGE/UPDATE/MODIFY, RETAIN, FIRST./LAST.,
               FILENAME access methods, INFILE/FILE, LINK, DATA step GOTO.
analyzer.py    ComplexityAnalyzer — aggregation only; owns no tier of its own.
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

`config.json`, section `complexity` — per-tier score weights only:

```json
"complexity": {
  "weight_low": null,
  "weight_medium": null,
  "weight_high": null
}
```

Precedence is the repo standard: explicit constructor argument > `config.json`
> catalogue default (`rules.WEIGHT_LOW` / `WEIGHT_MEDIUM` / `WEIGHT_HIGH`).
Weights only rank units within a tier — they can never change a tier. To
retune **which construct means what**, edit `rules.py`, not the config.

`ComplexityAnalyzer(use_detectors=False)` restricts the analysis to what the
chunker's own metadata reports, dropping the supplementary scans.

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

## Tests

`tests/test_complexity.py` — 51 tests, no LLM and no disk I/O. Covers each
tier against the constructs the brief names, the max-tier/worst-parity
aggregation rules, detector precision (comments, string literals, `%DO` vs
`DO`, `if…then do;` blocks), batch aggregation, report rendering, the config
and `use_detectors` switches, and catalogue integrity.

```
python -m pytest tests/test_complexity.py -v
```
