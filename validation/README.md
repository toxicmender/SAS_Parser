# validation

Validation harness for the SAS → LLM pipeline, with two front doors over one
scoring core:

- **Offline cases**: run declarative evaluation cases through
  `SasLLMPipeline` and score the outputs (`ValidationRunner`).
- **Live conversations** (post-hoc, observe-only): score a conversation that
  already happened — an existing memory-store thread by `thread_id`, or any
  arbitrary (prompt, response) transcript — without re-running the pipeline.

Both produce the same result models, scored by deterministic metrics (plus
an optional LLM judge), and optionally append to a **Spark-backed history**
— a local parquet directory by default (`./validation_runs`; no server, no
service), a Delta table on Databricks.

## Why pyspark for tracking

- **LangSmith** requires its hosted service — ruled out by the local-only
  requirement.
- **DeepEval** runs locally, but its interesting metrics are LLM-judged and
  it drags in a heavy dependency tree for what this repo needs.
- **MLflow** works locally, but adds a large optional dependency and (as of
  MLflow 3) pushes local users onto a SQLite store anyway.
- **pyspark** is already a core dependency of this repo, and production runs
  on Databricks: locally `log_report` appends parquet rows to a directory,
  in production the same call appends to a managed Delta table — identical
  to how `memory.store` splits its in-memory and Delta backends.

The scoring layer itself is storage-free: metrics are plain functions of the
pipeline's inputs/outputs, and Spark is only booted inside
`log_report`/`load_runs` — never by the metrics or the runner.

## Layout

```
models.py        ValidationCase, EvaluationRun (case-free scoring unit),
                 CaseRun (case-derived subclass), MetricResult, CaseResult,
                 ValidationReport (with to_markdown()).
metrics.py       Deterministic metrics + default_metrics():
                   response_coverage    every unit answered            (>= 1.0)
                   dataset_fidelity     item's dataset names appear
                                        in its response                (>= 0.75)
                   python_syntax        fenced Python blocks ast.parse (>= 1.0)
                   required_terms       declared substrings            (>= 1.0)
                   reference_similarity token-F1 vs golden reference   (>= 0.5)
judge.py         LLMJudgeMetric — grades functional equivalence 1–5 with any
                 LangChain-style model (or an llm_client.LLMClient). Opt-in;
                 never part of default_metrics().
evaluator.py     Evaluator — the scoring core: one EvaluationRun in, one
                 CaseResult out. Everything funnels through here.
runner.py        ValidationRunner: cases -> pipeline -> Evaluator -> report.
conversation.py  validate_thread() / validate_transcript() (and their
                 run_from_*() builders): post-hoc live-conversation scoring.
live.py          LiveValidator: inline per-item scoring during a run, with
                 the verdict stored in that conversation's memory.
                 validations_for_thread() reads the verdicts back.
dataset.py       load_cases(): *.json case files (inline sas_source or a
                 sas_path reference next to the JSON).
tracking.py      log_report(): one row per (run, case, metric) appended to a
                 Spark target — parquet directory locally, saveAsTable
                 (Delta) on Databricks. load_runs() reads the history back
                 as a DataFrame for trend queries.
pdf.py           report_to_pdf(): render a report's to_markdown() to a PDF
                 (markdown-it-py -> HTML -> PyMuPDF Story). publish_report_pdf()
                 renders and uploads it to a SharePoint document library via
                 app_config.sharepoint.
cases/           Sample cases.
```

Thresholds resolve with the repo-wide precedence rule (`app_config`):
explicit constructor argument > config.json `validation.<name>_threshold` >
the metric's class default. Metrics that a run carries no signal for
(no reference translation, no required terms, no datasets) report
`skipped` — they pass and are excluded from the case score.

## Usage

CLI (exit code gates CI — 0 iff every case passed):

```bash
# deterministic metrics against a live model (needs ANTHROPIC_API_KEY):
python -m validation validation/cases --model claude-sonnet-4-5

# + LLM judge, + append to the local run history (./validation_runs):
python -m validation validation/cases --judge-model claude-sonnet-4-5 --track

# on Databricks, target a Delta table instead:
python -m validation validation/cases --track --table main.qa.validation_runs

# render the markdown report to a PDF — locally, and/or into SharePoint:
python -m validation validation/cases --pdf report.pdf
python -m validation validation/cases --pdf-sharepoint Reports/Validation
```

Programmatic, fully offline (this is what tests/test_validation.py does):

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from chunker import SasLLMPipeline
from validation import ValidationRunner, load_cases

pipeline = SasLLMPipeline(llm=FakeListChatModel(responses=["..."]))
report = ValidationRunner(pipeline).run(load_cases("validation/cases"))
print(report.to_markdown())
```

## PDF report (and SharePoint)

The same `to_markdown()` report renders to a paginated PDF —
markdown-it-py turns it into HTML (the metric table included), PyMuPDF's
`Story` lays it across A4 pages — and, on request, uploads to a SharePoint
document library through `app_config.sharepoint`:

```python
from validation import report_to_pdf, publish_report_pdf

pdf_bytes = report_to_pdf(report)               # or report_to_pdf(markdown_str)
open("report.pdf", "wb").write(pdf_bytes)

# render + upload; a folder dest gets a timestamped filename, a *.pdf dest is
# used verbatim. Auth/site settings come from app_config.sharepoint (needs the
# `sharepoint` extra and an Entra ID service principal).
publish_report_pdf(report, "Reports/Validation")
```

The SharePoint destination resolves with the repo-wide precedence rule:
the explicit argument, else config.json `validation.report_sharepoint_path`,
else the library root. Rendering needs only PyMuPDF (a core dependency) and
markdown-it-py — no SharePoint extra; uploading imports `app_config.sharepoint`
lazily, so `import validation` stays cheap.

## Inline validation (during the run)

Score each item **as it is answered**, and store the verdict in the same
conversation memory the run uses — no post-hoc pass, no separate history.
Opt in by handing the pipeline a `LiveValidator`:

```python
from chunker import SasLLMPipeline
from validation import LiveValidator, validations_for_thread

pipeline = SasLLMPipeline(model="claude-sonnet-4-5", validator=LiveValidator())
pipeline.run_text(sas_source, source_id="job1.sas", thread_id="run::job1.sas")

# One verdict per item, filed beside that item's run fact:
facts = pipeline.get_validation_facts("run::job1.sas")   # or
facts = validations_for_thread(pipeline._memory.kv, "run::job1.sas")
for f in facts:
    print(f["item_id"], f["score"], f["passed"])
```

The per-item verdicts also aggregate into the same `ValidationReport` an
offline run produces, so an inline run reuses `to_markdown()`, the PDF renderer,
and the Spark history without a second scoring pass:

```python
from validation import report_from_thread, report_to_pdf, publish_report_pdf

report = report_from_thread(
    pipeline._memory.kv, "run::job1.sas",
    model="claude-sonnet-4-5",
    instructions_fingerprint=pipeline.instructions_fingerprint,
)
open("inline_report.pdf", "wb").write(report_to_pdf(report))   # local, or
publish_report_pdf(report, "Reports/Validation")               # SharePoint
```

`report_from_verdicts` is the same builder over a raw list of verdict dicts —
e.g. the `out["validation"]` values a `run_*` call returns. `demo_run.py` uses
exactly this: `--pdf` writes the inline report locally in `local` mode, and
`sharepoint` mode uploads it as `.../validation/report.pdf` beside the JSON.

Each item is scored the instant its response returns, through the same
`Evaluator` core as every other mode, so an inline verdict equals a post-hoc
one over that single item. Because it scores **one item at a time**, the item
carries its own metadata: `dataset_fidelity` scores precisely (it does not
skip the way a metadata-less thread does), and `response_coverage` counts
that one item. The default suite is deterministic and offline, so inline
validation adds no model call; append an `LLMJudgeMetric`
(`LiveValidator(metrics=[...])`) to grade each item with a judge — that call
is per-item and blocking.

Storage mirrors the run facts (`validation::{thread_id}::item::{item_id}`
against `run::{thread_id}::item::{item_id}`): same thread, same per-item
granularity, small facts only (the response itself stays in the `msg::`
history). Each `run_*` output dict also carries the item's verdict under a
`"validation"` key. The pipeline always swallows any validator error, so a
scoring bug can never break a translation run.

### Acting on the verdict — `validation_retries`

By default (`validation_retries=0`) inline validation is **observe-only**: a
failing item is neither retried nor allowed to abort the run. Set
`SasLLMPipeline(validator=LiveValidator(), validation_retries=N)` to make the
verdict *actionable*:

```python
pipeline = SasLLMPipeline(validator=LiveValidator(), validation_retries=2)
```

- **Improving the batch (inline).** When an item fails, its just-produced
  turn is rolled back off the thread (`KVChatMessageHistory.truncate_to`) and
  the item is re-prompted with a corrective note naming the metrics that fell
  short — ephemeral, like reference guidance: prompted, never persisted. The
  loop stops as soon as an attempt passes or the budget is spent, and the
  final attempt's turn and verdict are the ones that persist. Exactly one
  (human, AI) pair survives per item, so the history invariants hold. The
  run fact records the `attempts` it took.
- **Resuming.** The same switch makes `resume=True` validation-aware: an item
  whose *stored* verdict failed no longer counts as done. The run rewinds to
  the earliest unsatisfied item (missing, errored, or failed), drops that
  item's and every later item's turns and facts, and regenerates from there
  on a clean history — the passing prefix is kept and recovered as before.

With no validator attached, `validation_retries` has no effect (a warning is
logged) and the run stays observe-only.

## Live conversations (post-hoc)

Score a thread the pipeline already ran — reconstructed from the memory
store's (human, AI) turn pairs, never re-executed:

```python
from validation import validate_thread

# after pipeline.run_text(..., thread_id="run::job1.sas") has happened:
result = validate_thread(pipeline, "run::job1.sas", required_terms=["groupBy"])
print(result.passed, result.score)
```

`validate_thread` accepts a `SasLLMPipeline` or a bare `MemoryHub`; with a
pipeline the outputs carry their real item ids (recovered from the run
facts), otherwise turns are labelled `turn-<n>`. From the CLI, against a
Delta-backed store:

```bash
python -m validation --thread run::job1.sas --delta-table main.ml.langchain_memory
```

Arbitrary transcripts work too — `(prompt, response)` pairs or a LangChain
message list:

```python
from validation import validate_transcript

result = validate_transcript(
    [("translate: data a; run;", "```python\ndf = spark.table('a')\n```")],
    run_id="adhoc",
)
```

Caveats of item-less scoring: the chunker/batcher items are not persisted,
so `dataset_fidelity` skips ("no item metadata"), `response_coverage` counts
turns instead of items, and the LLM judge grades against each turn's prompt
(which, for pipeline threads, embeds the SAS chunk text). Failure handling
is observe-only: results are returned/logged, nothing gates or retries.
Wrap results in a `ValidationReport(model=..., results=[...])` to reuse
`to_markdown()` / `log_report()` — `case_id` simply carries the thread or
transcript id.

Querying the accumulated history:

```python
from validation import load_runs

runs = load_runs()  # or load_runs(spark=spark, table="main.qa.validation_runs")
runs.groupBy("run_id", "model").avg("case_score").orderBy("run_id").show()
runs.filter("metric = 'dataset_fidelity'").groupBy("run_id").avg("score").show()
```

## Case files

One JSON object (or a list) per `*.json` file:

```json
{
  "case_id": "simple_etl",
  "description": "what this exercises",
  "sas_source": "data work.a; ... run;",
  "reference_translation": "optional golden output",
  "required_terms": ["groupBy"]
}
```

Use `"sas_path": "programs/job1.sas"` (relative to the JSON file) instead of
`sas_source` for real programs.

## Caveats

- The deterministic metrics validate *shape and fidelity signals* — coverage,
  dataset accounting, syntactic validity, expected terms, drift vs a golden
  baseline. They do not prove functional equivalence; that is what the
  opt-in LLM judge (and ultimately human review) is for.
- `reference_similarity` is lexical token-F1: treat it as a regression alarm
  against a known-good baseline, not a correctness score.
- `log_report`/`load_runs` boot a local Spark session, which needs a JVM
  (and `winutils` on Windows). Everything else in the package — runner,
  metrics, judge, report — is Spark-free, and the tracking test skips itself
  where no local Spark session can start.
