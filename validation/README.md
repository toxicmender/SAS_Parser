# validation

Validation harness for the SAS → LLM pipeline, with two front doors over one
scoring core:

- **Offline cases**: run declarative evaluation cases through
  `SasLLMPipeline` and score the outputs (`ValidationRunner`).
- **Live conversations** (post-hoc, observe-only): score a conversation that
  already happened — an existing memory-store thread by `thread_id`, or any
  arbitrary (prompt, response) transcript — without re-running the pipeline.

Both produce the same result models, scored by deterministic metrics (plus
an optional suite of [LLM-judged metrics](#judged-metrics)), and optionally
append to a **Spark-backed history**
— a local parquet directory by default (`./validation_runs`; no server, no
service), a Delta table on Databricks.

## Why pyspark for tracking

- **LangSmith** requires its hosted service — ruled out by the local-only
  requirement.
- **DeepEval** runs locally, but it drags in a heavy dependency tree and a
  second LLM-configuration path outside `llm_client`/the AI Gateway. Its
  metric *definitions* are worth having, so the interesting ones are
  reimplemented natively here — see [Judged metrics](#judged-metrics).
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
metrics.py       Deterministic metrics + default_metrics(output_language):
                   response_coverage    every unit answered            (>= 1.0)
                   dataset_fidelity     item's dataset names appear
                                        in its response                (>= 0.75)
                   language_compliance  translated blocks are in the
                                        run's target language          (>= 1.0)
                   python_syntax        those blocks parse as the
                                        target (historical name)       (>= 1.0)
                   required_terms       declared substrings            (>= 1.0)
                   reference_similarity token-F1 vs golden reference   (>= 0.5)
judge.py         LLMJudgeMetric — grades functional equivalence 1–5 with any
                 LangChain-style model (or an llm_client.LLMClient). Opt-in;
                 never part of default_metrics().
judged.py        JudgedMetric — base for every LLM-judged metric: one call
                 path over structured output or prose JSON, shared verdict
                 schemas, judge token accounting.
rag_metrics.py   faithfulness, answer_relevancy, hallucination,
                 contextual_precision, contextual_relevancy.
agentic_metrics.py
                 prompt_alignment, plan_adherence, task_completion.
summarization.py summarization (the rolling thread summary) and
                 analysis_summarization (each item's ## Analysis).
                 All twelve judged metrics are opt-in via
                 metrics.judged_metrics().
memory_metrics.py
                 policy_adherence, override_compliance (judged) and
                 memory_extraction, memory_leakage (deterministic) — the
                 memory layers' metrics, built by memory_metrics().
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

## Judged metrics

Twelve opt-in metrics — ten implementing deepeval's published definitions
(deepeval.com/docs/metrics-*) natively, on top of `JudgedMetric`. They cover
what the deterministic suite structurally cannot: whether the model followed
the system prompt's contract, whether the translation is grounded in the
reference guidance `prompt_builder` retrieved (until now the retrieval layer
had no end-to-end measurement at all), whether the `## Analysis` reasoning
matches the code that follows it, and whether the rolling summary keeps the
identifiers `memory.summarize` promises to keep. The other two
(`policy_adherence`, `override_compliance`, in `memory_metrics.py`) score the
instruction memories — see [Memory metrics](#memory-metrics).

| metric | scores | against | scope | default |
|---|---|---|---|---|
| `prompt_alignment` | instructions followed / total | the declared instruction list | item | 0.80 |
| `policy_adherence` | policy instructions followed / total | the run's long-term `TaskPolicy` | item | 0.80 |
| `override_compliance` | overrides honoured + fixed rules respected / total | the thread's short-term notes and the policy's fixed rules | item | 0.90 |
| `answer_relevancy` | relevant statements / total | the item's prompt | item | 0.70 |
| `faithfulness` | truthful claims / total | the retrieved guidance | item | 0.70 |
| `hallucination` | 1 − contradicted contexts / total | the item's SAS source (+ golden translation) | item | 0.80 |
| `contextual_precision` | rank-weighted precision | the retrieved guidance vs the golden translation | item | 0.60 |
| `contextual_relevancy` | relevant statements / total | the retrieved guidance | item | 0.50 |
| `plan_adherence` | plan-vs-execution alignment | `analysis` vs `mapping` + `cells` | item | 0.70 |
| `analysis_summarization` | min(alignment, coverage) | `## Analysis` vs the SAS source | item | 0.60 |
| `summarization` | min(alignment, coverage) | the rolling summary vs the turns it covers | run | 0.60 |
| `task_completion` | task-vs-outcome alignment | the whole run's responses | run | 0.70 |

Three departures from deepeval, each also documented on the class:

- **`hallucination` is inverted.** deepeval scores contradicted/total, where
  lower is better and the threshold is a maximum. Everything here is
  higher-is-better (`MetricResult.passed` is `score >= threshold`,
  `CaseResult.score` is a mean, and a failing metric is fed back to the model
  as "score X < threshold Y" on retry), so this reports **groundedness** and
  puts deepeval's raw rate in `details`.
- **No signal reports `skipped`, not an auto-pass 1.0.** deepeval's Plan
  Adherence auto-passes when the trace states no plan; a skip passes *and*
  stays out of the case mean, which is what `dataset_fidelity` and
  `reference_similarity` already do.
- **`faithfulness` skips deepeval's separate truths-extraction call** — the
  retrieval context is already a short word-budgeted set of manual sections,
  so it is handed to the classifier verbatim (2 calls per item, not 3).

Per-item metrics average over the run's items, as `llm_judge` does: one
`EvaluationRun` here is a whole run, not a single deepeval test case.

```python
from llm_client import LLMClient, LLMClientConfig
from validation import ValidationRunner, default_metrics, judged_metrics

judge = LLMClient(LLMClientConfig(model="claude-sonnet-4-5"))
runner = ValidationRunner(
    pipeline,
    metrics=[*default_metrics(), *judged_metrics(judge)],       # all twelve
)
# or a subset:
judged_metrics(judge, include=["faithfulness", "contextual_relevancy"])
```

From the CLI (`--judge-metrics` needs `--judge-model`; `--judge-model` alone
still means just `llm_judge`):

```bash
python -m validation validation/cases --judge-model claude-sonnet-4-5 --judge-metrics all
python -m validation validation/cases --judge-model claude-sonnet-4-5 \
    --judge-metrics faithfulness,contextual_precision
```

**Cost.** The full suite is roughly **17 judge calls per item plus 5 per run**,
against `llm_judge`'s one — enable a subset unless you mean it. The spend lands
in `ValidationReport.judge_token_usage`, separate from the pipeline's own, with
no extra wiring: `ValidationRunner.judge_token_usage` sums `token_usage` across
every metric that exposes one.

**What each mode can score.** The judged metrics read `input` and retrieved
context off the pipeline's output dicts (`prompt` / `retrieval_context`, which
`SasLLMPipeline._process` records for fresh *and* resumed items). So:

| mode | judged coverage |
|---|---|
| offline cases, inline `LiveValidator` | everything; `contextual_precision` additionally needs a `reference_translation` |
| post-hoc thread / transcript | no retrieval context (it is ephemeral and never persisted) and no item metadata, so the three context metrics, `hallucination` and `analysis_summarization` skip; `summarization` is the one that only works here |

Any judge works: an `LLMClient` is asked for a schema through
`invoke_structured`; anything else is asked in prose and its JSON parsed out
(fenced or embedded). An unusable reply is a warning plus a zero for that unit,
never an exception — a scoring bug must not break a run.

Thresholds resolve with the repo-wide precedence rule (`app_config`):
explicit constructor argument > config.json `validation.<name>_threshold` >
the metric's class default. Metrics that a run carries no signal for
(no reference translation, no required terms, no datasets) report
`skipped` — they pass and are excluded from the case score.

## Memory metrics

Four metrics over the memory layers (`memory.policy`, `memory.thread_mem`,
`memory.extractor`). Two are judged and already sit in the full judged suite;
two are deterministic and need no model at all.

| metric | scores | needs on the run | judged | default |
|---|---|---|---|---|
| `policy_adherence` | policy instructions followed / total | `task_policy` | yes | 0.80 |
| `override_compliance` | overrides honoured **and** fixed rules respected / total | `thread_notes` (+ fixed rules from `task_policy`) | yes | 0.90 |
| `memory_extraction` | F1 of extracted vs expected memories | `expected_memories`, `extracted_memories` | no | 0.70 |
| `memory_leakage` | 1 − foreign notes that surfaced / total | `foreign_notes` | no | 1.00 |

```python
from validation import memory_metrics, validate_thread

validate_thread(pipeline, thread_id, metrics=memory_metrics(judge))
```

`validation.conversation.run_from_thread` fills the first three fields
automatically — the policy from the pipeline that ran the thread, the thread's
own notes, and every *other* thread's notes as the leakage corpus — so
auditing a finished conversation needs no extra bookkeeping. A run that
carries none of them skips all four.

Two things worth knowing about what they mean:

- **`override_compliance` scores both directions.** A model that ignores an
  approved exception and a model that lets an exception talk it past a fixed
  guardrail are both failures, and only the first looks like disobedience.
  That symmetry is what makes short-term memory safe to enable.
- **`memory_leakage` is deliberately narrow.** It catches a foreign note
  quoted verbatim or near-verbatim (word overlap inside a note-sized window
  of the answer). Leakage visible only as changed *behaviour* is not
  detectable from text — that is `override_compliance`'s job.

`memory_metrics()` is kept out of `default_metrics()`: the deterministic pair
is cheap but scores nothing unless a run declares memory expectations, and
always-skipped rows only make a report harder to read.

The policy in force is identified by `SasLLMPipeline.policy_fingerprint`
(fixed at construction, alongside the prompt text it describes). It reaches
the run history as its own `policy_fingerprint` column — see
[Run history](#run-history) for the schema change that implies — and is
repeated in `policy_adherence`'s `details` when the metric is given an
explicit policy object.

## Usage

CLI (exit code gates CI — 0 iff every case passed):

```bash
# deterministic metrics against a live model (needs OPENAI_API_KEY — the
# gateway is OpenAI-compatible for every model it fronts):
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
from pipeline import SasLLMPipeline
from validation import ValidationRunner, load_cases

pipeline = SasLLMPipeline(llm=FakeListChatModel(responses=["..."]))
report = ValidationRunner(pipeline).run(load_cases("validation/cases"))
print(report.to_markdown())
```

`ValidationRunner` takes the target language from the pipeline it is given, so
the code is scored as the language it was prompted for. The entry points that
have no pipeline to ask take it explicitly:

```python
from validation import Evaluator, LiveValidator, default_metrics

LiveValidator(output_language="SparkSQL")          # inline, during a run
Evaluator(output_language="SparkSQL")              # a prepared EvaluationRun
default_metrics("SparkSQL")                        # to build a suite by hand
```

Omitting it resolves config.json `pipeline.output_language`, then the code
default. Getting it wrong is loud rather than silent: a `LiveValidator` built
for one target and handed to a pipeline prompting for another logs a WARNING at
construction, because every item would otherwise fail `language_compliance` and
— with `validation_retries` on — burn the whole retry budget.

## Run history

`log_report()` appends one row per (run, case, metric); run- and case-level
values repeat on each row so any slice of the table is self-describing.
`tracking._COLUMNS` **is** the schema — the DDL is generated from it and the
row builder is checked against it by a test, so a column cannot be added to
one and forgotten in the other.

| group | columns |
|---|---|
| run | `run_id`, `logged_at`, `model`, `instructions_fingerprint`, `policy_fingerprint`, `run_score`, `run_passed`, `case_count` |
| case | `case_id`, `case_score`, `case_passed`, `item_count` |
| metric | `metric`, `score`, `threshold`, `passed`, `skipped`, `details` |

The two fingerprints are the "was this comparable?" columns: runs prompted
under different reference-guidance instructions
(`instructions_fingerprint`, `prompt_builder`) or a different long-term task
policy (`policy_fingerprint`, `memory.policy`) are not equals, and a trend
query should group by them:

```python
(load_runs()
    .groupBy("policy_fingerprint", "metric")
    .avg("score"))
```

> **⚠️ BREAKING — `policy_fingerprint` was added to the schema.** No migration
> is provided. Appending to a history written before this column existed fails
> on the mismatch (a parquet directory raises on read/append, a saved table on
> `saveAsTable`). Point `validation.path` / `validation.table` at a **fresh
> target**, or drop the old one. Old history stays readable where it is; it
> just cannot be appended to or unioned with new rows.

## Markdown report

`to_markdown()` is the report's delivery format — the metric table, per-case
rows, and the run's token totals. Both CLIs write it to a file on request:

```bash
python -m validation validation/cases --md report.md
python demo_run.py local path/to/sas_dir --md report.md
```

`demo_run.py sharepoint` always uploads it as
`<application_name>/output/<timestamp>/validation/report.md`.

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
from pipeline import SasLLMPipeline
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
    policy_fingerprint=pipeline.policy_fingerprint,
)
open("inline_report.pdf", "wb").write(report_to_pdf(report))   # local, or
publish_report_pdf(report, "Reports/Validation")               # SharePoint
```

`report_from_verdicts` is the same builder over a raw list of verdict dicts —
e.g. the `out["validation"]` values a `run_*` call returns. `demo_run.py` uses
exactly this: `--pdf` writes the inline report locally in `local` mode, and
`sharepoint` mode uploads it as `.../validation/report.pdf` beside the JSON.

## Token cost

A report carries two independent token figures, both `llm_client.TokenUsage`
and both `None` when nothing reported usage:

| Field | What it counts |
|---|---|
| `token_usage` | What the run under test cost — the pipeline's own LLM calls. |
| `judge_token_usage` | What *grading* cost — every `JudgedMetric`'s calls (`LLMJudgeMetric` included), i.e. any metric exposing a `token_usage`. |

They are kept apart deliberately: folded together, a run would look more
expensive the more thoroughly it was checked, and judging is a choice about the
eval, not a property of the model being evaluated. Both render into
`to_markdown()` and therefore into the PDF.

`ValidationRunner` attributes them to *its* run by snapshotting before and
subtracting after, since the pipeline and judge belong to the caller and may
have been used before the run or reused after it. Post-hoc thread mode reports
only the judge figure — the translation it scores was billed in some earlier
run. A judge wired to a raw chat model rather than an `LLMClient` reports
nothing, and the field stays `None` rather than claiming zero.

Inline runs (`report_from_verdicts`) take `token_usage=pipeline.token_usage`
explicitly: verdicts are per item, usage is per run, so it cannot be recovered
from the verdicts. `demo_run.py` passes it, and also writes it into
`.../validation/summary.json`.

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
(which, for pipeline threads, embeds the SAS chunk text). The reference
guidance is ephemeral for the same reason, so the retrieval-context metrics
skip too — see the coverage table under [Judged metrics](#judged-metrics).
What thread mode uniquely *can* score is `summarization`: `run_from_thread`
reads the thread's rolling summary (`summary::{thread_id}`) and the prefix of
turns it covers straight off the store. Failure handling
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
  "required_terms": ["groupBy"],
  "prompt_instructions": ["Never use collect() on a full DataFrame"]
}
```

`prompt_instructions` is checked one by one by `prompt_alignment`; omit it and
that metric falls back to config.json `validation.prompt_instructions`, then to
`validation.agentic_metrics.DEFAULT_PROMPT_INSTRUCTIONS` — the system prompt's
own standing contract — rather than skipping.

Use `"sas_path": "programs/job1.sas"` (relative to the JSON file) instead of
`sas_source` for real programs.

## Caveats

- The deterministic metrics validate *shape and fidelity signals* — coverage,
  dataset accounting, syntactic validity, expected terms, drift vs a golden
  baseline. They do not prove functional equivalence; that is what the
  opt-in [judged metrics](#judged-metrics) (and ultimately human review) are
  for. Those are themselves model judgements: treat them as a strong,
  reproducible signal, not a proof — and note that a judged run is only
  comparable to another judged run on the same judge model.
- `reference_similarity` is lexical token-F1: treat it as a regression alarm
  against a known-good baseline, not a correctness score.
- `log_report`/`load_runs` boot a local Spark session, which needs a JVM
  (and `winutils` on Windows). Everything else in the package — runner,
  metrics, judge, report — is Spark-free, and the tracking test skips itself
  where no local Spark session can start.
