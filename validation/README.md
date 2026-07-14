# validation

Offline validation harness for the SAS → LLM pipeline: run a set of
declarative evaluation cases through `SasLLMPipeline`, score the LLM outputs
with deterministic metrics (plus an optional LLM judge), and optionally
append each run to a **Spark-backed history** — a local parquet directory by
default (`./validation_runs`; no server, no service), a Delta table on
Databricks.

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
models.py    ValidationCase, CaseRun, MetricResult, CaseResult,
             ValidationReport (with to_markdown()).
metrics.py   Deterministic metrics + default_metrics():
               response_coverage    every item answered            (>= 1.0)
               dataset_fidelity     item's dataset names appear
                                    in its response                (>= 0.75)
               python_syntax        fenced Python blocks ast.parse (>= 1.0)
               required_terms       case-declared substrings       (>= 1.0)
               reference_similarity token-F1 vs golden reference   (>= 0.5)
judge.py     LLMJudgeMetric — grades functional equivalence 1–5 with any
             LangChain-style model (or an llm_client.LLMClient). Opt-in;
             never part of default_metrics().
runner.py    ValidationRunner: cases -> pipeline -> metrics -> report.
dataset.py   load_cases(): *.json case files (inline sas_source or a
             sas_path reference next to the JSON).
tracking.py  log_report(): one row per (run, case, metric) appended to a
             Spark target — parquet directory locally, saveAsTable (Delta)
             on Databricks. load_runs() reads the history back as a
             DataFrame for trend queries.
cases/       Sample cases.
```

Thresholds resolve with the repo-wide precedence rule (`app_config`):
explicit constructor argument > config.json `validation.<name>_threshold` >
the metric's class default. Metrics that a case carries no signal for
(no reference translation, no required terms, no datasets) report
`skipped` — they pass and are excluded from the case score.

## Usage

CLI (exit code gates CI — 0 iff every case passed):

```bash
# deterministic metrics against a live model (needs ANTHROPIC_API_KEY):
python -m validation validation/cases --model claude-haiku-4-5-20251001

# + LLM judge, + append to the local run history (./validation_runs):
python -m validation validation/cases --judge-model claude-haiku-4-5-20251001 --track

# on Databricks, target a Delta table instead:
python -m validation validation/cases --track --table main.qa.validation_runs
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
