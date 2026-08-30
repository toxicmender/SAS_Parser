# V2 validation and assessment contracts

Status: Phase 8 implemented. The top-level `validation` and `complexity`
packages remain shipped compatibility surfaces until the Phase 10 composition
cutover.

## Validation ownership

- `sas_migrate.application.validation.models` owns source-neutral cases,
  transcript units, metric results, reports, and token-budget policy/output.
- `metrics` owns the six deterministic offline metrics. Their names, default
  thresholds, ordering, and no-signal skip behavior remain characterized.
- `judged` owns the judged/RAG/agentic/summarization/policy catalogue and calls
  only `ValidationJudge`; no provider or translation service is imported.
- `memory_metrics` owns deterministic extraction and cross-thread leakage
  checks.
- `conversation`, `live`, and `runner` reconstruct transcripts, validate an
  already-produced response, and drive offline cases through
  `ValidationRunProducer` respectively.
- `budgeting` derives `token_budget_compliance` from the attempt-level core
  ledger. Input categories such as SAS source, project instructions, target
  directives, history, and policy remain individually attributable.
- `reporting` renders Markdown/JSON. The validation adapters own append-only
  JSONL tracking and PDF bytes.

Translation and judge ledgers, policies, summaries, violations, retry
overhead, and recovered usage are separate fields throughout. Target
resolution validation is rendered for both structured and raw-fallback
responses through the common `ResponseValidationResult` contract.

## Assessment ownership

- `sas_migrate.resources.assessment` is the single PySpark/Spark SQL profile
  catalogue. The legacy complexity analyzer reads these same resources so the
  two paths cannot drift.
- `sas_migrate.application.assessment.profiles` resolves recursive inheritance,
  merges sizing blocks/constructs/flags, validates rules, and rejects cycles.
- `service` owns target-profile selection, construct scoring, sizing,
  uncertainty, cross-file dataset edges, and optional review through the
  `AssessmentReviewer` port.
- `reporting` renders JSON/Markdown; assessment adapters own packaged/custom
  profile stores and PDF bytes.

Spark SQL remains the public `spark_sql` target while the compatibility
profile filename remains `sparksql.json`. SQL syntax validation calls SQLGlot
with `read="databricks"`; no module uses `sparksql` as a SQLGlot dialect.

## Executable gates

- `tests/test_v2_validation.py` covers deterministic metrics, target results,
  component token budgets, separate translation/judge accounting, JSON,
  tracking, and PDF.
- `tests/test_v2_validation_workflows.py` covers the complete judged metric
  catalogue, skips, memory checks, transcript reconstruction, inline
  validation, and the producer-port offline runner.
- `tests/test_v2_assessment.py` covers packaged/custom profiles, exact
  inheritance, cycles, target-specific parity, sizing, cross-file edges,
  review, JSON/Markdown/PDF, and the shared legacy catalogue.
- Legacy validation and complexity characterization suites run alongside the
  v2 suites. The focused Phase 8 CI gate requires 90% combined line/branch
  coverage.
- Architecture, inventory, schema, Ruff, Pyright ratchet, installed-wheel,
  and full test jobs include both feature families.

## Remaining cutover work

Phase 9 conversion composition must call these use cases through ports and
choose concrete tracking/profile/review adapters. Phase 10 adds the operational
`validate` and `assess` subcommands, then removes the top-level compatibility
packages and references. Those tasks are tracked by G-010, G-012, G-013,
G-014, and G-021 rather than by duplicate Phase 8 gaps.
