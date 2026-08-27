# V2 operational CLI contracts

Phase 10 moves operations onto the `sas-migrate` console script in bounded
slices. The first slice exposes the two offline report workflows without
loading Spark, Graph, Databricks Model Serving, or credentials.

## Assessment

```text
sas-migrate assess UNITS.json \
  --target pyspark|spark_sql \
  --format json|markdown|pdf \
  [--output REPORT] [--profiles DIRECTORY]
```

`UNITS.json` is a JSON array of existing `AssessmentUnit` contracts. The
default target is `spark_sql`; Spark Scala is not accepted. Packaged profiles
are used unless `--profiles` points to a directory containing `pyspark.json`
and `sparksql.json`.

## Validation

```text
sas-migrate validate RUN.json \
  [--model LABEL] \
  [--translation-ledger LEDGER.json] \
  [--judge-ledger LEDGER.json] \
  [--translation-policy POLICY.json] \
  [--judge-policy POLICY.json] \
  --format json|markdown|pdf [--output REPORT]
```

`RUN.json` is the existing versioned `EvaluationRun` contract. Optional token
files use `TokenCallLedger` and validation `TokenBudgetPolicy`; translation and
judge accounting remain separate in every report. Validation exits zero only
when deterministic metrics, target-resolution validation, and supplied budget
policies pass. It still writes the report when validation fails.

JSON is the default and may be written to stdout. Markdown may be written to
stdout or a file. PDF requires `--output` so binary data is never emitted to a
terminal. Invalid contracts and filesystem errors return status 2 with an
operator-facing diagnostic.

The clean-wheel smoke executes both commands with packaged profiles and no
repository imports. Focused unit tests cover successful and failed validation,
unsupported targets, token budgets, all report formats, invalid contracts,
and I/O failures at a 90% combined line/branch CI threshold.

G-013 remains open. Later Phase 10 slices add conversion, hydration,
SharePoint preflight, memory maintenance, and remove the legacy `sas-parser`
entry point.
