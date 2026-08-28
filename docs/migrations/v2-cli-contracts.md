# V2 operational CLI contracts

Phase 10 moves operations onto the `sas-migrate` console script in bounded
slices. The offline report workflows do not load Spark, Graph, Databricks Model
Serving, or credentials. Local conversion adds the first live provider path.

## Local conversion

```text
sas-migrate convert local SOURCE_DIRECTORY \
  [--output-dir ARTIFACT_DIRECTORY] \
  [--target pyspark|spark_sql] [--model MODEL] \
  [--gateway-base-url URL] [--gateway-version VERSION] \
  [--api-key-env ENVIRONMENT_VARIABLE] \
  [--max-input-tokens N] [--reserved-output-tokens N] \
  [--safety-margin-tokens N] [--max-run-tokens N] \
  [--max-attempts N] [--dry-run]
```

The command discovers `.sas` and `.txt` files, runs the migrated semantic
chunker and cross-file batcher, invokes an OpenAI-compatible gateway through
the v2 `LLMPort`, and applies the same structured/raw-fallback target validator
before writing canonical Markdown, notebooks, attempt audits, token records,
and a run summary. Spark SQL validation uses SQLGlot's `databricks` dialect.

The gateway credential is read only at runtime from
`SAS_MIGRATE_GATEWAY_API_KEY`, or the variable named by `--api-key-env`.
Settings documents may contain the variable name, gateway URL, version,
timeout, and retry count, but reject an `api_key` value. The adapter requests
the versioned `TranslationDocument` JSON schema while retaining raw output for
the mandatory fallback validator when a gateway ignores structured output.

`--dry-run` resolves no credential and constructs no provider client. It writes
a versioned `conversion-plan.json` containing source names, target identity,
SQLGlot dialect, model, and the complete token policy. Live and dry-run results
use the existing `ConversionBatchOutcome` contract and return status 1 when a
request fails; invalid operator configuration returns status 2.

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

G-013 remains open. Later Phase 10 slices add SharePoint conversion, hydration,
SharePoint preflight, memory maintenance, and remove the legacy `sas-parser`
entry point.
