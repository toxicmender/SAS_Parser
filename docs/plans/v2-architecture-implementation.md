# SAS migration v2: consolidated architecture and implementation plan

Status: implementation in progress. Phases 0 through 8 and the Phase 9
conversion, hydration, infrastructure, SharePoint Graph, and advanced
knowledge-retrieval slices are implemented on the v2 migration branch. Native
Delta memory persistence and Databricks AI adapters remain in Phase 9 before
the Phase 10 operational cutover.

This is the authoritative plan for the fresh version of the application. It
consolidates the architecture audit, the functional-parity review, the test
coverage findings, and the following product decisions:

1. The only translation targets are **PySpark** and **Spark SQL**.
2. Structured and raw-fallback responses pass through the same target-aware
   normalization and validation contract.
3. Token budgeting is captured per prompt component, per attempt, per item,
   and per run, and is rendered in the validation report.
4. Breaking Python APIs, package names, CLI shapes, and persisted schemas are
   allowed. Existing functional behavior is retained unless this plan marks a
   deliberate v2 change.

The older plan in `docs/plans/pipeline-decoupling-tiktoken.md` remains useful
as historical context but does not govern v2 implementation.

The authoritative open-work ledger is
[`v2-gap-register.md`](v2-gap-register.md), backed by the machine-checked
[`v2-gap-legacy-inventory.json`](v2-gap-legacy-inventory.json). The complete
legacy code and reference index is [`../legacy/README.md`](../legacy/README.md).
Phase sections below define implementation order; they do not create separate
gap or legacy inventories.

## 1. Outcomes

The v2 implementation will deliver:

- one installable `sas_migrate` namespace under `src/`;
- an acyclic dependency graph with pure domain code, application use cases,
  ports, and infrastructure adapters;
- a cleanly installed wheel whose CLI and programmatic APIs work without the
  repository on `PYTHONPATH`;
- behavioral parity for parsing, batching, prompting, memory, validation,
  assessment, XREF, conversion, hydration, reporting, and deployment;
- target-safe output handling for both structured and raw model responses;
- auditable token budgets that distinguish estimated prompt composition from
  provider-reported billing;
- mandatory coverage and integration gates for production-critical paths.

## 2. Scope and deliberate changes

### 2.1 Supported targets

`ResolvedTarget` has exactly two values:

| key | display name | code language | syntax validator |
|---|---|---|---|
| `pyspark` | PySpark | Python | `ast.parse` plus target fence/language validation |
| `spark_sql` | Spark SQL | SQL | `sqlglot` with the Databricks dialect |

`scala`, `Spark Scala`, and all Scala aliases are rejected at every public
boundary: CLI, configuration, SharePoint request projection, programmatic API,
notebook rendering, XREF dispatch, validation, and assessment profile lookup.

The following Scala references are not application targets and must not be
removed accidentally:

- Delta Lake's `SCALA_BINARY_VERSION` used to resolve the Spark JVM artifact;
- generic parsing of historical Markdown fences, when preserving an old raw
  artifact byte-for-byte. Such a fence is evidence of an off-target response,
  not a supported target.

The current resolver already exposes only PySpark and Spark SQL. The v2 work is
therefore a consistency sweep: remove stale architecture text, remove any
target-specific Scala mappings, retain rejection tests, and ensure no adapter
can reintroduce Scala as a target.

### 2.2 Breaking changes

- Imports move from top-level packages such as `chunker`, `pipeline`, and
  `validation` to `sas_migrate.*`.
- The command becomes `sas-migrate` with explicit subcommands.
- Configuration gains `config_version = 2` and typed sections.
- Run facts, validation history, token accounting, and response artifacts use
  new v2 schemas. No in-place migration of old Delta/parquet records is
  required; v2 writes to fresh targets.
- A response that cannot be normalized and validated for its resolved target
  is not publishable. The raw response remains available as an audit artifact.

### 2.3 Non-goals

- Replacing the SAS parser with a third-party parser.
- Proving semantic equivalence through deterministic validation alone.
- Treating estimated tiktoken counts as provider billing.
- Making production external services mandatory for fast unit tests.
- Preserving undocumented private functions or old import paths.

## 3. Functional parity ledger

The following capabilities are in scope and must have characterization or
contract tests before their current implementations are removed.

### 3.1 SAS analysis and batching

- source-preserving scanning, regions, line offsets, diagnostics, timeouts;
- DATA, PROC, macro, global and control chunk classification;
- oversized split windows and overlap behavior;
- dataset input/output, implicit `_LAST_`/`_DATA_`, macros, macro variables,
  functions, CALL routines, paths, LIBNAME engines and external references;
- single- and multi-file edge discovery in corpus order;
- nearest-preceding producer semantics and sorted producer indexes;
- weak-edge resolution, context absorption, batch reasons and stable ordering;
- dependency batches, merged singletons, and token-budgeted packing;
- source/dataset/macro metadata serialization.

### 3.2 Translation and artifacts

- per-item target resolution and Spark SQL-to-PySpark fallback derived from
  compatibility profiles;
- prompt construction, target directives, batch attribution and diagnostics;
- sync/async LLM invocation, structured output, raw fallback, retries,
  provider throttling and prompt-cache fallback;
- inline validation, corrective retry, history rollback and final-attempt
  persistence;
- run facts, resume, validation-aware rewind, fork and recovery;
- effective-prompt Markdown artifacts;
- structured Analysis/Mapping/Translation/Risks documents;
- one notebook per source file and `_cross_file.ipynb` for incomplete
  multi-source attribution;
- token usage, validation summary and reporting artifacts.

### 3.3 Knowledge and instructions

- PDF TOC and font-heuristic extraction with graceful degradation;
- instruction chunking, catalog loading and SHA-based extraction cache;
- construct, hazard and topical retrieval;
- BM25, optional dense retrieval, RRF, reranking and embedding caches;
- user instructions scoped by construct, statement, category, target, chunk
  kind, metadata flag and topic;
- member attribution, fingerprints, focus hints and reasoning directives.

### 3.4 Memory

- in-memory and Delta KV backends behind one behavioral contract;
- chat/thread/message indexing and incremental reads;
- snapshots, restoration, retention, pruning, fork and rewind;
- optional BM25/dense relevant-history selection and token packing;
- rolling thread summaries;
- long-term task policy with fixed and overridable rules;
- thread notes with TTL and fork inheritance;
- memory extraction, temporary-note writes and approval-held policy proposals;
- Delta CDF/audit caches, guarded optimization/vacuum operations and optional
  Databricks AI adapters;
- Spark-free in-memory import and runtime behavior.

### 3.5 XREF

- SharePoint list and CSV mapping sources;
- exact dataset, libref and longest-directory-prefix path resolution;
- `pre`, `post`, and `both` modes;
- SAS-side dataset and physical-path rewriting;
- target-side Spark SQL and PySpark table/path rewriting;
- unresolved macro-path reporting;
- byte-identical output on a non-fatal target rewrite failure;
- one SAS path grammar and one path resolver.

### 3.6 Assessment, validation, conversion and hydration

- complexity tiers, translation parity, file sizing, story points, profile
  inheritance, cross-file analysis, dependency graph, migration order, LLM
  review, Markdown/PDF reports and SharePoint publishing;
- deterministic, judged, RAG, agentic, summarization and memory validation
  metrics;
- offline cases, inline validation, thread/transcript validation, tracking,
  Markdown/PDF reports and separate translation/judge token costs;
- local and SharePoint conversion, request/conversion lists, model/target
  selection, status lifecycle, dry-run behavior, XREF, folder conventions,
  `.sas`/`.txt` loading, upload and per-row failure isolation;
- pure hydration planning, probes, partition strategy, credential resolution,
  Oracle/SFTP/ADLS/Blob/SAS/SPDE readers, ranged I/O, managed Delta sink,
  dry-run and per-item outcomes.

### 3.7 Cross-cutting operations

- typed configuration precedence and secret-free configuration models;
- Azure, Vault, Databricks and SharePoint credential chains;
- SharePoint's single worker-thread/event-loop ownership;
- read-only SharePoint deployment preflight;
- redacted console/file logging, HTTP tracing and unhandled exception capture;
- Docker app/Spark/Vault images, Compose, Spark/Delta compatibility checks and
  dependency warmup.

## 4. Target architecture

```text
src/sas_migrate/
  core/
    sas/
      models.py
      scanner.py
      chunking.py
      paths.py
      metadata/
      dependencies/
      batching.py
    targets/
      models.py
      registry.py
      compatibility.py
      validation.py
    responses/
      models.py
      markdown_parser.py
      normalization.py
      validation.py
    runs/
      models.py
      events.py
    tokens/
      counting.py
      models.py
      policy.py

  application/
    ports/
      llm.py
      memory.py
      source_repository.py
      artifact_repository.py
      credential_provider.py
      validation.py
      clock.py
    translation/
      service.py
      prompting.py
      prompt_assembly.py
      packing.py
      run_state/
    knowledge/
    memory/
    xref/
      target_rewriters/
    validation/
    assessment/
    conversion/
    hydration/
    artifacts/
    operations/

  adapters/
    llm/
    memory/
    sharepoint/
    auth/
    hydration/
    validation/
    documents/
    files/

  config/
  observability/
  cli/
  resources/
    complexity_profiles/
    prompt_instructions/

deploy/
  docker/
  compose/
```

Dependency direction:

```text
CLI/composition -> application use cases -> core
                         ^
                 adapters implement ports
```

Rules enforced in CI:

- `core` imports no application or adapter module;
- application code imports ports and core contracts, never concrete adapters;
- adapters do not import one another to coordinate a use case;
- only the CLI composition root chooses concrete implementations;
- no feature package imports another feature's CLI or concrete service;
- imports between top-level v2 areas form a directed acyclic graph.

## 5. Target resolution and response acceptance

### 5.1 Resolution contract

Target resolution occurs at each public boundary and then travels as a
`ResolvedTarget` object. Free-form strings do not pass beyond the boundary.

Resolution order depends on the public boundary:

- local/programmatic: explicit argument, typed v2 configuration, then default
  `Spark SQL`;
- SharePoint: the request row, the command's explicit fallback, typed v2
  configuration, then default `Spark SQL`.

An item may fall back from Spark SQL to PySpark only when the compatibility
profile says the item is not implementable in Spark SQL and PySpark is
strictly better. There is no reverse fallback and no third target.

### 5.2 Response models

The structured response adds explicit target identity:

```text
TranslationDocument
  schema_version: 2
  target: "pyspark" | "spark_sql"
  analysis: str
  mapping: list[MappingEntry]
  cells: list[TranslationCell]
  risks: list[RiskNote]

TranslationCell
  kind: "code" | "markdown"
  source: str
  language: "python" | "sql" | null
  chunk_id: str | null
```

The model must not be allowed to select the target freely. The prompt contains
the resolved item target, and the returned `target` must equal it.

### 5.3 One normalization path

Every call produces a `ResponseEnvelope`:

```text
ResponseEnvelope
  mode: structured | raw_fallback
  raw_message: provider message
  document: TranslationDocument | null
  structured_error: string | null
  resolved_target: ResolvedTarget
  validation: ResponseValidationResult
```

Processing order:

1. Resolve the run target and the item's optional fallback target.
2. Invoke the target-specific structured schema.
3. If the provider returns a parsed document, normalize it.
4. If structured parsing fails, save the raw provider response and parse its
   four Markdown sections and code fences into the same `TranslationDocument`.
5. Run the same target validation over either document.
6. Only a validated document is persisted as the accepted response and used
   to build notebooks.
7. Persist the original raw response, structured parsing error, normalized
   document and validation result in the audit artifact.

There is no second notebook-only Markdown parser. Raw fallback is normalized
once, and downstream memory, validation, resume and artifact generation use
the normalized document rendered through one canonical Markdown renderer.

### 5.4 Mandatory target validation

`ResponseTargetValidator` checks:

- document target equals the resolved item target;
- every code cell has either the target's canonical language or no language;
- an explicit foreign fence/language is a target mismatch;
- every code cell passes the target syntax checker;
- at least one non-empty code cell exists;
- multi-member cell attribution uses known chunk ids;
- no document mixes PySpark and Spark SQL code cells;
- rendered Markdown round-trips without changing target identity.

For raw fallback:

- an untagged fence is interpreted as the already resolved target, then syntax
  checked;
- an explicit `scala` or other unsupported fence is retained in the raw audit
  text but fails target validation;
- prose without target code fails validation;
- mixed target fences fail rather than being guessed into one language.

Target validation is a mandatory pre-publication gate, separate from optional
quality metrics. A failed document enters the existing corrective retry loop.
After the retry budget is exhausted, the item is recorded as `failed`; its raw
response is retained, but no runnable notebook cell is published. This is a
deliberate v2 behavior change from accepting the final invalid attempt.

The result also appears in validation as the deterministic `response_target`
metric. Its details record structured versus raw-fallback mode, resolved and
reported targets, fence/language mismatches, syntax failures and attribution
errors. The publication gate does not depend on whether the broader validation
suite is enabled; the metric is the reporting view of an always-on check.

### 5.5 Response tests

For each target, cover:

- valid structured response;
- structured response with wrong target;
- structured response with foreign cell language;
- structured response with syntax error;
- provider ignoring the schema but returning valid raw Markdown;
- raw response with untagged valid code;
- raw response with explicit wrong fence;
- mixed PySpark/SQL raw response;
- missing Translation section or empty code;
- malformed Markdown retained as an audit artifact;
- retry repair and exhausted-retry failure;
- resumed/forked runs preserving the normalized document and target result;
- notebook rendering only validated documents.

## 6. Token budgeting and validation reporting

### 6.1 Accounting principles

The system records two different truths and never conflates them:

- **estimated composition**: counts produced from exact prompt components by
  the configured counter before the request;
- **provider usage**: billed/reported input, output, cache read and cache write
  tokens returned after the request.

The provider total is authoritative for cost. The component breakdown is an
estimate used for budgeting and diagnosis. Reports show both, the estimator
and encoding used, and their signed reconciliation delta.

### 6.2 Typed prompt assembly

Prompt text is not flattened before accounting. `PromptAssembly` contains
typed `PromptComponent` records and renders the final message list:

```text
PromptComponent
  category: TokenCategory
  text: str
  message_role: system | user | assistant
  source_id: str | null
  cacheable: bool
  ephemeral: bool
```

Input categories:

| category | content |
|---|---|
| `system_static` | base translation contract and output rules |
| `structured_schema` | structured-output/tool schema when countable |
| `target_directive` | resolved target and fallback explanation |
| `sas_source` | exact SAS text submitted for the item |
| `batch_context` | member ids, source ids, datasets, diagnostics and ordering |
| `reference_guidance` | selected reference-document instructions |
| `project_instructions` | operator/user instruction files and scoped rules |
| `task_policy` | long-term task policy included in the cached prefix |
| `thread_notes` | short-term thread-scoped exceptions |
| `rolling_summary` | rolling conversation summary |
| `selected_history` | history turns selected for this call |
| `retry_feedback` | validation feedback on a repair attempt |
| `chat_framing` | estimated per-message and reply-primer overhead |

Project instructions and reference guidance must remain distinct even if they
are delivered in one system message. This requires the knowledge builder to
return attributed components, not only a rendered string.

Output categories, derived from the normalized document:

| category | content |
|---|---|
| `analysis_output` | Analysis field |
| `mapping_output` | Mapping entries |
| `code_output` | target code cells |
| `markdown_output` | non-code document cells |
| `risk_output` | risk notes |
| `raw_output_overhead` | raw content not represented in the normalized fields |

### 6.3 Per-call and per-attempt records

Every LLM call returns a `CallTokenRecord` associated with run, thread, item
and attempt:

```text
CallTokenRecord
  run_id, thread_id, item_id, attempt
  target
  estimator, encoding, approximate
  estimated_input_by_category
  estimated_input_total
  provider_input_tokens
  provider_output_tokens
  provider_cache_read_tokens
  provider_cache_write_tokens
  provider_total_tokens
  provider_input_delta
  estimated_output_by_category
  accepted_attempt
```

`provider_input_delta = provider_input_tokens - estimated_input_total` is
signed. It may be negative when a gateway/provider tokenizer differs from the
estimator. It is not forced into an invented prompt category.

Retries remain visible. A discarded attempt consumed tokens and contributes to
run cost even though only the accepted response enters conversation history.
Resume does not claim new token use for skipped items; recovered historical
records are identified separately from current-run calls.

### 6.4 Budget policy

`TokenBudgetPolicy` supports:

- `max_input_tokens` per call;
- `reserved_output_tokens` per call;
- `safety_margin_tokens` for tokenizer/gateway variance;
- `max_run_tokens` optional hard run cap;
- optional per-category caps such as `max_sas_source_tokens`,
  `max_instruction_tokens`, and `max_history_tokens`;
- optional warning shares, for example instructions exceeding 40% of the
  estimated input;
- `on_exceeded = reject | shrink_optional_context`.

Required content (`sas_source`, target directive and batch identity) is never
silently truncated. Under `shrink_optional_context`, trimming order is:

1. older selected history;
2. lower-ranked reference guidance;
3. rolling summary compression, if configured;
4. fail before removing project instructions, fixed policy or SAS source.

Packing derives its available budget from the same policy and the same prompt
assembly used at invocation. There must not be a separate packing estimator.

### 6.5 Validation report schema

`ValidationReport` retains separate translation and judge usage and adds:

```text
token_budget: RunTokenBudgetReport | null
judge_token_budget: RunTokenBudgetReport | null
```

The translation report includes:

- provider totals: input, output, total, cache read/write and calls;
- estimated input totals by component;
- estimated output totals by normalized response component;
- per-target, per-item and per-attempt breakdowns;
- retry overhead and discarded-attempt cost;
- packing efficiency: SAS tokens per call and items per call;
- configured caps, peak use, violations and warnings;
- estimator/encoding and provider reconciliation deltas;
- missing-usage status when the provider reports no counts.

The Markdown/PDF report renders a concise aggregate table followed by an
optional item/attempt detail table. JSON stores the nested report directly.
Delta/parquet tracking uses two related fact tables rather than repeating a
large breakdown on every metric row:

- `validation_metric_fact`, keyed by run/case/metric;
- `token_call_fact`, keyed by run/thread/item/attempt and carrying the token
  categories, provider totals and acceptance state.

Reports and tracking must not imply that estimated category counts are billed
provider counts.

### 6.6 Token-budget validation metric

Add deterministic `token_budget_compliance` to the default suite when a token
budget report is present. It evaluates:

- no call exceeded `max_input_tokens` before transmission;
- no accepted item violated a configured category cap;
- run total stayed within `max_run_tokens`, if set;
- reconciliation deltas stayed within an optional warning tolerance;
- every non-resumed call has an attempt-level token record.

No usage data produces `skipped`, not a pass. A preflight rejection is reported
as a budget failure without sending an LLM request.

### 6.7 Token tests

- exact component sums and framing;
- SAS source separated from batch metadata;
- reference guidance separated from project instructions;
- policy, thread notes, summary and history separated;
- retry feedback counted only on retry attempts;
- accepted and discarded attempts aggregated correctly;
- provider totals reconciled without changing component estimates;
- absent/malformed provider usage;
- cache read/write reporting;
- tiktoken-unavailable approximation label;
- prompt packing and invocation using the same budget calculation;
- validation Markdown, PDF, JSON and tracking round-trips;
- secrets and raw credential headers never entering token audit artifacts.

## 7. Detailed implementation sequence

Each phase ends in a mergeable, releasable state. Do not combine code motion
with behavior changes unless the phase explicitly says so.

### Phase 0 - baseline, decisions and emergency packaging repair

Deliverables:

1. Freeze representative golden corpora and output snapshots.
2. Convert the documented load-bearing invariants into named tests.
3. Record current public workflows and v2 replacements.
4. Repair the current wheel configuration so `token_budget` and `validation`
   ship while v2 is being built.
5. Add an explicit build backend and a clean-wheel smoke test.
6. Add branch coverage and publish unambiguous package-relative coverage XML.
7. Update the pyright baseline so every currently clean shipped module is
   gated.

Tests/gates:

- current unit suite remains green;
- built wheel installs in an empty environment;
- `sas-parser --help` and a fake-model local conversion work from the wheel;
- parser/batcher golden fixtures are committed and reproducible.

### Phase 1 - v2 skeleton, contracts and architecture enforcement

Deliverables:

1. Create `src/sas_migrate` and explicit package/resource configuration.
2. Define core ids, result types, ports, errors and versioned serialization.
3. Define `ResolvedTarget`, `TranslationDocument`, `ResponseEnvelope`, token
   models and run-event contracts.
4. Add import-boundary tests and an architecture graph check.
5. Add the new `sas-migrate` CLI shell without operational subcommands.

Tests/gates:

- core imports with only core dependencies installed;
- architecture graph is acyclic;
- resource files appear in the wheel;
- serialization models round-trip with `schema_version = 2`.

### Phase 2 - SAS core extraction

Deliverables:

1. Move models, scanner, metadata extractors, path grammar, chunking,
   dependency discovery and batching.
2. Split large files by edge family and metadata concern without changing the
   single corpus-order discovery walk.
3. Move token packing behind the shared `TokenBudgetPolicy` interface.
4. Retain temporary compatibility adapters from old models only in tests.

Tests/gates:

- golden chunks, metadata, edges, reasons and ordering match;
- source coverage and overlap invariants hold;
- property tests cover source preservation and graph determinism;
- no `app_config`, LLM, memory or SharePoint import reaches SAS core.

### Phase 3 - targets, response normalization and mandatory validation

Deliverables:

1. Implement the two-target registry and compatibility fallback.
2. Remove stale Spark Scala target references and mappings.
3. Implement target-bearing structured schemas.
4. Implement raw Markdown normalization into the same document model.
5. Implement `ResponseTargetValidator` and canonical Markdown rendering.
6. Integrate target failure with retry and failed-item state.

Tests/gates:

- the response matrix in section 5.5 passes for both targets;
- any Scala target input is rejected;
- raw fallback cannot bypass target or syntax checks;
- invalid output cannot produce a runnable notebook.

### Phase 4 - prompt assembly and token accounting

Deliverables:

1. Introduce typed prompt components and render messages from them.
2. Instrument every input category before flattening.
3. Return per-call provider usage from the LLM port.
4. Associate usage with run/item/attempt, including retries.
5. Instrument normalized output fields.
6. Implement budget enforcement, optional-context trimming and run caps.
7. Persist token records with run facts and prompt audit artifacts.

Tests/gates:

- section 6.7 passes;
- existing max-input and packing behavior remains compatible;
- no prompt component is double-counted;
- retry and resume accounting is correct.

### Phase 5 - translation orchestration and artifacts

Deliverables:

1. Implement `TranslateCorpus` against LLM, memory, validation and artifact
   ports.
2. Move prompt formatting, instruction injection and per-item target choice.
3. Move run ledger, resume, rewind, fork and recovery.
4. Move structured/raw response processing onto the phase 3 contract.
5. Move effective prompts, canonical Markdown and notebook production.
6. Preserve multi-source attribution and `_cross_file.ipynb` behavior.

Tests/gates:

- fake-model end-to-end translation for both targets;
- resume/fork/retry characterization tests;
- notebook and prompt golden tests;
- one accepted turn and one accepted token record per item.

### Phase 6 - knowledge and memory

Deliverables:

1. Move PDF ingestion, instruction chunking, catalog/cache and user rules.
2. Move retrieval and return attributed `PromptComponent` objects.
3. Move memory history, relevance selection, summarization, policy, thread
   notes, extraction and context assembly.
4. Implement in-memory and Delta adapters separately.
5. Move CDF, retention, audit and maintenance operations.

Tests/gates:

- retrieval and memory contract suites pass;
- ephemeral content never enters chat history;
- prompt categories remain distinguishable in token reports;
- in-memory installation imports and runs without Spark;
- dedicated Delta job passes without skips.

### Phase 7 - XREF

Deliverables:

1. Move mapping models and the single path resolver.
2. Move SAS pre-rewriting and target post-rewriters.
3. Implement SharePoint and CSV mapping-source adapters.
4. Preserve `pre`, `post`, `both`, unresolved reporting and byte-identical
   non-fatal fallback.

Tests/gates:

- existing XREF characterization suite passes;
- no Scala target dispatch exists;
- source grammar is imported from SAS core rather than copied;
- XREF application makes no network call unless its source adapter is invoked.

### Phase 8 - validation and assessment

Deliverables:

1. Move validation models, deterministic metrics and evaluator.
2. Move judged, RAG, agentic, summarization and memory metrics.
3. Move offline runner, inline validator, thread/transcript reconstruction and
   history tracking.
4. Add token budget models, `token_budget_compliance`, report tables and
   tracking schema.
5. Move complexity profiles, scoring, sizing, cross-file analysis, dependency
   graph, LLM review and reports.
6. Make both features consume run/core contracts, never the translation
   service implementation.

Tests/gates:

- all metric names, thresholds and skip semantics are preserved;
- target and token validation appear in Markdown/PDF/JSON reports;
- translation and judge usage remain separate;
- assessment output and profile inheritance match golden fixtures;
- neither feature imports another feature's CLI or concrete service.

### Phase 9 - conversion, hydration and infrastructure adapters

Deliverables:

1. **Implemented:** move local and SharePoint conversion workflows onto
   repositories/ports.
2. **Partially implemented:** request status lifecycle, model/target selection,
   and source paths are v2-owned. SharePoint publication remains with the
   Phase 10 presenter/publication cutover gate.
3. **Implemented:** move hydration planning into application code and all
   driver/sink boundaries into adapters, including ranged I/O and Delta.
4. **Implemented:** split secret-free settings from Azure/Vault/Databricks/
   SharePoint infrastructure and place credentials behind ports.
5. **Implemented:** move the concrete Graph transport while preserving
   SharePoint worker-loop ownership and versioned read-only preflight
   diagnostics.
6. **Implemented:** move logging/redaction and HTTP tracing to observability.
7. **Implemented:** move BM25/FAISS knowledge ranking, reciprocal-rank fusion,
   reranking, and provider-scoped memory/disk embedding caches behind lazy v2
   ports and adapters.
8. **Pending:** move physical Delta memory MERGE/CDF behavior out of
   `memory.store`.
9. **Pending:** move Databricks chat and embedding factories out of
   `memory.databricks_ai`.

Tests/gates:

- local and SharePoint fake-adapter end-to-end workflows pass;
- failure paths update status once and preserve per-row isolation;
- hydration planning is pure and driver imports remain lazy;
- adapter extras each get an install/import/contract CI job;
- SharePoint site-resolved drive tests run with no pinned drive id.
- advanced knowledge retrieval passes lexical, dense, fusion, reranking,
  cache persistence/corruption, lazy-import, and application integration gates.

### Phase 10 - CLI, reporting, deployment and cutover

Deliverables:

1. Implement subcommands:
   - `sas-migrate convert local`;
   - `sas-migrate convert sharepoint`;
   - `sas-migrate assess`;
   - `sas-migrate validate`;
   - `sas-migrate hydrate`;
   - `sas-migrate check sharepoint`;
   - `sas-migrate memory status|optimize|vacuum`.
2. Move Markdown/PDF/notebook presenters and SharePoint publication adapters.
3. Update Docker images, Compose files and Spark/Delta warmup.
4. Replace Architecture.md with a short overview plus focused ADRs.
5. Remove old package and CLI compatibility layers.
6. Publish v2 configuration, API, schema and operator migration guides.

Tests/gates:

- every subcommand runs from the installed wheel;
- CLI target choices list only PySpark and Spark SQL;
- report and notebook visual/golden checks pass;
- Docker app and Spark/Delta smoke tests pass;
- repository contains no stale application-target claim for Spark Scala;
- old packages are absent from the wheel.

## 8. Test and CI design

```text
tests/
  unit/
    core/
    application/
  characterization/
    parser/
    batching/
    outputs/
  contract/
    llm/
    memory/
    source_repositories/
    artifact_repositories/
    hydration/
  integration/
    delta/
    sharepoint/
    gateway/
    hydration_drivers/
  e2e/
    wheel/
    cli/
    docker/
  quality/
    model_evaluations/
```

Coverage policy:

- branch coverage enabled;
- core: at least 95% line / 90% branch;
- application: at least 90% line / 85% branch;
- adapters: at least 80% line / 70% branch, with contract tests mandatory;
- changed-line coverage at least 90%;
- no global percentage may hide a package below its own threshold;
- CLI modules, token accounting and installed-wheel behavior are measured.

CI jobs:

1. formatting/lint and architecture boundaries;
2. core unit/property tests with minimal dependencies;
3. full offline application suite;
4. clean wheel install and CLI smoke tests;
5. Spark/Delta contract tests where a skip is a failure;
6. SharePoint/Graph extra contract tests;
7. Databricks/auth extra contract tests;
8. hydration adapter matrix;
9. Docker build and warmup smoke tests;
10. scheduled real-model quality evaluations with cost limits.

## 9. Suggested commit sequence

1. `test(architecture): freeze v1 behavior and invariants`
2. `fix(packaging): ship all current runtime packages`
3. `build(v2): add src namespace and wheel smoke test`
4. `feat(core): add v2 targets response and token contracts`
5. `refactor(sas): move scanner and chunking into v2 core`
6. `refactor(sas): move metadata dependencies and batching`
7. `feat(responses): normalize and validate structured and raw output`
8. `feat(tokens): add typed prompt assembly and call records`
9. `feat(tokens): enforce shared budgets and retry accounting`
10. `refactor(translation): move orchestration and run state`
11. `refactor(artifacts): move prompt markdown and notebook outputs`
12. `refactor(knowledge): move ingestion retrieval and user rules`
13. `refactor(memory): move services and in-memory adapter`
14. `refactor(memory): move Delta adapter and operations`
15. `refactor(xref): move mapping and target rewriters`
16. `refactor(validation): move metrics runners and tracking`
17. `feat(validation): report component token budgets`
18. `refactor(assessment): move scoring reports and profiles`
19. `refactor(conversion): move local and SharePoint use cases`
20. `refactor(hydration): move planning drivers and sink`
21. `refactor(infrastructure): split config auth and SharePoint adapters`
22. `feat(cli): add v2 subcommands and composition root`
23. `build(deploy): migrate containers and compatibility checks`
24. `docs(v2): publish ADRs and migration guides`
25. `refactor(v2): remove legacy packages and entry points`

Each commit must leave its selected test tier green. Commits that change a
serialized contract include the schema fixture and migration note together.

## 10. Completion criteria

The work is complete only when:

- the installed wheel contains one `sas_migrate` namespace and all resources;
- all current user workflows have a v2 command or programmatic API;
- PySpark and Spark SQL are the only resolved targets;
- raw fallback cannot bypass target and syntax validation;
- invalid target output is retained for audit but is never published as a
  runnable notebook;
- token reports reconcile prompt components, attempts, provider usage, cache
  usage, translation cost and judge cost without conflating estimates with
  billing;
- all load-bearing parser, batching, memory, XREF and SharePoint invariants are
  executable tests;
- dependency-boundary tests report no cycle;
- mandatory Delta and adapter jobs pass without silent skips;
- Markdown, PDF, JSON and tracking outputs contain target-validation and token
  budget results;
- v2 configuration, persisted schemas, CLI and operational runbooks are
  documented;
- old top-level packages and legacy entry points are removed.
