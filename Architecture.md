# Architecture

SAS_Parser turns Base SAS source into LLM-ready work items. It has three
layers, each usable on its own:

1. **Chunker** — splits SAS source into source-preserving semantic chunks
   (DATA steps, PROC steps, macro definitions, …) with extracted metadata.
2. **Batcher** — discovers dataset/macro/macro-variable dependencies between
   chunks (within and across files) and groups inter-dependent chunks into
   batches that must be translated together.
3. **Pipeline** — feeds work items, in dependency order, through a
   LangChain/LangGraph chat model with per-run conversational memory
   persisted to a KV store (in-memory dict locally, Databricks Delta in
   production). Every LLM call is made per `SasBatch`: the batcher's ordered
   items are coalesced first, so dependency batches and merged runs of
   independent singletons are the only units prompted — and, with
   token-budgeted packing (on by default; `max_merged_tokens`, `0` to
   disable), adjacent small items share a call as `packed-NNN` batches
   under a prompt-cost budget. The deliverable is a
   **notebook** — one `.ipynb` per SAS source file (`pipeline.notebook`) —
   because the output is code and code should be runnable. Multi-source
   batches split back per file via per-cell `chunk_id` attribution (the
   structured prompt asks for it; all-or-nothing per item, falling back to
   a shared `_cross_file.ipynb` when attribution is incomplete); the two
   report surfaces, validation and complexity, are **Markdown**.

An optional fourth component, **prompt_builder**, reads reference PDFs (SAS
manuals, target-platform guides) into retrieval-ready instruction chunks and,
when passed to the pipeline, injects per-item guidance relevant to each work
item's constructs — prompted to the LLM but never persisted (see invariant 5).

A fifth, **complexity**, scores chunks and batches for migration effort on two
axes — a LOW/MEDIUM/HIGH data-complexity tier (a property of the SAS source)
and a feature-parity rating against the output language (a property of the
SAS/target pair) — for triage and estimation. It also sizes each **source
file** on the agile T-shirt scale (Small/Medium/Large/Extra Large, with
Fibonacci story points), which the two presence-based axes cannot express: a
long file of trivial steps raises no signal at all yet is still real work.
Sizing combines three declared dimensions — effort, complexity, and
uncertainty — measured relative to a documented reference file, and accounts
for what each file borrows from and lends to the rest of the corpus. The
catalogue that assigns those ratings is JSON data under
`complexity/profiles/`, one file per target, so the analysis retargets from
Spark SQL to PySpark (or anything else) without a code change. It delivers two
levels of Markdown — one corpus report plus one per source SAS script, each
printing the chunk text behind every verdict — and can optionally ask an LLM
where its rules are wrong. It reads the chunker's output and is deliberately
not wired into the pipeline.

```
                 +----------------------+
  SAS source(s) ─▶ SasSemanticChunker   │──▶ SasChunkResult (per file)
                 +----------------------+
                          │  SasCorpus wraps N file results
                          ▼
                 +----------------------+
                 │ MultiFileBatcher /   │──▶ SasBatchResult
                 │ SasChunkBatcher      │    (batches + singletons)
                 +----------------------+
                          │  all_ordered_items → coalesce_into_batches
                          ▼
                 +----------------------+       +--------------------+
                 │ SasLLMPipeline       │──────▶│ memory.store   │
                 │ (LangGraph graph,    │ turns │ (KV chat history)  │
                 │  one thread per run) │       +--------------------+
                 +----------------------+
                    ▲ ephemeral   │
                    │ guidance    ▼
   +----------------------+   LLM responses, one per SasBatch
   │ prompt_builder       │◀── reference PDFs (SAS + target manuals)
   │ (PromptBuilder, opt) │
   +----------------------+
```

## Package layout

```
main.py                 The entry point (console script: `sas-parser`).
                        Argument parsing, validation-before-work, credential
                        resolution, pipeline construction, and dispatch —
                        nothing else; the per-request orchestration lives in
                        conversion/run.py so it is testable without the CLI.
                        SharePoint is the DEFAULT flow and a positional source
                        directory is the explicit opt-out into local mode.
                        Neither falls back to the other silently.

chunker/
  models.py             Pydantic models: SasChunk(+Kind), SasChunkMetadata,
                        SasChunkResult, SasCorpus, SasBatch, SasBatchResult,
                        SasDiagnostic(+Severity)
  keywords.py           SAS keyword catalogues transcribed from the SAS docs
                        (reserved macro words, autocall macros, function /
                        CALL-routine dictionaries) + patterns compiled from
                        them. Pure data; no package imports, no logging.
  scanner.py            Lexical layer: _Unit/_Region parse primitives, the
                        statement classifier (_classify), text normalisation
                        and sanitisation, line-offset helpers, and the
                        _Deadline/_ParseWatchdog stuck-parser machinery.
  metadata.py           Per-chunk semantic extraction: _metadata_for, _io_for
                        (directed dataset I/O), _macro_body_io (literal vs
                        parameterised body refs), symput / SQL-INTO / CALL
                        EXECUTE extractors, _merge_meta, and the extraction
                        regex catalogue.
  chunker.py            SasSemanticChunker orchestration (scan → group →
                        build chunks, oversized-split with overlap).
  batcher.py            _EdgeDiscovery + Union-Find grouping, weak-edge
                        resolution, context absorption, batch construction.
                        SasChunkBatcher is a one-file convenience over
                        MultiFileBatcher.
  _repl.py              print_iterable REPL helper (also used by conversion.run
                        to render per-item summary lines into its logs).
pipeline/
  setup.py              The pipeline constructor's grouped configs, one per
                        concern: MemorySetup (store hub, task policy, thread
                        memory, extractor, chat identity, history policy —
                        plus the cross-injection logic in build()),
                        ChunkingSetup, PromptingSetup, ValidationSetup. LLM
                        transport groups under llm_client.LLMClientConfig the
                        same way. ONE SPELLING PER KNOB: these are fields on
                        those objects, never keyword arguments on the pipeline
                        as well — two places to set one thing means a
                        rejection branch to stop them being set in both, which
                        is what the 22-argument constructor used to carry.
  run_ledger.py         RunLedger: KV-side run bookkeeping — per-item
                        run/validation facts, resume (skip/redo resolution
                        and the rewind), and the fact-copying half of fork.
                        Never calls an LLM.
  engine.py             SasLLMPipeline: the LangGraph StateGraph wiring,
                        memory/validation integration, resume and fork, and
                        opt-in Anthropic prompt caching on the system prompt.
  prompting.py          Item -> retrieval query / construct keys / scope
                        tokens (the sole SAS-metadata -> prompt_builder
                        mapping) and chunk/batch prompt formatting.
  constants.py          Prompt templates — the Markdown-sections system
                        prompt and its structured-output counterpart.
  response_models.py    Pydantic models for the structured answer the pipeline
                        asks for: TranslationDocument (analysis, mapping,
                        ordered cells, risks) + to_markdown(), which renders it
                        back to the four Markdown sections that get persisted
                        and scored. Pydantic only; no langchain import.
  notebook.py           Renders pipeline outputs as nbformat v4.5 notebooks —
                        one .ipynb per SAS source file — from a
                        TranslationDocument (multi-source items split per
                        file by each cell's chunk_id; unattributed items land
                        in _cross_file.ipynb with pointers), or by parsing
                        the Markdown response when there is none.

llm_client/
  tokens.py             Shared tiktoken-backed token estimation: model id ->
                        encoding by explicit prefix map (o200k_base for the
                        modern GPT families and every non-OpenAI id,
                        cl100k_base for older GPTs), ChatML-style message
                        counting, chars//4 degradation when the encoding
                        data cannot load. A leaf module.
  client.py             LLMClient / LLMClientConfig: chat-model construction
                        over the AI Gateway's OpenAI-compatible API
                        (temperature, max output tokens, endpoint overrides —
                        base_url / api_key / headers / timeout / model_kwargs,
                        proactive InMemoryRateLimiter) and sync + async
                        invocation (input-token budget via llm_client.tokens
                        -> InputTokenLimitError, transient-error retry with
                        exponential backoff). WHICH client is built follows the
                        model's provider (provider_client="auto"): anthropic
                        -> a raw openai.OpenAI wrapped to the surface this
                        class invokes, openai/google -> ChatOpenAI, anything
                        else -> LLMClientError. That is what the reference
                        deployment does against this same gateway; an explicit
                        provider_client pins it either way.
                        from_ai_gateway() / from_vault_secret() are the two
                        constructions that perform I/O, which is why they are
                        classmethods rather than defaults.
                        Imports nothing from chunker or memory.

memory/
  turns.py              Dependency-light turn grouping + the package's
                        default token counter (tiktoken o200k_base when its
                        data is loadable, else the ~4-chars/token estimate),
                        shared by relevance and summarize (so summarize never
                        imports the bm25s/faiss stack). A leaf module.
  relevance.py          HybridRanker: shared BM25 (bm25s) + optional dense
                        retrieval (LangChain Embeddings), RRF fusion,
                        optional reranker hook, content-hashed embedding and
                        tokenization caches — with a stateless per-call mode
                        (dense scores are a numpy matrix-vector product over
                        the normalised vectors; ties break toward recency)
                        and an index-once/query-many static-corpus mode
                        (FAISS IndexFlatIP, imported lazily inside index() so
                        BM25-only pipelines never pay for faiss) — plus
                        DiskCachedEmbeddings (on-disk .npz document-embedding
                        cache). RelevantHistorySelector layers history policy
                        on top: relevance-based selection of prompted history
                        turn pairs, always-keep-last tail, recency fallback,
                        optional max_tokens packing.
                        Imports nothing from chunker or memory.store.
  summarize.py          RollingSummarizer: one rolling summary per thread —
                        turns older than a recency tail fold monotonically
                        into a KV-stored summary (prompted, never persisted).
                        Store is duck-typed; imports only memory.turns.
  store.py              KVStore façade over two backends
                        (_InMemoryBackend dict / _DeltaBackend Spark+Delta),
                        KVChatMessageHistory (BaseChatMessageHistory, with
                        optional after-write retention and the chat:: index
                        that gives Thread -> Chat -> Message), ThreadMemoryManager
                        (incl. fork_thread), KVMemoryStore (optional injected
                        HybridRanker upgrades kv.search to hybrid retrieval),
                        and the MemoryHub entry-point façade.
  policy.py             TaskPolicy: long-term, task-scoped standing
                        instructions (policy::{task_id}), each flagged
                        overridable or fixed, with a content fingerprint.
                        Rendered into the cacheable system prompt. Store is
                        duck-typed; imports nothing from memory.
  thread_mem.py         ThreadMemory: short-term, thread-scoped notes and
                        exceptions (tmem::{thread_id}) with TTL expiry and a
                        fork() that follows memory.store's thread fork.
                        Prompted, never persisted. Store is duck-typed.
  context.py            MemoryContext.assemble(): resolves both instruction
                        channels for one turn — policy -> system suffix,
                        notes -> ephemeral messages — and states their
                        precedence. Duck-typed on both; imports neither.
  extractor.py          MemoryExtractor: classifies a kept turn into
                        permanent (a policy proposal held for approval) or
                        temporary (a thread note, applied), behind an offline
                        cue gate. Never raises; model and store duck-typed.

prompt_builder/
  models.py             Pydantic models: InstructionChunk, DocSection,
                        InstructionDoc, InstructionDiagnostic, ConstructKey
                        (frozen), DocRole / ExtractionStrategy.
  pdf_reader.py         PdfReader: reference PDF -> DocSections via a TOC or a
                        font-heuristic strategy, with shared text cleanup,
                        SAS-title -> ConstructKey parsing, and chunker-style
                        diagnostics (never raises).
  doc_chunker.py        InstructionChunker: DocSections -> word-budgeted
                        InstructionChunks (same-parent merge, oversized
                        paragraph-window split, breadcrumb prefix).
  catalog.py            DocumentSpec, default_catalog (the bundled
                        reference_docs set), and CorpusLoader with a per-doc
                        on-disk extraction cache keyed by file SHA-256.
  selector.py           InstructionSelector: construct-key lookup (hazard-
                        first, stop-listed) + HybridRanker topical ranking
                        under a word budget; the dense stage uses
                        memory.relevance.DiskCachedEmbeddings. Imports
                        memory.relevance only.
  builder.py            PromptBuilder façade: read -> chunk -> index at
                        construction, then build(query, constructs) -> a
                        Markdown guidance block or None.
  user_instructions.py  UserInstructionSet: operator-supplied rules (plain
                        str / file) -> scoped InstructionChunks (always /
                        when:constructs / topic). Selected ahead of all
                        reference tiers, rendered as a separate "Project
                        instructions" block, fingerprinted for validation
                        run history. Degrades toward over-inclusion, never
                        raises.

app_config/
  __init__.py           Dependency-free loader for the repo-root config.json
                        (word/token limits). Precedence: explicit constructor
                        argument > config.json > hard default; JSON null means
                        "unset". Searched via SAS_PARSER_CONFIG env var, cwd,
                        then repo root. A leaf package — imports nothing.
  vault.py              HashiCorp Vault credential client (get_secret).
                        Non-secret connection settings from VAULT_* env vars >
                        config.json `vault`; token / AppRole creds from the
                        environment only. Auth: token > approle > azuread
                        (Entra ID OIDC — JWT minted via azure.py, presented to
                        Vault's jwt auth mount). hvac imported lazily
                        (extra: vault).
  databricks.py         Databricks workspace settings: host, warehouse/cluster,
                        Unity Catalog catalog+schema (full_table_name()), and
                        the credential. Auth: notebook (on-cluster) > pat
                        (DATABRICKS_TOKEN) > azure-ad (via azure.py). SDK /
                        SQL connector imported lazily (extra: databricks).
  azure.py              Microsoft Entra ID auth: AzureAuthClient.get_token()
                        via client_credentials (secret or certificate) or
                        device_code, with per-scope expiry-aware caching.
                        Client secret from AZURE_CLIENT_SECRET only; msal
                        imported lazily (extra: azure). `verify` / `proxies`
                        reach MSAL's own session, which is what makes the
                        login work on a TLS-intercepting network.
  sharepoint.py         Microsoft Graph transport: folders, files, and list
                        items, addressed drive-relative (SharePointConfig
                        .drive_path owns the joining, and strips the
                        "Shared Documents/" prefix a SharePoint URL shows).
                        Authenticates as SharePoint's OWN service principal
                        from the Databricks secret scope (saact-hsv-*), or the
                        shared azure.py identity when no scope is configured.
                        Domain-free: what a folder or a column MEANS belongs
                        to conversion/, xref/, and complexity/sharepoint.py.
                        msgraph-sdk imported lazily (extra: sharepoint).
  sharepoint_check.py   Read-only preflight for the SharePoint deployment
                        (`python -m app_config.sharepoint_check`, or
                        `sas-parser --check`): resolves the config REPORTING
                        THE SOURCE of each value, reads the service principal
                        out of the Databricks secret scope, mints a Graph token
                        and decodes its `roles` claim (the granted application
                        permissions — the 403 diagnosis), then reads the
                        library and each configured list. Writes nothing and
                        calls no model. --offline stops after the config.
  logging_setup.py      Console/file logging for the three CLI entry points.
                        --debug does NOT raise the HTTP transport libraries
                        (TRANSPORT_LOGGERS) to DEBUG; --trace-http is the
                        opt-in for the wire. A RedactingFilter masks bearer
                        tokens and secret-shaped key/values on every handler.
                        Standard library only.

reporting/
  pdf.py                Markdown -> PDF: markdown-it parses, PyMuPDF's Story
                        lays out. ONE implementation for both report surfaces
                        -- complexity renders to a file beside the Markdown,
                        validation to bytes for upload, and they share the
                        stylesheet, the code folding (Story CLIPS a wide <pre>
                        line rather than wrapping it), the image archive, and
                        the paging-loop cap. A leaf: imports nothing from this
                        repo.

conversion/
  paths.py              The folder conventions one application's scripts live
                        under: scripts_original, scripts_converted,
                        scripts_converted/validation, and the run's own
                        {model}/{timestamp} upload target.
  requests.py           The requests and conversions lists: the SharePoint
                        INTERNAL column names (transcribed, encoded characters
                        and trailing spaces included), row projection, and the
                        one write this repo makes to a list --
                        update_request_status.
  sources.py            Discovering and loading an application's SAS sources
                        (.sas + .txt). No temporary files: the text goes
                        straight to chunk_text with the library path as its
                        source_id.
  upload.py             Writing converted scripts and validation artefacts
                        back, delegating notebook rendering to
                        pipeline/notebook.py rather than reimplementing it.
  run.py                The orchestration over the other four: read a row's
                        scripts, apply XREF, translate the application as ONE
                        corpus on one thread, upload, write the row's Status
                        (on the failure path too). Takes a transport and a
                        pipeline FACTORY, so a whole run is testable without a
                        network or an LLM. Also the run-reporting helpers the
                        CLI prints through.

xref/
  sourcing.py           XREF rows for one application, classified into exact /
                        by_libref / by_path. Title carries an optional type
                        marker; unmarked rows are table mappings, so existing
                        rows need no backfill. Also the CSV file backend
                        (load_databricks_mapping_sharepoint) -- fetching a
                        mapping is this package's job, never chunker's.
  apply.py              WHEN the substitution happens: "pre" (default, over
                        the SAS-side metadata via chunker.batcher
                        .replace_dataset_names), "post" (over generated code),
                        or "both" (each, reporting what only post reached).
  pre.py                The other half of "pre": the physical paths in
                        LIBNAME / INFILE / %INCLUDE, keyed off by_path. Runs
                        over the RAW TEXT before chunking, which is what keeps
                        chunker and pipeline free of XREF knowledge. Matched by
                        statement, not by blind sweep, and keys are applied
                        longest-first (a path is routinely a prefix of a longer
                        one). A path carrying an unresolved macro reference is
                        counted and reported, never guessed at.
  rewrite.py            The post-conversion rewriter: sqlglot for Spark SQL,
                        the ast module + source-span substitution for PySpark
                        (so comments and formatting survive). Unparseable
                        input is returned BYTE-IDENTICAL -- a rewriter that
                        corrupts generated code is worse than one that no-ops.

target_language/
  __init__.py           The run's target output language as one resolved
                        object: TargetLanguage (display name, fence tags,
                        notebook kernel/cell language, complexity profile,
                        syntax checker) + resolve_target_language(), which
                        folds spelling ("SparkSQL"/"spark sql"/"sql" are one
                        target) and raises UnknownTargetLanguage rather than
                        degrading to a Python run. PySpark, Spark SQL, Spark
                        Scala. A leaf package — stdlib only — imported by
                        pipeline, prompt_builder, validation, and complexity,
                        which is what keeps them agreeing on one target.
                        sqlglot core dependency for the SQL check.

validation/
  models.py             Pydantic models: ValidationCase, CaseRun,
                        MetricResult, CaseResult, ValidationReport
                        (score/passed are computed fields; to_markdown()).
  metrics.py            Deterministic metrics + default_metrics(language):
                        response_coverage, dataset_fidelity,
                        language_compliance, target_syntax, required_terms,
                        reference_similarity. The two language-aware ones
                        score against the run's target. target_syntax parses
                        the selected target language. Thresholds resolve via app_config
                        (validation.<name>_threshold).
  judge.py              LLMJudgeMetric — opt-in LLM-as-judge (any
                        LangChain-style model / llm_client.LLMClient);
                        never part of default_metrics().
  runner.py             ValidationRunner: cases -> SasLLMPipeline -> metrics
                        -> ValidationReport; fresh thread_id per case run.
  dataset.py            load_cases(): *.json case files (inline sas_source
                        or a sibling sas_path).
  tracking.py           log_report() / load_runs(): Spark-backed run history,
                        one row per (run, case, metric) — local parquet dir
                        (./validation_runs) by default, saveAsTable (Delta)
                        via `table` on Databricks. Spark boots lazily inside
                        these two functions only.
  __main__.py           CLI: python -m validation <cases_dir> [--judge-model
                        ...] [--track] [--md report.md] [--pdf report.pdf];
                        exit code gates CI.
  cases/                Sample cases. Like tests/, the package does not ship
                        in the wheel.

complexity/
  tiers.py              The verdict vocabulary: ComplexityTier (LOW/MEDIUM/
                        HIGH), TranslationParity (DIRECT..MANUAL), and
                        TShirtSize (SMALL..EXTRA_LARGE, carrying Fibonacci
                        points and the needs_breakdown flag) — ordered scales
                        with max_tier() / worst_parity() / max_size().
  models.py             The scored units: ComplexitySignal (evidence vs
                        catalogue note), ChunkComplexity, BatchComplexity,
                        FileComplexity (the sized unit, reporting
                        effort/complexity/uncertainty separately),
                        CrossFileProfile, CorpusComplexityReport (computed
                        tier_counts / overall_tier / overall_difficulty /
                        overall_size / total_points; to_markdown()).
                        Every result records the target it was scored against.
                        Re-exports tiers.py and dependencies.py, so this stays
                        the one import site.
  dependencies.py       DependencyEdge + DependencyGraph: which file must be
                        migrated before which. The only models here describing
                        relationships BETWEEN files rather than scoring one.
  rules.py              RuleSet + the JSON profile loader: resolution
                        (explicit path > target > config > default), "extends"
                        inheritance with per-construct deep merge,
                        construct_groups shorthand, schema validation
                        (RuleSetError names the offending key), and caching.
                        Holds no ratings of its own.
  sizing.py             SizeModel and its calibration constants: how counted
                        work becomes a T-shirt size and a story-point number.
                        The other half of what a profile declares — the
                        catalogue says what a construct means, this says how
                        the counting scales.
  scoring.py            The measurement behind the analyzer's rules: counting
                        contained steps, deduping datasets, spanning lines,
                        turning a rule match into a ComplexitySignal, merging
                        signals, and writing a verdict's rationale.
  sharepoint.py         The complexity request list (Application /
                        Output_Language / Preferred_LLM) and where a run's
                        reports go. READ-ONLY: no status write-back, no
                        pending concept, timestamped folders for idempotence,
                        and a run-summary.md standing in for a Status column.
  crossfile.py          CrossFileIndex: resolves each chunk's macro, dataset,
                        macro-variable, and libref references against the rest
                        of the corpus into internal / import / export /
                        unresolved, from metadata the chunker already
                        extracts. HIGH/MANUAL "unresolved" is raised only when
                        two or more files were in scope — with one file,
                        absence proves nothing, so the reference is reported
                        as merely external. Reuses the chunker's autocall-macro
                        and default-libref sets rather than re-listing them.
  profiles/*.json       The catalogues themselves, one per target language:
                        construct -> (category, tier, parity, note), keyed by
                        PROC, component object, function, CALL routine, global
                        statement, chunk kind, metadata flag, and detector
                        name. An allowlist: an unlisted construct contributes
                        no signal at all. Tiers are target-independent (they
                        describe the SAS side); only parity moves between
                        profiles. sparksql.json is the default; pyspark.json
                        extends it and restates only what differs. Ratings are
                        grounded in reference_docs/ plus the published Spark
                        SQL function reference, and the load-bearing ones
                        quote their source in the entry's note (a SAS ARRAY is
                        "not a data structure"; MERGE with vs without BY is a
                        join vs a positional pairing; LAG returns "values from
                        a queue", not the previous row).
  detectors.py          Regex scans for what SasChunkMetadata does not extract
                        — ARRAY, DO loops, MERGE/UPDATE/MODIFY, RETAIN,
                        FIRST./LAST., FILENAME access methods (SFTP/FTP/EMAIL/
                        URL/PIPE/SOCKET), INFILE/FILE, LINK, DATA step GOTO.
                        Runs on chunker.scanner._sanitise output, so comments
                        and string literals never fire a signal; negative
                        lookbehinds keep macro %DO out of the DATA step forms.
  analyzer.py           ComplexityAnalyzer: aggregation and sizing, owns no
                        tier of its own. Tier = max signal tier
                        (presence-based), difficulty = worst signal parity,
                        score = sum of distinct construct weights (ranks
                        within a tier only). T-shirt size = effort + complexity
                        + uncertainty, scaled against the profile's anchor and
                        banded on the Fibonacci rungs, then floored by chunk
                        kind. Files, not batches, are the sized unit: a batch
                        may span several files while every chunk belongs to
                        exactly one. Reported points are the *rung* the final
                        size lands on — always a planning-poker entry (2/3/5/8),
                        floors included, so the number and the label can never
                        disagree; the continuous position the banding read is
                        kept alongside for ranking within a rung.
  report.py             Markdown rendering, and nothing else: the corpus report
                        (to_markdown() plus an index) and one report per source
                        SAS script, each printing the chunk text behind every
                        verdict it states. That text is passed in through
                        chunk_texts(), keyed (source_id, chunk_id) — a
                        ChunkComplexity carries no source of its own, and the
                        batcher re-ids chunks per file, so the lookup must be
                        built from the same items that were scored.
  llm_eval.py           The optional second opinion: the evaluation prompt
                        (static verdict + drivers + coupling + full source,
                        asking where the rules are wrong rather than for the
                        rules again), the FileEvaluation shape asked back, and
                        the invocation. Duck-typed on the client (anything with
                        invoke(); invoke_structured() used when offered), so
                        the package gains no LLM dependency. An unparseable
                        reply is kept as prose and costs only its own file.
  __main__.py           CLI: python -m complexity <sas_dir> [--target ...]
                        [--top N] [--out report.md] [--out-dir reports/]
                        [--llm-eval | --prompt-only]. Chunks, batches with
                        MultiFileBatcher, and scores the *batched* units — the
                        same work items the pipeline translates — then writes
                        the corpus report, and with --out-dir the per-file ones
                        too. Offline unless --llm-eval is passed, which is the
                        only path here that reaches the network.
```

Import direction is strictly downward: `keywords` and `models` import
nothing from the `chunker` package; `scanner` and `metadata` import from
them; `chunker.py` imports from all four; `batcher` imports from `keywords`,
`metadata`, `models`. The top-level `pipeline` package sits above the whole
stack and is the **only** package that imports `chunker` together with
`memory.store`, `memory.relevance`, `memory.summarize`, `llm_client`, and
`prompt_builder` (`pipeline.engine` orchestrates; `pipeline.prompting` holds
the SAS-metadata → `(query, constructs)` mapping). `memory`, `llm_client`,
and `prompt_builder` never import `chunker` or `pipeline` (or each other) —
`prompt_builder` reuses `memory.relevance` for retrieval, and the metadata
mapping that feeds it lives in `pipeline.prompting`, precisely so
`prompt_builder` needs no `chunker` import.
`app_config` is a leaf every package may import (like `chunker.keywords`, it
imports nothing from this repo outside itself): `chunker`, `llm_client`, and
`prompt_builder` read their word/token-limit defaults through it. Its
credential submodules — `vault`, `azure`, `databricks` — import only the
`app_config` loader and, in `databricks`'s case, `azure`; each defers its
third-party client to a lazy import inside the call that needs it, so the
package stays dependency-free to import. `validation` sits *above* the whole
stack, beside the CLI entry points: it drives `pipeline.engine` and may
import anything, and nothing imports it back. `complexity` sits above
`chunker` on the same footing — it reads `chunker.models` (plus
`chunker.scanner._sanitise`, `chunker.keywords._STANDARD_AUTOCALL_MACROS`, and
`chunker.batcher._DEFAULT_LIBREFS`, all deliberately, rather than
re-implementing SAS comment and quote rules or re-listing sets that would
drift) and `app_config`, and nothing imports it back. It is
never wired into the pipeline: scoring a corpus for complexity must not change
what the LLM is asked to translate. Its own optional LLM pass
(`complexity.llm_eval`) does not change that — the client is duck-typed on
`invoke()` and supplied by the caller, so the package still imports neither
`llm_client` nor `pipeline`, and `complexity/__main__.py` is the one place that
constructs a real client, lazily and only when `--llm-eval` is passed.

## Chunking model

The chunker is deliberately a **statement scanner + regex extractor**, not a
grammar-driven parser. It degrades gracefully on malformed source (emitting
`SasDiagnostic`s such as `UNCLOSED_MACRO`, `UNRECOGNIZED_SOURCE_REGION`,
`PARSER_TIMEOUT`) instead of failing, which a strict parser would not.
Replacing it with a full SAS grammar would be a rewrite, not a
simplification — this is a considered decision, not an accident.

- **Block collection rule:** only a new DATA/PROC/%MACRO header or an
  explicit RUN;/QUIT;/%MEND closes the current block. FORMAT, OPTIONS,
  LIBNAME, ODS, etc. inside a block body are collected, never treated as
  boundaries. A %MACRO block closes only on its own (nesting-balanced)
  %MEND.
- **Oversized splits:** a region exceeding `max_words` yields a *parent*
  chunk (full text) plus overlapping *child* chunks (`parent_id` set). The
  parent/child text redundancy is intentional context for the LLM. Child
  metadata is merged with the parent's via `_merge_meta` (see invariants).
- **Stuck-parser protection:** a wall-clock deadline gives a graceful
  partial-result exit at statement boundaries; a watchdog thread names the
  stuck phase in the logs when the parser is wedged inside a C-level regex
  call it cannot preempt.

### Metadata: stored vs computed

`SasChunkMetadata` stores one field per concept. Two views are
**computed fields** derived at access time, not stored:

- `referenced_automatic_vars` — the `&sys*` subset of
  `referenced_macro_vars` (all SAS automatic variables carry the reserved
  `SYS` prefix; see `models._is_automatic_macro_var`).
- `consumes_macrovars` — `referenced_macro_vars` minus automatics minus the
  macro's own `macro_param_names` (call-site-resolved, so never a
  corpus-level dependency).

Consequences: both appear in `model_dump()` but are silently ignored as
constructor kwargs, and they do not appear in `__str__` (which walks
`__dict__`). `defines_macros` / `invokes_macros` are the single
authoritative macro fields (`invokes_macros` includes CALL EXECUTE-invoked
macros).

## Batching model

`_EdgeDiscovery` builds producer indices, then walks the flattened corpus
once, emitting typed edges:

| Edge kind            | Tier   | Meaning |
|----------------------|--------|---------|
| `dataset_flow`       | strong | chunk reads a dataset a preceding chunk wrote |
| `macro_body_dataset` | strong | call-site-resolved parameterised macro-body I/O |
| `macro_invocation`   | weak   | chunk invokes a macro defined elsewhere |
| `macro_var_flow`     | weak   | chunk reads `&name`; links to the nearest preceding creator |
| `macro_arg_dataset`  | weak   | dataset name appears in a macro call's argument |

Strong edges union their endpoints in a Union-Find immediately. Weak edges
are resolved afterwards at *component* granularity: a producer feeding
exactly one component is absorbed into it; a producer feeding two or more
otherwise-independent components is promoted into a single **global-context
batch**, emitted first (`is_global_context=True`) — so one widely-used
`%let` or utility macro cannot fuse the whole corpus into one mega-batch.
OPTIONS/GLOBAL_STATEMENT (and optionally comment) chunks are then absorbed
into the following substantive chunk's component, same-file only.

Dataset names are canonicalised (`_canon_ds`): one-level names become
`work.<name>` (a `USER_LIBRARY_ASSIGNED` diagnostic flags the case where
that rewrite is inexact). Consumers link to the **nearest preceding
producer** in corpus order — the state a sequential SAS session would
actually read — so unrelated jobs reusing `work.tmp` stay separate.
Macro-variable consumers follow the same rule: a chunk reading `&name`
links only to the nearest preceding `%let`/`%global`/SYMPUT/SQL-INTO
creator (the last assignment before the reference is the one whose value
SAS resolves), so a name reassigned across unrelated jobs cannot fuse
them, and a reference before any assignment gets no edge.

## Pipeline and memory

`SasLLMPipeline` compiles a one-node LangGraph `StateGraph(MessagesState)`.
The model node loads the thread's history from `KVChatMessageHistory`, runs
`_trim | prompt | LLMClient` (trimming only limits what is *prompted*;
storage keeps every turn), and persists exactly the item message plus the
response in one bulk `add_messages` write. When a `prompt_builder` is set the
prompt additionally carries a block of reference guidance (see
`prompt_builder/`), injected via an ephemeral `instructions` placeholder that
is **prompted but never persisted**. `llm_client.LLMClient` owns
model construction (temperature, output-token cap, endpoint overrides,
gateway TLS trust via a configured `cert_file` exported as `SSL_CERT_FILE`,
an optional proactive rate limiter that paces request starts — on for the
`from_ai_gateway` credential path) and sync + async invocation (input-token
budget, transient-error retry that honors a gateway `Retry-After` /
`retry-after-ms` header when present, else capped exponential backoff); an
injected `llm` still gets the retry/budget layers.

### Instruction memory: task policy and thread notes

Beside the conversation itself, a run carries two instruction memories
(`memory/`), on the Task → Thread → Chat → Message model:

- **Long-term** (`task_id` / `task_policy`): a `TaskPolicy` of standing
  instructions for the task, each marked overridable or fixed. Loaded once,
  rendered into the system prompt inside the cache breakpoint, fingerprinted
  for the run record.
- **Short-term** (`thread_memory`): per-thread notes, exceptions and
  overrides, prompted through the ephemeral `instructions` channel with the
  precedence rule (a note beats an overridable instruction, never a fixed
  one) and expiring on a TTL. They travel with `fork_run`.
- **Writes** (`memory_extractor`): each accepted turn is classified into a
  temporary note (applied) or a permanent policy proposal (held for
  `approve()`), behind an offline cue gate so ordinary items cost no extra
  call.

A **chat** is one pipeline instance's span of a thread, recorded as a
`chat::` index row and readable via `get_chats(thread_id)`; a resumed or
forked run is a second chat on the same thread. `validation.memory_metrics`
scores all of this after the fact (policy adherence, override compliance,
extraction quality, note leakage across threads).

### Structured output and the notebook deliverable

With `structured_output` on (constructor argument, else config.json
`pipeline.structured_output`, default on) the model is asked for a
`TranslationDocument` — analysis, per-construct mapping, an **ordered list of
notebook cells**, and risks — via `LLMClient.invoke_structured`, which is
`invoke` with a schema-bound model, so the input budget, retries,
`cache_control` fallback, and usage accounting all still apply
(`include_raw=True`, so usage rides on the raw message).

What gets *persisted* is unchanged: the AI turn's content is
`TranslationDocument.to_markdown()` — the same four `##` sections the
unstructured prompt asks for, code in fenced blocks — with the document
carried alongside on `additional_kwargs["translation_document"]`. That is
deliberate. Storing the raw content would store an empty turn whenever the
gateway answers by tool call, breaking resume (`_recovered_response`),
relevance-based history selection, and every validation metric; rendering
instead means conversation memory, `validation`, and the resume path never
learn that structured output exists. The document is what `pipeline.notebook`
uses to build cells it knows are runnable, rather than guessing from fences.

Degradation is two-stage and never fails a run: a model whose integration has
no `with_structured_output` is detected at construction (the pipeline then
sends the Markdown system prompt), and a gateway that rejects the schema
mid-run is demoted once, for the rest of the pipeline's life, and re-sent
unstructured — the same one-shot demotion `llm_client` applies to a refused
`cache_control` breakpoint. A model that accepts the schema but answers badly
arrives as `parsing_error`, and its prose is used. In every fallback the
notebook is built by parsing the Markdown instead.

With `prompt_caching` enabled (constructor argument, else config.json
`llm_client.prompt_caching`) and an Anthropic model (`claude-*` or
`anthropic:`-prefixed), the system prompt travels as a content block
carrying a `cache_control` breakpoint, so every item after a run's first
reads it from the provider cache at a fraction of input cost. The
breakpoint sits on the system block only — prompted history varies per
item under trimming/selection, so the system prompt is the one stable
prefix. Non-Anthropic models keep the plain template (with a WARNING if
caching was requested); prompts under the model's minimum cacheable
prefix are silently not cached, which is harmless.

Prompted-history trimming has two modes: the default `window_k` recency
window, or — when a `memory.relevance.RelevantHistorySelector` is passed as
`history_selector` — relevance-based selection: each call keeps the
`top_k` turn pairs most relevant to the current batch/chunk message (BM25
lexical retrieval, optional dense retrieval over embeddings, RRF
fusion, optional reranker), always including the most recent
`always_keep_last` pairs and preserving chronological order. Scorers with
no signal (all scores tied) are excluded from fusion; with no signal at
all, selection degrades to recency. The selector optionally packs its picks
into a `max_tokens` budget (tail exempt, oversized pairs skipped). Either
way, trimming affects only the prompt — storage keeps every turn. All items
of one `run_file`/`run_text`/`run_files` call share one thread
(`thread_id = "run::<source ids>"`), so the LLM sees the run's accumulated
context batch by batch.

Two optional layers complement trimming. A `memory.summarize.RollingSummarizer`
(passed as `summarizer=`) folds turns older than its recency tail into one
running summary per thread, prepended as a SystemMessage *after*
trimming/selection — so it is never dropped by the window and never scored
by the selector; a store-less summarizer is auto-wired to the pipeline's
`memory.kv`. And the pipeline records one small **run fact** per processed
item into the KV layer (`run::<thread>::item::<item_id>`: status, index,
timing — never the response text, which already lives in `msg::`), readable
via `get_run_facts(thread_id)`.

Run facts power two control features. **Resume**: `run_file` / `run_text` /
`run_files` accept `resume=True` — items whose fact reads `ok` on the
thread are skipped (their stored responses recovered from the thread's
turn pairs, `skipped: True` in the output; error facts are reprocessed and
overwritten), so a crashed run continues instead of replaying completed
turns. **Fork**: `fork_run(src, dst, upto_items=k)` copies the first *k*
turn pairs plus their `ok` facts onto an empty thread
(`MemoryHub.fork_thread` underneath, preserving keys/timestamps);
rerunning with `thread_id=dst, resume=True` redoes everything after item
*k* on the branched history — KV-native time travel, no checkpointer.
Storage growth is bounded, when wanted, by
`MemoryHub(retention_max_age_s=..., retention_max_messages=...)`,
applied after each write.

`memory.store` stores everything as namespaced KV rows
(`msg::<thread>::<μs-timestamp>-<rand>` for messages). The
`KVStore` façade owns all JSON (de)serialisation, tag queries, search,
and snapshot/restore; a backend only stores/retrieves/deletes raw rows.
Message reads are incremental: after one full load per
`KVChatMessageHistory` instance, `.messages` fetches only rows past the
last seen key (`records_after` — keys are time-ordered), invalidating on
clear/prune/retention/restore, so an n-item run reads O(n) message rows
instead of O(n²).
`_InMemoryBackend` (default) is a plain dict and requires neither pyspark
nor a JVM; `_DeltaBackend` requires both and uses MERGE INTO / DELETE FROM
against a Delta table.

## Load-bearing invariants

Things that look like implementation details but are contracts. Breaking
any of these silently changes behavior.

1. **Edge discovery is one walk, in corpus order.**
   `_EdgeDiscovery._resolve_macro_body` mutates `produces_ds` mid-walk: a
   macro call site's resolved outputs are registered as producers at the
   moment the call is visited, which is what implements "a macro's output
   exists only once the call has executed" under nearest-preceding-producer
   bisection. Splitting the edge families into separate corpus walks would
   let a consumer link to a producer that does not exist yet at its
   position — or miss one that does.

2. **Producer lists stay sorted by global index.** The nearest-preceding
   lookups are `bisect_left` over `produces_ds[name]`; mid-walk
   registration therefore uses `insort`, never `append`.

3. **`output_datasets` is insertion-ordered, never sorted.**
   `_resolve_implicit_datasets` treats `output_datasets[-1]` as "the last
   dataset named" when resolving `_LAST_`/`_DATA_`/missing-`data=`
   references. Sorting it breaks that convention (list-merge in
   `_merge_meta` is the deliberate exception: split children lose ordering,
   and implicit resolution operates on unsplit metadata).

4. **Every `SasChunkMetadata` field must have a merge rule.** `_merge_meta`
   dispatches on field annotation (`list[str]` → sorted union, `bool` → OR,
   `str | None` → child-or-parent, `_MERGE_PARENT_WINS` → parent's value)
   and raises `TypeError` for anything else. The default-instance test in
   `tests/test_chunker.py` trips the guard for every stored field, so a new
   field shape cannot ship without a conscious decision. Signature-derived
   fields (`macro_param_names`, `body_param_*`) are parent-wins because only
   the split slice containing the `%MACRO` header can parse them.

5. **The LangGraph graph is compiled *without* a checkpointer, on
   purpose.** Durable per-thread persistence lives in the KV `msg::` row
   schema that `snapshot()`, `prune_before()`, and `list_threads()` depend
   on. A `BaseCheckpointSaver` would store full state blobs per turn
   (O(n²) growth in the Delta table) and duplicate the canonical store.
   Corollary: one graph invocation is one conversational turn — the node
   persists exactly the last state message plus the response. **Ephemeral
   context is prompted but never persisted to the `msg::` history** — two
   kinds: (a) reference guidance — when a `prompt_builder` is set, per-item
   instruction chunks are injected through an `instructions` placeholder
   carried in the run config; (b) the rolling summary — when a `summarizer`
   is set, its SystemMessage is prepended after trimming/selection and its
   state lives under the KV `summary::` key. Both are re-derivable, would
   bloat the O(n) history, and must stay out of
   `RelevantHistorySelector`'s scoring. A third kind joins them: (c)
   short-term **thread notes** — when a `thread_memory` is set, its live
   notes for the thread are appended to the same `instructions` list — so
   *stored = the item message; prompted = policy (in the system block) +
   summary + selected history + guidance + thread notes + item message*.

6. **Instruction memory splits on the cache breakpoint.** The long-term task
   policy (`memory.policy`) is folded into the system prompt at pipeline
   construction, *inside* the `cache_control` block: it is identical for
   every thread and item, so it costs one cache write and is then served
   from cache. Short-term thread notes must **not** go there — they differ
   per thread, and folding them into the cached prefix would miss the cache
   on every one — which is why they are ephemeral context (invariant 5c)
   rather than part of the prompt template. Corollary: the policy is a
   *snapshot*. `SasLLMPipeline.policy_fingerprint` is fixed at construction
   along with the prompt text it describes, and editing the `TaskPolicy`
   object afterwards changes neither — a new pipeline (a new chat) is what
   picks an edit up.

7. **The chat id is not part of the message key.** `chat::{thread}::{id}`
   is an index record, written in the *same batch* as the messages it covers
   (so it can never describe a write that did not land). Thread ids are
   recovered from `msg::{thread}::{tick}` keys by splitting off the last
   segment (`list_threads`), `fork_thread` copies a key-prefix range, and
   the incremental read frontier is a key comparison — a third key segment
   would break all three at once, plus every legacy row. One pipeline
   instance opens one chat on every thread it writes; a fork clamps each
   copied chat to the slice that was actually copied.

8. **In-memory mode must stay Spark-free.** `_InMemoryBackend` (and
   therefore `MemoryHub()` with no arguments) must import and run
   without pyspark installed; the pyspark requirement lives inside
   `_DeltaBackend.__init__` only. The pipeline never boots a SparkSession
   unless `delta_table` is set. Accordingly pyspark is declared as the
   optional `spark` extra in pyproject.toml, never a core dependency —
   CI installs the extra in the test and pyright jobs.

9. **`SasBatch.reason` strings and item ordering are pinned by tests.**
   Edge-emission order is observable output, not an implementation detail.

10. **`_RESERVED_WORDS` is Appendix 1 verbatim (94 words).** Genuine macro
   functions missing from Appendix 1 go in
   `_ADDITIONAL_MACRO_FUNCTION_WORDS`, and SAS-provided autocall macros in
   `_STANDARD_AUTOCALL_MACROS` — the three sets have distinct, citable
   identities and distinct consumers; do not fold them together.

11. **The target output language is resolved once and passed as an object.**
   `SasLLMPipeline.__init__` calls `resolve_target_language` and every later
   stage takes the resulting `TargetLanguage` from `.target_language` — the
   prompt, the `[lang: ...]` axis, the notebook kernel and fence tags, the
   validation suite. Re-deriving it from a string downstream is what let the
   layers disagree: the syntax metric checked Python on a Spark SQL run and
   scored a correct translation 0.0. Metric *names* are also part of this
   contract — `target_syntax` is the config, stored-verdict, and report key
   for syntax checks of the selected target.

12. **One owner per cross-cutting concern, and the owner is not the caller.**
   Three of these, each of which regressed once by growing a second
   implementation next to the first, and each of which fails *silently* when
   it does:

   - **One XREF owner.** `xref/` fetches mappings; `chunker` only
     *substitutes* what it is handed (`replace_dataset_names`,
     `parse_databricks_mapping_csv`). `chunker` must import
     `app_config.sharepoint` nowhere — `tests/test_xref.py` asserts this
     directly, because the failure mode is a network call appearing inside a
     package documented as network-free.
   - **One credential chain.** `LLMClientConfig.from_ai_gateway()` is the only
     way to build a gateway config. Assembling one by hand from
     `vault.ai_gateway_token()` looks equivalent and silently drops the
     `ai-gateway-version` header and the gateway's rate-limit pacing, both of
     which that classmethod adds. It is also walked **once per run** and
     copied per row (`model_copy`), never re-resolved.
   - **One entry point.** `main.py` parses and dispatches; `conversion/run.py`
     orchestrates. A second flow that reads the library and uploads to it will
     drift on the folder conventions — which is exactly how a previous one
     ended up reading from the drive root and writing to `{app}/output/`
     instead of `{base}/{app}/scripts_converted/{model}/{timestamp}`.

13. **`SharePointClient` owns a worker thread, not just a loop.** The blocking
    facade must stay callable from a caller that already has a running event
    loop, because a Jupyter or Databricks notebook keeps one in its main thread
    for the whole session — and SharePoint mode is deployed *in* a notebook.
    Blocking on the calling thread (`run_until_complete` there) made every
    SharePoint operation raise inside the deployment target. The invariant that
    actually matters is that the `httpx` connection pool stays bound to one
    loop driven by one thread; `max_workers=1` gets that unconditionally and
    stops caring what the caller's thread is doing. Corollary: one client
    serialises its calls, so it is not a way to parallelise Graph traffic.

14. **A Graph call built inside a coroutine must resolve the drive with
    `await`, never through `SharePointClient._run`.** `_run` drives its loop on
    the worker thread of invariant 13, so a helper that reaches `_drive_id()`
    from within a coroutine would block that worker on itself — `_run` detects
    the re-entry and raises rather than deadlock. It bites only when
    `SHAREPOINT_DRIVE_ID` is unset and the library has to be resolved from the
    site, which is the documented default. `_item_async` / `_drive_id_async`
    exist for exactly this; `_collect_children` and `_create_folder` use them.
    The failure is invisible to any test that pins a `drive_id`, so
    `tests/test_sharepoint.py` covers `list_directory`, `list_files` and
    `create_folder` against a site-resolved drive specifically.

## Conventions

- **Logging:** f-string messages everywhere (never lazy `%`-style).
  Per-iteration debug logs inside parse/batch loops are guarded with
  `if logger.isEnabledFor(logging.DEBUG):` so the f-string is never built
  when DEBUG is off; per-call entry/exit and LLM-paced logs are unguarded.
  Logger names follow modules: `chunker.chunker`, `chunker.scanner`,
  `chunker.metadata`, `chunker.batcher`, `pipeline.engine`,
  `pipeline.prompting`, `pipeline.notebook`, `memory.store`,
  `memory.relevance`, `memory.summarize`, `llm_client.client`,
  `conversion.run`, `xref.pre`, `xref.sourcing`, `target_language`, and
  `main` for the entry point. All four CLI entry points (`main`,
  `python -m complexity`, `python -m validation`,
  `python -m app_config.sharepoint_check`) configure logging through
  `app_config.logging_setup.configure_logging()` rather than calling
  `logging.basicConfig` themselves, which is what gives them `--log-file`,
  `--debug`, `--trace-http`, and secret redaction uniformly. That call also
  routes unhandled exceptions — main thread and worker threads — through the
  handlers, so `--log-file` captures the traceback rather than losing it to
  stderr; redaction covers the traceback text, not just the message.
- **Names:** dataset/macro/libref names are lowercased at extraction;
  quoted physical paths keep a leading `'` so they can never collide with
  identifiers.
- **Commits:** conventional-commit style (`refactor(scope): …`), one
  logical change per commit.

## Testing

`tests/` runs without a JVM, network, or API keys: the memory
tests use the in-memory backend, and the pipeline and validation tests inject
`FakeListChatModel`. The two KV backends share one behavioral contract suite
(`tests/test_backend_contract.py`): the in-memory half always runs, and the
Delta half runs the identical tests against a local delta-spark session,
skipping itself where pyspark + delta-spark + a JVM are unavailable — where
it cannot run, `_DeltaBackend` changes still need manual verification
against Databricks. Behavior-preserving
refactors of the chunker/batcher have historically been verified by
snapshotting full batcher output (batch membership, I/O fields, reason
strings, ordering) on a synthetic multi-file corpus and diffing against the
pre-change code; prefer that over trusting the suite alone for pure
code-motion changes.

The unit suite asserts *code behavior*; the *output quality* of a real model
run is the `validation` package's job — declarative cases scored by
deterministic metrics (coverage, dataset fidelity, Python syntax, required
terms, reference similarity) plus an opt-in LLM judge, with per-run history
appended via Spark (`python -m validation validation/cases --track`): local
parquet by default, a Delta table on Databricks. Like the Delta memory
backend, the Spark write path needs a JVM, so its test skips itself where no
local Spark session can start.
