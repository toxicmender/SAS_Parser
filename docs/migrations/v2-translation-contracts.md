# V2 translation orchestration — Phase 5 migration note

Phase 5 adds the application-level translation contract and deliberately does
not preserve the legacy `SasLLMPipeline` Python API or its mutable output
dictionaries. New integrations compose `TranslateCorpus` with explicit ports
for the model, accepted-response memory, token records, run events, artifacts,
and time.

The new versioned contracts are:

- `TranslationItem`, which preserves ordered SAS chunks and source attribution;
- `TranslateCorpusRequest`, including the resolved run target and token policy;
- `TranslationItemOutcome` and `TranslationRunOutcome`;
- `PromptContext` and attributed prompt components;
- `ArtifactLocator` for persisted prompts, responses, Markdown, and notebooks.

Run control is append-only. Resume replays run events and recovers accepted
responses from memory without making another model call. Rewind forgets the
affected accepted responses and reopens the run. Fork copies only the accepted
prefix and marks copied token records as recovered usage.

Every provider attempt uses the same Phase 3 structured/raw response envelope.
Only target-valid documents enter accepted memory, canonical Markdown, or
notebooks. Effective-prompt artifacts retain the exact attributed request for
each new attempt; token-audit artifacts remain redacted.

Notebook output is one file per SAS source where code-cell `chunk_id`
attribution is complete. An indivisible multi-source result is written once to
`_cross_file.ipynb`, with pointer cells in its source notebooks. Mixed PySpark
and Spark SQL notebooks use a Python kernel and prefix SQL cells with `%sql`.

The v2 Spark SQL target continues to validate SQL with SQLGlot's `databricks`
dialect. There is no Spark Scala translation target.
