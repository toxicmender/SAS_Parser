# V2 token contracts — Phase 4 migration note

Phase 4 adds the following `schema_version = 2` wire contracts:

- `PromptAssembly` and attributed `PromptComponent` input estimates;
- `PromptBudgetDecision` preflight results;
- `CallTokenRecord` attempt-level estimates and provider usage;
- `TokenCallLedger` retry, resume, and run-total aggregation;
- `TokenAuditArtifact` redacted component manifests.

These are pre-release v2 contracts, so their addition does not migrate or
rewrite legacy `token_budget`, pipeline-memory, Delta, or validation records.
V2 writers use fresh targets as required by the consolidated architecture
plan.

Estimated component counts remain diagnostic estimates. Provider input,
output, cache-read, and cache-write counts remain separate and are the source
for billed totals whenever present. Missing provider usage is represented as
missing rather than as zero.

Token audit artifacts do not store prompt text or raw source identifiers.
They retain category, role, flags, counts, and SHA-256 fingerprints; effective
prompt artifacts with source text belong to Phase 5 and use the artifact
repository's access controls.
