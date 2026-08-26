# Consolidated legacy code and reference index

This index defines what “legacy” means during the v2 migration. Legacy code is
still the active implementation for features not yet moved; it is not dead
code and must not be relocated or deleted before its owning phase passes the
replacement and cutover gates. The authoritative machine inventory is
[`../plans/v2-gap-legacy-inventory.json`](../plans/v2-gap-legacy-inventory.json).

## Code inventory

| Surface | Python files | Owner | V2 replacement/status |
|---|---:|---:|---|
| `app_config/` | 9 | Phase 9/10 | v2 settings, credentials, auth, observability, Graph worker, and preflight exist; active legacy composition remains |
| `chunker/` | 9 | Phase 2/10 | `sas_migrate.core.sas` exists; compatibility package retained until cutover |
| `complexity/` | 18 | Phase 8/10 | v2 assessment replacement exists; compatibility package retained |
| `conversion/` | 6 | Phase 9/10 | v2 conversion application/adapters exist; compatibility package and publication path retained until cutover |
| `data_hydration/` | 20 | Phase 9/10 | v2 application/adapters exist; compatibility CLI/config composition retained |
| `llm_client/` | 2 | Phase 9 | application port exists; gateway adapter pending |
| `memory/` | 11 | Phase 6/9/10 | services and physical Delta/CDF moved; Databricks AI and active legacy composition remain |
| `pipeline/` | 9 | Phase 5/10 | translation orchestration exists; active CLI compatibility remains |
| `prompt_builder/` | 8 | Phase 6/10 | v2 ingestion and advanced retrieval replacements exist; compatibility package retained until cutover |
| `reporting/` | 2 | Phase 10 | presenters/document adapters pending |
| `target_language/` | 1 | Phase 3/10 | `sas_migrate.core.targets` exists; compatibility package retained |
| `token_budget/` | 1 | Phase 4/8/10 | core tokens exist; validation reporting and cutover remain |
| `validation/` | 17 | Phase 8/10 | v2 validation replacement exists; compatibility package retained |
| `xref/` | 6 | Phase 7/10 | v2 replacement exists; compatibility package retained until cutover |
| `main.py` | 1 module | Phase 10 | active `sas-parser` entry point; replaced by operational `sas-migrate` commands |

Total: 14 packages, 119 package Python files, 167 tracked package files
(including instruction/profile resources), and one top-level entry module.

## V2 dependencies on legacy code

No production import from `src/sas_migrate` into a top-level legacy package is
permitted. The previous `memory.store.KVStore` allowance was removed when v2
took ownership of Delta schema upgrades, MERGE/delete persistence, CDF,
checkpoint, audit, retry, and diagnostics behavior.

CI rejects any new v2-to-legacy import or any unrecorded change to this
allowlist. Test-only SAS aliases in `scripts/v2_sas_test_adapter.py` are parity
infrastructure and do not create a production dependency.

## Reference inventory

Legacy references are intentionally grouped by purpose:

- Active runtime documentation: `README.md`, `Architecture.md`, and the 14
  package README files.
- Active legacy configuration: `config.json` and `.env.example`.
- Packaging and static analysis: `pyproject.toml` package discovery, package
  data, coverage sources, and Pyright inputs.
- CI and smoke compatibility: `.github/workflows/ci.yml`,
  `scripts/smoke_wheel.py`, `scripts/v2_sas_test_adapter.py`, and
  `tests/test_v2_sas_parity.py`.
- Deployment: `docker-compose.yml`, `docker/README.md`, and Spark/Delta install
  and warmup scripts.
- Behavior characterization: legacy-focused files under `tests/`.
- Historical wire-contract decisions: `docs/migrations/` and the superseded
  `docs/plans/pipeline-decoupling-tiktoken.md` plan.

Historical migration notes remain valid evidence and are not rewritten to look
like current APIs. Runtime documents carry a banner linking back to the v2 plan
and consolidated registers.

The Phase 9 conversion replacement now owns request and source boundaries,
target/model selection, lifecycle transitions, dry-run behavior, and queue
isolation. The top-level `conversion/` package remains shipped only because the
legacy CLI still composes it and because SharePoint deliverable publication is
part of the Phase 10 presenter/cutover gate.

The Phase 9 hydration replacement now owns pure planning, versioned work and
report contracts, partition probes, lazy source-driver boundaries, ranged I/O,
failure-isolated execution, and managed Delta writes. The top-level
`data_hydration/` package remains shipped for the legacy CLI and its environment
and credential composition; those dependencies move with G-012 and Phase 10.

The Phase 9 infrastructure foundation now owns strict non-secret settings,
credential and token ports, lazy environment/Databricks/Vault/MSAL adapters,
redacted logging/HTTP tracing, and the concrete Graph single-loop worker and
read-only preflight. `app_config` remains active only for legacy composition
roots and is removed with the Phase 10 cutover.

The Phase 9 knowledge replacement now owns deterministic lexical retrieval,
lazy BM25/FAISS hybrid ranking, reciprocal-rank fusion, reranking, and
provider-scoped in-memory and atomic disk embedding caches. `prompt_builder`
and its shared `memory.relevance` helper remain shipped only because the legacy
pipeline still composes them; their removal is part of G-021, not an open
knowledge-functionality gap.

The Phase 9 memory persistence replacement now owns the physical Delta table
contract and has no import back to `memory.store`. The top-level `memory/`
package remains shipped because the active legacy pipeline still composes its
history services and because Databricks AI factories remain G-005; physical
Delta persistence is no longer a transition dependency.

## Removal order

1. Move behavior and data contracts into the owning v2 feature.
2. Pass characterization, adapter, installed-wheel, and required no-skip jobs.
3. Switch composition and operational entry points to v2.
4. Remove the package from wheel discovery, coverage, Pyright, smoke, and CI.
5. Remove or archive its legacy tests and README references.
6. Update the machine inventory and close the corresponding gap.

Physical consolidation under a new `legacy/` Python namespace is deliberately
not an intermediate step: it would churn active imports without retiring a
dependency. The consolidation unit is this checked inventory; code moves once,
directly into its final v2 owner.
