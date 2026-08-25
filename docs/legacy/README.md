# Consolidated legacy code and reference index

This index defines what “legacy” means during the v2 migration. Legacy code is
still the active implementation for features not yet moved; it is not dead
code and must not be relocated or deleted before its owning phase passes the
replacement and cutover gates. The authoritative machine inventory is
[`../plans/v2-gap-legacy-inventory.json`](../plans/v2-gap-legacy-inventory.json).

## Code inventory

| Surface | Python files | Owner | V2 replacement/status |
|---|---:|---:|---|
| `app_config/` | 9 | Phase 9 | `sas_migrate.config` and infrastructure adapters; not migrated |
| `chunker/` | 9 | Phase 2/10 | `sas_migrate.core.sas` exists; compatibility package retained until cutover |
| `complexity/` | 18 | Phase 8/10 | v2 assessment replacement exists; compatibility package retained |
| `conversion/` | 6 | Phase 9/10 | v2 conversion application/adapters exist; compatibility package and publication path retained until cutover |
| `data_hydration/` | 20 | Phase 9 | hydration application/adapters; not migrated |
| `llm_client/` | 2 | Phase 9 | application port exists; gateway adapter pending |
| `memory/` | 11 | Phase 6/9 | services moved; physical Delta/CDF and Databricks AI transition remains |
| `pipeline/` | 9 | Phase 5/10 | translation orchestration exists; active CLI compatibility remains |
| `prompt_builder/` | 8 | Phase 6/9 | base knowledge moved; advanced retrieval adapters remain |
| `reporting/` | 2 | Phase 10 | presenters/document adapters pending |
| `target_language/` | 1 | Phase 3/10 | `sas_migrate.core.targets` exists; compatibility package retained |
| `token_budget/` | 1 | Phase 4/8/10 | core tokens exist; validation reporting and cutover remain |
| `validation/` | 17 | Phase 8/10 | v2 validation replacement exists; compatibility package retained |
| `xref/` | 6 | Phase 7/10 | v2 replacement exists; compatibility package retained until cutover |
| `main.py` | 1 module | Phase 10 | active `sas-parser` entry point; replaced by operational `sas-migrate` commands |

Total: 14 packages, 118 package Python files, 166 tracked package files
(including instruction/profile resources), and one top-level entry module.

## V2 dependencies on legacy code

Only one production import is permitted:

| V2 importer | Legacy dependency | Removal gate |
|---|---|---|
| `src/sas_migrate/adapters/memory/delta.py` | `memory.store.KVStore` | V2 owns physical Delta MERGE, CDF, checkpoint, and audit implementation |

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
