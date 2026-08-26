# V2 consolidated gap register

Status: authoritative during Phase 9. The machine-readable source is
[`v2-gap-legacy-inventory.json`](v2-gap-legacy-inventory.json); CI checks its
package counts, v2 import allowlist, ownership, references, and exit gates.
PR descriptions and phase migration notes summarize this register but do not
define separate gap lists.

## Coverage and hardening

| ID | Owner | Gap | Exit gate |
|---|---:|---|---|
| G-001 | Phase 2 | V2 SAS core is below the final 95% line / 90% branch policy. | Package-level core thresholds pass. |
| G-002 | Phase 5 | Translation orchestration, budgeting, artifacts, models, and run-state retain uncovered guard/error branches. | Every Phase 5 application module reaches 90% line / 85% branch. |
| G-019 | Phase 10 | Repository-wide Ruff is not clean; CI gates the migration surface. | Ruff gates the complete repository at zero diagnostics. |
| G-020 | Phase 10 | Reference-document and live-service tests remain conditional outside provisioned jobs. | Required corpora are reproducible and production-critical integrations have no-skip jobs. |

## Knowledge and memory

| ID | Owner | Gap | Exit gate |
|---|---:|---|---|
| G-004 | Phase 9 | V2 Delta memory delegates physical MERGE/CDF work to `memory.store.KVStore`. | The v2 adapter owns Delta persistence and the legacy-import allowlist becomes empty. |
| G-005 | Phase 9 | Databricks chat and embedding factories remain in `memory.databricks_ai`. | Lazy v2 Databricks AI adapter jobs pass. |

## Closed Phase 9 replacements

| ID | Owner | Replacement | Evidence |
|---|---:|---|---|
| G-003 | Phase 9 | V2 ports and lazy adapters own BM25, FAISS dense retrieval, provider-scoped embedding caches, reciprocal-rank fusion, and reranking. | Lexical/dense/fusion/reranker, corruption, dimension, namespace isolation, lazy-import, and application-integration contracts pass at 95% combined line/branch coverage. |
| G-010 | Phase 9 | Local and SharePoint conversion requests, target/model selection, lifecycle state, source adapters, and per-row isolation are v2-owned. | Fake local and SharePoint end-to-end contracts pass at 98% combined line/branch coverage. |
| G-011 | Phase 9 | Pure hydration planning, versioned contracts, driver/sink ports, lazy adapters, ranged I/O, and managed Delta writes are v2-owned. | Parity and failure contracts pass at 95% combined coverage; optional imports are no-skip and the Delta sink runs in the real Spark/Delta container. |
| G-012 | Phase 9 | Lazy Graph SDK gateway, one-thread/one-loop transport, site/default-drive resolution, and versioned read-only deployment preflight are v2-owned. | File/list parity, running-loop safety, dependency ordering, redaction, and no-write preflight contracts pass at 96% combined line/branch coverage; the real SDK is constructed in the no-skip infrastructure job. |
| G-017 | Phase 9 | Hydration and infrastructure optional-adapter CI matrices. | Real SDK imports cannot skip; settings/auth/credential/observability contracts pass above 90% combined coverage. |

## Cutover, operations, and documentation

| ID | Owner | Gap | Exit gate |
|---|---:|---|---|
| G-013 | Phase 10 | `sas-migrate` is a shell; operations remain on `sas-parser`/`main.py`. | All v2 subcommands run from the wheel and `sas-parser` is removed. |
| G-014 | Phase 10 | Report presenters and SharePoint publication remain legacy-only. | V2 visual/golden and publication contracts pass. |
| G-015 | Phase 10 | A wheel-only non-root v2 image is gated, but Compose, warmup, and operator commands still compose legacy entry points. | Compose and operator commands cut over to the verified v2 image. |
| G-016 | Phase 10 | Top-level README and Architecture primarily describe the active legacy runtime. | V2 overview, config, API, operator, and cutover guides replace them. |
| G-018 | Phase 10 | Containerized PySpark/Delta and wheel-only v2 deployment smoke jobs are present; scheduled real-model quality evaluation remains absent. | Budgeted scheduled real-model evaluation passes. |
| G-021 | Phase 10 | Fourteen legacy packages, `main.py`, dual entry points, compatibility tests, and packaging references are shipped. | The legacy inventory and compatibility references are empty or archived outside the wheel. |

## Closed gates

| ID | Closed | Replacement | Evidence |
|---|---:|---|---|
| G-006 | Phase 7 | `sas_migrate.application.xref` and `sas_migrate.adapters.xref` | V2/legacy characterization, Databricks SQLGlot dialect, lazy adapter, architecture, type, wheel, and CI gates. |
| G-007 | Phase 8 | `sas_migrate.application.validation` and `sas_migrate.adapters.validation` | Deterministic/judged/memory metric, transcript, inline/offline, tracking, PDF, architecture, and parity gates. |
| G-008 | Phase 8 | Validation token-budget contracts and report presenters | Markdown/PDF/JSON/tracking expose input components and separate translation/judge compliance. |
| G-009 | Phase 8 | `sas_migrate.application.assessment` and `sas_migrate.adapters.assessment` | Profile inheritance, target scoring, sizing, cross-file dependency, review, golden report, PDF, and legacy parity gates. |

The deployment portion of G-018 is executable in the `V2 deployment smoke`
job. It builds `docker/v2.Dockerfile`, then runs the installed `sas-migrate
smoke` command as a non-root user with a read-only filesystem. G-018 remains
open only for its scheduled real-model quality gate.

The workspace deployment is also executable through the root Databricks
bundle. It deploys a two-task Lakeflow Job onto an explicitly supplied existing
general-purpose cluster, requires DBR 18 LTS before conversion starts, and runs
the current SharePoint entry point. The remaining operational cutover is still
tracked by G-013 through G-015: the job must switch to the v2 CLI and presenter
once those commands are implemented.

## Closure protocol

A gap closes only when its exit gate is executable and green. The implementing
commit must update the JSON status, this rendered register, the owning phase,
and any legacy package/import/reference entry removed by the change. Deleting a
reference without removing the underlying compatibility dependency does not
close a gap.
