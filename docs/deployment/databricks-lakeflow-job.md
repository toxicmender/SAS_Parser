# Databricks Lakeflow Job

The root [`databricks.yml`](../../databricks.yml) deploys the current
SharePoint-driven application as a Databricks Lakeflow Job. Both tasks attach
to one existing general-purpose (all-purpose) cluster:

1. `runtime_preflight` requires Databricks Runtime 18 LTS and verifies the
   active Spark/Python and installed-library environment;
2. `convert_pending_requests` runs only after that preflight succeeds and calls
   the installed `sas-parser` Lakeflow wheel entry point.

The bundle does not create or resize the cluster. General-purpose compute is a
workspace resource with cloud-specific node types, policies, permissions, and
cost controls, so its ID is an explicit deployment input. The absence of a
`new_cluster`, `job_cluster_key`, or serverless `environment_key` is a tested
contract.

## Prerequisites

- Databricks CLI 0.258.0 or newer;
- `uv` available on the deployment host;
- a Databricks workspace profile or equivalent unified authentication;
- an existing general-purpose cluster configured with **Databricks Runtime 18
  LTS** and, unless a workload requires otherwise, standard access mode;
- the existing SharePoint, Entra ID, Vault, and model-gateway configuration
  described in [the root README](../../README.md#configuration).

Databricks Runtime 18 supplies Spark and Delta. Do not install the project's
`spark` extra on the cluster. The job installs the built application wheel and
[`databricks/requirements.txt`](../../databricks/requirements.txt), whose
constraints are derived for DBR 18 LTS.

## Validate and deploy

Supply the cluster ID as a bundle variable. Keep workspace identity and
credentials in the Databricks profile rather than committing them:

```bash
databricks bundle validate -t dev \
  --var general_purpose_cluster_id=0123-456789-example \
  --profile MY_WORKSPACE

databricks bundle deploy -t dev \
  --var general_purpose_cluster_id=0123-456789-example \
  --profile MY_WORKSPACE

databricks bundle run -t dev sas_parser \
  --var general_purpose_cluster_id=0123-456789-example \
  --profile MY_WORKSPACE
```

Use `-t prod` for the production target. No schedule is enabled by default;
operators can run the deployed job manually or add a workspace-approved
schedule after deployment.

The runtime requirement defaults to `18`. It can be passed explicitly for
auditability, but weakening it to another family changes the deployment
contract:

```bash
databricks bundle validate -t prod \
  --var general_purpose_cluster_id=0123-456789-example \
  --var databricks_runtime_family=18 \
  --profile MY_WORKSPACE
```

## Current cutover boundary

This job runs the full current SharePoint conversion workflow through the
failure-propagating `main:lakeflow_main` wrapper. The v2 conversion application
service is already implemented, but
the `sas-migrate convert sharepoint` composition root and publication presenter
remain Phase 10 gaps (G-013/G-014). At that cutover, the second wheel task can
switch entry points without changing the compute or DBR 18 preflight contract.

Static bundle, entry-point, dependency, and runtime-mismatch contracts run in
CI without workspace credentials. A live `databricks bundle validate`, deploy,
and job run require the target workspace and cluster and therefore remain a
deployment-environment gate.
