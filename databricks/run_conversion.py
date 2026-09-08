# Databricks notebook source
# ruff: noqa: E402 -- a Databricks source notebook is a sequence of cells,
# and the first one puts the checkout on sys.path. Every import after it
# depends on that having run, so none of them can be hoisted to the top.
# MAGIC %md
# MAGIC # Run a SAS conversion on this cluster
# MAGIC
# MAGIC Import this file into the workspace (Workspace > Import > File) and it
# MAGIC arrives as a notebook. It is checked in as Python source rather than
# MAGIC `.ipynb` so it diffs like code and carries no output.
# MAGIC
# MAGIC **Do not run the conversion with `!python main.py ...` or
# MAGIC `%sh python main.py ...`.** Those start a *child process*, and on a
# MAGIC cluster a child inherits `DATABRICKS_RUNTIME_VERSION` but not the
# MAGIC notebook's workspace credential. So it detects Databricks, takes the
# MAGIC notebook auth path, and fails the Databricks secret-scope read from
# MAGIC inside the SDK with a message about the Azure CLI or
# MAGIC `'NoneType' object has no attribute 'parent_header'` -- neither of
# MAGIC which is about the real problem. Every cell below runs in the
# MAGIC notebook's own Python, where the credential is simply present.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Put the checkout on `sys.path`
# MAGIC Skip this if the notebook already lives inside the repo folder.

# COMMAND ----------

import sys

REPO = "/Workspace/Users/<you>/SAS_Parser"  # <- your checkout
if REPO not in sys.path:
    sys.path.insert(0, REPO)

print(sys.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Is this really the notebook's own Python?
# MAGIC Expect `[ ok ] runtime ... in the notebook's own Python`, and
# MAGIC `notebook REPL: yes (IPython shell)` in the detail. A `[FAIL]` here is
# MAGIC the child-process problem, and everything after it will fail too.

# COMMAND ----------

from app_config.databricks_check import render, run_checks

print(render(run_checks(), verbose=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. The credential-chain dry run
# MAGIC Offline first -- this contacts nothing and reports where every setting
# MAGIC came from. Run it before the live form: if the offline report is wrong,
# MAGIC the live one has nothing to add.

# COMMAND ----------

from app_config.auth_check import render as render_auth
from app_config.auth_check import run_checks as run_auth_checks

print(render_auth(run_auth_checks(), verbose=True))

# COMMAND ----------

# MAGIC %md
# MAGIC Now the live form: it reads the secret scope, mints a Graph token and
# MAGIC reports its granted roles, logs in to Vault, and reads the AI gateway
# MAGIC credential. It still writes nothing and calls no model.

# COMMAND ----------

print(render_auth(run_auth_checks(live=True), verbose=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. The run
# MAGIC The direct replacement for the `!python main.py ...` cell: the string
# MAGIC is the same arguments, `shlex`-split, so an existing command pastes
# MAGIC across unchanged.

# COMMAND ----------

import main

status = main.run_in_notebook(
    "--reference-dir '/Workspace/Users/<you>/reference_docs' --request-id 80"
)
print(f"exit status: {status}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Re-running after changing an environment variable
# MAGIC A notebook process outlives the environment it first resolved, and the
# MAGIC credential chain is cached per process on purpose (it is walked once
# MAGIC per run, not once per row). Pass `reset_caches=True` to drop it:
# MAGIC
# MAGIC ```python
# MAGIC main.run_in_notebook("--request-id 80", reset_caches=True)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Regenerating the compute library set (maintainers only)
# MAGIC Run this on a **stock** cluster of the runtime you are targeting,
# MAGIC before attaching any library, and paste the output into
# MAGIC `databricks/constraints-dbr<N>.txt`. See `databricks/README.md` for
# MAGIC what to keep and what to drop.

# COMMAND ----------

from importlib.metadata import distributions

print(f"# runtime: {__import__('os').environ.get('DATABRICKS_RUNTIME_VERSION')}")
for name, version in sorted(
    {dist.metadata["Name"].lower(): dist.version for dist in distributions()}.items()
):
    print(f"{name}=={version}")
