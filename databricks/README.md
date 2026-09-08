# Compute library sets

One pair of files per Databricks Runtime this repo has been resolved against:

| File | What it is |
|---|---|
| `requirements-dbr<N>.txt` | The set to install on a DBR *N* cluster. Point the Libraries tab at it. |
| `constraints-dbr<N>.txt` | What DBR *N* already ships, of the packages the requirements file therefore omits. |
| `run_conversion.py` | The notebook that runs a conversion (and, in its last cell, regenerates a constraints file). |

Both files in a pair carry a machine-readable tag on line 2:

```
# databricks-runtime: 19
```

`app_config.databricks_check`'s `libraries` stage reads that tag rather than a
constant in the code, so adding a set is adding two files and nothing else. On
a cluster whose major version has no set, it WARNs — the installed libraries
were resolved against a different runtime, and a package that runtime already
ships may have been silently upgraded cluster-wide.

## Deploying the code to a cluster

However the code gets onto the cluster, **what you copy decides what the
preflight can check.** `app_config.databricks_check` walks up from its own
location looking for a checkout, and falls back to the installed
distribution's metadata; a deployment that provides neither gets three honest
skips rather than three green checks — see the `runtime` stage's `app_config:`
line for where it is actually running from.

| You copy / install | `packages`, `extras` | `libraries` | `main.run_in_notebook` |
|---|---|---|---|
| The repo (Git folder, or the whole tree) | ✅ pyproject | ✅ | ✅ |
| Package folders **+ `pyproject.toml`** | ✅ pyproject | only with `databricks/` too | needs `main.py` |
| `%pip install` the built wheel | ✅ distribution metadata | ❌ excluded from the wheel | ✅ |
| Package folders alone | ❌ skip | ❌ skip | needs `main.py` |

Three things are easy to leave behind when copying folders:

- **`pyproject.toml`** — one file, and `packages` and `extras` start working.
- **`databricks/`** — the library sets, which `[tool.setuptools.packages.find]`
  deliberately excludes from the wheel, so installing one never brings them.
- **`main.py`** — a top-level *module*, not a package
  (`[tool.setuptools] py-modules = ["main"]`), so a copy that took only the
  package directories will not have it and `main.run_in_notebook(...)` fails on
  `import main`.

## Pointing a cluster at one

```json
{ "requirements": "/Volumes/<catalog>/<schema>/sas_parser/requirements-dbr19.txt" }
```

`requirements-dbr<N>.txt`'s own header lists what is deliberately **absent**
from it and why — `pyspark` and `delta-spark` above all, because the runtime
*is* Spark and installing the `spark` extra puts a mismatched Python client
under the runtime's JVM.

## Adding a set for a new runtime

The versions a runtime ships cannot be derived from this repository; they are
a property of the image. So this is not a desk exercise — step 1 needs a
cluster.

**1. Capture what the runtime ships.** On a **stock** cluster of that runtime,
before attaching any library, run the last cell of `run_conversion.py` (or
paste it into a notebook):

```python
from importlib.metadata import distributions

for name, version in sorted(
    {dist.metadata["Name"].lower(): dist.version for dist in distributions()}.items()
):
    print(f"{name}=={version}")
```

**2. Write `constraints-dbr<N>.txt`.** Take the export the project resolves to:

```bash
uv export --frozen --no-dev --no-hashes --no-emit-project --no-annotate \
    --no-header --extra sharepoint --extra vault --extra azure
```

Keep the lines from step 1 whose package name appears in that export. Those are
the packages the runtime already provides — that set, at the runtime's
versions, *is* the constraints file. Add the header (copy `constraints-dbr19.txt`'s
and change the numbers) and the `# databricks-runtime: <N>` tag.

**3. Write `requirements-dbr<N>.txt`.** Take the same export and drop every
package the constraints file now lists at a version the lock is happy with.
What remains is the requirements file. Add the header, the tag, and the
"deliberately absent" list.

**4. Verify it resolves against a stock runtime, without one to hand:**

```bash
uv pip compile databricks/requirements-dbr<N>.txt \
    -c databricks/constraints-dbr<N>.txt \
    --python-version <the runtime's python> --python-platform linux
```

A failure here means something in the set now demands a package newer than the
runtime ships, and the install would silently upgrade it cluster-wide. That is
the signal to look, not to widen the constraint.

**Never hand-write a constraints file from documentation or memory.** A guessed
version passes every check in this repo and is wrong only on the cluster, where
it upgrades a runtime package for every job on it.

## Known runtimes

- **DBR 19** — Spark 4.2.0, Python 3.12.3. Set checked in.
- **DBR 18 LTS** — Spark 4.1.0, JDK 21. **No set checked in yet**; a cluster is
  needed for step 1. Until one lands, `databricks_check`'s `libraries` stage
  WARNs on that runtime, which is the intended behaviour rather than a gap in
  the check. Note that the `spark` extra's `pyspark>=4.1.2,<4.2` is
  minor-compatible with 18 LTS where it is not with 19 — that does not make
  installing the extra correct there (4.1.2+ over 4.1.0 is still a mismatched
  client), but it does change what the failure looks like.
