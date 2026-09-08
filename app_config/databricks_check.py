"""A read-only preflight for running this repo *on* a Databricks cluster.

The sibling of :mod:`app_config.sharepoint_check`, for the other half of a
deployment: not "can we reach SharePoint" but "is this process actually the
Databricks process it thinks it is, and is the environment around it the one
the code was resolved against". Three failures motivate it, and all three are
expensive precisely because none of them says what it is::

    runtime   are we on a cluster — and if so, in the notebook's own Python?
    session   is there a live SparkSession, and is its master what we assume?
    pyspark   does the installed Python client match the JVM it is talking to?
    packages  is every declared dependency installed, at or above its floor?
    extras    which optional extras are present — and is `spark` wrongly among them?
    libraries does this repo ship a compute library set for the running DBR?

Nothing here writes, starts a session, or costs an LLM call. Every stage runs
offline; ``--check-secrets`` adds the one network stage, a real read of the
Databricks secret scope.

A stage that cannot establish its answer reports ``SKIP`` and names what it
could not read -- never ``PASS``. ``extras`` did once return a pass having
opened nothing at all, which is the same defect as the ``runtime`` stage's old
discriminator and worse than a failure, because it is believed. The
``packages`` and ``extras`` stages resolve the requirement set through
:func:`requirement_sources`: the checkout's ``pyproject.toml`` first, then the
installed distribution's own metadata, then neither and a skip that says so.

The three failures
------------------
**A child process is not the notebook.** ``DATABRICKS_RUNTIME_VERSION`` is
inherited by every child of a cluster process, but the workspace *credential*
lives in the REPL, not the environment. So a ``%sh python main.py ...`` cell
detects "I am on Databricks", takes the notebook auth path, and fails the
secret read — whereupon the SDK walks its whole auth chain and reports either
something about ``az account show`` or, when its ``runtime`` strategy gets far
enough to reach for the REPL context, ``'NoneType' object has no attribute
'parent_header'``. Neither names anything real. The ``runtime`` stage detects
the shape of that process directly, before any credential is touched, and says
so in one line. See :func:`app_config.databricks.read_workspace_secrets`, whose
``_subprocess_hint`` is the same diagnosis delivered too late to be cheap.

It discriminates on the notebook REPL
(:func:`app_config.databricks.in_notebook_repl`), not on an active
SparkSession. The session was the original signal and it is a *lagging* one:
the SDK's ``runtime`` strategy imports ``dbruntime`` on its way to failing,
that import attaches Py4J to the driver's JVM, and the child therefore acquires
a session moments after being refused a credential. Anything that asked the old
question after the first failed secret read got a pass.

**The cluster already has a session.** Building one by master is wrong on
Databricks — silently ignored on classic Dedicated compute, and refused
outright under Spark Connect (serverless, and classic *Standard* access mode
since DBR 14.0). :func:`app_config.spark.active_or_new_session` is the fix; the
``session`` stage is how you confirm it took, and reports the master actually
in force rather than the one a log line claims.

**The `spark` extra must never be installed on a cluster.** ``pyproject.toml``
pins ``pyspark>=4.1.2,<4.2`` — correct for a laptop and for the Docker stack,
and wrong for any runtime whose own Spark is outside that range. Installed on a
cluster it puts a mismatched Python client under the runtime's JVM, and the
resulting ``NoSuchMethodError`` arrives much later, from somewhere else. The
``pyspark`` stage compares client against server directly, which needs no table
of runtime-to-Spark mappings to stay true.

Usage
-----
::

    python -m app_config.databricks_check                 # the full preflight
    python -m app_config.databricks_check --json          # machine-readable
    python -m app_config.databricks_check --check-secrets # + read the scope

From a notebook, import it rather than shelling out — shelling out is the very
thing the ``runtime`` stage exists to catch::

    from app_config.databricks_check import run_checks, render
    print(render(run_checks()))

Use *this* module's :func:`render`, which the line above imports. It exists
because a bare re-export of :func:`app_config.sharepoint_check.render` gave
that documented call SharePoint's title and all-clear line over a Databricks
report, while ``main()`` passed the right ones at its own call site and so
never showed it.

Logger name: ``app_config.databricks_check``.
"""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import logging
import os
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

from . import load_dotenv_file
from .logging_setup import configure_logging
from .sharepoint_check import FAIL, PASS, SKIP, WARN, CheckResult, to_json
from .sharepoint_check import render as _render

logger = logging.getLogger(__name__)

#: The report's title and its all-clear line. They live here, not at main()'s
#: call site, because the module docstring tells notebook users to import
#: `render` directly -- and while this module simply re-exported
#: sharepoint_check's, that documented form printed "SharePoint preflight" and
#: "PASSED: SharePoint is reachable and configured" over a Databricks report.
_TITLE = "Databricks runtime preflight"
_PASSED = "PASSED: the runtime environment matches what the code expects"

#: Extras that a Databricks deployment may legitimately have installed. Absence
#: is not a failure - a local-mode run needs none of them - so these are
#: reported, not required.
_OPTIONAL_EXTRAS = ("sharepoint", "vault", "azure", "databricks", "databricks-ai")

#: The extra that must NOT be installed on a cluster, and why in one clause.
_FORBIDDEN_EXTRA = "spark"

#: Distributions the `spark` extra brings, by the name pip files them under.
_FORBIDDEN_DISTRIBUTIONS = ("pyspark", "delta-spark")

#: This project, as pip files it. The fallback source for the requirement set
#: when there is no checkout above this module.
_DISTRIBUTION = "sas-parser"

#: `Requires-Dist: hvac>=2.0; extra == "vault"` -> "vault".
_EXTRA_MARKER = re.compile(r"""extra\s*==\s*['"]([^'"]+)['"]""")


# ---------------------------------------------------------------------------
# pyproject, as the single source of truth for what is required
# ---------------------------------------------------------------------------


def find_pyproject(start: Path | None = None) -> Path | None:
    """The repo's ``pyproject.toml``, found by walking up from this module.

    Walking up rather than trusting the working directory on purpose: under a
    Databricks job the CWD is not the repo, which is the same reason
    :mod:`app_config` resolves ``config.json`` explicitly. Returns ``None``
    when there is no checkout above us - an installed wheel, say - and the
    dependency stage then degrades to a skip rather than a false failure.
    """
    here = (start or Path(__file__)).resolve()
    for parent in here.parents:
        candidate = parent / "pyproject.toml"
        if candidate.is_file():
            return candidate
    return None


def _read_requirements(pyproject: Path) -> tuple[list[str], dict[str, list[str]]]:
    """``(core, {extra: requirements})`` out of *pyproject*'s project table."""
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    core = list(project.get("dependencies", []))
    extras = {
        name: list(reqs)
        for name, reqs in project.get("optional-dependencies", {}).items()
    }
    return core, extras


def _parse_requirement(spec: str) -> tuple[str, Any]:
    """``(distribution_name, specifier_or_None)`` for one requirement string.

    Uses ``packaging`` when it is importable - it is present on every
    Databricks runtime and in every resolved environment - and degrades to a
    name-only parse otherwise, so :mod:`app_config` keeps needing nothing.
    """
    try:
        from packaging.requirements import Requirement

        parsed = Requirement(spec)
        return parsed.name, parsed.specifier
    except ImportError:
        # "name[extra]>=1.2,<2" -> "name". Enough to check presence.
        name = spec.split(";")[0].strip()
        for sep in ("[", ">", "<", "=", "!", "~", " "):
            name = name.split(sep)[0]
        return name.strip(), None


def _installed_version(distribution: str) -> str | None:
    """The installed version of *distribution*, or ``None`` when absent."""
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _requirements_from_metadata() -> tuple[list[str], dict[str, list[str]]] | None:
    """This project's requirements out of its *installed distribution*.

    The second source for the same question ``pyproject.toml`` answers, and
    deliberately the one that exists when the file does not: a wheel carries
    ``Requires-Dist`` (with ``; extra == "vault"`` markers) and
    ``Provides-Extra``, which is the requirement set in another spelling. So a
    deployment installed rather than checked out still gets a dependency check
    instead of a stage that quietly passes.

    ``None`` when this project is not an installed distribution either --
    running from a source tree, or from package folders copied somewhere
    without their pyproject. The caller must skip, not pass.
    """
    try:
        meta = importlib_metadata.metadata(_DISTRIBUTION)
    except importlib_metadata.PackageNotFoundError:
        return None

    core: list[str] = []
    extras: dict[str, list[str]] = {
        name: [] for name in (meta.get_all("Provides-Extra") or [])
    }
    for spec in meta.get_all("Requires-Dist") or []:
        requirement, _, marker = str(spec).partition(";")
        extra = _EXTRA_MARKER.search(marker)
        if extra is None:
            core.append(requirement.strip())
        else:
            extras.setdefault(extra.group(1), []).append(requirement.strip())
    return core, extras


def requirement_sources(
    pyproject: Path | None = None,
) -> tuple[list[str], dict[str, list[str]], str] | None:
    """``(core, extras, where_it_came_from)``, or ``None`` when nothing declares them.

    One resolution order, shared by :func:`check_packages` and
    :func:`check_extras`, so the two cannot disagree about whether they are
    able to check anything -- which is how one of them ended up reporting a
    pass while having read nothing at all.
    """
    path = pyproject or find_pyproject()
    if path is not None:
        core, extras = _read_requirements(path)
        return core, extras, str(path)

    resolved = _requirements_from_metadata()
    if resolved is not None:
        return resolved[0], resolved[1], f"the installed {_DISTRIBUTION} distribution"
    return None


def _nothing_declares_them(stage: str) -> CheckResult:
    """The shared skip for "there is nothing here to check against".

    Names both sources and where it looked, because on a cluster this is
    normally a *deployment* fact -- package folders copied without their
    pyproject -- and the report is the only place that is visible.
    """
    return CheckResult(
        stage,
        SKIP,
        f"nothing declares this project's dependencies; {stage} cannot be checked",
        {
            "pyproject.toml": "not found above this module",
            "installed distribution": f"{_DISTRIBUTION} is not installed",
            "app_config": str(Path(__file__).resolve().parent),
        },
        fix=(
            "Copy pyproject.toml alongside the package folders, or install the "
            "built wheel, so there is something to check against. See "
            "databricks/README.md."
        ),
    )


def _satisfies(version: str, specifier: Any) -> bool:
    """Whether *version* meets *specifier* (``True`` when there is no specifier)."""
    if specifier is None:
        return True
    try:
        return specifier.contains(version, prereleases=True)
    except Exception:  # pragma: no cover - a specifier packaging itself parsed
        return True


# ---------------------------------------------------------------------------
# The stages
# ---------------------------------------------------------------------------


def check_runtime() -> CheckResult:
    """Are we on a Databricks cluster, and in the notebook's own Python?

    The child-process case is the one worth catching: it looks exactly like a
    cluster to every environment check, and exactly like an Azure CLI problem
    to the credential chain.

    The discriminator is the **notebook REPL**, not an active SparkSession.
    A session used to stand in for one, and it does not: a child process
    acquires a session as a *side effect* of the very failure this stage
    predicts — the SDK's ``runtime`` auth strategy imports ``dbruntime`` on its
    way to failing, and that import attaches Py4J to the driver's JVM. So the
    session appears moments after the credential is refused, and any run of
    this stage after the first failed secret read would have reported a pass.
    :func:`app_config.databricks.in_notebook_repl` has no such lag: its
    ``False`` is the same missing ``get_ipython()`` the SDK reports as
    ``'NoneType' object has no attribute 'parent_header'``.
    """
    from .databricks import (
        PROCESS_CHILD,
        PROCESS_OFF_CLUSTER,
        notebook_evidence,
        process_shape,
    )

    version = os.environ.get("DATABRICKS_RUNTIME_VERSION")
    detail: dict[str, Any] = {
        "DATABRICKS_RUNTIME_VERSION": version or "unset",
        "python": sys.version.split()[0],
        "executable": sys.executable,
        # The pid the runtime's own "Connection to spark using Py4J from PID
        # N" line names, so a report and a stray log line can be tied together.
        "pid": os.getpid(),
        # Which copy of this code is running. `executable` names the
        # interpreter, not the package, and on a cluster those routinely come
        # from different places -- a notebook-scoped env plus a Workspace
        # folder. It is also the single fact that explains what the packages,
        # extras and libraries stages below were and were not able to read.
        "app_config": str(Path(__file__).resolve().parent),
    }

    shape = process_shape()
    if shape == PROCESS_OFF_CLUSTER:
        return CheckResult(
            "runtime",
            SKIP,
            "not running on a Databricks cluster",
            detail,
        )

    detail.update(notebook_evidence())
    # Reported, never decisive - see this function's docstring. Kept below the
    # in_databricks_runtime() branch above so an off-cluster run still never
    # imports pyspark (Architecture.md invariant 8).
    detail["active SparkSession"] = "yes" if _active_session() is not None else "no"
    detail["workspace credential"] = (
        "DATABRICKS_TOKEN is set"
        if os.environ.get("DATABRICKS_TOKEN")
        else "the cluster runtime's own"
    )

    if shape == PROCESS_CHILD:
        if os.environ.get("DATABRICKS_TOKEN"):
            # The one legitimate child: it carries a credential of its own.
            # _subprocess_hint makes the same exemption, and this stays a WARN
            # so main's child-process warning (which logs only on FAIL) is
            # silent for a setup that works.
            return CheckResult(
                "runtime",
                WARN,
                "a Databricks child process, authenticating with DATABRICKS_TOKEN",
                detail,
                fix=(
                    "That is a supported setup - the secret-scope read will "
                    "use the PAT rather than the notebook's credential. "
                    "Nothing to change unless you expected the run to act as "
                    "the notebook's own identity."
                ),
            )
        return CheckResult(
            "runtime",
            FAIL,
            "looks like a Databricks child process, not the notebook itself",
            detail,
            fix=(
                "Only DATABRICKS_RUNTIME_VERSION is inherited by child "
                "processes - the notebook's workspace credential is not, so "
                "every secret-scope read from here will fail with a "
                "misleading Azure CLI error. Run this in the notebook's own "
                "Python (import the module and call it, or %run) instead of "
                "'%sh python ...', '!python ...' or subprocess. If a child "
                "process is genuinely intended, set DATABRICKS_TOKEN for it. "
                "The notebook form of a run is: import main; "
                "main.run_in_notebook('--request-id 80')."
            ),
        )
    return CheckResult(
        "runtime",
        PASS,
        f"on a Databricks cluster (DBR {version}), in the notebook's own Python",
        detail,
    )


def check_session() -> CheckResult:
    """Report the live SparkSession and the master actually in force."""
    from .databricks import PROCESS_CHILD, in_databricks_runtime, process_shape

    active = _active_session()
    if active is None:
        # These two used to be one branch, because "on a cluster with no
        # session" meant "a child process" back when the runtime stage
        # discriminated that way. It does not any more: the notebook's own
        # Python legitimately has no session until something touches Spark, so
        # pointing at the runtime stage there would send a reader to a check
        # that passes.
        if process_shape() == PROCESS_CHILD:
            return CheckResult(
                "session",
                SKIP,
                "no active SparkSession (see the runtime stage)",
            )
        if in_databricks_runtime():
            return CheckResult(
                "session",
                SKIP,
                "no active SparkSession yet - the runtime builds one on demand",
            )
        return CheckResult(
            "session",
            SKIP,
            "no active SparkSession - one is built on demand off Databricks",
        )

    detail: dict[str, Any] = {}
    try:
        detail["server Spark"] = active.version
    except Exception as exc:  # pragma: no cover - depends on the session type
        detail["server Spark"] = f"unavailable ({type(exc).__name__})"

    # sparkContext is absent under Spark Connect: serverless, and classic
    # Standard access mode since DBR 14.0. Its absence IS the finding.
    try:
        master = active.sparkContext.master
        detail["master"] = master
        detail["access mode"] = "dedicated / full session"
    except Exception:
        detail["master"] = "not exposed (Spark Connect)"
        detail["access mode"] = "standard or serverless (Spark Connect)"
        return CheckResult(
            "session",
            WARN,
            "the session is a Spark Connect client, not a full session",
            detail,
            fix=(
                "Delta-backed memory works, but anything reaching "
                "spark.sparkContext will not. If that is unintended, switch "
                "the cluster to Dedicated (single user) access mode."
            ),
        )

    if master.startswith("local") and in_databricks_runtime():
        return CheckResult(
            "session",
            FAIL,
            f"on a cluster but the active session is local ({master})",
            detail,
            fix=(
                "Something built a local session before the runtime's was "
                "reached. Restart the Python process and let "
                "app_config.spark.active_or_new_session pick up the "
                "runtime's session."
            ),
        )
    return CheckResult("session", PASS, f"active session on {master}", detail)


def check_pyspark() -> CheckResult:
    """Does the installed Python client match the Spark it is talking to?

    Compared directly rather than against a table of runtime-to-Spark
    mappings, which would need editing every release to stay true.
    """
    client = _installed_version("pyspark")
    if client is None:
        # No pip-installed pyspark. What that MEANS depends entirely on whether
        # Spark is running, and reporting one answer for both cases is how this
        # stage came to print "correct for an in-memory run" directly above a
        # live Spark 4.1.0 session on a cluster.
        #
        # On a cluster the runtime's own pyspark has no dist-info visible from
        # the notebook-scoped environment, so absence here is not absence of
        # pyspark -- it is the absence of a pip install shadowing the runtime's,
        # which is exactly the state this stage exists to confirm.
        running = _active_session()
        if running is None:
            return CheckResult(
                "pyspark",
                SKIP,
                "pyspark is not installed (correct for an in-memory run)",
            )
        try:
            server = running.version
        except Exception:  # pragma: no cover - depends on the session type
            server = "unknown"
        return CheckResult(
            "pyspark",
            PASS,
            f"no pip-installed pyspark is shadowing the runtime's Spark {server}",
            {
                "client (pyspark)": "no distribution metadata (the runtime's own)",
                "server Spark": server,
            },
        )

    detail: dict[str, Any] = {"client (pyspark)": client}
    active = _active_session()
    if active is None:
        return CheckResult(
            "pyspark",
            SKIP,
            f"pyspark {client} installed; no session to compare it against",
            detail,
        )

    try:
        server = active.version
    except Exception as exc:  # pragma: no cover - depends on the session type
        return CheckResult(
            "pyspark",
            SKIP,
            f"pyspark {client} installed; server version unavailable ({exc})",
            detail,
        )

    detail["server (JVM)"] = server
    # Compare the release, not the build suffix: Databricks reports its own
    # patch strings, and only a major.minor divergence is the failure meant.
    if _release(client) != _release(server):
        return CheckResult(
            "pyspark",
            FAIL,
            f"pyspark {client} does not match the running Spark {server}",
            detail,
            fix=(
                "The 'spark' extra was installed over the runtime's own "
                "pyspark. Remove pyspark and delta-spark from the compute "
                "libraries and restart the cluster - the runtime provides "
                "both. pyproject's pyspark pin is for laptops and the Docker "
                "stack; it is not meant to be satisfied on a cluster."
            ),
        )
    return CheckResult(
        "pyspark", PASS, f"pyspark {client} matches the running Spark", detail
    )


def check_packages(pyproject: Path | None = None) -> CheckResult:
    """Is every core dependency installed, at or above its declared floor?

    Reads ``pyproject.toml`` rather than a copied list, so this cannot drift
    from what the code actually requires.
    """
    sources = requirement_sources(pyproject)
    if sources is None:
        return _nothing_declares_them("packages")

    core, _, declared_by = sources
    missing: list[str] = []
    outdated: list[str] = []
    detail: dict[str, Any] = {"declared by": declared_by}

    for spec in core:
        name, specifier = _parse_requirement(spec)
        installed = _installed_version(name)
        if installed is None:
            missing.append(name)
        elif not _satisfies(installed, specifier):
            outdated.append(f"{name} {installed} (needs {specifier})")

    detail["declared"] = len(core)
    if missing:
        detail["missing"] = ", ".join(sorted(missing))
    if outdated:
        detail["below floor"] = ", ".join(sorted(outdated))

    if missing or outdated:
        return CheckResult(
            "packages",
            FAIL,
            f"{len(missing) + len(outdated)} of {len(core)} core dependencies "
            f"are missing or too old",
            detail,
            fix=(
                "On a cluster this is a failed library install - the Libraries "
                "tab shows FAILED while the notebook attaches anyway. Check "
                "the cluster event log, and that the compute library points at "
                "databricks/requirements-dbr<N>.txt."
            ),
        )
    return CheckResult(
        "packages",
        PASS,
        f"all {len(core)} core dependencies installed and satisfied",
        detail,
    )


def check_extras(pyproject: Path | None = None) -> CheckResult:
    """Which optional extras are present - and is ``spark`` wrongly among them?

    This used to guard its whole body on having found a ``pyproject.toml`` and
    then return PASS regardless, so a deployment with no checkout above it got
    ``[ ok ] optional extras resolved`` having read nothing and resolved
    nothing. A green check that did not check is the same defect this module's
    ``runtime`` stage was rewritten for, one size down: it is worse than a red
    one, because it is believed.
    """
    from .databricks import in_databricks_runtime

    sources = requirement_sources(pyproject)
    if sources is None:
        return _nothing_declares_them("extras")

    _, extras, declared_by = sources
    detail: dict[str, Any] = {"declared by": declared_by}
    for extra in _OPTIONAL_EXTRAS:
        specs = extras.get(extra)
        if not specs:
            continue
        names = [_parse_requirement(s)[0] for s in specs]
        present = [n for n in names if _installed_version(n) is not None]
        detail[extra] = (
            "installed"
            if len(present) == len(names)
            else f"partial ({', '.join(present)})"
            if present
            else "not installed"
        )

    on_cluster = in_databricks_runtime()
    installed = {d: _installed_version(d) for d in _FORBIDDEN_DISTRIBUTIONS}
    detail |= {d: v or "not installed" for d, v in installed.items()}

    # Only delta-spark is judged here. A wrong *pyspark* is the `pyspark`
    # stage's job, which compares it against the running Spark and so can tell
    # a shadowing install from the runtime's own; a version alone cannot.
    if on_cluster and installed["delta-spark"]:
        return CheckResult(
            "extras",
            FAIL,
            "delta-spark is installed on a cluster, where Delta is built in",
            detail,
            fix=(
                f"Remove the '{_FORBIDDEN_EXTRA}' extra (pyspark, delta-spark) "
                "from the compute libraries and restart the cluster. The "
                "runtime provides both, and the OSS delta-spark will fight it."
            ),
        )
    return CheckResult(
        "extras",
        PASS,
        "optional extras resolved" + (" (on a cluster)" if on_cluster else ""),
        detail,
    )


#: Line the library-set files carry so this stage reads the runtime they were
#: resolved against out of the file rather than out of a constant here, which
#: would drift the moment a set is added. Same reasoning as `check_packages`
#: reading pyproject.toml instead of a copied list.
_RUNTIME_TAG = re.compile(r"^#\s*databricks-runtime:\s*(\d+)", re.MULTILINE)


def check_libraries(pyproject: Path | None = None) -> CheckResult:
    """Does this repo ship a compute library set for the runtime we are on?

    A *different* question from ``packages``, and neither answers the other.
    The Libraries tab points at a path outside this process — a Volume or a
    Workspace file — so nothing here can know which file was actually
    installed. This stage answers "is there a set for this runtime at all",
    which is what catches a cluster upgraded to a DBR the repo has never been
    resolved against; ``packages`` answers "is what *is* installed sufficient",
    by reading pyproject against the installed distributions. Together they are
    the answer; alone, neither is.

    A WARN, never a FAIL: an unlisted runtime may work perfectly well. It is
    the silence that is expensive, not the mismatch.
    """
    from .databricks import in_databricks_runtime

    version = os.environ.get("DATABRICKS_RUNTIME_VERSION")
    if not in_databricks_runtime():
        return CheckResult(
            "libraries", SKIP, "not running on a Databricks cluster"
        )

    found = pyproject or find_pyproject()
    directory = (found.parent / "databricks") if found else None
    if directory is None or not directory.is_dir():
        # Unlike packages/extras there is no second source to fall back to:
        # databricks/ is excluded from the wheel by [tool.setuptools.packages
        # .find], so an installed deployment has no library sets to compare
        # against however it was installed. Only a checkout does.
        return CheckResult(
            "libraries",
            SKIP,
            "no databricks/ directory above this module; the shipped library "
            "sets cannot be read",
            {"app_config": str(Path(__file__).resolve().parent)},
            fix=(
                "Copy databricks/ (and pyproject.toml) alongside the package "
                "folders to compare the running runtime against the sets this "
                "repo ships. It is excluded from the wheel, so installing one "
                "does not bring it. See databricks/README.md."
            ),
        )

    shipped: dict[str, str] = {}
    for path in sorted(directory.glob("requirements-dbr*.txt")):
        match = _RUNTIME_TAG.search(path.read_text(encoding="utf-8"))
        if match:
            shipped[match.group(1)] = path.name

    running = version.split(".")[0] if version else ""
    detail: dict[str, Any] = {
        "running": version or "unset",
        "library sets": ", ".join(
            f"DBR {major} ({name})" for major, name in sorted(shipped.items())
        )
        or "none found",
        "directory": str(directory),
    }

    if not shipped:
        return CheckResult(
            "libraries",
            SKIP,
            "no tagged library sets found",
            detail,
            fix="Each requirements-dbr<N>.txt needs a '# databricks-runtime: <N>' line.",
        )
    if running not in shipped:
        return CheckResult(
            "libraries",
            WARN,
            f"running DBR {version}, but this repo ships sets for "
            f"DBR {', '.join(sorted(shipped))} only",
            detail,
            fix=(
                "The installed libraries were resolved against a different "
                "runtime, so a version this one already ships may have been "
                "silently upgraded cluster-wide. See databricks/README.md for "
                "how to resolve a set against this runtime."
            ),
        )
    return CheckResult(
        "libraries",
        PASS,
        f"a library set for DBR {running} is checked in ({shipped[running]})",
        detail,
    )


def check_secret_scope() -> CheckResult:
    """Actually read the configured secret scope - the one network stage."""
    from .databricks import (
        DEFAULT_SECRET_KEYS,
        DatabricksError,
        get_databricks_config,
        read_workspace_secrets,
    )

    config = get_databricks_config()
    detail: dict[str, Any] = {
        "auth_method": config.auth_method or "none",
        "secret_scope": config.secret_scope or "unset",
    }
    if not config.secret_scope:
        return CheckResult(
            "secrets",
            SKIP,
            "no secret scope configured",
            detail,
            fix="Set DATABRICKS_SECRET_SCOPE or databricks.secret_scope.",
        )
    try:
        values = read_workspace_secrets(
            config.secret_scope, DEFAULT_SECRET_KEYS.keys, config=config
        )
    except DatabricksError as exc:
        return CheckResult(
            "secrets",
            FAIL,
            f"could not read secret scope '{config.secret_scope}'",
            detail | {"error": str(exc)},
            fix=(
                "If this process is a %sh/subprocess child, that is the cause "
                "- see the runtime stage. Otherwise check that the cluster's "
                "principal has READ on the scope."
            ),
        )
    detail["keys read"] = ", ".join(sorted(values))
    return CheckResult(
        "secrets", PASS, f"read {len(values)} keys from the secret scope", detail
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _active_session() -> Any:
    """The active SparkSession, or ``None`` - without importing pyspark eagerly."""
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        return None
    try:
        return SparkSession.getActiveSession()
    except Exception:  # pragma: no cover - a broken/partial pyspark install
        return None


def _release(version: str) -> tuple[str, str]:
    """``(major, minor)`` of *version*, ignoring patch and any build suffix."""
    parts = version.split(".")
    return (parts[0], parts[1] if len(parts) > 1 else "0")


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run_checks(*, check_secrets: bool = False) -> list[CheckResult]:
    """Every stage, in dependency order.

    Offline by default: nothing here contacts the workspace unless
    *check_secrets* asks for the one stage that does.
    """
    results = [
        check_runtime(),
        check_session(),
        check_pyspark(),
        check_packages(),
        check_extras(),
        check_libraries(),
    ]
    if check_secrets:
        results.append(check_secret_scope())
    logger.info(
        f"run_checks: {sum(1 for r in results if r.status == FAIL)} failure(s) "
        f"across {len(results)} stage(s)"
    )
    return results


def render(results: list[CheckResult], *, verbose: bool = False) -> str:
    """The results as the human-readable report, under this preflight's title.

    A wrapper rather than a re-export of :func:`sharepoint_check.render`, whose
    defaults name SharePoint. The module docstring documents importing this
    from a notebook, and for as long as the re-export stood, that documented
    call printed a SharePoint heading and "PASSED: SharePoint is reachable and
    configured" over a Databricks report. ``main()`` supplied the right strings
    at its own call site, which is exactly why nobody running the CLI ever saw
    it.
    """
    return _render(results, verbose=verbose, title=_TITLE, passed=_PASSED)


def main(argv: list[str] | None = None) -> int:
    """Run the preflight from the command line. Non-zero if any stage failed."""
    parser = argparse.ArgumentParser(
        prog="python -m app_config.databricks_check",
        description=(
            "Read-only preflight for running on a Databricks cluster: the "
            "process identity, the Spark session, the pyspark/JVM pairing, and "
            "the installed dependencies. Writes nothing and starts no session."
        ),
    )
    parser.add_argument(
        "--check-secrets",
        action="store_true",
        help="Also read the configured Databricks secret scope (the one "
        "network stage). Needs the cluster's own credential or a PAT.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit the results as JSON instead of the readable report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show what every check resolved, not just the failing ones.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Also write the log to this file, appending to it. Secrets are "
        "redacted, but treat the file as sensitive.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging. The HTTP transport libraries stay at INFO.",
    )
    args = parser.parse_args(argv)

    configure_logging(debug=args.debug, log_file=args.log_file)
    load_dotenv_file()

    results = run_checks(check_secrets=args.check_secrets)
    if args.as_json:
        print(to_json(results))
    else:
        print(
            render(results, verbose=args.verbose)
        )
    return 1 if any(r.status == FAIL for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
