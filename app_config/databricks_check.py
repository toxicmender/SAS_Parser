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

Nothing here writes, starts a session, or costs an LLM call. Every stage runs
offline; ``--check-secrets`` adds the one network stage, a real read of the
Databricks secret scope.

The three failures
------------------
**A child process is not the notebook.** ``DATABRICKS_RUNTIME_VERSION`` is
inherited by every child of a cluster process, but the workspace *credential*
lives in the REPL, not the environment. So a ``%sh python main.py ...`` cell
detects "I am on Databricks", takes the notebook auth path, and fails the
secret read — whereupon the SDK walks its whole auth chain and reports
something about ``az account show`` that names nothing real. The ``runtime``
stage detects the shape of that process directly, before any credential is
touched, and says so in one line. See
:func:`app_config.databricks.read_workspace_secrets`, whose ``_subprocess_hint``
is the same diagnosis delivered too late to be cheap.

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

Logger name: ``app_config.databricks_check``.
"""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import logging
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

from . import load_dotenv_file
from .logging_setup import configure_logging
from .sharepoint_check import FAIL, PASS, SKIP, WARN, CheckResult, render, to_json

logger = logging.getLogger(__name__)

#: Extras that a Databricks deployment may legitimately have installed. Absence
#: is not a failure - a local-mode run needs none of them - so these are
#: reported, not required.
_OPTIONAL_EXTRAS = ("sharepoint", "vault", "azure", "databricks", "databricks-ai")

#: The extra that must NOT be installed on a cluster, and why in one clause.
_FORBIDDEN_EXTRA = "spark"

#: Distributions the `spark` extra brings, by the name pip files them under.
_FORBIDDEN_DISTRIBUTIONS = ("pyspark", "delta-spark")


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
    to the credential chain. An active SparkSession is the discriminator - the
    notebook REPL has one, a ``%sh``/``subprocess`` child does not.
    """
    from .databricks import in_databricks_runtime

    version = os.environ.get("DATABRICKS_RUNTIME_VERSION")
    detail: dict[str, Any] = {
        "DATABRICKS_RUNTIME_VERSION": version or "unset",
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }

    if not in_databricks_runtime():
        return CheckResult(
            "runtime",
            SKIP,
            "not running on a Databricks cluster",
            detail,
        )

    active = _active_session()
    detail["active SparkSession"] = "yes" if active is not None else "no"
    if active is None:
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
                "process is genuinely intended, set DATABRICKS_TOKEN for it."
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
    from .databricks import in_databricks_runtime

    active = _active_session()
    if active is None:
        if in_databricks_runtime():
            return CheckResult(
                "session",
                SKIP,
                "no active SparkSession (see the runtime stage)",
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
        return CheckResult(
            "pyspark",
            SKIP,
            "pyspark is not installed (correct for an in-memory run)",
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
    path = pyproject or find_pyproject()
    if path is None:
        return CheckResult(
            "packages",
            SKIP,
            "no pyproject.toml above this module; nothing to check against",
        )

    core, _ = _read_requirements(path)
    missing: list[str] = []
    outdated: list[str] = []
    detail: dict[str, Any] = {"pyproject": str(path)}

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
                "databricks/requirements.txt."
            ),
        )
    return CheckResult(
        "packages",
        PASS,
        f"all {len(core)} core dependencies installed and satisfied",
        detail,
    )


def check_extras(pyproject: Path | None = None) -> CheckResult:
    """Which optional extras are present - and is ``spark`` wrongly among them?"""
    from .databricks import in_databricks_runtime

    detail: dict[str, Any] = {}
    path = pyproject or find_pyproject()
    if path is not None:
        _, extras = _read_requirements(path)
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
    ]
    if check_secrets:
        results.append(check_secret_scope())
    logger.info(
        f"run_checks: {sum(1 for r in results if r.status == FAIL)} failure(s) "
        f"across {len(results)} stage(s)"
    )
    return results


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
            render(
                results,
                verbose=args.verbose,
                title="Databricks runtime preflight",
                passed="PASSED: the runtime environment matches what the code expects",
            )
        )
    return 1 if any(r.status == FAIL for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
