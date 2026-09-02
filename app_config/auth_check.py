"""A dry run of the credential chain — every hop, offline by default.

The third preflight, and it asks a question neither sibling does.
:mod:`app_config.sharepoint_check` asks "can we reach SharePoint";
:mod:`app_config.databricks_check` asks "is this process the Databricks process
it thinks it is". This one asks **"which identity does each hop use, where did
each value come from, and which hop breaks?"** — which crosses both, because a
single Databricks secret scope feeds *two* service principals, and those feed
Microsoft Graph on one side and Vault → the AI gateway on the other.

The chain is not a line, and the report says so::

    process ──▶ workspace ──▶ bootstrap ─┬─▶ principal (sp-hsv-*) ──▶ vault ──▶ gateway
                                         └─▶ sharepoint (saact-hsv-*) ──▶ graph

The incident this came from
---------------------------
A run started from a Databricks notebook cell as ``!python main.py …`` failed
with::

    AzureAuthError: could not read the Entra ID service principal from
    Databricks: could not build a Databricks client to read secret scope
    '…': ValueError: default auth: runtime: 'NoneType' object has no
    attribute 'parent_header'

Eight hops of credential resolution, and the message names none of them. The
cause was the first: ``!python`` is a child process, it inherits
``DATABRICKS_RUNTIME_VERSION`` but not the notebook's workspace credential, and
so every later hop was doomed before it was reached. The whole point of a dry
run is that the *offline* form of this report says so — ``process`` FAIL,
``workspace`` WARN, ``bootstrap`` WARN, everything downstream skipped with a
reason — in a second, having contacted nothing.

Offline by default
------------------
The inverse of ``sharepoint_check --offline``, deliberately. This command is
for a deployment whose credentials are in doubt, so the safe default is to
touch nothing; ``--live`` is the opt-in that actually reads the scope, mints
the tokens, and logs in to Vault. Even ``--live`` writes nothing and pays no
LLM: it reads the *credential*, never a model.

Usage
-----
::

    python -m app_config.auth_check              # the dry run: no network at all
    python -m app_config.auth_check --live       # read the scope, mint the tokens
    python -m app_config.auth_check --json       # machine-readable
    sas-parser --check-auth                      # the offline form, via the CLI

From a notebook, import it rather than shelling out — a child process is the
first thing the ``process`` stage reports, and shelling out is how you become
one::

    from app_config.auth_check import render, run_checks
    print(render(run_checks(), verbose=True))

Secrets
-------
No stage puts a credential in its result. Identifiers (tenant, client, scope
and key *names*, hosts, audiences, granted roles), presence (``"set"`` /
``"unset"``), and lengths are reported; values are not. ``render`` and
``to_json`` additionally pass everything through
:func:`app_config.logging_setup.redact`, but that is a net under the rule, not
the rule.

Logger name: ``app_config.auth_check``.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import logging
import os
import sys
from typing import Any

from . import load_dotenv_file
from .logging_setup import configure_logging
from .sharepoint_check import (
    FAIL,
    PASS,
    SKIP,
    WARN,
    CheckResult,
    setting_source,
    to_json,
)
from .sharepoint_check import render as _render

logger = logging.getLogger(__name__)

__all__ = [
    "FAIL",
    "PASS",
    "SKIP",
    "WARN",
    "CheckResult",
    "check_bootstrap",
    "check_gateway",
    "check_graph",
    "check_principal",
    "check_process",
    "check_sharepoint",
    "check_vault",
    "check_workspace",
    "main",
    "render",
    "run_checks",
    "to_json",
]

#: The report's title and its all-clear line, for :func:`render`.
_TITLE = "Credential chain dry run"
_PASSED = "PASSED: every hop of the credential chain resolved"


def _renamed(result: CheckResult, name: str) -> CheckResult:
    """*result* under a different stage name.

    The reused stages arrive carrying their own module's vocabulary
    (``"runtime"``, ``"identity"``, ``"secrets"``, ``"token"``). Renaming them
    is what lets this report name the *hop* while the logic stays in the one
    place that owns it.
    """
    return dataclasses.replace(result, name=name)


def _skipped(name: str, why: str) -> CheckResult:
    """A stage that could not run because an earlier one failed."""
    return CheckResult(name, SKIP, why)


def _installed(distribution: str) -> bool:
    """True when *distribution*'s import package is present, without importing it."""
    try:
        return importlib.util.find_spec(distribution) is not None
    except (ImportError, ValueError):  # pragma: no cover - a broken install
        return False


def _present(env_var: str) -> str:
    """``"set"`` or ``"unset"`` — for a value that must never be printed."""
    return "set" if os.environ.get(env_var) else "unset"


def _sourced(env_var: str, section: str, key: str) -> str:
    """One setting as ``"<value> (from <source>)"``, or ``"unset"``."""
    value, source = setting_source(env_var, section, key)
    return "unset" if value is None else f"{value} (from {source})"


# ---------------------------------------------------------------------------
# The stages
# ---------------------------------------------------------------------------


def check_process() -> CheckResult:
    """Hop 0: is this the notebook's own Python, or a child of it?

    Reused wholesale from :func:`app_config.databricks_check.check_runtime`,
    because it is the same question and there must be one answer to it. It
    leads the chain because on a cluster it is the most common cause of every
    later failure and it costs nothing to ask.
    """
    from .databricks_check import check_runtime

    return _renamed(check_runtime(), "process")


def check_workspace() -> CheckResult:
    """Hop 1: which credential authenticates to the Databricks workspace?

    Offline in full — :meth:`~app_config.databricks.DatabricksConfig.from_env`
    performs no I/O — and the provenance is the payload. A host that is right
    in ``config.json`` and stale in the environment looks exactly like a
    correct one until you know which of them won.
    """
    from .databricks import get_databricks_config, in_databricks_runtime
    from .databricks_check import check_runtime

    config = get_databricks_config()
    method = config.auth_method
    detail: dict[str, Any] = {
        "auth_method": method or "(nothing usable)",
        "host": _sourced("DATABRICKS_HOST", "databricks", "host"),
        "secret scope": _sourced(
            "DATABRICKS_SECRET_SCOPE", "databricks", "secret_scope"
        ),
        "catalog": _sourced("DATABRICKS_CATALOG", "databricks", "catalog"),
        "ARM_TENANT_ID": _sourced("ARM_TENANT_ID", "databricks", "azure_tenant_id"),
        "ARM_CLIENT_ID": _sourced("ARM_CLIENT_ID", "databricks", "azure_client_id"),
        # Secret-shaped: presence only, never the value.
        "ARM_CLIENT_SECRET": _present("ARM_CLIENT_SECRET"),
        "DATABRICKS_TOKEN": _present("DATABRICKS_TOKEN"),
        "azure_resource_id": _sourced(
            "DATABRICKS_AZURE_RESOURCE_ID",
            "databricks",
            "azure_workspace_resource_id",
        ),
    }

    if method is None:
        return CheckResult(
            "workspace",
            FAIL,
            "no usable Databricks credential",
            detail,
            fix=(
                "Run on a cluster, or set DATABRICKS_TOKEN, or configure the "
                "ARM_TENANT_ID / ARM_CLIENT_ID / ARM_CLIENT_SECRET service "
                "principal."
            ),
        )
    # The reported bug, named at the hop where it originates: `notebook` means
    # "the runtime authenticates itself", and a child process is on the cluster
    # without being the thing that holds the credential.
    if method == "notebook" and check_runtime().status == FAIL:
        return CheckResult(
            "workspace",
            WARN,
            "auth_method is 'notebook', but this process is not the notebook",
            detail,
            fix=(
                "The runtime credential belongs to the REPL, so nothing here "
                "can use it. Run from a cell (import main; "
                "main.run_in_notebook(...)), or set DATABRICKS_TOKEN for this "
                "process. See the process stage."
            ),
        )
    if not in_databricks_runtime() and not config.host:
        return CheckResult(
            "workspace",
            FAIL,
            "no Databricks host configured, and not running on a cluster",
            detail,
            fix="Set DATABRICKS_HOST or databricks.host in config.json.",
        )
    return CheckResult(
        "workspace", PASS, f"the workspace credential is '{method}'", detail
    )


def check_bootstrap(*, live: bool = False) -> CheckResult:
    """Hop 2: can the secret scope be read *at all*?

    The headline of the dry run. Reading the scope needs a credential that
    cannot have come out of it — the cluster runtime's own, or a PAT — and this
    stage answers that with no network and no SDK import, by mirroring
    :func:`app_config.databricks._bootstrap_client`'s preconditions rather than
    calling it.

    ``--live`` hands over to
    :func:`app_config.databricks_check.check_secret_scope`, which performs the
    real read of the workspace key set.
    """
    from .databricks import (
        get_databricks_config,
        in_databricks_runtime,
        in_notebook_repl,
    )

    config = get_databricks_config()
    on_cluster = in_databricks_runtime()
    credential = (
        "DATABRICKS_TOKEN"
        if config.token
        else "the cluster runtime's own"
        if on_cluster
        else "none"
    )
    detail: dict[str, Any] = {
        "secret scope": config.secret_scope or "unset",
        "bootstrap credential": credential,
        "host": config.host or "(the runtime's own)",
        "databricks-sdk": "installed" if _installed("databricks.sdk") else "missing",
    }

    if not config.secret_scope:
        return CheckResult(
            "bootstrap",
            SKIP,
            "no secret scope configured",
            detail,
            fix=(
                "Set DATABRICKS_SECRET_SCOPE or databricks.secret_scope. "
                "Without one the ARM_* principal is the only path."
            ),
        )
    if not _installed("databricks.sdk"):
        return CheckResult(
            "bootstrap",
            FAIL,
            "databricks-sdk is not installed, so the scope cannot be read",
            detail,
            fix="pip install \"sas-parser[databricks]\" (a cluster already has it).",
        )
    if credential == "none":
        return CheckResult(
            "bootstrap",
            FAIL,
            "no credential that could authenticate the secret-scope read",
            detail,
            fix=(
                "Reading the scope needs a credential that does not come from "
                "it: run on a Databricks cluster, or set DATABRICKS_TOKEN."
            ),
        )
    if on_cluster and not config.token and not in_notebook_repl():
        # The offline prediction of the reported failure, made before any
        # client is built: the only credential on offer is the runtime's, and
        # this process is not the one holding it.
        return CheckResult(
            "bootstrap",
            WARN,
            "the only credential is the runtime's, and this is not the notebook",
            detail,
            fix=(
                "Every read of this scope will fail from here, inside the SDK, "
                "with a message about the Azure CLI or 'parent_header'. Run "
                "from a cell (import main; main.run_in_notebook(...)) or set "
                "DATABRICKS_TOKEN. See the process stage."
            ),
        )
    if not live:
        return CheckResult(
            "bootstrap",
            PASS,
            f"the scope read would authenticate with {credential}",
            detail,
        )

    from .databricks_check import check_secret_scope

    return _renamed(check_secret_scope(), "bootstrap")


def check_principal(*, live: bool = False) -> CheckResult:
    """Hop 3: the workspace/Vault service principal (``sp-hsv-*``).

    Offline this reports *which source will win* rather than claiming success.
    A complete local ``ARM_*`` triple pre-empts the scope read for this key set
    only — see
    :meth:`app_config.databricks.DatabricksConfig.service_principal` — and
    which of the two is in force is exactly what a later failure will not say.
    """
    from .databricks import DEFAULT_SECRET_KEYS, DatabricksError, get_databricks_config

    config = get_databricks_config()
    local = config.has_service_principal
    detail: dict[str, Any] = {
        "principal": "the workspace / Vault service principal",
        "source": (
            "the local ARM_* triple (it pre-empts the scope for this key set)"
            if local
            else f"secret scope '{config.secret_scope}'"
            if config.secret_scope
            else "(nothing configured)"
        ),
        "keys": ", ".join(DEFAULT_SECRET_KEYS.keys),
    }

    if not local and not config.secret_scope:
        return CheckResult(
            "principal",
            SKIP,
            "no service principal configured, locally or in a scope",
            detail,
            fix=(
                "Set ARM_TENANT_ID / ARM_CLIENT_ID / ARM_CLIENT_SECRET, or "
                "point DATABRICKS_SECRET_SCOPE at the scope holding "
                f"{'/'.join(DEFAULT_SECRET_KEYS.keys)}."
            ),
        )
    if not live:
        return CheckResult(
            "principal", PASS, f"resolves from {detail['source']}", detail
        )

    try:
        principal = config.service_principal()
    except DatabricksError as exc:
        return CheckResult(
            "principal",
            FAIL,
            "could not resolve the workspace service principal",
            detail | {"error": str(exc)},
            fix=(
                "If this is a child process, that is the cause - see the "
                "process stage. Otherwise check the cluster principal has "
                "READ on the scope and that the keys exist under these names."
            ),
        )
    detail["tenant id"] = principal.tenant_id
    detail["client id"] = principal.client_id
    detail["client secret"] = "read (not shown)"
    return CheckResult(
        "principal", PASS, f"resolved principal {principal.client_id}", detail
    )


def check_sharepoint(*, live: bool = False) -> CheckResult:
    """Hop 3': SharePoint's *own* service principal (``saact-hsv-*``).

    A second principal in the same scope under different keys, so it fails
    independently of hop 3 — and when one resolves and the other does not, that
    difference is the diagnosis. The scope itself falls through
    ``SHAREPOINT_SECRET_SCOPE`` → ``sharepoint.secret_scope`` →
    ``DATABRICKS_SECRET_SCOPE`` → ``databricks.secret_scope``, so the report
    names which of the four supplied it.
    """
    from .sharepoint import SharePointConfig
    from .sharepoint_check import check_identity, check_secret_scope

    config = SharePointConfig.from_env()
    scope_source = _sourced("SHAREPOINT_SECRET_SCOPE", "sharepoint", "secret_scope")
    if scope_source == "unset" and config.secret_scope:
        scope_source = (
            f"{config.secret_scope} (fell through to the databricks section)"
        )

    stage = check_secret_scope(config) if live else check_identity(config)
    result = _renamed(stage, "sharepoint")
    return dataclasses.replace(
        result, detail=result.detail | {"secret scope source": scope_source}
    )


def check_graph(*, live: bool = False) -> CheckResult:
    """Hop 4: the Microsoft Graph token, and what its ``roles`` claim grants.

    Offline this is a skip with the audience it *would* request, which is worth
    reporting on its own: a token mints perfectly well with no application
    permissions at all and then every call returns 403, and the claim is where
    the difference is visible.
    """
    from .sharepoint import SharePointConfig
    from .sharepoint_check import check_token

    config = SharePointConfig.from_env()
    detail: dict[str, Any] = {"scopes": ", ".join(config.scopes)}

    if not live:
        return CheckResult(
            "graph",
            SKIP,
            "not minted (add --live)",
            detail | {"msal": "installed" if _installed("msal") else "missing"},
        )
    if not _installed("msal"):
        return CheckResult(
            "graph",
            SKIP,
            "msal is not installed, so no token can be minted",
            detail,
            fix='pip install "sas-parser[azure]" (or [sharepoint], which includes it).',
        )
    # SharePointClient.get_token goes through app_config.azure, not the Graph
    # SDK, so msgraph-sdk is genuinely not needed on this path.
    from .sharepoint import SharePointClient

    return _renamed(check_token(SharePointClient(config)), "graph")


def check_vault() -> CheckResult:
    """Hop 5: the Vault login — offline resolution only.

    The highest-value offline finding here is the **JWT audience**: a Vault
    role bound to a different one fails the login with a bare 400, and the
    audience resolves with no I/O as ``vault.azure_scopes`` > the azure
    section's scopes > the ARM default. That chain is exact rather than a
    guess, because :meth:`app_config.azure.AzureAuthConfig.for_principal` is
    :func:`dataclasses.replace` over ``from_env()`` and so inherits its scopes
    even when the identity comes out of the secret scope.

    Deliberately does not call :func:`app_config.azure.get_azure_client` to
    find that out: with no ``AZURE_*`` identity that falls through to the
    Databricks secret scope, which is a network read.
    """
    from .azure import AzureAuthConfig
    from .vault import _ARM_DEFAULT_SCOPE, VaultConfig

    config = VaultConfig.from_env()
    method = config.auth_method
    audience = config.azure_scopes or AzureAuthConfig.from_env().scopes or (
        _ARM_DEFAULT_SCOPE,
    )
    detail: dict[str, Any] = {
        "auth_method": method or "(nothing usable)",
        "address": _sourced("VAULT_ADDR", "vault", "address"),
        "namespace": _sourced("VAULT_NAMESPACE", "vault", "namespace"),
        "mount point": config.mount_point,
        "auth path": _sourced("VAULT_AUTH_PATH", "vault", "auth_path"),
        "role (app_name)": _sourced("VAULT_APP_NAME", "vault", "app_name"),
        "JWT audience": ", ".join(audience),
        "TLS verify": config.verify,
        "VAULT_TOKEN": _present("VAULT_TOKEN"),
        "VAULT_ROLE_ID": _present("VAULT_ROLE_ID"),
        "VAULT_SECRET_ID": _present("VAULT_SECRET_ID"),
    }

    if not config.address:
        return CheckResult(
            "vault",
            SKIP,
            "Vault is not configured (the OPENAI_API_KEY path)",
            detail,
        )
    if method is None:
        return CheckResult(
            "vault",
            FAIL,
            "VAULT_ADDR is set but there is no way to log in",
            detail,
            fix=(
                "Set VAULT_TOKEN; or VAULT_ROLE_ID and VAULT_SECRET_ID; or "
                "VAULT_APP_NAME for the Entra ID OIDC login."
            ),
        )
    if method != "azuread":
        return CheckResult(
            "vault",
            WARN,
            f"logging in with '{method}', not the deployment's azuread chain",
            detail,
            fix=(
                "Functional, but it bypasses the service principal the rest of "
                "the chain uses. Unset VAULT_TOKEN / the AppRole pair to fall "
                "back to VAULT_APP_NAME."
            ),
        )
    return CheckResult("vault", PASS, "logs in with the azuread JWT", detail)


def check_vault_live() -> CheckResult:
    """Hop 5, live: perform the login. Reads no secret, only authenticates."""
    from .vault import VaultError, get_vault_client

    try:
        get_vault_client().client
    except VaultError as exc:
        return CheckResult(
            "vault",
            FAIL,
            "the Vault login failed",
            {"error": str(exc)},
            fix=(
                "Check the role name (VAULT_APP_NAME) is bound at the auth "
                "path, and that its bound audience matches the JWT audience "
                "the offline run reports."
            ),
        )
    return CheckResult("vault", PASS, "logged in to Vault")


def check_gateway(*, live: bool = False) -> CheckResult:
    """Hop 6: the AI gateway credential.

    Reads the gateway secret through :mod:`app_config.vault` and reports its
    *fields*. It deliberately does **not** build an
    :class:`llm_client.LLMClientConfig`: :mod:`app_config` is a dependency leaf
    and must not import ``llm_client`` (``app_config/README.md``, "Boundary"),
    and ``LLMClientConfig.from_ai_gateway()`` is the single owner of that
    construction (Architecture.md invariant 12) — hand-assembling one here
    would silently drop the ``ai-gateway-version`` header and the gateway's
    rate-limit pacing. So this verifies the *credential*, not the client. A
    real run, or ``--check``, is what exercises the classmethod.

    No model is called on either path. This preflight never pays an LLM.
    """
    from .vault import VaultConfig

    config = VaultConfig.from_env()
    path = config.resolved_ai_gateway_path
    detail: dict[str, Any] = {
        "secret path": path or "unresolved",
        "token field": config.ai_gateway_key or "(the first known key)",
        "mount point": config.mount_point,
        "OPENAI_API_KEY": _present("OPENAI_API_KEY"),
    }

    if not config.address:
        return CheckResult(
            "gateway",
            SKIP,
            "Vault is not configured; the LLM key comes from OPENAI_API_KEY",
            detail,
        )
    if path is None:
        return CheckResult(
            "gateway",
            FAIL,
            "no AI gateway secret path resolves",
            detail,
            fix="Set VAULT_APP_NAME, or vault.ai_gateway_path in config.json.",
        )
    if not live:
        return CheckResult(
            "gateway", PASS, f"the gateway token would be read from '{path}'", detail
        )

    from .vault import (
        VaultError,
        ai_gateway_base_url,
        ai_gateway_token,
        ai_gateway_version,
        get_ai_gateway_secret,
    )

    try:
        secret = get_ai_gateway_secret()
        token = ai_gateway_token(secret)
    except VaultError as exc:
        return CheckResult(
            "gateway",
            FAIL,
            "could not read the AI gateway credential",
            detail | {"error": str(exc)},
            fix=(
                "Check the role can read this path, and that the secret has "
                "one of the known token keys."
            ),
        )
    detail["token"] = f"read (not shown), {len(token)} chars"
    detail["base_url"] = ai_gateway_base_url(secret) or "(not in the secret)"
    detail["gateway_version"] = ai_gateway_version(secret) or "(not in the secret)"
    if not detail["base_url"].startswith("http"):
        return CheckResult(
            "gateway",
            WARN,
            "the gateway token read, but the secret carries no base_url",
            detail,
            fix="Set llm_client.base_url in config.json, or add it to the secret.",
        )
    return CheckResult("gateway", PASS, "the gateway credential is readable", detail)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

#: Stages that cannot mean anything once an upstream hop has failed. The two
#: principal branches are NOT in each other's list: when `sp-hsv-*` resolves
#: and `saact-hsv-*` does not, running both is what tells you it is a key
#: naming problem rather than a credential one.
_AFTER_BOOTSTRAP = ("principal", "sharepoint", "graph", "vault", "gateway")


def run_checks(*, live: bool = False) -> list[CheckResult]:
    """Every hop, in dependency order.

    Offline unless *live*: nothing below contacts a workspace, a token
    endpoint, or Vault, and even ``live=True`` writes nothing and calls no
    model.
    """
    results = [check_process(), check_workspace()]

    bootstrap = check_bootstrap(live=live)
    results.append(bootstrap)
    if bootstrap.status == FAIL:
        why = "the secret-scope read cannot be authenticated (see bootstrap)"
        results.extend(_skipped(name, why) for name in _AFTER_BOOTSTRAP)
        logger.info(f"run_checks: stopped after bootstrap ({len(results)} stages)")
        return results

    principal = check_principal(live=live)
    results.append(principal)

    sharepoint = check_sharepoint(live=live)
    results.append(sharepoint)
    results.append(
        check_graph(live=live)
        if sharepoint.status != FAIL
        else _skipped("graph", "SharePoint's principal did not resolve")
    )

    if principal.status == FAIL:
        why = "the workspace principal did not resolve (see principal)"
        results.extend((_skipped("vault", why), _skipped("gateway", why)))
    else:
        vault = check_vault()
        if live and vault.status == PASS:
            vault = check_vault_live()
        results.append(vault)
        results.append(
            check_gateway(live=live)
            if vault.status != FAIL
            else _skipped("gateway", "the Vault login did not succeed")
        )

    logger.info(
        f"run_checks: {sum(1 for r in results if r.status == FAIL)} failure(s) "
        f"across {len(results)} stage(s)"
    )
    return results


def render(results: list[CheckResult], *, verbose: bool = False) -> str:
    """The results as the human-readable report, under this preflight's title."""
    return _render(results, verbose=verbose, title=_TITLE, passed=_PASSED)


def main(argv: list[str] | None = None) -> int:
    """Run the dry run from the command line. Non-zero if any hop failed."""
    parser = argparse.ArgumentParser(
        prog="python -m app_config.auth_check",
        description=(
            "Resolve every hop of the credential chain - the process shape, "
            "the Databricks workspace credential, the secret-scope bootstrap, "
            "both service principals in the scope, the Graph token, the Vault "
            "login and the AI gateway secret - reporting where each value came "
            "from. Contacts nothing unless --live. Never writes, and never "
            "calls a model."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Actually read the secret scope, mint the tokens and log in to "
        "Vault. Off by default: this command exists to be safe to run on a "
        "deployment whose credentials are in doubt.",
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
        help="Show what every hop resolved, not just the failing ones.",
    )
    parser.add_argument("--log-file", default=None, help="Also write the log here.")
    parser.add_argument(
        "--debug", action="store_true", help="DEBUG for the first-party loggers."
    )
    args = parser.parse_args(argv)

    configure_logging(debug=args.debug, log_file=args.log_file)
    load_dotenv_file()

    results = run_checks(live=args.live)
    print(to_json(results) if args.as_json else render(results, verbose=args.verbose))
    return 1 if any(r.status == FAIL for r in results) else 0


if __name__ == "__main__":  # pragma: no cover - exercised via main(argv)
    sys.exit(main())
