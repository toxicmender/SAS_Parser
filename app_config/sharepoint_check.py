"""A read-only preflight for the SharePoint deployment.

Every other way to find out whether SharePoint is wired up correctly is to
start a conversion — which reads the request list, then the scripts, then pays
an LLM, and only then discovers that the base path was pasted with the document
library prefix still on it. This module answers the same question in a few
seconds, reads nothing it does not have to, and **never writes**: no folder is
created, no file uploaded, no list item patched, no ``Status`` column touched.

The checks run in dependency order and each reports where its inputs came from,
because the failures a live deployment actually produces are configuration
failures, and "which of the three places this setting can come from won" is the
thing a stack trace never says::

    config    which SHAREPOINT_* / config.json values resolved, and from where
    imports   whether the optional `sharepoint` extra is actually installed
    identity  which Entra ID principal will be used, and where it comes from
    secrets   actually read that principal out of the Databricks secret scope
    token     mint a real Graph token, and report what the claims say it is
    site      resolve the site id and the document library
    base      list the applications root the base path points at
    lists     read one item from each configured list, reporting its columns

A stage that cannot run because an earlier one failed is reported as skipped
rather than silently passed, and the exit status is non-zero if any stage
failed. Stages after ``imports`` are omitted entirely under ``--offline``,
which checks the configuration alone and is the form that runs in CI.

The token stage is the one worth reading twice. A client-credentials Graph
token carries the granted application permissions in its ``roles`` claim, so
when the library returns 403 this is what distinguishes "the app registration
was never granted ``Sites.ReadWrite.All``" from "it was granted but admin
consent was never clicked" — a distinction that costs an afternoon to make any
other way. The claims are decoded locally and never verified: this is a
description of the token we hold, not an authentication decision.

Usage
-----
::

    python -m app_config.sharepoint_check              # the full preflight
    python -m app_config.sharepoint_check --offline    # config only, no network
    python -m app_config.sharepoint_check --json       # machine-readable
    sas-parser --check                                 # the same, via the CLI

Logger name: ``app_config.sharepoint_check``.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import importlib.util
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from . import get_value, load_dotenv_file
from .logging_setup import configure_logging, redact

logger = logging.getLogger(__name__)

PASS = "pass"
FAIL = "fail"
WARN = "warn"
SKIP = "skip"

#: How many children of a folder to name in the report before summarising.
_SAMPLE = 5

#: Marks in the human-readable report. ASCII, because this output is read over
#: an RDP session to a Windows host as often as in a modern terminal.
_MARKS = {PASS: "[ ok ]", FAIL: "[FAIL]", WARN: "[warn]", SKIP: "[skip]"}


@dataclass
class CheckResult:
    """One stage's outcome.

    Attributes
    ----------
    name : str
        The stage id, as the module docstring lists them.
    status : str
        :data:`PASS`, :data:`FAIL`, :data:`WARN` or :data:`SKIP`.
    summary : str
        One line saying what happened.
    detail : dict[str, Any]
        Structured findings, rendered as indented ``key: value`` lines and
        emitted as-is by ``--json``. Values pass through
        :func:`~app_config.logging_setup.redact` on the way out.
    fix : str | None
        What to change when this stage failed — the config key or grant to go
        and set, not a restatement of the error.
    """

    name: str
    status: str
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)
    fix: str | None = None

    @property
    def ok(self) -> bool:
        """True when this stage did not fail (a warning or skip is not a failure)."""
        return self.status != FAIL


# ---------------------------------------------------------------------------
# Where a setting came from
# ---------------------------------------------------------------------------


def setting_source(env_var: str, section: str, key: str) -> tuple[Any, str]:
    """The resolved value of one setting and the name of the place it came from.

    Mirrors the ``os.environ.get(...) or get_value(...)`` precedence every
    ``from_env`` in this package uses. Reporting the source is the point:
    a value that is right in ``config.json`` and wrong (or stale) in the
    environment looks identical to a correct one until you know which won.
    """
    env = os.environ.get(env_var)
    if env:
        return env, f"${env_var}"
    configured = get_value(section, key)
    if configured is not None:
        return configured, f"config.json {section}.{key}"
    return None, "unset"


def _redacted(value: Any) -> Any:
    """*value* with any secret-shaped content masked, for display."""
    return redact(value) if isinstance(value, str) else value


# ---------------------------------------------------------------------------
# JWT claims
# ---------------------------------------------------------------------------


def decode_claims(token: str) -> dict[str, Any]:
    """The payload of a JWT *token*, decoded without verifying its signature.

    Purely descriptive — the token was just minted by MSAL over TLS and is
    about to be handed to Graph, which is what actually validates it. Decoding
    it here answers "which app is this, in which tenant, for which audience,
    with which granted roles", which is the whole of a Graph 403 diagnosis.

    Returns ``{}`` when *token* is not a three-part JWT (Entra can issue opaque
    tokens, and a stand-in provider in a test returns whatever it likes).
    """
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)  # b64url without padding
    try:
        raw = base64.urlsafe_b64decode(payload)
        claims = json.loads(raw)
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return {}
    return claims if isinstance(claims, dict) else {}


def _claim_summary(claims: dict[str, Any]) -> dict[str, Any]:
    """The handful of claims worth printing, from a decoded token."""
    roles = claims.get("roles")
    return {
        "audience (aud)": claims.get("aud"),
        "tenant (tid)": claims.get("tid"),
        "app id (appid)": claims.get("appid") or claims.get("azp"),
        "app name": claims.get("app_displayname"),
        "granted roles": ", ".join(sorted(roles)) if isinstance(roles, list) else roles,
    }


# The application permission every operation in app_config.sharepoint needs.
# Sites.Selected is the least-privilege alternative, and works only when the
# specific site has been granted to the app separately, so it is reported as a
# warning rather than accepted silently.
_REQUIRED_ROLE = "Sites.ReadWrite.All"
_SUFFICIENT_ROLES = frozenset({"Sites.ReadWrite.All", "Sites.FullControl.All"})


# ---------------------------------------------------------------------------
# The stages
# ---------------------------------------------------------------------------


def check_config(config: Any) -> CheckResult:
    """Report every resolved SharePoint setting and where it came from."""
    settings = [
        ("site_hostname", "SHAREPOINT_SITE_HOSTNAME", "site_hostname"),
        ("site_path", "SHAREPOINT_SITE_PATH", "site_path"),
        ("site_id", "SHAREPOINT_SITE_ID", "site_id"),
        ("drive_id", "SHAREPOINT_DRIVE_ID", "drive_id"),
        (
            "file_server_base_path",
            "SHAREPOINT_FILE_SERVER_BASE_PATH",
            "file_server_base_path",
        ),
        ("secret_scope", "SHAREPOINT_SECRET_SCOPE", "secret_scope"),
        ("list_id_sas_requests", "SHAREPOINT_LIST_ID_SAS_REQUESTS", "list_id_sas_requests"),
        (
            "list_id_sas_conversions",
            "SHAREPOINT_LIST_ID_SAS_CONVERSIONS",
            "list_id_sas_conversions",
        ),
        ("list_id_xref", "SHAREPOINT_LIST_ID_XREF", "list_id_xref"),
        (
            "list_id_sas_complexity",
            "SHAREPOINT_LIST_ID_SAS_COMPLEXITY",
            "list_id_sas_complexity",
        ),
    ]
    detail: dict[str, Any] = {}
    for attr, env_var, key in settings:
        raw, source = setting_source(env_var, "sharepoint", key)
        resolved = getattr(config, attr, None)
        # The resolved value is what the client will use; the raw one is what
        # was written down. They differ where a normaliser ran, and that
        # difference is itself a finding — a base path shortened by the
        # "Shared Documents/" strip is the single most common surprise here.
        if raw is not None and str(raw).strip("/ ") != str(resolved or "").strip("/ "):
            detail[attr] = f"{resolved!r} (from {source} as {raw!r}, normalised)"
        else:
            detail[attr] = f"{resolved!r} (from {source})"
    detail["scopes"] = ", ".join(config.scopes)
    detail["timeout"] = f"{config.timeout}s"
    detail["config.json"] = _config_path()

    if not config.resolved_site_id:
        return CheckResult(
            "config",
            FAIL,
            "no SharePoint site is configured",
            detail,
            fix=(
                "set SHAREPOINT_SITE_ID, or SHAREPOINT_SITE_HOSTNAME together "
                "with SHAREPOINT_SITE_PATH (or the sharepoint.* equivalents in "
                "config.json)"
            ),
        )
    if not config.list_id_sas_requests:
        return CheckResult(
            "config",
            WARN,
            f"site {config.resolved_site_id!r} is configured, but no requests list is",
            detail,
            fix=(
                "set SHAREPOINT_LIST_ID_SAS_REQUESTS to run from the request "
                "queue; without it only --app/--request-id-free local mode works"
            ),
        )
    return CheckResult(
        "config", PASS, f"site {config.resolved_site_id!r}", detail
    )


def _config_path() -> str:
    """Which ``config.json`` the settings above were read from, if any."""
    from . import ENV_VAR, _candidate_paths

    for path in _candidate_paths():
        if path.is_file():
            override = f" (via ${ENV_VAR})" if os.environ.get(ENV_VAR) else ""
            return f"{path}{override}"
    return "none found (hard defaults apply)"


def check_imports(*, needed: bool = True) -> CheckResult:
    """Report whether the optional ``sharepoint`` extra is installed.

    Parameters
    ----------
    needed : bool
        Whether this run would have to build a Graph client of its own. When a
        caller supplies a pre-built one, the SDK is not required and a missing
        extra is reported as skipped rather than failed — the question the
        stage asks ("can we build a client?") is moot when we were handed one.
        That is not a testing convenience: it is the same reasoning
        :func:`run_checks` already applies when deciding whether a missing SDK
        should stop the later stages.
    """
    modules = ("msgraph", "msgraph_core", "kiota_abstractions", "msal", "httpx")
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    detail = {
        name: ("missing" if name in missing else "installed") for name in modules
    }
    if missing and not needed:
        return CheckResult(
            "imports",
            SKIP,
            f"not needed (a client was supplied); missing: {', '.join(missing)}",
            detail,
        )
    if missing:
        return CheckResult(
            "imports",
            FAIL,
            f"missing: {', '.join(missing)}",
            detail,
            fix='uv pip install -e ".[sharepoint]"',
        )
    return CheckResult("imports", PASS, "the sharepoint extra is installed", detail)


def check_identity(config: Any) -> CheckResult:
    """Report which Entra ID principal will authenticate, without minting."""
    from .azure import AzureAuthConfig

    if config.secret_scope:
        from .databricks import get_databricks_config, in_databricks_runtime

        databricks = get_databricks_config()
        detail = {
            "principal": "SharePoint's own service principal",
            "secret scope": config.secret_scope,
            "tenant id key": config.tenant_id_key,
            "client id key": config.client_id_key,
            "client secret key": config.client_secret_key,
            "scope read as": databricks.auth_method or "(nothing usable)",
            "on a cluster": in_databricks_runtime(),
        }
        return CheckResult(
            "identity",
            PASS,
            f"the service principal in Databricks secret scope "
            f"{config.secret_scope!r} (not yet read — see the secrets check)",
            detail,
        )

    azure_config = AzureAuthConfig.from_env()
    tenant, tenant_source = setting_source("AZURE_TENANT_ID", "azure", "tenant_id")
    client, client_source = setting_source("AZURE_CLIENT_ID", "azure", "client_id")
    detail = {
        "principal": "the shared app_config.azure identity",
        "tenant_id": f"{tenant!r} (from {tenant_source})",
        "client_id": f"{client!r} (from {client_source})",
        "flow": azure_config.auth_flow,
        "authority": azure_config.authority,
        "credential": (
            "certificate"
            if azure_config.has_certificate and not azure_config.client_secret
            else "client secret"
            if azure_config.client_secret
            else "none"
        ),
        "TLS verify": azure_config.verify,
        "proxies": sorted(azure_config.proxies) if azure_config.proxies else None,
    }
    if not (azure_config.tenant_id and azure_config.client_id):
        return CheckResult(
            "identity",
            FAIL,
            "no Entra ID identity is configured",
            detail,
            fix=(
                "point SHAREPOINT_SECRET_SCOPE (or DATABRICKS_SECRET_SCOPE) at "
                "the scope holding SharePoint's service principal, or set "
                "AZURE_TENANT_ID and AZURE_CLIENT_ID plus AZURE_CLIENT_SECRET"
            ),
        )
    return CheckResult(
        "identity", PASS, f"the shared identity {azure_config.client_id}", detail
    )


def check_secret_scope(config: Any) -> CheckResult:
    """Actually read SharePoint's service principal out of the secret scope.

    Skipped when no scope is configured (the local path authenticates with the
    shared ``AZURE_*`` identity, which the identity stage already resolved).

    This is a stage of its own because the scope read is a *separate*
    authentication from the one it produces: reaching the scope needs a
    workspace credential — the Databricks runtime's or a PAT — and the
    principal it hands back is what then authenticates to Graph. Folding it
    into the identity check made a configured scope look like a working one,
    which is precisely the gap that hides the most common cluster failure:
    running the entry point from a ``!python`` cell, where the runtime
    environment variable is inherited but the notebook's credential is not.

    Only the non-secret half of the principal is reported; the client secret is
    never read into the result.
    """
    if not config.secret_scope:
        return CheckResult(
            "secrets", SKIP, "no secret scope; using the shared Entra identity"
        )
    from .azure import AzureAuthError, databricks_service_principal
    from .databricks import DatabricksError

    try:
        principal = databricks_service_principal(config.secret_scope, config.secret_keys)
    # AzureAuthError is the wrapper databricks_service_principal raises;
    # DatabricksError is caught too so a future direct call reports the same.
    except (AzureAuthError, DatabricksError) as exc:
        return CheckResult(
            "secrets",
            FAIL,
            f"could not read the principal out of scope {config.secret_scope!r}",
            {"error": str(exc)},
            fix=(
                "the scope read needs a workspace credential of its own — the "
                "cluster runtime's, or DATABRICKS_TOKEN. A '!python ...' cell "
                "inherits the runtime env var but not the credential, so run "
                "in the notebook's own Python or set DATABRICKS_TOKEN"
            ),
        )
    return CheckResult(
        "secrets",
        PASS,
        f"read SharePoint's principal from scope {config.secret_scope!r}",
        {
            "tenant id": principal.tenant_id,
            "client id": principal.client_id,
            "client secret": "read (not shown)",
        },
    )


def check_token(client: Any) -> CheckResult:
    """Mint a Graph token and report what its claims say the caller is."""
    from .sharepoint import SharePointError

    try:
        token = client.get_token()
    except SharePointError as exc:
        return CheckResult(
            "token",
            FAIL,
            "could not mint a Microsoft Graph token",
            {"error": str(exc)},
            fix=(
                "check the client secret has not expired, that the tenant and "
                "client ids match the app registration, and — on a "
                "TLS-intercepting network — that AZURE_VERIFY names the "
                "corporate CA bundle"
            ),
        )

    claims = decode_claims(token)
    if not claims:
        return CheckResult(
            "token",
            PASS,
            "a Graph token was minted (opaque — no claims to report)",
            {"token length": len(token)},
        )

    detail = _claim_summary(claims)
    roles = claims.get("roles")
    granted = set(roles) if isinstance(roles, list) else set()
    if granted & _SUFFICIENT_ROLES:
        return CheckResult(
            "token", PASS, f"a Graph token with {_REQUIRED_ROLE}", detail
        )
    if granted:
        return CheckResult(
            "token",
            WARN,
            f"a Graph token, but without {_REQUIRED_ROLE}",
            detail,
            fix=(
                f"the app has {', '.join(sorted(granted))}. Reads may work and "
                f"writes will 403. Grant {_REQUIRED_ROLE} (or Sites.Selected "
                f"plus a per-site grant for this site) and click admin consent"
            ),
        )
    return CheckResult(
        "token",
        WARN,
        "a Graph token with no application permissions at all",
        detail,
        fix=(
            f"the 'roles' claim is empty, which means no application permission "
            f"has been consented. Add {_REQUIRED_ROLE} to the app registration "
            f"and grant admin consent; a token mints fine without it, and every "
            f"call then returns 403"
        ),
    )


def check_site(client: Any) -> CheckResult:
    """Resolve the drive for the configured site — the first real Graph call."""
    from .sharepoint import SharePointError

    site_id = client.config.resolved_site_id
    try:
        drive_id = client._drive_id()
    except SharePointError as exc:
        return CheckResult(
            "site",
            FAIL,
            f"could not reach site {site_id!r}",
            {"error": str(exc)},
            fix=(
                "a 404 here means the site id or hostname+path is wrong (the "
                "path is the /sites/Name part of the URL); a 403 means the "
                "token's permissions do not cover this site"
            ),
        )
    return CheckResult(
        "site",
        PASS,
        f"site {site_id!r} resolves",
        {
            "site id": site_id,
            "drive id": drive_id,
            "drive source": (
                "SHAREPOINT_DRIVE_ID"
                if client.config.drive_id
                else "the site's default document library"
            ),
        },
    )


def check_base_path(client: Any) -> CheckResult:
    """List the applications root, so a wrong base path fails here and not mid-run."""
    from .sharepoint import SharePointError

    base = client.config.file_server_base_path
    try:
        entries = client.list_directory(base)
    except SharePointError as exc:
        return CheckResult(
            "base",
            FAIL,
            f"could not list the applications root {base or '/'!r}",
            {"error": str(exc)},
            fix=(
                "SHAREPOINT_FILE_SERVER_BASE_PATH is relative to the document "
                "library, not the site: a path copied from the browser keeps a "
                "leading 'Shared Documents/', which is stripped automatically, "
                "but any other prefix is looked for literally"
            ),
        )
    folders = [e["name"] for e in entries if e["is_folder"]]
    detail: dict[str, Any] = {
        "path": base or "(library root)",
        "children": f"{len(entries)} ({len(folders)} folder(s))",
    }
    if folders:
        shown = ", ".join(folders[:_SAMPLE])
        more = f", +{len(folders) - _SAMPLE} more" if len(folders) > _SAMPLE else ""
        detail["applications"] = f"{shown}{more}"
    if not folders:
        return CheckResult(
            "base",
            WARN,
            f"{base or 'the library root'!r} exists but holds no folders",
            detail,
            fix=(
                "each application is a folder here, holding scripts_original/. "
                "An empty root usually means the base path points one level too "
                "high or too low"
            ),
        )
    return CheckResult(
        "base", PASS, f"{len(folders)} application folder(s) under {base or '/'!r}", detail
    )


def check_lists(client: Any) -> list[CheckResult]:
    """Read one item from each configured list, reporting the columns it has.

    One item, not the list: enough to prove the id resolves and the identity
    can read it, and — because SharePoint's *internal* column names are what
    :mod:`conversion.requests` addresses and they are not the display names —
    enough to show what those internal names actually are on this tenant. That
    mapping is transcribed by hand in the domain layers, so seeing the real
    thing is what catches a renamed or re-created column.
    """
    from .sharepoint import LIST_KINDS, SharePointError

    results: list[CheckResult] = []
    for kind in sorted(LIST_KINDS):
        name = f"list:{kind}"
        try:
            list_id = client.config.list_id(kind)
        except SharePointError as exc:
            results.append(
                CheckResult(name, SKIP, "not configured", {"reason": str(exc)})
            )
            continue
        try:
            rows = client.list_items(list_id, top=1)
        except SharePointError as exc:
            results.append(
                CheckResult(
                    name,
                    FAIL,
                    f"could not read list {list_id!r}",
                    {"error": str(exc)},
                    fix=(
                        "a 404 means the list id is wrong — take it from the "
                        "list's Settings page URL, not its display name; a 403 "
                        "means the token cannot read this site's lists"
                    ),
                )
            )
            continue
        detail: dict[str, Any] = {"list id": list_id}
        if not rows:
            results.append(
                CheckResult(name, PASS, "readable, but empty", detail)
            )
            continue
        columns = sorted(rows[0].get("fields", {}))
        detail["columns"] = ", ".join(columns) if columns else "(none expanded)"
        results.append(
            CheckResult(name, PASS, f"readable, {len(columns)} column(s)", detail)
        )
    return results


# ---------------------------------------------------------------------------
# Driving them
# ---------------------------------------------------------------------------


def run_checks(*, offline: bool = False, client: Any | None = None) -> list[CheckResult]:
    """Run the preflight and return one :class:`CheckResult` per stage.

    Parameters
    ----------
    offline : bool
        Stop after ``imports``: resolve and report the configuration without
        contacting Entra ID or Graph. Everything after is reported as skipped.
    client : Any | None
        A pre-built :class:`~app_config.sharepoint.SharePointClient` to probe.
        ``None`` (default) builds one from the environment. The injection point
        for tests, matching the one the transport itself offers.

    Never writes: the network stages issue reads only.
    """
    from .sharepoint import SharePointClient, SharePointConfig

    config = client.config if client is not None else SharePointConfig.from_env()
    results = [check_config(config)]
    imports = check_imports(needed=client is None)
    results.append(imports)

    later = ("identity", "secrets", "token", "site", "base", "lists")
    if offline:
        results.extend(
            CheckResult(stage, SKIP, "not run (--offline)") for stage in later
        )
        return results
    # A missing SDK is only fatal for a client we would have to build ourselves;
    # an injected one already works without it.
    if not imports.ok and client is None:
        results.extend(
            CheckResult(stage, SKIP, "the sharepoint extra is not installed")
            for stage in later
        )
        return results

    identity = check_identity(config)
    results.append(identity)
    if not identity.ok:
        results.extend(
            CheckResult(stage, SKIP, "no identity to authenticate with")
            for stage in later[1:]
        )
        return results

    if client is None:
        client = SharePointClient(config)

    stages: list[tuple[str, Callable[[], CheckResult]]] = [
        ("secrets", lambda: check_secret_scope(config)),
        ("token", lambda: check_token(client)),
        ("site", lambda: check_site(client)),
        ("base", lambda: check_base_path(client)),
    ]
    for index, (stage, run) in enumerate(stages):
        result = run()
        results.append(result)
        # Each of these gates what follows: without a token nothing
        # authenticates, and without a drive there is no library to look in.
        # A WARN (e.g. missing roles) is deliberately not a gate — letting the
        # later stages run is what shows *which* calls the grant actually
        # blocks, which is more informative than stopping.
        if not result.ok:
            remaining = [name for name, _ in stages[index + 1 :]]
            results.extend(
                CheckResult(skipped, SKIP, f"the {stage} check failed")
                for skipped in [*remaining, "lists"]
            )
            return results

    results.extend(check_lists(client))
    return results


def render(results: list[CheckResult], *, verbose: bool = False) -> str:
    """The results as the human-readable report."""
    lines: list[str] = ["SharePoint preflight", "=" * 20, ""]
    for result in results:
        lines.append(f"{_MARKS.get(result.status, '[ ?? ]')} {result.name}: {result.summary}")
        show_detail = verbose or result.status in (FAIL, WARN)
        if show_detail:
            for key, value in result.detail.items():
                lines.append(f"         {key}: {_redacted(value)}")
        if result.fix:
            lines.append(f"         -> {result.fix}")
    failures = [r for r in results if r.status == FAIL]
    warnings = [r for r in results if r.status == WARN]
    lines.append("")
    if failures:
        lines.append(
            f"FAILED: {len(failures)} check(s) failed "
            f"({', '.join(r.name for r in failures)})"
        )
    elif warnings:
        lines.append(
            f"PASSED with {len(warnings)} warning(s) "
            f"({', '.join(r.name for r in warnings)})"
        )
    else:
        lines.append("PASSED: SharePoint is reachable and configured")
    if not verbose and not failures and not warnings:
        lines.append("(re-run with --verbose to see what each check resolved)")
    return "\n".join(lines)


def to_json(results: list[CheckResult]) -> str:
    """The results as JSON, for a caller that wants to assert on them."""
    return json.dumps(
        [
            {
                "name": r.name,
                "status": r.status,
                "summary": r.summary,
                "detail": {k: _redacted(v) for k, v in r.detail.items()},
                "fix": r.fix,
            }
            for r in results
        ],
        indent=2,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the preflight from the command line. Non-zero if any stage failed."""
    parser = argparse.ArgumentParser(
        prog="python -m app_config.sharepoint_check",
        description=(
            "Read-only preflight for the SharePoint deployment: resolves the "
            "configuration, mints a Graph token, and reads the library and "
            "lists. Writes nothing."
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Check the configuration only — contact neither Entra ID nor Graph.",
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
        help="Also write the log to this file (appended).",
    )
    parser.add_argument(
        "--trace-http",
        action="store_true",
        help="Log every Graph request the SDK makes. Verbose; pair with --log-file.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable DEBUG logging."
    )
    args = parser.parse_args(argv)

    configure_logging(
        debug=args.debug, log_file=args.log_file, trace_http=args.trace_http
    )
    # Before any setting is resolved, and with the same precedence a real run
    # uses, so the preflight reports the environment the run would see.
    load_dotenv_file()

    results = run_checks(offline=args.offline)
    print(to_json(results) if args.as_json else render(results, verbose=args.verbose))
    return 1 if any(r.status == FAIL for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
