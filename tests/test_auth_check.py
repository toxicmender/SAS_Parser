"""Tests for the credential-chain dry run (app_config.auth_check).

Two properties carry most of the value, and both are about what the module
does *not* do.

**Offline is offline.** The default run must resolve all eight hops without
building a client, minting a token, or logging in anywhere. That is asserted
directly: every live seam is replaced with something that raises, and a clean
run is the proof.

**Nothing secret reaches the report.** A preflight prints its findings, so a
credential in a ``detail`` dict is a credential on someone's screen and in
their scrollback. Asserted against the serialised output, not the dict.

The seams here are one level *above* ``tests/test_auth_chain.py``'s. That suite
fakes the three external systems — the Databricks SDK, msal, hvac — and proves
the chain's links line up end to end. This module is a *reporter* over that
chain, so what it must be tested on is that it reports what those functions
return; re-faking the SDK to get there would test the plumbing twice and this
module's job zero times.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

import app_config
from app_config import auth_check, azure, databricks, sharepoint, vault
from app_config.sharepoint_check import FAIL, PASS, SKIP, WARN

SPN_TENANT = "check-tenant"
SPN_CLIENT = "check-client"
SPN_SECRET = "check-client-secret-VALUE"
GATEWAY_TOKEN = "gw-token-VALUE"
GRAPH_TOKEN = "graph-token-VALUE"
SCOPE = "udap_scripts_udappipelines_comp_APPSVC102630_dev"

_ENV = (
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_SECRET_SCOPE",
    "DATABRICKS_RUNTIME_VERSION",
    "DATABRICKS_AZURE_RESOURCE_ID",
    "DATABRICKS_CATALOG",
    "ARM_TENANT_ID",
    "ARM_CLIENT_ID",
    "ARM_CLIENT_SECRET",
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "SHAREPOINT_SECRET_SCOPE",
    "SHAREPOINT_SITE_ID",
    "VAULT_ADDR",
    "VAULT_TOKEN",
    "VAULT_ROLE_ID",
    "VAULT_SECRET_ID",
    "VAULT_APP_NAME",
    "OPENAI_API_KEY",
)

_MODULES = (app_config, azure, databricks, sharepoint, vault)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(app_config.ENV_VAR, str(cfg))
    for var in _ENV:
        monkeypatch.delenv(var, raising=False)
    for module in _MODULES:
        module.clear_cache()
    yield
    for module in _MODULES:
        module.clear_cache()


class _Shell:
    """An IPython shell stand-in; only its non-None-ness is read."""


@pytest.fixture
def child_process(monkeypatch):
    """The reported incident's shape: on a cluster, but not the notebook."""
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "18.3")
    monkeypatch.setenv("DATABRICKS_HOST", "https://adb-751152.4.azuredatabricks.net")
    monkeypatch.setenv("DATABRICKS_SECRET_SCOPE", SCOPE)


@pytest.fixture
def in_repl(monkeypatch):
    import types

    monkeypatch.setitem(
        sys.modules, "IPython", types.SimpleNamespace(get_ipython=lambda: _Shell())
    )


def _by_name(results: list[Any]) -> dict[str, Any]:
    return {result.name: result for result in results}


@pytest.fixture
def extras_installed(monkeypatch):
    """Report every optional auth dependency as present.

    CI's test job installs neither `databricks`, `azure` nor `sharepoint` — the
    in-memory paths need none of them — so without this the whole chain would
    stop at "databricks-sdk is not installed" and these tests would assert the
    venv rather than the logic. Owning the fact here also means the *absence*
    branch gets a test of its own instead of being whatever the environment did.
    """
    monkeypatch.setattr(auth_check, "_installed", lambda _distribution: True)


# ---------------------------------------------------------------------------
# Offline is offline
# ---------------------------------------------------------------------------


@pytest.fixture
def no_network(monkeypatch):
    """Make every live seam explode, so an offline run proves itself."""

    def _forbidden(*_args: Any, **_kwargs: Any):
        raise AssertionError("the offline dry run reached a live credential path")

    monkeypatch.setattr(databricks, "read_workspace_secrets", _forbidden)
    monkeypatch.setattr(databricks, "_import_workspace_client", _forbidden)
    monkeypatch.setattr(azure, "get_client_for_principal", _forbidden)
    monkeypatch.setattr(azure, "get_azure_client", _forbidden)
    monkeypatch.setattr(vault, "get_vault_client", _forbidden)
    monkeypatch.setattr(vault, "get_ai_gateway_secret", _forbidden)


def test_the_dry_run_contacts_nothing(extras_installed, no_network, child_process, monkeypatch):
    """The headline property, with the chain fully configured so it is tempted."""
    monkeypatch.setenv("VAULT_ADDR", "https://vault.example:8200")
    monkeypatch.setenv("VAULT_APP_NAME", "sas-parser")

    results = auth_check.run_checks()

    assert len(results) == 8


def test_the_dry_run_reproduces_the_incident(extras_installed, no_network, child_process):
    """The whole point: the offline report names the cause of the real failure.

    A `!python main.py ...` cell on a cluster, with the scope configured and
    the runtime credential unreachable. This is the shape that failed with
    `default auth: runtime: 'NoneType' object has no attribute 'parent_header'`
    eight hops later, and here it is diagnosed at hop zero, with no network.
    """
    stages = _by_name(auth_check.run_checks())

    assert stages["process"].status == FAIL
    assert "child process" in stages["process"].summary
    assert stages["workspace"].status == WARN
    assert "not the notebook" in stages["workspace"].summary
    assert stages["bootstrap"].status == WARN
    # The fix names the actual remedy, not the symptom.
    assert "run_in_notebook" in (stages["bootstrap"].fix or "")


def test_the_same_deployment_is_clean_from_the_notebook(
    extras_installed, no_network, child_process, in_repl
):
    """One thing differs — the process — and three hops change verdict."""
    stages = _by_name(auth_check.run_checks())

    assert stages["process"].status == PASS
    assert stages["workspace"].status == PASS
    assert stages["bootstrap"].status == PASS
    assert "the cluster runtime's own" in stages["bootstrap"].summary


def test_a_pat_makes_a_child_process_workable(extras_installed, no_network, child_process, monkeypatch):
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-child")

    stages = _by_name(auth_check.run_checks())

    assert stages["process"].status == WARN
    assert stages["workspace"].status == PASS
    assert stages["bootstrap"].status == PASS
    assert stages["bootstrap"].detail["bootstrap credential"] == "DATABRICKS_TOKEN"


def test_provenance_names_which_source_won(extras_installed, no_network, monkeypatch, tmp_path):
    """A value right in config.json and stale in the environment looks the same."""
    cfg = tmp_path / "config.json"
    cfg.write_text(
        '{"databricks": {"host": "https://from-config.net", '
        '"secret_scope": "from-config"}}',
        encoding="utf-8",
    )
    monkeypatch.setenv(app_config.ENV_VAR, str(cfg))
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-x")
    monkeypatch.setenv("DATABRICKS_SECRET_SCOPE", "from-the-environment")
    app_config.clear_cache()
    databricks.clear_cache()

    detail = _by_name(auth_check.run_checks())["workspace"].detail

    assert detail["host"] == "https://from-config.net (from config.json databricks.host)"
    assert detail["secret scope"] == (
        "from-the-environment (from $DATABRICKS_SECRET_SCOPE)"
    )


def test_secret_shaped_settings_are_reported_as_presence_only(no_network, monkeypatch):
    monkeypatch.setenv("ARM_TENANT_ID", "t")
    monkeypatch.setenv("ARM_CLIENT_ID", "c")
    monkeypatch.setenv("ARM_CLIENT_SECRET", "the-actual-arm-secret")
    monkeypatch.setenv("DATABRICKS_HOST", "https://adb-1.net")

    results = auth_check.run_checks()

    assert _by_name(results)["workspace"].detail["ARM_CLIENT_SECRET"] == "set"
    assert "the-actual-arm-secret" not in auth_check.to_json(results)
    assert "the-actual-arm-secret" not in auth_check.render(results, verbose=True)


def test_a_missing_sdk_fails_the_bootstrap_hop(no_network, child_process, monkeypatch):
    """Without databricks-sdk the scope cannot be read, and saying so is the job.

    A cluster always has it. Off one, a checkout without the `databricks` extra
    is a normal state — the in-memory paths need none of it — so this must
    report a missing dependency rather than a missing credential, which is the
    conclusion the SDK's own ImportError would lead someone to.
    """
    monkeypatch.setattr(
        auth_check, "_installed", lambda distribution: distribution != "databricks.sdk"
    )

    stages = _by_name(auth_check.run_checks())

    assert stages["bootstrap"].status == FAIL
    assert "databricks-sdk is not installed" in stages["bootstrap"].summary
    assert stages["bootstrap"].detail["databricks-sdk"] == "missing"


def test_a_missing_msal_skips_the_graph_hop(live_chain, monkeypatch):
    """No msal, no token — but that is a skip, not a failed credential."""
    monkeypatch.setattr(
        auth_check, "_installed", lambda distribution: distribution != "msal"
    )

    graph = _by_name(auth_check.run_checks(live=True))["graph"]

    assert graph.status == SKIP
    assert "msal is not installed" in graph.summary


# ---------------------------------------------------------------------------
# Gating — what a failed hop does to the ones after it
# ---------------------------------------------------------------------------


def test_a_failed_bootstrap_skips_everything_downstream(extras_installed, no_network, monkeypatch):
    """No credential can authenticate the scope read, so nothing after it can run."""
    monkeypatch.setenv("DATABRICKS_HOST", "https://adb-1.net")
    monkeypatch.setenv("DATABRICKS_SECRET_SCOPE", SCOPE)

    stages = _by_name(auth_check.run_checks())

    assert stages["bootstrap"].status == FAIL
    for name in ("principal", "sharepoint", "graph", "vault", "gateway"):
        assert stages[name].status == SKIP
        assert "bootstrap" in stages[name].summary


def test_the_two_principals_fail_independently(extras_installed, no_network, child_process, in_repl):
    """One scope, two principals under different keys.

    When `sp-hsv-*` resolves and `saact-hsv-*` does not, running both is what
    tells you it is a key-naming problem rather than a credential one — so
    neither gates the other.
    """
    stages = _by_name(auth_check.run_checks())

    assert stages["principal"].status == PASS
    assert stages["sharepoint"].status == PASS
    # Both branches were reached.
    assert "graph" in stages and "vault" in stages


def test_every_stage_is_present_even_when_the_chain_is_unconfigured(no_network):
    names = [result.name for result in auth_check.run_checks()]

    assert names == [
        "process",
        "workspace",
        "bootstrap",
        "principal",
        "sharepoint",
        "graph",
        "vault",
        "gateway",
    ]


# ---------------------------------------------------------------------------
# --live
# ---------------------------------------------------------------------------


class _FakeAzureClient:
    def __init__(self) -> None:
        self.scopes: list[Any] = []

    def get_token(self, scopes=None):
        self.scopes.append(scopes)
        return GRAPH_TOKEN


@pytest.fixture
def live_chain(monkeypatch, extras_installed, child_process, in_repl):
    """Every credential readable, faked at app_config's own seams."""
    monkeypatch.setenv("VAULT_ADDR", "https://vault.example:8200")
    monkeypatch.setenv("VAULT_APP_NAME", "sas-parser")

    def _secrets(scope, keys, *, config=None):
        assert scope == SCOPE
        return {
            key: {
                "tenantid": SPN_TENANT,
                "appid": SPN_CLIENT,
                "secret": SPN_SECRET,
            }[key.rsplit("-", 1)[-1]]
            for key in keys
        }

    monkeypatch.setattr(databricks, "read_workspace_secrets", _secrets)
    monkeypatch.setattr(
        azure, "get_client_for_principal", lambda *_a, **_k: _FakeAzureClient()
    )
    monkeypatch.setattr(vault, "get_vault_client", lambda: _FakeVaultClient())
    monkeypatch.setattr(
        vault,
        "get_ai_gateway_secret",
        lambda path=None: {"token": GATEWAY_TOKEN, "base_url": "https://gw.example/v1"},
    )


class _FakeVaultClient:
    """Enough of VaultClient for the login check and the gateway token read."""

    def __init__(self) -> None:
        self.config = vault.VaultConfig.from_env()

    @property
    def client(self):
        return object()


def test_live_resolves_every_hop(live_chain):
    stages = _by_name(auth_check.run_checks(live=True))

    assert stages["principal"].status == PASS
    assert stages["principal"].detail["client id"] == SPN_CLIENT
    assert stages["principal"].detail["client secret"] == "read (not shown)"
    assert stages["sharepoint"].status == PASS
    assert stages["vault"].status == PASS
    assert stages["gateway"].status == PASS
    assert stages["gateway"].detail["base_url"] == "https://gw.example/v1"


def test_live_never_prints_a_credential(live_chain):
    """The rule is that no stage puts one in `detail`; this is the check."""
    results = auth_check.run_checks(live=True)
    rendered = auth_check.render(results, verbose=True) + auth_check.to_json(results)

    for secret in (SPN_SECRET, GATEWAY_TOKEN, GRAPH_TOKEN):
        assert secret not in rendered


def test_live_reports_the_gateway_token_by_length_only(live_chain):
    detail = _by_name(auth_check.run_checks(live=True))["gateway"].detail

    assert detail["token"] == f"read (not shown), {len(GATEWAY_TOKEN)} chars"


def test_a_failed_scope_read_fails_the_bootstrap_hop(live_chain, monkeypatch):
    """The read is the bootstrap, so its failure is reported there, once.

    Everything downstream needs that same read, so the gate skips it rather
    than producing five copies of one error -- which is the shape the original
    incident had, where the same cause was reported as a SharePoint problem.
    """

    def _boom(scope, keys, *, config=None):
        raise databricks.DatabricksError("could not read secret 'sp-hsv-appid'")

    monkeypatch.setattr(databricks, "read_workspace_secrets", _boom)

    stages = _by_name(auth_check.run_checks(live=True))

    assert stages["bootstrap"].status == FAIL
    assert "sp-hsv-appid" in stages["bootstrap"].detail["error"]
    for name in ("principal", "sharepoint", "graph", "vault", "gateway"):
        assert stages[name].status == SKIP


# ---------------------------------------------------------------------------
# The report and the CLI
# ---------------------------------------------------------------------------


def test_the_report_is_ascii(extras_installed, no_network, child_process):
    """Read over RDP to a Windows host as often as in a modern terminal.

    Asserted over the stages this module authors. The `sharepoint` hop is
    `sharepoint_check`'s own result, and that module's summaries carry
    em-dashes; reporting on them here would only pin someone else's text.
    """
    authored = [
        result
        for result in auth_check.run_checks()
        if result.name != "sharepoint"
    ]
    report = auth_check.render(authored, verbose=True)

    assert report.isascii(), [c for c in report if not c.isascii()]
    assert "Credential chain dry run" in report


def test_the_cli_is_offline_by_default(extras_installed, no_network, child_process, capsys):
    code = auth_check.main([])

    assert code == 1  # the child-process FAIL
    assert "child process" in capsys.readouterr().out


def test_the_cli_emits_json(extras_installed, no_network, child_process, capsys):
    import json

    auth_check.main(["--json"])

    payload = json.loads(capsys.readouterr().out)
    assert [stage["name"] for stage in payload][:3] == [
        "process",
        "workspace",
        "bootstrap",
    ]


def test_the_cli_returns_zero_when_nothing_failed(extras_installed, no_network, child_process, in_repl):
    assert auth_check.main([]) == 0
