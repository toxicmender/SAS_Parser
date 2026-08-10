"""
Tests for the read-only SharePoint preflight (app_config.sharepoint_check).

No live SharePoint and no msgraph-sdk: every network stage is driven through a
stub client with the same surface run_checks() uses, and the config stage reads
a controlled environment plus a tmp config.json. The point of most of these is
the *diagnosis* rather than the pass/fail — a check that says "failed" without
naming the setting to change is the situation this module exists to end.
"""

from __future__ import annotations

import base64
import json
import logging
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

import app_config
from app_config import azure, sharepoint, sharepoint_check
from app_config.sharepoint_check import FAIL, PASS, SKIP, WARN

_ENV = (
    "SHAREPOINT_SITE_HOSTNAME",
    "SHAREPOINT_SITE_PATH",
    "SHAREPOINT_SITE_ID",
    "SHAREPOINT_DRIVE_ID",
    "SHAREPOINT_SCOPES",
    "SHAREPOINT_FILE_SERVER_BASE_PATH",
    "SHAREPOINT_SECRET_SCOPE",
    "SHAREPOINT_LIST_ID_SAS_REQUESTS",
    "SHAREPOINT_LIST_ID_SAS_CONVERSIONS",
    "SHAREPOINT_LIST_ID_XREF",
    "SHAREPOINT_LIST_ID_SAS_COMPLEXITY",
    "DATABRICKS_SECRET_SCOPE",
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Empty config file, no SharePoint/Azure env vars, all caches cleared."""
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(app_config.ENV_VAR, str(cfg))
    for var in _ENV:
        monkeypatch.delenv(var, raising=False)
    app_config.clear_cache()
    sharepoint.clear_cache()
    azure.clear_cache()
    handlers = list(logging.getLogger().handlers)
    yield cfg
    app_config.clear_cache()
    sharepoint.clear_cache()
    azure.clear_cache()
    # configure_logging() replaces the root handlers; put back what pytest had
    # so a test that opens a log file does not keep it open for the next one.
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers


def _set(cfg_path, mapping) -> None:
    cfg_path.write_text(json.dumps(mapping), encoding="utf-8")
    app_config.clear_cache()


def _by_name(results) -> dict[str, Any]:
    return {r.name: r for r in results}


# ---------------------------------------------------------------------------
# A stub client with the surface run_checks() drives
# ---------------------------------------------------------------------------


def _jwt(claims: dict[str, Any]) -> str:
    """A three-part token whose payload decodes to *claims*, unsigned."""

    def segment(payload: dict[str, Any]) -> str:
        raw = json.dumps(payload).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{segment({'alg': 'none'})}.{segment(claims)}.signature"


_GRANTED = {
    "aud": "https://graph.microsoft.com",
    "tid": "tenant-1",
    "appid": "client-1",
    "app_displayname": "SAS Parser",
    "roles": ["Sites.ReadWrite.All"],
}


class _StubClient:
    """Stands in for SharePointClient across every stage the preflight calls."""

    def __init__(
        self,
        config,
        *,
        token: str | None = None,
        drive_id: str = "DRV",
        entries: list[dict[str, Any]] | None = None,
        rows: dict[str, list[dict[str, Any]]] | None = None,
        fail: dict[str, str] | None = None,
    ):
        self.config = config
        self._token = token if token is not None else _jwt(_GRANTED)
        self._drive = drive_id
        self._entries = entries if entries is not None else [
            {"name": "MyApp", "is_folder": True},
            {"name": "readme.txt", "is_folder": False},
        ]
        self._rows = rows or {}
        self._fail = fail or {}
        self.list_item_calls: list[tuple[str, Any]] = []

    def _boom(self, stage: str) -> None:
        message = self._fail.get(stage)
        if message:
            raise sharepoint.SharePointError(message)

    def get_token(self, scopes=None):
        self._boom("token")
        return self._token

    def _drive_id(self):
        self._boom("site")
        return self._drive

    def list_directory(self, path=""):
        self._boom("base")
        return list(self._entries)

    def list_items(self, list_id, *, top=None, **_kwargs):
        self.list_item_calls.append((list_id, top))
        self._boom("lists")
        return list(self._rows.get(list_id, []))


@pytest.fixture
def _readable_scope(monkeypatch):
    """The secret scope reads cleanly, so run_checks() reaches the Graph stages.

    Without this the `secrets` stage tries a real Databricks scope read and
    fails first — which is correct behaviour, and exactly what
    test_a_failed_secrets_check_skips_the_graph_stages asserts.
    """
    from app_config import azure as azure_mod
    from app_config.databricks import AzureServicePrincipal

    monkeypatch.setattr(
        azure_mod,
        "databricks_service_principal",
        lambda scope, keys: AzureServicePrincipal("t-1", "c-1", "s-1"),
    )


def _configured(**overrides) -> sharepoint.SharePointConfig:
    """A config with a site, a drive and a requests list — enough to reach the end."""
    settings: dict[str, Any] = {
        "site_id": "SITE",
        "drive_id": "DRV",
        "secret_scope": "kv-scope",
        "list_id_sas_requests": "L-REQ",
    }
    settings.update(overrides)
    return sharepoint.SharePointConfig(**settings)


# ---------------------------------------------------------------------------
# The config stage — provenance is the product
# ---------------------------------------------------------------------------


def test_config_reports_which_source_each_setting_came_from(monkeypatch, _isolated):
    _set(_isolated, {"sharepoint": {"site_id": "FROM-CONFIG", "drive_id": "D-CONFIG"}})
    monkeypatch.setenv("SHAREPOINT_SITE_ID", "FROM-ENV")

    result = sharepoint_check.check_config(sharepoint.SharePointConfig.from_env())

    assert result.status == WARN  # no requests list configured
    assert "$SHAREPOINT_SITE_ID" in result.detail["site_id"]
    assert "FROM-ENV" in result.detail["site_id"]
    # The environment wins, and the shadowed config.json value is not reported
    # as the source — which is the confusion the stage exists to remove.
    assert "FROM-CONFIG" not in result.detail["site_id"]
    assert "config.json sharepoint.drive_id" in result.detail["drive_id"]


def test_config_flags_a_base_path_that_was_normalised(monkeypatch):
    monkeypatch.setenv(
        "SHAREPOINT_FILE_SERVER_BASE_PATH", "Shared Documents/Apps/Migration"
    )
    result = sharepoint_check.check_config(sharepoint.SharePointConfig.from_env())

    reported = result.detail["file_server_base_path"]
    assert "'Apps/Migration'" in reported
    assert "normalised" in reported
    assert "Shared Documents/Apps/Migration" in reported


def test_config_without_a_site_fails_and_names_the_keys():
    result = sharepoint_check.check_config(sharepoint.SharePointConfig())

    assert result.status == FAIL
    assert result.fix is not None
    assert "SHAREPOINT_SITE_ID" in result.fix
    assert "SHAREPOINT_SITE_HOSTNAME" in result.fix


def test_config_with_a_site_but_no_requests_list_warns(monkeypatch):
    monkeypatch.setenv("SHAREPOINT_SITE_ID", "SITE")
    result = sharepoint_check.check_config(sharepoint.SharePointConfig.from_env())

    assert result.status == WARN
    assert result.fix is not None
    assert "SHAREPOINT_LIST_ID_SAS_REQUESTS" in result.fix


def test_config_passes_when_site_and_requests_list_are_set(monkeypatch):
    monkeypatch.setenv("SHAREPOINT_SITE_ID", "SITE")
    monkeypatch.setenv("SHAREPOINT_LIST_ID_SAS_REQUESTS", "L-REQ")
    result = sharepoint_check.check_config(sharepoint.SharePointConfig.from_env())

    assert result.status == PASS


# ---------------------------------------------------------------------------
# The identity stage — which principal, without minting
# ---------------------------------------------------------------------------


def test_identity_reports_the_secret_scope_principal():
    config = sharepoint.SharePointConfig(site_id="SITE", secret_scope="kv-scope")
    result = sharepoint_check.check_identity(config)

    assert result.status == PASS
    assert result.detail["secret scope"] == "kv-scope"
    assert result.detail["tenant id key"] == sharepoint.DEFAULT_TENANT_ID_KEY


def test_identity_reports_the_shared_principal(monkeypatch):
    monkeypatch.setenv("AZURE_TENANT_ID", "t-1")
    monkeypatch.setenv("AZURE_CLIENT_ID", "c-1")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "s-1")
    result = sharepoint_check.check_identity(sharepoint.SharePointConfig(site_id="S"))

    assert result.status == PASS
    assert "c-1" in result.detail["client_id"]
    assert result.detail["credential"] == "client secret"
    # The secret itself is never part of the report.
    assert "s-1" not in json.dumps(result.detail)


def test_identity_without_any_principal_fails():
    result = sharepoint_check.check_identity(sharepoint.SharePointConfig(site_id="S"))

    assert result.status == FAIL
    assert result.fix is not None
    assert "SHAREPOINT_SECRET_SCOPE" in result.fix


# ---------------------------------------------------------------------------
# The secrets stage — a configured scope is not a readable one
# ---------------------------------------------------------------------------


def test_secrets_is_skipped_without_a_scope():
    config = sharepoint.SharePointConfig(site_id="SITE")
    result = sharepoint_check.check_secret_scope(config)

    assert result.status == SKIP


def test_secrets_reports_the_principal_without_its_secret(monkeypatch):
    from app_config import azure as azure_mod
    from app_config.databricks import AzureServicePrincipal

    monkeypatch.setattr(
        azure_mod,
        "databricks_service_principal",
        lambda scope, keys: AzureServicePrincipal("t-9", "c-9", "super-secret"),
    )
    result = sharepoint_check.check_secret_scope(_configured())

    assert result.status == PASS
    assert result.detail["tenant id"] == "t-9"
    assert result.detail["client id"] == "c-9"
    assert "super-secret" not in json.dumps(result.detail)


def test_secrets_failure_names_the_subprocess_pitfall(monkeypatch):
    from app_config import azure as azure_mod
    from app_config.databricks import DatabricksError

    def _boom(scope, keys):
        raise DatabricksError("could not read secret 'saact-hsv-tenantid'")

    monkeypatch.setattr(azure_mod, "databricks_service_principal", _boom)
    result = sharepoint_check.check_secret_scope(_configured())

    assert result.status == FAIL
    assert result.fix is not None
    assert "DATABRICKS_TOKEN" in result.fix
    assert "!python" in result.fix


def test_a_failed_secrets_check_skips_the_graph_stages(monkeypatch):
    """The failure that used to surface as an Azure CLI error three layers on."""
    from app_config import azure as azure_mod
    from app_config.databricks import DatabricksError

    def _boom(scope, keys):
        raise DatabricksError("no workspace credential")

    monkeypatch.setattr(azure_mod, "databricks_service_principal", _boom)
    client = _StubClient(_configured())
    results = _by_name(sharepoint_check.run_checks(client=client))

    assert results["identity"].status == PASS  # a scope IS configured
    assert results["secrets"].status == FAIL  # but it cannot be read
    assert results["token"].status == SKIP
    assert results["site"].status == SKIP
    assert results["lists"].status == SKIP


# ---------------------------------------------------------------------------
# The token stage — the roles claim is the 403 diagnosis
# ---------------------------------------------------------------------------


def test_token_reports_the_granted_roles():
    client = _StubClient(_configured())
    result = sharepoint_check.check_token(client)

    assert result.status == PASS
    assert result.detail["granted roles"] == "Sites.ReadWrite.All"
    assert result.detail["tenant (tid)"] == "tenant-1"
    assert result.detail["app name"] == "SAS Parser"


def test_token_without_the_required_role_warns_and_predicts_the_403():
    claims = {**_GRANTED, "roles": ["Sites.Read.All"]}
    client = _StubClient(_configured(), token=_jwt(claims))
    result = sharepoint_check.check_token(client)

    assert result.status == WARN
    assert result.fix is not None
    assert "Sites.Read.All" in result.fix
    assert "403" in result.fix


def test_token_with_no_roles_at_all_warns_about_admin_consent():
    claims = {k: v for k, v in _GRANTED.items() if k != "roles"}
    client = _StubClient(_configured(), token=_jwt(claims))
    result = sharepoint_check.check_token(client)

    assert result.status == WARN
    assert result.fix is not None
    assert "admin consent" in result.fix


def test_token_that_is_opaque_still_passes():
    client = _StubClient(_configured(), token="not-a-jwt")
    result = sharepoint_check.check_token(client)

    assert result.status == PASS
    assert "opaque" in result.summary


def test_token_failure_points_at_the_usual_causes():
    client = _StubClient(_configured(), fail={"token": "AADSTS7000215: bad secret"})
    result = sharepoint_check.check_token(client)

    assert result.status == FAIL
    assert "AADSTS7000215" in result.detail["error"]
    assert result.fix is not None
    assert "AZURE_VERIFY" in result.fix


@pytest.mark.parametrize("token", ["", "a.b", "a.b.c.d", "a.!!!.c", "a.e30.c"])
def test_decode_claims_never_raises_on_a_malformed_token(token):
    assert isinstance(sharepoint_check.decode_claims(token), dict)


def test_decode_claims_round_trips_a_payload():
    assert sharepoint_check.decode_claims(_jwt({"aud": "x"})) == {"aud": "x"}


# ---------------------------------------------------------------------------
# The library stages
# ---------------------------------------------------------------------------


def test_base_path_lists_the_application_folders():
    client = _StubClient(
        _configured(file_server_base_path="Apps"),
        entries=[
            {"name": "AppA", "is_folder": True},
            {"name": "AppB", "is_folder": True},
            {"name": "notes.txt", "is_folder": False},
        ],
    )
    result = sharepoint_check.check_base_path(client)

    assert result.status == PASS
    assert result.detail["applications"] == "AppA, AppB"
    assert "3 (2 folder(s))" == result.detail["children"]


def test_base_path_with_no_folders_warns():
    client = _StubClient(_configured(), entries=[{"name": "a.txt", "is_folder": False}])
    result = sharepoint_check.check_base_path(client)

    assert result.status == WARN
    assert result.fix is not None
    assert "scripts_original" in result.fix


def test_base_path_failure_explains_the_document_library_prefix():
    client = _StubClient(_configured(), fail={"base": "HTTP 404 itemNotFound"})
    result = sharepoint_check.check_base_path(client)

    assert result.status == FAIL
    assert result.fix is not None
    assert "Shared Documents/" in result.fix


def test_lists_report_the_internal_column_names():
    config = _configured(list_id_xref="L-XREF")
    client = _StubClient(
        config,
        rows={
            "L-REQ": [
                {"fields": {"Application Name": "A", "Macro File Name x003f ": "m"}}
            ],
            "L-XREF": [{"fields": {"OriginalValue": "a", "NewValue": "b"}}],
        },
    )
    results = _by_name(sharepoint_check.check_lists(client))

    assert results["list:requests"].status == PASS
    # The encoded internal name is exactly what conversion.requests addresses,
    # so seeing the real one is the point of reading a row at all.
    assert "Macro File Name x003f " in results["list:requests"].detail["columns"]
    assert results["list:xref"].status == PASS
    # Unconfigured lists are skipped, not failed: not every deployment has them.
    assert results["list:conversions"].status == SKIP
    assert results["list:complexity"].status == SKIP


def test_lists_read_only_one_row():
    client = _StubClient(_configured(), rows={"L-REQ": [{"fields": {"Title": "a"}}]})
    sharepoint_check.check_lists(client)

    assert client.list_item_calls == [("L-REQ", 1)]


def test_an_empty_list_still_passes():
    client = _StubClient(_configured())
    results = _by_name(sharepoint_check.check_lists(client))

    assert results["list:requests"].status == PASS
    assert "empty" in results["list:requests"].summary


def test_list_failure_explains_where_the_id_comes_from():
    client = _StubClient(_configured(), fail={"lists": "HTTP 404 itemNotFound"})
    results = _by_name(sharepoint_check.check_lists(client))

    assert results["list:requests"].status == FAIL
    assert results["list:requests"].fix is not None
    assert "Settings page URL" in results["list:requests"].fix


# ---------------------------------------------------------------------------
# run_checks: ordering, gating, and never writing
# ---------------------------------------------------------------------------


def test_run_checks_end_to_end_passes(_readable_scope):
    client = _StubClient(_configured(), rows={"L-REQ": [{"fields": {"Title": "a"}}]})
    results = sharepoint_check.run_checks(client=client)

    assert [r.name for r in results][:7] == [
        "config",
        "imports",
        "identity",
        "secrets",
        "token",
        "site",
        "base",
    ]
    assert all(r.status != FAIL for r in results)


def test_run_checks_never_writes(_readable_scope):
    """The stub has no write surface at all, so a write would be an AttributeError."""
    client = _StubClient(_configured())
    for name in ("write_file", "upload_file", "create_folder", "update_list_item"):
        assert not hasattr(client, name)

    results = sharepoint_check.run_checks(client=client)

    assert all(r.status != FAIL for r in results)


def test_a_failed_site_check_skips_the_rest_rather_than_passing_them(_readable_scope):
    client = _StubClient(_configured(), fail={"site": "HTTP 403 accessDenied"})
    results = _by_name(sharepoint_check.run_checks(client=client))

    assert results["site"].status == FAIL
    assert results["base"].status == SKIP
    assert results["lists"].status == SKIP
    assert "site" in results["base"].summary


def test_a_token_warning_does_not_gate_the_later_stages(_readable_scope):
    """A missing role is more informative once you see which calls it blocks."""
    claims = {**_GRANTED, "roles": ["Sites.Read.All"]}
    client = _StubClient(_configured(), token=_jwt(claims))
    results = _by_name(sharepoint_check.run_checks(client=client))

    assert results["token"].status == WARN
    assert results["site"].status == PASS
    assert results["base"].status == PASS


def test_offline_checks_the_config_and_stops(monkeypatch):
    monkeypatch.setenv("SHAREPOINT_SITE_ID", "SITE")
    monkeypatch.setenv("SHAREPOINT_LIST_ID_SAS_REQUESTS", "L-REQ")
    results = _by_name(sharepoint_check.run_checks(offline=True))

    assert results["config"].status == PASS
    assert results["identity"].status == SKIP
    assert results["token"].status == SKIP
    assert results["lists"].status == SKIP


def test_a_failing_identity_skips_the_network_stages():
    # No secret scope and no AZURE_* identity, but a site: config passes and
    # identity is the stage that stops the run.
    config = sharepoint.SharePointConfig(site_id="SITE", list_id_sas_requests="L")
    client = _StubClient(config)
    results = _by_name(sharepoint_check.run_checks(client=client))

    assert results["identity"].status == FAIL
    assert results["token"].status == SKIP
    assert results["site"].status == SKIP


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_shows_detail_for_failures_but_not_for_passes():
    results = [
        sharepoint_check.CheckResult("a", PASS, "fine", {"hidden": "x"}),
        sharepoint_check.CheckResult("b", FAIL, "broken", {"shown": "y"}, fix="do it"),
    ]
    report = sharepoint_check.render(results)

    assert "hidden" not in report
    assert "shown: y" in report
    assert "-> do it" in report
    assert "FAILED: 1 check(s) failed (b)" in report


def test_render_verbose_shows_everything():
    results = [sharepoint_check.CheckResult("a", PASS, "fine", {"hidden": "x"})]
    report = sharepoint_check.render(results, verbose=True)

    assert "hidden: x" in report
    assert "PASSED" in report


def test_render_redacts_a_secret_that_reached_the_detail():
    results = [
        sharepoint_check.CheckResult(
            "a", FAIL, "boom", {"header": "Authorization: Bearer abcdef1234567890"}
        )
    ]
    report = sharepoint_check.render(results)

    assert "abcdef1234567890" not in report
    assert "<redacted>" in report


def test_json_output_is_parseable_and_redacted():
    results = [
        sharepoint_check.CheckResult(
            "a", WARN, "hm", {"token": "Bearer abcdef1234567890"}, fix="f"
        )
    ]
    payload = json.loads(sharepoint_check.to_json(results))

    assert payload[0]["name"] == "a"
    assert payload[0]["status"] == WARN
    assert payload[0]["fix"] == "f"
    assert "abcdef1234567890" not in payload[0]["detail"]["token"]


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_main_offline_returns_nonzero_without_a_site(capsys):
    code = sharepoint_check.main(["--offline"])

    assert code == 1
    assert "no SharePoint site is configured" in capsys.readouterr().out


def test_main_offline_returns_zero_when_configured(monkeypatch, capsys):
    monkeypatch.setenv("SHAREPOINT_SITE_ID", "SITE")
    monkeypatch.setenv("SHAREPOINT_LIST_ID_SAS_REQUESTS", "L-REQ")
    code = sharepoint_check.main(["--offline"])

    assert code == 0
    assert "PASSED" in capsys.readouterr().out


def test_main_json_flag_emits_json(monkeypatch, capsys):
    monkeypatch.setenv("SHAREPOINT_SITE_ID", "SITE")
    code = sharepoint_check.main(["--offline", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert {r["name"] for r in payload} >= {"config", "imports"}


def test_main_writes_a_log_file(monkeypatch, tmp_path):
    log = tmp_path / "logs" / "preflight.log"
    monkeypatch.setenv("SHAREPOINT_SITE_ID", "SITE")
    sharepoint_check.main(["--offline", "--log-file", str(log)])

    assert log.is_file()
    assert "logging to" in log.read_text(encoding="utf-8")
