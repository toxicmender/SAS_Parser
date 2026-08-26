"""Phase 9 v2 settings, authentication, credentials, and observability contracts."""

from __future__ import annotations

import io
import json
import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

import sas_migrate.adapters.auth.azure as azure_adapter
import sas_migrate.adapters.credentials.databricks as databricks_adapter
import sas_migrate.adapters.credentials.vault as vault_adapter
import sas_migrate.observability.logging as observability_logging
from sas_migrate.adapters.auth import MsalAccessTokenProvider
from sas_migrate.adapters.credentials import (
    ChainedCredentialProvider,
    CredentialProviderUnavailable,
    DatabricksSecretCredentialProvider,
    DatabricksSecretReference,
    EnvironmentCredentialProvider,
    VaultCredentialProvider,
    VaultSecretReference,
)
from sas_migrate.application.ports import AccessToken, CredentialValue
from sas_migrate.config import (
    AzureSettings,
    ConfigurationError,
    ObservabilitySettings,
    SharePointSettings,
    VaultSettings,
    load_settings,
    load_settings_file,
)
from sas_migrate.observability import (
    REDACTED,
    RedactingFilter,
    configure_observability,
    redact_mapping,
    redact_text,
)


def test_settings_are_versioned_immutable_and_secret_free() -> None:
    settings = load_settings(document={}, environ={})

    assert settings.schema_version == 2
    assert settings.azure.authority_host == "https://login.microsoftonline.com"
    assert settings.sharepoint.scopes == ("https://graph.microsoft.com/.default",)
    payload = settings.model_dump_json()
    assert '"client_secret":' not in payload
    assert '"access_token":' not in payload
    with pytest.raises(ValidationError, match="frozen"):
        settings.azure.timeout = 1  # type: ignore[misc]


def test_environment_overrides_document_and_inherits_safe_shared_settings() -> None:
    settings = load_settings(
        document={
            "azure": {"timeout": 5, "verify": True},
            "vault": {"verify": False},
            "databricks": {"secret_scope": "shared-scope", "schema": "bronze"},
            "sharepoint": {
                "site_hostname": "contoso.sharepoint.com",
                "site_path": "sites/Engineering/",
                "file_server_base_path": "Shared Documents/Apps/",
                "list_id_sas_requests": "requests",
            },
        },
        environ={
            "AZURE_TIMEOUT": "12.5",
            "AZURE_SCOPES": "scope-a, scope-b",
            "SHAREPOINT_DRIVE_ID": "drive-1",
            "SAS_MIGRATE_TRACE_HTTP": "yes",
        },
    )

    assert settings.azure.timeout == 12.5
    assert settings.azure.scopes == ("scope-a", "scope-b")
    assert settings.azure.verify is True
    assert settings.databricks.schema_name == "bronze"
    assert settings.sharepoint.site_path == "/sites/Engineering"
    assert settings.sharepoint.file_server_base_path == "Apps"
    assert settings.sharepoint.secret_scope == "shared-scope"
    assert settings.sharepoint.resolved_site_id == (
        "contoso.sharepoint.com:/sites/Engineering"
    )
    assert settings.sharepoint.drive_path("One", "/Two/") == "Apps/One/Two"
    assert settings.sharepoint.configuration_issues() == ()
    assert settings.observability.trace_http is True


def test_vault_tls_environment_precedence_and_azure_inheritance() -> None:
    with_ca = load_settings(
        document={"vault": {"verify": False}},
        environ={"VAULT_CACERT": "/certs/company.pem"},
    )
    assert with_ca.vault.verify == "/certs/company.pem"
    assert with_ca.azure.verify == "/certs/company.pem"

    skipped = load_settings(document={}, environ={"VAULT_SKIP_VERIFY": "true"})
    assert skipped.vault.verify is False
    assert skipped.azure.verify is False


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"azure": {"client_secret": "forbidden"}}, "credential"),
        ({"vault": []}, "must be a JSON object"),
        ({"azure": {"unknown": "value"}}, "extra_forbidden"),
    ],
)
def test_settings_reject_secret_wrong_shape_and_unknown_keys(
    document: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        load_settings(document=document, environ={})


def test_settings_reject_invalid_environment_values() -> None:
    with pytest.raises(ConfigurationError, match="SAS_MIGRATE_DEBUG"):
        load_settings(document={}, environ={"SAS_MIGRATE_DEBUG": "sometimes"})
    with pytest.raises(ConfigurationError, match="VAULT_SKIP_VERIFY"):
        load_settings(document={}, environ={"VAULT_SKIP_VERIFY": "perhaps"})
    with pytest.raises(ConfigurationError, match="AZURE_TIMEOUT"):
        load_settings(document={}, environ={"AZURE_TIMEOUT": "slow"})
    assert load_settings(
        document={}, environ={"SAS_MIGRATE_DEBUG": "off"}
    ).observability.debug is False


def test_settings_file_accepts_utf8_bom_and_reports_bad_documents(tmp_path: Path) -> None:
    valid = tmp_path / "settings.json"
    valid.write_text(
        "\ufeff" + json.dumps({"sharepoint": {"drive_id": "drive"}}),
        encoding="utf-8",
    )
    assert load_settings_file(valid, environ={}).sharepoint.drive_id == "drive"
    assert load_settings(
        environ={"SAS_PARSER_CONFIG": str(valid)}
    ).sharepoint.drive_id == "drive"

    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="must be a JSON object"):
        load_settings_file(invalid, environ={})
    with pytest.raises(ConfigurationError, match="cannot load settings"):
        load_settings_file(tmp_path / "missing.json", environ={})


def test_sharepoint_settings_name_actionable_missing_configuration() -> None:
    settings = SharePointSettings()
    assert settings.configuration_issues() == (
        "set site_id or site_hostname + site_path (or provide drive_id)",
        "set list_id_sas_requests",
    )
    assert SharePointSettings(file_server_base_path="Shared Documents").drive_path(
        "Reports"
    ) == "Reports"


@pytest.mark.anyio
async def test_environment_and_chain_providers_preserve_order_and_mask_values() -> None:
    empty = EnvironmentCredentialProvider(
        {"gateway": "MISSING"},
        environ={},
    )
    environment = EnvironmentCredentialProvider(
        {"gateway": "GATEWAY_TOKEN"},
        environ={"GATEWAY_TOKEN": "very-secret-token"},
    )
    value = await ChainedCredentialProvider((empty, environment)).get("gateway")

    assert value is not None
    assert value.source == "environment:GATEWAY_TOKEN"
    assert value.value.get_secret_value() == "very-secret-token"
    assert "very-secret-token" not in repr(value)
    assert "very-secret-token" not in value.model_dump_json()
    assert await environment.get("unknown") is None
    assert await ChainedCredentialProvider((empty,)).get("gateway") is None


class _SecretsAccessor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get(self, *, scope: str, key: str) -> str:
        self.calls.append((scope, key))
        return "scope-secret"


@pytest.mark.anyio
async def test_databricks_provider_uses_only_explicit_secret_references() -> None:
    accessor = _SecretsAccessor()
    provider = DatabricksSecretCredentialProvider(
        {
            "azure_client_secret": DatabricksSecretReference(
                scope="apps",
                key="client-secret",
            )
        },
        accessor=accessor,
    )

    assert await provider.get("missing") is None
    value = await provider.get("azure_client_secret")
    assert value is not None
    assert value.value.get_secret_value() == "scope-secret"
    assert value.source == "databricks:apps/client-secret"
    assert accessor.calls == [("apps", "client-secret")]

    class _EmptyAccessor:
        def get(self, *, scope: str, key: str) -> str:
            return ""

    empty = DatabricksSecretCredentialProvider(
        {"name": DatabricksSecretReference(scope="scope", key="key")},
        accessor=_EmptyAccessor(),
    )
    assert await empty.get("name") is None


@pytest.mark.anyio
async def test_databricks_provider_resolves_runtime_accessor_once(monkeypatch) -> None:
    accessor = _SecretsAccessor()
    calls = 0

    def runtime_accessor() -> _SecretsAccessor:
        nonlocal calls
        calls += 1
        return accessor

    monkeypatch.setattr(databricks_adapter, "_runtime_accessor", runtime_accessor)
    provider = DatabricksSecretCredentialProvider(
        {"name": DatabricksSecretReference(scope="scope", key="key")}
    )
    assert await provider.get("name") is not None
    assert await provider.get("name") is not None
    assert calls == 1


class _MsalApplication:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.scopes: list[list[str]] = []

    def acquire_token_for_client(self, scopes: list[str]) -> dict[str, Any]:
        self.scopes.append(scopes)
        return self.response


@pytest.mark.anyio
async def test_msal_adapter_resolves_secret_lazily_and_returns_masked_token() -> None:
    credentials = EnvironmentCredentialProvider(
        {"azure_client_secret": "AZURE_CLIENT_SECRET"},
        environ={"AZURE_CLIENT_SECRET": "client-secret-value"},
    )
    application = _MsalApplication(
        {"access_token": "issued-access-token", "expires_in": "120"}
    )
    secrets_seen: list[str] = []

    def factory(settings: AzureSettings, secret: str) -> _MsalApplication:
        assert settings.client_id == "client"
        secrets_seen.append(secret)
        return application

    provider = MsalAccessTokenProvider(
        AzureSettings(
            tenant_id="tenant",
            client_id="client",
            scopes=("scope/.default",),
        ),
        credentials,
        application_factory=factory,
    )
    token = await provider.get_token()

    assert token.value.get_secret_value() == "issued-access-token"
    assert "issued-access-token" not in token.model_dump_json()
    assert token.expires_at_epoch is not None
    assert application.scopes == [["scope/.default"]]
    await provider.get_token(("other",))
    assert secrets_seen == ["client-secret-value"]


@pytest.mark.anyio
async def test_msal_adapter_reports_missing_credentials_and_redacts_errors() -> None:
    empty = EnvironmentCredentialProvider(
        {"azure_client_secret": "AZURE_CLIENT_SECRET"},
        environ={},
    )
    provider = MsalAccessTokenProvider(
        AzureSettings(scopes=("scope",)),
        empty,
        application_factory=lambda _settings, _secret: _MsalApplication({}),
    )
    with pytest.raises(CredentialProviderUnavailable, match="not configured"):
        await provider.get_token()

    failing = MsalAccessTokenProvider(
        AzureSettings(scopes=("scope",)),
        EnvironmentCredentialProvider(
            {"azure_client_secret": "SECRET"},
            environ={"SECRET": "bootstrap-secret"},
        ),
        application_factory=lambda _settings, _secret: _MsalApplication(
            {
                "error": "unauthorized_client",
                "error_description": "client_secret=leaked-value",
            }
        ),
    )
    with pytest.raises(CredentialProviderUnavailable) as captured:
        await failing.get_token()
    assert "leaked-value" not in str(captured.value)
    assert REDACTED in str(captured.value)

    with pytest.raises(CredentialProviderUnavailable, match="at least one"):
        await MsalAccessTokenProvider(
            AzureSettings(),
            empty,
            application_factory=lambda _settings, _secret: _MsalApplication({}),
        ).get_token()

    with pytest.raises(CredentialProviderUnavailable, match="client_credentials"):
        await MsalAccessTokenProvider(
            AzureSettings(flow="device_code", scopes=("scope",)),
            empty,
            application_factory=lambda _settings, _secret: _MsalApplication({}),
        ).get_token()


def test_default_msal_factory_is_lazy_and_validates_identity(monkeypatch) -> None:
    built: list[tuple[str, str, str]] = []

    def factory(client_id: str, *, client_credential: str, authority: str) -> object:
        built.append((client_id, client_credential, authority))
        return object()

    monkeypatch.setitem(
        sys.modules,
        "msal",
        SimpleNamespace(ConfidentialClientApplication=factory),
    )
    application = azure_adapter._msal_application(
        AzureSettings(tenant_id="tenant", client_id="client"),
        "secret",
    )
    assert application is not None
    assert built == [
        ("client", "secret", "https://login.microsoftonline.com/tenant")
    ]
    with pytest.raises(CredentialProviderUnavailable, match="client_id and tenant_id"):
        azure_adapter._msal_application(AzureSettings(), "secret")


class _VaultV2:
    def __init__(self, data: dict[str, Any] | Exception) -> None:
        self.data = data
        self.calls: list[tuple[str, str]] = []

    def read_secret_version(
        self,
        *,
        path: str,
        mount_point: str,
        raise_on_deleted_version: bool,
    ) -> dict[str, Any]:
        assert raise_on_deleted_version is True
        self.calls.append((mount_point, path))
        if isinstance(self.data, Exception):
            raise self.data
        return {"data": {"data": self.data}}


class _VaultV1:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def read_secret(self, *, path: str, mount_point: str) -> dict[str, Any]:
        return {"data": self.data}


class _VaultClient:
    def __init__(self, data: dict[str, Any] | Exception) -> None:
        self.token: str | None = None
        self.approle_calls: list[dict[str, Any]] = []
        self.jwt_calls: list[dict[str, Any]] = []
        self.secrets = SimpleNamespace(
            kv=SimpleNamespace(v2=_VaultV2(data), v1=_VaultV1({"legacy": "v1"}))
        )
        self.auth = SimpleNamespace(
            approle=SimpleNamespace(login=self._approle),
            jwt=SimpleNamespace(jwt_login=self._jwt),
        )

    def _approle(self, **values: Any) -> None:
        self.approle_calls.append(values)

    def _jwt(self, **values: Any) -> None:
        self.jwt_calls.append(values)


class _StaticTokenProvider:
    async def get_token(self, scopes: tuple[str, ...] = ()) -> AccessToken:
        return AccessToken(value=SecretStr("azure-jwt"), source=str(scopes))


@pytest.mark.anyio
async def test_vault_provider_supports_token_and_kv2_without_leaking_bootstrap() -> None:
    client = _VaultClient({"api_key": "vault-secret"})
    provider = VaultCredentialProvider(
        VaultSettings(address="https://vault.example"),
        {"gateway": VaultSecretReference(path="app/gateway", key="api_key")},
        bootstrap=EnvironmentCredentialProvider(
            {"vault_token": "VAULT_TOKEN"},
            environ={"VAULT_TOKEN": "root-bootstrap"},
        ),
        client_factory=lambda _settings: client,
    )

    assert await provider.get("unknown") is None
    value = await provider.get("gateway")
    assert value is not None
    assert client.token == "root-bootstrap"
    assert value.value.get_secret_value() == "vault-secret"
    assert value.source == "vault:secret/app/gateway#api_key"
    assert client.secrets.kv.v2.calls == [("secret", "app/gateway")]


@pytest.mark.anyio
async def test_vault_provider_supports_approle_kv1_and_azure_jwt() -> None:
    approle_client = _VaultClient({})
    approle = VaultCredentialProvider(
        VaultSettings(address="https://vault.example", kv_version=1),
        {"legacy": VaultSecretReference(path="old", key="legacy")},
        bootstrap=EnvironmentCredentialProvider(
            {
                "vault_role_id": "VAULT_ROLE_ID",
                "vault_secret_id": "VAULT_SECRET_ID",
            },
            environ={"VAULT_ROLE_ID": "role", "VAULT_SECRET_ID": "secret"},
        ),
        client_factory=lambda _settings: approle_client,
    )
    value = await approle.get("legacy")
    assert value is not None
    assert value.value.get_secret_value() == "v1"
    assert approle_client.approle_calls == [
        {"role_id": "role", "secret_id": "secret", "use_token": True}
    ]

    jwt_client = _VaultClient({"key": "jwt-secret"})
    jwt = VaultCredentialProvider(
        VaultSettings(address="https://vault.example", app_name="sas-parser"),
        {"name": VaultSecretReference(path="path", key="key")},
        bootstrap=EnvironmentCredentialProvider({}, environ={}),
        azure_tokens=_StaticTokenProvider(),
        client_factory=lambda _settings: jwt_client,
    )
    assert (await jwt.get("name")) is not None
    assert jwt_client.jwt_calls == [
        {
            "role": "sas-parser",
            "jwt": "azure-jwt",
            "path": "jwt",
            "use_token": True,
        }
    ]


@pytest.mark.anyio
async def test_vault_provider_normalizes_missing_non_text_and_sdk_errors() -> None:
    bootstrap = EnvironmentCredentialProvider(
        {"vault_token": "TOKEN"},
        environ={"TOKEN": "bootstrap"},
    )
    reference = {"name": VaultSecretReference(path="path", key="key")}

    missing = VaultCredentialProvider(
        VaultSettings(address="https://vault.example"),
        reference,
        bootstrap=bootstrap,
        client_factory=lambda _settings: _VaultClient({}),
    )
    assert await missing.get("name") is None

    non_text = VaultCredentialProvider(
        VaultSettings(address="https://vault.example"),
        reference,
        bootstrap=bootstrap,
        client_factory=lambda _settings: _VaultClient({"key": 7}),
    )
    with pytest.raises(CredentialProviderUnavailable, match="must be text"):
        await non_text.get("name")

    failed = VaultCredentialProvider(
        VaultSettings(address="https://vault.example"),
        reference,
        bootstrap=bootstrap,
        client_factory=lambda _settings: _VaultClient(
            RuntimeError("token=secret-in-error")
        ),
    )
    with pytest.raises(CredentialProviderUnavailable) as captured:
        await failed.get("name")
    assert "secret-in-error" not in str(captured.value)
    assert REDACTED in str(captured.value)


@pytest.mark.anyio
async def test_vault_provider_requires_one_authentication_method() -> None:
    provider = VaultCredentialProvider(
        VaultSettings(address="https://vault.example"),
        {"name": VaultSecretReference(path="path", key="key")},
        bootstrap=EnvironmentCredentialProvider({}, environ={}),
        client_factory=lambda _settings: _VaultClient({"key": "value"}),
    )
    with pytest.raises(CredentialProviderUnavailable, match="Vault needs"):
        await provider.get("name")


def test_default_vault_factory_is_lazy_and_requires_address(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def client(**values: Any) -> object:
        calls.append(values)
        return object()

    monkeypatch.setitem(sys.modules, "hvac", SimpleNamespace(Client=client))
    settings = VaultSettings(address="https://vault.example", timeout=1.5)
    assert vault_adapter._hvac_client(settings) is not None
    assert calls[0]["url"] == "https://vault.example"
    assert calls[0]["timeout"] == 1.5
    with pytest.raises(CredentialProviderUnavailable, match="address"):
        vault_adapter._hvac_client(VaultSettings())


def test_redaction_masks_text_nested_values_and_secret_types() -> None:
    assert redact_text("Authorization: Bearer abcdefghijklmnop") == (
        f"Authorization: Bearer {REDACTED}"
    )
    redacted = redact_mapping(
        {
            "client_secret": "value",
            "nested": {"url": "https://example.test?sig=signature-value"},
            "items": [SecretStr("hidden"), "password=also-hidden"],
            "safe": "visible",
        }
    )
    assert redacted == {
        "client_secret": REDACTED,
        "nested": {"url": f"https://example.test?sig={REDACTED}"},
        "items": [REDACTED, f"password={REDACTED}"],
        "safe": "visible",
    }
    assert redact_mapping({"count": 3}) == {"count": 3}


def test_redacting_filter_masks_message_and_traceback() -> None:
    try:
        raise RuntimeError("access_token=traceback-secret")
    except RuntimeError:
        record = logging.LogRecord(
            "test",
            logging.ERROR,
            __file__,
            1,
            "Bearer %s",
            ("message-secret-value",),
            sys.exc_info(),
        )
    assert RedactingFilter().filter(record) is True
    assert "message-secret-value" not in record.getMessage()
    assert record.exc_text is not None
    assert "traceback-secret" not in record.exc_text

    broken = logging.LogRecord(
        "test", logging.INFO, __file__, 1, "%d", ("not-a-number",), None
    )
    assert RedactingFilter().filter(broken) is True


def test_observability_configures_redacted_console_file_and_transport_levels(
    tmp_path: Path,
) -> None:
    stream = io.StringIO()
    log_path = tmp_path / "logs" / "run.log"
    original_sys_hook = sys.excepthook
    original_thread_hook = threading.excepthook
    handlers = configure_observability(
        ObservabilitySettings(
            debug=True,
            trace_http=True,
            log_file=log_path,
            capture_crashes=False,
        ),
        stream=stream,
    )
    try:
        logging.getLogger("sas_migrate.test").debug(
            "client_secret=%s", "console-secret"
        )
        for handler in handlers:
            handler.flush()
        assert "console-secret" not in stream.getvalue()
        assert REDACTED in stream.getvalue()
        assert "console-secret" not in log_path.read_text("utf-8")
        assert logging.getLogger("httpx").level == logging.DEBUG
        assert sys.excepthook is original_sys_hook
        assert threading.excepthook is original_thread_hook
    finally:
        for handler in handlers:
            handler.close()
        logging.getLogger().handlers.clear()


def test_configure_observability_can_install_crash_handlers() -> None:
    original_sys_hook = sys.excepthook
    original_thread_hook = threading.excepthook
    handlers = configure_observability(
        ObservabilitySettings(capture_crashes=True),
        stream=io.StringIO(),
    )
    try:
        assert sys.excepthook is not original_sys_hook
        assert threading.excepthook is not original_thread_hook
        assert logging.getLogger("httpx").level == logging.INFO
    finally:
        sys.excepthook = original_sys_hook
        threading.excepthook = original_thread_hook
        for handler in handlers:
            handler.close()
        logging.getLogger().handlers.clear()


def test_crash_hook_logs_failures_and_defers_keyboard_interrupt(monkeypatch) -> None:
    critical: list[tuple[str, Any]] = []
    keyboard: list[type[BaseException]] = []
    monkeypatch.setattr(
        observability_logging.LOGGER,
        "critical",
        lambda message, *, exc_info: critical.append((message, exc_info)),
    )
    monkeypatch.setattr(
        sys,
        "__excepthook__",
        lambda exc_type, _exc, _traceback: keyboard.append(exc_type),
    )
    error = RuntimeError("failed")
    observability_logging._log_crash(RuntimeError, error, None)
    assert critical[0][0] == "unhandled exception; the run did not finish"
    observability_logging._log_crash(KeyboardInterrupt, KeyboardInterrupt(), None)
    assert keyboard == [KeyboardInterrupt]


def test_credential_contract_masks_secret_json() -> None:
    value = CredentialValue(
        name="one",
        value=SecretStr("never-serialize"),
        source="test",
    )
    assert "never-serialize" not in value.model_dump_json()
