"""Versioned, immutable infrastructure settings with no credential values."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from sas_migrate.core.models import ContractModel, VersionedContract

DEFAULT_AUTHORITY_HOST = "https://login.microsoftonline.com"
GRAPH_DEFAULT_SCOPE = "https://graph.microsoft.com/.default"
DEFAULT_SHAREPOINT_CREDENTIAL_KEYS = (
    "saact-hsv-tenantid",
    "saact-hsv-appid",
    "saact-hsv-secret",
)


def _clean(value: object) -> object:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    return stripped or None


class AzureSettings(ContractModel):
    tenant_id: str | None = None
    client_id: str | None = None
    authority_host: str = DEFAULT_AUTHORITY_HOST
    scopes: tuple[str, ...] = Field(default_factory=tuple)
    flow: Literal["client_credentials", "device_code"] = "client_credentials"
    timeout: float = Field(default=30.0, gt=0)
    certificate_path: Path | None = None
    certificate_thumbprint: str | None = None
    verify: bool | str = True
    proxies: dict[str, str] = Field(default_factory=dict)

    _strip_strings = field_validator(
        "tenant_id",
        "client_id",
        "authority_host",
        "certificate_thumbprint",
        mode="before",
    )(_clean)


class VaultSettings(ContractModel):
    address: str | None = None
    namespace: str | None = None
    app_name: str | None = None
    mount_point: str = "secret"
    kv_version: Literal[1, 2] = 2
    timeout: float = Field(default=30.0, gt=0)
    verify: bool | str = True
    auth_path: str = "jwt"
    azure_scopes: tuple[str, ...] = ("https://management.azure.com//.default",)
    ai_gateway_path: str | None = None
    ai_gateway_key: str | None = None

    _strip_strings = field_validator(
        "address",
        "namespace",
        "app_name",
        "mount_point",
        "auth_path",
        "ai_gateway_path",
        "ai_gateway_key",
        mode="before",
    )(_clean)


class DatabricksSettings(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    host: str | None = None
    http_path: str | None = None
    warehouse_id: str | None = None
    cluster_id: str | None = None
    catalog: str | None = None
    schema_name: str | None = Field(default=None, alias="schema")
    timeout: float = Field(default=30.0, gt=0)
    azure_tenant_id: str | None = None
    azure_client_id: str | None = None
    azure_workspace_resource_id: str | None = None
    secret_scope: str | None = None

    _strip_strings = field_validator(
        "host",
        "http_path",
        "warehouse_id",
        "cluster_id",
        "catalog",
        "schema_name",
        "azure_tenant_id",
        "azure_client_id",
        "azure_workspace_resource_id",
        "secret_scope",
        mode="before",
    )(_clean)


class SharePointSettings(ContractModel):
    site_hostname: str | None = None
    site_path: str | None = None
    site_id: str | None = None
    drive_id: str | None = None
    scopes: tuple[str, ...] = (GRAPH_DEFAULT_SCOPE,)
    timeout: float = Field(default=60.0, gt=0)
    file_server_base_path: str = ""
    secret_scope: str | None = None
    tenant_id_key: str = DEFAULT_SHAREPOINT_CREDENTIAL_KEYS[0]
    client_id_key: str = DEFAULT_SHAREPOINT_CREDENTIAL_KEYS[1]
    client_secret_key: str = DEFAULT_SHAREPOINT_CREDENTIAL_KEYS[2]
    list_id_sas_requests: str | None = None
    list_id_sas_conversions: str | None = None
    list_id_xref: str | None = None
    list_id_sas_complexity: str | None = None

    _strip_strings = field_validator(
        "site_hostname",
        "site_id",
        "drive_id",
        "secret_scope",
        "tenant_id_key",
        "client_id_key",
        "client_secret_key",
        "list_id_sas_requests",
        "list_id_sas_conversions",
        "list_id_xref",
        "list_id_sas_complexity",
        mode="before",
    )(_clean)

    @field_validator("site_path", mode="before")
    @classmethod
    def normalize_site_path(cls, value: object) -> object:
        cleaned = _clean(value)
        if not isinstance(cleaned, str):
            return cleaned
        return f"/{cleaned.strip('/')}"

    @field_validator("file_server_base_path", mode="before")
    @classmethod
    def normalize_drive_path(cls, value: object) -> str:
        if not isinstance(value, str):
            return ""
        path = value.strip().strip("/")
        prefix = "shared documents"
        if path.casefold() == prefix:
            return ""
        if path.casefold().startswith(f"{prefix}/"):
            return path[len(prefix) + 1 :].strip("/")
        return path

    @property
    def resolved_site_id(self) -> str | None:
        if self.site_id:
            return self.site_id
        if self.site_hostname and self.site_path:
            return f"{self.site_hostname}:{self.site_path}"
        return None

    def drive_path(self, *parts: str) -> str:
        values = [self.file_server_base_path, *parts]
        return "/".join(value.strip().strip("/") for value in values if value.strip())

    def configuration_issues(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.resolved_site_id is None and self.drive_id is None:
            issues.append(
                "set site_id or site_hostname + site_path (or provide drive_id)"
            )
        if self.list_id_sas_requests is None:
            issues.append("set list_id_sas_requests")
        return tuple(issues)


class ObservabilitySettings(ContractModel):
    debug: bool = False
    log_file: Path | None = None
    trace_http: bool = False
    capture_crashes: bool = True


class InfrastructureSettings(VersionedContract):
    azure: AzureSettings = Field(default_factory=AzureSettings)
    vault: VaultSettings = Field(default_factory=VaultSettings)
    databricks: DatabricksSettings = Field(default_factory=DatabricksSettings)
    sharepoint: SharePointSettings = Field(default_factory=SharePointSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)


__all__ = [
    "DEFAULT_AUTHORITY_HOST",
    "DEFAULT_SHAREPOINT_CREDENTIAL_KEYS",
    "GRAPH_DEFAULT_SCOPE",
    "AzureSettings",
    "DatabricksSettings",
    "InfrastructureSettings",
    "ObservabilitySettings",
    "SharePointSettings",
    "VaultSettings",
]
