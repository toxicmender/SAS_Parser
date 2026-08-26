"""Resolve v2 infrastructure settings without ever loading a credential value."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import InfrastructureSettings

CONFIG_PATH_ENV = "SAS_PARSER_CONFIG"
_SECTIONS = ("azure", "vault", "databricks", "sharepoint", "observability")
_FORBIDDEN_FILE_FIELDS = {
    ("azure", "client_secret"),
    ("databricks", "token"),
    ("databricks", "azure_client_secret"),
    ("vault", "token"),
    ("vault", "role_id"),
    ("vault", "secret_id"),
    ("sharepoint", "client_secret"),
}


class ConfigurationError(ValueError):
    pass


def _text(value: str) -> str:
    return value.strip()


def _number(value: str) -> float:
    return float(value.strip())


def _integer(value: str) -> int:
    return int(value.strip())


def _boolean(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


def _verify(value: str) -> bool | str:
    try:
        return _boolean(value)
    except ValueError:
        path = value.strip()
        if not path:
            raise ValueError("TLS verification value cannot be blank") from None
        return path


def _scopes(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.replace(",", " ").split() if part)


Parser = Callable[[str], Any]
EnvSpec = tuple[str, Parser]

_ENV: dict[str, dict[str, EnvSpec]] = {
    "azure": {
        "tenant_id": ("AZURE_TENANT_ID", _text),
        "client_id": ("AZURE_CLIENT_ID", _text),
        "authority_host": ("AZURE_AUTHORITY_HOST", _text),
        "scopes": ("AZURE_SCOPES", _scopes),
        "flow": ("AZURE_FLOW", _text),
        "timeout": ("AZURE_TIMEOUT", _number),
        "certificate_path": ("AZURE_CLIENT_CERTIFICATE_PATH", _text),
        "certificate_thumbprint": (
            "AZURE_CLIENT_CERTIFICATE_THUMBPRINT",
            _text,
        ),
        "verify": ("AZURE_VERIFY", _verify),
    },
    "vault": {
        "address": ("VAULT_ADDR", _text),
        "namespace": ("VAULT_NAMESPACE", _text),
        "app_name": ("VAULT_APP_NAME", _text),
        "mount_point": ("VAULT_MOUNT_POINT", _text),
        "kv_version": ("VAULT_KV_VERSION", _integer),
        "timeout": ("VAULT_TIMEOUT", _number),
        "auth_path": ("VAULT_AUTH_PATH", _text),
        "azure_scopes": ("VAULT_AZURE_SCOPES", _scopes),
        "ai_gateway_path": ("VAULT_AI_GATEWAY_PATH", _text),
        "ai_gateway_key": ("VAULT_AI_GATEWAY_KEY", _text),
    },
    "databricks": {
        "host": ("DATABRICKS_HOST", _text),
        "http_path": ("DATABRICKS_HTTP_PATH", _text),
        "warehouse_id": ("DATABRICKS_WAREHOUSE_ID", _text),
        "cluster_id": ("DATABRICKS_CLUSTER_ID", _text),
        "catalog": ("DATABRICKS_CATALOG", _text),
        "schema": ("DATABRICKS_SCHEMA", _text),
        "timeout": ("DATABRICKS_TIMEOUT", _number),
        "azure_tenant_id": ("ARM_TENANT_ID", _text),
        "azure_client_id": ("ARM_CLIENT_ID", _text),
        "azure_workspace_resource_id": (
            "DATABRICKS_AZURE_RESOURCE_ID",
            _text,
        ),
        "secret_scope": ("DATABRICKS_SECRET_SCOPE", _text),
    },
    "sharepoint": {
        "site_hostname": ("SHAREPOINT_SITE_HOSTNAME", _text),
        "site_path": ("SHAREPOINT_SITE_PATH", _text),
        "site_id": ("SHAREPOINT_SITE_ID", _text),
        "drive_id": ("SHAREPOINT_DRIVE_ID", _text),
        "scopes": ("SHAREPOINT_SCOPES", _scopes),
        "timeout": ("SHAREPOINT_TIMEOUT", _number),
        "file_server_base_path": ("SHAREPOINT_FILE_SERVER_BASE_PATH", _text),
        "secret_scope": ("SHAREPOINT_SECRET_SCOPE", _text),
        "tenant_id_key": ("SHAREPOINT_TENANT_ID_KEY", _text),
        "client_id_key": ("SHAREPOINT_CLIENT_ID_KEY", _text),
        "client_secret_key": ("SHAREPOINT_CLIENT_SECRET_KEY", _text),
        "list_id_sas_requests": ("SHAREPOINT_LIST_ID_SAS_REQUESTS", _text),
        "list_id_sas_conversions": (
            "SHAREPOINT_LIST_ID_SAS_CONVERSIONS",
            _text,
        ),
        "list_id_xref": ("SHAREPOINT_LIST_ID_XREF", _text),
        "list_id_sas_complexity": (
            "SHAREPOINT_LIST_ID_SAS_COMPLEXITY",
            _text,
        ),
    },
    "observability": {
        "debug": ("SAS_MIGRATE_DEBUG", _boolean),
        "log_file": ("SAS_MIGRATE_LOG_FILE", _text),
        "trace_http": ("SAS_MIGRATE_TRACE_HTTP", _boolean),
        "capture_crashes": ("SAS_MIGRATE_CAPTURE_CRASHES", _boolean),
    },
}


def _document_sections(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    for section, field in _FORBIDDEN_FILE_FIELDS:
        value = document.get(section)
        if isinstance(value, Mapping) and field in value:
            raise ConfigurationError(
                f"{section}.{field} is a credential and cannot be stored in "
                "the settings document"
            )
    sections: dict[str, dict[str, Any]] = {}
    for section in _SECTIONS:
        raw = document.get(section, {})
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ConfigurationError(f"{section} must be a JSON object")
        sections[section] = {
            str(key): value
            for key, value in raw.items()
            if value is not None and not str(key).startswith("_")
        }
    return sections


def _apply_environment(
    sections: dict[str, dict[str, Any]],
    environ: Mapping[str, str],
) -> None:
    for section, fields in _ENV.items():
        for field, (name, parser) in fields.items():
            raw = environ.get(name)
            if raw is None or not raw.strip():
                continue
            try:
                sections[section][field] = parser(raw)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(f"invalid {name}: {exc}") from exc

    vault_ca = environ.get("VAULT_CACERT")
    vault_skip = environ.get("VAULT_SKIP_VERIFY")
    if vault_ca and vault_ca.strip():
        sections["vault"]["verify"] = vault_ca.strip()
    elif vault_skip and vault_skip.strip():
        try:
            sections["vault"]["verify"] = not _boolean(vault_skip)
        except ValueError as exc:
            raise ConfigurationError(f"invalid VAULT_SKIP_VERIFY: {exc}") from exc

    if "verify" not in sections["azure"] and "verify" in sections["vault"]:
        sections["azure"]["verify"] = sections["vault"]["verify"]
    if (
        "secret_scope" not in sections["sharepoint"]
        and "secret_scope" in sections["databricks"]
    ):
        sections["sharepoint"]["secret_scope"] = sections["databricks"][
            "secret_scope"
        ]


def load_settings(
    *,
    document: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> InfrastructureSettings:
    """Resolve a supplied JSON-like document with environment overrides."""

    resolved_environment = os.environ if environ is None else environ
    if document is None:
        configured_path = resolved_environment.get(CONFIG_PATH_ENV)
        if configured_path and configured_path.strip():
            return load_settings_file(
                configured_path.strip(),
                environ=resolved_environment,
            )
    sections = _document_sections(document or {})
    _apply_environment(sections, resolved_environment)
    try:
        return InfrastructureSettings.model_validate(sections)
    except ValidationError as exc:
        raise ConfigurationError(str(exc)) from exc


def load_settings_file(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> InfrastructureSettings:
    """Read one explicit UTF-8 JSON settings file and apply environment values."""

    resolved = Path(path)
    try:
        document = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot load settings from {resolved}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ConfigurationError(f"settings document {resolved} must be a JSON object")
    return load_settings(document=document, environ=environ)


__all__ = ["CONFIG_PATH_ENV", "ConfigurationError", "load_settings", "load_settings_file"]
