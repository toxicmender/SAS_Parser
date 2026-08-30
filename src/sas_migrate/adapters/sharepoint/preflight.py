"""Read-only SharePoint deployment checks with versioned output."""

from __future__ import annotations

import base64
import binascii
import importlib.util
import json
from collections.abc import Callable
from typing import Any, Literal, Protocol

from pydantic import Field

from sas_migrate.application.ports import AccessToken
from sas_migrate.config import SharePointSettings
from sas_migrate.core.models import ContractModel, VersionedContract
from sas_migrate.observability import redact_mapping, redact_text

PreflightStatus = Literal["pass", "fail", "warn", "skip"]
REQUIRED_GRAPH_ROLE = "Sites.ReadWrite.All"
SUFFICIENT_GRAPH_ROLES = frozenset(
    {REQUIRED_GRAPH_ROLE, "Sites.FullControl.All"}
)


class SharePointPreflightProbe(Protocol):
    def access_token(self) -> AccessToken: ...

    def resolve_drive_id(self) -> str: ...

    def list_directory(self, path: str = "") -> list[dict[str, Any]]: ...

    def list_items(
        self,
        list_id: str,
        *,
        select: list[str] | None = None,
        expand: str = "fields",
        top: int | None = None,
        filter: str | None = None,
    ) -> list[dict[str, Any]]: ...


class PreflightCheck(ContractModel):
    name: str = Field(min_length=1)
    status: PreflightStatus
    summary: str = Field(min_length=1)
    detail: dict[str, Any] = Field(default_factory=dict)
    fix: str | None = None

    @property
    def passed(self) -> bool:
        return self.status != "fail"


class SharePointPreflightReport(VersionedContract):
    offline: bool = False
    checks: tuple[PreflightCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1


def decode_token_claims(token: str) -> dict[str, Any]:
    """Decode JWT claims for diagnostics only, without authenticating them."""

    parts = token.split(".")
    if len(parts) != 3:
        return {}
    payload = f"{parts[1]}{'=' * (-len(parts[1]) % 4)}"
    try:
        decoded = json.loads(base64.urlsafe_b64decode(payload))
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _token_detail(token: AccessToken) -> tuple[dict[str, Any], PreflightStatus, str]:
    claims = decode_token_claims(token.value.get_secret_value())
    roles = claims.get("roles")
    normalized_roles = (
        sorted(str(role) for role in roles) if isinstance(roles, list) else []
    )
    detail = {
        "source": token.source,
        "expires_at_epoch": token.expires_at_epoch,
        "audience": claims.get("aud"),
        "tenant_id": claims.get("tid"),
        "application_id": claims.get("appid") or claims.get("azp"),
        "roles": normalized_roles,
    }
    if normalized_roles and not SUFFICIENT_GRAPH_ROLES.intersection(normalized_roles):
        return (
            detail,
            "warn",
            f"token does not advertise {REQUIRED_GRAPH_ROLE}",
        )
    return detail, "pass", f"token acquired from {token.source}"


class SharePointPreflight:
    """Validate configuration and read access without mutating SharePoint."""

    _MODULES = ("msgraph", "msgraph_core", "kiota_abstractions", "msal", "httpx")

    def __init__(
        self,
        settings: SharePointSettings,
        probe: SharePointPreflightProbe | None = None,
        *,
        module_finder: Callable[[str], object | None] = importlib.util.find_spec,
    ) -> None:
        self._settings = settings
        self._probe = probe
        self._module_finder = module_finder

    def _config(self) -> PreflightCheck:
        issues = self._settings.configuration_issues()
        detail = {
            "site_id": self._settings.resolved_site_id,
            "drive_id": self._settings.drive_id,
            "base_path": self._settings.file_server_base_path,
            "list_ids": {
                "requests": self._settings.list_id_sas_requests,
                "conversions": self._settings.list_id_sas_conversions,
                "xref": self._settings.list_id_xref,
                "complexity": self._settings.list_id_sas_complexity,
            },
            "scopes": list(self._settings.scopes),
            "timeout_seconds": self._settings.timeout,
        }
        if issues:
            return PreflightCheck(
                name="config",
                status="fail",
                summary="SharePoint configuration is incomplete",
                detail=detail,
                fix="; ".join(issues),
            )
        return PreflightCheck(
            name="config",
            status="pass",
            summary=f"site {self._settings.resolved_site_id!r} is configured",
            detail=detail,
        )

    def _imports(self) -> PreflightCheck:
        installed = {
            module: self._module_finder(module) is not None for module in self._MODULES
        }
        missing = [module for module, present in installed.items() if not present]
        if missing and self._probe is None:
            return PreflightCheck(
                name="imports",
                status="fail",
                summary=f"missing: {', '.join(missing)}",
                detail=installed,
                fix='install "sas-parser[sharepoint]"',
            )
        if missing:
            return PreflightCheck(
                name="imports",
                status="skip",
                summary="not needed because a prebuilt probe was supplied",
                detail=installed,
            )
        return PreflightCheck(
            name="imports",
            status="pass",
            summary="the sharepoint extra is installed",
            detail=installed,
        )

    @staticmethod
    def _skip(name: str, dependency: str) -> PreflightCheck:
        return PreflightCheck(
            name=name,
            status="skip",
            summary=f"blocked by failed {dependency} check",
        )

    @staticmethod
    def _failure(name: str, summary: str, exc: BaseException) -> PreflightCheck:
        return PreflightCheck(
            name=name,
            status="fail",
            summary=summary,
            detail={"error": redact_text(str(exc))},
        )

    def run(self, *, offline: bool = False) -> SharePointPreflightReport:
        checks = [self._config(), self._imports()]
        if offline:
            return SharePointPreflightReport(offline=True, checks=tuple(checks))

        if any(not check.passed for check in checks):
            dependency = next(check.name for check in checks if not check.passed)
            checks.extend(
                self._skip(name, dependency)
                for name in ("token", "site", "base", "lists")
            )
            return SharePointPreflightReport(checks=tuple(checks))
        if self._probe is None:
            checks.append(
                PreflightCheck(
                    name="token",
                    status="fail",
                    summary="online preflight requires a SharePoint probe",
                    fix="compose SharePointGraphTransport with an access-token provider",
                )
            )
            checks.extend(self._skip(name, "token") for name in ("site", "base", "lists"))
            return SharePointPreflightReport(checks=tuple(checks))

        try:
            detail, status, summary = _token_detail(self._probe.access_token())
            checks.append(
                PreflightCheck(
                    name="token",
                    status=status,
                    summary=summary,
                    detail=redact_mapping(detail),
                )
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic boundary
            checks.append(self._failure("token", "could not acquire Graph token", exc))
            checks.extend(
                self._skip(name, "token") for name in ("site", "base", "lists")
            )
            return SharePointPreflightReport(checks=tuple(checks))

        try:
            drive_id = self._probe.resolve_drive_id()
            checks.append(
                PreflightCheck(
                    name="site",
                    status="pass",
                    summary="site and document library resolved",
                    detail={
                        "site_id": self._settings.resolved_site_id,
                        "drive_id": drive_id,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic boundary
            checks.append(self._failure("site", "could not resolve site drive", exc))
            checks.extend(self._skip(name, "site") for name in ("base", "lists"))
            return SharePointPreflightReport(checks=tuple(checks))

        try:
            children = self._probe.list_directory(
                self._settings.file_server_base_path
            )
            checks.append(
                PreflightCheck(
                    name="base",
                    status="pass",
                    summary="configured base path is readable",
                    detail={
                        "path": self._settings.file_server_base_path or "/",
                        "child_count": len(children),
                        "sample": [item.get("name") for item in children[:5]],
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic boundary
            checks.append(self._failure("base", "base path is not readable", exc))
            checks.append(self._skip("lists", "base"))
            return SharePointPreflightReport(checks=tuple(checks))

        configured_lists = {
            "requests": self._settings.list_id_sas_requests,
            "conversions": self._settings.list_id_sas_conversions,
            "xref": self._settings.list_id_xref,
            "complexity": self._settings.list_id_sas_complexity,
        }
        try:
            list_detail: dict[str, Any] = {}
            for name, list_id in configured_lists.items():
                if list_id is None:
                    list_detail[name] = {"status": "not configured"}
                    continue
                rows = self._probe.list_items(list_id, top=1)
                fields = sorted((rows[0].get("fields") or {}).keys()) if rows else []
                list_detail[name] = {
                    "id": list_id,
                    "sample_count": len(rows),
                    "sample_fields": fields,
                }
            checks.append(
                PreflightCheck(
                    name="lists",
                    status="pass",
                    summary="configured SharePoint lists are readable",
                    detail=list_detail,
                )
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic boundary
            checks.append(self._failure("lists", "a configured list is not readable", exc))
        return SharePointPreflightReport(checks=tuple(checks))


__all__ = [
    "PreflightCheck",
    "PreflightStatus",
    "SharePointPreflight",
    "SharePointPreflightProbe",
    "SharePointPreflightReport",
    "decode_token_claims",
]
