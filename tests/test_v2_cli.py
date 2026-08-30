"""Operational contracts for the Phase 10 v2 CLI composition root."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Self

import pytest

from sas_migrate.cli import build_parser, main

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _assessment_units(path: Path) -> Path:
    return _write_json(
        path,
        [
            {
                "source_id": "producer.sas",
                "line_count": 12,
                "chunk_count": 1,
                "step_count": 1,
                "output_datasets": ["work.customer"],
            },
            {
                "source_id": "consumer.sas",
                "line_count": 8,
                "chunk_count": 1,
                "step_count": 1,
                "input_datasets": ["work.customer"],
            },
        ],
    )


def _evaluation_run(path: Path, *, response: str = "```sql\nSELECT 1\n```") -> Path:
    return _write_json(
        path,
        {
            "schema_version": 2,
            "run_id": "offline-case",
            "target": "spark_sql",
            "units": [
                {
                    "unit_id": "unit-1",
                    "source": "proc sql; select 1; quit;",
                    "response": response,
                }
            ],
        },
    )


def _hydration_plan(path: Path, *, kind: str = "file") -> Path:
    return _write_json(
        path,
        {
            "schema_version": 2,
            "run_date": "20260830",
            "items": [
                {
                    "schema_version": 2,
                    "source": {
                        "schema_version": 2,
                        "kind": kind,
                        "locator": str(path.parent),
                        "object_name": "customers",
                        "source_name": "customers.csv",
                    },
                    "target_table": "main.bronze.customers",
                }
            ],
        },
    )


def test_parser_exposes_operational_commands_and_only_supported_targets() -> None:
    parser = build_parser()
    args = parser.parse_args(["assess", "units.json", "--target", "pyspark"])
    assert args.command == "assess"
    assert args.target == "pyspark"

    with pytest.raises(SystemExit) as unsupported:
        parser.parse_args(["assess", "units.json", "--target", "spark-scala"])
    assert unsupported.value.code == 2

    local = parser.parse_args(
        ["convert", "local", "source", "--target", "spark_sql", "--dry-run"]
    )
    assert local.command == "convert"
    assert local.convert_command == "local"
    assert local.target == "spark_sql"
    with pytest.raises(SystemExit) as scala:
        parser.parse_args(
            ["convert", "local", "source", "--target", "spark-scala"]
        )
    assert scala.value.code == 2

    check = parser.parse_args(["check", "sharepoint", "--offline"])
    assert check.command == "check"
    assert check.check_command == "sharepoint"
    assert check.offline is True

    hydrate = parser.parse_args(["hydrate", "plan.json", "--dry-run"])
    assert hydrate.command == "hydrate"
    assert hydrate.dry_run is True
    assert hydrate.on_error == "continue"


def test_assess_emits_json_and_markdown_from_packaged_profiles(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = _assessment_units(tmp_path / "units.json")

    assert main(["assess", str(input_path), "--target", "pyspark"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["target"] == "pyspark"
    assert report["profile"] == "pyspark"
    assert report["dependencies"] == [
        {
            "producer": "producer.sas",
            "consumer": "consumer.sas",
            "dataset": "work.customer",
        }
    ]

    output = tmp_path / "assessment.md"
    assert (
        main(
            [
                "assess",
                str(input_path),
                "--format",
                "markdown",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert "Migration assessment" in output.read_text("utf-8")

    assert (
        main(
            [
                "assess",
                str(input_path),
                "--profiles",
                str(ROOT / "src" / "sas_migrate" / "resources" / "assessment"),
                "--format",
                "markdown",
            ]
        )
        == 0
    )
    assert "Migration assessment" in capsys.readouterr().out


def test_assess_pdf_requires_a_path_and_writes_a_real_document(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = _assessment_units(tmp_path / "units.json")
    assert main(["assess", str(input_path), "--format", "pdf"]) == 2
    assert "PDF output requires --output PATH" in capsys.readouterr().err

    output = tmp_path / "assessment.pdf"
    assert (
        main(
            [
                "assess",
                str(input_path),
                "--format",
                "pdf",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.read_bytes().startswith(b"%PDF")


def test_validate_emits_report_and_returns_failure_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    passing = _evaluation_run(tmp_path / "passing.json")
    assert main(["validate", str(passing), "--model", "offline-test"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["model"] == "offline-test"
    assert report["target"] == "spark_sql"
    assert all(metric["passed"] for metric in report["results"][0]["metrics"])

    failing = _evaluation_run(tmp_path / "failing.json", response="not target code")
    assert main(["validate", str(failing)]) == 1
    failed_report = json.loads(capsys.readouterr().out)
    language = next(
        metric
        for metric in failed_report["results"][0]["metrics"]
        if metric["metric"] == "language_compliance"
    )
    assert not language["passed"]


def test_validate_reports_component_token_budget_and_pdf(
    tmp_path: Path,
) -> None:
    run = _evaluation_run(tmp_path / "run.json")
    ledger = _write_json(
        tmp_path / "ledger.json",
        {"schema_version": 2, "records": []},
    )
    policy = _write_json(tmp_path / "policy.json", {"max_run_tokens": 10})
    markdown = tmp_path / "validation.md"

    assert (
        main(
            [
                "validate",
                str(run),
                "--translation-ledger",
                str(ledger),
                "--translation-policy",
                str(policy),
                "--format",
                "markdown",
                "--output",
                str(markdown),
            ]
        )
        == 0
    )
    text = markdown.read_text("utf-8")
    assert "Translation token budget" in text
    assert "token_budget_compliance: **PASS**" in text

    pdf = tmp_path / "validation.pdf"
    assert (
        main(
            [
                "validate",
                str(run),
                "--format",
                "pdf",
                "--output",
                str(pdf),
            ]
        )
        == 0
    )
    assert pdf.read_bytes().startswith(b"%PDF")


def test_cli_returns_operator_error_for_invalid_or_missing_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    assert main(["assess", str(invalid)]) == 2
    assert "invalid assessment units" in capsys.readouterr().err

    assert main(["validate", str(tmp_path / "missing.json")]) == 2
    assert "could not read" in capsys.readouterr().err

    empty = _write_json(tmp_path / "empty.json", [])
    assert main(["assess", str(empty)]) == 2
    assert "at least one unit" in capsys.readouterr().err


def test_cli_reports_invalid_optional_contract_and_output_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = _evaluation_run(tmp_path / "run.json")
    invalid_ledger = _write_json(tmp_path / "ledger.json", {"records": "invalid"})
    assert (
        main(["validate", str(run), "--translation-ledger", str(invalid_ledger)])
        == 2
    )
    assert "invalid TokenCallLedger" in capsys.readouterr().err

    missing_parent = tmp_path / "missing" / "report.json"
    assert main(["validate", str(run), "--output", str(missing_parent)]) == 2
    assert "could not write report" in capsys.readouterr().err


def test_smoke_human_and_quiet_presentations(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["smoke"]) == 0
    output = capsys.readouterr().out
    assert "v2 deployment smoke: PASSED" in output
    assert "v2_application_flow: pass" in output

    assert main(["smoke", "--quiet"]) == 0
    assert capsys.readouterr().out == ""


def test_convert_local_dry_run_needs_no_credential_and_records_dialect(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "program.sas").write_text("proc sql; select 1; quit;", encoding="utf-8")
    output = tmp_path / "artifacts"

    assert (
        main(
            [
                "convert",
                "local",
                str(source),
                "--output-dir",
                str(output),
                "--target",
                "spark_sql",
                "--dry-run",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["outcomes"][0]["status"] == "Completed"
    plan_path = Path(result["outcomes"][0]["artifacts"][0]["location"])
    plan = json.loads(plan_path.read_text("utf-8"))
    assert plan["sqlglot_dialect"] == "databricks"
    assert plan["token_policy"]["max_input_tokens"] == 128_000


def test_convert_local_live_uses_gateway_port_and_validates_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sas_migrate.adapters.ai as ai_adapters
    from sas_migrate.application.ports import ProviderResponse, ProviderTokenUsage
    from sas_migrate.core.responses import (
        TranslationCell,
        TranslationCellKind,
        TranslationDocument,
    )

    class _Gateway:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def invoke(self, _prompt, target, *, attempt: int) -> ProviderResponse:
            del attempt
            document = TranslationDocument(
                target=target.target,
                analysis="Preserve semantics.",
                cells=(
                    TranslationCell(
                        kind=TranslationCellKind.CODE,
                        source="SELECT 1",
                        language="sql",
                    ),
                ),
            )
            return ProviderResponse(
                raw_message=document.model_dump_json(),
                structured_document=document,
                usage=ProviderTokenUsage(input_tokens=100, output_tokens=20),
            )

    monkeypatch.setattr(ai_adapters, "OpenAICompatibleLLM", _Gateway)
    monkeypatch.setenv("TEST_GATEWAY_TOKEN", "secret")
    source = tmp_path / "source"
    source.mkdir()
    (source / "program.sas").write_text("proc sql; select 1; quit;", encoding="utf-8")
    output = tmp_path / "artifacts"

    assert (
        main(
            [
                "convert",
                "local",
                str(source),
                "--output-dir",
                str(output),
                "--api-key-env",
                "TEST_GATEWAY_TOKEN",
                "--gateway-base-url",
                "https://gateway.example/v1",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    artifacts = result["outcomes"][0]["artifacts"]
    assert any(artifact["kind"] == "notebook" for artifact in artifacts)
    assert any(artifact["kind"] == "conversion_run_summary" for artifact in artifacts)


def test_convert_local_reports_operator_configuration_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing"
    assert main(["convert", "local", str(missing), "--dry-run"]) == 2
    assert "source directory does not exist" in capsys.readouterr().err

    source = tmp_path / "source"
    source.mkdir()
    (source / "program.sas").write_text("data out; run;", encoding="utf-8")
    monkeypatch.delenv("MISSING_GATEWAY_TOKEN", raising=False)
    assert (
        main(
            [
                "convert",
                "local",
                str(source),
                "--api-key-env",
                "MISSING_GATEWAY_TOKEN",
            ]
        )
        == 2
    )
    assert "gateway credential is not set" in capsys.readouterr().err

    assert (
        main(
            [
                "convert",
                "local",
                str(source),
                "--dry-run",
                "--max-input-tokens",
                "100",
                "--reserved-output-tokens",
                "100",
            ]
        )
        == 2
    )
    assert "invalid local conversion settings" in capsys.readouterr().err


def test_hydrate_dry_run_is_versioned_and_resolves_no_runtime(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = _hydration_plan(tmp_path / "plan.json", kind="oracle")
    output = tmp_path / "report.json"

    assert (
        main(
            [
                "hydrate",
                str(plan),
                "--dry-run",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == ""
    report = json.loads(output.read_text("utf-8"))
    assert report["schema_version"] == 2
    assert report["dry_run"] is True
    assert report["outcomes"][0]["status"] == "skipped"
    assert report["outcomes"][0]["error"] == "dry run"


def test_hydrate_live_composes_driver_and_delta_sink(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sas_migrate.cli.hydration as hydration_cli

    calls: list[tuple[object, ...]] = []

    class _Driver:
        def batches(self, item: object) -> tuple[list[int], ...]:
            calls.append(("batches", item))
            return ([1, 2, 3],)

        def close(self) -> None:
            calls.append(("close",))

    class _Registry:
        def driver_for(self, kind: object) -> _Driver:
            calls.append(("driver", kind))
            return _Driver()

    class _Sink:
        def write(self, item: object, batches: object) -> int:
            calls.append(("write", item, tuple(batches)))  # type: ignore[arg-type]
            return 3

    def registry(*, batch_rows: int) -> _Registry:
        calls.append(("registry", batch_rows))
        return _Registry()

    def sink(*, apply_index_clustering: bool) -> _Sink:
        calls.append(("sink", apply_index_clustering))
        return _Sink()

    monkeypatch.setattr(hydration_cli, "hydration_driver_registry", registry)
    monkeypatch.setattr(hydration_cli, "hydration_delta_sink", sink)
    plan = _hydration_plan(tmp_path / "plan.json")

    assert (
        main(
            [
                "hydrate",
                str(plan),
                "--batch-rows",
                "25",
                "--apply-index-clustering",
                "--on-error",
                "stop",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["outcomes"][0]["status"] == "written"
    assert report["outcomes"][0]["rows"] == 3
    assert calls[0:2] == [("registry", 25), ("sink", True)]
    assert calls[-1] == ("close",)


def test_hydrate_reports_unconfigured_driver_as_item_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = _hydration_plan(tmp_path / "plan.json", kind="sftp")

    assert main(["hydrate", str(plan)]) == 1
    report = json.loads(capsys.readouterr().out)
    outcome = report["outcomes"][0]
    assert outcome["status"] == "failed"
    assert "no sftp hydration driver is configured" in outcome["error"]


def test_hydrate_reports_invalid_plan_batch_size_and_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid = _write_json(tmp_path / "invalid.json", {"items": "invalid"})
    assert main(["hydrate", str(invalid), "--dry-run"]) == 2
    assert "invalid HydrationPlan" in capsys.readouterr().err

    empty = _write_json(tmp_path / "empty.json", {"schema_version": 2, "items": []})
    assert main(["hydrate", str(empty), "--dry-run"]) == 2
    assert "at least one item" in capsys.readouterr().err

    plan = _hydration_plan(tmp_path / "plan.json")
    assert main(["hydrate", str(plan), "--dry-run", "--batch-rows", "0"]) == 2
    assert "--batch-rows must be at least 1" in capsys.readouterr().err

    assert (
        main(
            [
                "hydrate",
                str(plan),
                "--dry-run",
                "--output",
                str(tmp_path / "missing" / "report.json"),
            ]
        )
        == 2
    )
    assert "could not write report" in capsys.readouterr().err


def _sharepoint_config(path: Path, **sharepoint: object) -> Path:
    values = {
        "site_id": "example.sharepoint.com,site-id,web-id",
        "drive_id": "drive-1",
        "file_server_base_path": "Applications",
        "list_id_sas_requests": "requests",
        **sharepoint,
    }
    return _write_json(
        path,
        {
            "azure": {
                "tenant_id": "tenant-1",
                "client_id": "client-1",
            },
            "sharepoint": values,
        },
    )


def _jwt(payload: dict[str, object]) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    return f"header.{encoded.rstrip('=')}.signature"


def test_check_sharepoint_offline_emits_versioned_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _sharepoint_config(tmp_path / "settings.json")
    output = tmp_path / "preflight.json"

    assert (
        main(
            [
                "check",
                "sharepoint",
                "--config",
                str(config),
                "--offline",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == ""
    report = json.loads(output.read_text("utf-8"))
    assert report["schema_version"] == 2
    assert report["offline"] is True
    assert [check["name"] for check in report["checks"]] == ["config", "imports"]
    assert all(check["status"] == "pass" for check in report["checks"])


def test_check_sharepoint_live_is_read_only_and_redacts_token(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic import SecretStr

    import sas_migrate.adapters.sharepoint as sharepoint_adapters
    from sas_migrate.application.ports import AccessToken

    calls: list[tuple[object, ...]] = []

    class _Transport:
        def __init__(self, settings: object, token_provider: object) -> None:
            calls.append(("init", settings, token_provider))

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_exc: object) -> None:
            calls.append(("close",))

        def access_token(self) -> AccessToken:
            calls.append(("token",))
            return AccessToken(
                value=SecretStr(
                    _jwt(
                        {
                            "aud": "graph",
                            "tid": "tenant-1",
                            "appid": "client-1",
                            "roles": ["Sites.ReadWrite.All"],
                        }
                    )
                ),
                source="test:sharepoint",
                expires_at_epoch=9999,
            )

        def resolve_drive_id(self) -> str:
            calls.append(("drive",))
            return "drive-1"

        def list_directory(self, path: str = "") -> list[dict[str, object]]:
            calls.append(("directory", path))
            return [{"name": "Application A"}]

        def list_items(
            self,
            list_id: str,
            *,
            top: int | None = None,
            **_options: object,
        ) -> list[dict[str, object]]:
            calls.append(("list", list_id, top))
            return [{"fields": {"Application Name": "Application A"}}]

    monkeypatch.setattr(
        sharepoint_adapters,
        "SharePointGraphTransport",
        _Transport,
    )
    config = _sharepoint_config(tmp_path / "settings.json")

    assert main(["check", "sharepoint", "--config", str(config)]) == 0
    report_text = capsys.readouterr().out
    report = json.loads(report_text)
    assert [check["status"] for check in report["checks"]] == [
        "pass",
        "pass",
        "pass",
        "pass",
        "pass",
        "pass",
    ]
    assert "signature" not in report_text
    assert calls[-1] == ("close",)
    assert all(call[0] not in {"write", "update", "mkdir"} for call in calls)


def test_check_sharepoint_reports_config_and_output_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "check",
                "sharepoint",
                "--config",
                str(tmp_path / "missing.json"),
                "--offline",
            ]
        )
        == 2
    )
    assert "invalid v2 settings" in capsys.readouterr().err

    incomplete = _write_json(tmp_path / "incomplete.json", {})
    assert (
        main(
            ["check", "sharepoint", "--config", str(incomplete), "--offline"]
        )
        == 1
    )
    report = json.loads(capsys.readouterr().out)
    assert report["checks"][0]["status"] == "fail"

    config = _sharepoint_config(tmp_path / "settings.json")
    assert (
        main(
            [
                "check",
                "sharepoint",
                "--config",
                str(config),
                "--offline",
                "--output",
                str(tmp_path / "missing" / "report.json"),
            ]
        )
        == 2
    )
    assert "could not write report" in capsys.readouterr().err


def test_sharepoint_token_provider_selects_environment_or_secret_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from pydantic import SecretStr

    import sas_migrate.adapters.auth as auth_adapters
    import sas_migrate.adapters.credentials as credential_adapters
    from sas_migrate.application.ports import (
        AccessToken,
        CredentialProviderUnavailable,
        CredentialValue,
    )
    from sas_migrate.cli.sharepoint import (
        DatabricksSharePointTokenProvider,
        sharepoint_token_provider,
    )
    from sas_migrate.config import InfrastructureSettings

    environment = sharepoint_token_provider(
        InfrastructureSettings.model_validate(
            {
                "azure": {"tenant_id": "tenant", "client_id": "client"},
                "sharepoint": {"list_id_sas_requests": "requests"},
            }
        ),
        environ={"AZURE_CLIENT_SECRET": "environment-secret"},
    )
    assert type(environment).__name__ == "MsalAccessTokenProvider"

    fetched: list[str] = []

    class _Secrets:
        def __init__(self, references: object) -> None:
            assert references

        async def get(self, name: str) -> CredentialValue | None:
            fetched.append(name)
            values = {
                "sharepoint_tenant_id": "tenant-from-scope",
                "sharepoint_client_id": "client-from-scope",
                "sharepoint_client_secret": "secret-from-scope",
            }
            return CredentialValue(
                name=name,
                value=SecretStr(values[name]),
                source="test:scope",
            )

    class _Msal:
        def __init__(
            self,
            settings: object,
            credentials: _Secrets,
            *,
            credential_name: str,
        ) -> None:
            assert settings.tenant_id == "tenant-from-scope"  # type: ignore[attr-defined]
            assert settings.client_id == "client-from-scope"  # type: ignore[attr-defined]
            assert credential_name == "sharepoint_client_secret"
            self.credentials = credentials

        async def get_token(self, scopes: tuple[str, ...] = ()) -> AccessToken:
            credential = await self.credentials.get("sharepoint_client_secret")
            assert credential is not None
            return AccessToken(value=credential.value, source="test:msal")

    monkeypatch.setattr(
        credential_adapters,
        "DatabricksSecretCredentialProvider",
        _Secrets,
    )
    monkeypatch.setattr(auth_adapters, "MsalAccessTokenProvider", _Msal)
    scoped_settings = InfrastructureSettings.model_validate(
        {
            "sharepoint": {
                "secret_scope": "sharepoint-scope",
                "list_id_sas_requests": "requests",
            }
        }
    )
    scoped = sharepoint_token_provider(scoped_settings)
    assert isinstance(scoped, DatabricksSharePointTokenProvider)
    assert asyncio.run(scoped.get_token()).source == "test:msal"
    assert asyncio.run(scoped.get_token(("custom-scope",))).source == "test:msal"
    assert fetched == [
        "sharepoint_tenant_id",
        "sharepoint_client_id",
        "sharepoint_client_secret",
        "sharepoint_client_secret",
    ]

    unconfigured = DatabricksSharePointTokenProvider(
        InfrastructureSettings.model_validate(
            {"sharepoint": {"list_id_sas_requests": "requests"}}
        )
    )
    with pytest.raises(CredentialProviderUnavailable, match="not configured"):
        asyncio.run(unconfigured.get_token())

    class _MissingSecrets(_Secrets):
        async def get(self, name: str) -> CredentialValue | None:
            if name in {"sharepoint_tenant_id", "sharepoint_client_id"}:
                return None
            return await super().get(name)

    monkeypatch.setattr(
        credential_adapters,
        "DatabricksSecretCredentialProvider",
        _MissingSecrets,
    )
    missing = DatabricksSharePointTokenProvider(scoped_settings)
    with pytest.raises(
        CredentialProviderUnavailable,
        match="lacks tenant id and client id",
    ):
        asyncio.run(missing.get_token())
