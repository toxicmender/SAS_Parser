"""V2 conversion workflow contracts and adapter-level end-to-end tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sas_migrate.adapters.conversion import (
    LocalConversionRequestRepository,
    LocalConversionSourceRepository,
    SharePointConversionConfig,
    SharePointConversionRequestRepository,
    SharePointConversionSourceRepository,
    request_from_row,
)
from sas_migrate.application.conversion import (
    ConversionModelPreference,
    ConversionRequest,
    ConversionStatus,
    ConversionTranslationCommand,
    ConversionTranslationResult,
    ConversionWorkflow,
    model_for,
    select_requests,
)
from sas_migrate.application.translation import ArtifactLocator
from sas_migrate.core.targets import SPARK_SQL, TargetId, TargetSource


class _Translator:
    def __init__(self, *, failing_requests: set[str] | None = None) -> None:
        self.commands: list[ConversionTranslationCommand] = []
        self._failing = failing_requests or set()

    async def translate(
        self,
        command: ConversionTranslationCommand,
    ) -> ConversionTranslationResult:
        self.commands.append(command)
        if command.request.request_id in self._failing:
            return ConversionTranslationResult(ok=False, error="translation refused")
        return ConversionTranslationResult(
            ok=True,
            artifacts=(
                ArtifactLocator(
                    artifact_id=f"{command.request.request_id}-notebook",
                    location=f"runs/{command.request.request_id}.ipynb",
                    kind="notebook",
                    media_type="application/x-ipynb+json",
                ),
            ),
        )


def _request(request_id: str = "7", **overrides: Any) -> ConversionRequest:
    values: dict[str, Any] = {
        "request_id": request_id,
        "application_name": f"App{request_id}",
        "output_language": "SQL",
        "status": "New",
    }
    values.update(overrides)
    return ConversionRequest(**values)


def test_select_requests_filters_pending_id_and_application() -> None:
    requests = (
        _request("1", application_name="One"),
        _request("2", application_name="Two", status="Completed"),
        _request("3", application_name="Three"),
    )
    assert [item.request_id for item in select_requests(requests)] == ["1", "3"]
    assert [
        item.request_id
        for item in select_requests(
            requests,
            include_completed=True,
            application_name=" two ",
        )
    ] == ["2"]
    assert [item.request_id for item in select_requests(requests, request_id=" 3 ")] == [
        "3"
    ]


def test_model_for_uses_first_matching_non_blank_preference() -> None:
    preferences = (
        ConversionModelPreference(request_id="7", model=" model-a "),
        ConversionModelPreference(request_id="7", model="model-b"),
    )
    assert model_for(_request(), preferences, "fallback") == "model-a"
    assert model_for(_request("8"), preferences, "fallback") == "fallback"
    assert model_for(
        _request(),
        (ConversionModelPreference(request_id="7", model="  "),),
        "fallback",
    ) == "fallback"


def test_translation_result_enforces_success_and_failure_shape() -> None:
    with pytest.raises(ValueError, match="successful conversion"):
        ConversionTranslationResult(ok=True, error="unexpected")
    with pytest.raises(ValueError, match="failed conversion"):
        ConversionTranslationResult(ok=False)


@pytest.mark.anyio
async def test_local_conversion_runs_end_to_end(tmp_path: Path) -> None:
    (tmp_path / "b.sas").write_text("proc print; run;", encoding="utf-8")
    (tmp_path / "a.txt").write_text("data a; run;", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("no", encoding="utf-8")
    request = _request()
    requests = LocalConversionRequestRepository(
        request,
        preferences=(ConversionModelPreference(request_id="7", model="preferred"),),
    )
    translator = _Translator()
    workflow = ConversionWorkflow(
        requests=requests,
        sources=LocalConversionSourceRepository(tmp_path),
        translator=translator,
        default_model="fallback",
    )

    batch = await workflow.run()

    assert batch.exit_code == 0
    assert requests.statuses == [ConversionStatus.IN_PROGRESS, ConversionStatus.COMPLETED]
    assert (await requests.list_requests())[0].status == "Completed"
    command = translator.commands[0]
    assert command.model == "preferred"
    assert command.target.target is TargetId.SPARK_SQL
    assert command.target.source is TargetSource.REQUEST
    assert SPARK_SQL.sqlglot_dialect == "databricks"
    assert [source.name for source in command.sources] == ["a.txt", "b.sas"]
    assert batch.outcomes[0].artifacts[0].kind == "notebook"


@pytest.mark.anyio
async def test_dry_run_leaves_request_status_unchanged(tmp_path: Path) -> None:
    (tmp_path / "a.sas").write_text("data a; run;", encoding="utf-8")
    requests = LocalConversionRequestRepository(_request())
    translator = _Translator()
    batch = await ConversionWorkflow(
        requests=requests,
        sources=LocalConversionSourceRepository(tmp_path),
        translator=translator,
        default_model="fallback",
    ).run(dry_run=True)

    assert batch.exit_code == 0
    assert requests.statuses == []
    assert translator.commands[0].dry_run is True


@pytest.mark.anyio
async def test_local_adapters_reject_unknown_request_and_missing_directory(
    tmp_path: Path,
) -> None:
    requests = LocalConversionRequestRepository(_request())
    with pytest.raises(KeyError, match="unknown local conversion request"):
        await requests.set_status("other", ConversionStatus.FAILED)
    with pytest.raises(FileNotFoundError, match="source directory does not exist"):
        await LocalConversionSourceRepository(tmp_path / "missing").sources_for(
            _request()
        )


@pytest.mark.anyio
async def test_failure_has_one_terminal_status_and_does_not_raise(tmp_path: Path) -> None:
    (tmp_path / "a.sas").write_text("data a; run;", encoding="utf-8")
    requests = LocalConversionRequestRepository(_request())
    batch = await ConversionWorkflow(
        requests=requests,
        sources=LocalConversionSourceRepository(tmp_path),
        translator=_Translator(failing_requests={"7"}),
        default_model="fallback",
    ).run()

    assert batch.exit_code == 1
    assert requests.statuses == [ConversionStatus.IN_PROGRESS, ConversionStatus.FAILED]
    assert requests.statuses.count(ConversionStatus.FAILED) == 1
    assert batch.outcomes[0].error == "translation refused"


@pytest.mark.anyio
async def test_empty_source_set_fails_without_calling_translator(tmp_path: Path) -> None:
    requests = LocalConversionRequestRepository(_request())
    translator = _Translator()
    batch = await ConversionWorkflow(
        requests=requests,
        sources=LocalConversionSourceRepository(tmp_path),
        translator=translator,
        default_model="fallback",
    ).run()

    assert batch.exit_code == 1
    assert translator.commands == []
    assert "no supported source files" in (batch.outcomes[0].error or "")


@pytest.mark.anyio
async def test_status_repository_failures_are_isolated(tmp_path: Path, caplog) -> None:
    (tmp_path / "a.sas").write_text("data a; run;", encoding="utf-8")

    class _FailingStatuses(LocalConversionRequestRepository):
        def __init__(self, *, fail_on: ConversionStatus) -> None:
            super().__init__(_request())
            self.fail_on = fail_on

        async def set_status(self, request_id: str, status: ConversionStatus) -> None:
            if status is self.fail_on:
                raise RuntimeError(f"cannot write {status.value}")
            await super().set_status(request_id, status)

    start_failure = await ConversionWorkflow(
        requests=_FailingStatuses(fail_on=ConversionStatus.IN_PROGRESS),
        sources=LocalConversionSourceRepository(tmp_path),
        translator=_Translator(),
        default_model="fallback",
    ).run()
    terminal_failure = await ConversionWorkflow(
        requests=_FailingStatuses(fail_on=ConversionStatus.FAILED),
        sources=LocalConversionSourceRepository(tmp_path),
        translator=_Translator(failing_requests={"7"}),
        default_model="fallback",
    ).run()

    assert "could not mark request in progress" in (start_failure.outcomes[0].error or "")
    assert terminal_failure.exit_code == 1
    assert "could not mark conversion request 7 failed" in caplog.text


@pytest.mark.anyio
async def test_bad_target_is_isolated_from_later_request(tmp_path: Path) -> None:
    (tmp_path / "a.sas").write_text("data a; run;", encoding="utf-8")
    class _TwoRequests(LocalConversionRequestRepository):
        async def list_requests(self) -> tuple[ConversionRequest, ...]:
            return (_request("7", output_language="Scala"), _request("8"))

        async def set_status(self, request_id: str, status: ConversionStatus) -> None:
            self.statuses.append(status)

    requests = _TwoRequests(_request(output_language="Scala"))
    translator = _Translator()
    batch = await ConversionWorkflow(
        requests=requests,
        sources=LocalConversionSourceRepository(tmp_path),
        translator=translator,
        default_model="fallback",
    ).run()

    assert [outcome.status for outcome in batch.outcomes] == [
        ConversionStatus.FAILED,
        ConversionStatus.COMPLETED,
    ]
    assert [command.request.request_id for command in translator.commands] == ["8"]
    assert "unsupported target" in (batch.outcomes[0].error or "")


class _SharePointTransport:
    def __init__(self) -> None:
        self.updated: list[tuple[str, str, dict[str, Any]]] = []
        self.request_rows = [
            {
                "id": "7",
                "fields": {
                    "Application Name": "MyApp",
                    "Source Language": "SAS",
                    "Destination Language": "PySpark",
                    "Validation x0020 Documents x0020 ": "Yes",
                    "Status": "New",
                },
            }
        ]

    def list_items(self, list_id: str, **_options: Any) -> list[dict[str, Any]]:
        if list_id == "requests":
            return self.request_rows
        return [{"id": "9", "fields": {"Request_ID": "7", "Model": "preferred"}}]

    def update_list_item(
        self,
        list_id: str,
        item_id: str,
        fields: dict[str, Any],
    ) -> None:
        self.updated.append((list_id, item_id, fields))

    def list_files(
        self,
        folder: str,
        extensions: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        assert extensions == {"sas", "txt"}
        return [
            {"name": "b.sas", "path": f"{folder}/b.sas", "is_folder": False},
            {"name": "a.txt", "path": f"{folder}/a.txt", "is_folder": False},
        ]

    def download_file_as_text(self, path: str, *, encoding: str = "utf-8") -> str:
        assert encoding == "utf-8"
        return f"source: {path}"


@pytest.mark.anyio
async def test_sharepoint_conversion_runs_end_to_end_through_transport() -> None:
    transport = _SharePointTransport()
    config = SharePointConversionConfig(
        request_list_id="requests",
        conversion_list_id="conversions",
        base_path="Kit/Applications",
    )
    requests = SharePointConversionRequestRepository(transport, config)
    translator = _Translator()
    batch = await ConversionWorkflow(
        requests=requests,
        sources=SharePointConversionSourceRepository(transport, config),
        translator=translator,
        default_model="fallback",
    ).run()

    assert batch.exit_code == 0
    assert [fields["Status"] for _, _, fields in transport.updated] == [
        "In Progress",
        "Completed",
    ]
    command = translator.commands[0]
    assert command.model == "preferred"
    assert command.request.validation_required is True
    assert command.target.target is TargetId.PYSPARK
    assert [source.name for source in command.sources] == ["a.txt", "b.sas"]
    assert command.sources[0].source_id.startswith(
        "Kit/Applications/MyApp/scripts_original/"
    )


@pytest.mark.anyio
async def test_sharepoint_skips_malformed_and_unreadable_rows(caplog) -> None:
    transport = _SharePointTransport()
    transport.request_rows.append({"id": "8", "fields": {"Status": "New"}})
    config = SharePointConversionConfig(
        request_list_id="requests",
        base_path="Kit/Applications",
    )
    requests = await SharePointConversionRequestRepository(
        transport,
        config,
    ).list_requests()

    assert [request.request_id for request in requests] == ["7"]
    assert "skipping malformed SharePoint request row" in caplog.text


@pytest.mark.anyio
async def test_sharepoint_optional_preferences_and_source_failures(caplog) -> None:
    transport = _SharePointTransport()
    config = SharePointConversionConfig(
        request_list_id="requests",
        base_path="Kit/Applications",
    )
    request_repository = SharePointConversionRequestRepository(transport, config)
    assert await request_repository.model_preferences() == ()

    with pytest.raises(ValueError, match="requires an id"):
        request_from_row({"fields": {"Application Name": "App"}})

    sources = SharePointConversionSourceRepository(transport, config)
    with pytest.raises(ValueError, match="no source extensions known"):
        await sources.sources_for(_request(input_language="COBOL"))

    class _UnreadableTransport(_SharePointTransport):
        def list_files(
            self,
            folder: str,
            extensions: set[str] | None = None,
        ) -> list[dict[str, Any]]:
            return [
                {"name": "folder", "path": f"{folder}/folder", "is_folder": True},
                {"name": "bad.sas", "path": f"{folder}/bad.sas", "is_folder": False},
            ]

        def download_file_as_text(
            self,
            path: str,
            *,
            encoding: str = "utf-8",
        ) -> str:
            raise OSError(f"unreadable: {path} ({encoding})")

    unreadable = SharePointConversionSourceRepository(_UnreadableTransport(), config)
    assert await unreadable.sources_for(_request(application_name="MyApp")) == ()
    assert "skipping unreadable SharePoint source" in caplog.text
