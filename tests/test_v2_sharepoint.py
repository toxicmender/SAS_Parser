"""V2 Graph transport, loop ownership, and read-only preflight contracts."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

import sas_migrate.adapters.sharepoint.graph as graph_module
from sas_migrate.adapters.sharepoint import (
    GraphSdkGateway,
    SharePointGraphTransport,
    SharePointPreflight,
    SharePointTransportError,
    SingleLoopWorker,
    WorkerClosedError,
    decode_token_claims,
)
from sas_migrate.application.ports import AccessToken
from sas_migrate.config import SharePointSettings


class TokenProvider:
    def __init__(self, value: str = "opaque") -> None:
        self.value = value
        self.calls: list[tuple[str, ...]] = []

    async def get_token(self, scopes: tuple[str, ...] = ()) -> AccessToken:
        self.calls.append(scopes)
        return AccessToken(
            value=SecretStr(self.value),
            source="test:token",
            expires_at_epoch=1234,
        )


class FakeGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.loop_ids: list[int] = []
        self.fail: str | None = None

    def _called(self, *call: Any) -> None:
        self.calls.append(call)
        self.loop_ids.append(id(asyncio.get_running_loop()))
        if self.fail == call[0]:
            raise RuntimeError("client_secret=super-secret")

    async def access_token(self) -> AccessToken:
        self._called("token")
        return AccessToken(
            value=SecretStr(_jwt({"roles": ["Sites.ReadWrite.All"]})),
            source="fake:graph",
            expires_at_epoch=4567,
        )

    async def resolve_drive_id(self) -> str:
        self._called("drive")
        return "drive-1"

    async def list_directory(self, path: str) -> list[dict[str, Any]]:
        self._called("directory", path)
        return [
            {"name": "nested", "is_folder": True},
            {"name": "A.SAS", "is_folder": False},
            {"name": "notes.txt", "is_folder": False},
            {"name": "data.csv", "is_folder": False},
        ]

    async def read_file(self, path: str) -> bytes:
        self._called("read", path)
        if path == "payload.json":
            return b'{"ready": true}'
        return b"\xef\xbb\xbfhello\xff"

    async def write_file(self, path: str, content: bytes) -> dict[str, Any]:
        self._called("write", path, content)
        return {"name": path.rsplit("/", 1)[-1], "is_folder": False}

    async def create_directory(
        self, path: str, conflict_behavior: str
    ) -> dict[str, Any]:
        self._called("mkdir", path, conflict_behavior)
        return {"name": path.rsplit("/", 1)[-1], "is_folder": True}

    async def list_items(
        self,
        list_id: str,
        *,
        select: list[str] | None,
        expand: str,
        top: int | None,
        filter: str | None,
    ) -> list[dict[str, Any]]:
        self._called("list", list_id, select, expand, top, filter)
        return [{"id": "1", "fields": {"Title": "ready"}}]

    async def get_list_item(
        self, list_id: str, item_id: str | int
    ) -> dict[str, Any]:
        self._called("get", list_id, item_id)
        return {"id": str(item_id), "fields": {"Title": "ready"}}

    async def update_list_item(
        self, list_id: str, item_id: str | int, fields: dict[str, Any]
    ) -> dict[str, Any]:
        self._called("update", list_id, item_id, fields)
        return fields

    async def close(self) -> None:
        self._called("close")


def _settings(**values: Any) -> SharePointSettings:
    return SharePointSettings(
        site_id="site-1",
        file_server_base_path="Apps",
        list_id_sas_requests="requests",
        **values,
    )


def _jwt(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload).encode()
    encoded = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    return f"header.{encoded}.signature"


def test_single_loop_worker_reuses_one_loop_from_sync_and_async_callers() -> None:
    worker = SingleLoopWorker(name="test-sharepoint")

    async def identity() -> tuple[int, int]:
        return id(asyncio.get_running_loop()), threading.get_ident()

    first = worker.run(identity())

    async def from_running_loop() -> tuple[int, int]:
        return worker.run(identity())

    second = asyncio.run(from_running_loop())
    assert first == second
    assert first[1] == worker.thread_id
    worker.close()
    worker.close()
    with pytest.raises(WorkerClosedError, match="closed"):
        worker.run(identity())


def test_single_loop_worker_rejects_self_reentry_and_context_closes() -> None:
    worker = SingleLoopWorker(name="reentry")

    async def nested() -> None:
        with pytest.raises(RuntimeError, match="re-enter"):
            worker.run(asyncio.sleep(0))

    worker.run(nested())
    worker.close()
    with SingleLoopWorker(name="context") as context:
        assert context.run(asyncio.sleep(0, result=3)) == 3
    with pytest.raises(WorkerClosedError):
        context.run(asyncio.sleep(0))


def test_transport_covers_file_folder_and_list_contracts_on_one_loop() -> None:
    gateway = FakeGateway()
    transport = SharePointGraphTransport(
        _settings(), TokenProvider(), gateway=gateway
    )
    try:
        assert transport.resolve_drive_id() == "drive-1"
        assert transport.list_files("Apps", {"sas", ".TXT"}) == [
            {"name": "A.SAS", "is_folder": False, "path": "Apps/A.SAS"},
            {"name": "notes.txt", "is_folder": False, "path": "Apps/notes.txt"},
        ]
        assert len(transport.list_files()) == 3
        assert transport.download_file_as_text("A.SAS") == "hello�"
        assert transport.read_json_text("payload.json") == {"ready": True}
        assert transport.write_file("Apps/out.sql", "select 1")["name"] == "out.sql"
        assert transport.upload_file("Apps", "out.py", b"x")["name"] == "out.py"
        assert transport.create_directory("Apps/new", conflict_behavior="rename")[
            "name"
        ] == "new"
        assert transport.create_folder("Apps/a/b")["name"] == "b"
        assert transport.list_items("requests", top=1)[0]["id"] == "1"
        assert transport.get_list_item("requests", 7)["id"] == "7"
        assert transport.update_list_item("requests", 7, {"Status": "Done"}) == {
            "Status": "Done"
        }
        assert transport.access_token().source == "fake:graph"
        assert len(set(gateway.loop_ids)) == 1
    finally:
        transport.close()
    assert gateway.calls[-1] == ("close",)
    transport.close()


@pytest.mark.parametrize(
    ("operation", "call"),
    [
        ("directory", lambda client: client.list_directory("bad")),
        ("read", lambda client: client.read_file("bad")),
        ("write", lambda client: client.write_file("bad", b"x")),
        ("mkdir", lambda client: client.create_directory("bad")),
        ("list", lambda client: client.list_items("bad")),
        ("get", lambda client: client.get_list_item("bad", 1)),
        ("update", lambda client: client.update_list_item("bad", 1, {"x": 1})),
    ],
)
def test_transport_normalizes_and_redacts_gateway_errors(
    operation: str, call: Any
) -> None:
    gateway = FakeGateway()
    gateway.fail = operation
    with SharePointGraphTransport(
        _settings(), TokenProvider(), gateway=gateway
    ) as transport, pytest.raises(SharePointTransportError) as raised:
        call(transport)
    assert "super-secret" not in str(raised.value)
    assert "<redacted>" in str(raised.value)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda client: client.write_file("/", b"x"), "file path"),
        (lambda client: client.upload_file("Apps", "", b"x"), "file name"),
        (lambda client: client.create_directory("/"), "folder path"),
        (lambda client: client.create_folder("/"), "folder path"),
        (
            lambda client: client.create_directory("Apps/x", conflict_behavior="merge"),
            "conflict_behavior",
        ),
        (lambda client: client.update_list_item("requests", 1, {}), "one field"),
    ],
)
def test_transport_rejects_invalid_mutations(call: Any, message: str) -> None:
    with SharePointGraphTransport(
        _settings(), TokenProvider(), gateway=FakeGateway()
    ) as transport, pytest.raises(SharePointTransportError, match=message):
        call(transport)


def test_transport_reports_invalid_json_with_path() -> None:
    with SharePointGraphTransport(
        _settings(), TokenProvider(), gateway=FakeGateway()
    ) as transport, pytest.raises(SharePointTransportError, match="bad.json"):
        transport.read_json_text("bad.json")


class PageBuilder:
    def __init__(self, pages: list[Any]) -> None:
        self.pages = pages
        self.index = 0

    async def get(self, *_args: Any) -> Any:
        page = self.pages[self.index]
        self.index += 1
        return page

    def with_url(self, _url: str) -> PageBuilder:
        return self

    async def post(self, body: Any) -> Any:
        return SimpleNamespace(name=body.name, folder=body.folder)


class FakeItemBuilder:
    def __init__(self, pages: list[Any], content: bytes = b"body") -> None:
        self.children = PageBuilder(pages)
        self.content = SimpleNamespace(get=self._get, put=self._put)
        self._content = content

    async def _get(self) -> bytes:
        return self._content

    async def _put(self, _content: bytes) -> Any:
        return SimpleNamespace(name="written", folder=None)


def _graph_client_stub() -> Any:
    file_item = SimpleNamespace(
        name="a.sas",
        id="file-1",
        folder=None,
        size=5,
        web_url="https://example/file",
        last_modified_date_time=None,
    )
    folder = SimpleNamespace(
        name="folder",
        id="folder-1",
        folder=SimpleNamespace(child_count=2),
        size=0,
        web_url="https://example/folder",
        last_modified_date_time=None,
    )
    pages = [
        SimpleNamespace(value=[file_item], odata_next_link="next"),
        SimpleNamespace(value=[folder], odata_next_link=None),
    ]
    item_builder = FakeItemBuilder(pages)
    items = SimpleNamespace(by_drive_item_id=lambda _item_id: item_builder)
    drives = SimpleNamespace(
        by_drive_id=lambda drive_id: SimpleNamespace(id=drive_id, items=items)
    )
    list_row = SimpleNamespace(
        id="11",
        web_url="https://example/list/11",
        fields=SimpleNamespace(additional_data={"Title": "Ready"}),
    )
    list_pages = [
        SimpleNamespace(value=[list_row], odata_next_link="next-list"),
        SimpleNamespace(value=[], odata_next_link=None),
    ]

    class Fields:
        async def patch(self, body: Any) -> Any:
            return SimpleNamespace(additional_data=body.additional_data)

    class ListItems(PageBuilder):
        def by_list_item_id(self, item_id: str) -> Any:
            row = SimpleNamespace(
                id=item_id,
                web_url=f"https://example/list/{item_id}",
                fields=SimpleNamespace(additional_data={"Title": "Ready"}),
            )
            return SimpleNamespace(get=_async_value(row), fields=Fields())

    list_items = ListItems(list_pages)
    lists = SimpleNamespace(
        by_list_id=lambda _list_id: SimpleNamespace(items=list_items)
    )
    drive = SimpleNamespace(get=_async_value(SimpleNamespace(id="resolved-drive")))
    site = SimpleNamespace(drive=drive, lists=lists)
    sites = SimpleNamespace(by_site_id=lambda _site_id: site)
    return SimpleNamespace(sites=sites, drives=drives)


def _async_value(value: Any) -> Any:
    async def get(*_args: Any) -> Any:
        return value

    return get


def test_graph_gateway_is_lazy_resolves_and_caches_drive_and_pages() -> None:
    provider = TokenProvider()
    clients: list[Any] = []

    def factory(_settings: SharePointSettings, _provider: TokenProvider) -> Any:
        clients.append(_graph_client_stub())
        return clients[0]

    gateway = GraphSdkGateway(_settings(), provider, client_factory=factory)

    async def exercise() -> None:
        assert clients == []
        assert await gateway.resolve_drive_id() == "resolved-drive"
        assert await gateway.resolve_drive_id() == "resolved-drive"
        assert [row["name"] for row in await gateway.list_directory("")] == [
            "a.sas",
            "folder",
        ]
        assert await gateway.read_file("a.sas") == b"body"
        assert (await gateway.write_file("b.sas", b"x"))["name"] == "written"
        assert (await gateway.create_directory("Apps/new", "replace"))["name"] == "new"
        rows = await gateway.list_items(
            "requests", select=["id"], expand="fields", top=1, filter="id gt 0"
        )
        assert rows[0]["fields"] == {"Title": "Ready"}
        assert (await gateway.get_list_item("requests", 12))["id"] == "12"
        assert await gateway.update_list_item(
            "requests", 12, {"Status": "Done"}
        ) == {"Status": "Done"}
        assert (await gateway.access_token()).source == "test:token"
        await gateway.close()

    asyncio.run(exercise())
    assert len(clients) == 1
    query = graph_module._ListQueryParameters(top=1)
    assert query.get_query_parameter("top") == "%24top"


def test_graph_gateway_uses_explicit_drive_without_building_client() -> None:
    gateway = GraphSdkGateway(_settings(drive_id="fixed"), TokenProvider())
    assert asyncio.run(gateway.resolve_drive_id()) == "fixed"


def test_graph_gateway_rejects_missing_site_and_empty_drive() -> None:
    missing = GraphSdkGateway(
        SharePointSettings(list_id_sas_requests="requests"), TokenProvider()
    )
    with pytest.raises(SharePointTransportError, match="no SharePoint site"):
        asyncio.run(missing.resolve_drive_id())

    client = SimpleNamespace(
        sites=SimpleNamespace(
            by_site_id=lambda _site: SimpleNamespace(
                drive=SimpleNamespace(get=_async_value(SimpleNamespace(id=None)))
            )
        )
    )
    empty = GraphSdkGateway(_settings(), TokenProvider(), client=client)
    with pytest.raises(SharePointTransportError, match="no accessible"):
        asyncio.run(empty.resolve_drive_id())


def test_graph_gateway_rejects_empty_file_and_missing_list_item() -> None:
    item = FakeItemBuilder([], content=None)  # type: ignore[arg-type]
    drives = SimpleNamespace(
        by_drive_id=lambda _drive: SimpleNamespace(
            items=SimpleNamespace(by_drive_item_id=lambda _item: item)
        )
    )
    missing_item = SimpleNamespace(get=_async_value(None))
    list_items = SimpleNamespace(by_list_item_id=lambda _item: missing_item)
    site = SimpleNamespace(
        lists=SimpleNamespace(
            by_list_id=lambda _list: SimpleNamespace(items=list_items)
        )
    )
    client = SimpleNamespace(
        drives=drives,
        sites=SimpleNamespace(by_site_id=lambda _site: site),
    )
    gateway = GraphSdkGateway(
        _settings(drive_id="drive"), TokenProvider(), client=client
    )
    with pytest.raises(SharePointTransportError, match="returned no content"):
        asyncio.run(gateway.read_file("empty"))
    with pytest.raises(SharePointTransportError, match="has no item"):
        asyncio.run(gateway.get_list_item("requests", 99))


def test_graph_token_credential_is_a_lazy_sdk_bridge() -> None:
    pytest.importorskip("msgraph")
    provider = TokenProvider()
    credential = graph_module._GraphTokenCredential(
        provider, ("default-scope",)
    )
    token = asyncio.run(credential.get_token())
    assert token.token == "opaque"
    assert token.expires_on == 1234
    assert provider.calls == [("default-scope",)]

    async def context_identity() -> Any:
        async with credential as entered:
            return entered

    assert asyncio.run(context_identity()) is credential

    provider_without_expiry = TokenProvider()

    async def no_expiry(scopes: tuple[str, ...] = ()) -> AccessToken:
        provider_without_expiry.calls.append(scopes)
        return AccessToken(value=SecretStr("short-lived"), source="test")

    provider_without_expiry.get_token = no_expiry  # type: ignore[method-assign]
    token = asyncio.run(credential.__class__(provider_without_expiry, ()).get_token("one"))
    assert token.token == "short-lived"
    assert token.expires_on > 0


def test_graph_gateway_closes_async_and_sync_request_adapters() -> None:
    calls: list[str] = []

    async def async_close() -> None:
        calls.append("async")

    gateway = GraphSdkGateway(
        _settings(),
        TokenProvider(),
        client=SimpleNamespace(request_adapter=SimpleNamespace(close=async_close)),
    )
    asyncio.run(gateway.close())
    gateway = GraphSdkGateway(
        _settings(),
        TokenProvider(),
        client=SimpleNamespace(
            request_adapter=SimpleNamespace(close=lambda: calls.append("sync"))
        ),
    )
    asyncio.run(gateway.close())
    assert calls == ["async", "sync"]


def test_graph_error_description_includes_service_metadata() -> None:
    class BrokenHeaders:
        def get(self, _name: str) -> Any:
            raise RuntimeError("bad headers")

    detailed = RuntimeError("fallback")
    detailed.response_status_code = 403  # type: ignore[attr-defined]
    detailed.error = SimpleNamespace(code="accessDenied", message="denied")  # type: ignore[attr-defined]
    detailed.response_headers = {"client-request-id": {"b", "a"}}  # type: ignore[attr-defined]
    rendered = graph_module._describe(detailed)
    assert "HTTP 403" in rendered
    assert "accessDenied" in rendered
    assert "request-id=a, b" in rendered

    broken = RuntimeError()
    broken.response_headers = BrokenHeaders()  # type: ignore[attr-defined]
    assert graph_module._describe(broken) == "RuntimeError"


def test_graph_transport_preserves_existing_transport_errors() -> None:
    gateway = FakeGateway()

    async def fail() -> str:
        raise SharePointTransportError("already normalized")

    gateway.resolve_drive_id = fail  # type: ignore[method-assign]
    with SharePointGraphTransport(
        _settings(), TokenProvider(), gateway=gateway
    ) as transport, pytest.raises(SharePointTransportError, match="already normalized"):
        transport.resolve_drive_id()


class PreflightProbe(FakeGateway):
    def access_token(self) -> AccessToken:  # type: ignore[override]
        self.calls.append(("token",))
        if self.fail == "token":
            raise RuntimeError("access_token=secret-value")
        return AccessToken(
            value=SecretStr(_jwt({"aud": "graph", "roles": ["Sites.ReadWrite.All"]})),
            source="preflight:test",
            expires_at_epoch=9999,
        )

    def resolve_drive_id(self) -> str:  # type: ignore[override]
        self.calls.append(("drive",))
        if self.fail == "drive":
            raise RuntimeError("drive unavailable")
        return "drive-1"

    def list_directory(self, path: str = "") -> list[dict[str, Any]]:  # type: ignore[override]
        self.calls.append(("directory", path))
        if self.fail == "directory":
            raise RuntimeError("base unavailable")
        return [{"name": "one"}, {"name": "two"}]

    def list_items(  # type: ignore[override]
        self, list_id: str, *, top: int | None = None, **_options: Any
    ) -> list[dict[str, Any]]:
        self.calls.append(("list", list_id, top))
        if self.fail == "list":
            raise RuntimeError("list unavailable")
        return [{"fields": {"Title": "one", "Status": "Ready"}}]


def test_preflight_success_is_versioned_read_only_and_serializable() -> None:
    probe = PreflightProbe()
    settings = _settings(
        list_id_sas_conversions="conversions",
        list_id_xref="xref",
        list_id_sas_complexity="complexity",
    )
    report = SharePointPreflight(
        settings,
        probe,
        module_finder=lambda _module: object(),
    ).run()
    assert report.passed is True
    assert report.exit_code == 0
    assert [check.name for check in report.checks] == [
        "config",
        "imports",
        "token",
        "site",
        "base",
        "lists",
    ]
    assert all(check.status == "pass" for check in report.checks)
    assert all(call[0] not in {"write", "mkdir", "update"} for call in probe.calls)
    restored = type(report).from_json(report.to_json())
    assert restored.schema_version == 2
    assert restored.checks[-1].detail["requests"]["sample_fields"] == [
        "Status",
        "Title",
    ]


def test_preflight_offline_reports_config_and_imports_only() -> None:
    report = SharePointPreflight(
        _settings(), module_finder=lambda _module: None
    ).run(offline=True)
    assert report.offline is True
    assert report.exit_code == 1
    assert [check.name for check in report.checks] == ["config", "imports"]
    assert report.checks[-1].fix == 'install "sas-parser[sharepoint]"'


def test_preflight_config_failure_skips_dependent_network_checks() -> None:
    settings = SharePointSettings()
    report = SharePointPreflight(
        settings,
        PreflightProbe(),
        module_finder=lambda _module: object(),
    ).run()
    assert report.passed is False
    assert [check.status for check in report.checks] == [
        "fail",
        "pass",
        "skip",
        "skip",
        "skip",
        "skip",
    ]


def test_preflight_supplied_probe_can_skip_optional_imports() -> None:
    report = SharePointPreflight(
        _settings(),
        PreflightProbe(),
        module_finder=lambda _module: None,
    ).run()
    assert report.checks[1].status == "skip"
    assert report.passed is True


@pytest.mark.parametrize(
    ("failure", "statuses"),
    [
        ("token", ["fail", "skip", "skip", "skip"]),
        ("drive", ["pass", "fail", "skip", "skip"]),
        ("directory", ["pass", "pass", "fail", "skip"]),
        ("list", ["pass", "pass", "pass", "fail"]),
    ],
)
def test_preflight_failure_order_and_redaction(
    failure: str, statuses: list[str]
) -> None:
    probe = PreflightProbe()
    probe.fail = failure
    report = SharePointPreflight(
        _settings(), probe, module_finder=lambda _module: object()
    ).run()
    assert [check.status for check in report.checks[2:]] == statuses
    assert report.exit_code == 1
    assert "secret-value" not in report.to_json()


def test_preflight_without_probe_skips_live_checks() -> None:
    report = SharePointPreflight(
        _settings(), module_finder=lambda _module: object()
    ).run()
    assert [check.status for check in report.checks[2:]] == [
        "fail",
        "skip",
        "skip",
        "skip",
    ]
    assert report.passed is False


def test_token_claim_decoder_handles_opaque_invalid_and_wrong_shape() -> None:
    assert decode_token_claims("opaque") == {}
    assert decode_token_claims("a.%%%25.b") == {}
    encoded = base64.urlsafe_b64encode(b"[]").decode().rstrip("=")
    assert decode_token_claims(f"a.{encoded}.b") == {}


def test_preflight_warns_for_token_without_required_graph_role() -> None:
    probe = PreflightProbe()

    def token() -> AccessToken:
        return AccessToken(
            value=SecretStr(_jwt({"roles": ["User.Read.All"]})),
            source="preflight:test",
        )

    probe.access_token = token  # type: ignore[method-assign]
    report = SharePointPreflight(
        _settings(), probe, module_finder=lambda _module: object()
    ).run()
    assert report.checks[2].status == "warn"
    assert report.passed is True
