"""Lazy Microsoft Graph transport for SharePoint files and lists."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, Protocol, Self, TypeVar, cast

from sas_migrate.application.ports import AccessToken, AccessTokenProvider
from sas_migrate.config import SharePointSettings
from sas_migrate.observability import redact_text

from .worker import SingleLoopWorker

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")

if TYPE_CHECKING:
    from azure.core.credentials import AccessToken as AzureAccessToken


class SharePointTransportError(RuntimeError):
    """A normalized configuration, authentication, or Graph failure."""


class AsyncSharePointGateway(Protocol):
    async def access_token(self) -> AccessToken: ...

    async def resolve_drive_id(self) -> str: ...

    async def list_directory(self, path: str) -> list[dict[str, Any]]: ...

    async def read_file(self, path: str) -> bytes: ...

    async def write_file(self, path: str, content: bytes) -> dict[str, Any]: ...

    async def create_directory(
        self, path: str, conflict_behavior: str
    ) -> dict[str, Any]: ...

    async def list_items(
        self,
        list_id: str,
        *,
        select: list[str] | None,
        expand: str,
        top: int | None,
        filter: str | None,
    ) -> list[dict[str, Any]]: ...

    async def get_list_item(
        self, list_id: str, item_id: str | int
    ) -> dict[str, Any]: ...

    async def update_list_item(
        self, list_id: str, item_id: str | int, fields: dict[str, Any]
    ) -> dict[str, Any]: ...

    async def close(self) -> None: ...


def _drive_item_id(path: str) -> str:
    clean = path.strip().strip("/")
    return "root" if not clean else f"root:/{clean}:"


def _drive_item(item: Any) -> dict[str, Any]:
    folder = getattr(item, "folder", None)
    modified = getattr(item, "last_modified_date_time", None)
    return {
        "name": getattr(item, "name", None),
        "id": getattr(item, "id", None),
        "is_folder": folder is not None,
        "size": getattr(item, "size", None),
        "web_url": getattr(item, "web_url", None),
        "last_modified": modified.isoformat() if modified is not None else None,
        "child_count": getattr(folder, "child_count", None),
    }


def _list_item(item: Any) -> dict[str, Any]:
    fields = getattr(item, "fields", None)
    data = getattr(fields, "additional_data", None) if fields is not None else None
    return {
        "id": getattr(item, "id", None),
        "web_url": getattr(item, "web_url", None),
        "fields": dict(data) if data else {},
    }


def _describe(exc: BaseException) -> str:
    parts: list[str] = []
    status = getattr(exc, "response_status_code", None)
    if status:
        parts.append(f"HTTP {status}")
    error = getattr(exc, "error", None)
    code = getattr(error, "code", None)
    if code:
        parts.append(str(code))
    message = str(
        getattr(error, "message", None) or getattr(exc, "message", None) or exc
    ).strip()
    if message and message not in parts:
        parts.append(redact_text(message))
    headers = getattr(exc, "response_headers", None)
    if headers is not None:
        for name in ("client-request-id", "request-id"):
            try:
                request_id = headers.get(name)
            except Exception:  # noqa: BLE001 - tolerate SDK header variants
                break
            if request_id:
                if isinstance(request_id, (set, frozenset, list, tuple)):
                    request_id = ", ".join(sorted(str(item) for item in request_id))
                parts.append(f"request-id={request_id}")
                break
    detail = "; ".join(parts)
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


class _GraphTokenCredential:
    def __init__(self, provider: AccessTokenProvider, scopes: tuple[str, ...]) -> None:
        self._provider = provider
        self._scopes = scopes

    async def get_token(
        self,
        *scopes: str,
        claims: str | None = None,
        tenant_id: str | None = None,
        enable_cae: bool = False,
        **kwargs: Any,
    ) -> AzureAccessToken:
        del claims, tenant_id, enable_cae, kwargs
        token = await self._provider.get_token(tuple(scopes) or self._scopes)
        try:
            from azure.core.credentials import AccessToken as AzureAccessToken
        except ImportError as exc:
            raise SharePointTransportError(
                "azure-core is required for SharePoint access; install "
                "'sas-parser[sharepoint]'"
            ) from exc
        expires_at = token.expires_at_epoch or int(time.time()) + 300
        return AzureAccessToken(token.value.get_secret_value(), expires_at)

    async def close(self) -> None:
        """The delegated token provider owns no Graph SDK resources."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        del exc_type, exc_value, traceback
        await self.close()


GraphClientFactory = Callable[[SharePointSettings, AccessTokenProvider], Any]


@dataclass
class _ListQueryParameters:
    """Stable Kiota query shape without generated deprecated wrappers."""

    expand: list[str] | None = None
    select: list[str] | None = None
    top: int | None = None
    filter: str | None = None

    def get_query_parameter(self, original_name: str) -> str:
        return f"%24{original_name}"


def _graph_client(
    settings: SharePointSettings, token_provider: AccessTokenProvider
) -> Any:
    try:
        import httpx
        from kiota_authentication_azure.azure_identity_authentication_provider import (
            AzureIdentityAuthenticationProvider,
        )
        from msgraph.graph_request_adapter import GraphRequestAdapter
        from msgraph.graph_service_client import GraphServiceClient
        from msgraph_core import GraphClientFactory
    except ImportError as exc:
        raise SharePointTransportError(
            "msgraph-sdk is required for SharePoint access; install "
            "'sas-parser[sharepoint]'"
        ) from exc
    credential = _GraphTokenCredential(token_provider, settings.scopes)
    auth = AzureIdentityAuthenticationProvider(
        credentials=credential,
        scopes=list(settings.scopes),
    )
    http_client = GraphClientFactory.create_with_default_middleware(
        client=httpx.AsyncClient(timeout=settings.timeout)
    )
    return GraphServiceClient(request_adapter=GraphRequestAdapter(auth, http_client))


class GraphSdkGateway:
    """Async Graph SDK operations, with SDK construction deferred to first use."""

    def __init__(
        self,
        settings: SharePointSettings,
        token_provider: AccessTokenProvider,
        *,
        client: Any | None = None,
        client_factory: GraphClientFactory = _graph_client,
    ) -> None:
        self.settings = settings
        self._token_provider = token_provider
        self._client = client
        self._client_factory = client_factory
        self._resolved_drive_id: str | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory(self.settings, self._token_provider)
        return self._client

    def _site_id(self) -> str:
        site_id = self.settings.resolved_site_id
        if not site_id:
            raise SharePointTransportError(
                "no SharePoint site configured: set site_id, or site_hostname "
                "and site_path"
            )
        return site_id

    async def access_token(self) -> AccessToken:
        return await self._token_provider.get_token(self.settings.scopes)

    async def resolve_drive_id(self) -> str:
        if self.settings.drive_id:
            return self.settings.drive_id
        if self._resolved_drive_id is None:
            site_id = self._site_id()
            drive = await self.client.sites.by_site_id(site_id).drive.get()
            drive_id = getattr(drive, "id", None)
            if not drive_id:
                raise SharePointTransportError(
                    f"site {site_id!r} has no accessible default document library"
                )
            self._resolved_drive_id = str(drive_id)
        return self._resolved_drive_id

    async def _item(self, path: str) -> Any:
        drive = self.client.drives.by_drive_id(await self.resolve_drive_id())
        return drive.items.by_drive_item_id(_drive_item_id(path))

    async def list_directory(self, path: str) -> list[dict[str, Any]]:
        builder = (await self._item(path)).children
        response = await builder.get()
        items: list[dict[str, Any]] = []
        while response is not None:
            items.extend(_drive_item(item) for item in response.value or [])
            next_link = getattr(response, "odata_next_link", None)
            response = await builder.with_url(next_link).get() if next_link else None
        return items

    async def read_file(self, path: str) -> bytes:
        content = await (await self._item(path)).content.get()
        if content is None:
            raise SharePointTransportError(f"SharePoint file {path!r} returned no content")
        return cast(bytes, content)

    async def write_file(self, path: str, content: bytes) -> dict[str, Any]:
        item = await (await self._item(path)).content.put(content)
        return _drive_item(item)

    async def create_directory(
        self, path: str, conflict_behavior: str
    ) -> dict[str, Any]:
        from msgraph.generated.models.drive_item import DriveItem
        from msgraph.generated.models.folder import Folder

        parent, _, name = path.rpartition("/")
        body = DriveItem(
            name=name,
            folder=Folder(),
            additional_data={"@microsoft.graph.conflictBehavior": conflict_behavior},
        )
        item = await (await self._item(parent)).children.post(body)
        return _drive_item(item)

    async def list_items(
        self,
        list_id: str,
        *,
        select: list[str] | None,
        expand: str,
        top: int | None,
        filter: str | None,
    ) -> list[dict[str, Any]]:
        from kiota_abstractions.base_request_configuration import RequestConfiguration

        builder = self.client.sites.by_site_id(self._site_id()).lists.by_list_id(
            list_id
        ).items
        query = _ListQueryParameters(
            expand=[expand] if expand else None,
            select=select,
            top=top,
            filter=filter,
        )
        response = await builder.get(RequestConfiguration(query_parameters=query))
        items: list[dict[str, Any]] = []
        while response is not None:
            items.extend(_list_item(item) for item in response.value or [])
            next_link = getattr(response, "odata_next_link", None)
            response = await builder.with_url(next_link).get() if next_link else None
        return items

    async def get_list_item(
        self, list_id: str, item_id: str | int
    ) -> dict[str, Any]:
        item = await (
            self.client.sites.by_site_id(self._site_id())
            .lists.by_list_id(list_id)
            .items.by_list_item_id(str(item_id))
            .get()
        )
        if item is None:
            raise SharePointTransportError(
                f"SharePoint list {list_id!r} has no item {item_id!r}"
            )
        return _list_item(item)

    async def update_list_item(
        self, list_id: str, item_id: str | int, fields: dict[str, Any]
    ) -> dict[str, Any]:
        from msgraph.generated.models.field_value_set import FieldValueSet

        updated = await (
            self.client.sites.by_site_id(self._site_id())
            .lists.by_list_id(list_id)
            .items.by_list_item_id(str(item_id))
            .fields.patch(FieldValueSet(additional_data=dict(fields)))
        )
        data = getattr(updated, "additional_data", None)
        return dict(data) if data else {}

    async def close(self) -> None:
        client = self._client
        adapter = getattr(client, "request_adapter", None) if client else None
        close = getattr(adapter, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result
        self._client = None


class SharePointGraphTransport:
    """Blocking SharePoint transport shared by conversion and XREF adapters."""

    def __init__(
        self,
        settings: SharePointSettings,
        token_provider: AccessTokenProvider,
        *,
        gateway: AsyncSharePointGateway | None = None,
        worker: SingleLoopWorker | None = None,
    ) -> None:
        self.settings = settings
        self._gateway = gateway or GraphSdkGateway(settings, token_provider)
        self._worker = worker or SingleLoopWorker()
        self._closed = False

    def _run(self, operation: str, coroutine: Coroutine[Any, Any, T]) -> T:
        try:
            return self._worker.run(coroutine)
        except SharePointTransportError:
            raise
        except Exception as exc:
            raise SharePointTransportError(
                f"SharePoint {operation} failed: {_describe(exc)}"
            ) from exc

    def access_token(self) -> AccessToken:
        return self._run("token acquisition", self._gateway.access_token())

    def resolve_drive_id(self) -> str:
        return self._run("drive resolution", self._gateway.resolve_drive_id())

    def list_directory(self, path: str = "") -> list[dict[str, Any]]:
        return self._run(
            f"directory listing for {path or '/'!r}",
            self._gateway.list_directory(path),
        )

    def list_files(
        self,
        path: str = "",
        extensions: set[str] | tuple[str, ...] | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        wanted = (
            {extension.strip().lstrip(".").casefold() for extension in extensions}
            if extensions is not None
            else None
        )
        folder = path.strip().strip("/")
        files: list[dict[str, Any]] = []
        for entry in self.list_directory(path):
            if entry.get("is_folder"):
                continue
            name = str(entry.get("name") or "")
            if wanted is not None:
                _, dot, suffix = name.rpartition(".")
                if not dot or suffix.casefold() not in wanted:
                    continue
            files.append({**entry, "path": f"{folder}/{name}" if folder else name})
        return files

    def read_file(self, path: str) -> bytes:
        return self._run(f"file read for {path!r}", self._gateway.read_file(path))

    def download_file_as_text(self, path: str, *, encoding: str = "utf-8") -> str:
        codec = "utf-8-sig" if encoding.casefold() in {"utf-8", "utf8"} else encoding
        return self.read_file(path).decode(codec, errors="replace")

    def read_json_text(self, path: str, *, encoding: str = "utf-8") -> Any:
        try:
            return json.loads(self.download_file_as_text(path, encoding=encoding))
        except json.JSONDecodeError as exc:
            raise SharePointTransportError(
                f"SharePoint file {path!r} is not valid JSON: {exc}"
            ) from exc

    def write_file(self, path: str, content: bytes | str) -> dict[str, Any]:
        clean = path.strip().strip("/")
        if not clean:
            raise SharePointTransportError(
                "write_file needs a file path, not the library root"
            )
        body = content.encode("utf-8") if isinstance(content, str) else content
        return self._run(
            f"file write for {path!r}", self._gateway.write_file(clean, body)
        )

    def upload_file(
        self, folder: str, name: str, content: bytes | str
    ) -> dict[str, Any]:
        leaf = name.strip().strip("/")
        if not leaf:
            raise SharePointTransportError("upload_file needs a file name")
        parent = folder.strip().strip("/")
        return self.write_file(f"{parent}/{leaf}" if parent else leaf, content)

    def create_directory(
        self, path: str, *, conflict_behavior: str = "fail"
    ) -> dict[str, Any]:
        clean = path.strip().strip("/")
        if not clean:
            raise SharePointTransportError(
                "create_directory needs a folder path, not the library root"
            )
        if conflict_behavior not in {"fail", "replace", "rename"}:
            raise SharePointTransportError(
                "conflict_behavior must be 'fail', 'replace', or 'rename'"
            )
        return self._run(
            f"directory creation for {path!r}",
            self._gateway.create_directory(clean, conflict_behavior),
        )

    def create_folder(self, path: str) -> dict[str, Any]:
        clean = path.strip().strip("/")
        if not clean:
            raise SharePointTransportError(
                "create_folder needs a folder path, not the library root"
            )
        result: dict[str, Any] = {}
        walked = ""
        for segment in clean.split("/"):
            if segment:
                walked = f"{walked}/{segment}" if walked else segment
                result = self.create_directory(walked, conflict_behavior="replace")
        return result

    def list_items(
        self,
        list_id: str,
        *,
        select: list[str] | None = None,
        expand: str = "fields",
        top: int | None = None,
        filter: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._run(
            f"list read for {list_id!r}",
            self._gateway.list_items(
                list_id,
                select=select,
                expand=expand,
                top=top,
                filter=filter,
            ),
        )

    def get_list_item(self, list_id: str, item_id: str | int) -> dict[str, Any]:
        return self._run(
            f"item read for {item_id!r} in {list_id!r}",
            self._gateway.get_list_item(list_id, item_id),
        )

    def update_list_item(
        self, list_id: str, item_id: str | int, fields: dict[str, Any]
    ) -> dict[str, Any]:
        if not fields:
            raise SharePointTransportError("update_list_item requires at least one field")
        return self._run(
            f"item update for {item_id!r} in {list_id!r}",
            self._gateway.update_list_item(list_id, item_id, fields),
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._run("client close", self._gateway.close())
        finally:
            self._closed = True
            self._worker.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "AsyncSharePointGateway",
    "GraphSdkGateway",
    "SharePointGraphTransport",
    "SharePointTransportError",
]
