"""
Tests for the app_config.sharepoint Microsoft Graph client.

No live SharePoint (and no msgraph-sdk install) is needed: settings are
resolved from a controlled environment + tmp config.json, and the Graph
operations are exercised through an injected fake GraphServiceClient that mimics
the fluent request-builder chain. Each test isolates SAS_PARSER_CONFIG, the
SHAREPOINT_* and AZURE_* env vars, and clears the app_config file cache plus the
sharepoint client cache around itself.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

import app_config
from app_config import azure, sharepoint

# msgraph-sdk is the optional 'sharepoint' extra: the CI test job installs only
# --extra dev, so it is absent there. Most tests inject a fake GraphServiceClient
# and never touch it, but a few operations import SDK model / request-config
# classes lazily to build request bodies; those tests skip when it is missing
# (the types job type-checks the module with the extra installed instead).
requires_msgraph = pytest.mark.skipif(
    importlib.util.find_spec("msgraph") is None,
    reason="msgraph-sdk (the 'sharepoint' extra) is not installed",
)

_SHAREPOINT_ENV = (
    "SHAREPOINT_SITE_HOSTNAME",
    "SHAREPOINT_SITE_PATH",
    "SHAREPOINT_SITE_ID",
    "SHAREPOINT_DRIVE_ID",
    "SHAREPOINT_SCOPES",
    "SHAREPOINT_FILE_SERVER_BASE_PATH",
    "SHAREPOINT_SECRET_SCOPE",
    "SHAREPOINT_TENANT_ID_KEY",
    "SHAREPOINT_CLIENT_ID_KEY",
    "SHAREPOINT_CLIENT_SECRET_KEY",
    "SHAREPOINT_LIST_ID_SAS_REQUESTS",
    "SHAREPOINT_LIST_ID_SAS_CONVERSIONS",
    "SHAREPOINT_LIST_ID_XREF",
    "SHAREPOINT_LIST_ID_SAS_COMPLEXITY",
    # The SharePoint secret scope defaults across from the workspace's.
    "DATABRICKS_SECRET_SCOPE",
)

_AZURE_ENV = (
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
    for var in _SHAREPOINT_ENV + _AZURE_ENV:
        monkeypatch.delenv(var, raising=False)
    app_config.clear_cache()
    sharepoint.clear_cache()
    azure.clear_cache()
    yield cfg
    app_config.clear_cache()
    sharepoint.clear_cache()
    azure.clear_cache()


def _set(cfg_path, mapping) -> None:
    cfg_path.write_text(json.dumps(mapping), encoding="utf-8")
    app_config.clear_cache()


def _service_principal(monkeypatch) -> None:
    """The env of a workspace reached with an Entra ID service principal."""
    monkeypatch.setenv("AZURE_TENANT_ID", "t-1")
    monkeypatch.setenv("AZURE_CLIENT_ID", "c-1")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "s-1")


# ---------------------------------------------------------------------------
# Fake Graph SDK — just enough of the fluent chain to drive the client
# ---------------------------------------------------------------------------


class _Awaitable:
    """A coroutine stand-in returning a fixed value, for the SDK's `await`.

    Provides ``close()`` so it behaves like the real coroutines the client's
    ``_run`` may discard (e.g. when refusing to nest in a running loop).
    """

    def __init__(self, value):
        self._value = value

    def __await__(self):
        async def _coro():
            return self._value

        return _coro().__await__()

    def close(self):
        pass


class _Recorder:
    """Records calls and returns a preset value as an awaitable."""

    def __init__(self, value=None):
        self.value = value
        self.calls: list[Any] = []

    def __call__(self, *args):
        self.calls.append(args)
        return _Awaitable(self.value)


class _Folder:
    def __init__(self, child_count=0):
        self.child_count = child_count


class _DriveItem:
    def __init__(self, name=None, id=None, folder=None, size=None, web_url=None):
        self.name = name
        self.id = id
        self.folder = folder
        self.size = size
        self.web_url = web_url
        self.last_modified_date_time = None


class _Collection:
    def __init__(self, value, next_link=None):
        self.value = value
        self.odata_next_link = next_link


class _ContentBuilder:
    def __init__(self, get_value=None, put_value=None):
        self.get = _Recorder(get_value)
        # put(body) records the uploaded body
        self.put = _Recorder(put_value)


class _ChildrenBuilder:
    def __init__(self, pages):
        # pages: list of _Collection to hand out in order across with_url()
        self._pages = list(pages)
        self._i = 0
        self.get = self._get
        self.posted: list[Any] = []

    def _get(self, config=None):
        page = self._pages[self._i]
        return _Awaitable(page)

    def with_url(self, url):
        self._i += 1
        return self

    def post(self, body):
        self.posted.append(body)
        return _Awaitable(body)


class _ItemBuilder:
    def __init__(self, *, content=None, children=None):
        self.content = content or _ContentBuilder()
        self.children = children or _ChildrenBuilder([_Collection([])])


class _ItemsBuilder:
    def __init__(self, item):
        self._item = item
        self.requested_ids: list[str] = []

    def by_drive_item_id(self, item_id):
        self.requested_ids.append(item_id)
        return self._item


class _DriveBuilder:
    def __init__(self, item):
        self.items = _ItemsBuilder(item)


class _DrivesBuilder:
    # A single persistent drive/items builder so recorded ids accumulate across
    # calls on the instance the assertions read.
    def __init__(self, item):
        self.drive_builder = _DriveBuilder(item)
        self.requested_ids: list[str] = []

    def by_drive_id(self, drive_id):
        self.requested_ids.append(drive_id)
        return self.drive_builder


class _ListItem:
    """One list item as the SDK returns it, with a patchable `fields`."""

    def __init__(self, item_id, values):
        self.id = item_id
        self.web_url = f"https://sp.example/items/{item_id}"
        fields: Any = type("Fields", (), {})()
        fields.additional_data = dict(values)
        self.fields = fields


class _ItemFieldsBuilder:
    def __init__(self, item):
        self._item = item
        self.patched: list[dict] = []

    def patch(self, body):
        self.patched.append(dict(body.additional_data))
        self._item.fields.additional_data.update(body.additional_data)
        result: Any = type("Fields", (), {})()
        result.additional_data = dict(self._item.fields.additional_data)
        return _Awaitable(result)


class _SingleItemBuilder:
    def __init__(self, item):
        self._item = item
        self.fields = _ItemFieldsBuilder(item)

    def get(self, config=None):
        return _Awaitable(self._item)


class _ListItemsBuilder:
    def __init__(self, pages, by_id=None):
        self._pages = list(pages)
        self._i = 0
        self.configs: list[Any] = []
        # item id -> _ListItem, for by_list_item_id()
        self._by_id = dict(by_id or {})
        self.item_builders: dict[str, _SingleItemBuilder] = {}
        self.requested_item_ids: list[str] = []

    def get(self, config=None):
        self.configs.append(config)
        return _Awaitable(self._pages[self._i])

    def with_url(self, url):
        self._i += 1
        return self

    def by_list_item_id(self, item_id):
        self.requested_item_ids.append(item_id)
        if item_id not in self._by_id:
            raise RuntimeError(f"itemNotFound: {item_id}")
        builder = self.item_builders.get(item_id)
        if builder is None:
            builder = _SingleItemBuilder(self._by_id[item_id])
            self.item_builders[item_id] = builder
        return builder


class _ListBuilder:
    def __init__(self, items_builder):
        self.items = items_builder


class _ListsBuilder:
    def __init__(self, items_builder):
        self._items_builder = items_builder
        self.requested_ids: list[str] = []

    def by_list_id(self, list_id):
        self.requested_ids.append(list_id)
        return _ListBuilder(self._items_builder)


class _SiteDriveBuilder:
    def __init__(self, drive_item):
        self.get = _Recorder(drive_item)


class _SiteBuilder:
    def __init__(self, drive_item=None, list_items_builder=None):
        self.drive = _SiteDriveBuilder(drive_item)
        self.lists = _ListsBuilder(list_items_builder)


class _SitesBuilder:
    def __init__(self, site_builder):
        self._site_builder = site_builder
        self.requested_ids: list[str] = []

    def by_site_id(self, site_id):
        self.requested_ids.append(site_id)
        return self._site_builder


class _FakeGraphClient:
    def __init__(self, *, item=None, site=None):
        self.drives = _DrivesBuilder(item or _ItemBuilder())
        self.sites = _SitesBuilder(site or _SiteBuilder())


def _client(config=None, **kwargs):
    """A SharePointClient over a fake Graph client with a drive_id set."""
    cfg = config or sharepoint.SharePointConfig(drive_id="DRV")
    fake = _FakeGraphClient(**kwargs)
    return sharepoint.SharePointClient(cfg, client=fake), fake


# ---------------------------------------------------------------------------
# SharePointConfig resolution
# ---------------------------------------------------------------------------


def test_from_env_reads_env_first(monkeypatch, _isolated):
    monkeypatch.setenv("SHAREPOINT_SITE_HOSTNAME", "contoso.sharepoint.com")
    monkeypatch.setenv("SHAREPOINT_SITE_PATH", "/sites/Eng")
    monkeypatch.setenv("SHAREPOINT_DRIVE_ID", "DRV-1")
    cfg = sharepoint.SharePointConfig.from_env()
    assert cfg.site_hostname == "contoso.sharepoint.com"
    assert cfg.site_path == "/sites/Eng"
    assert cfg.drive_id == "DRV-1"
    assert cfg.scopes == (sharepoint.GRAPH_DEFAULT_SCOPE,)


def test_from_env_falls_back_to_config_json(_isolated):
    _set(
        _isolated,
        {
            "sharepoint": {
                "site_hostname": "cfg.sharepoint.com",
                "site_path": "sites/Cfg",
                "timeout": 5,
            }
        },
    )
    cfg = sharepoint.SharePointConfig.from_env()
    assert cfg.site_hostname == "cfg.sharepoint.com"
    assert cfg.site_path == "/sites/Cfg"  # normalised with a leading slash
    assert cfg.timeout == 5


def test_env_beats_config(monkeypatch, _isolated):
    _set(_isolated, {"sharepoint": {"site_hostname": "cfg.sharepoint.com"}})
    monkeypatch.setenv("SHAREPOINT_SITE_HOSTNAME", "env.sharepoint.com")
    assert (
        sharepoint.SharePointConfig.from_env().site_hostname == "env.sharepoint.com"
    )


def test_defaults_without_env_or_config(_isolated):
    cfg = sharepoint.SharePointConfig.from_env()
    assert cfg.site_hostname is None and cfg.site_path is None
    assert cfg.site_id is None and cfg.drive_id is None
    assert cfg.resolved_site_id is None
    assert cfg.scopes == (sharepoint.GRAPH_DEFAULT_SCOPE,)
    assert cfg.timeout == sharepoint.DEFAULT_TIMEOUT


def test_wrong_typed_timeout_degrades(_isolated):
    _set(_isolated, {"sharepoint": {"timeout": "slow"}})
    assert (
        sharepoint.SharePointConfig.from_env().timeout == sharepoint.DEFAULT_TIMEOUT
    )


def test_scopes_from_env_space_or_comma_separated(monkeypatch, _isolated):
    monkeypatch.setenv("SHAREPOINT_SCOPES", "api://x/.default, api://y/.default")
    assert sharepoint.SharePointConfig.from_env().scopes == (
        "api://x/.default",
        "api://y/.default",
    )


def test_wrong_typed_scopes_degrade_to_graph_default(_isolated):
    _set(_isolated, {"sharepoint": {"scopes": "not-a-list"}})
    assert sharepoint.SharePointConfig.from_env().scopes == (
        sharepoint.GRAPH_DEFAULT_SCOPE,
    )


# ---------------------------------------------------------------------------
# resolved_site_id
# ---------------------------------------------------------------------------


def test_resolved_site_id_from_hostname_and_path():
    cfg = sharepoint.SharePointConfig(
        site_hostname="contoso.sharepoint.com", site_path="/sites/Eng"
    )
    assert cfg.resolved_site_id == "contoso.sharepoint.com:/sites/Eng"


def test_explicit_site_id_beats_hostname():
    cfg = sharepoint.SharePointConfig(
        site_id="explicit-id",
        site_hostname="contoso.sharepoint.com",
        site_path="/sites/Eng",
    )
    assert cfg.resolved_site_id == "explicit-id"


def test_hostname_without_path_has_no_site_id():
    assert (
        sharepoint.SharePointConfig(site_hostname="contoso.sharepoint.com").resolved_site_id
        is None
    )


# ---------------------------------------------------------------------------
# Path -> drive-item id addressing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path, expected",
    [
        ("", "root"),
        ("/", "root"),
        ("   ", "root"),
        ("Reports", "root:/Reports:"),
        ("/Reports/2024/", "root:/Reports/2024:"),
        ("Reports/q1.txt", "root:/Reports/q1.txt:"),
    ],
)
def test_drive_item_id(path, expected):
    assert sharepoint._drive_item_id(path) == expected


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------


def test_list_directory_returns_flattened_items():
    children = _ChildrenBuilder(
        [
            _Collection(
                [
                    _DriveItem(name="a.txt", id="1", size=10, web_url="u1"),
                    _DriveItem(name="sub", id="2", folder=_Folder(child_count=3)),
                ]
            )
        ]
    )
    item = _ItemBuilder(children=children)
    client, fake = _client(item=item)
    entries = client.list_directory("Reports")
    assert [(e["name"], e["is_folder"], e["child_count"]) for e in entries] == [
        ("a.txt", False, None),
        ("sub", True, 3),
    ]
    # Addressed the drive and the folder by its path alias.
    assert fake.drives.requested_ids == ["DRV"]
    assert fake.drives.drive_builder.items.requested_ids == ["root:/Reports:"]


def test_list_directory_follows_paging():
    children = _ChildrenBuilder(
        [
            _Collection([_DriveItem(name="a")], next_link="https://next"),
            _Collection([_DriveItem(name="b")]),
        ]
    )
    client, _ = _client(item=_ItemBuilder(children=children))
    names = [e["name"] for e in client.list_directory()]
    assert names == ["a", "b"]


def test_list_directory_addresses_root_by_default():
    item = _ItemBuilder(children=_ChildrenBuilder([_Collection([])]))
    client, fake = _client(item=item)
    client.list_directory()
    assert fake.drives.drive_builder.items.requested_ids == ["root"]


def test_list_directory_wraps_errors():
    class _Boom(_ChildrenBuilder):
        def _get(self, config=None):
            raise OSError("network down")

    client, _ = _client(item=_ItemBuilder(children=_Boom([_Collection([])])))
    with pytest.raises(sharepoint.SharePointError, match="could not list SharePoint"):
        client.list_directory("Reports")


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


def test_read_file_returns_bytes():
    item = _ItemBuilder(content=_ContentBuilder(get_value=b"hello"))
    client, fake = _client(item=item)
    assert client.read_file("Reports/a.txt") == b"hello"
    assert fake.drives.drive_builder.items.requested_ids == ["root:/Reports/a.txt:"]


def test_read_file_missing_content_raises():
    item = _ItemBuilder(content=_ContentBuilder(get_value=None))
    client, _ = _client(item=item)
    with pytest.raises(sharepoint.SharePointError, match="returned no content"):
        client.read_file("Reports/a.txt")


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


def test_write_file_encodes_str_and_uploads():
    uploaded = _DriveItem(name="a.txt", id="9", web_url="u")
    content = _ContentBuilder(put_value=uploaded)
    item = _ItemBuilder(content=content)
    client, fake = _client(item=item)
    result = client.write_file("Reports/a.txt", "héllo")
    assert result["id"] == "9" and result["name"] == "a.txt"
    # str body was UTF-8 encoded to bytes.
    assert content.put.calls == [("héllo".encode("utf-8"),)]
    assert fake.drives.drive_builder.items.requested_ids == ["root:/Reports/a.txt:"]


def test_write_file_passes_bytes_through():
    content = _ContentBuilder(put_value=_DriveItem(name="a"))
    client, _ = _client(item=_ItemBuilder(content=content))
    client.write_file("a.bin", b"\x00\x01")
    assert content.put.calls == [(b"\x00\x01",)]


def test_write_file_rejects_root():
    client, _ = _client()
    with pytest.raises(sharepoint.SharePointError, match="needs a file path"):
        client.write_file("/", "x")


# ---------------------------------------------------------------------------
# create_directory
# ---------------------------------------------------------------------------


@requires_msgraph
def test_create_directory_posts_a_folder():
    children = _ChildrenBuilder([_Collection([])])
    item = _ItemBuilder(children=children)
    client, fake = _client(item=item)
    client.create_directory("Reports/2024")
    # Parent addressed by path; a Folder DriveItem posted with the leaf name.
    assert fake.drives.drive_builder.items.requested_ids == ["root:/Reports:"]
    (body,) = children.posted
    assert body.name == "2024"
    assert body.folder is not None
    assert body.additional_data["@microsoft.graph.conflictBehavior"] == "fail"


@requires_msgraph
def test_create_directory_at_root_uses_root_parent():
    children = _ChildrenBuilder([_Collection([])])
    item = _ItemBuilder(children=children)
    client, fake = _client(item=item)
    client.create_directory("Reports", conflict_behavior="replace")
    assert fake.drives.drive_builder.items.requested_ids == ["root"]
    (body,) = children.posted
    assert body.name == "Reports"
    assert body.additional_data["@microsoft.graph.conflictBehavior"] == "replace"


def test_create_directory_rejects_root():
    client, _ = _client()
    with pytest.raises(sharepoint.SharePointError, match="needs a folder path"):
        client.create_directory("")


# ---------------------------------------------------------------------------
# read_list_items (PowerApps)
# ---------------------------------------------------------------------------


def _site_client(config=None, pages=None):
    items_builder = _ListItemsBuilder(pages or [_Collection([])])
    site = _SiteBuilder(list_items_builder=items_builder)
    cfg = config or sharepoint.SharePointConfig(site_id="SITE")
    fake = _FakeGraphClient(site=site)
    return sharepoint.SharePointClient(cfg, client=fake), fake, items_builder


class _ListItem:
    def __init__(self, id, fields):
        self.id = id
        self.web_url = f"u/{id}"

        class _F:
            additional_data = fields

        self.fields = _F()


@requires_msgraph
def test_read_list_items_flattens_fields():
    pages = [
        _Collection(
            [
                _ListItem("1", {"Title": "Task A", "Status": "Open"}),
                _ListItem("2", {"Title": "Task B", "Status": "Done"}),
            ]
        )
    ]
    client, fake, _ = _site_client(pages=pages)
    rows = client.read_list_items("Tasks")
    assert rows == [
        {"id": "1", "web_url": "u/1", "fields": {"Title": "Task A", "Status": "Open"}},
        {"id": "2", "web_url": "u/2", "fields": {"Title": "Task B", "Status": "Done"}},
    ]
    assert fake.sites.requested_ids == ["SITE"]
    assert fake.sites._site_builder.lists.requested_ids == ["Tasks"]


@requires_msgraph
def test_read_list_items_expands_fields_by_default():
    client, _, items_builder = _site_client()
    client.read_list_items("Tasks")
    (config,) = items_builder.configs
    assert config.query_parameters.expand == ["fields"]


@requires_msgraph
def test_read_list_items_follows_paging():
    pages = [
        _Collection([_ListItem("1", {"Title": "A"})], next_link="https://next"),
        _Collection([_ListItem("2", {"Title": "B"})]),
    ]
    client, _, _ = _site_client(pages=pages)
    rows = client.read_list_items("Tasks")
    assert [r["id"] for r in rows] == ["1", "2"]


def test_read_list_items_without_site_raises():
    cfg = sharepoint.SharePointConfig(drive_id="DRV")  # a drive, but no site
    client = sharepoint.SharePointClient(cfg, client=_FakeGraphClient())
    with pytest.raises(sharepoint.SharePointError, match="no SharePoint site"):
        client.read_list_items("Tasks")


# ---------------------------------------------------------------------------
# Drive resolution from the site's default library
# ---------------------------------------------------------------------------


def test_drive_id_resolved_from_site_default_library():
    default_drive = _DriveItem(id="SITE-DRV")
    site = _SiteBuilder(drive_item=default_drive)
    item = _ItemBuilder(content=_ContentBuilder(get_value=b"x"))
    fake = _FakeGraphClient(item=item, site=site)
    cfg = sharepoint.SharePointConfig(site_id="SITE")  # no explicit drive_id
    client = sharepoint.SharePointClient(cfg, client=fake)
    client.read_file("a.txt")
    # The site's default drive id was resolved and then used to address the drive.
    assert fake.sites.requested_ids == ["SITE"]
    assert fake.drives.requested_ids == ["SITE-DRV"]


def test_drive_resolution_is_cached():
    default_drive = _DriveItem(id="SITE-DRV")
    site = _SiteBuilder(drive_item=default_drive)
    item = _ItemBuilder(content=_ContentBuilder(get_value=b"x"))
    fake = _FakeGraphClient(item=item, site=site)
    client = sharepoint.SharePointClient(
        sharepoint.SharePointConfig(site_id="SITE"), client=fake
    )
    client.read_file("a.txt")
    client.read_file("b.txt")
    # Site drive resolved once, not per call.
    assert site.drive.get.calls == [()]


def test_no_default_library_raises():
    site = _SiteBuilder(drive_item=_DriveItem(id=None))
    fake = _FakeGraphClient(site=site)
    client = sharepoint.SharePointClient(
        sharepoint.SharePointConfig(site_id="SITE"), client=fake
    )
    with pytest.raises(sharepoint.SharePointError, match="no accessible default"):
        client.read_file("a.txt")


# ---------------------------------------------------------------------------
# get_token (authentication)
# ---------------------------------------------------------------------------


def test_get_token_requests_the_graph_scope(monkeypatch, _isolated):
    _service_principal(monkeypatch)
    asked: list = []

    def _fake_get_token(scopes=None):
        asked.append(scopes)
        return "graph-token"

    monkeypatch.setattr(azure, "get_token", _fake_get_token)
    client = sharepoint.SharePointClient(sharepoint.SharePointConfig())
    assert client.get_token() == "graph-token"
    assert asked == [(sharepoint.GRAPH_DEFAULT_SCOPE,)]


def test_get_token_wraps_azure_errors(monkeypatch, _isolated):
    def _boom(scopes=None):
        raise azure.AzureAuthError("bad secret")

    monkeypatch.setattr(azure, "get_token", _boom)
    client = sharepoint.SharePointClient(sharepoint.SharePointConfig())
    with pytest.raises(sharepoint.SharePointError, match="could not mint a Microsoft Graph"):
        client.get_token()


# ---------------------------------------------------------------------------
# Token credential adapter
# ---------------------------------------------------------------------------


def test_credential_wraps_token_in_access_token():
    pytest.importorskip("azure.core.credentials", reason="azure-core not installed")
    cred = sharepoint._GraphTokenCredential(
        lambda scopes: f"tok-for-{','.join(scopes)}", (sharepoint.GRAPH_DEFAULT_SCOPE,)
    )
    token = cred.get_token("https://graph.microsoft.com/.default")
    assert token.token == "tok-for-https://graph.microsoft.com/.default"
    assert token.expires_on > 0


def test_credential_falls_back_to_default_scopes():
    pytest.importorskip("azure.core.credentials", reason="azure-core not installed")
    cred = sharepoint._GraphTokenCredential(
        lambda scopes: "|".join(scopes), ("api://default/.default",)
    )
    assert cred.get_token().token == "api://default/.default"


# ---------------------------------------------------------------------------
# _build_client validation (runs before the SDK import)
# ---------------------------------------------------------------------------


def test_build_client_without_identity_raises(_isolated):
    # No AZURE_* env: there is no service principal to mint a Graph token with.
    client = sharepoint.SharePointClient(sharepoint.SharePointConfig(drive_id="DRV"))
    with pytest.raises(sharepoint.SharePointError, match="no Entra ID identity"):
        _ = client.client


def test_missing_sdk_raises_helpful_error(monkeypatch, _isolated):
    # msgraph-sdk is an optional extra; when it is not installed a
    # fully-configured client still fails at import with an install hint.
    try:
        import msgraph  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("msgraph-sdk is installed; the import path is unreachable")
    _service_principal(monkeypatch)
    client = sharepoint.SharePointClient(sharepoint.SharePointConfig(drive_id="DRV"))
    with pytest.raises(sharepoint.SharePointError, match="msgraph-sdk is required"):
        _ = client.client


# ---------------------------------------------------------------------------
# Synchronous facade guardrail
# ---------------------------------------------------------------------------


def test_calling_from_a_running_loop_raises():
    client, _ = _client(item=_ItemBuilder(content=_ContentBuilder(get_value=b"x")))

    async def _inside():
        # Inside a running loop the blocking facade must refuse rather than
        # nest and strand the httpx pool on another loop.
        client.read_file("a.txt")

    with pytest.raises(sharepoint.SharePointError, match="running event loop"):
        asyncio.run(_inside())


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------


def test_get_sharepoint_client_is_cached():
    first = sharepoint.get_sharepoint_client()
    assert sharepoint.get_sharepoint_client() is first
    sharepoint.clear_cache()
    assert sharepoint.get_sharepoint_client() is not first


# ---------------------------------------------------------------------------
# Base-path normalisation and drive_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Shared Documents/Kit/Applications", "Kit/Applications"),
        ("/Shared Documents/Kit/Applications", "Kit/Applications"),
        ("shared documents/Kit/Applications", "Kit/Applications"),
        ("SHARED DOCUMENTS/Kit/Applications", "Kit/Applications"),
        ("Kit/Applications", "Kit/Applications"),
        ("/Kit/Applications/", "Kit/Applications"),
        ("Shared Documents/", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_base_path_is_normalised_to_drive_relative(raw, expected, _isolated):
    # Graph addresses items INSIDE the library, so the "Shared Documents/"
    # prefix a SharePoint URL shows must not survive into a lookup.
    _set(_isolated, {"sharepoint": {"file_server_base_path": raw}})
    assert sharepoint.SharePointConfig.from_env().file_server_base_path == expected


def test_base_path_from_env_beats_config(monkeypatch, _isolated):
    _set(_isolated, {"sharepoint": {"file_server_base_path": "Cfg/Apps"}})
    monkeypatch.setenv(
        "SHAREPOINT_FILE_SERVER_BASE_PATH", "Shared Documents/Env/Apps"
    )
    assert sharepoint.SharePointConfig.from_env().file_server_base_path == "Env/Apps"


def test_drive_path_joins_onto_the_base():
    cfg = sharepoint.SharePointConfig(file_server_base_path="Kit/Applications")
    assert cfg.drive_path("MyApp", "scripts_original") == (
        "Kit/Applications/MyApp/scripts_original"
    )
    # Stray slashes on either side are absorbed, so callers need not care.
    assert cfg.drive_path("/MyApp/", "/scripts_original/") == (
        "Kit/Applications/MyApp/scripts_original"
    )
    assert cfg.drive_path("MyApp/scripts_original") == (
        "Kit/Applications/MyApp/scripts_original"
    )


def test_drive_path_with_no_base_is_library_relative():
    cfg = sharepoint.SharePointConfig()
    assert cfg.drive_path("MyApp") == "MyApp"
    assert cfg.drive_path() == ""


# ---------------------------------------------------------------------------
# List ids
# ---------------------------------------------------------------------------


def test_list_ids_read_from_config(_isolated):
    _set(
        _isolated,
        {
            "sharepoint": {
                "list_id_sas_requests": "L-req",
                "list_id_sas_conversions": "L-conv",
                "list_id_xref": "L-xref",
                "list_id_sas_complexity": "L-cx",
            }
        },
    )
    cfg = sharepoint.SharePointConfig.from_env()
    assert cfg.list_id("requests") == "L-req"
    assert cfg.list_id("conversions") == "L-conv"
    assert cfg.list_id("xref") == "L-xref"
    assert cfg.list_id("complexity") == "L-cx"


def test_list_id_env_beats_config(monkeypatch, _isolated):
    _set(_isolated, {"sharepoint": {"list_id_xref": "cfg"}})
    monkeypatch.setenv("SHAREPOINT_LIST_ID_XREF", "env")
    assert sharepoint.SharePointConfig.from_env().list_id("xref") == "env"


def test_missing_list_id_names_the_config_key():
    # Better than Graph reporting that None is not a list.
    cfg = sharepoint.SharePointConfig()
    with pytest.raises(sharepoint.SharePointError, match="sharepoint.list_id_xref"):
        cfg.list_id("xref")


def test_unknown_list_kind_raises():
    with pytest.raises(sharepoint.SharePointError, match="unknown SharePoint list"):
        sharepoint.SharePointConfig().list_id("nope")


# ---------------------------------------------------------------------------
# SharePoint's own service principal
# ---------------------------------------------------------------------------


def test_secret_keys_default_to_the_sharepoint_principal(_isolated):
    cfg = sharepoint.SharePointConfig.from_env()
    assert cfg.secret_keys.tenant_id_key == "saact-hsv-tenantid"
    assert cfg.secret_keys.client_id_key == "saact-hsv-appid"
    assert cfg.secret_keys.client_secret_key == "saact-hsv-secret"


def test_secret_scope_defaults_across_from_databricks(monkeypatch, _isolated):
    # Both principals live in one scope, under different keys.
    monkeypatch.setenv("DATABRICKS_SECRET_SCOPE", "workspace-scope")
    assert sharepoint.SharePointConfig.from_env().secret_scope == "workspace-scope"


def test_sharepoint_secret_scope_beats_the_databricks_one(monkeypatch, _isolated):
    monkeypatch.setenv("DATABRICKS_SECRET_SCOPE", "workspace-scope")
    monkeypatch.setenv("SHAREPOINT_SECRET_SCOPE", "sharepoint-scope")
    assert sharepoint.SharePointConfig.from_env().secret_scope == "sharepoint-scope"


def test_token_is_minted_as_the_sharepoint_principal(monkeypatch, _isolated):
    calls: list[tuple] = []

    class _FakeClient:
        def get_token(self, scopes):
            calls.append(("token", tuple(scopes)))
            return "graph-token"

    def _fake_get_databricks_client(secret_scope=None, keys=None):
        calls.append(("client", secret_scope, keys))
        return _FakeClient()

    monkeypatch.setattr(azure, "get_databricks_client", _fake_get_databricks_client)
    cfg = sharepoint.SharePointConfig(secret_scope="kv")
    client = sharepoint.SharePointClient(cfg, client=_FakeGraphClient())

    assert client.get_token() == "graph-token"
    # The saact-hsv-* key set reaches the secret read — not the sp-hsv-* one
    # that authenticates Vault.
    assert calls[0][1] == "kv"
    assert calls[0][2].client_id_key == "saact-hsv-appid"
    assert calls[1] == ("token", (sharepoint.GRAPH_DEFAULT_SCOPE,))


def test_token_falls_back_to_the_shared_identity(monkeypatch, _isolated):
    seen: list[tuple] = []
    monkeypatch.setattr(
        azure, "get_token", lambda scopes=None: seen.append(scopes) or "shared-token"
    )
    client = sharepoint.SharePointClient(
        sharepoint.SharePointConfig(), client=_FakeGraphClient()
    )

    assert client.get_token() == "shared-token"
    assert seen == [(sharepoint.GRAPH_DEFAULT_SCOPE,)]


# ---------------------------------------------------------------------------
# File primitives: list_files, download_file_as_text, read_json_text,
# upload_file, create_folder
# ---------------------------------------------------------------------------


def _folder_listing(*entries):
    """A children builder serving one page of (name, is_folder) entries."""
    items = [
        _DriveItem(name=name, id=name, folder=_Folder() if is_folder else None)
        for name, is_folder in entries
    ]
    return _ChildrenBuilder([_Collection(items)])


def test_list_files_drops_folders():
    children = _folder_listing(("etl.sas", False), ("archive", True))
    client, _ = _client(item=_ItemBuilder(children=children))

    names = [entry["name"] for entry in client.list_files("MyApp")]
    assert names == ["etl.sas"]


def test_list_files_filters_by_extension():
    children = _folder_listing(
        ("etl.sas", False),
        ("notes.TXT", False),
        ("report.pdf", False),
        ("no_extension", False),
    )
    client, _ = _client(item=_ItemBuilder(children=children))

    # Case-insensitive, and a leading dot is optional.
    found = [entry["name"] for entry in client.list_files("MyApp", ("sas", ".txt"))]
    assert found == ["etl.sas", "notes.TXT"]


def test_list_files_without_a_filter_keeps_every_file():
    children = _folder_listing(("a.sas", False), ("b.pdf", False), ("sub", True))
    client, _ = _client(item=_ItemBuilder(children=children))

    assert len(client.list_files("MyApp")) == 2


def test_list_files_carries_the_drive_relative_path():
    children = _folder_listing(("etl.sas", False))
    client, _ = _client(item=_ItemBuilder(children=children))

    assert client.list_files("Kit/MyApp")[0]["path"] == "Kit/MyApp/etl.sas"
    # At the library root there is no prefix to add.
    assert client.list_files("")[0]["path"] == "etl.sas"


def test_download_file_as_text_decodes():
    content = _ContentBuilder(get_value="data a; run;".encode("utf-8"))
    client, _ = _client(item=_ItemBuilder(content=content))

    assert client.download_file_as_text("a.sas") == "data a; run;"


def test_download_file_as_text_strips_a_bom():
    # Windows editors add one, and a leading ﻿ would break the first
    # statement of every file that has it.
    content = _ContentBuilder(get_value="﻿data a; run;".encode("utf-8"))
    client, _ = _client(item=_ItemBuilder(content=content))

    assert client.download_file_as_text("a.sas") == "data a; run;"


def test_download_file_as_text_survives_undecodable_bytes():
    # One stray byte in a decades-old SAS file must not lose the whole file.
    content = _ContentBuilder(get_value=b"data a; \xff run;")
    client, _ = _client(item=_ItemBuilder(content=content))

    assert client.download_file_as_text("a.sas").startswith("data a; ")


def test_read_json_text_parses():
    content = _ContentBuilder(get_value=b'{"a": [1, 2]}')
    client, _ = _client(item=_ItemBuilder(content=content))

    assert client.read_json_text("x.json") == {"a": [1, 2]}


def test_read_json_text_names_the_file_on_bad_json():
    content = _ContentBuilder(get_value=b"not json")
    client, _ = _client(item=_ItemBuilder(content=content))

    with pytest.raises(sharepoint.SharePointError, match="x.json"):
        client.read_json_text("x.json")


def test_upload_file_composes_the_same_path_as_write_file():
    item = _ItemBuilder(content=_ContentBuilder(put_value=_DriveItem(name="a.sas")))
    client, fake = _client(item=item)

    client.upload_file("Kit/MyApp", "a.sas", "data a; run;")
    by_folder = list(fake.drives.drive_builder.items.requested_ids)

    client2, fake2 = _client(item=item)
    client2.write_file("Kit/MyApp/a.sas", "data a; run;")
    assert by_folder == list(fake2.drives.drive_builder.items.requested_ids)
    assert by_folder == ["root:/Kit/MyApp/a.sas:"]


def test_upload_file_at_the_library_root():
    item = _ItemBuilder(content=_ContentBuilder(put_value=_DriveItem(name="a.sas")))
    client, fake = _client(item=item)

    client.upload_file("", "a.sas", "x")
    assert fake.drives.drive_builder.items.requested_ids == ["root:/a.sas:"]


def test_upload_file_needs_a_name():
    client, _ = _client()
    with pytest.raises(sharepoint.SharePointError, match="needs a file name"):
        client.upload_file("Kit", "  ", "x")


@requires_msgraph
def test_create_folder_is_idempotent():
    # The upload flows call it unconditionally before every upload, so an
    # existing folder is a success, not a conflict.
    children = _ChildrenBuilder([_Collection([])])
    client, _ = _client(item=_ItemBuilder(children=children))

    client.create_folder("MyApp/complexity")
    client.create_folder("MyApp/complexity")

    behaviours = [
        body.additional_data["@microsoft.graph.conflictBehavior"]
        for body in children.posted
    ]
    assert behaviours and set(behaviours) == {"replace"}


@requires_msgraph
def test_create_folder_creates_missing_parents():
    children = _ChildrenBuilder([_Collection([])])
    client, _ = _client(item=_ItemBuilder(children=children))

    client.create_folder("MyApp/complexity/2026-08-04")

    assert [body.name for body in children.posted] == [
        "MyApp",
        "complexity",
        "2026-08-04",
    ]


def test_create_folder_refuses_the_library_root():
    client, _ = _client()
    with pytest.raises(sharepoint.SharePointError, match="not the library root"):
        client.create_folder("/")


# ---------------------------------------------------------------------------
# List-item primitives: get_list_item, update_list_item
# ---------------------------------------------------------------------------


def _list_client(items):
    """A client over a site whose one list serves *items* ({id: fields})."""
    by_id = {str(k): _ListItem(str(k), v) for k, v in items.items()}
    items_builder = _ListItemsBuilder(
        [_Collection(list(by_id.values()))], by_id=by_id
    )
    site = _SiteBuilder(list_items_builder=items_builder)
    cfg = sharepoint.SharePointConfig(site_id="SITE", drive_id="DRV")
    client = sharepoint.SharePointClient(cfg, client=_FakeGraphClient(site=site))
    return client, items_builder


@requires_msgraph
def test_list_items_is_read_list_items():
    client, _ = _list_client({1: {"Title": "one"}, 2: {"Title": "two"}})
    rows = client.list_items("L-xref")

    assert [row["fields"]["Title"] for row in rows] == ["one", "two"]


def test_get_list_item_returns_one_row():
    client, builder = _list_client({7: {"Application": "MyApp", "Status": "New"}})
    row = client.get_list_item("L-req", 7)

    assert row["id"] == "7"
    assert row["fields"] == {"Application": "MyApp", "Status": "New"}
    assert builder.requested_item_ids == ["7"]


def test_get_list_item_missing_raises():
    client, _ = _list_client({1: {"Title": "one"}})
    with pytest.raises(sharepoint.SharePointError, match="could not read item"):
        client.get_list_item("L-req", 99)


@requires_msgraph
def test_update_list_item_round_trips():
    client, builder = _list_client({7: {"Application": "MyApp", "Status": "New"}})
    written = client.update_list_item("L-req", 7, {"Status": "Completed"})

    assert builder.item_builders["7"].fields.patched == [{"Status": "Completed"}]
    assert written["Status"] == "Completed"
    # A partial mapping is a partial update: untouched columns survive.
    assert written["Application"] == "MyApp"
    assert client.get_list_item("L-req", 7)["fields"]["Status"] == "Completed"


def test_update_list_item_refuses_an_empty_write():
    client, _ = _list_client({7: {"Status": "New"}})
    with pytest.raises(sharepoint.SharePointError, match="no fields to write"):
        client.update_list_item("L-req", 7, {})
