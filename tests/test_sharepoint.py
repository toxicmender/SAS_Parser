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


class _ListItemsBuilder:
    def __init__(self, pages):
        self._pages = list(pages)
        self._i = 0
        self.configs: list[Any] = []

    def get(self, config=None):
        self.configs.append(config)
        return _Awaitable(self._pages[self._i])

    def with_url(self, url):
        self._i += 1
        return self


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
