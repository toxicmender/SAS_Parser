"""No-skip import contract for Phase 9 infrastructure adapter extras."""

from __future__ import annotations

import asyncio
import importlib
import os

import pytest
from pydantic import SecretStr

from sas_migrate.adapters.sharepoint import GraphSdkGateway
from sas_migrate.application.ports import AccessToken
from sas_migrate.config import SharePointSettings

pytestmark = pytest.mark.skipif(
    os.environ.get("REQUIRE_INFRASTRUCTURE_TESTS") != "1",
    reason="requires the dedicated infrastructure-adapter environment",
)

OPTIONAL_INFRASTRUCTURE_MODULES = (
    ("msal", "azure"),
    ("hvac", "vault"),
    ("databricks.sdk", "databricks"),
    ("msgraph", "sharepoint"),
)


@pytest.mark.parametrize(("module_name", "extra"), OPTIONAL_INFRASTRUCTURE_MODULES)
def test_infrastructure_dependency_is_installed_and_importable(
    module_name: str,
    extra: str,
) -> None:
    assert extra in {"azure", "vault", "databricks", "sharepoint"}
    assert importlib.import_module(module_name) is not None


def test_real_graph_sdk_client_is_constructible_without_network_access() -> None:
    class Provider:
        async def get_token(self, scopes: tuple[str, ...] = ()) -> AccessToken:
            return AccessToken(value=SecretStr("not-used"), source=str(scopes))

    gateway = GraphSdkGateway(
        SharePointSettings(site_id="site", list_id_sas_requests="requests"),
        Provider(),
    )
    assert gateway.client is not None
    asyncio.run(gateway.close())
