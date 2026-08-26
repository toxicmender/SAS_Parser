"""No-skip import contract for Phase 9 infrastructure adapter extras."""

from __future__ import annotations

import importlib
import os

import pytest

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
