"""No-skip import contract for the Databricks AI optional adapter."""

from __future__ import annotations

import importlib
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("REQUIRE_INFRASTRUCTURE_TESTS") != "1",
    reason="requires the dedicated infrastructure-adapter environment",
)


def test_databricks_ai_dependency_is_installed_and_exports_required_models() -> None:
    module = importlib.import_module("databricks_langchain")
    assert module.ChatDatabricks is not None
    assert module.DatabricksEmbeddings is not None
