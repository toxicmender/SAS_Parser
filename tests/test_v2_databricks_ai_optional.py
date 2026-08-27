"""No-skip import contract for the Databricks AI optional adapter."""

from __future__ import annotations

import importlib


def test_databricks_ai_dependency_is_installed_and_exports_required_models() -> None:
    module = importlib.import_module("databricks_langchain")
    assert module.ChatDatabricks is not None
    assert module.DatabricksEmbeddings is not None
