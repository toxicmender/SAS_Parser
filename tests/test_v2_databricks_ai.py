"""Phase 9 Databricks Model Serving adapter contracts."""

from __future__ import annotations

import builtins
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from sas_migrate.adapters.ai import (
    DatabricksAIUnavailable,
    DatabricksEmbeddingProvider,
    create_chat_model,
    create_embedding_model,
)

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def test_factories_preserve_provider_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def make(kind: str) -> Any:
        def factory(**kwargs: Any) -> Any:
            calls.append((kind, kwargs))
            return SimpleNamespace(kind=kind, kwargs=kwargs)

        return factory

    monkeypatch.setitem(
        sys.modules,
        "databricks_langchain",
        SimpleNamespace(
            ChatDatabricks=make("chat"),
            DatabricksEmbeddings=make("embedding"),
        ),
    )

    workspace = object()
    chat = create_chat_model(
        " chat-endpoint ",
        workspace_client=workspace,
        temperature=0,
        max_tokens=4096,
        timeout=30,
        max_retries=2,
        use_responses_api=True,
    )
    create_embedding_model(
        "embedding-endpoint",
        query_params={"instruction": "query"},
        documents_params={"instruction": "document"},
    )

    assert chat.kind == "chat"
    assert calls[0][1] == {
        "endpoint": "chat-endpoint",
        "use_responses_api": True,
        "workspace_client": workspace,
        "temperature": 0,
        "max_tokens": 4096,
        "timeout": 30,
        "max_retries": 2,
    }
    assert calls[1][0] == "embedding"
    assert calls[1][1] == {
        "endpoint": "embedding-endpoint",
        "target_uri": "databricks",
        "query_params": {"instruction": "query"},
        "documents_params": {"instruction": "document"},
    }


def test_factories_omit_unset_options_and_validate_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setitem(
        sys.modules,
        "databricks_langchain",
        SimpleNamespace(
            ChatDatabricks=lambda **kwargs: calls.append(kwargs),
            DatabricksEmbeddings=lambda **kwargs: calls.append(kwargs),
        ),
    )
    create_chat_model("chat")
    create_embedding_model("embedding", target_uri="custom")
    assert calls == [
        {"endpoint": "chat"},
        {"endpoint": "embedding", "target_uri": "custom"},
    ]
    with pytest.raises(ValueError, match="non-empty"):
        create_chat_model(" ")
    with pytest.raises(ValueError, match="non-empty"):
        create_embedding_model("")


def test_missing_optional_dependency_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def missing(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "databricks_langchain":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "databricks_langchain", raising=False)
    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(DatabricksAIUnavailable, match="databricks-ai"):
        create_chat_model("endpoint")


def test_embedding_provider_converts_mutable_sdk_vectors_to_port_contract() -> None:
    class Model:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            assert texts == ["first", "second"]
            return [[1, 2.5], [3, 4]]

        def embed_query(self, text: str) -> list[float]:
            assert text == "query"
            return [5, 6.5]

    provider = DatabricksEmbeddingProvider(Model())
    assert provider.embed_documents(("first", "second")) == (
        (1.0, 2.5),
        (3.0, 4.0),
    )
    assert provider.embed_query("query") == (5.0, 6.5)


def test_embedding_provider_from_endpoint_uses_lazy_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SimpleNamespace(
        embed_documents=lambda texts: [[float(len(text))] for text in texts],
        embed_query=lambda text: [float(len(text))],
    )
    monkeypatch.setattr(
        "sas_migrate.adapters.ai.databricks.create_embedding_model",
        lambda *args, **kwargs: model,
    )
    provider = DatabricksEmbeddingProvider.from_endpoint(
        "embedding", query_params={"truncate": True}
    )
    assert provider.embed_query("four") == (4.0,)


def test_ai_adapter_import_does_not_eagerly_import_provider_sdk() -> None:
    code = """
import sys
from sas_migrate.adapters.ai import create_chat_model
assert create_chat_model
assert 'databricks_langchain' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=pathlib.Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": str(SRC)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
