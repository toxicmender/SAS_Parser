"""Lazy Databricks Model Serving adapters.

The optional ``databricks_langchain`` package is imported only when a factory
is called. Importing v2 application or adapter packages remains credential-free
and does not initialize a Databricks workspace client.
"""

from __future__ import annotations

from typing import Any, Protocol

from sas_migrate.application.ports import EmbeddingVector


class DatabricksAIUnavailable(RuntimeError):
    """The optional Databricks AI adapter dependency is unavailable."""


class EmbeddingModel(Protocol):
    """Structural surface exposed by DatabricksEmbeddings."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _endpoint(value: str) -> str:
    endpoint = value.strip() if isinstance(value, str) else ""
    if not endpoint:
        raise ValueError("Databricks serving endpoint must be a non-empty string")
    return endpoint


def _langchain() -> Any:
    try:
        import databricks_langchain  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - optional-environment contract
        raise DatabricksAIUnavailable(
            "Databricks AI support requires the 'databricks-ai' project extra"
        ) from exc
    return databricks_langchain


def create_chat_model(
    endpoint: str,
    *,
    workspace_client: Any | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    **extra: Any,
) -> Any:
    """Construct a ChatDatabricks model without importing its SDK eagerly."""

    kwargs: dict[str, Any] = {"endpoint": _endpoint(endpoint), **extra}
    for name, value in {
        "workspace_client": workspace_client,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "max_retries": max_retries,
    }.items():
        if value is not None:
            kwargs[name] = value
    return _langchain().ChatDatabricks(**kwargs)


def create_embedding_model(
    endpoint: str,
    *,
    target_uri: str = "databricks",
    query_params: dict[str, Any] | None = None,
    documents_params: dict[str, Any] | None = None,
) -> EmbeddingModel:
    """Construct the provider SDK embedding model on explicit selection."""

    kwargs: dict[str, Any] = {
        "endpoint": _endpoint(endpoint),
        "target_uri": target_uri,
    }
    if query_params is not None:
        kwargs["query_params"] = query_params
    if documents_params is not None:
        kwargs["documents_params"] = documents_params
    return _langchain().DatabricksEmbeddings(**kwargs)


class DatabricksEmbeddingProvider:
    """Adapt DatabricksEmbeddings to the immutable v2 embedding port."""

    def __init__(self, model: EmbeddingModel) -> None:
        self._model = model

    @classmethod
    def from_endpoint(
        cls,
        endpoint: str,
        *,
        target_uri: str = "databricks",
        query_params: dict[str, Any] | None = None,
        documents_params: dict[str, Any] | None = None,
    ) -> DatabricksEmbeddingProvider:
        return cls(
            create_embedding_model(
                endpoint,
                target_uri=target_uri,
                query_params=query_params,
                documents_params=documents_params,
            )
        )

    def embed_documents(
        self, texts: tuple[str, ...]
    ) -> tuple[EmbeddingVector, ...]:
        return tuple(
            tuple(float(value) for value in vector)
            for vector in self._model.embed_documents(list(texts))
        )

    def embed_query(self, text: str) -> EmbeddingVector:
        return tuple(float(value) for value in self._model.embed_query(text))


__all__ = [
    "DatabricksAIUnavailable",
    "DatabricksEmbeddingProvider",
    "EmbeddingModel",
    "create_chat_model",
    "create_embedding_model",
]
