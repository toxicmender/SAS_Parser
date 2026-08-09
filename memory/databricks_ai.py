"""Optional Databricks AI Bridge factories for the memory seams.

This module deliberately imports ``databricks_langchain`` only inside its
factories. Installing and importing :mod:`memory` therefore remains fully
offline unless the application explicitly selects Databricks model serving.
The returned chat model can be passed as ``SasLLMPipeline(llm=...)`` or as the
``model`` for :class:`MemoryExtractor` and :class:`RollingSummarizer`.
"""

from __future__ import annotations

from typing import Any


class DatabricksAIUnavailable(RuntimeError):
    """The optional Databricks AI Bridge dependency is not installed."""


def _langchain() -> Any:
    try:
        import databricks_langchain
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise DatabricksAIUnavailable(
            "Databricks AI support needs the optional dependency: "
            "pip install 'sas-parser[databricks-ai]'"
        ) from exc
    return databricks_langchain


def chat_model(
    endpoint: str,
    *,
    workspace_client: Any | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    **extra: Any,
) -> Any:
    """Create a lazy ``ChatDatabricks`` model for pipeline and memory work."""
    kwargs: dict[str, Any] = {"endpoint": endpoint, **extra}
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


def embeddings(
    endpoint: str,
    *,
    target_uri: str = "databricks",
    query_params: dict[str, Any] | None = None,
    documents_params: dict[str, Any] | None = None,
) -> Any:
    """Create ``DatabricksEmbeddings`` for ``HybridRanker(embeddings=...)``."""
    kwargs: dict[str, Any] = {"endpoint": endpoint, "target_uri": target_uri}
    if query_params is not None:
        kwargs["query_params"] = query_params
    if documents_params is not None:
        kwargs["documents_params"] = documents_params
    return _langchain().DatabricksEmbeddings(**kwargs)
