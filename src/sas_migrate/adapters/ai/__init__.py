"""Provider-specific model and embedding adapters."""

from .databricks import (
    DatabricksAIUnavailable,
    DatabricksEmbeddingProvider,
    EmbeddingModel,
    create_chat_model,
    create_embedding_model,
)

__all__ = [
    "DatabricksAIUnavailable",
    "DatabricksEmbeddingProvider",
    "EmbeddingModel",
    "create_chat_model",
    "create_embedding_model",
]
