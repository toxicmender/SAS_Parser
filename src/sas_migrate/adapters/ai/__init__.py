"""Provider-specific model and embedding adapters."""

from .databricks import (
    DatabricksAIUnavailable,
    DatabricksEmbeddingProvider,
    EmbeddingModel,
    create_chat_model,
    create_embedding_model,
)
from .openai_compatible import GatewayLLMError, OpenAICompatibleLLM

__all__ = [
    "DatabricksAIUnavailable",
    "DatabricksEmbeddingProvider",
    "EmbeddingModel",
    "GatewayLLMError",
    "OpenAICompatibleLLM",
    "create_chat_model",
    "create_embedding_model",
]
