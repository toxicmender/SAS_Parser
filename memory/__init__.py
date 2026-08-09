"""Chat-history / KV persistence, relevance-based history selection, and
rolling thread summarization. See memory/README.md.

A regular package (not a PEP-420 namespace package) so packaging tools and
import machinery treat it uniformly.
"""

from .databricks_ai import DatabricksAIUnavailable, chat_model, embeddings
from .operations import DeltaMemoryMaintenance, VacuumPolicy

__all__ = [
    "DatabricksAIUnavailable",
    "DeltaMemoryMaintenance",
    "VacuumPolicy",
    "chat_model",
    "embeddings",
]
