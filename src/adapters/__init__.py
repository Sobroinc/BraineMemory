"""Internal adapters (not exposed to LLM)."""

from src.adapters.embeddings import EmbeddingsAdapter
from src.adapters.vision import VisionAdapter

__all__ = ["EmbeddingsAdapter", "VisionAdapter"]
