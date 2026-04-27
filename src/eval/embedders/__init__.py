"""Phase 2 embedder package — pluggable Chroma EmbeddingFunction adapters."""

from src.eval.embedders.bge_small import BgeEmbedder

__all__ = ["BgeEmbedder"]
