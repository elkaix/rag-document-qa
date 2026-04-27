"""BgeEmbedder — Chroma EmbeddingFunction adapter for BAAI/bge-small-en-v1.5.

Pipeline position:
    Document → Chunks → [BgeEmbedder] → Vectors (384-dim) → ChromaDB

Phase 2 lever 2b. The factory installs this on the Chroma collection at
creation time; ChromaVectorStore.upsert/query then auto-embeds via this
function with no per-call code change.

Why bge-small-en-v1.5:
    - 384 dim — same as ChromaDB's default ONNX MiniLM, so dimension-comparable.
    - Strong on MTEB retrieval benchmarks (top-tier 33M-param model).
    - Loadable via `sentence-transformers`, which is already a project dep.
"""

from __future__ import annotations

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


class BgeEmbedder(EmbeddingFunction[Documents]):
    """Chroma-compatible embedding function backed by sentence-transformers.

    Caches the SentenceTransformer model on the instance to avoid re-loading
    on every call. Each instance is safe to share across one collection.
    """

    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self) -> None:
        # WHY lazy import: sentence-transformers is heavy. Only import when an
        # instance is created so module import remains cheap for tests that
        # never construct one.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.MODEL_NAME)

    def __call__(self, input: Documents) -> Embeddings:
        """Encode a batch of documents into 384-dim vectors.

        Args:
            input: List of strings to embed.

        Returns:
            List of 384-element float lists, one per input document.
        """
        # WHY tolist(): sentence-transformers returns a numpy array; Chroma
        # expects a plain list[list[float]] for serialization.
        vectors = self._model.encode(list(input), normalize_embeddings=True)
        return vectors.tolist()

    @staticmethod
    def name() -> str:
        """Required by Chroma >= 0.4.x for embedding-function identification."""
        return "bge_small_en_v1_5"
