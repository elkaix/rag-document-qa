"""Document retrieval engine using cosine similarity.

The retriever is the SEARCH component of RAG — given a user query,
it finds the most relevant document chunks from the embedded corpus.

Retrieval flow:
1. Embed the user's query using the same embedder
2. Compute similarity between query and all chunk embeddings
3. Return the top-K most similar chunks

This is essentially a vector search over the embedding space.
"""
import logging
from typing import List, Dict, Any
import numpy as np
from .embeddings import TfidfEmbedder

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieve relevant document chunks based on query similarity."""

    def __init__(self, embedder: TfidfEmbedder, chunks: List[Dict[str, Any]], top_k: int = 5) -> None:
        self.embedder = embedder
        self.chunks = chunks
        self.top_k = top_k
        self._embeddings: np.ndarray = np.array([])

    def build_index(self) -> None:
        """Build the search index by embedding all chunks."""
        texts = [c["text"] for c in self.chunks]
        self._embeddings = self.embedder.fit_transform(texts)
        logger.info("Built retrieval index: %d chunks indexed", len(self.chunks))

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Find the most relevant chunks for a given query."""
        if len(self._embeddings) == 0:
            self.build_index()

        # Embed the query
        query_vec = self.embedder.transform([query])[0]

        # Compute similarities
        similarities = []
        for i, chunk_emb in enumerate(self._embeddings):
            sim = self.embedder.similarity(query_vec, chunk_emb)
            similarities.append({"chunk": self.chunks[i], "score": round(sim, 4)})

        # Sort by similarity and return top-K
        similarities.sort(key=lambda x: x["score"], reverse=True)
        results = similarities[:self.top_k]
        logger.info("Retrieved %d chunks for query: '%s' (top score: %.4f)", len(results), query[:50], results[0]["score"] if results else 0)
        return results
