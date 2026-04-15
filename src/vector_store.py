"""
Vector store module for RAG pipeline.

Provides an abstract VectorStore interface with:
- InMemoryVectorStore: pure-numpy implementation
- QdrantVectorStore: wraps qdrant_client (optional dependency)
- Factory function: get_vector_store()
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single result returned from a vector store search."""

    content: str
    metadata: Dict[str, Any]
    score: float
    doc_id: str
    chunk_id: str


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        """Add documents with their embeddings to the store.

        Args:
            documents: List of dicts with keys: content, metadata, doc_id, chunk_id.
            embeddings: Corresponding embedding vectors.
        """

    @abstractmethod
    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[SearchResult]:
        """Search for nearest neighbours.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.

        Returns:
            List of SearchResult sorted by descending score.
        """

    @abstractmethod
    def delete(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document.

        Args:
            doc_id: Document identifier.

        Returns:
            Number of chunks deleted.
        """

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the store.

        Returns:
            Dict with at least 'total_chunks' and 'total_documents'.
        """


# --------------------------------------------------------------------------- #
# In-memory implementation                                                     #
# --------------------------------------------------------------------------- #

class InMemoryVectorStore(VectorStore):
    """Numpy-based in-memory vector store.

    Stores all vectors in RAM and computes cosine similarity at query time.
    Suitable for prototyping and small document sets (< 100k chunks).
    """

    def __init__(self) -> None:
        self._embeddings: List[np.ndarray] = []
        self._documents: List[Dict[str, Any]] = []

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        """Add documents and their embeddings.

        Args:
            documents: Each dict must have: content, metadata, doc_id, chunk_id.
            embeddings: Matching list of embedding vectors.
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"documents and embeddings must have the same length "
                f"(got {len(documents)} and {len(embeddings)})"
            )
        for doc, emb in zip(documents, embeddings):
            vec = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self._embeddings.append(vec)
            self._documents.append(doc)

        logger.debug("Added %d chunks. Total: %d", len(documents), len(self._documents))

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[SearchResult]:
        """Return top_k most similar chunks by cosine similarity.

        Args:
            query_embedding: Query vector.
            top_k: Number of results.

        Returns:
            List of SearchResult sorted by descending similarity.
        """
        if not self._embeddings:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        matrix = np.stack(self._embeddings)  # (N, D)
        scores = matrix.dot(q)  # cosine similarity (since vectors are normalised)

        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: List[SearchResult] = []
        for idx in top_indices:
            doc = self._documents[idx]
            results.append(
                SearchResult(
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    score=float(scores[idx]),
                    doc_id=doc.get("doc_id", ""),
                    chunk_id=doc.get("chunk_id", ""),
                )
            )
        return results

    def delete(self, doc_id: str) -> int:
        """Delete all chunks for the given doc_id.

        Args:
            doc_id: Document identifier.

        Returns:
            Number of chunks removed.
        """
        indices_to_keep = [
            i for i, d in enumerate(self._documents) if d.get("doc_id") != doc_id
        ]
        deleted = len(self._documents) - len(indices_to_keep)
        self._documents = [self._documents[i] for i in indices_to_keep]
        self._embeddings = [self._embeddings[i] for i in indices_to_keep]
        logger.debug("Deleted %d chunks for doc_id=%s", deleted, doc_id)
        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """Return store statistics."""
        doc_ids = {d.get("doc_id") for d in self._documents}
        dim = int(self._embeddings[0].shape[0]) if self._embeddings else 0
        return {
            "total_chunks": len(self._documents),
            "total_documents": len(doc_ids),
            "embedding_dimension": dim,
            "backend": "memory",
        }


# --------------------------------------------------------------------------- #
# Qdrant implementation                                                        #
# --------------------------------------------------------------------------- #

class QdrantVectorStore(VectorStore):
    """Vector store backed by Qdrant (optional dependency).

    Automatically falls back gracefully if qdrant_client is not installed.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "documents",
        vector_dim: int = 384,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Args:
            host: Qdrant host.
            port: Qdrant port.
            collection_name: Default collection name.
            vector_dim: Embedding dimension.
            url: Full Qdrant URL (overrides host/port).
            api_key: Qdrant API key for cloud.
        """
        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.models import Distance, VectorParams  # type: ignore

            self._Distance = Distance
            self._VectorParams = VectorParams
            connect_kwargs: Dict[str, Any] = {}
            if url:
                connect_kwargs["url"] = url
                if api_key:
                    connect_kwargs["api_key"] = api_key
            else:
                connect_kwargs["host"] = host
                connect_kwargs["port"] = port

            self._client = QdrantClient(**connect_kwargs)
            self._available = True
            logger.info("QdrantVectorStore connected to %s", url or f"{host}:{port}")
        except ImportError:
            logger.warning(
                "qdrant_client not installed. QdrantVectorStore unavailable. "
                "Install with: pip install qdrant-client"
            )
            self._client = None
            self._available = False

        self.collection_name = collection_name
        self.vector_dim = vector_dim

        if self._available:
            self.create_collection(collection_name, vector_dim)

    def _require_client(self) -> None:
        if not self._available or self._client is None:
            raise RuntimeError("qdrant_client is not installed or Qdrant is unreachable.")

    def create_collection(self, collection_name: str, vector_dim: int) -> None:
        """Create a Qdrant collection if it does not already exist.

        Args:
            collection_name: Name of the collection.
            vector_dim: Dimensionality of vectors.
        """
        self._require_client()
        existing = [c.name for c in self._client.get_collections().collections]  # type: ignore[union-attr]
        if collection_name not in existing:
            self._client.create_collection(  # type: ignore[union-attr]
                collection_name=collection_name,
                vectors_config=self._VectorParams(
                    size=vector_dim,
                    distance=self._Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection '%s' (dim=%d)", collection_name, vector_dim)
        else:
            logger.debug("Collection '%s' already exists", collection_name)

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        """Upsert documents into Qdrant.

        Args:
            documents: Each dict must have: content, metadata, doc_id, chunk_id.
            embeddings: Matching embedding vectors.
        """
        self._require_client()
        from qdrant_client.models import PointStruct  # type: ignore

        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings length mismatch")

        points = []
        for doc, emb in zip(documents, embeddings):
            payload = {
                "content": doc.get("content", ""),
                "doc_id": doc.get("doc_id", ""),
                "chunk_id": doc.get("chunk_id", ""),
                "metadata": doc.get("metadata", {}),
            }
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload=payload,
                )
            )

        self._client.upsert(collection_name=self.collection_name, points=points)  # type: ignore[union-attr]
        logger.debug("Upserted %d points to Qdrant", len(points))

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[SearchResult]:
        """Search Qdrant with cosine similarity.

        Args:
            query_embedding: Query vector.
            top_k: Number of results.

        Returns:
            List of SearchResult.
        """
        self._require_client()
        hits = self._client.search(  # type: ignore[union-attr]
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
        )
        results: List[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    content=payload.get("content", ""),
                    metadata=payload.get("metadata", {}),
                    score=float(hit.score),
                    doc_id=payload.get("doc_id", ""),
                    chunk_id=payload.get("chunk_id", ""),
                )
            )
        return results

    def delete(self, doc_id: str) -> int:
        """Delete all points with the given doc_id.

        Args:
            doc_id: Document identifier.

        Returns:
            Number of points deleted (approximated from scroll).
        """
        self._require_client()
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore

        filter_ = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )
        # Count before deletion
        count_result = self._client.count(  # type: ignore[union-attr]
            collection_name=self.collection_name,
            count_filter=filter_,
            exact=True,
        )
        deleted_count = count_result.count
        self._client.delete(  # type: ignore[union-attr]
            collection_name=self.collection_name,
            points_selector=filter_,
        )
        logger.debug("Deleted %d points for doc_id=%s from Qdrant", deleted_count, doc_id)
        return deleted_count

    def get_stats(self) -> Dict[str, Any]:
        """Return Qdrant collection statistics."""
        self._require_client()
        info = self._client.get_collection(self.collection_name)  # type: ignore[union-attr]
        return {
            "total_chunks": info.points_count,
            "total_documents": None,  # Qdrant doesn't track this natively
            "embedding_dimension": self.vector_dim,
            "backend": "qdrant",
            "collection": self.collection_name,
            "status": str(info.status),
        }


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #

def get_vector_store(
    backend: str = "memory",
    **kwargs: Any,
) -> VectorStore:
    """Factory function to create a VectorStore.

    Args:
        backend: 'memory' or 'qdrant'.
        **kwargs: Additional kwargs passed to the store constructor.

    Returns:
        A VectorStore instance.

    Raises:
        ValueError: If backend is unknown.
    """
    if backend == "memory":
        return InMemoryVectorStore()
    if backend == "qdrant":
        return QdrantVectorStore(**kwargs)
    raise ValueError(f"Unknown vector store backend: '{backend}'. Choose 'memory' or 'qdrant'.")
