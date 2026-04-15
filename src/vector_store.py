"""
Vector store module — ChromaDB-backed storage for chunk embeddings.

RAG Pipeline Position:
  Document → Chunks → Embeddings → [VECTOR STORE] → Retrieval → Generator → Answer
                                        ^^^
  This module is the STORAGE step. It accepts chunk text + metadata + embeddings
  from the ingestion pipeline, persists them in ChromaDB, and serves the
  nearest-neighbour queries that power retrieval.

WHY ChromaDB over plain numpy (InMemoryVectorStore):
  - Persistent: survives process restarts (EphemeralClient for tests, PersistentClient for prod)
  - HNSW index: sub-linear search time at scale vs O(N) linear scan
  - Metadata filtering: WHERE clauses let us scope search to a single document
  - Built-in upsert: idempotent ingestion — re-uploading a file overwrites, never duplicates

WHY replace InMemoryVectorStore and QdrantVectorStore:
  Qdrant requires a running Docker container or Qdrant Cloud account. ChromaDB ships
  as a pure-Python package with no external service required, making it a better
  default for a portfolio project that should run `pip install && python` with zero ops.

TRADE-OFF: ChromaDB stores data on disk by default (PersistentClient). For unit
  tests we use EphemeralClient which is fully in-memory and isolated per test run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import chromadb

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data model                                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class SearchResult:
    """
    A single result returned from a ChromaDB similarity search.

    WHY a dataclass rather than a TypedDict: dataclasses give us attribute access
    (result.score), type checking, and a clean repr — all useful for debugging
    and for the response models in the FastAPI layer.
    """

    content: str          # The raw chunk text shown to the LLM as context
    metadata: dict[str, Any]  # Source info: filename, page, chunk_index, etc.
    score: float          # Cosine similarity 0..1 (1 = identical, 0 = orthogonal)
    doc_id: str           # Which document this chunk came from
    chunk_id: str         # Unique ID for this specific chunk


# --------------------------------------------------------------------------- #
# ChromaVectorStore                                                            #
# --------------------------------------------------------------------------- #

class ChromaVectorStore:
    """
    Vector store backed by a ChromaDB Collection.

    Wraps a single ChromaDB collection and exposes a minimal interface for the
    RAG pipeline: upsert, query, delete, and stats.

    PATTERN: Thin wrapper — this class does not own the ChromaDB client or collection
    lifecycle. The caller creates the client and collection (using EphemeralClient
    for tests, PersistentClient for production) and passes the collection in.
    This makes the class easy to test without mocking and easy to configure in prod.

    Example (production):
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )
        store = ChromaVectorStore(collection=collection)

    Example (testing):
        client = chromadb.EphemeralClient()
        collection = client.get_or_create_collection(
            name="test_docs",
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        store = ChromaVectorStore(collection=collection)
    """

    def __init__(self, collection: chromadb.Collection) -> None:
        """
        Args:
            collection: A pre-configured ChromaDB Collection instance.
                        Must use cosine space (metadata={"hnsw:space": "cosine"})
                        for scores to be meaningful in the 0..1 range.
        """
        self._collection = collection
        logger.debug(
            "ChromaVectorStore initialised with collection '%s'",
            collection.name,
        )

    # ---------------------------------------------------------------------- #
    # Write operations                                                        #
    # ---------------------------------------------------------------------- #

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """
        Add or update chunks in the collection.

        WHY upsert over add: ChromaDB's add() raises if an ID already exists.
        Upsert silently overwrites, giving us idempotent ingestion — re-processing
        a document won't create duplicates. This is crucial for user-facing apps
        where users may re-upload a file after editing it.

        Args:
            ids:        Unique chunk IDs (e.g., "doc_abc123_chunk_0"). Must be
                        stable across re-ingestion for idempotency to work.
            documents:  Raw text of each chunk — stored verbatim in ChromaDB.
            metadatas:  Per-chunk metadata dicts. Must include 'doc_id' so that
                        delete_by_doc_id() can find all chunks for a document.
            embeddings: Optional pre-computed embedding vectors. Pass these for
                        deterministic tests. Omit in production — ChromaDB will
                        auto-embed using the collection's embedding function.

        Note:
            All four lists must have the same length.
        """
        kwargs: dict[str, Any] = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        if embeddings is not None:
            # WHY: only include embeddings key when provided — passing embeddings=None
            # to ChromaDB triggers auto-embedding via the collection's embedding function.
            kwargs["embeddings"] = embeddings

        self._collection.upsert(**kwargs)
        logger.debug("Upserted %d chunks into '%s'", len(ids), self._collection.name)

    # ---------------------------------------------------------------------- #
    # Read operations                                                         #
    # ---------------------------------------------------------------------- #

    def query(
        self,
        query_text: str | None = None,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Find the most similar chunks for a query.

        Exactly one of query_text or query_embedding must be provided:
        - Use query_text in production (ChromaDB auto-embeds with the collection's
          embedding function).
        - Use query_embedding in tests (explicit, deterministic vectors).

        WHY separate query_text / query_embedding params rather than overloading:
        Explicit is better than implicit. The caller declares their intent.
        Passing both is an error; passing neither is an error. This surfaces
        mis-use at call time rather than producing silent wrong results.

        Args:
            query_text:       Natural language query string. ChromaDB embeds it.
            query_embedding:  Pre-computed query vector. Bypasses auto-embedding.
            top_k:            Number of top results to return.
            where:            Optional metadata filter dict (ChromaDB WHERE clause).
                              Example: {"doc_id": "abc123"} to search within one doc.

        Returns:
            List of SearchResult ordered by descending similarity score (0..1).
            Returns an empty list if the collection has no documents.

        Raises:
            ValueError: If neither or both of query_text/query_embedding are provided.
        """
        # PATTERN: guard clause — validate inputs before any I/O
        if query_text is None and query_embedding is None:
            raise ValueError("Provide either query_text or query_embedding.")
        if query_text is not None and query_embedding is not None:
            raise ValueError("Provide query_text OR query_embedding, not both.")

        # WHY: ChromaDB raises if you query an empty collection; return early to
        # give callers a clean empty-list contract with no exception handling needed.
        if self._collection.count() == 0:
            return []

        # Build ChromaDB query kwargs based on which input was provided
        query_kwargs: dict[str, Any] = {"n_results": top_k, "include": ["documents", "metadatas", "distances"]}
        if query_text is not None:
            query_kwargs["query_texts"] = [query_text]
        else:
            query_kwargs["query_embeddings"] = [query_embedding]  # type: ignore[list-item]

        if where is not None:
            query_kwargs["where"] = where

        raw = self._collection.query(**query_kwargs)

        # WHY: ChromaDB returns batched results (outer list = one entry per query).
        # We always send a single query, so we index [0] to get the per-chunk lists.
        ids = raw["ids"][0]
        documents = raw["documents"][0]       # type: ignore[index]
        metadatas = raw["metadatas"][0]       # type: ignore[index]
        distances = raw["distances"][0]       # type: ignore[index]

        results: list[SearchResult] = []
        for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
            # PATTERN: ChromaDB cosine distance is in [0, 2] where 0 = identical.
            # Convert to similarity score in [0, 1]:
            #   score = max(0, 1 - distance)
            # We clamp to 0 to handle floating-point noise that might produce
            # a tiny negative value for completely dissimilar vectors.
            score = max(0.0, 1.0 - distance)

            doc_id = meta.get("doc_id", "") if meta else ""
            results.append(
                SearchResult(
                    content=text or "",
                    metadata=meta or {},
                    score=score,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                )
            )

        return results

    # ---------------------------------------------------------------------- #
    # Delete operations                                                       #
    # ---------------------------------------------------------------------- #

    def delete_by_doc_id(self, doc_id: str) -> None:
        """
        Delete all chunks that belong to the given document.

        WHY: Chunk IDs are opaque to the caller — the caller only tracks doc_id.
        We use ChromaDB's WHERE clause to find and delete all chunks whose
        metadata["doc_id"] matches, without the caller needing to enumerate
        chunk IDs.

        TRADE-OFF: ChromaDB's delete(where=...) performs a metadata scan, which
        is O(N) in the number of chunks. For large collections (>1M chunks), a
        secondary index on doc_id would be faster. This is acceptable at current scale.

        Args:
            doc_id: The document identifier. All chunks with this doc_id are removed.
        """
        self._collection.delete(where={"doc_id": doc_id})
        logger.debug("Deleted chunks for doc_id='%s' from '%s'", doc_id, self._collection.name)

    # ---------------------------------------------------------------------- #
    # Stats                                                                   #
    # ---------------------------------------------------------------------- #

    def get_stats(self) -> dict[str, Any]:
        """
        Return runtime statistics about the collection.

        Used by the Documents dashboard to show how many chunks are indexed
        and which storage backend is active.

        Returns:
            Dict with keys:
                total_chunks (int):  Total number of chunks in the collection.
                backend      (str):  Always "chromadb".
                collection   (str):  The ChromaDB collection name.
        """
        return {
            "total_chunks": self._collection.count(),
            "backend": "chromadb",
            "collection": self._collection.name,
        }
