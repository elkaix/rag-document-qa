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
from typing import Any, ClassVar

import chromadb

from src.domain import SearchResult

# Sentinel: "argument not supplied", distinct from an explicit None.
_UNSET: Any = object()

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# ChromaVectorStore                                                            #
# --------------------------------------------------------------------------- #

def _first_occurrence_indices(ids: list[str]) -> list[int]:
    """Return the positions of each id's first appearance, in order.

    Args:
        ids: Chunk ids, possibly with repeats.

    Returns:
        Indices to keep so every id appears exactly once, earliest wins.
    """
    seen: set[str] = set()
    keep: list[int] = []
    for index, chunk_id in enumerate(ids):
        if chunk_id not in seen:
            seen.add(chunk_id)
            keep.append(index)
    return keep


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
        store = ChromaVectorStore.open(client, "documents")

    Example (testing):
        client = chromadb.EphemeralClient()
        store = ChromaVectorStore.open(client, "test_docs", embedding_function=None)
    """

    # WHY a module constant: the cosine setting was spelled out at nine
    #      construction sites. The score conversion below is only correct in
    #      cosine space, so a site that forgot it produced silently wrong
    #      similarity scores rather than an error.
    SPACE_METADATA: ClassVar[dict[str, str]] = {"hnsw:space": "cosine"}

    @classmethod
    def open(
        cls,
        client: chromadb.ClientAPI,
        name: str,
        embedding_function: Any = _UNSET,
    ) -> ChromaVectorStore:
        """Get or create a cosine-space collection and wrap it.

        This is the supported way to build a store: it owns the one invariant
        the score conversion depends on, so callers cannot forget it.

        Args:
            client: Any ChromaDB client — persistent in production, ephemeral
                in tests.
            name: Collection name.
            embedding_function: Passed through to ChromaDB when supplied.
                Omit it to accept ChromaDB's built-in embedder; pass ``None``
                to supply raw embeddings yourself.

        Returns:
            A store over a collection guaranteed to use cosine distance.
        """
        kwargs: dict[str, Any] = {"name": name, "metadata": dict(cls.SPACE_METADATA)}
        if embedding_function is not _UNSET:
            kwargs["embedding_function"] = embedding_function
        return cls(collection=client.get_or_create_collection(**kwargs))

    def __init__(self, collection: chromadb.Collection) -> None:
        """
        Prefer :meth:`open`, which creates the collection with the required
        cosine space. Use this constructor directly only when a collection
        already exists and is known to be cosine.

        Args:
            collection: A ChromaDB Collection configured for cosine space.
        """
        self._collection = collection
        logger.debug(
            "ChromaVectorStore initialised with collection '%s'",
            collection.name,
        )

    @property
    def collection(self) -> chromadb.Collection:
        """The wrapped ChromaDB collection.

        Exposed for the two callers that legitimately need the collection object
        itself — wiring a facade and naming a collection for teardown — so they
        do not have to touch the private attribute. Reading *data* through this
        is a seam breach; use the query and lookup methods instead.
        """
        return self._collection

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
            All four lists must have the same length. Ids repeated *within* one
            call are collapsed to their first occurrence.

        BUG FIX: chunk ids are content-addressed, so a document containing the
            same text twice — a repeated boilerplate footer, a disclaimer page,
            a CSV with duplicate rows — produced the same id twice in a single
            batch. ChromaDB rejects such a batch with DuplicateIDError, so the
            whole upload failed rather than storing the document. Two chunks
            with the same content-addressed id *are* the same chunk, so
            collapsing them is what the id scheme already means.
        """
        keep = _first_occurrence_indices(ids)
        if len(keep) != len(ids):
            logger.debug(
                "Collapsed %d repeated chunk id(s) within one upsert batch",
                len(ids) - len(keep),
            )

        kwargs: dict[str, Any] = {
            "ids": [ids[i] for i in keep],
            "documents": [documents[i] for i in keep],
            "metadatas": [metadatas[i] for i in keep],
        }
        if embeddings is not None:
            # WHY: only include embeddings key when provided — passing embeddings=None
            # to ChromaDB triggers auto-embedding via the collection's embedding function.
            kwargs["embeddings"] = [embeddings[i] for i in keep]

        self._collection.upsert(**kwargs)
        logger.debug(
            "Upserted %d chunks into '%s'", len(keep), self._collection.name
        )

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
        for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances, strict=False):
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

    def get_by_doc_id(self, doc_id: str) -> list[dict[str, Any]]:
        """
        Fetch every chunk belonging to one document, unranked.

        WHY a dict shape rather than SearchResult: this is a metadata filter,
        not a similarity search — there is no meaningful score to attach.
        Reusing SearchResult would force a fake score field onto results
        that were never ranked, which is worse than a smaller, honest shape.

        WHY this method exists: callers that need "every chunk of doc X"
        (e.g. a document detail view) previously reached into
        ``self._collection`` directly to run this ChromaDB get() query.
        Wrapping it here keeps ChromaDB's raw batch-response shape internal
        to this module.

        Args:
            doc_id: The document identifier. All chunks with this doc_id
                    are returned.

        Returns:
            List of dicts with keys chunk_id, content, metadata — one per
            chunk, in ChromaDB's storage order (not similarity-ranked).
            Returns an empty list if no chunks match.
        """
        raw = self._collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"],
        )
        return [
            {
                "chunk_id": chunk_id,
                "content": raw["documents"][i] if raw["documents"] else "",
                "metadata": raw["metadatas"][i] if raw["metadatas"] else {},
            }
            for i, chunk_id in enumerate(raw["ids"])
        ]

    def all_chunk_texts(self) -> dict[str, str]:
        """Return every indexed chunk as ``{chunk_id: text}``.

        WHY this method exists: a sparse retriever (BM25) needs the whole corpus
        as text keyed by chunk id, which it cannot get from a similarity search.
        The eval pipeline used to reach into ``vector_store._collection`` and
        call ChromaDB's ``get()`` itself — twice, redundantly — parsing the raw
        batch-response shape at the call site. That is the same seam breach
        ``get_by_doc_id`` was added to close.

        Returns:
            Mapping of chunk_id to chunk text for the whole collection. Empty
            when nothing has been indexed yet.

        TRADE-OFF: this materialises the entire collection in memory, which is
            what a BM25 corpus requires. It is a corpus-build call, not a
            per-query one.
        """
        raw = self._collection.get(include=["documents"])
        ids = raw.get("ids") or []
        documents = raw.get("documents") or []
        return {
            chunk_id: documents[i] if i < len(documents) else ""
            for i, chunk_id in enumerate(ids)
        }

    # ---------------------------------------------------------------------- #
    # Delete operations                                                       #
    # ---------------------------------------------------------------------- #

    def delete_by_doc_id(self, doc_id: str) -> int:
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

        Returns:
            Number of chunks removed — useful for the delete endpoint's
            response body, which advertises `chunks_deleted` as a count.
        """
        # WHY count-then-delete: Chroma's delete() does not return a count, so
        # we look the matching IDs up first to report an accurate number.
        matching = self._collection.get(where={"doc_id": doc_id}, include=[])
        count = len(matching.get("ids", []))
        self._collection.delete(where={"doc_id": doc_id})
        logger.debug(
            "Deleted %d chunks for doc_id='%s' from '%s'",
            count, doc_id, self._collection.name,
        )
        return count

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
