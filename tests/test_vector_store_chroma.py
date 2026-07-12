"""
Tests for ChromaVectorStore — the STORAGE step in the RAG pipeline.

RAG Pipeline Position:
  Document → Chunks → Embeddings → [VECTOR STORE] → Retrieval → Generator → Answer
                                        ^^^
  These tests verify that ChromaVectorStore correctly stores, retrieves,
  and manages chunk embeddings using ChromaDB as the backend.

WHY: We test with explicit 3-dimensional embeddings to keep assertions
     deterministic and independent of any embedding model. This is possible
     because ChromaDB's EphemeralClient accepts pre-computed embeddings
     when embedding_function=None.

PATTERN: TDD — tests are written first so the interface is designed from
         the caller's perspective, not the implementation's perspective.
"""

from __future__ import annotations

import uuid

import pytest
import chromadb

from src.vector_store import ChromaVectorStore, SearchResult


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture
def chroma_collection():
    """
    An isolated in-memory ChromaDB collection for each test.

    WHY: EphemeralClient creates a fresh in-process database — no files on disk,
         no shared state between test runs. This is the correct unit-test pattern
         for ChromaDB.

    WHY embedding_function=None: We pass explicit embeddings in each test so
         results are deterministic and don't require an embedding model installed.

    WHY unique collection name: ChromaDB's EphemeralClient shares an in-process
         store — two clients calling get_or_create_collection("test_docs") will
         see the same underlying collection. Using a UUID per test guarantees
         complete isolation between test runs in the same pytest session.
    """
    # PATTERN: EphemeralClient is the test-friendly equivalent of SQLite's ":memory:"
    client = chromadb.EphemeralClient()
    # WHY uuid: prevents collection name collision when tests run in the same process
    collection_name = f"test_docs_{uuid.uuid4().hex}"
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,  # explicit embeddings only — no auto-embedding
    )
    return collection


@pytest.fixture
def store(chroma_collection):
    """A ChromaVectorStore wrapping the isolated test collection."""
    return ChromaVectorStore(collection=chroma_collection)


# --------------------------------------------------------------------------- #
# Helper data — simple 3D unit vectors for deterministic cosine similarity    #
# --------------------------------------------------------------------------- #

# Three orthogonal unit vectors in 3D space.
# Their pairwise cosine similarities are all 0.0 (completely unrelated).
# A query matching vec_a exactly will return chunk_a as the top hit.
VEC_A = [1.0, 0.0, 0.0]
VEC_B = [0.0, 1.0, 0.0]
VEC_C = [0.0, 0.0, 1.0]


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #

class TestUpsertAndQuery:
    """Verify basic upsert + semantic query flow."""

    def test_upsert_and_query(self, store: ChromaVectorStore):
        """
        Upsert 3 chunks with explicit embeddings; query returns the closest match.

        WHY: This is the core contract. Chunk A has embedding [1,0,0] and the query
             is also [1,0,0], so cosine distance = 0 (identical). Chunks B and C
             are orthogonal and should rank lower.
        """
        store.upsert(
            ids=["chunk_a", "chunk_b", "chunk_c"],
            documents=["Text about topic A", "Text about topic B", "Text about topic C"],
            metadatas=[
                {"doc_id": "doc1", "source": "file_a.txt"},
                {"doc_id": "doc1", "source": "file_b.txt"},
                {"doc_id": "doc2", "source": "file_c.txt"},
            ],
            embeddings=[VEC_A, VEC_B, VEC_C],
        )

        results = store.query(query_embedding=VEC_A, top_k=1)

        assert len(results) == 1
        top = results[0]
        assert top.chunk_id == "chunk_a", (
            f"Expected 'chunk_a' as top result, got '{top.chunk_id}'"
        )
        # Score should be close to 1.0 — identical vectors, cosine distance ≈ 0
        assert top.score >= 0.99, f"Expected score ≥ 0.99, got {top.score}"

    def test_query_returns_correct_metadata(self, store: ChromaVectorStore):
        """
        Verify SearchResult fields are populated from stored metadata.

        WHY: Metadata propagation is critical for source citations — the UI needs
             doc_id and other metadata to show users where an answer came from.
        """
        store.upsert(
            ids=["chunk_a"],
            documents=["Text about topic A"],
            metadatas=[{"doc_id": "doc1", "source": "file_a.txt"}],
            embeddings=[VEC_A],
        )

        results = store.query(query_embedding=VEC_A, top_k=1)

        assert results[0].doc_id == "doc1"
        assert results[0].content == "Text about topic A"
        assert results[0].metadata["source"] == "file_a.txt"


class TestDeleteByDocId:
    """Verify document-level deletion removes all associated chunks."""

    def test_delete_by_doc_id(self, store: ChromaVectorStore):
        """
        Upsert chunks from 2 docs, delete one, verify only the other remains.

        WHY: Document deletion is essential for keeping the knowledge base current.
             We verify at the stats level (not just query level) to confirm the
             chunks are truly gone and not just filtered out of results.
        """
        store.upsert(
            ids=["chunk_a", "chunk_b", "chunk_c"],
            documents=["Doc1 chunk1", "Doc1 chunk2", "Doc2 chunk1"],
            metadatas=[
                {"doc_id": "doc1"},
                {"doc_id": "doc1"},
                {"doc_id": "doc2"},
            ],
            embeddings=[VEC_A, VEC_B, VEC_C],
        )

        store.delete_by_doc_id("doc1")

        stats = store.get_stats()
        assert stats["total_chunks"] == 1, (
            f"Expected 1 chunk remaining after deleting doc1, got {stats['total_chunks']}"
        )

        # Verify the remaining chunk belongs to doc2
        results = store.query(query_embedding=VEC_C, top_k=5)
        assert len(results) == 1
        assert results[0].doc_id == "doc2"


class TestUpsertIdempotency:
    """Verify that upserting the same ID multiple times doesn't create duplicates."""

    def test_upsert_is_idempotent(self, store: ChromaVectorStore):
        """
        Upsert the same chunk ID 3 times; collection count must remain 1.

        WHY: Idempotency prevents double-ingestion bugs. If a user re-uploads a file,
             we want to overwrite, not duplicate. ChromaDB's upsert semantics give
             us this for free — we test that we're using upsert (not add).

        TRADE-OFF: Upsert requires an ID per chunk. We use a deterministic
                   hash-based ID (filename + chunk_index) in production so the
                   same chunk always maps to the same ID.
        """
        for _ in range(3):
            store.upsert(
                ids=["chunk_a"],
                documents=["Repeated text"],
                metadatas=[{"doc_id": "doc1"}],
                embeddings=[VEC_A],
            )

        stats = store.get_stats()
        assert stats["total_chunks"] == 1, (
            f"Expected exactly 1 chunk after 3 identical upserts, got {stats['total_chunks']}"
        )


class TestGetStats:
    """Verify the stats dict has required fields with correct values."""

    def test_get_stats(self, store: ChromaVectorStore):
        """
        Verify get_stats() returns total_chunks and backend fields.

        WHY: Stats are surfaced in the Documents dashboard so users can see
             how many chunks are indexed. The 'backend' field lets monitoring
             tools know which storage layer is in use.
        """
        store.upsert(
            ids=["chunk_a", "chunk_b"],
            documents=["Text A", "Text B"],
            metadatas=[{"doc_id": "doc1"}, {"doc_id": "doc1"}],
            embeddings=[VEC_A, VEC_B],
        )

        stats = store.get_stats()

        assert "total_chunks" in stats, "get_stats() must include 'total_chunks'"
        assert "backend" in stats, "get_stats() must include 'backend'"
        assert stats["total_chunks"] == 2
        assert stats["backend"] == "chromadb"

    def test_get_stats_empty_store(self, store: ChromaVectorStore):
        """Stats on an empty store should report 0 chunks."""
        stats = store.get_stats()
        assert stats["total_chunks"] == 0


class TestQueryEmptyStore:
    """Querying an empty store must not raise — return an empty list."""

    def test_query_empty_store(self, store: ChromaVectorStore):
        """
        Query before any documents are indexed returns an empty list.

        WHY: Defensive contract — the caller should never need to check
             'is the store empty?' before calling query(). Empty results
             are a valid, expected state (e.g., fresh deployment).
        """
        results = store.query(query_embedding=VEC_A, top_k=5)
        assert results == [], f"Expected empty list, got {results}"


class TestGetByDocId:
    """Verify metadata-filtered chunk lookup by doc_id."""

    def test_get_by_doc_id_returns_only_matching_chunks(self, store: ChromaVectorStore):
        """
        Upsert chunks from 2 docs; get_by_doc_id returns only the requested doc's chunks.

        WHY: This is a metadata lookup, not a similarity search — it must
             return every chunk for a doc_id, unordered by relevance, so
             callers building a "view this document" page see the whole
             document rather than a top-k slice.
        """
        store.upsert(
            ids=["chunk_a", "chunk_b", "chunk_c"],
            documents=["Doc1 chunk1", "Doc1 chunk2", "Doc2 chunk1"],
            metadatas=[
                {"doc_id": "doc1", "chunk_index": 0},
                {"doc_id": "doc1", "chunk_index": 1},
                {"doc_id": "doc2", "chunk_index": 0},
            ],
            embeddings=[VEC_A, VEC_B, VEC_C],
        )

        chunks = store.get_by_doc_id("doc1")

        assert len(chunks) == 2
        assert {c["chunk_id"] for c in chunks} == {"chunk_a", "chunk_b"}
        contents = {c["content"] for c in chunks}
        assert contents == {"Doc1 chunk1", "Doc1 chunk2"}
        for c in chunks:
            assert c["metadata"]["doc_id"] == "doc1"

    def test_get_by_doc_id_unknown_doc_returns_empty(self, store: ChromaVectorStore):
        """A doc_id with no chunks returns an empty list, not an error."""
        store.upsert(
            ids=["chunk_a"],
            documents=["Doc1 chunk1"],
            metadatas=[{"doc_id": "doc1"}],
            embeddings=[VEC_A],
        )

        assert store.get_by_doc_id("nonexistent") == []
