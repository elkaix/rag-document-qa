"""Tests for ChunkCorpus — the corpus mirror that notices when it is stale.

The staleness rule is the whole reason ADR 0004 deferred the hybrid strategy:
``BM25HybridRetriever`` took a corpus snapshot at construction, so every
document ingested afterwards was invisible to the sparse side. These tests pin
the replacement rule — refresh when the store's published revision moves — at
the seam that owns it, rather than only through the retriever that consumes it.
"""

from __future__ import annotations

import uuid

import chromadb
import pytest

from src.retrieval.corpus import ChunkCorpus
from src.vector_store import ChromaVectorStore

VEC_A = [1.0, 0.0, 0.0]
VEC_B = [0.0, 1.0, 0.0]


@pytest.fixture
def store() -> ChromaVectorStore:
    """A store with no embedding function, so tests supply vectors explicitly.

    WHY the random collection name: chromadb.EphemeralClient() returns a shared
    in-process instance for identical settings, so a fixed name would let one
    test's chunks leak into the next test's corpus.
    """
    client = chromadb.EphemeralClient()
    return ChromaVectorStore.open(
        client, f"corpus_test_{uuid.uuid4().hex[:8]}", embedding_function=None
    )


def _upsert(store: ChromaVectorStore, chunk_id: str, text: str, doc_id: str) -> None:
    store.upsert(
        ids=[chunk_id],
        documents=[text],
        metadatas=[{"doc_id": doc_id, "filename": f"{doc_id}.txt"}],
        embeddings=[VEC_A],
    )


class TestFreshness:
    def test_empty_store_yields_an_empty_snapshot(self, store):
        assert len(ChunkCorpus(store).snapshot()) == 0

    def test_repeated_calls_return_the_identical_object(self, store):
        """Consumers cache expensive derived values on snapshot identity."""
        _upsert(store, "c1", "alpha beta", "d1")
        corpus = ChunkCorpus(store)
        assert corpus.snapshot() is corpus.snapshot()

    def test_a_write_produces_a_new_snapshot(self, store):
        _upsert(store, "c1", "alpha beta", "d1")
        corpus = ChunkCorpus(store)
        first = corpus.snapshot()

        _upsert(store, "c2", "gamma delta", "d2")
        second = corpus.snapshot()

        assert second is not first
        assert set(second.chunks) == {"c1", "c2"}
        assert second.revision > first.revision

    def test_a_deletion_removes_the_chunk_from_the_next_snapshot(self, store):
        _upsert(store, "c1", "alpha beta", "d1")
        _upsert(store, "c2", "gamma delta", "d2")
        corpus = ChunkCorpus(store)
        assert len(corpus.snapshot()) == 2

        store.delete_by_doc_id("d1")
        assert set(corpus.snapshot().chunks) == {"c2"}

    def test_a_populated_store_is_read_on_the_first_call(self, store):
        """The cold-start case: revision 0, but the collection is not empty.

        Seeding the cache with an empty snapshot at revision 0 would look fresh
        and serve nothing until the process's first write — which, on a
        persistent collection restored from disk, could be never.
        """
        _upsert(store, "c1", "alpha beta", "d1")
        # The bare constructor (rather than .open) deliberately simulates a
        # process that reopened an existing collection: same data, revision 0.
        reopened = ChromaVectorStore(collection=store.collection)
        assert reopened.revision == 0
        assert set(ChunkCorpus(reopened).snapshot().chunks) == {"c1"}


class TestTheCounterIsTheContract:
    """Writes that bypass the store bypass the freshness rule — deliberately.

    tests/test_retrieval_hybrid.py's end-to-end case upserts through the raw
    Chroma collection, which never moves `revision`. It works because the cache
    starts empty rather than seeded, so the first snapshot still reads the
    collection. Pinning that here keeps a reader from taking the raw-collection
    write as the sanctioned pattern: every production write goes through
    ChromaVectorStore, which counts.
    """

    def test_a_raw_collection_write_does_not_invalidate_the_snapshot(self, store):
        _upsert(store, "c1", "alpha beta", "d1")
        corpus = ChunkCorpus(store)
        first = corpus.snapshot()

        store.collection.upsert(
            ids=["c2"], documents=["gamma delta"], metadatas=[{"doc_id": "d2"}], embeddings=[VEC_B]
        )
        assert store.revision == first.revision
        assert corpus.snapshot() is first

    def test_the_same_write_through_the_store_does_invalidate_it(self, store):
        _upsert(store, "c1", "alpha beta", "d1")
        corpus = ChunkCorpus(store)
        first = corpus.snapshot()

        _upsert(store, "c2", "gamma delta", "d2")
        assert corpus.snapshot() is not first


class TestSnapshotContents:
    def test_chunks_carry_metadata_and_doc_id(self, store):
        _upsert(store, "c1", "alpha beta", "d1")
        chunk = ChunkCorpus(store).snapshot().chunks["c1"]
        assert chunk.chunk_id == "c1"
        assert chunk.doc_id == "d1"
        assert chunk.metadata["filename"] == "d1.txt"
        assert chunk.content == "alpha beta"

    def test_a_chunk_without_a_doc_id_gets_an_empty_one_not_a_derived_one(self, store):
        """Chunk derives ids when blank; the corpus must supply the real one."""
        store.upsert(
            ids=["c1"],
            documents=["alpha"],
            metadatas=[{"filename": "orphan.txt"}],
            embeddings=[VEC_B],
        )
        chunk = ChunkCorpus(store).snapshot().chunks["c1"]
        assert chunk.chunk_id == "c1"
        assert chunk.doc_id == ""
