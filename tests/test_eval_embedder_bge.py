"""Tests for BgeEmbedder — a Chroma EmbeddingFunction adapter for BAAI/bge-small-en-v1.5."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def embedder():
    """Module-scoped to amortize the model-load cost across tests."""
    from src.eval.embedders import BgeEmbedder
    return BgeEmbedder()


def test_returns_384_dim_vectors(embedder):
    import numpy as np
    out = embedder(["hello world"])
    assert len(out) == 1
    assert len(out[0]) == 384
    # WHY (float, np.floating): chromadb 1.5.8's EmbeddingFunction.__init_subclass__
    # wraps __call__ with normalize_embeddings(), which always converts scalars to
    # numpy.float32 regardless of what the adapter returns. Python float alone fails.
    assert all(isinstance(x, (float, np.floating)) for x in out[0])


def test_synonyms_closer_than_unrelated(embedder):
    """Sanity check that the right model is loaded — not a stub."""
    import numpy as np
    a, b, c = embedder(["cat", "feline", "airplane"])
    a, b, c = np.array(a), np.array(b), np.array(c)
    cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    assert cos(a, b) > cos(a, c), "BGE should rank cat~feline > cat~airplane"


def test_chroma_collection_uses_embedder(embedder):
    """End-to-end: a Chroma collection created with BgeEmbedder retrieves the right doc."""
    import chromadb
    client = chromadb.EphemeralClient()
    coll = client.get_or_create_collection(
        name="test_bge_e2e",
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )
    coll.upsert(
        ids=["d1", "d2", "d3"],
        documents=[
            "Cats are small carnivorous mammals often kept as pets.",
            "Airplanes are powered flying vehicles with fixed wings.",
            "Dogs are domesticated descendants of wolves.",
        ],
    )
    res = coll.query(query_texts=["What is a feline?"], n_results=1)
    assert res["ids"][0][0] == "d1"
