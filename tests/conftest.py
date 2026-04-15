"""
Shared pytest fixtures for the RAG Document Q&A test suite.

All fixtures use mock/in-memory data — no external dependencies required.
Uses ChromaDB EphemeralClient for vector store fixtures (no disk I/O, no cleanup needed).
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import List
import hashlib

# Ensure project root is on sys.path so `from src.x import ...` works
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import chromadb

from src.document_loader import Chunk, Document
from src.vector_store import ChromaVectorStore


# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

EMBEDDING_DIM = 384
SAMPLE_TEXT = (
    "Retrieval-Augmented Generation (RAG) is a technique that enhances large language "
    "models by retrieving relevant documents from an external knowledge base before "
    "generating a response. This allows the model to access up-to-date information "
    "and produce more accurate, grounded answers.\n\n"
    "The retrieval step typically uses dense vector search, where both the query and "
    "documents are embedded into a shared vector space. The most similar documents "
    "are then passed as context to the language model along with the original query."
)

SAMPLE_TEXT_2 = (
    "Vector databases store embeddings and support efficient approximate nearest-neighbour "
    "search. Popular options include Qdrant, Pinecone, Weaviate, and Chroma. "
    "They enable semantic search at scale, handling millions of vectors with low latency."
)


# --------------------------------------------------------------------------- #
# Document fixtures                                                            #
# --------------------------------------------------------------------------- #

@pytest.fixture
def sample_document() -> Document:
    """A single Document instance with realistic content."""
    return Document(
        content=SAMPLE_TEXT,
        metadata={
            "filename": "rag_overview.txt",
            "file_type": "txt",
            "file_size_bytes": len(SAMPLE_TEXT.encode()),
        },
    )


@pytest.fixture
def sample_document_2() -> Document:
    """A second Document with different content."""
    return Document(
        content=SAMPLE_TEXT_2,
        metadata={
            "filename": "vector_dbs.txt",
            "file_type": "txt",
            "file_size_bytes": len(SAMPLE_TEXT_2.encode()),
        },
    )


@pytest.fixture
def sample_chunks(sample_document: Document) -> List[Chunk]:
    """Pre-built chunks from the sample document."""
    texts = [
        "Retrieval-Augmented Generation (RAG) is a technique that enhances large language models.",
        "The retrieval step uses dense vector search where documents are embedded.",
        "The most similar documents are passed as context to the language model.",
    ]
    chunks = []
    for i, text in enumerate(texts):
        chunk = Chunk(
            content=text,
            metadata={
                "filename": "rag_overview.txt",
                "chunk_index": i,
                "chunk_strategy": "fixed",
            },
            doc_id=sample_document.doc_id,
        )
        chunks.append(chunk)
    return chunks


# --------------------------------------------------------------------------- #
# Embedding fixtures                                                           #
# --------------------------------------------------------------------------- #

def _make_deterministic_embedding(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """Create a deterministic unit-norm embedding from text."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "little")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


@pytest.fixture
def chroma_collection():
    """
    Ephemeral ChromaDB collection for testing.

    WHY EphemeralClient: no disk I/O, no teardown needed — each test gets
    a fresh in-memory collection that vanishes when the fixture goes out of scope.
    PATTERN: cosine distance matches how the production vector store is configured.
    """
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(
        name="test_docs",
        metadata={"hnsw:space": "cosine"},
        # WHY None: we supply our own deterministic embeddings via upsert(),
        # so ChromaDB must not auto-embed — passing embedding_function=None
        # disables the default all-MiniLM-L6-v2 auto-embedder.
        embedding_function=None,
    )


@pytest.fixture
def populated_vector_store(sample_chunks: List[Chunk], chroma_collection) -> ChromaVectorStore:
    """
    A ChromaVectorStore pre-loaded with sample_chunks and deterministic embeddings.

    Replaces the old InMemoryVectorStore-based fixture now that TF-IDF has been
    removed. Tests that need a ready-to-query store use this fixture directly.
    """
    store = ChromaVectorStore(chroma_collection)
    store.upsert(
        ids=[c.chunk_id for c in sample_chunks],
        documents=[c.content for c in sample_chunks],
        metadatas=[
            {
                "doc_id": c.doc_id,
                "filename": c.metadata.get("filename", ""),
                "chunk_index": c.metadata.get("chunk_index", 0),
            }
            for c in sample_chunks
        ],
        embeddings=[_make_deterministic_embedding(c.content) for c in sample_chunks],
    )
    return store


# --------------------------------------------------------------------------- #
# Tmp file helper                                                              #
# --------------------------------------------------------------------------- #

@pytest.fixture
def tmp_text_file(tmp_path: Path) -> Path:
    """A temporary .txt file with sample content."""
    file = tmp_path / "test_document.txt"
    file.write_text(SAMPLE_TEXT, encoding="utf-8")
    return file


@pytest.fixture
def tmp_json_file(tmp_path: Path) -> Path:
    """A temporary .json file."""
    import json

    data = {"title": "Test Document", "body": SAMPLE_TEXT, "tags": ["rag", "nlp"]}
    file = tmp_path / "test_data.json"
    file.write_text(json.dumps(data), encoding="utf-8")
    return file


@pytest.fixture
def tmp_csv_file(tmp_path: Path) -> Path:
    """A temporary .csv file."""
    import csv

    file = tmp_path / "test_data.csv"
    rows = [
        ["name", "description", "category"],
        ["RAG", "Retrieval-Augmented Generation", "NLP"],
        ["BERT", "Bidirectional Encoder Representations", "NLP"],
        ["GPT", "Generative Pre-trained Transformer", "LLM"],
    ]
    with file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)
    return file
