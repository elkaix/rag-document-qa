"""Value objects that cross module seams — the vocabulary of the pipeline.

RAG Pipeline Position:
    Document -> Chunk -> Embeddings -> Vector Store -> SearchResult -> Answer
    ^^^^^^^^    ^^^^^                                  ^^^^^^^^^^^^
    Every arrow above carries one of these types. They are the currency the
    modules trade in, so they belong to none of them.

What concept it teaches:
    A leaf module. It imports nothing from this package, so anything may import
    it without creating a cycle or dragging a dependency along.

Why this approach over alternatives:
    ``SearchResult`` used to live in ``src/vector_store.py``, the module that
    does ``import chromadb``. Because ``SearchResult`` is what every Retriever
    returns, ten modules — the whole ``retrieval`` package, the whole
    ``query_engine`` package, and the eval pipeline — had to import the storage
    vendor merely to *name* the type at the seam. The seam could not be
    described without the implementation behind it.

    ``Document`` and ``Chunk`` had the same shape of problem one step earlier:
    naming a chunk meant importing the file-parsing module.

Design Decision:
    Plain frozen-by-convention dataclasses, not Pydantic models. These cross
    internal seams where both sides are trusted; validation belongs at the API
    boundary, which has its own Pydantic schemas.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


def content_hash(text: str) -> str:
    """Return a stable content-addressed id for a piece of text.

    Args:
        text: The content to identify.

    Returns:
        The full SHA-256 hex digest.

    WHY content-addressed: re-ingesting the same document must produce the same
        ids so the upsert is idempotent rather than duplicating chunks.

    WHY the full digest and not a prefix: an earlier version truncated to 16 hex
        characters, which is too little entropy for a content-addressed id as a
        corpus grows, and DocumentRecord's contract promises a full SHA-256.
        These ids are persisted in SQLite and ChromaDB, so shortening them would
        also orphan every stored chunk.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Document:
    """One loaded source document, before chunking.

    Attributes:
        content: The extracted plain text.
        metadata: Source facts — filename, page count, and so on.
        doc_id: Content-addressed identifier, derived when not supplied.
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = content_hash(self.content)


@dataclass
class Chunk:
    """One retrievable slice of a Document.

    Attributes:
        content: The chunk text sent to the embedder and shown to the LLM.
        metadata: Inherited source facts plus the chunk's own index.
        chunk_id: Content-addressed identifier, derived when not supplied.
        doc_id: The parent document's identifier.

    WHY the id mixes in doc_id: two documents can legitimately contain the same
        paragraph, and they must stay distinct chunks.
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(default="")
    doc_id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = content_hash(self.content + self.doc_id)


@dataclass
class SearchResult:
    """One chunk returned by a Retriever, with its relevance score.

    Attributes:
        content: The raw chunk text shown to the LLM as context.
        metadata: Source facts — filename, page, chunk_index.
        score: Relevance, oriented so higher is better. The *scale* is the
            producing strategy's own and is NOT comparable across
            strategies: dense retrieval reports a cosine similarity in
            [0, 1], the cross-encoder reranker reports a raw logit
            (roughly -11..+11), and hybrid retrieval reports 0.0 for a
            sparse-only hit because no cosine similarity exists for one.
            See ``Retriever.retrieve`` for what the seam does and does not
            guarantee about ordering.
        doc_id: The document this chunk came from.
        chunk_id: This chunk's identifier.

    WHY a dataclass rather than a TypedDict: attribute access (``result.score``)
        type-checks and reads better than string keys, and the repr is useful
        when a retrieval chain is being debugged.

    WHY the score is a similarity, not a distance: every Retriever presents the
        same orientation — higher is better — so composing adapters (reranking
        over dense, multi-query over either) never has to ask which convention
        an inner Retriever used. Converting from a store's native distance is
        that store's job.
    """

    content: str
    metadata: dict[str, Any]
    score: float
    doc_id: str
    chunk_id: str
