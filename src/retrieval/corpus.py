"""ChunkCorpus — the whole indexed corpus, kept in sync with the vector store.

RAG Pipeline Position:
    Vector Store -> [CHUNK CORPUS] -> BM25 index -> BM25HybridRetriever
                         ^^^
    Dense retrieval asks the store for the *nearest* chunks. Sparse retrieval
    needs *every* chunk, up front, to build a term-frequency index over. This
    module is the one place that materialises that corpus and decides when the
    materialised copy has gone stale.

What concept it teaches:
    Cache invalidation by published revision. The store counts its own writes
    (``ChromaVectorStore.revision``); anything derived from the corpus records
    the revision it was built at and rebuilds when the two differ.

Why this approach over alternatives:
    ADR 0004 deferred the ``hybrid`` strategy because ``BM25HybridRetriever``
    took a ``dict[str, str]`` snapshot at construction time — the corpus was
    frozen at whatever had been ingested when the process booted, so every
    later upload was invisible to the sparse side and every deletion left
    tombstones that sparse retrieval would still surface. Two fixes were
    considered:

      - **Explicit invalidation**: have ingest and delete call ``invalidate()``.
        Correct only as long as every current *and future* mutation site
        remembers to, which is the class of bug that does not show up in tests.
      - **Published revision** (chosen): the store already funnels every write
        through two methods, so it can count them itself. Derived indexes ask.
        A new mutation method cannot forget, because the counter lives with the
        write, not with the caller.

Design Decision:
    ``snapshot()`` returns the *same object* while the corpus is unchanged, so
    a consumer that builds something expensive from it (the BM25 index) can
    cache on identity — ``if snap is not self._built_from`` — with no second
    copy of the freshness rule.

TRADE-OFF: the first query after an upload pays a full collection read, so
    ingestion moves a latency cost onto the next query rather than onto the
    upload. That is the right side of the trade for a corpus that is read far
    more often than it is written, but it is a cliff, not a constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.domain import Chunk
from src.vector_store import ChromaVectorStore


@dataclass(frozen=True)
class CorpusSnapshot:
    """Every indexed chunk as of one store revision.

    Attributes:
        revision: The ``ChromaVectorStore.revision`` this was read at.
        chunks: Mapping of chunk_id to Chunk for the whole collection.
    """

    revision: int
    chunks: dict[str, Chunk] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the number of chunks in the snapshot."""
        return len(self.chunks)


class ChunkCorpus:
    """A lazily-materialised, self-refreshing view of the whole collection."""

    def __init__(self, vector_store: ChromaVectorStore) -> None:
        """Track a vector store without reading it yet.

        Args:
            vector_store: The collection to mirror. Nothing is read until the
                first :meth:`snapshot` call, so constructing a corpus is free
                and safe at application startup, before anything is ingested.
        """
        self._vector_store = vector_store
        # WHY None and not ``CorpusSnapshot(revision=0)``: a persistent
        #     collection that already holds documents starts at revision 0 too.
        #     Seeding an empty snapshot at that revision would make the corpus
        #     look fresh-and-empty, and hybrid retrieval would silently serve
        #     dense-only results until the first write of the process.
        self._snapshot: CorpusSnapshot | None = None

    def snapshot(self) -> CorpusSnapshot:
        """Return the corpus, re-reading it only if the store has been written.

        Returns:
            The current snapshot. Repeated calls return the *identical* object
            while the store's revision is unchanged, so consumers may cache
            derived values on its identity.
        """
        revision = self._vector_store.revision
        cached = self._snapshot
        if cached is not None and cached.revision == revision:
            return cached

        fresh = CorpusSnapshot(revision=revision, chunks=self._vector_store.all_chunks())
        self._snapshot = fresh
        return fresh
