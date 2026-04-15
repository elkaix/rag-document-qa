"""
Unified RAG backend — stateful facade that wires all pipeline components.

Used by both the FastAPI API and the Streamlit frontend.
Manages document ingestion, index rebuilding, querying, and deletion.
"""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .document_loader import SUPPORTED_EXTENSIONS, Document, DocumentLoader, TextChunker
from .embeddings import TfidfEmbedder
from .llm_handler import LLMHandler
# NOTE: Task 6 will fully rewrite RAGBackend to use ChromaVectorStore.
# For now we import only what still exists after the Task 3 vector store replacement.
from .vector_store import ChromaVectorStore, SearchResult

logger = logging.getLogger(__name__)


class RAGBackend:
    """Stateful RAG pipeline that persists across requests / Streamlit reruns.

    Lifecycle:
        1. ``ingest_file()`` or ``ingest_bytes()`` — load, chunk, embed, store
        2. ``query()`` — retrieve relevant chunks, generate answer via LLM
        3. ``delete_document()`` — remove a document and rebuild index
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        top_k: int = 5,
        llm_model: str = "glm-5.1",
        max_tokens: int = 4096,
        chroma_collection: Any = None,
    ) -> None:
        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy="recursive",
        )
        self.embedder = TfidfEmbedder()

        # WHY: Task 6 will replace TF-IDF + ChromaVectorStore with a proper
        # ChromaDB-native backend. For now, use a temporary EphemeralClient if
        # no collection is provided, so the class remains instantiable.
        if chroma_collection is not None:
            self.vector_store = ChromaVectorStore(collection=chroma_collection)
        else:
            import chromadb as _chromadb
            _client = _chromadb.EphemeralClient()
            _col = _client.get_or_create_collection(
                name="rag_backend_tmp",
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )
            self.vector_store = ChromaVectorStore(collection=_col)

        self.llm = LLMHandler(model=llm_model, max_tokens=max_tokens)
        self.top_k = top_k

        # Internal registries
        self._documents: Dict[str, Dict[str, Any]] = {}  # doc_id → metadata
        self._all_chunks: list = []  # list of Chunk objects
        self._index_built = False

    # ------------------------------------------------------------------ #
    # Ingestion                                                            #
    # ------------------------------------------------------------------ #

    def ingest_file(
        self,
        file_path: str | Path,
        original_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load a file from disk, chunk, embed, and index it.

        Returns:
            Dict with doc_id, filename, chunks_count, status.
        """
        document = self.loader.load(file_path)
        if original_filename:
            document.metadata["filename"] = original_filename
        return self._ingest_document(document)

    def ingest_bytes(
        self,
        filename: str,
        data: bytes,
    ) -> Dict[str, Any]:
        """Ingest raw file bytes (e.g. from an upload).

        Writes to a temp file, loads via DocumentLoader, then cleans up.
        """
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            return self.ingest_file(tmp_path, original_filename=filename)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _ingest_document(self, document: Document) -> Dict[str, Any]:
        """Chunk a Document, rebuild the index, and register metadata."""
        chunks = self.chunker.chunk(document)
        self._all_chunks.extend(chunks)
        self._rebuild_index()

        meta = {
            "doc_id": document.doc_id,
            "filename": document.metadata.get("filename", "unknown"),
            "chunks": len(chunks),
            "upload_date": datetime.now(timezone.utc).isoformat(),
            "file_type": document.metadata.get("file_type", ""),
            "file_size_bytes": document.metadata.get("file_size_bytes", 0),
        }
        self._documents[document.doc_id] = meta

        logger.info(
            "Ingested '%s' → doc_id=%s (%d chunks)",
            meta["filename"],
            document.doc_id,
            len(chunks),
        )
        return {
            "doc_id": document.doc_id,
            "filename": meta["filename"],
            "chunks_count": len(chunks),
            "status": "success",
        }

    # ------------------------------------------------------------------ #
    # Index management                                                     #
    # ------------------------------------------------------------------ #

    def _rebuild_index(self) -> None:
        """Re-embed all chunks and rebuild the vector store.

        TF-IDF is a corpus-level method — vocabulary and IDF weights change
        when documents are added or removed, so we must re-embed everything.

        NOTE: Task 5 will remove TF-IDF; Task 6 will replace this entire backend
        with one that uses ChromaDB's built-in embedding function so rebuild is
        no longer needed. This implementation bridges Tasks 3 and 6.
        """
        if not self._all_chunks:
            self._index_built = False
            return

        texts = [c.content for c in self._all_chunks]
        embeddings = self.embedder.fit_transform(texts)

        # BEFORE (Task 3): InMemoryVectorStore().add_documents(docs, embeddings)
        # AFTER  (Task 3): ChromaVectorStore.upsert(ids, documents, metadatas, embeddings)
        # WHY: ChromaVectorStore uses ChromaDB's upsert API which requires explicit
        #      IDs, documents, metadatas, and embeddings as separate lists.

        # Recreate the ChromaDB collection on each rebuild (TF-IDF vocab changes)
        import chromadb as _chromadb
        _client = _chromadb.EphemeralClient()
        _col = _client.get_or_create_collection(
            name="rag_backend_tmp",
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        self.vector_store = ChromaVectorStore(collection=_col)

        self.vector_store.upsert(
            ids=[c.chunk_id for c in self._all_chunks],
            documents=[c.content for c in self._all_chunks],
            metadatas=[{**c.metadata, "doc_id": c.doc_id} for c in self._all_chunks],
            embeddings=embeddings.tolist(),
        )
        self._index_built = True

        logger.info("Rebuilt index: %d chunks, %d dimensions", len(texts), embeddings.shape[1])

    # ------------------------------------------------------------------ #
    # Querying                                                             #
    # ------------------------------------------------------------------ #

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a full RAG query: retrieve → build context → generate answer.

        Returns:
            Dict with answer, sources, confidence.
        """
        if not self._index_built:
            return {
                "answer": "No documents indexed yet. Please upload documents first.",
                "sources": [],
                "confidence": 0.0,
            }

        k = top_k or self.top_k
        query_emb = self.embedder.transform([question])[0].tolist()
        results = self.vector_store.query(query_embedding=query_emb, top_k=k)

        # Build context string
        context = "\n\n".join(
            f"[{r.metadata.get('filename', 'unknown')}] {r.content}"
            for r in results
        )

        # Generate answer (create a per-query handler if model differs)
        handler = self.llm
        if model and model != self.llm.model:
            handler = LLMHandler(model=model)

        answer = handler.generate_with_context(question, context)

        sources = [
            {
                "doc_id": r.doc_id,
                "chunk_id": r.chunk_id,
                "filename": r.metadata.get("filename"),
                "score": round(r.score, 4),
                "excerpt": r.content[:300],
                "chunk_index": r.metadata.get("chunk_index"),
            }
            for r in results
        ]

        # Confidence: clamped average of top-3 scores
        if results:
            top_scores = [r.score for r in results[: min(3, len(results))]]
            confidence = max(0.0, min(1.0, sum(top_scores) / len(top_scores)))
        else:
            confidence = 0.0

        return {
            "answer": answer,
            "sources": sources,
            "confidence": round(confidence, 4),
        }

    def stream_query(
        self,
        question: str,
        top_k: Optional[int] = None,
        model: Optional[str] = None,
    ):
        """Retrieve context and yield LLM tokens, then return sources.

        Yields:
            Tuples of ("token", str) for each token, then ("done", sources_list).
        """
        if not self._index_built:
            yield ("token", "No documents indexed yet. Please upload documents first.")
            yield ("done", [])
            return

        k = top_k or self.top_k
        query_emb = self.embedder.transform([question])[0].tolist()
        results = self.vector_store.query(query_embedding=query_emb, top_k=k)

        context = "\n\n".join(
            f"[{r.metadata.get('filename', 'unknown')}] {r.content}"
            for r in results
        )

        handler = self.llm
        if model and model != self.llm.model:
            handler = LLMHandler(model=model)

        system_prompt = (
            "You are a helpful assistant. Answer the user's question based solely on the "
            "provided context. If the context does not contain enough information, say so."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        for token in handler.stream_response(user_prompt, system_prompt=system_prompt):
            yield ("token", token)

        sources = [
            {
                "doc_id": r.doc_id,
                "chunk_id": r.chunk_id,
                "filename": r.metadata.get("filename"),
                "score": round(r.score, 4),
                "excerpt": r.content[:300],
            }
            for r in results
        ]
        yield ("done", sources)

    # ------------------------------------------------------------------ #
    # Document management                                                  #
    # ------------------------------------------------------------------ #

    def delete_document(self, doc_id: str) -> int:
        """Remove a document and all its chunks, then rebuild the index."""
        before = len(self._all_chunks)
        self._all_chunks = [c for c in self._all_chunks if c.doc_id != doc_id]
        deleted = before - len(self._all_chunks)

        self._documents.pop(doc_id, None)
        self._rebuild_index()

        logger.info("Deleted doc_id=%s (%d chunks removed)", doc_id, deleted)
        return deleted

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return metadata for all indexed documents."""
        docs = list(self._documents.values())
        docs.sort(key=lambda d: d.get("upload_date", ""), reverse=True)
        return docs

    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Return all chunks for a specific document."""
        return [
            {
                "chunk_id": c.chunk_id,
                "index": c.metadata.get("chunk_index", 0),
                "text": c.content,
                "metadata": c.metadata,
            }
            for c in self._all_chunks
            if c.doc_id == doc_id
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics."""
        total_size = sum(
            d.get("file_size_bytes", 0) for d in self._documents.values()
        )
        return {
            "total_docs": len(self._documents),
            "total_chunks": len(self._all_chunks),
            "index_size_mb": round(total_size / (1024 * 1024), 2),
            "index_built": self._index_built,
        }
