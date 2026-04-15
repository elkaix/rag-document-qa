"""End-to-end RAG pipeline — combines all components.

This is the orchestration layer that ties chunking, embedding,
retrieval, and generation into a single query flow.
"""
import logging
from typing import List, Dict, Any
from .chunker import FixedSizeChunker
from .embeddings import TfidfEmbedder
from .retriever import Retriever
from .generator import ResponseGenerator

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50, top_k: int = 5) -> None:
        self.chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedder = TfidfEmbedder()
        self.generator = ResponseGenerator()
        self._chunks: List[Dict] = []
        self._retriever = None
        self.top_k = top_k

    def ingest(self, documents: List[str]) -> Dict[str, int]:
        """Process documents into searchable chunks."""
        all_chunks = []
        for i, doc in enumerate(documents):
            chunks = self.chunker.chunk(doc)
            for c in chunks:
                c["doc_id"] = i
            all_chunks.extend(chunks)
        self._chunks = all_chunks
        self._retriever = Retriever(self.embedder, self._chunks, top_k=self.top_k)
        self._retriever.build_index()
        logger.info("Ingested %d documents → %d chunks", len(documents), len(self._chunks))
        return {"documents": len(documents), "chunks": len(self._chunks)}

    def query(self, question: str) -> Dict[str, Any]:
        """Process a query through the full RAG pipeline."""
        if not self._retriever:
            return {"error": "No documents ingested. Call ingest() first."}
        results = self._retriever.retrieve(question)
        response = self.generator.generate(question, results)
        return response
