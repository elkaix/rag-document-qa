"""Document chunking strategies for RAG pipelines.

Chunking is the FIRST step in any RAG system — it breaks documents into
smaller pieces that can be embedded and searched individually.

Why chunking matters:
- LLMs have limited context windows (e.g., 128K tokens)
- Smaller chunks = more precise retrieval
- Overlap prevents context loss at chunk boundaries
"""
import logging
from typing import List
from .config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class FixedSizeChunker:
    """Split documents into fixed-size chunks with optional overlap.

    This is the simplest and most common chunking strategy.
    Good for: Uniform documents, technical docs, code.
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> None:
        self.chunk_size = chunk_size
        self.overlap = min(overlap, chunk_size // 2)  # Safety: overlap < 50%

    def chunk(self, text: str) -> List[dict]:
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({"id": idx, "text": chunk_text, "start": start, "end": end})
                idx += 1
            start = end - self.overlap
        logger.info("Chunked text into %d chunks (size=%d, overlap=%d)", len(chunks), self.chunk_size, self.overlap)
        return chunks


class SentenceChunker:
    """Split documents at sentence boundaries.

    Better than fixed-size for natural language because it preserves
    sentence integrity — a chunk never cuts mid-sentence.
    """

    def __init__(self, max_sentences: int = 10, overlap_sentences: int = 2) -> None:
        self.max_sentences = max_sentences
        self.overlap_sentences = overlap_sentences

    def _split_sentences(self, text: str) -> List[str]:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str) -> List[dict]:
        sentences = self._split_sentences(text)
        chunks = []
        idx = 0
        start = 0
        while start < len(sentences):
            end = min(start + self.max_sentences, len(sentences))
            chunk_text = " ".join(sentences[start:end])
            chunks.append({"id": idx, "text": chunk_text, "start": start, "end": end, "type": "sentence"})
            idx += 1
            start = end - self.overlap_sentences
        logger.info("Sentence-chunked into %d chunks", len(chunks))
        return chunks
