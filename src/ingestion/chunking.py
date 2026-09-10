"""Chunking strategies — how a document becomes retrievable slices.

RAG Pipeline Position:
    Document -> [CHUNKING] -> Chunk -> Embedding -> Vector Store
                    ^^^

What concept it teaches:
    Chunking is a retrieval-quality lever, not a formatting detail. Too small
    and a chunk loses the context that makes it answerable; too large and the
    embedding averages several topics into one vector that matches none of them
    well. Three tiers are offered so the trade-off can be measured rather than
    assumed.

Why this is separate from loading:
    Parsing and chunking shared a 504-line module and nothing else — not a
    function call in either direction, only the value types. They change for
    entirely different reasons: adding a format touches parsing, tuning
    retrieval quality touches chunking.

Design Decision:
    Quality filters (a minimum length, a table-of-contents detector) live with
    chunking rather than with parsing, because what counts as a useless chunk
    depends on the chunk size, not on the source format.
"""

from __future__ import annotations

import logging

from src.domain import Chunk, Document

logger = logging.getLogger(__name__)


class TextChunker:
    """Splits documents into overlapping chunks for embedding."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        strategy: str = "recursive",
        separators: list[str] | None = None,
    ) -> None:
        """
        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Number of overlapping characters between chunks.
            strategy: 'fixed', 'recursive', or 'semantic'.
            separators: Custom separators for recursive strategy.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        if strategy not in ("fixed", "recursive", "semantic"):
            raise ValueError("strategy must be 'fixed', 'recursive', or 'semantic'")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    # WHY 20 chars: shorter chunks are almost always PDF artifacts — page
    # numbers ("109"), stray headers, or section labels.  They carry no
    # semantic value and pollute retrieval results with false matches.
    MIN_CHUNK_LENGTH = 20

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a Document into chunks.

        Args:
            document: Document to split.

        Returns:
            List of Chunk objects.
        """
        if self.strategy == "fixed":
            raw_chunks = self._fixed_chunk(document.content)
        elif self.strategy == "recursive":
            # WHY overlap is applied here instead of inside _recursive_chunk:
            # The recursive splitter calls itself at multiple depths.  If
            # overlap were applied at each depth it would cascade — the tail
            # of a depth-1 chunk (already overlapped) gets overlapped again
            # at depth 0, tripling text.  Applying once at the top avoids this.
            raw_chunks = self._recursive_chunk(document.content)
            raw_chunks = self._apply_word_overlap(raw_chunks)
        else:  # semantic
            raw_chunks = self._semantic_chunk(document.content)

        chunks: list[Chunk] = []
        for idx, text in enumerate(raw_chunks):
            stripped = text.strip()
            if not stripped or len(stripped) < self.MIN_CHUNK_LENGTH:
                continue
            # Filter ToC dot-leader chunks (". . . . . . . . . 42").
            # WHY: PDF tables of contents extract as dot-filled lines
            # mixed with section titles.  They carry no semantic value
            # and pollute retrieval.  Content chunks have < 5% dots;
            # ToC chunks have > 20% dots — a clean bimodal split.
            dot_ratio = stripped.count(".") / len(stripped)
            if dot_ratio > 0.15:
                continue
            meta = {**document.metadata, "chunk_index": idx, "chunk_strategy": self.strategy}
            chunks.append(Chunk(content=stripped, metadata=meta, doc_id=document.doc_id))

        logger.debug(
            "Chunked document %s into %d chunks (strategy=%s)",
            document.doc_id,
            len(chunks),
            self.strategy,
        )
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk multiple documents.

        Args:
            documents: List of Document objects.

        Returns:
            Flattened list of all Chunk objects.
        """
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk(doc))
        logger.info("Total chunks from %d documents: %d", len(documents), len(all_chunks))
        return all_chunks

    # ------------------------------------------------------------------ #
    # Chunking strategies                                                  #
    # ------------------------------------------------------------------ #

    def _fixed_chunk(self, text: str) -> list[str]:
        """Split text into fixed-size character windows with overlap."""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _recursive_chunk(self, text: str, depth: int = 0) -> list[str]:
        """Recursively split text using a hierarchy of separators.

        Splits on the current-depth separator, merges small parts into
        chunks up to ``chunk_size``, then applies word-boundary-safe
        overlap via ``_apply_word_overlap``.
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if depth >= len(self.separators):
            return self._fixed_chunk(text)

        sep = self.separators[depth]
        if sep == "":
            return self._fixed_chunk(text)

        parts = text.split(sep)
        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for part in parts:
            added_len = len(part) + (len(sep) if current_parts else 0)

            if current_len + added_len <= self.chunk_size:
                current_parts.append(part)
                current_len += added_len
            else:
                if current_parts:
                    committed = sep.join(current_parts)
                    if len(committed) > self.chunk_size:
                        chunks.extend(self._recursive_chunk(committed, depth + 1))
                    else:
                        chunks.append(committed)

                current_parts = [part]
                current_len = len(part)

        if current_parts:
            remaining = sep.join(current_parts)
            if len(remaining) > self.chunk_size:
                chunks.extend(self._recursive_chunk(remaining, depth + 1))
            else:
                chunks.append(remaining)

        return chunks

    def _apply_word_overlap(self, chunks: list[str]) -> list[str]:
        """Prepend the trailing words of chunk N to chunk N+1.

        BEFORE (broken _apply_overlap):
            Sliced last N raw *characters* and concatenated with no separator,
            producing "fine-tuningsystems." and doubled content.

        AFTER:
            Takes the last ``chunk_overlap`` characters, snaps *forward* to the
            nearest word boundary (first space), and prepends with ``" ... "``
            as a visual separator.  Result is always clean, readable text.

        WHY word-boundary snapping:
            Character-level slicing can cut mid-word ("optimisa|tion").
            Snapping to the next space guarantees whole words.
        """
        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return chunks

        result: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            # Grab roughly chunk_overlap chars from the end of previous chunk
            raw_tail = prev[-self.chunk_overlap :]
            # Snap forward to the nearest word boundary (skip partial word)
            space_idx = raw_tail.find(" ")
            if space_idx != -1 and space_idx < len(raw_tail) - 1:
                tail = raw_tail[space_idx + 1 :]
            else:
                # The tail is a single long word — use it as-is
                tail = raw_tail
            tail = tail.strip()
            if tail:
                result.append(tail + " " + chunks[i])
            else:
                result.append(chunks[i])
        return result

    def _semantic_chunk(self, text: str) -> list[str]:
        """Sentence-aware chunking: accumulate sentences until chunk_size is exceeded."""
        import re

        # Split on sentence boundaries
        sentence_endings = re.compile(r"(?<=[.!?])\s+")
        sentences = sentence_endings.split(text)

        chunks: list[str] = []
        current_sentences: list[str] = []
        current_len = 0

        for sentence in sentences:
            s_len = len(sentence)
            if current_len + s_len > self.chunk_size and current_sentences:
                chunks.append(" ".join(current_sentences))
                # keep overlap
                overlap_sentences: list[str] = []
                overlap_len = 0
                for sent in reversed(current_sentences):
                    if overlap_len + len(sent) <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_len += len(sent)
                    else:
                        break
                current_sentences = overlap_sentences
                current_len = overlap_len

            current_sentences.append(sentence)
            current_len += s_len

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return [c for c in chunks if c.strip()]
