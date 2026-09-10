"""
Tests for the ingestion package — loading and chunking.

All tests use local fixtures — no external services required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.domain import Chunk, Document
from src.ingestion import DocumentLoader, TextChunker

# --------------------------------------------------------------------------- #
# DocumentLoader tests                                                         #
# --------------------------------------------------------------------------- #


class TestDocumentLoader:
    """Tests for DocumentLoader.load() and load_directory()."""

    def test_load_text_basic(self, tmp_text_file: Path) -> None:
        """load() on a .txt file returns a Document with non-empty content."""
        loader = DocumentLoader()
        doc = loader.load(tmp_text_file)

        assert isinstance(doc, Document)
        assert len(doc.content) > 0
        assert doc.doc_id  # hash-based ID should be populated

    def test_load_text_content_matches_file(self, tmp_text_file: Path) -> None:
        """Content of the loaded Document matches the file's content."""
        loader = DocumentLoader()
        doc = loader.load(tmp_text_file)
        expected = tmp_text_file.read_text(encoding="utf-8")
        assert doc.content == expected

    def test_load_text_metadata(self, tmp_text_file: Path) -> None:
        """Metadata includes filename, file_type, and file_size_bytes."""
        loader = DocumentLoader()
        doc = loader.load(tmp_text_file)

        assert doc.metadata["filename"] == tmp_text_file.name
        assert doc.metadata["file_type"] == "txt"
        assert doc.metadata["file_size_bytes"] == tmp_text_file.stat().st_size

    def test_load_json_file(self, tmp_json_file: Path) -> None:
        """JSON files are loaded and content is pretty-printed JSON text."""
        loader = DocumentLoader()
        doc = loader.load(tmp_json_file)

        assert doc.content  # should have content
        # Valid JSON in the output
        parsed = json.loads(doc.content)
        assert parsed["title"] == "Test Document"

    def test_load_csv_file(self, tmp_csv_file: Path) -> None:
        """CSV files are loaded as structured text with headers."""
        loader = DocumentLoader()
        doc = loader.load(tmp_csv_file)

        assert "name" in doc.content
        assert "RAG" in doc.content
        assert doc.metadata["row_count"] == 3
        assert doc.metadata["column_count"] == 3

    def test_load_unsupported_type_raises(self, tmp_path: Path) -> None:
        """Unsupported file extension raises ValueError."""
        bad_file = tmp_path / "archive.zip"
        bad_file.write_bytes(b"PK\x03\x04")

        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load(bad_file)

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """Non-existent file raises FileNotFoundError."""
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "does_not_exist.txt")

    def test_load_directory(self, tmp_path: Path) -> None:
        """load_directory() loads all supported files."""
        # Create multiple files
        (tmp_path / "a.txt").write_text("Content A", encoding="utf-8")
        (tmp_path / "b.txt").write_text("Content B", encoding="utf-8")
        (tmp_path / "c.json").write_text('{"x": 1}', encoding="utf-8")
        (tmp_path / "skip.zip").write_bytes(b"PK")  # should be skipped

        loader = DocumentLoader()
        docs = loader.load_directory(tmp_path)

        assert len(docs) == 3
        filenames = {d.metadata["filename"] for d in docs}
        assert "a.txt" in filenames
        assert "b.txt" in filenames
        assert "c.json" in filenames
        assert "skip.zip" not in filenames

    def test_load_directory_recursive(self, tmp_path: Path) -> None:
        """load_directory() with recursive=True finds files in subdirectories."""
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "root.txt").write_text("root", encoding="utf-8")
        (sub / "nested.txt").write_text("nested", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load_directory(tmp_path, recursive=True)
        filenames = {d.metadata["filename"] for d in docs}
        assert "root.txt" in filenames
        assert "nested.txt" in filenames

    def test_metadata_extraction_file_path(self, tmp_text_file: Path) -> None:
        """Metadata includes the resolved file path."""
        loader = DocumentLoader()
        doc = loader.load(tmp_text_file)
        assert "file_path" in doc.metadata
        assert Path(doc.metadata["file_path"]).is_absolute()


# --------------------------------------------------------------------------- #
# TextChunker tests                                                            #
# --------------------------------------------------------------------------- #


class TestTextChunkerFixed:
    """Tests for fixed-size chunking strategy."""

    def test_chunk_fixed_basic(self, sample_document: Document) -> None:
        """Fixed chunking produces non-empty chunks."""
        chunker = TextChunker(chunk_size=200, chunk_overlap=20, strategy="fixed")
        chunks = chunker.chunk(sample_document)

        assert len(chunks) > 0
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.content.strip()

    def test_chunk_fixed_max_size(self, sample_document: Document) -> None:
        """Fixed chunks do not exceed chunk_size + overlap."""
        chunk_size = 200
        overlap = 20
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=overlap, strategy="fixed")
        chunks = chunker.chunk(sample_document)

        for c in chunks:
            # Fixed chunks should be at most chunk_size chars (last may be shorter)
            assert len(c.content) <= chunk_size + overlap + 5  # small tolerance

    def test_chunk_fixed_doc_id_preserved(self, sample_document: Document) -> None:
        """Each chunk has the parent doc_id."""
        chunker = TextChunker(chunk_size=200, strategy="fixed")
        chunks = chunker.chunk(sample_document)

        for c in chunks:
            assert c.doc_id == sample_document.doc_id

    def test_chunk_fixed_unique_ids(self, sample_document: Document) -> None:
        """Each chunk has a unique chunk_id."""
        chunker = TextChunker(chunk_size=200, strategy="fixed")
        chunks = chunker.chunk(sample_document)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_fixed_metadata_index(self, sample_document: Document) -> None:
        """Chunks include chunk_index in metadata."""
        chunker = TextChunker(chunk_size=300, strategy="fixed")
        chunks = chunker.chunk(sample_document)
        for i, c in enumerate(chunks):
            assert c.metadata["chunk_index"] == i


class TestTextChunkerRecursive:
    """Tests for recursive chunking strategy."""

    def test_chunk_recursive_basic(self, sample_document: Document) -> None:
        """Recursive chunking produces chunks."""
        chunker = TextChunker(chunk_size=300, chunk_overlap=30, strategy="recursive")
        chunks = chunker.chunk(sample_document)
        assert len(chunks) > 0

    def test_chunk_recursive_respects_size(self, sample_document: Document) -> None:
        """Most recursive chunks are at or near chunk_size."""
        chunker = TextChunker(chunk_size=300, chunk_overlap=30, strategy="recursive")
        chunks = chunker.chunk(sample_document)
        # The last chunk may be shorter, but no chunk should far exceed chunk_size + overlap
        for c in chunks:
            assert len(c.content) <= 300 + 50  # reasonable upper bound with overlap

    def test_chunk_recursive_short_document(self) -> None:
        """A short document produces a single chunk."""
        short_text = "This is a very short document."
        doc = Document(content=short_text, metadata={})
        chunker = TextChunker(chunk_size=500, strategy="recursive")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert short_text in chunks[0].content

    def test_chunk_recursive_empty_document(self) -> None:
        """Empty document produces no chunks."""
        doc = Document(content="   ", metadata={})
        chunker = TextChunker(chunk_size=200, strategy="recursive")
        chunks = chunker.chunk(doc)
        assert chunks == []


class TestTextChunkerDocuments:
    """Tests for chunk_documents() bulk method."""

    def test_chunk_documents_multiple(
        self, sample_document: Document, sample_document_2: Document
    ) -> None:
        """chunk_documents() produces chunks from all documents."""
        chunker = TextChunker(chunk_size=200, strategy="fixed")
        chunks = chunker.chunk_documents([sample_document, sample_document_2])

        doc1_ids = {c.doc_id for c in chunks if c.doc_id == sample_document.doc_id}
        doc2_ids = {c.doc_id for c in chunks if c.doc_id == sample_document_2.doc_id}
        assert doc1_ids
        assert doc2_ids

    def test_chunk_documents_empty_list(self) -> None:
        """chunk_documents([]) returns an empty list."""
        chunker = TextChunker(chunk_size=200, strategy="fixed")
        assert chunker.chunk_documents([]) == []


class TestTextChunkerValidation:
    """Validation and edge-case tests."""

    def test_invalid_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            TextChunker(chunk_size=0)

    def test_invalid_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            TextChunker(chunk_size=100, chunk_overlap=200)

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="strategy"):
            TextChunker(strategy="unknown")


class TestChunkQualityFilters:
    """The filters that decide a chunk is not worth indexing.

    Both were documented but untested; the review listed them among the
    module's uncovered paths.
    """

    def _doc(self, content: str) -> Document:
        return Document(content=content, metadata={"filename": "f.txt"})

    def test_chunks_shorter_than_the_floor_are_dropped(self):
        """Short chunks are almost always PDF artifacts — page numbers, labels."""
        chunker = TextChunker(chunk_size=64, chunk_overlap=0, strategy="fixed")
        assert chunker.chunk(self._doc("109")) == []

    def test_a_chunk_exactly_at_the_floor_is_kept(self):
        chunker = TextChunker(chunk_size=64, chunk_overlap=0, strategy="fixed")
        chunks = chunker.chunk(self._doc("x" * TextChunker.MIN_CHUNK_LENGTH))
        assert len(chunks) == 1

    def test_table_of_contents_dot_leaders_are_dropped(self):
        """PDF contents pages extract as dot-filled lines and match everything."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=0, strategy="fixed")
        toc = "Introduction . . . . . . . . . . . . . . . . . . . . . . . . 42"
        assert chunker.chunk(self._doc(toc)) == []

    def test_ordinary_prose_with_full_stops_survives(self):
        """Content is ~5% dots; a contents page is >20% — the filter sits between."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=0, strategy="fixed")
        prose = (
            "Retrieval augmented generation works in two steps. First it "
            "retrieves. Then it generates. This is a normal paragraph."
        )
        assert len(chunker.chunk(self._doc(prose))) == 1


class TestSemanticStrategy:
    """The third chunking tier, which no test constructed before."""

    def _doc(self, content: str) -> Document:
        return Document(content=content, metadata={})

    def test_semantic_is_an_accepted_strategy(self):
        assert TextChunker(strategy="semantic").strategy == "semantic"

    def test_it_produces_chunks_and_labels_them(self):
        chunker = TextChunker(chunk_size=120, chunk_overlap=20, strategy="semantic")
        text = " ".join(
            f"This is sentence number {i} about retrieval augmented generation." for i in range(12)
        )
        chunks = chunker.chunk(self._doc(text))
        assert chunks
        assert all(c.metadata["chunk_strategy"] == "semantic" for c in chunks)

    def test_it_respects_the_chunk_size_budget(self):
        chunker = TextChunker(chunk_size=150, chunk_overlap=20, strategy="semantic")
        text = " ".join(f"Sentence {i} carries some content." for i in range(20))
        for chunk in chunker.chunk(self._doc(text)):
            assert len(chunk.content) <= 300, "a chunk should not run far past the budget"

    def test_an_unknown_strategy_is_rejected(self):
        with pytest.raises(ValueError, match="strategy must be"):
            TextChunker(strategy="nonsense")


class TestWordOverlap:
    """The overlap helper whose docstring documents a specific bug it fixes."""

    def test_overlap_does_not_split_a_word(self):
        chunker = TextChunker(chunk_size=80, chunk_overlap=20, strategy="recursive")
        pieces = ["alpha beta gamma delta epsilon", "zeta eta theta iota kappa"]
        overlapped = chunker._apply_word_overlap(pieces)
        assert len(overlapped) == 2
        # every token in the result must be a whole word from the input
        words = set(" ".join(pieces).split())
        assert all(tok in words for tok in overlapped[1].split())

    def test_the_first_chunk_is_never_prefixed(self):
        chunker = TextChunker(chunk_size=80, chunk_overlap=20, strategy="recursive")
        pieces = ["alpha beta gamma", "delta epsilon zeta"]
        assert chunker._apply_word_overlap(pieces)[0] == "alpha beta gamma"

    def test_a_single_chunk_is_returned_unchanged(self):
        chunker = TextChunker(chunk_size=80, chunk_overlap=20, strategy="recursive")
        assert chunker._apply_word_overlap(["only one"]) == ["only one"]

    def test_zero_overlap_leaves_chunks_alone(self):
        chunker = TextChunker(chunk_size=80, chunk_overlap=0, strategy="recursive")
        pieces = ["alpha beta", "gamma delta"]
        assert chunker._apply_word_overlap(pieces) == pieces


class TestChunkerValidation:
    def test_chunk_size_must_be_positive(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextChunker(chunk_size=0)

    def test_overlap_must_be_smaller_than_the_chunk(self):
        with pytest.raises(ValueError, match="chunk_overlap must be"):
            TextChunker(chunk_size=100, chunk_overlap=100)

    def test_overlap_cannot_be_negative(self):
        with pytest.raises(ValueError, match="chunk_overlap must be"):
            TextChunker(chunk_size=100, chunk_overlap=-1)
