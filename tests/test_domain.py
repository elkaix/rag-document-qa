"""Tests for src.domain — the value objects that cross module seams."""

from __future__ import annotations

import subprocess
import sys

from src.domain import Chunk, Document, SearchResult, content_hash


class TestContentHash:
    def test_is_a_full_sha256_digest(self):
        """A prefix was tried once and reverted: too little entropy, and these
        ids are persisted in SQLite and ChromaDB."""
        assert len(content_hash("hello")) == 64

    def test_is_stable_across_calls(self):
        assert content_hash("hello") == content_hash("hello")

    def test_handles_non_ascii(self):
        assert len(content_hash("ünïcödé 中文")) == 64


class TestDerivedIds:
    def test_document_derives_its_id_from_content(self):
        assert Document(content="abc").doc_id == content_hash("abc")

    def test_explicit_document_id_wins(self):
        assert Document(content="abc", doc_id="given").doc_id == "given"

    def test_chunk_id_mixes_in_the_parent_document(self):
        """The same paragraph in two documents must stay two distinct chunks."""
        a = Chunk(content="same text", doc_id="doc-a")
        b = Chunk(content="same text", doc_id="doc-b")
        assert a.chunk_id != b.chunk_id

    def test_reingesting_identical_content_reuses_ids(self):
        """Idempotent upsert depends on this."""
        assert Chunk(content="x", doc_id="d").chunk_id == Chunk(content="x", doc_id="d").chunk_id


class TestSeamIsVendorFree:
    """Naming the Retriever seam's result type must not import a storage vendor.

    BEFORE: SearchResult lived in src/vector_store.py, the module that does
            `import chromadb`, so ten modules imported the vendor merely to name
            the type at the seam.
    """

    def test_importing_the_value_types_pulls_in_no_vendor(self):
        code = (
            "import sys; import src.domain; "
            "print('chromadb' in sys.modules or 'openai' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "False"

    def test_search_result_carries_a_similarity_not_a_distance(self):
        r = SearchResult(content="c", metadata={}, score=1.0, doc_id="d", chunk_id="c1")
        assert 0.0 <= r.score <= 1.0


class TestAllowedOrigins:
    """CORS parsing — a security-relevant setting, so it gets direct tests."""

    def test_unset_stays_open_for_local_dev(self, monkeypatch):
        from src.config import allowed_origins

        monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
        assert allowed_origins() == ["*"]

    def test_blank_is_treated_as_unset(self, monkeypatch):
        from src.config import allowed_origins

        monkeypatch.setenv("ALLOWED_ORIGINS", "   ")
        assert allowed_origins() == ["*"]

    def test_parses_and_strips_a_comma_separated_list(self, monkeypatch):
        from src.config import allowed_origins

        monkeypatch.setenv("ALLOWED_ORIGINS", " https://a.example , https://b.example ")
        assert allowed_origins() == ["https://a.example", "https://b.example"]

    def test_drops_empty_entries(self, monkeypatch):
        from src.config import allowed_origins

        monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.example,,")
        assert allowed_origins() == ["https://a.example"]
