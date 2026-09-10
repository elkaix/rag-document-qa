"""Tests for src.ingestion.loader — path handling and batch error policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion import DocumentLoader


@pytest.fixture
def loader() -> DocumentLoader:
    return DocumentLoader()


class TestLoadOne:
    def test_missing_file_raises(self, loader, tmp_path):
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "absent.txt")

    def test_unsupported_extension_raises(self, loader, tmp_path):
        f = tmp_path / "a.exe"
        f.write_text("x")
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load(f)

    def test_source_metadata_is_attached(self, loader, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("hello world")
        doc = loader.load(f)
        assert doc.metadata["filename"] == "notes.txt"
        assert doc.metadata["file_type"] == "txt"
        assert doc.metadata["file_size_bytes"] == len("hello world")
        assert doc.metadata["file_path"].endswith("notes.txt")

    def test_parser_metadata_is_merged_in(self, loader, tmp_path):
        f = tmp_path / "a.csv"
        f.write_text("h1,h2\n1,2\n")
        doc = loader.load(f)
        assert doc.metadata["row_count"] == 1
        assert doc.metadata["filename"] == "a.csv"

    def test_extension_case_does_not_matter(self, loader, tmp_path):
        f = tmp_path / "a.TXT"
        f.write_text("hello")
        assert loader.load(f).content == "hello"

    def test_the_doc_id_is_content_addressed(self, loader, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("identical")
        b.write_text("identical")
        assert loader.load(a).doc_id == loader.load(b).doc_id


class TestLoadDirectory:
    def test_a_non_directory_raises(self, loader, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("x")
        with pytest.raises(NotADirectoryError):
            loader.load_directory(f)

    def test_loads_every_supported_file(self, loader, tmp_path):
        (tmp_path / "a.txt").write_text("alpha")
        (tmp_path / "b.md").write_text("beta")
        (tmp_path / "skip.exe").write_text("gamma")
        assert len(loader.load_directory(tmp_path)) == 2

    def test_extension_filter_narrows_the_set(self, loader, tmp_path):
        (tmp_path / "a.txt").write_text("alpha")
        (tmp_path / "b.md").write_text("beta")
        docs = loader.load_directory(tmp_path, extensions=[".md"])
        assert [d.metadata["filename"] for d in docs] == ["b.md"]

    def test_recursion_can_be_disabled(self, loader, tmp_path):
        (tmp_path / "top.txt").write_text("top")
        nested = tmp_path / "sub"
        nested.mkdir()
        (nested / "deep.txt").write_text("deep")

        assert len(loader.load_directory(tmp_path, recursive=False)) == 1
        assert len(loader.load_directory(tmp_path, recursive=True)) == 2

    def test_one_bad_file_does_not_abort_the_batch(self, loader, tmp_path, monkeypatch):
        """A bulk upload should index what it can and log the casualty."""
        (tmp_path / "good.txt").write_text("fine")
        (tmp_path / "bad.txt").write_text("boom")

        real_load = DocumentLoader.load

        def sometimes_fails(self, path):
            if Path(path).name == "bad.txt":
                raise OSError("simulated read failure")
            return real_load(self, path)

        monkeypatch.setattr(DocumentLoader, "load", sometimes_fails)

        docs = loader.load_directory(tmp_path)

        assert [d.metadata["filename"] for d in docs] == ["good.txt"]
