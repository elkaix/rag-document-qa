"""Tests for src.eval.datasets.ml_papers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.eval.datasets.ml_papers import (
    DEFAULT_QUESTIONS_PATH,
    DEFAULT_MANIFEST_PATH,
    ManifestVerificationError,
    load_questions,
    verify_corpus_manifest,
)
from src.eval.schemas import EvalQuestion


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    return tmp_path


def _write_questions(path: Path, questions: list[EvalQuestion]) -> None:
    with path.open("w") as f:
        for q in questions:
            f.write(q.model_dump_json() + "\n")


class TestLoadQuestions:
    def test_loads_existing_jsonl(self, temp_data_dir: Path):
        path = temp_data_dir / "questions.jsonl"
        sample = [
            EvalQuestion(
                id="q1", question="What is attention?",
                gold_answer="A weighted sum.", gold_chunk_ids=["c1"],
            ),
        ]
        _write_questions(path, sample)
        loaded = load_questions(path)
        assert loaded == sample

    def test_empty_file_returns_empty_list(self, temp_data_dir: Path):
        path = temp_data_dir / "questions.jsonl"
        path.write_text("")
        assert load_questions(path) == []

    def test_missing_file_raises(self, temp_data_dir: Path):
        with pytest.raises(FileNotFoundError):
            load_questions(temp_data_dir / "missing.jsonl")


class TestVerifyCorpusManifest:
    def test_valid_manifest_returns_papers(self, temp_data_dir: Path):
        pdf_path = temp_data_dir / "fake.pdf"
        pdf_path.write_bytes(b"hello world")
        sha = hashlib.sha256(b"hello world").hexdigest()

        manifest_path = temp_data_dir / "manifest.json"
        manifest_path.write_text(json.dumps({
            "version": "v1",
            "description": "test",
            "papers": [{
                "id": "fake",
                "title": "Fake Paper",
                "source_url": "https://example.com",
                "local_path": str(pdf_path),
                "sha256": sha,
            }],
        }))
        papers = verify_corpus_manifest(manifest_path)
        assert len(papers) == 1
        assert papers[0]["id"] == "fake"

    def test_tampered_sha_raises(self, temp_data_dir: Path):
        pdf_path = temp_data_dir / "fake.pdf"
        pdf_path.write_bytes(b"hello world")
        bad_sha = "0" * 64

        manifest_path = temp_data_dir / "manifest.json"
        manifest_path.write_text(json.dumps({
            "version": "v1",
            "description": "test",
            "papers": [{
                "id": "fake", "title": "Fake", "source_url": "https://x",
                "local_path": str(pdf_path), "sha256": bad_sha,
            }],
        }))
        with pytest.raises(ManifestVerificationError, match="sha256 mismatch"):
            verify_corpus_manifest(manifest_path)

    def test_missing_pdf_raises(self, temp_data_dir: Path):
        manifest_path = temp_data_dir / "manifest.json"
        manifest_path.write_text(json.dumps({
            "version": "v1",
            "description": "test",
            "papers": [{
                "id": "missing", "title": "Missing", "source_url": "https://x",
                "local_path": str(temp_data_dir / "absent.pdf"),
                "sha256": "0" * 64,
            }],
        }))
        with pytest.raises(ManifestVerificationError, match="not found"):
            verify_corpus_manifest(manifest_path)

    def test_empty_papers_list_is_ok(self, temp_data_dir: Path):
        manifest_path = temp_data_dir / "manifest.json"
        manifest_path.write_text(json.dumps({
            "version": "v1", "description": "skeleton", "papers": [],
        }))
        assert verify_corpus_manifest(manifest_path) == []


class TestDefaults:
    def test_default_paths_point_at_v1(self):
        assert "ml_papers_v1" in str(DEFAULT_QUESTIONS_PATH)
        assert "ml_papers_v1" in str(DEFAULT_MANIFEST_PATH)
