"""
ML Papers v1 dev-set loader.

Eval Harness Position:
  hand-labeled questions.jsonl + corpus_manifest.json
                ↓
       load_questions / verify_corpus_manifest
                ↓
       runner reads questions, ingests pinned PDFs into Chroma

Design decisions:
  - Corpus manifest pins each PDF by SHA-256 so the eval corpus is
    BYTE-STABLE across machines. If a PDF on disk is replaced or
    corrupted, the loader raises rather than silently producing
    different chunks downstream.
  - Empty papers list and empty questions.jsonl are both valid: the
    skeleton ships with both empty so the test suite passes before any
    labeling has happened. Labeling fills them in incrementally.
  - load_questions returns a plain list[EvalQuestion]; the runner is
    responsible for the ingest step.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from src.eval.schemas import EvalQuestion

logger = logging.getLogger(__name__)

DEFAULT_QUESTIONS_PATH = Path("eval_data/ml_papers_v1/questions.jsonl")
DEFAULT_MANIFEST_PATH = Path("eval_data/ml_papers_v1/corpus_manifest.json")


class ManifestVerificationError(Exception):
    """Raised when a corpus PDF is missing or has a SHA mismatch."""


def load_questions(path: Path = DEFAULT_QUESTIONS_PATH) -> list[EvalQuestion]:
    """Read the hand-labeled questions JSONL.

    An empty file returns an empty list (the skeleton state before any
    labeling). A missing file raises.
    """
    if not path.exists():
        raise FileNotFoundError(f"ML Papers questions file not found: {path}")
    out: list[EvalQuestion] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(EvalQuestion.model_validate_json(line))
    return out


def _sha256_of(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file in 64KB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_corpus_manifest(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> list[dict[str, Any]]:
    """Verify every paper listed in the manifest matches its pinned SHA-256.

    Args:
        manifest_path: Path to corpus_manifest.json.

    Returns:
        The ``papers`` list from the manifest, after successful verification.
        May be empty if no papers have been added yet.

    Raises:
        ManifestVerificationError: If any pinned PDF is missing on disk
            or its SHA-256 does not match the recorded hash.
    """
    with manifest_path.open() as f:
        manifest = json.load(f)
    papers = manifest.get("papers", [])
    for paper in papers:
        local_path = Path(paper["local_path"])
        if not local_path.exists():
            raise ManifestVerificationError(
                f"Paper {paper['id']!r} not found at {local_path}"
            )
        actual_sha = _sha256_of(local_path)
        expected_sha = paper["sha256"]
        if actual_sha != expected_sha:
            raise ManifestVerificationError(
                f"Paper {paper['id']!r} sha256 mismatch: "
                f"expected {expected_sha}, got {actual_sha}"
            )
    logger.info("Verified corpus manifest: %d paper(s)", len(papers))
    return papers
