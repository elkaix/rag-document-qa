"""
Utility functions for the RAG pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .document_loader import SUPPORTED_EXTENSIONS  # single source of truth


def format_source_citation(source: Dict[str, Any] | str) -> str:
    """Format a source reference into a human-readable citation string.

    Args:
        source: Either a dict with metadata keys (filename, page, chunk_index,
                doc_id, score) or a plain string.

    Returns:
        Formatted citation string.

    Examples:
        >>> format_source_citation({"filename": "report.pdf", "page": 3, "score": 0.91})
        '[report.pdf, page 3, score: 0.91]'
        >>> format_source_citation("report.pdf")
        '[report.pdf]'
    """
    if isinstance(source, str):
        return f"[{source}]"

    parts: list[str] = []

    filename = source.get("filename") or source.get("file_path") or source.get("doc_id", "unknown")
    if filename:
        # Show only the base name to keep citations short
        parts.append(str(Path(filename).name))

    page = source.get("page")
    if page is not None:
        parts.append(f"page {page}")

    chunk_index = source.get("chunk_index")
    if chunk_index is not None:
        parts.append(f"chunk {chunk_index}")

    score = source.get("score")
    if score is not None:
        parts.append(f"score: {score:.2f}")

    return "[" + ", ".join(parts) + "]"


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to max_length characters, appending '…' if truncated.

    Args:
        text: Input string.
        max_length: Maximum allowed length (including the ellipsis).

    Returns:
        Possibly-truncated string.

    Examples:
        >>> truncate_text("Hello world", max_length=5)
        'Hell…'
        >>> truncate_text("Short", max_length=200)
        'Short'
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    if max_length < 1:
        raise ValueError("max_length must be at least 1")

    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "…"


def calculate_hash(text: str) -> str:
    """Return a SHA-256 hex digest of the given text.

    Uses UTF-8 encoding. The full 64-character hex string is returned;
    callers may slice it for shorter IDs.

    Args:
        text: Input string.

    Returns:
        64-character lowercase hex string.

    Examples:
        >>> len(calculate_hash("hello"))
        64
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def setup_logging(level: str = "INFO", name: str = "rag") -> logging.Logger:
    """Configure and return a logger with a consistent format.

    Configures the root logger if not already configured.

    Args:
        level: Logging level name ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        name: Logger name.

    Returns:
        Configured Logger instance.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Avoid adding duplicate handlers on repeated calls
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(numeric_level)
    return logger


def validate_file_type(filename: str) -> bool:
    """Check whether a filename has a supported document extension.

    Args:
        filename: Filename or path string.

    Returns:
        True if the file extension is supported, False otherwise.

    Examples:
        >>> validate_file_type("report.pdf")
        True
        >>> validate_file_type("archive.zip")
        False
    """
    ext = Path(filename).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS
