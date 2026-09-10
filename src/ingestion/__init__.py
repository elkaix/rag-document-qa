"""Ingestion — turning files into retrievable chunks.

RAG Pipeline Position:
    File -> [INGESTION] -> Document -> Chunk -> Embedding -> Vector Store

What this package holds:
    - ``parsers``: one function per format behind a registry, plus the pure
      text normalisation PDFs need.
    - ``loader``: path handling, source metadata, and batch error policy.
    - ``chunking``: the three chunking strategies and the quality filters.

Why this is a package:
    Loading and chunking shared a 504-line module — twice the project's ceiling
    — while sharing no code with each other. They change for different reasons:
    adding a format touches parsing, tuning retrieval quality touches chunking.
"""

from src.ingestion.chunking import TextChunker
from src.ingestion.loader import DocumentLoader
from src.ingestion.parsers import (
    PARSERS,
    SUPPORTED_EXTENSIONS,
    normalise_pdf_text,
    parser_for,
)

__all__ = [
    "PARSERS",
    "SUPPORTED_EXTENSIONS",
    "DocumentLoader",
    "TextChunker",
    "normalise_pdf_text",
    "parser_for",
]
