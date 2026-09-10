"""Document loading — find a file, pick its parser, attach source metadata.

RAG Pipeline Position:
    File -> [LOADER] -> Document -> Chunk -> Embedding -> Vector Store
              ^^^

What concept it teaches:
    A thin orchestrator over a registry. The loader knows about paths, source
    metadata and error handling; it knows nothing about any file format. Adding
    a format means adding a parser, not editing this module.

Why this approach over alternatives:
    Dispatch used to be a dict of private methods bound to the loader, rebuilt
    on every call, gated by a separate hardcoded extension set. A new format
    meant editing three places inside one class, and no parser could be
    substituted or called on its own in a test.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

from src.domain import Document
from src.ingestion.parsers import SUPPORTED_EXTENSIONS, parser_for

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Turns files into Documents by delegating to a registered parser."""

    def load(self, file_path: str | Path) -> Document:
        """Load one file.

        Args:
            file_path: Path to the file.

        Returns:
            A Document carrying the extracted text and its source metadata.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If no parser is registered for the extension.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        parse = parser_for(path.suffix)

        metadata: dict[str, Any] = {
            "filename": path.name,
            "file_path": str(path.resolve()),
            "file_type": path.suffix.lower().lstrip("."),
            "file_size_bytes": path.stat().st_size,
        }

        logger.info("Loading document: %s", path)
        content, parser_metadata = parse(path)
        metadata.update(parser_metadata)

        document = Document(content=content, metadata=metadata)
        logger.debug("Loaded document %s (%d chars)", path.name, len(content))
        return document

    def load_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
    ) -> List[Document]:
        """Load every supported file in a directory.

        Args:
            directory: Directory to scan.
            recursive: Whether to descend into subdirectories.
            extensions: Restrict to these extensions; defaults to all supported.

        Returns:
            The documents that loaded successfully.

        Raises:
            NotADirectoryError: If the path is not a directory.

        WHY one bad file does not abort the batch: a directory upload is a bulk
            operation, and failing all of it because one PDF is corrupt is worse
            than indexing the rest and logging the casualty.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        allowed = {e.lower() for e in (extensions or SUPPORTED_EXTENSIONS)}
        pattern = "**/*" if recursive else "*"
        files = [
            p
            for p in dir_path.glob(pattern)
            if p.is_file() and p.suffix.lower() in allowed
        ]
        logger.info("Found %d files in %s", len(files), dir_path)

        documents: List[Document] = []
        for file in files:
            try:
                documents.append(self.load(file))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", file, exc)

        logger.info("Successfully loaded %d/%d documents", len(documents), len(files))
        return documents
