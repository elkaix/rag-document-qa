"""Per-format parsers behind one seam.

RAG Pipeline Position:
    File -> [PARSERS] -> text + metadata -> Chunk -> Embedding -> Vector Store
               ^^^

What concept it teaches:
    A registry of adapters instead of a dispatch table of private methods. Each
    parser is a module-level function of ``Path -> (text, metadata)``, so a test
    can call one directly and a new format is one function plus one registry
    entry.

Why this approach over alternatives:
    Format dispatch used to be a dict of methods bound to ``self``, rebuilt on
    every call and gated by a second hardcoded extension set that had to be kept
    in sync by hand. Nothing could be substituted, and the three formats needing
    an optional dependency — PDF, DOCX, HTML — had **no tests at all**, because
    reaching them meant writing a real binary file to disk.

    The registry now derives the supported-extension set, so the two cannot
    disagree.

Design Decision:
    The valuable part of PDF handling — undoing the hard line breaks pypdf emits
    at the column width — is a pure ``str -> str`` transform. It lives in
    ``normalise_pdf_text`` where it can be tested with a string, rather than
    trapped behind file I/O and an optional import.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ParseResult = tuple[str, dict[str, Any]]
Parser = Callable[[Path], ParseResult]

# Sentinel used while paragraph breaks are protected from line-break collapsing.
_PARAGRAPH_MARK = "\x00"


def normalise_pdf_text(text: str) -> str:
    """Undo PDF layout line breaks while preserving paragraph structure.

    Args:
        text: Raw text as extracted from a PDF, with hard newlines at the
            column width.

    Returns:
        Text where single newlines have become spaces, blank-line paragraph
        breaks survive, hyphenated line-wraps are rejoined, and runs of spaces
        are collapsed.

    WHY this matters: pypdf emits a newline wherever the *layout* wrapped,
        not where a sentence ended. Left alone, the recursive chunker treats
        each of those as a boundary and over-fragments the text, so retrieval
        returns half-sentences.

    Example:
        "Fine-Tuning LLMs from\\nBasics" becomes "Fine-Tuning LLMs from Basics".
    """
    text = text.replace("\n\n", _PARAGRAPH_MARK)
    text = text.replace("\n", " ")
    text = text.replace(_PARAGRAPH_MARK, "\n\n")

    # WHY hyphen-space-lowercase: after the newline became a space, a wrapped
    #     word reads "develop- ment". A real compound ("self-attention") has no
    #     space after the hyphen, so this pattern separates the two cases.
    text = re.sub(r"(\w)- ([a-z])", r"\1\2", text)

    return re.sub(r" {2,}", " ", text)


def parse_text(path: Path) -> ParseResult:
    """Read a plain-text or Markdown file.

    Args:
        path: File to read.

    Returns:
        The file's text and its encoding.
    """
    return path.read_text(encoding="utf-8", errors="replace"), {"encoding": "utf-8"}


def parse_pdf(path: Path) -> ParseResult:
    """Extract text from a PDF, normalising its layout line breaks.

    Args:
        path: File to read.

    Returns:
        The document text, plus page count and any title/author/subject the
        PDF declares.

    Note:
        Falls back to a raw read when pypdf is absent, so a missing optional
        dependency degrades rather than raising.
    """
    try:
        import pypdf  # type: ignore
    except ImportError:
        logger.warning("pypdf not installed; reading PDF as binary text")
        return path.read_text(errors="replace"), {}

    reader = pypdf.PdfReader(str(path))
    text = normalise_pdf_text("\n\n".join(page.extract_text() or "" for page in reader.pages))

    meta: dict[str, Any] = {"page_count": len(reader.pages)}
    if reader.metadata:
        for key in ("title", "author", "subject"):
            value = getattr(reader.metadata, key, None)
            if value:
                meta[key] = value
    return text, meta


def parse_docx(path: Path) -> ParseResult:
    """Extract paragraph text from a Word document.

    Args:
        path: File to read.

    Returns:
        Non-empty paragraphs joined by blank lines, plus core properties.

    Note:
        Returns empty text when python-docx is absent rather than raising.
    """
    try:
        import docx  # type: ignore
    except ImportError:
        logger.warning("python-docx not installed; cannot load DOCX")
        return "", {"error": "python-docx not installed"}

    document = docx.Document(str(path))
    text = "\n\n".join(p.text for p in document.paragraphs if p.text.strip())

    meta: dict[str, Any] = {}
    props = document.core_properties
    for attr in ("author", "title", "subject", "created", "modified"):
        value = getattr(props, attr, None)
        if value:
            meta[attr] = str(value)
    return text, meta


def parse_html(path: Path) -> ParseResult:
    """Extract readable text from an HTML file.

    Args:
        path: File to read.

    Returns:
        Visible text with script, style, nav, header and footer removed, plus
        the document title.

    Note:
        Falls back to a naive tag strip when beautifulsoup4 is absent.
    """
    html = path.read_text(encoding="utf-8", errors="replace")
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:
        logger.warning("beautifulsoup4 not installed; stripping HTML tags naively")
        stripped = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", stripped).strip(), {}

    soup = BeautifulSoup(html, "html.parser")
    # WHY these tags: they carry chrome, not content — indexing them pollutes
    #     retrieval with menus and cookie banners.
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.string if soup.title else ""
    return soup.get_text(separator="\n", strip=True), {"html_title": title or ""}


def parse_csv(path: Path) -> ParseResult:
    """Render a CSV as one labelled line per row.

    Args:
        path: File to read.

    Returns:
        A header line followed by ``column: value`` pairs per row, plus row and
        column counts.

    WHY labelled pairs rather than raw rows: a retrieved chunk has to make sense
        on its own, and a bare row of values does not carry its column names.
    """
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return "", {"row_count": 0, "column_count": 0}

    headers = rows[0]
    lines = [", ".join(headers)]
    for row in rows[1:]:
        lines.append("; ".join(f"{h}: {v}" for h, v in zip(headers, row, strict=False)))

    return "\n".join(lines), {
        "row_count": len(rows) - 1,
        "column_count": len(headers),
    }


def parse_json(path: Path) -> ParseResult:
    """Render a JSON file as indented text.

    Args:
        path: File to read.

    Returns:
        Pretty-printed JSON when it parses, the raw text otherwise, with a
        ``json_valid`` flag either way.

    WHY invalid JSON is not an error: the text is still indexable, and refusing
        the upload would be worse than indexing it as-is.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw, {"json_valid": False}
    return json.dumps(data, indent=2, ensure_ascii=False), {"json_valid": True}


# PATTERN: one registry, and the supported-extension set derived from it. The
#          two used to be separate literals that had to be kept in step by hand.
PARSERS: dict[str, Parser] = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".txt": parse_text,
    ".md": parse_text,
    ".html": parse_html,
    ".htm": parse_html,
    ".csv": parse_csv,
    ".json": parse_json,
}

SUPPORTED_EXTENSIONS = frozenset(PARSERS)


def parser_for(extension: str) -> Parser:
    """Return the parser registered for a file extension.

    Args:
        extension: Extension including the dot, any case.

    Returns:
        The parser function.

    Raises:
        ValueError: If no parser is registered for the extension.
    """
    try:
        return PARSERS[extension.lower()]
    except KeyError:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        ) from None
