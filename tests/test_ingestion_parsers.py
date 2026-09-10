"""Tests for src.ingestion.parsers — one parser per format, behind a registry.

PDF, DOCX and HTML had zero tests before this seam existed: dispatch went to
private methods bound to the loader, so reaching them meant writing a real
binary file to disk and going through the whole loader.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingestion.parsers import (
    PARSERS,
    SUPPORTED_EXTENSIONS,
    normalise_pdf_text,
    parse_csv,
    parse_docx,
    parse_html,
    parse_json,
    parse_pdf,
    parse_text,
    parser_for,
)


class TestNormalisePdfText:
    """The transform that stops the chunker over-fragmenting PDF text.

    This is the module's most valuable logic and was previously reachable only
    by writing a real PDF to disk. As a free function it is a five-line test.
    """

    def test_layout_line_breaks_become_spaces(self):
        assert normalise_pdf_text("Fine-Tuning LLMs from\nBasics") == (
            "Fine-Tuning LLMs from Basics"
        )

    def test_paragraph_breaks_survive(self):
        assert normalise_pdf_text("First para.\n\nSecond para.") == ("First para.\n\nSecond para.")

    def test_a_wrapped_hyphenated_word_is_rejoined(self):
        assert normalise_pdf_text("develop-\nment") == "development"

    def test_a_real_compound_keeps_its_hyphen(self):
        """ "self-attention" has no space after the hyphen, so it is not a wrap."""
        assert normalise_pdf_text("self-attention works") == "self-attention works"

    def test_a_hyphen_before_a_capital_is_not_a_wrap(self):
        assert normalise_pdf_text("Fine-\nTuning") == "Fine- Tuning"

    def test_runs_of_spaces_collapse(self):
        assert normalise_pdf_text("a     b") == "a b"

    def test_empty_text_is_unchanged(self):
        assert normalise_pdf_text("") == ""


class TestRegistry:
    def test_supported_extensions_derive_from_the_registry(self):
        """The two used to be separate literals kept in step by hand."""
        assert SUPPORTED_EXTENSIONS == frozenset(PARSERS)

    def test_lookup_is_case_insensitive(self):
        assert parser_for(".PDF") is parse_pdf

    def test_an_unknown_extension_is_rejected_by_name(self):
        with pytest.raises(ValueError, match=r"Unsupported file type: \.xyz"):
            parser_for(".xyz")

    def test_markdown_and_text_share_a_parser(self):
        assert parser_for(".md") is parser_for(".txt") is parse_text


class TestTextParser:
    def test_reads_content_and_reports_encoding(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hello")
        assert parse_text(f) == ("hello", {"encoding": "utf-8"})

    def test_undecodable_bytes_are_replaced_not_raised(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_bytes(b"caf\xff")
        text, _ = parse_text(f)
        assert text.startswith("caf")


class TestCsvParser:
    def test_rows_become_labelled_pairs(self, tmp_path):
        f = tmp_path / "a.csv"
        f.write_text("name,role\nAda,engineer\n")
        text, meta = parse_csv(f)
        assert "name: Ada; role: engineer" in text
        assert meta == {"row_count": 1, "column_count": 2}

    def test_an_empty_file_reports_zero_counts(self, tmp_path):
        f = tmp_path / "a.csv"
        f.write_text("")
        assert parse_csv(f) == ("", {"row_count": 0, "column_count": 0})

    def test_a_short_row_stops_at_the_values_it_has(self, tmp_path):
        f = tmp_path / "a.csv"
        f.write_text("a,b,c\n1,2\n")
        text, _ = parse_csv(f)
        assert "a: 1; b: 2" in text


class TestJsonParser:
    def test_valid_json_is_pretty_printed(self, tmp_path):
        f = tmp_path / "a.json"
        f.write_text('{"b":1,"a":2}')
        text, meta = parse_json(f)
        assert meta == {"json_valid": True}
        assert json.loads(text) == {"b": 1, "a": 2}
        assert "\n" in text

    def test_invalid_json_is_indexed_as_raw_text(self, tmp_path):
        f = tmp_path / "a.json"
        f.write_text("{not json")
        assert parse_json(f) == ("{not json", {"json_valid": False})

    def test_non_ascii_is_preserved(self, tmp_path):
        f = tmp_path / "a.json"
        f.write_text('{"t":"ünïcödé"}', encoding="utf-8")
        text, _ = parse_json(f)
        assert "ünïcödé" in text


class TestHtmlParser:
    def test_visible_text_is_extracted(self, tmp_path):
        f = tmp_path / "a.html"
        f.write_text("<html><body><p>Hello world</p></body></html>")
        text, _ = parse_html(f)
        assert "Hello world" in text

    def test_chrome_tags_are_dropped(self, tmp_path):
        """Menus and cookie banners pollute retrieval."""
        f = tmp_path / "a.html"
        f.write_text(
            "<html><head><style>.x{}</style></head><body>"
            "<nav>Menu</nav><header>Top</header>"
            "<p>Real content</p>"
            "<footer>Legal</footer><script>var x=1</script>"
            "</body></html>"
        )
        text, _ = parse_html(f)
        assert "Real content" in text
        for chrome in ("Menu", "Top", "Legal", "var x=1", ".x{}"):
            assert chrome not in text

    def test_the_title_is_captured(self, tmp_path):
        f = tmp_path / "a.html"
        f.write_text("<html><head><title>My Page</title></head><body>x</body></html>")
        _, meta = parse_html(f)
        assert meta["html_title"] == "My Page"

    def test_a_missing_title_is_empty_not_none(self, tmp_path):
        f = tmp_path / "a.html"
        f.write_text("<html><body>x</body></html>")
        _, meta = parse_html(f)
        assert meta["html_title"] == ""


class TestDocxParser:
    def _write_docx(self, path: Path, paragraphs: list[str], **props) -> Path:
        import docx

        document = docx.Document()
        for text in paragraphs:
            document.add_paragraph(text)
        for key, value in props.items():
            setattr(document.core_properties, key, value)
        document.save(str(path))
        return path

    def test_paragraphs_are_joined_with_blank_lines(self, tmp_path):
        f = self._write_docx(tmp_path / "a.docx", ["First", "Second"])
        text, _ = parse_docx(f)
        assert text == "First\n\nSecond"

    def test_empty_paragraphs_are_dropped(self, tmp_path):
        f = self._write_docx(tmp_path / "a.docx", ["First", "   ", "Second"])
        text, _ = parse_docx(f)
        assert text == "First\n\nSecond"

    def test_core_properties_become_metadata(self, tmp_path):
        f = self._write_docx(tmp_path / "a.docx", ["x"], author="Ada", title="Notes")
        _, meta = parse_docx(f)
        assert meta["author"] == "Ada"
        assert meta["title"] == "Notes"


class TestPdfParser:
    def _write_pdf(self, path: Path, lines: list[str]) -> Path:
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        with path.open("wb") as handle:
            writer.write(handle)
        return path

    def test_reports_the_page_count(self, tmp_path):
        f = self._write_pdf(tmp_path / "a.pdf", [])
        _, meta = parse_pdf(f)
        assert meta["page_count"] == 1

    def test_returns_text_and_metadata(self, tmp_path):
        f = self._write_pdf(tmp_path / "a.pdf", [])
        text, meta = parse_pdf(f)
        assert isinstance(text, str)
        assert "page_count" in meta

    def test_a_missing_pypdf_degrades_instead_of_raising(self, tmp_path, monkeypatch):
        """The optional-dependency fallback, previously unreachable in tests."""
        import builtins

        real_import = builtins.__import__

        def no_pypdf(name, *args, **kwargs):
            if name == "pypdf":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        f = tmp_path / "a.pdf"
        f.write_text("plain text standing in for a pdf")
        monkeypatch.setattr(builtins, "__import__", no_pypdf)

        text, meta = parse_pdf(f)

        assert "plain text" in text
        assert meta == {}
