from pathlib import Path
import pytest
from backend.app.parsers.docx_parser import DOCXParser
from backend.app.parsers.html_parser import HTMLParser
from backend.app.parsers.markdown_parser import MarkdownParser
from backend.app.parsers.models import ParsedElementType
from backend.app.parsers.pdf_parser import PDFParser
from backend.app.parsers.registry import ParserRegistry, get_parser_registry
from backend.app.parsers.text_parser import TextParser

FIXTURES_DIR = Path("backend/tests/fixtures")


def test_text_parser():
    """Verify plain text parser extracts sections and paragraphs."""
    parser = TextParser()
    parsed = parser.parse(FIXTURES_DIR / "sample.txt", "doc-1", "sample.txt")

    assert parsed.document_id == "doc-1"
    assert parsed.page_count == 1
    assert len(parsed.pages[0].elements) >= 3

    # Check first element is detected heading
    first_elem = parsed.pages[0].elements[0]
    assert first_elem.element_type == ParsedElementType.HEADING
    assert "1. Executive Summary" in first_elem.text


def test_markdown_parser():
    """Verify markdown parser preserves heading hierarchy and code blocks."""
    parser = MarkdownParser()
    parsed = parser.parse(FIXTURES_DIR / "sample.md", "doc-2", "sample.md")

    assert parsed.document_id == "doc-2"
    assert parsed.page_count == 1

    headings = [e for e in parsed.pages[0].elements if e.element_type == ParsedElementType.HEADING]
    assert len(headings) >= 3

    h1 = headings[0]
    assert h1.level == 1
    assert "Enterprise Copilot Specification" in h1.text

    h2 = headings[1]
    assert h2.level == 2
    assert "Overview" in h2.text

    code_blocks = [e for e in parsed.pages[0].elements if e.element_type == ParsedElementType.CODE]
    assert len(code_blocks) >= 1
    assert "query_copilot" in code_blocks[0].text


def test_html_parser():
    """Verify HTML parser extracts headings, paragraphs, and tables."""
    parser = HTMLParser()
    parsed = parser.parse(FIXTURES_DIR / "sample.html", "doc-3", "sample.html")

    assert parsed.document_id == "doc-3"
    assert parsed.page_count == 1

    headings = [e for e in parsed.pages[0].elements if e.element_type == ParsedElementType.HEADING]
    assert len(headings) >= 2
    assert headings[0].level == 1
    assert "Enterprise Knowledge Platform" in headings[0].text

    tables = [e for e in parsed.pages[0].elements if e.element_type == ParsedElementType.TABLE]
    assert len(tables) >= 1
    assert "Hybrid RAG" in tables[0].text


def test_docx_parser():
    """Verify DOCX parser extracts paragraphs, headings, and tables."""
    parser = DOCXParser()
    parsed = parser.parse(FIXTURES_DIR / "sample.docx", "doc-4", "sample.docx")

    assert parsed.document_id == "doc-4"
    assert parsed.page_count == 1

    headings = [e for e in parsed.pages[0].elements if e.element_type == ParsedElementType.HEADING]
    assert len(headings) >= 2
    assert headings[0].level == 1
    assert "Research Findings Report" in headings[0].text

    tables = [e for e in parsed.pages[0].elements if e.element_type == ParsedElementType.TABLE]
    assert len(tables) >= 1
    assert "NDCG@10" in tables[0].text


def test_pdf_parser():
    """Verify PDF parser extracts pages and maintains 1-indexed page numbers."""
    parser = PDFParser()
    parsed = parser.parse(FIXTURES_DIR / "sample.pdf", "doc-5", "sample.pdf")

    assert parsed.document_id == "doc-5"
    assert parsed.page_count == 2
    assert len(parsed.pages) == 2

    # Check page numbers
    assert parsed.pages[0].page_number == 1
    assert parsed.pages[1].page_number == 2

    # Check content in pages
    assert "Executive Summary" in parsed.pages[0].full_text
    assert "Methodology" in parsed.pages[1].full_text


def test_parser_registry_resolution():
    """Verify parser registry maps extensions and MIME types correctly."""
    registry = get_parser_registry()

    assert isinstance(registry.get_parser("pdf"), PDFParser)
    assert isinstance(registry.get_parser("docx"), DOCXParser)
    assert isinstance(registry.get_parser("txt"), TextParser)
    assert isinstance(registry.get_parser("md"), MarkdownParser)
    assert isinstance(registry.get_parser("html"), HTMLParser)

    # Test MIME type fallback
    assert isinstance(registry.get_parser("unknown", "application/pdf"), PDFParser)

    # Test unsupported extension raises ValueError
    with pytest.raises(ValueError, match="Unsupported file format"):
        registry.get_parser("exe")
