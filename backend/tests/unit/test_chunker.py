import pytest
from backend.app.parsers.models import (
    ParsedDocument,
    ParsedElement,
    ParsedElementType,
    ParsedPage,
)
from backend.app.services.chunking_service import StructureAwareChunker


@pytest.fixture
def chunker():
    """Provides a StructureAwareChunker with explicit test sizing."""
    class MockSettings:
        CHUNK_SIZE = 300
        CHUNK_OVERLAP = 50
        MIN_CHUNK_SIZE = 30

    return StructureAwareChunker(settings=MockSettings())


def test_chunker_deterministic_ordering(chunker):
    """Verify chunks receive sequential indices starting from 0."""
    elements = [
        ParsedElement(element_type=ParsedElementType.HEADING, text="Chapter 1: Foundations", level=1),
        ParsedElement(element_type=ParsedElementType.PARAGRAPH, text="Paragraph A " * 15),
        ParsedElement(element_type=ParsedElementType.PARAGRAPH, text="Paragraph B " * 15),
        ParsedElement(element_type=ParsedElementType.PARAGRAPH, text="Paragraph C " * 15),
    ]
    doc = ParsedDocument(
        document_id="doc-chunk-1",
        filename="test.txt",
        pages=[ParsedPage(page_number=1, elements=elements)],
        page_count=1,
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    for idx, chunk in enumerate(chunks):
        assert chunk.chunk_index == idx
        assert chunk.character_count == len(chunk.text)
        assert chunk.token_count > 0


def test_chunker_section_path_and_heading_preservation(chunker):
    """Verify chunks retain their hierarchical section path and section heading."""
    elements = [
        ParsedElement(element_type=ParsedElementType.HEADING, text="1. System Architecture", level=1),
        ParsedElement(element_type=ParsedElementType.HEADING, text="1.1 Storage Layer", level=2),
        ParsedElement(
            element_type=ParsedElementType.PARAGRAPH,
            text="The storage layer uses isolated directories per document ID.",
        ),
    ]
    doc = ParsedDocument(
        document_id="doc-chunk-2",
        filename="arch.md",
        pages=[ParsedPage(page_number=1, elements=elements)],
        page_count=1,
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 1
    chunk = chunks[0]
    assert chunk.section_title == "1.1 Storage Layer"
    assert chunk.section_path == "1. System Architecture > 1.1 Storage Layer"
    assert chunk.page_number == 1


def test_chunker_page_boundaries(chunker):
    """Verify chunks do not cross physical page boundaries."""
    page1_elements = [
        ParsedElement(element_type=ParsedElementType.PARAGRAPH, text="Page 1 content text for retrieval evaluation.")
    ]
    page2_elements = [
        ParsedElement(element_type=ParsedElementType.PARAGRAPH, text="Page 2 content text with independent citation target.")
    ]
    doc = ParsedDocument(
        document_id="doc-chunk-3",
        filename="multi_page.pdf",
        pages=[
            ParsedPage(page_number=1, elements=page1_elements),
            ParsedPage(page_number=2, elements=page2_elements),
        ],
        page_count=2,
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert "Page 1" in chunks[0].text
    assert chunks[1].page_number == 2
    assert "Page 2" in chunks[1].text


def test_chunker_empty_document(chunker):
    """Verify empty document produces 0 chunks without error."""
    doc = ParsedDocument(
        document_id="doc-empty",
        filename="empty.txt",
        pages=[],
        page_count=0,
    )
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 0


def test_chunker_large_paragraph_split(chunker):
    """Verify an oversized single paragraph is split into multiple sub-chunks."""
    large_text = "This is a sentence in a very large document block. " * 30
    elements = [ParsedElement(element_type=ParsedElementType.PARAGRAPH, text=large_text)]
    doc = ParsedDocument(
        document_id="doc-large",
        filename="large.txt",
        pages=[ParsedPage(page_number=1, elements=elements)],
        page_count=1,
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 3
    for chunk in chunks:
        assert chunk.character_count <= chunker.max_chunk_size + 50
