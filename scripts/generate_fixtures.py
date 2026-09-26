import io
from pathlib import Path
import docx
from pypdf import PdfReader, PdfWriter


def create_fixtures():
    fixtures_dir = Path("backend/tests/fixtures")
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plain Text Fixture
    sample_txt = fixtures_dir / "sample.txt"
    sample_txt.write_text(
        "1. Executive Summary\n\n"
        "This is an enterprise knowledge copilot designed for production agentic retrieval and grounded reasoning.\n\n"
        "2. Architecture\n\n"
        "The architecture combines dense vector retrieval, lexical BM25 matching, and structure-aware chunking.\n"
        "It provides high precision across complex multi-document queries.\n\n"
        "3. Conclusion\n\n"
        "All claims are verified with source page and section citations.",
        encoding="utf-8",
    )

    # 2. Markdown Fixture
    sample_md = fixtures_dir / "sample.md"
    sample_md.write_text(
        "# Enterprise Copilot Specification\n\n"
        "## Overview\n\n"
        "The system provides grounded answers with source citations.\n\n"
        "## Core Capabilities\n\n"
        "- Hybrid dense and sparse search\n"
        "- Reciprocal rank fusion\n"
        "- Citation verification agent\n\n"
        "```python\n"
        "def query_copilot(prompt: str) -> str:\n"
        "    return 'Answer with citations'\n"
        "```\n\n"
        "### Limitations\n\n"
        "Scanned documents require future OCR capabilities.",
        encoding="utf-8",
    )

    # 3. HTML Fixture
    sample_html = fixtures_dir / "sample.html"
    sample_html.write_text(
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head><title>System Specs</title></head>\n"
        "<body>\n"
        "<h1>Enterprise Knowledge Platform</h1>\n"
        "<p>This platform indexes enterprise documents with structure preservation.</p>\n"
        "<h2>Supported Formats</h2>\n"
        "<ul>\n"
        "  <li>PDF documents with page citations</li>\n"
        "  <li>DOCX reports with heading styles</li>\n"
        "  <li>Markdown and plain text specifications</li>\n"
        "</ul>\n"
        "<h2>Benchmark Results</h2>\n"
        "<table>\n"
        "  <tr><th>Model</th><th>Accuracy</th></tr>\n"
        "  <tr><td>Hybrid RAG</td><td>94.2%</td></tr>\n"
        "  <tr><td>Dense Only</td><td>81.5%</td></tr>\n"
        "</table>\n"
        "</body>\n"
        "</html>",
        encoding="utf-8",
    )

    # 4. DOCX Fixture
    sample_docx = fixtures_dir / "sample.docx"
    doc = docx.Document()
    doc.add_heading("Research Findings Report", level=1)
    doc.add_paragraph("This document outlines the findings of our multi-document evaluation.")
    doc.add_heading("Methodology", level=2)
    doc.add_paragraph("We evaluated 50 benchmark queries across dense and lexical retrieval modes.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "Score"
    table.cell(1, 0).text = "NDCG@10"
    table.cell(1, 1).text = "0.89"
    doc.save(str(sample_docx))

    # 5. Multi-page PDF Fixture with extracted text
    sample_pdf = fixtures_dir / "sample.pdf"
    raw_pdf_stream = (
        b"%PDF-1.4\n"
        b"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
        b"2 0 obj <</Type /Pages /Kids [3 0 R 4 0 R] /Count 2>> endobj\n"
        b"3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R>> endobj\n"
        b"4 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R>> endobj\n"
        b"5 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj\n"
        b"6 0 obj <</Length 73>> stream\nBT /F1 14 Tf 50 700 Td (1. Executive Summary) Tj 0 -30 Td (This is page 1 content.) Tj ET\nendstream\nendobj\n"
        b"7 0 obj <</Length 68>> stream\nBT /F1 14 Tf 50 700 Td (2. Methodology) Tj 0 -30 Td (This is page 2 content.) Tj ET\nendstream\nendobj\n"
        b"xref\n0 8\n0000000000 65535 f \n0000000009 00000 n \n0000000056 00000 n \n0000000122 00000 n \n0000000244 00000 n \n0000000366 00000 n \n0000000435 00000 n \n0000000557 00000 n \ntrailer <</Size 8 /Root 1 0 R>>\nstartxref\n674\n%%EOF\n"
    )
    reader = PdfReader(io.BytesIO(raw_pdf_stream))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    with open(sample_pdf, "wb") as f:
        writer.write(f)

    print("Fixtures created successfully in backend/tests/fixtures/")


if __name__ == "__main__":
    create_fixtures()
