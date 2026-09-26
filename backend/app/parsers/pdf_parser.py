from pathlib import Path
from typing import List
from pypdf import PdfReader
from backend.app.parsers.base import BaseParser
from backend.app.parsers.models import (
    ParsedDocument,
    ParsedElement,
    ParsedElementType,
    ParsedPage,
)
from backend.app.utils.logger import logger


class PDFParser(BaseParser):
    """
    Parser for PDF documents using pypdf.
    Preserves exact 1-indexed page numbers, separates paragraphs,
    and detects scanned/image-only documents.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ["pdf"]

    @property
    def supported_mime_types(self) -> List[str]:
        return ["application/pdf", "application/x-pdf"]

    def parse(
        self, file_path: Path, document_id: str, original_filename: str
    ) -> ParsedDocument:
        logger.info(f"Parsing PDF document: {file_path.name}")
        try:
            reader = PdfReader(str(file_path))
        except Exception as exc:
            logger.error(f"Failed to open PDF {file_path}: {exc}")
            raise ValueError(f"Malformed or corrupt PDF file: {exc}")

        pages: List[ParsedPage] = []
        total_text_length = 0

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception as extract_exc:
                logger.warning(f"Error extracting text from page {page_idx}: {extract_exc}")
                raw_text = ""

            cleaned_text = raw_text.strip()
            total_text_length += len(cleaned_text)

            elements: List[ParsedElement] = []
            if cleaned_text:
                # Split page into logical paragraphs by double newlines or long single breaks
                raw_paragraphs = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]
                for para in raw_paragraphs:
                    # Heuristic: short single-line capitalized string could be a section heading
                    is_potential_heading = (
                        len(para) < 100
                        and "\n" not in para
                        and (para.isupper() or para.istitle() or para.startswith(tuple(f"{i}." for i in range(1, 20))))
                    )
                    elements.append(
                        ParsedElement(
                            element_type=ParsedElementType.HEADING if is_potential_heading else ParsedElementType.PARAGRAPH,
                            text=para,
                            level=2 if is_potential_heading else None,
                            metadata={"page_number": page_idx},
                        )
                    )

            pages.append(ParsedPage(page_number=page_idx, elements=elements))

        # Detect scanned/empty documents
        is_scanned = total_text_length < 20 and len(reader.pages) > 0

        return ParsedDocument(
            document_id=document_id,
            filename=original_filename,
            pages=pages,
            page_count=len(pages),
            metadata={
                "parser": "PDFParser",
                "total_pages": len(pages),
                "is_scanned": is_scanned,
                "character_count": total_text_length,
            },
        )
