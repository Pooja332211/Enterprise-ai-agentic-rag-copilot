import re
from pathlib import Path
from typing import List
from backend.app.parsers.base import BaseParser
from backend.app.parsers.models import (
    ParsedDocument,
    ParsedElement,
    ParsedElementType,
    ParsedPage,
)
from backend.app.utils.logger import logger


class TextParser(BaseParser):
    """
    Parser for plain text (.txt) documents.
    Normalizes line breaks, segments content into paragraphs,
    and identifies section headers.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ["txt"]

    @property
    def supported_mime_types(self) -> List[str]:
        return ["text/plain"]

    def parse(
        self, file_path: Path, document_id: str, original_filename: str
    ) -> ParsedDocument:
        logger.info(f"Parsing plain text document: {file_path.name}")
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.error(f"Failed to read text file {file_path}: {exc}")
            raise ValueError(f"Could not read text file: {exc}")

        # Normalize line endings
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        raw_blocks = [b.strip() for b in normalized.split("\n\n") if b.strip()]

        elements: List[ParsedElement] = []

        for block in raw_blocks:
            # Check if block looks like a section heading: e.g. "SECTION 1", "1. Introduction", all-caps title
            is_heading = (
                len(block) < 80
                and "\n" not in block
                and (
                    block.isupper()
                    or bool(re.match(r"^\d+(\.\d+)*\.?\s+[A-Z]", block))
                    or block.startswith("Chapter ")
                )
            )

            elements.append(
                ParsedElement(
                    element_type=ParsedElementType.HEADING if is_heading else ParsedElementType.PARAGRAPH,
                    text=block,
                    level=1 if is_heading else None,
                )
            )

        pages = [ParsedPage(page_number=1, elements=elements)]

        return ParsedDocument(
            document_id=document_id,
            filename=original_filename,
            pages=pages,
            page_count=1,
            metadata={
                "parser": "TextParser",
                "paragraph_count": len(elements),
            },
        )
