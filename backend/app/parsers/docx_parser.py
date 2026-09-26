import re
from pathlib import Path
from typing import List
import docx
from backend.app.parsers.base import BaseParser
from backend.app.parsers.models import (
    ParsedDocument,
    ParsedElement,
    ParsedElementType,
    ParsedPage,
)
from backend.app.utils.logger import logger


class DOCXParser(BaseParser):
    """
    Parser for Microsoft Word (.docx) documents.
    Preserves heading styles (Heading 1-6), list items, paragraphs, and tables.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ["docx"]

    @property
    def supported_mime_types(self) -> List[str]:
        return [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ]

    def parse(
        self, file_path: Path, document_id: str, original_filename: str
    ) -> ParsedDocument:
        logger.info(f"Parsing DOCX document: {file_path.name}")
        try:
            doc = docx.Document(str(file_path))
        except Exception as exc:
            logger.error(f"Failed to open DOCX {file_path}: {exc}")
            raise ValueError(f"Malformed or corrupt DOCX file: {exc}")

        elements: List[ParsedElement] = []

        # 1. Process document paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            style_name = para.style.name if para.style else ""

            # Detect heading level from style
            heading_match = re.search(r"Heading\s*(\d)", style_name, re.IGNORECASE)
            if heading_match:
                level = int(heading_match.group(1))
                elements.append(
                    ParsedElement(
                        element_type=ParsedElementType.HEADING,
                        text=text,
                        level=level,
                        metadata={"style": style_name},
                    )
                )
            elif "List" in style_name:
                elements.append(
                    ParsedElement(
                        element_type=ParsedElementType.LIST_ITEM,
                        text=text,
                        metadata={"style": style_name},
                    )
                )
            else:
                elements.append(
                    ParsedElement(
                        element_type=ParsedElementType.PARAGRAPH,
                        text=text,
                        metadata={"style": style_name},
                    )
                )

        # 2. Process tables
        for table_idx, table in enumerate(doc.tables, start=1):
            table_rows = []
            for row in table.rows:
                row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                if any(row_cells):
                    table_rows.append(" | ".join(row_cells))

            if table_rows:
                table_text = "\n".join(table_rows)
                elements.append(
                    ParsedElement(
                        element_type=ParsedElementType.TABLE,
                        text=table_text,
                        metadata={"table_index": table_idx},
                    )
                )

        pages = [ParsedPage(page_number=1, elements=elements)]

        return ParsedDocument(
            document_id=document_id,
            filename=original_filename,
            pages=pages,
            page_count=1,
            metadata={
                "parser": "DOCXParser",
                "total_elements": len(elements),
                "has_tables": len(doc.tables) > 0,
            },
        )
