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


class MarkdownParser(BaseParser):
    """
    Parser for Markdown (.md) documents.
    Preserves heading hierarchies (# to ######), code blocks, list items, and paragraphs.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ["md", "markdown"]

    @property
    def supported_mime_types(self) -> List[str]:
        return ["text/markdown", "text/x-markdown"]

    def parse(
        self, file_path: Path, document_id: str, original_filename: str
    ) -> ParsedDocument:
        logger.info(f"Parsing Markdown document: {file_path.name}")
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.error(f"Failed to read Markdown file {file_path}: {exc}")
            raise ValueError(f"Could not read markdown file: {exc}")

        elements: List[ParsedElement] = []
        lines = content.splitlines()

        in_code_block = False
        code_lines: List[str] = []
        code_lang = ""

        current_para: List[str] = []

        def flush_para():
            if current_para:
                para_text = " ".join(current_para).strip()
                if para_text:
                    elements.append(
                        ParsedElement(
                            element_type=ParsedElementType.PARAGRAPH,
                            text=para_text,
                        )
                    )
                current_para.clear()

        for line in lines:
            trimmed = line.strip()

            # Handle code blocks
            if trimmed.startswith("```"):
                if in_code_block:
                    # Closing code block
                    code_text = "\n".join(code_lines)
                    elements.append(
                        ParsedElement(
                            element_type=ParsedElementType.CODE,
                            text=code_text,
                            metadata={"language": code_lang},
                        )
                    )
                    code_lines = []
                    code_lang = ""
                    in_code_block = False
                else:
                    # Opening code block
                    flush_para()
                    in_code_block = True
                    code_lang = trimmed[3:].strip()
                continue

            if in_code_block:
                code_lines.append(line)
                continue

            # Check headings: # to ######
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", trimmed)
            if heading_match:
                flush_para()
                hashes, heading_text = heading_match.groups()
                elements.append(
                    ParsedElement(
                        element_type=ParsedElementType.HEADING,
                        text=heading_text.strip(),
                        level=len(hashes),
                    )
                )
                continue

            # Check list items
            list_match = re.match(r"^[-*+]\s+(.+)$|^\d+\.\s+(.+)$", trimmed)
            if list_match:
                flush_para()
                item_text = list_match.group(1) or list_match.group(2)
                elements.append(
                    ParsedElement(
                        element_type=ParsedElementType.LIST_ITEM,
                        text=item_text.strip(),
                    )
                )
                continue

            # Empty line indicates paragraph boundary
            if not trimmed:
                flush_para()
            else:
                current_para.append(trimmed)

        flush_para()

        # Catch unclosed code block
        if in_code_block and code_lines:
            elements.append(
                ParsedElement(
                    element_type=ParsedElementType.CODE,
                    text="\n".join(code_lines),
                    metadata={"language": code_lang},
                )
            )

        pages = [ParsedPage(page_number=1, elements=elements)]

        return ParsedDocument(
            document_id=document_id,
            filename=original_filename,
            pages=pages,
            page_count=1,
            metadata={
                "parser": "MarkdownParser",
                "total_elements": len(elements),
            },
        )
