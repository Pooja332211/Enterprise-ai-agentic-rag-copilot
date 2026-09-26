from pathlib import Path
from typing import List
from bs4 import BeautifulSoup
from backend.app.parsers.base import BaseParser
from backend.app.parsers.models import (
    ParsedDocument,
    ParsedElement,
    ParsedElementType,
    ParsedPage,
)
from backend.app.utils.logger import logger


class HTMLParser(BaseParser):
    """
    Parser for HTML documents (.html, .htm).
    Strips scripts and styles, and extracts headings (H1-H6), paragraphs, lists, and tables.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ["html", "htm"]

    @property
    def supported_mime_types(self) -> List[str]:
        return ["text/html", "application/xhtml+xml"]

    def parse(
        self, file_path: Path, document_id: str, original_filename: str
    ) -> ParsedDocument:
        logger.info(f"Parsing HTML document: {file_path.name}")
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.error(f"Failed to read HTML file {file_path}: {exc}")
            raise ValueError(f"Could not read HTML file: {exc}")

        soup = BeautifulSoup(content, "html.parser")

        # Strip scripts, styles, and comments
        for script_or_style in soup(["script", "style", "meta", "noscript", "header", "footer"]):
            script_or_style.extract()

        elements: List[ParsedElement] = []

        # Find structural elements
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table"]):
            tag_name = tag.name.lower()
            text = tag.get_text(separator=" ", strip=True)
            if not text:
                continue

            if tag_name.startswith("h") and len(tag_name) == 2 and tag_name[1].isdigit():
                level = int(tag_name[1])
                elements.append(
                    ParsedElement(
                        element_type=ParsedElementType.HEADING,
                        text=text,
                        level=level,
                    )
                )
            elif tag_name == "li":
                elements.append(
                    ParsedElement(
                        element_type=ParsedElementType.LIST_ITEM,
                        text=text,
                    )
                )
            elif tag_name == "table":
                # Convert table rows to text
                rows = []
                for tr in tag.find_all("tr"):
                    cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
                    if any(cells):
                        rows.append(" | ".join(cells))
                if rows:
                    elements.append(
                        ParsedElement(
                            element_type=ParsedElementType.TABLE,
                            text="\n".join(rows),
                        )
                    )
            else:
                elements.append(
                    ParsedElement(
                        element_type=ParsedElementType.PARAGRAPH,
                        text=text,
                    )
                )

        pages = [ParsedPage(page_number=1, elements=elements)]

        return ParsedDocument(
            document_id=document_id,
            filename=original_filename,
            pages=pages,
            page_count=1,
            metadata={
                "parser": "HTMLParser",
                "total_elements": len(elements),
            },
        )
