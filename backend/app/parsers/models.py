from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ParsedElementType(str, Enum):
    """Categorization of structured document elements."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CODE = "code"


class ParsedElement(BaseModel):
    """A single atomic structural element extracted from a document."""
    element_type: ParsedElementType
    text: str
    level: Optional[int] = None  # Heading depth 1-6 if element_type == HEADING
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParsedPage(BaseModel):
    """A page container preserving 1-indexed page boundaries for citation accuracy."""
    page_number: int
    elements: List[ParsedElement] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Concatenated text of all elements on this page."""
        return "\n\n".join(el.text for el in self.elements if el.text.strip())


class ParsedDocument(BaseModel):
    """Normalized structured document representation produced by all parsers."""
    document_id: str
    filename: str
    pages: List[ParsedPage] = Field(default_factory=list)
    page_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def total_elements(self) -> int:
        """Total number of structural elements across all pages."""
        return sum(len(page.elements) for page in self.pages)

    @property
    def full_text(self) -> str:
        """Concatenated full text across all pages."""
        return "\n\n".join(page.full_text for page in self.pages if page.full_text.strip())
