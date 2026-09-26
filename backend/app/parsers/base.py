from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from backend.app.parsers.models import ParsedDocument


class BaseParser(ABC):
    """Abstract base class for all file format parsers."""

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """List of lowercase file extensions without leading dot (e.g. ['pdf'])."""
        pass

    @property
    @abstractmethod
    def supported_mime_types(self) -> List[str]:
        """List of supported MIME types (e.g. ['application/pdf'])."""
        pass

    @abstractmethod
    def parse(
        self, file_path: Path, document_id: str, original_filename: str
    ) -> ParsedDocument:
        """
        Parses the document file into a normalized ParsedDocument representation.
        Raises ValueError or RuntimeError if file is malformed.
        """
        pass
