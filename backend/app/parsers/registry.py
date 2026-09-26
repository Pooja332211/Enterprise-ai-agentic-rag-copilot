from typing import Dict, List, Optional
from backend.app.parsers.base import BaseParser
from backend.app.parsers.docx_parser import DOCXParser
from backend.app.parsers.html_parser import HTMLParser
from backend.app.parsers.markdown_parser import MarkdownParser
from backend.app.parsers.pdf_parser import PDFParser
from backend.app.parsers.text_parser import TextParser
from backend.app.utils.logger import logger


class ParserRegistry:
    """
    Central registry and factory for document parsers.
    Maps file extensions and MIME types to appropriate parser instances.
    """

    def __init__(self):
        self._extension_map: Dict[str, BaseParser] = {}
        self._mime_map: Dict[str, BaseParser] = {}
        self._register_default_parsers()

    def _register_default_parsers(self) -> None:
        """Registers built-in parsers for PDF, DOCX, TXT, Markdown, and HTML."""
        default_parsers: List[BaseParser] = [
            PDFParser(),
            DOCXParser(),
            MarkdownParser(),
            TextParser(),
            HTMLParser(),
        ]
        for parser in default_parsers:
            self.register_parser(parser)

    def register_parser(self, parser: BaseParser) -> None:
        """Registers a parser instance for its supported extensions and MIME types."""
        for ext in parser.supported_extensions:
            clean_ext = ext.lower().lstrip(".")
            self._extension_map[clean_ext] = parser

        for mime in parser.supported_mime_types:
            self._mime_map[mime.lower()] = parser

    def get_parser(
        self, file_extension: str, mime_type: Optional[str] = None
    ) -> BaseParser:
        """
        Retrieves matching parser for a given file extension or MIME type.
        Raises ValueError if file format is unsupported.
        """
        clean_ext = file_extension.lower().lstrip(".")

        if clean_ext in self._extension_map:
            return self._extension_map[clean_ext]

        if mime_type:
            clean_mime = mime_type.lower().split(";")[0].strip()
            if clean_mime in self._mime_map:
                return self._mime_map[clean_mime]

        supported = list(self._extension_map.keys())
        logger.warning(f"No parser available for extension '{file_extension}' (MIME: {mime_type})")
        raise ValueError(
            f"Unsupported file format '.{clean_ext}'. Supported extensions: {', '.join(supported)}"
        )

    def is_supported(self, file_extension: str) -> bool:
        """Returns True if the file extension is supported."""
        return file_extension.lower().lstrip(".") in self._extension_map

    @property
    def supported_extensions(self) -> List[str]:
        """Returns a list of all registered file extensions."""
        return sorted(list(self._extension_map.keys()))


# Singleton parser registry instance
_parser_registry_instance: Optional[ParserRegistry] = None


def get_parser_registry() -> ParserRegistry:
    """Returns the singleton parser registry instance."""
    global _parser_registry_instance
    if _parser_registry_instance is None:
        _parser_registry_instance = ParserRegistry()
    return _parser_registry_instance
