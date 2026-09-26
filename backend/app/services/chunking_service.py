import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from backend.app.config.settings import Settings, get_settings
from backend.app.parsers.models import (
    ParsedDocument,
    ParsedElement,
    ParsedElementType,
    ParsedPage,
)
from backend.app.utils.logger import logger


@dataclass
class ChunkData:
    """Represents a generated chunk ready for persistence and embedding."""
    chunk_index: int
    text: str
    token_count: int
    character_count: int
    page_number: Optional[int]
    section_title: Optional[str]
    section_path: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class StructureAwareChunker:
    """
    Structure-aware document chunker that respects heading hierarchies,
    section paths, and physical page boundaries.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.max_chunk_size = self.settings.CHUNK_SIZE
        self.overlap = self.settings.CHUNK_OVERLAP
        self.min_chunk_size = self.settings.MIN_CHUNK_SIZE

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimates token count from text using word and character heuristics (~4 chars/token).
        Provides consistent metric without requiring heavy external tokenizer model.
        """
        if not text:
            return 0
        char_estimate = len(text) // 4
        word_estimate = int(len(text.split()) * 1.3)
        return max(1, max(char_estimate, word_estimate))

    def chunk_document(self, document: ParsedDocument) -> List[ChunkData]:
        """
        Chunks a parsed document preserving heading hierarchy, section paths,
        and page numbers with deterministic ordering.
        """
        chunks: List[ChunkData] = []
        chunk_counter = 0

        # Section hierarchy tracking: stack of (level, heading_text)
        section_stack: List[str] = []
        current_section_title: Optional[str] = None

        for page in document.pages:
            page_num = page.page_number
            page_buffer: List[str] = []
            buffer_chars = 0

            for element in page.elements:
                # 1. Update section hierarchy if element is a heading
                if element.element_type == ParsedElementType.HEADING:
                    current_section_title = element.text
                    level = element.level or 1

                    # Truncate stack to maintain depth
                    if len(section_stack) >= level:
                        section_stack = section_stack[: level - 1]
                    section_stack.append(element.text)

                current_path = " > ".join(section_stack) if section_stack else None

                elem_text = element.text.strip()
                if not elem_text:
                    continue

                elem_len = len(elem_text)

                # 2. Check if adding element exceeds max chunk size
                if buffer_chars + elem_len > self.max_chunk_size and page_buffer:
                    chunk_text = "\n\n".join(page_buffer).strip()
                    page_has_chunks = any(c.page_number == page_num for c in chunks)
                    if (len(chunk_text) >= self.min_chunk_size or not page_has_chunks) and chunk_text:
                        chunks.append(
                            ChunkData(
                                chunk_index=chunk_counter,
                                text=chunk_text,
                                token_count=self.estimate_tokens(chunk_text),
                                character_count=len(chunk_text),
                                page_number=page_num,
                                section_title=current_section_title,
                                section_path=current_path,
                                metadata={
                                    "document_id": document.document_id,
                                    "filename": document.filename,
                                    "page": page_num,
                                },
                            )
                        )
                        chunk_counter += 1

                    # Setup overlap from the end of the previous buffer
                    overlap_seed = ""
                    if self.overlap > 0 and page_buffer:
                        last_item = page_buffer[-1]
                        if len(last_item) <= self.overlap:
                            overlap_seed = last_item
                        else:
                            overlap_seed = last_item[-self.overlap :]

                    page_buffer = [overlap_seed, elem_text] if overlap_seed else [elem_text]
                    buffer_chars = sum(len(x) for x in page_buffer)
                elif elem_len > self.max_chunk_size:
                    # Single element is huge: split it into sub-chunks
                    sub_chunks = self._split_large_text(elem_text, self.max_chunk_size, self.overlap)
                    for sc in sub_chunks:
                        chunks.append(
                            ChunkData(
                                chunk_index=chunk_counter,
                                text=sc,
                                token_count=self.estimate_tokens(sc),
                                character_count=len(sc),
                                page_number=page_num,
                                section_title=current_section_title,
                                section_path=current_path,
                                metadata={
                                    "document_id": document.document_id,
                                    "filename": document.filename,
                                    "page": page_num,
                                },
                            )
                        )
                        chunk_counter += 1
                    page_buffer = []
                    buffer_chars = 0
                else:
                    page_buffer.append(elem_text)
                    buffer_chars += elem_len

            # Flush remaining buffer at page end
            if page_buffer:
                chunk_text = "\n\n".join(page_buffer).strip()
                page_has_chunks = any(c.page_number == page_num for c in chunks)
                if (len(chunk_text) >= self.min_chunk_size or not page_has_chunks) and chunk_text:
                    current_path = " > ".join(section_stack) if section_stack else None
                    chunks.append(
                        ChunkData(
                            chunk_index=chunk_counter,
                            text=chunk_text,
                            token_count=self.estimate_tokens(chunk_text),
                            character_count=len(chunk_text),
                            page_number=page_num,
                            section_title=current_section_title,
                            section_path=current_path,
                            metadata={
                                "document_id": document.document_id,
                                "filename": document.filename,
                                "page": page_num,
                            },
                        )
                    )
                    chunk_counter += 1

        logger.info(f"Chunked document '{document.filename}' into {len(chunks)} structure-aware chunks.")
        return chunks

    def _split_large_text(self, text: str, max_size: int, overlap: int) -> List[str]:
        """Splits an oversized text block on sentence or word boundaries."""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + max_size
            if end >= text_len:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Try to break on newline or space
            split_at = text.rfind("\n", start + min(100, max_size), end)
            if split_at == -1:
                split_at = text.rfind(". ", start + min(100, max_size), end)
                if split_at != -1:
                    split_at += 1
            if split_at == -1:
                split_at = text.rfind(" ", start + min(100, max_size), end)

            if split_at == -1 or split_at <= start:
                split_at = end

            chunk = text[start:split_at].strip()
            if chunk:
                chunks.append(chunk)

            start = split_at - overlap if overlap < split_at else split_at

        return chunks


_chunker_instance: Optional[StructureAwareChunker] = None


def get_chunker() -> StructureAwareChunker:
    """Returns singleton structure-aware chunker instance."""
    global _chunker_instance
    if _chunker_instance is None:
        _chunker_instance = StructureAwareChunker()
    return _chunker_instance
