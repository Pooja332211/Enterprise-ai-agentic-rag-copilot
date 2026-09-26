# Architecture Decision Record (ADR) 02: Document Ingestion, Parsing & Structure-Aware Chunking

## Status
Accepted

## Context
Milestone 2 establishes the core ingestion and transformation pipeline for the Enterprise AI Knowledge & Research Copilot. Enterprise documents arrive in multiple heterogeneous formats (PDF, DOCX, TXT, Markdown, HTML), with varying quality, layout conventions, and structures. 

Naive character-slicing or fixed token window chunking destroys section semantics, discards heading hierarchies, breaks tables, and loses physical page boundaries essential for grounded citations. Furthermore, unrestricted file uploads present security risks (path traversal, zip/bomb attacks, malicious extensions) and storage inefficiencies (duplicate uploads).

## Decisions

1. **Parser Registry Pattern & Normalized Document Model**:
   - Implemented an extensible `ParserRegistry` mapping MIME types and file extensions to specialized parser implementations (`BaseParser` abstraction).
   - All parsers normalize heterogeneous inputs into a uniform schema:
     ```
     ParsedDocument
       ├── metadata (title, author, creation_date, source_format, total_pages)
       └── pages: List[ParsedPage]
             └── elements: List[ParsedElement]
                   ├── type (heading, paragraph, list_item, table, code)
                   ├── text
                   ├── heading_level (1-6)
                   ├── section_path (breadcrumb path, e.g. "Spec > Architecture")
                   └── page_number (1-indexed physical citation boundary)
     ```
   - Parsers implemented:
     - **PDFParser** (`pypdf`): Extracts physical page boundaries and layout text for citations.
     - **DOCXParser** (`python-docx`): Extracts heading styles (Heading 1-6), paragraphs, list items, and table matrices.
     - **MarkdownParser**: Extracts `#` heading hierarchies, blockquotes, list items, and fenced code blocks.
     - **TextParser**: Heuristically extracts numbered/capitalized headings and double-newline paragraph blocks.
     - **HTMLParser** (`beautifulsoup4` + `lxml`): Strips non-content tags (`<script>`, `<style>`, `<nav>`, `<footer>`) and extracts `h1`-`h6`, `p`, `ul/ol`, and `table` contents.

2. **Structure-Aware Chunking Strategy**:
   - Replaced fixed-window chunkers with structure-aware chunking (`ChunkingService`):
     - **Heading Hierarchy Preservation**: Tracks active `section_title` and `section_path` (e.g. `Executive Summary > Methodology`).
     - **Page Boundary Respect**: Chunks do not cross physical page boundaries to ensure citations unambiguously reference exact pages.
     - **Element Grouping**: Groups semantic elements within the same section and page up to `CHUNK_SIZE` characters (default 800), with `CHUNK_OVERLAP` (default 100).
     - **Paragraph Boundary Splitting**: Large paragraphs exceeding chunk limits are split on sentence/paragraph breaks rather than arbitrary character cuts.
     - **Zero Data Loss**: Small pages or trailing elements below `MIN_CHUNK_SIZE` are preserved if the page has no other chunks.
     - **Deterministic Ordering**: Chunks are indexed sequentially with `chunk_index` starting at 0.

3. **Secure File Storage Service**:
   - `LocalStorageService` enforces strict path traversal protections:
     - Strips relative directory traversal sequences (`..`, `/`, `\`).
     - Cleans filenames using regex `[^a-zA-Z0-9_.-]`.
     - Validates resolved path is strictly within the configured root directory.
     - Storage layout: `data/uploads/<document_id>/original.<ext>` isolating documents by UUID.

4. **SHA-256 Deduplication & Idempotency**:
   - Computes SHA-256 hash across raw file contents before storage.
   - Rejects duplicate documents with `HTTP 409 Conflict` by default, returning the existing document metadata and chunk count.
   - Provides an optional `force=true` query parameter for administrative re-ingestion.

5. **Relational Persistence & Cascade Lifecycle**:
   - Database schema extended with `document_chunks` table linked to `documents` via `ON DELETE CASCADE`.
   - `DocumentChunk` stores:
     - `document_id`, `chunk_index`, `text`, `character_count`, `token_count`
     - `page_number`, `section_title`, `section_path`
     - Composite index `(document_id, chunk_index)` for fast sequential retrieval.
   - `DELETE /api/v1/documents/{document_id}` cascades across chunks, ingestion jobs, and deletes physical disk assets.

## Consequences
- High-quality, semantically intact chunks with rich structural metadata are ready for Milestone 3 (dense & sparse vector indexing).
- Physical page citation boundaries are guaranteed for grounded RAG generation (Milestone 5) and citation verification (Milestone 7).
- No external model API calls or quota dependencies are introduced.
