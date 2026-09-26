import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.config.settings import Settings, get_settings
from backend.app.database.models import Document, DocumentChunk, IngestionJob, generate_uuid
from backend.app.parsers.registry import get_parser_registry
from backend.app.services.chunking_service import get_chunker
from backend.app.services.storage_service import get_storage_service
from backend.app.utils.logger import logger


class IngestionService:
    """
    Orchestrates the complete document ingestion pipeline:
    validation -> checksum -> storage -> parsing -> chunking -> database persistence.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.storage = get_storage_service()
        self.parser_registry = get_parser_registry()
        self.chunker = get_chunker()

    @staticmethod
    def calculate_checksum(content: bytes) -> str:
        """Calculates SHA-256 hash of file bytes."""
        return hashlib.sha256(content).hexdigest()

    async def find_existing_document_by_hash(
        self, file_hash: str, db: AsyncSession
    ) -> Optional[Document]:
        """Finds completed document with identical content hash."""
        stmt = select(Document).where(
            Document.file_hash == file_hash, Document.status == "completed"
        )
        result = await db.execute(stmt)
        return result.scalars().first()

    async def ingest_file(
        self,
        filename: str,
        content: bytes,
        mime_type: Optional[str],
        db: AsyncSession,
        force: bool = False,
    ) -> Tuple[Document, IngestionJob, bool]:
        """
        Executes end-to-end document ingestion.
        Returns: (Document, IngestionJob, is_duplicate)
        """
        start_time = time.perf_counter()
        logger.info(f"document_received: filename='{filename}' size={len(content)} bytes")

        # 1. Validation: File size
        max_bytes = self.settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise ValueError(
                f"File size ({len(content)} bytes) exceeds maximum limit of {self.settings.MAX_UPLOAD_SIZE_MB} MB"
            )

        if len(content) == 0:
            raise ValueError("Uploaded file is empty (0 bytes)")

        # 2. Validation: Extension
        ext = filename.split(".")[-1].lower() if "." in filename else ""
        if not ext or ext not in self.settings.ALLOWED_EXTENSIONS:
            allowed = ", ".join(self.settings.ALLOWED_EXTENSIONS)
            raise ValueError(
                f"Unsupported file extension '.{ext}'. Allowed extensions: {allowed}"
            )

        # 3. Checksum & Duplicate Detection
        file_hash = self.calculate_checksum(content)
        existing_doc = await self.find_existing_document_by_hash(file_hash, db)
        if existing_doc and not force:
            logger.info(
                f"document_duplicate_detected: filename='{filename}' matches doc_id={existing_doc.id}"
            )
            # Fetch latest completed job for existing doc
            job_stmt = (
                select(IngestionJob)
                .where(IngestionJob.document_id == existing_doc.id)
                .order_by(IngestionJob.created_at.desc())
            )
            job_result = await db.execute(job_stmt)
            existing_job = job_result.scalars().first()
            return existing_doc, existing_job, True

        document_id = generate_uuid()

        # 4. Save file to storage
        storage_path, normalized_filename, size_bytes = self.storage.save_file(
            file_content=content, document_id=document_id, original_filename=filename
        )
        logger.info(f"document_stored: document_id={document_id} path='{storage_path}'")

        # 5. Create initial Document & IngestionJob records
        doc = Document(
            id=document_id,
            filename=filename,
            normalized_filename=normalized_filename,
            file_path=storage_path,
            file_size_bytes=size_bytes,
            mime_type=mime_type or f"application/{ext}",
            file_extension=ext,
            file_hash=file_hash,
            status="processing",
            chunk_count=0,
        )
        db.add(doc)

        job = IngestionJob(
            document_id=document_id,
            status="processing",
            stage="PARSING",
            progress=0.2,
        )
        db.add(job)
        await db.commit()
        await db.refresh(doc)
        await db.refresh(job)

        # 6. Parse document
        try:
            full_file_path = self.storage.get_file_path(storage_path)
            parser = self.parser_registry.get_parser(ext, mime_type)
            logger.info(f"parsing_started: document_id={document_id} parser={parser.__class__.__name__}")
            
            parsed_doc = parser.parse(full_file_path, document_id, filename)
            logger.info(
                f"parsing_completed: document_id={document_id} pages={parsed_doc.page_count} elements={parsed_doc.total_elements}"
            )

            # Check if document has no text (e.g. scanned PDF)
            if parsed_doc.metadata.get("is_scanned"):
                err_msg = "Document appears to be scanned or contains no extractable text. OCR is scheduled for a future milestone."
                logger.warning(f"document_scanned: document_id={document_id}")
                doc.status = "failed"
                job.status = "failed"
                job.stage = "FAILED"
                job.error_message = err_msg
                job.completed_at = datetime.utcnow()
                await db.commit()
                return doc, job, False

            # 7. Chunk document
            job.stage = "CHUNKING"
            job.progress = 0.5
            await db.commit()

            logger.info(f"chunking_started: document_id={document_id}")
            chunk_data_list = self.chunker.chunk_document(parsed_doc)
            logger.info(f"chunking_completed: document_id={document_id} chunk_count={len(chunk_data_list)}")

            # 8. Persist Chunks in Database
            job.stage = "PERSISTING"
            job.progress = 0.8
            await db.commit()

            created_db_chunks = []
            for chunk_data in chunk_data_list:
                db_chunk = DocumentChunk(
                    document_id=document_id,
                    chunk_index=chunk_data.chunk_index,
                    text=chunk_data.text,
                    token_count=chunk_data.token_count,
                    character_count=chunk_data.character_count,
                    page_number=chunk_data.page_number,
                    section_title=chunk_data.section_title,
                    section_path=chunk_data.section_path,
                    metadata_json=json.dumps(chunk_data.metadata),
                )
                db.add(db_chunk)
                created_db_chunks.append(db_chunk)

            # 9. Index Chunks (Dense Vector in Qdrant + Lexical BM25)
            job.stage = "INDEXING"
            job.progress = 0.9
            await db.commit()

            try:
                from backend.app.services.indexing_service import get_indexing_service
                indexing_service = get_indexing_service(self.settings)
                indexing_service.index_document_chunks(
                    document_id=document_id,
                    chunks=created_db_chunks,
                )
            except Exception as index_exc:
                logger.error(f"indexing_error: document_id={document_id} error={index_exc}", exc_info=True)
                doc.status = "indexing_failed"
                doc.chunk_count = len(chunk_data_list)
                doc.page_count = parsed_doc.page_count
                job.status = "failed"
                job.stage = "FAILED"
                job.error_message = f"Indexing failed: {index_exc}"
                job.completed_at = datetime.utcnow()
                await db.commit()
                await db.refresh(doc)
                await db.refresh(job)
                return doc, job, False

            # 10. Finalize Document & IngestionJob state
            doc.status = "completed"
            doc.chunk_count = len(chunk_data_list)
            doc.page_count = parsed_doc.page_count

            job.status = "completed"
            job.stage = "COMPLETED"
            job.progress = 1.0
            job.completed_at = datetime.utcnow()

            await db.commit()
            await db.refresh(doc)
            await db.refresh(job)

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info(
                f"ingestion_completed: document_id={document_id} chunks={doc.chunk_count} elapsed_ms={elapsed_ms}"
            )
            return doc, job, False

        except Exception as exc:
            logger.error(f"ingestion_failed: document_id={document_id} error={exc}", exc_info=True)
            await db.rollback()

            # Safely record failure state
            doc.status = "failed"
            job.status = "failed"
            job.stage = "FAILED"
            job.error_message = str(exc)
            job.completed_at = datetime.utcnow()

            db.add(doc)
            db.add(job)
            await db.commit()
            raise


_ingestion_service_instance: Optional[IngestionService] = None


def get_ingestion_service() -> IngestionService:
    """Returns singleton ingestion service instance."""
    global _ingestion_service_instance
    if _ingestion_service_instance is None:
        _ingestion_service_instance = IngestionService()
    return _ingestion_service_instance
