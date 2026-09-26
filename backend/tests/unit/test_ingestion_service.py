from pathlib import Path
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.database.models import Document, DocumentChunk, IngestionJob
from backend.app.services.ingestion_service import IngestionService

FIXTURES_DIR = Path("backend/tests/fixtures")


@pytest.mark.asyncio
async def test_ingest_txt_pipeline(test_db_session: AsyncSession):
    """Verify end-to-end ingestion of a plain text file."""
    service = IngestionService()
    content = (FIXTURES_DIR / "sample.txt").read_bytes()

    doc, job, is_duplicate = await service.ingest_file(
        filename="test_sample.txt",
        content=content,
        mime_type="text/plain",
        db=test_db_session,
    )

    assert is_duplicate is False
    assert doc.status == "completed"
    assert doc.chunk_count >= 1
    assert job.status == "completed"
    assert job.stage == "COMPLETED"
    assert job.progress == 1.0

    # Verify chunks in database
    chunks_stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
    res = await test_db_session.execute(chunks_stmt)
    chunks = res.scalars().all()
    assert len(chunks) == doc.chunk_count


@pytest.mark.asyncio
async def test_duplicate_detection(test_db_session: AsyncSession):
    """Verify uploading identical content triggers duplicate detection."""
    service = IngestionService()
    content = b"Unique enterprise report content for duplicate detection test."

    doc1, job1, is_dup1 = await service.ingest_file(
        filename="first_upload.txt",
        content=content,
        mime_type="text/plain",
        db=test_db_session,
    )
    assert is_dup1 is False
    assert doc1.status == "completed"

    # Second upload with identical content but different filename
    doc2, job2, is_dup2 = await service.ingest_file(
        filename="second_upload.txt",
        content=content,
        mime_type="text/plain",
        db=test_db_session,
    )
    assert is_dup2 is True
    assert doc2.id == doc1.id


@pytest.mark.asyncio
async def test_ingest_empty_file_fails(test_db_session: AsyncSession):
    """Verify empty file raises ValueError."""
    service = IngestionService()
    with pytest.raises(ValueError, match="Uploaded file is empty"):
        await service.ingest_file(
            filename="empty.txt",
            content=b"",
            mime_type="text/plain",
            db=test_db_session,
        )


@pytest.mark.asyncio
async def test_ingest_unsupported_extension_fails(test_db_session: AsyncSession):
    """Verify unsupported file extension raises ValueError."""
    service = IngestionService()
    with pytest.raises(ValueError, match="Unsupported file extension"):
        await service.ingest_file(
            filename="binary_program.exe",
            content=b"MZ\x90\x00",
            mime_type="application/octet-stream",
            db=test_db_session,
        )


@pytest.mark.asyncio
async def test_ingest_oversized_file_fails(test_db_session: AsyncSession):
    """Verify file exceeding size limit raises ValueError."""
    class MockSettings:
        MAX_UPLOAD_SIZE_MB = 1  # 1 MB
        ALLOWED_EXTENSIONS = ["txt"]
        UPLOAD_DIR = "./data/uploads"
        CHUNK_SIZE = 800
        CHUNK_OVERLAP = 100
        MIN_CHUNK_SIZE = 50

    service = IngestionService(settings=MockSettings())
    large_content = b"A" * (2 * 1024 * 1024)  # 2 MB

    with pytest.raises(ValueError, match="exceeds maximum limit"):
        await service.ingest_file(
            filename="too_large.txt",
            content=large_content,
            mime_type="text/plain",
            db=test_db_session,
        )
