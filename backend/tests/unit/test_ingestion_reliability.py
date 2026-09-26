"""Unit tests for Milestone 10 ingestion reliability, failure visibility, and duplicate handling."""

from unittest.mock import MagicMock, patch
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.services.ingestion_service import IngestionService


@pytest.mark.asyncio
async def test_indexing_failure_is_not_silently_marked_completed(test_db_session: AsyncSession):
    """Verify that when vector/lexical indexing fails, the document and job are marked failed."""
    service = IngestionService()
    content = b"# Document with Indexing Failure Simulation\n\nThis content triggers an indexing error."

    # Simulate an error in indexing service
    mock_indexer = MagicMock()
    mock_indexer.index_document_chunks.side_effect = RuntimeError("Qdrant connection dropped during upsert")

    with patch("backend.app.services.indexing_service.get_indexing_service", return_value=mock_indexer):
        doc, job, is_dup = await service.ingest_file(
            filename="indexing_fail_test.md",
            content=content,
            mime_type="text/markdown",
            db=test_db_session,
        )

        assert is_dup is False
        assert doc.status == "indexing_failed"
        assert job.status == "failed"
        assert job.stage == "FAILED"
        assert "Indexing failed" in job.error_message


@pytest.mark.asyncio
async def test_empty_file_rejected(test_db_session: AsyncSession):
    """Verify that empty file uploads are rejected with ValueError."""
    service = IngestionService()
    with pytest.raises(ValueError, match="Uploaded file is empty"):
        await service.ingest_file(
            filename="empty_file.txt",
            content=b"",
            mime_type="text/plain",
            db=test_db_session,
        )


@pytest.mark.asyncio
async def test_unsupported_extension_rejected(test_db_session: AsyncSession):
    """Verify that unsupported file extensions are rejected."""
    service = IngestionService()
    with pytest.raises(ValueError, match="Unsupported file extension"):
        await service.ingest_file(
            filename="malicious_payload.exe",
            content=b"binary-content-here",
            mime_type="application/x-dosexec",
            db=test_db_session,
        )
