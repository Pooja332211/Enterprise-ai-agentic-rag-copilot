import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.database.models import Document, IngestionJob, SystemAuditLog


@pytest.mark.asyncio
async def test_create_and_retrieve_document(test_db_session: AsyncSession):
    """Verify document model creation, persistence, and querying."""
    doc = Document(
        filename="research_paper.pdf",
        file_path="/data/uploads/research_paper.pdf",
        file_size_bytes=1024000,
        mime_type="application/pdf",
        file_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        status="pending",
    )
    test_db_session.add(doc)
    await test_db_session.commit()
    await test_db_session.refresh(doc)

    assert doc.id is not None
    assert doc.created_at is not None
    assert doc.status == "pending"

    # Query back
    result = await test_db_session.execute(select(Document).where(Document.id == doc.id))
    retrieved = result.scalar_one_or_none()
    assert retrieved is not None
    assert retrieved.filename == "research_paper.pdf"
    assert retrieved.file_size_bytes == 1024000


@pytest.mark.asyncio
async def test_document_ingestion_job_relationship(test_db_session: AsyncSession):
    """Verify relationship between Document and IngestionJob entities."""
    doc = Document(
        filename="benchmark_report.pdf",
        file_path="/data/uploads/benchmark_report.pdf",
        file_size_bytes=512000,
        mime_type="application/pdf",
        status="processing",
    )
    test_db_session.add(doc)
    await test_db_session.commit()

    job = IngestionJob(
        document_id=doc.id,
        status="queued",
        progress=0.0,
    )
    test_db_session.add(job)
    await test_db_session.commit()
    await test_db_session.refresh(job)

    assert job.id is not None
    assert job.document_id == doc.id
    assert job.status == "queued"
    assert job.progress == 0.0


@pytest.mark.asyncio
async def test_system_audit_log(test_db_session: AsyncSession):
    """Verify system audit log creation."""
    log = SystemAuditLog(
        event_type="SYSTEM_BOOT",
        details="Initial schema verified and database pool initialized.",
    )
    test_db_session.add(log)
    await test_db_session.commit()
    await test_db_session.refresh(log)

    assert log.id is not None
    assert log.event_type == "SYSTEM_BOOT"
    assert "Initial schema verified" in log.details
