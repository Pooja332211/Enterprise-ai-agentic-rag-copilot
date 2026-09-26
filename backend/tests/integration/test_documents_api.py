import io
from pathlib import Path
import pytest
from httpx import AsyncClient

FIXTURES_DIR = Path("backend/tests/fixtures")


@pytest.mark.asyncio
async def test_upload_and_retrieve_pdf(async_client: AsyncClient):
    """Verify uploading a PDF through the API, checking response and chunks."""
    pdf_bytes = (FIXTURES_DIR / "sample.pdf").read_bytes()
    files = {"file": ("api_test_sample.pdf", pdf_bytes, "application/pdf")}

    # 1. Upload
    response = await async_client.post("/api/v1/documents", files=files)
    assert response.status_code == 201
    data = response.json()
    assert data["is_duplicate"] is False
    assert "document" in data
    doc = data["document"]
    doc_id = doc["id"]
    assert doc["filename"] == "api_test_sample.pdf"
    assert doc["status"] == "completed"
    assert doc["chunk_count"] >= 1

    # 2. Get document metadata
    doc_resp = await async_client.get(f"/api/v1/documents/{doc_id}")
    assert doc_resp.status_code == 200
    assert doc_resp.json()["id"] == doc_id

    # 3. Get document chunks
    chunks_resp = await async_client.get(f"/api/v1/documents/{doc_id}/chunks")
    assert chunks_resp.status_code == 200
    chunks_data = chunks_resp.json()
    assert chunks_data["total"] >= 1
    assert len(chunks_data["items"]) == chunks_data["total"]

    first_chunk = chunks_data["items"][0]
    assert first_chunk["document_id"] == doc_id
    assert first_chunk["chunk_index"] == 0
    assert first_chunk["token_count"] > 0

    # 4. Check Ingestion Job
    job_id = data["job"]["id"]
    job_resp = await async_client.get(f"/api/v1/ingestion/{job_id}")
    assert job_resp.status_code == 200
    assert job_resp.json()["status"] == "completed"
    assert job_resp.json()["stage"] == "COMPLETED"


@pytest.mark.asyncio
async def test_upload_docx(async_client: AsyncClient):
    """Verify uploading a DOCX document."""
    docx_bytes = (FIXTURES_DIR / "sample.docx").read_bytes()
    files = {"file": ("api_sample.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}

    response = await async_client.post("/api/v1/documents", files=files)
    assert response.status_code == 201
    data = response.json()
    assert data["document"]["status"] == "completed"
    assert data["document"]["file_extension"] == "docx"


@pytest.mark.asyncio
async def test_upload_duplicate_conflict(async_client: AsyncClient):
    """Verify duplicate upload returns HTTP 409 Conflict with existing document."""
    content = b"Deterministic payload specifically for API duplicate test."
    files = {"file": ("first_file.txt", content, "text/plain")}

    # First upload
    res1 = await async_client.post("/api/v1/documents", files=files)
    assert res1.status_code == 201
    first_doc_id = res1.json()["document"]["id"]

    # Second upload with same content
    files2 = {"file": ("second_file.txt", content, "text/plain")}
    res2 = await async_client.post("/api/v1/documents", files=files2)
    assert res2.status_code == 409
    dup_data = res2.json()
    assert dup_data["is_duplicate"] is True
    assert dup_data["document"]["id"] == first_doc_id


@pytest.mark.asyncio
async def test_upload_invalid_extension(async_client: AsyncClient):
    """Verify uploading unsupported file type returns HTTP 400."""
    files = {"file": ("malicious.sh", b"#!/bin/bash\nrm -rf /", "application/x-sh")}
    response = await async_client.post("/api/v1/documents", files=files)
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_empty_file(async_client: AsyncClient):
    """Verify uploading 0-byte file returns HTTP 400."""
    files = {"file": ("empty.txt", b"", "text/plain")}
    response = await async_client.post("/api/v1/documents", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_list_and_delete_document(async_client: AsyncClient):
    """Verify listing documents and deleting a document."""
    content = b"Content to test document deletion."
    files = {"file": ("delete_me.txt", content, "text/plain")}

    upload_res = await async_client.post("/api/v1/documents", files=files)
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document"]["id"]

    # List documents
    list_res = await async_client.get("/api/v1/documents?limit=10")
    assert list_res.status_code == 200
    assert list_res.json()["total"] >= 1

    # Delete document
    del_res = await async_client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200

    # Verify not found after deletion
    get_res = await async_client.get(f"/api/v1/documents/{doc_id}")
    assert get_res.status_code == 404
